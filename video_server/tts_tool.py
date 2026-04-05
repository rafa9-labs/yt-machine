import os
import asyncio
import time
import re
import tempfile
from pathlib import Path
from typing import Optional
from mcp.server.fastmcp import FastMCP
import edge_tts
import struct
import wave
import numpy as np

# Kokoro TTS — primary engine (natural human-like speech)
try:
    from kokoro import KPipeline
    import soundfile as sf
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print("  [TTS] Kokoro TTS not installed, will use Edge TTS fallback")

# faster-whisper for high-precision word timestamps (bypasses whisperX's VAD issues)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("  [TTS] faster-whisper not installed, will use fallback timestamps")

server = FastMCP("tts-tool")

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOICE_MAPPING = {
    "authoritative": "en-US-GuyNeural",
    "professional": "en-GB-RyanNeural",
    "energetic": "en-US-AmberNeural",
    "calm": "en-US-AriaNeural",
    "deep": "en-US-GuyNeural",
    "female_professional": "en-GB-SoniaNeural",
    "female_energetic": "en-US-JennyNeural",
    "male_casual": "en-US-BrandonNeural"
}

# Phase 5.3: Content-type to voice-tone mapping
# Maps script mood/topic keywords to the most effective voice tone
CONTENT_VOICE_MAP = {
    "military": "authoritative",
    "conflict": "authoritative",
    "war": "authoritative",
    "economic": "professional",
    "financial": "professional",
    "sanctions": "professional",
    "technology": "professional",
    "cyber": "professional",
    "climate": "calm",
    "health": "calm",
    "pandemic": "calm",
    "diplomatic": "professional",
    "breaking": "energetic",
    "urgent": "energetic",
    "crisis": "authoritative",
    "collapse": "authoritative",
}


def select_voice_for_content(script_text: str, default_tone: str = "authoritative") -> str:
    """
    Phase 5.3: Select the best voice tone based on script content.

    Args:
        script_text: Full script text
        default_tone: Fallback voice tone

    Returns:
        Best matching voice_tone string
    """
    text_lower = script_text.lower()
    scores = {}
    for keyword, tone in CONTENT_VOICE_MAP.items():
        if keyword in text_lower:
            scores[tone] = scores.get(tone, 0) + 1

    if scores:
        best_tone = max(scores, key=lambda k: scores[k])
        return best_tone
    return default_tone


def _apply_audio_mastering(input_path: Path) -> bool:
    """
    Phase 5.2: Apply professional audio mastering to an MP3 file.
    Normalises loudness, reduces harsh sibilance, and ensures consistent levels.
    Uses numpy + scipy if available; silently skips if not installed.

    Args:
        input_path: Path to the MP3 file to master in-place

    Returns:
        True if mastering was applied, False if skipped
    """
    try:
        import numpy as np
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        from moviepy.audio.AudioClip import AudioArrayClip

        clip = AudioFileClip(str(input_path))
        arr = clip.to_soundarray(fps=44100)  # shape: (N, channels)
        clip.close()

        if arr is None or arr.size == 0:
            return False

        # 1. Peak normalise to -3 dBFS (prevent clipping)
        peak = np.max(np.abs(arr))
        if peak > 0:
            target_peak = 0.708  # -3 dBFS
            arr = arr * (target_peak / peak)

        # 2. RMS normalise for consistent perceived loudness
        rms = np.sqrt(np.mean(arr ** 2))
        target_rms = 0.12  # ~-18 LUFS equivalent
        if rms > 0:
            rms_gain = target_rms / rms
            # Cap gain to avoid over-amplifying silence
            rms_gain = min(rms_gain, 3.0)
            arr = arr * rms_gain
            arr = np.clip(arr, -1.0, 1.0)  # hard limit

        # 3. Write back — AudioArrayClip requires a 2D numpy ndarray (N, channels)
        fps = 44100
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        # Ensure contiguous float64 array for moviepy compatibility
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        mastered_clip = AudioArrayClip(arr, fps=fps)
        mastered_clip.write_audiofile(
            str(input_path),
            fps=fps,
            verbose=False,
            logger=None
        )
        mastered_clip.close()
        return True

    except Exception as e:
        print(f"  [TTS] Audio mastering skipped: {e}")
        return False

def _add_natural_pacing(text: str) -> str:
    """
    Clean script text and prepare it for natural speech with comedic timing.
    Removes stray formatting, then INJECTS pauses for:
    - Punchlines (after !): beat pause for laugh/impact
    - Rhetorical questions (after ?): thinking pause
    - Sarcastic tone shifts: comma pauses around key phrases
    - Quote marks: dramatic beat before/after quoted phrases
    
    Args:
        text: Original script text
    
    Returns:
        Clean plain text with natural pause markers for TTS
    """
    # Strip any leftover SSML/XML tags (safety net)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Replace underscores used as separators/formatting with a space
    text = re.sub(r'_+', ' ', text)
    
    # Replace equals signs used as separators
    text = re.sub(r'={2,}', ' ', text)
    
    # Replace dashes used as separators (3+ dashes)
    text = re.sub(r'-{3,}', ' ', text)
    
    # Strip markdown bold/italic markers
    text = re.sub(r'[*#`]', '', text)
    
    # ── COMEDIC PAUSE INJECTION ──
    
    # 1. Exclamation marks → beat pause (punchline impact)
    # "one shot before Easter!" → "one shot before Easter! ... "
    text = re.sub(r'!\s*', '! ...  ', text)
    
    # 2. Question marks → rhetorical pause (let it land)
    # "Think about it." before a question → add anticipation
    text = re.sub(r'\?\s*', '? ...  ', text)
    
    # 3. Quoted phrases → dramatic beat before the reveal
    # "one shot before Easter" → ... "one shot before Easter" ...
    text = re.sub(r'"([^"]+)"', r'... "\1" ...', text)
    
    # 4. Colons before reveals → dramatic pause
    # "The answer:" → "The answer ... "
    text = re.sub(r':\s*', ' ...  ', text)
    
    # 5. Double dashes (em-dash substitutes) → beat pause
    text = re.sub(r'\s*—\s*', ' ... ', text)
    text = re.sub(r'\s*--\s*', ' ... ', text)
    
    # 6. Sentence-ending periods followed by short sentences → extra pause
    # (This creates the "beat" timing for comedic delivery)
    text = re.sub(r'\.\s+([A-Z])', r'. ...  \1', text)
    
    # ── STORY SEPARATOR PAUSES ──
    
    # 7. Four dots (....) = longer pause between stories (~1.5s)
    # This is injected by the script builder after each punchline
    # Convert to a longer silence marker that TTS respects
    text = re.sub(r'\.{4}\s*', '. ... ... ...  ', text)
    
    # ── CLEANUP ──
    
    # Collapse multiple spaces/newlines into single space
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r' {2,}', '  ', text)
    
    # Remove any triple+ periods that aren't our deliberate "..."
    # But be careful not to collapse our injected pauses
    text = re.sub(r'\.{5,}', '...', text)
    
    # Remove pause markers at very start
    text = re.sub(r'^\s*\.{3}\s*', '', text)
    
    return text.strip()

def _get_voice_parameters(voice_tone: str) -> dict:
    """
    Get rate and pitch parameters for Edge TTS based on voice tone.
    Edge TTS does NOT support custom SSML - only plain text with API parameters.
    
    Args:
        voice_tone: Voice style to determine prosody settings
    
    Returns:
        Dict with 'rate' and 'pitch' keys for edge_tts.Communicate()
    """
    # Prosody settings for Edge TTS API parameters
    # Tuned for engaging, news-commentary style delivery
    prosody_settings = {
        "authoritative": {"rate": "+10%", "pitch": "+2Hz"},   # Faster, sharper
        "professional":  {"rate": "+5%",  "pitch": "+0Hz"},   # Confident but clear
        "energetic":     {"rate": "+15%", "pitch": "+4Hz"},   # High energy, urgent
        "calm":          {"rate": "+0%",  "pitch": "-2Hz"},   # Slower, thoughtful
        "deep":          {"rate": "+5%",  "pitch": "-3Hz"},   # Deep but not sluggish
        "female_professional": {"rate": "+5%", "pitch": "+0Hz"},
        "female_energetic":    {"rate": "+12%", "pitch": "+4Hz"},
        "male_casual":         {"rate": "+8%",  "pitch": "+1Hz"},
    }
    return prosody_settings.get(voice_tone, {"rate": "+0%", "pitch": "+0Hz"})

# Kokoro voice packs (American English voices)
KOKORO_VOICES = {
    "authoritative": "am_adam",        # Deep, authoritative male anchor voice
    "professional": "af_bella",        # Clear, professional female voice
    "energetic": "af_heart",           # Warm confident voice (works well for news)
    "calm": "af_bella",               # Clear, calm female voice
    "deep": "am_adam",                 # Deep, authoritative male voice
    "female_professional": "af_bella",
    "female_energetic": "af_heart",
    "male_casual": "am_adam",
}

# Default voice — authoritative male anchor for news-commentary style
KOKORO_DEFAULT_VOICE = "am_adam"


def _generate_kokoro_tts(clean_text: str, voice_tone: str, filepath: Path) -> Optional[dict]:
    """
    Generate speech using Kokoro TTS (primary engine).
    Returns metadata dict on success, None on failure.
    """
    if not KOKORO_AVAILABLE:
        return None
    
    try:
        voice_pack = KOKORO_VOICES.get(voice_tone, KOKORO_DEFAULT_VOICE)
        print(f"  [KOKORO] Generating speech with voice '{voice_pack}'...")
        
        # Initialize Kokoro pipeline (American English)
        pipeline = KPipeline(lang_code='a')
        
        # Generate audio — Kokoro returns grapheme-level segments
        audio_segments = []
        sample_rate = 24000  # Kokoro default sample rate
        
        for graphemes, phonemes, audio in pipeline(clean_text, voice=voice_pack):
            if audio is not None:
                audio_segments.append(audio)
        
        if not audio_segments:
            print(f"  [KOKORO] No audio segments generated")
            return None
        
        # Concatenate all audio segments
        full_audio = np.concatenate(audio_segments)
        
        # Save as WAV first (Kokoro outputs numpy arrays)
        wav_path = filepath.with_suffix('.wav')
        sf.write(str(wav_path), full_audio, sample_rate)
        
        # Convert WAV to MP3 using moviepy for smaller file size
        try:
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            clip = AudioFileClip(str(wav_path))
            clip.write_audiofile(str(filepath), verbose=False, logger=None)
            clip.close()
            # Remove temporary WAV
            wav_path.unlink(missing_ok=True)
        except Exception:
            # If MP3 conversion fails, just use the WAV
            filepath = wav_path
            print(f"  [KOKORO] Using WAV format (MP3 conversion failed)")
        
        duration = len(full_audio) / sample_rate
        file_size = filepath.stat().st_size
        
        print(f"  [KOKORO] Generated {duration:.1f}s of natural speech")
        
        return {
            "success": True,
            "filename": filepath.name,
            "path": str(filepath),
            "voice": f"kokoro_{voice_pack}",
            "voice_tone": voice_tone,
            "text_length": len(clean_text),
            "word_count": len(clean_text.split()),
            "file_size_bytes": file_size,
            "estimated_duration_seconds": round(duration, 2),
            "audio_mastered": False,
            "output_directory": str(OUTPUT_DIR),
            "engine": "kokoro",
        }
    
    except Exception as e:
        print(f"  [KOKORO] Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _generate_edge_tts(clean_text: str, voice_tone: str, filepath: Path) -> Optional[dict]:
    """
    Generate speech using Edge TTS (fallback engine).
    Returns metadata dict on success, None on failure.
    """
    voice = VOICE_MAPPING.get(voice_tone, VOICE_MAPPING["authoritative"])
    voice_params = _get_voice_parameters(voice_tone)
    
    try:
        async def generate_audio():
            communicate = edge_tts.Communicate(
                clean_text, 
                voice,
                rate=voice_params["rate"],
                pitch=voice_params["pitch"]
            )
            await communicate.save(str(filepath))
        
        # Retry logic for Edge TTS
        max_retries = 3
        for attempt in range(max_retries):
            try:
                asyncio.run(generate_audio())
                break
            except Exception as e:
                if "403" in str(e) and attempt < max_retries - 1:
                    print(f"  [WARN] Edge TTS 403 error, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(1)
                    continue
                else:
                    raise e
        
        if not filepath.exists():
            return None
        
        file_size = filepath.stat().st_size
        word_count = len(clean_text.split())
        estimated_duration = word_count / 2.5
        
        return {
            "success": True,
            "filename": filepath.name,
            "path": str(filepath),
            "voice": voice,
            "voice_tone": voice_tone,
            "text_length": len(clean_text),
            "word_count": word_count,
            "file_size_bytes": file_size,
            "estimated_duration_seconds": round(estimated_duration, 2),
            "audio_mastered": False,
            "output_directory": str(OUTPUT_DIR),
            "engine": "edge_tts",
        }
    
    except Exception as e:
        print(f"  [TTS] Edge TTS failed: {e}")
        return None


@server.tool()
def generate_voiceover(text: str, voice_tone: str = "authoritative") -> dict:
    """
    Generate high-fidelity voiceover using Kokoro TTS (primary) or Edge TTS (fallback).
    
    Args:
        text: Script text to convert to speech
        voice_tone: Voice style (authoritative, professional, energetic, calm, deep, 
                   female_professional, female_energetic, male_casual)
    
    Returns:
        dict with status, audio file path, and metadata
    """
    
    if not text or not text.strip():
        return {
            "success": False,
            "error": "Text cannot be empty"
        }
    
    # Phase 5.3: Auto-select best voice tone if caller used default
    if voice_tone == "authoritative":
        voice_tone = select_voice_for_content(text, default_tone="authoritative")

    # Clean the text (remove formatting, underscores, etc.)
    clean_text = _add_natural_pacing(text)
    
    filename = f"voiceover_{voice_tone}_{hash(text) % 10000}.mp3"
    filepath = OUTPUT_DIR / filename
    
    # === PRIMARY: Kokoro TTS (natural human-like speech) ===
    if KOKORO_AVAILABLE:
        print(f"  [TTS] Trying Kokoro TTS (primary engine)...")
        result = _generate_kokoro_tts(clean_text, voice_tone, filepath)
        
        if result and result.get('success'):
            # Apply audio mastering
            mastered = _apply_audio_mastering(Path(result['path']))
            if mastered:
                print(f"  [TTS] Audio mastering applied")
                result['audio_mastered'] = True
                result['file_size_bytes'] = Path(result['path']).stat().st_size
            
            # Get word timestamps for subtitle sync
            result['word_timestamps'] = _get_faster_whisper_timestamps(result['path'], clean_text)
            return result
    
    # === FALLBACK: Edge TTS ===
    print(f"  [TTS] Kokoro unavailable/failed, trying Edge TTS (fallback)...")
    result = _generate_edge_tts(clean_text, voice_tone, filepath)
    
    if result and result.get('success'):
        # Apply audio mastering
        mastered = _apply_audio_mastering(Path(result['path']))
        if mastered:
            print(f"  [TTS] Audio mastering applied")
            result['audio_mastered'] = True
            result['file_size_bytes'] = Path(result['path']).stat().st_size
        
        # Get word timestamps for subtitle sync
        result['word_timestamps'] = _get_faster_whisper_timestamps(result['path'], clean_text)
        return result
    
    # === LAST RESORT: Silent audio ===
    print(f"  [TTS] All TTS engines failed, generating silent audio fallback...")
    try:
        from moviepy.audio.AudioClip import AudioArrayClip
        
        word_count = len(text.split())
        estimated_duration = max(word_count / 2.5, 10.0)
        fps = 44100
        
        n_samples = int(estimated_duration * fps)
        silence = np.zeros((n_samples, 1), dtype=np.float32)
        
        audio_clip = AudioArrayClip(silence, fps=fps)
        audio_clip.write_audiofile(str(filepath), verbose=False, logger=None)
        audio_clip.close()
        
        file_size = filepath.stat().st_size
        
        return {
            "success": True,
            "filename": filepath.name,
            "path": str(filepath),
            "voice": "fallback_silent",
            "voice_tone": voice_tone,
            "text_length": len(text),
            "word_count": word_count,
            "file_size_bytes": file_size,
            "estimated_duration_seconds": round(estimated_duration, 2),
            "audio_mastered": False,
            "output_directory": str(OUTPUT_DIR),
            "fallback_used": True,
        }
            
    except Exception as fallback_error:
        return {
            "success": False,
            "error": f"All TTS engines failed. Silent fallback also failed: {str(fallback_error)}",
            "voice_tone": voice_tone
        }


def _get_faster_whisper_timestamps(audio_path: str, text: str) -> list:
    """
    Use faster-whisper direct word timestamps for high-precision alignment.
    Bypasses whisperX's VAD/pyannote/torchcodec issues.
    ~10-20s on CPU for 60s audio, ~5s on GPU.
    Returns list of {'word': str, 'start': float, 'end': float}.
    """
    if not FASTER_WHISPER_AVAILABLE:
        print(f"  [TTS] faster-whisper not available, using calibrated estimation")
        return _get_calibrated_timestamps(text, audio_path)
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"
        
        # Check if real audio file exists and has content
        if not os.path.exists(audio_path):
            print(f"  [TTS] Audio file not found, using estimation")
            return _get_calibrated_timestamps(text, None)
        
        # 1. Load faster-whisper model (base = fast, good accuracy)
        print(f"  [TTS] faster-whisper: Loading model ({device})...")
        model = WhisperModel("base", device=device, compute_type=compute_type)
        
        # 2. Transcribe with word timestamps enabled
        print(f"  [TTS] faster-whisper: Running transcription with word timestamps...")
        segments, info = model.transcribe(
            audio_path,
            language="en",
            word_timestamps=True,
            beam_size=5,
            vad_filter=False,  # Skip VAD to avoid pyannote issues
            condition_on_previous_text=True
        )
        
        # 3. Extract word-level timestamps
        timestamps = []
        for segment in segments:
            for word_data in segment.words:
                timestamps.append({
                    'word': word_data.word.strip(),
                    'start': round(word_data.start, 3),
                    'end': round(word_data.end, 3),
                })
        
        if timestamps:
            print(f"  [TTS] faster-whisper: {len(timestamps)} word timestamps (~50ms accuracy)")
        else:
            print(f"  [TTS] faster-whisper: No word timestamps returned, using calibration")
            return _get_calibrated_timestamps(text, audio_path)
        return timestamps
    
    except Exception as e:
        print(f"  [TTS] faster-whisper failed: {e}, using calibrated estimation")
        return _get_calibrated_timestamps(text, audio_path)


def _get_calibrated_timestamps(text: str, audio_path: str = None) -> list:
    """
    Calibrated word timestamps based on audio file duration.
    Reads actual audio file to get precise duration, then distributes words evenly.
    Much better than static 2.5 words/sec estimation.
    """
    import moviepy.audio.io.AudioFileClip as afc
    
    # Get actual audio duration
    if audio_path and os.path.exists(audio_path):
        try:
            clip = afc.AudioFileClip(audio_path)
            duration = clip.duration
            clip.close()
        except:
            duration = len(text.split()) / 2.5  # Fallback
    else:
        duration = len(text.split()) / 2.5
    
    words = text.split()
    n_words = len(words)
    
    # Calculate dynamic word rate based on duration
    words_per_sec = n_words / duration if duration > 0 else 2.5
    print(f"  [TTS] Calibrated timestamps: {n_words} words, {duration:.1f}s = {words_per_sec:.2f} words/sec")
    
    # Distribute words evenly with small gaps for natural spacing
    timestamps = []
    gap_duration = duration * 0.02  # 2% of duration for pauses
    word_duration = (duration - (gap_duration * n_words)) / n_words if n_words > 0 else duration
    
    current_time = 0
    for word in words:
        timestamps.append({
            'word': word,
            'start': round(current_time, 3),
            'end': round(current_time + word_duration, 3),
        })
        current_time += word_duration + gap_duration
    
    return timestamps


def _get_edge_tts_timestamps(text: str) -> list:
    """
    Fallback: try edge-tts WordBoundary events.
    Falls back to sentence-based estimation if WordBoundary not available.
    """
    try:
        communicate = edge_tts.Communicate(text)
        words = []
        
        async def _fetch_words():
            async for chunk in communicate.stream():
                if chunk["type"] == "WordBoundary":
                    words.append({
                        'word': chunk.get('text', '').strip(),
                        'start': chunk.get('offset', 0) / 10_000_000,
                        'end': (chunk.get('offset', 0) + chunk.get('duration', 0)) / 10_000_000,
                    })
        
        asyncio.run(_fetch_words())
        
        if words:
            print(f"  [TTS] Edge-tts WordBoundary: {len(words)} words")
            return words
        
        # Final fallback: estimation
        print(f"  [TTS] WordBoundary not available, estimating...")
        return _estimate_word_timestamps(text)
    
    except Exception as e:
        print(f"  [TTS] Edge-tts WordBoundary failed: {e}, estimating...")
        return _estimate_word_timestamps(text)


def _estimate_word_timestamps(text: str, words_per_sec: float = 2.5) -> list:
    """
    Fallback: estimate word timestamps from text length.
    Used when edge-tts boundary data is unavailable.
    """
    words = text.split()
    timestamps = []
    for i, word in enumerate(words):
        start = i / words_per_sec
        end = (i + 1) / words_per_sec
        timestamps.append({'word': word, 'start': start, 'end': end})
    return timestamps
