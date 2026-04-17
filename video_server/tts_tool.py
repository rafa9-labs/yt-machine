import os
import asyncio
import time
import re
import tempfile
import shutil
import requests as http_requests
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

# ElevenLabs TTS — highest quality natural voice (primary engine)
ELEVENLABS_VOICE_ID = "KDkfb6swL2m7BPf1K53e"
ELEVENLABS_MODEL = "eleven_multilingual_v2"
ELEVENLABS_SETTINGS = {
    "stability": 0.45,
    "similarity_boost": 0.70,
    "style_exaggeration": 0.45,
    "use_speaker_boost": True,
}

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


def _find_ffmpeg() -> Optional[str]:
    """
    Find ffmpeg executable — checks system PATH, then falls back to
    imageio_ffmpeg (bundled with moviepy/imageio). Returns full path
    to ffmpeg binary, or None if not found anywhere.
    """
    # 1. Check system PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    # 2. Try imageio_ffmpeg (bundled with moviepy)
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_path and os.path.isfile(ffmpeg_path):
            return ffmpeg_path
    except (ImportError, Exception):
        pass
    
    return None

# Cache the ffmpeg path on first lookup
_FFMPEG_PATH = None

def _get_ffmpeg() -> Optional[str]:
    """Get cached ffmpeg path (resolved once, reused)."""
    global _FFMPEG_PATH
    if _FFMPEG_PATH is None:
        _FFMPEG_PATH = _find_ffmpeg()
        if _FFMPEG_PATH:
            print(f"  [TTS] ffmpeg found: {_FFMPEG_PATH}")
        else:
            print(f"  [TTS] ffmpeg NOT found — audio mastering unavailable")
    return _FFMPEG_PATH


def _apply_audio_mastering(input_path: Path) -> bool:
    """
    Apply professional audio mastering to an audio file.
    Normalises loudness and ensures consistent levels.
    Uses soundfile for reliable WAV I/O (avoids moviepy array stacking issues).

    Args:
        input_path: Path to the audio file to master in-place

    Returns:
        True if mastering was applied, False if skipped
    """
    try:
        import numpy as np
        import soundfile as sf
        import tempfile
        import subprocess

        # Read audio directly with soundfile (supports WAV/FLAC/OGG, not MP3)
        # For MP3 files, fall back to ffmpeg → WAV → process → MP3
        suffix = input_path.suffix.lower()
        
        # Resolve ffmpeg path
        ffmpeg = _get_ffmpeg()
        if not ffmpeg:
            print(f"  [TTS] Mastering skipped: ffmpeg not available")
            return False
        
        if suffix == '.mp3':
            # Convert MP3 → temp WAV with ffmpeg for reliable reading
            temp_wav = input_path.with_suffix('.master.wav')
            subprocess.run(
                [ffmpeg, '-y', '-i', str(input_path), str(temp_wav)],
                capture_output=True, timeout=30
            )
            if not temp_wav.exists():
                print(f"  [TTS] Mastering: ffmpeg MP3→WAV conversion failed")
                return False
            arr, sr = sf.read(str(temp_wav))
        else:
            temp_wav = None
            arr, sr = sf.read(str(input_path))

        if arr is None or arr.size == 0:
            return False

        # Ensure 2D (samples × channels)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        # 1. High-pass filter at 80Hz (remove rumble, warm up voice)
        for ch in range(arr.shape[1]):
            n_samples = len(arr[:, ch])
            spectrum = np.fft.rfft(arr[:, ch])
            freqs = np.fft.rfftfreq(n_samples, 1.0 / sr)
            hp_mask = freqs < 80
            spectrum[hp_mask] *= 0.1  # -20dB below 80Hz
            presence_mask = (freqs >= 3000) & (freqs <= 5000)
            spectrum[presence_mask] *= 1.41  # +3dB presence boost for crispness
            arr[:, ch] = np.fft.irfft(spectrum, n=len(arr[:, ch]))
        
        # 2. Gentle compression (reduce dynamic range for consistent volume)
        threshold = 0.5
        ratio = 3.0
        for ch in range(arr.shape[1]):
            over = np.abs(arr[:, ch]) - threshold
            over = np.maximum(over, 0)
            gain_reduction = 1.0 - (over * (1.0 - 1.0 / ratio))
            gain_reduction = np.clip(gain_reduction, 0.3, 1.0)
            arr[:, ch] *= gain_reduction
        
        # 3. Peak normalize to -3 dBFS (prevent clipping)
        peak = np.max(np.abs(arr))
        if peak > 0:
            target_peak = 0.708  # -3 dBFS
            arr = arr * (target_peak / peak)

        # 4. Loudness normalize to ~-16 LUFS (YouTube-optimized for short-form)
        rms = np.sqrt(np.mean(arr ** 2))
        target_rms = 0.15  # slightly louder than broadcast standard
        if rms > 0:
            rms_gain = min(target_rms / rms, 3.0)
            arr = np.clip(arr * rms_gain, -1.0, 1.0)

        # 5. Noise gate: silence any sample below -70dB (kills residual noise floor)
        noise_gate_threshold = 10 ** (-70 / 20)  # -70dB
        arr[np.abs(arr) < noise_gate_threshold] = 0.0

        # 6. Write mastered audio back
        if suffix == '.mp3':
            # Write WAV, then convert back to MP3 with ffmpeg
            sf.write(str(temp_wav), arr, sr)
            subprocess.run(
                [ffmpeg, '-y', '-i', str(temp_wav), '-codec:a', 'libmp3lame', 
                 '-qscale:a', '2', str(input_path)],
                capture_output=True, timeout=30
            )
            temp_wav.unlink(missing_ok=True)
        else:
            sf.write(str(input_path), arr, sr)

        return True

    except Exception as e:
        print(f"  [TTS] Audio mastering skipped: {e}")
        return False

def _normalize_numbers_for_speech(text: str) -> str:
    """
    Pre-process numbers in text for better TTS pronunciation.
    Ported from prosody_processor.py to work with Kokoro TTS path.
    """
    # Dollar amounts: $70B → 70 billion dollars, $112 → 112 dollars
    text = re.sub(r'\$(\d+)B\b', r'\1 billion dollars', text)
    text = re.sub(r'\$(\d+)M\b', r'\1 million dollars', text)
    text = re.sub(r'\$(\d+)', r'\1 dollars', text)
    
    # Percentages: 15% → 15 percent
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    
    # Large numbers with commas: 3,000 → three thousand
    def _expand_number(match):
        num_str = match.group(1).replace(',', '')
        try:
            num = int(num_str)
            if num >= 1_000_000_000:
                return f'{num // 1_000_000_000} billion {num % 1_000_000_000}'.strip()
            elif num >= 1_000_000:
                return f'{num // 1_000_000} million {num % 1_000_000}'.strip()
            elif num >= 1_000:
                return f'{num // 1_000} thousand {num % 1_000}'.strip()
        except ValueError:
            pass
        return match.group(1)  # keep original if parsing fails
    text = re.sub(r'\b(\d{1,3}(,\d{3})+)\b', _expand_number, text)
    
    # Decimals (but not years or versions): 3.5 → 3 point 5
    text = re.sub(r'(\d+)\.(\d+)(?!\d)', lambda m: f"{m.group(1)} point {m.group(2)}", text)
    
    return text


def _add_natural_pacing(text: str, engine: str = "kokoro") -> str:
    """
    Clean and prepare script text for TTS input.
    
    Handles:
    - Strip markdown / formatting artifacts
    - Convert story separators (....) to comma-based long pauses
    - Em-dash (—) to comma pause for TTS compatibility
    - Normalize numbers for speech (percent, dollars, large numbers)
    - Strip LLM preamble text
    - Keep natural punctuation (! ? : , .) intact
    
    Args:
        text: Original script text
        engine: Target TTS engine ("kokoro" or "edge"). Both get the same treatment now.
    
    Returns:
        Clean plain text ready for neural TTS
    """
    # ── STRIP CURATOR/LLM PREAMBLES ──
    # Curator LLMs love to prepend meta-text. Strip it aggressively.
    preamble_patterns = [
        r"^here(?:'s|\s+is)\s+(?:your|the)\s+(?:transformed|curated|final|edited)\s+(?:news\s+)?script:?\s*",
        r"^here (?:it is|you go|is the script):?\s*",
        r"^(?:sure|certainly|here|ok)[,!]\s*(?:here'?s|i'?ve|i have)\s+",
        r"^here'?s\s+the\s+(?:curated|final|edited|updated)\s+(?:version|script|text):?\s*",
        r"^i(?:'ve| have)\s+(?:curated|edited|transformed|prepared)\s+",
        r"^the\s+(?:curated|edited|final)\s+(?:script|text|version)\s+(?:is|reads):?\s*",
    ]
    text_lower = text.lower()
    for pattern in preamble_patterns:
        match = re.match(pattern, text_lower)
        if match:
            text = text[match.end():]
            break
    
    # Nuclear fallback: if first line is short and doesn't contain story words, strip it
    lines = text.split('\n', 1)
    if len(lines) > 1:
        first_line = lines[0].strip().lower()
        story_words = ['good', 'hey', 'welcome', 'tonight', 'today', 'masker', 
                       'iran', 'russia', 'china', 'us', 'turkey', 'france', 'ukraine',
                       'the', 'so', 'but', 'and', 'well', 'alright']
        if (len(first_line) < 80 and 
            (not first_line.endswith(('!', '?', '.')) or
             not any(w in first_line for w in story_words))):
            text = lines[1]
    
    # ── STRIP FORMATTING ARTIFACTS ──
    text = re.sub(r'<[^>]+>', '', text)          # SSML/XML tags
    text = re.sub(r'_+', ' ', text)               # Underscores
    text = re.sub(r'={2,}', ' ', text)            # Equals signs
    text = re.sub(r'-{3,}', ' ', text)            # Triple+ dashes
    text = re.sub(r'[#`]', '', text)              # Markdown headers and backticks
    text = re.sub(r'[*]', '', text)               # Asterisks
    
    # ── PAUSE HANDLING ──
    # Story separator (....): strip entirely — silence buffers between batches handle pauses.
    # Avoids ElevenLabs interpreting comma chains as vocalized "CHET" sounds.
    text = re.sub(r'\.{4,}\s*', '.  ', text)
    
    # Dramatic pause (...): convert to period + double space for natural sentence break.
    # Double space causes a natural TTS pause without vocalized artifacts.
    text = re.sub(r'\.{3}\s*', '.  ', text)
    
    # Em-dash (—): convert to comma pause (all TTS engines respect commas)
    text = text.replace('—', ', ')
    
    # ── NUMBER NORMALIZATION ──
    text = _normalize_numbers_for_speech(text)
    
    # ── LIGHT CLEANUP ──
    text = re.sub(r'[\r\n]+', ' ', text)          # Collapse newlines
    text = re.sub(r' {2,}', ' ', text)            # Collapse multiple spaces
    text = text.strip()
    
    # ── SAFETY: ensure text ends with punctuation ──
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text

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
        
        # Concatenate all audio segments with 200ms silence buffers
        # Kokoro segments can cut abruptly; silence prevents word blending
        silence_duration = 0.20  # 200ms natural pause
        silence_samples = int(sample_rate * silence_duration)
        # Use zero silence with fade envelope — no noise floor artifacts
        silence = np.zeros(silence_samples, dtype=np.float32)
        fade_len = min(50, silence_samples // 2)
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
        silence[:fade_len] *= fade_in
        silence[-fade_len:] *= fade_out
        
        parts = []
        for i, seg in enumerate(audio_segments):
            parts.append(seg)
            if i < len(audio_segments) - 1:
                parts.append(silence)
        
        full_audio = np.concatenate(parts)
        print(f"  [KOKORO] Added {silence_duration}s silence x {len(audio_segments)-1} between segments")
        
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


def _generate_elevenlabs_tts(clean_text: str, voice_tone: str, filepath: Path) -> Optional[dict]:
    """
    Generate speech using ElevenLabs API (highest quality natural voice).
    Uses batch generation (1-3 sentences per call) for natural variation.
    Settings optimized per creator research: stability 45%, similarity 70%, style 45%.
    """
    api_key = os.environ.get("ELEVEN_LABS_KEY", "").strip()
    if not api_key:
        return None
    
    try:
        voice_id = ELEVENLABS_VOICE_ID
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        
        # ── BATCH GENERATION ──
        # Split into sentences, then group 1-3 per batch for natural variation
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        batches = []
        current_batch = []
        current_words = 0
        
        for sentence in sentences:
            words = len(sentence.split())
            if current_batch and (len(current_batch) >= 3 or current_words + words > 35):
                batches.append(' '.join(current_batch))
                current_batch = []
                current_words = 0
            current_batch.append(sentence)
            current_words += words
        
        if current_batch:
            batches.append(' '.join(current_batch))
        
        print(f"  [ELEVENLABS] Generating in {len(batches)} batches (voice: {voice_id[:8]}...)")
        
        audio_segments = []
        for i, batch_text in enumerate(batches):
            payload = {
                "text": batch_text,
                "model_id": ELEVENLABS_MODEL,
                "voice_settings": ELEVENLABS_SETTINGS,
            }
            
            resp = http_requests.post(url, json=payload, headers=headers, timeout=60)
            
            if resp.status_code != 200:
                print(f"  [ELEVENLABS] Batch {i+1}/{len(batches)} failed: HTTP {resp.status_code}")
                return None
            
            temp_path = filepath.with_suffix(f'.batch_{i}.mp3')
            temp_path.write_bytes(resp.content)
            audio_segments.append(str(temp_path))
        
        if not audio_segments:
            return None
        
        # ── CONCATENATE BATCHES ──
        if len(audio_segments) == 1:
            Path(audio_segments[0]).rename(filepath)
        else:
            # Use numpy-based concatenation with silence buffers between batches
            # This prevents word cutting at batch boundaries
            batch_arrays = []
            for p in audio_segments:
                arr, sr = sf.read(str(p))
                batch_arrays.append(arr)
            
            # Create 300ms silence buffer at native sample rate
            silence_duration = 0.30  # 300ms natural pause between batches
            silence_samples = int(sr * silence_duration)
            fade_len = min(50, silence_samples // 2)
            fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
            fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
            
            # Determine channels from first batch
            if batch_arrays[0].ndim == 1:
                # Zero silence with fade envelope — no noise floor artifacts
                silence = np.zeros(silence_samples, dtype=np.float32)
                silence[:fade_len] *= fade_in
                silence[-fade_len:] *= fade_out
            else:
                n_ch = batch_arrays[0].shape[1]
                silence = np.zeros((silence_samples, n_ch), dtype=np.float32)
                silence[:fade_len] *= fade_in[:, None]
                silence[-fade_len:] *= fade_out[:, None]
            
            # Interleave batches with silence
            parts = []
            for i, arr in enumerate(batch_arrays):
                parts.append(arr)
                if i < len(batch_arrays) - 1:
                    parts.append(silence)
            
            full_audio = np.concatenate(parts, axis=0)
            sf.write(str(filepath), full_audio, sr)
            
            print(f"  [ELEVENLABS] Added {silence_duration}s silence × {len(batches)-1} between batches")
            
            for p in audio_segments:
                Path(p).unlink(missing_ok=True)
        
        # Get duration
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        clip = AudioFileClip(str(filepath))
        duration = clip.duration
        clip.close()
        
        file_size = filepath.stat().st_size
        word_count = len(clean_text.split())
        
        print(f"  [ELEVENLABS] Generated {duration:.1f}s of natural speech ({len(batches)} batches)")
        
        return {
            "success": True,
            "filename": filepath.name,
            "path": str(filepath),
            "voice": f"elevenlabs_{voice_id[:8]}",
            "voice_tone": voice_tone,
            "text_length": len(clean_text),
            "word_count": word_count,
            "file_size_bytes": file_size,
            "estimated_duration_seconds": round(duration, 2),
            "audio_mastered": False,
            "output_directory": str(OUTPUT_DIR),
            "engine": "elevenlabs",
        }
    
    except Exception as e:
        print(f"  [ELEVENLABS] Failed: {e}")
        for p in filepath.parent.glob(f"{filepath.stem}.batch_*.mp3"):
            p.unlink(missing_ok=True)
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

    # Determine target engine
    target_engine = "kokoro" if KOKORO_AVAILABLE else "edge"
    
    # Clean the text — light formatting cleanup, keep punctuation intact
    # Both engines get the same treatment: natural punctuation, no synthetic markers
    clean_text = _add_natural_pacing(text, engine=target_engine)
    
    filename = f"voiceover_{voice_tone}_{hash(text) % 10000}.mp3"
    filepath = OUTPUT_DIR / filename
    
    # === PRIMARY: ElevenLabs (highest quality natural voice) ===
    result = _generate_elevenlabs_tts(clean_text, voice_tone, filepath)
    
    if result and result.get('success'):
        mastered = _apply_audio_mastering(Path(result['path']))
        if mastered:
            print(f"  [TTS] Audio mastering applied (EQ + compression)")
            result['audio_mastered'] = True
            result['file_size_bytes'] = Path(result['path']).stat().st_size
        
        result['word_timestamps'] = _get_faster_whisper_timestamps(result['path'], clean_text)
        return result
    
    # === FALLBACK 1: Kokoro TTS ===
    if KOKORO_AVAILABLE:
        print(f"  [TTS] ElevenLabs unavailable, trying Kokoro TTS...")
        result = _generate_kokoro_tts(clean_text, voice_tone, filepath)
        
        if result and result.get('success'):
            mastered = _apply_audio_mastering(Path(result['path']))
            if mastered:
                print(f"  [TTS] Audio mastering applied")
                result['audio_mastered'] = True
                result['file_size_bytes'] = Path(result['path']).stat().st_size
            
            result['word_timestamps'] = _get_faster_whisper_timestamps(result['path'], clean_text)
            return result
    
    # === FALLBACK 2: Edge TTS ===
    print(f"  [TTS] All primary engines failed, trying Edge TTS (last resort)...")
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
