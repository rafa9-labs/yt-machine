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

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# faster-whisper for high-precision word timestamps (bypasses whisperX's VAD issues)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("  [TTS] faster-whisper not installed, will use fallback timestamps")

server = FastMCP("tts-tool")

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ElevenLabs TTS — primary engine (highest quality natural voice)
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "G2BYBzpEHIacF1Bva0XL")
ELEVENLABS_MODEL = "eleven_multilingual_v2"
ELEVENLABS_SETTINGS = {
    "stability": 0.35,
    "similarity_boost": 0.70,
    "style_exaggeration": 0.65,
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

# Kokoro TTS — local free engine (runs on GPU/CPU)
KOKORO_VOICE_MAP = {
    "authoritative": "am_adam",
    "professional": "af_breeze",
    "energetic": "af_heart",
    "calm": "af_sky",
    "deep": "am_michael",
    "female_professional": "af_breeze",
    "female_energetic": "af_nova",
    "male_casual": "am_adam",
}
KOKORO_SPEED_MAP = {
    "authoritative": 1.0,
    "professional": 1.0,
    "energetic": 1.1,
    "calm": 0.95,
    "deep": 0.9,
    "female_professional": 1.0,
    "female_energetic": 1.1,
    "male_casual": 1.05,
}
USE_KOKORO = os.getenv("USE_KOKORO", "auto").lower() in ("true", "1", "yes", "auto")
_kokoro_pipeline = None


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


def _detect_outro_segment(text: str):
    """
    Split full script text into main content and outro segment.
    The outro is identified by the last '....' separator and trailing text.
    Returns (main_text, outro_text).
    """
    parts = re.split(r'\.{4,}', text)
    if len(parts) >= 2:
        main = '....'.join(parts[:-1])
        outro = parts[-1].strip()
        return main, outro
    return text, ""


def _apply_outro_tts_settings(voice_settings: dict, is_outro: bool = False) -> dict:
    """
    Modify TTS voice settings for outro segments.
    Creates a melancholy, natural spoken tone: slower, softer, more stable.
    """
    if not is_outro:
        return voice_settings

    settings = voice_settings.copy()

    # ElevenLabs: higher stability, lower style for melancholy
    if 'stability' in settings:
        settings['stability'] = min(1.0, float(settings.get('stability', 0.35)) + 0.25)
    if 'style_exaggeration' in settings:
        settings['style_exaggeration'] = max(0.0, float(settings.get('style_exaggeration', 0.65)) - 0.35)
    if 'similarity_boost' in settings:
        settings['similarity_boost'] = min(1.0, float(settings.get('similarity_boost', 0.70)) + 0.05)

    # Edge TTS: slower rate, lower pitch
    settings['rate'] = "-15%"
    settings['pitch'] = "-3Hz"

    # Kokoro: slower speed
    settings['speed'] = 0.85

    return settings


def _apply_outro_reverb(audio_path: str) -> str:
    """
    Apply subtle reverb tail ONLY to the last ~5 seconds (outro segment).
    Leaves the rest of the audio untouched.
    ffmpeg aecho: 0.8 gain, 0.88 feedback, 60ms delay, 0.4 decay.
    """
    ffmpeg_exe = _get_ffmpeg()
    if not ffmpeg_exe:
        return audio_path

    ext = audio_path.rsplit('.', 1)[-1] if '.' in audio_path else 'wav'
    output_path = audio_path.rsplit('.', 1)[0] + '_outro_reverb.' + ext

    try:
        import subprocess
        import json

        probecmd = [ffmpeg_exe, '-v', 'quiet', '-print_format', 'json', '-show_format', audio_path]
        probe = subprocess.run(probecmd, capture_output=True, timeout=15, text=True)
        duration = float(json.loads(probe.stdout)['format']['duration'])
        split_at = max(0, duration - 5)

        cmd = [
            ffmpeg_exe, '-y',
            '-i', audio_path,
            '-filter_complex',
            f'[0:a]atrim=0:{split_at},asetpts=PTS-STARTPTS[body];'
            f'[0:a]atrim=start={split_at},aecho=0.8:0.88:60:0.4,asetpts=PTS-STARTPTS[outro];'
            f'[body][outro]concat=n=2:v=0:a=1[out]',
            '-map', '[out]',
            '-ar', '44100',
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60, text=True)
        if result.returncode == 0 and Path(output_path).exists():
            try:
                Path(audio_path).unlink()
            except Exception:
                pass
            return output_path
    except Exception as e:
        print(f"  [TTS] Warning: Outro reverb failed: {e}")
    return audio_path


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
    Apply professional audio mastering using ffmpeg built-in DSP filters.
    Uses proper bandpass filtering (not crude FFT), compressor with attack/release
    (not hard sample limiter), and loudness normalization to -16 LUFS.
    """
    try:
        ffmpeg = _get_ffmpeg()
        if not ffmpeg:
            print(f"  [TTS] Mastering skipped: ffmpeg not available")
            return False

        import subprocess

        suffix = input_path.suffix.lower()
        tmp_path = input_path.with_suffix('.master_tmp' + suffix)

        filter_chain = (
            'highpass=f=80:t=0.7071,'
            'equalizer=f=4000:t=q:w=1.2:g=2,'
            'acompressor=threshold=0.125:ratio=2:attack=10:release=100:knee=4:makeup=1,'
            'loudnorm=I=-16:LRA=11:TP=-1.5'
        )

        cmd = [
            ffmpeg, '-y',
            '-i', str(input_path),
            '-af', filter_chain,
            '-ar', '44100',
            str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60, text=True)

        if result.returncode == 0 and tmp_path.exists():
            input_path.unlink(missing_ok=True)
            tmp_path.rename(input_path)
            return True
        else:
            tmp_path.unlink(missing_ok=True)
            return False

    except Exception as e:
        print(f"  [TTS] Mastering failed ({e}) — using unprocessed audio")
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
    text = re.sub(r'\*\[.*?\]\*', '', text)       # Glitch markers: *[system_warning]* etc.
    text = re.sub(r'\[STORY\s*\d+\]\s*', '', text)  # Story markers: [STORY 1], [STORY 2] etc.
    text = re.sub(r'[\[\]]', '', text)            # Leftover brackets
    text = re.sub(r'[*]', '', text)               # Remaining asterisks
    
    # ── PAUSE HANDLING ──
    # Story separator (....): stripped entirely — structural splitting in generate_voiceover()
    # handles inter-story pauses (0.8s silence). This avoids ElevenLabs interpreting
    # dot chains as vocalized "CHET" sounds or arbitrary batch-boundary pauses.
    text = re.sub(r'\.{4,}\s*', '. ', text)
    
    # Dramatic pause (...): convert to period for natural sentence break.
    # TTS engines naturally pause at periods; no special silence injection needed.
    text = re.sub(r'\.{3}\s*', '. ', text)
    
    # Em-dash (—): convert to comma pause (all TTS engines respect commas)
    text = text.replace('—', ', ')
    
    # ── NUMBER NORMALIZATION ──
    text = _normalize_numbers_for_speech(text)
    
    # ── GREETING NORMALIZATION ──
    # Prevent TTS from interpreting "Ssssmokin'" as a slow sibilant hiss
    text = re.sub(r"[Ss]{3,}mokin'", "Smokin'", text, flags=re.IGNORECASE)
    text = re.sub(r"[Ss]{3,}mokin", "Smokin", text, flags=re.IGNORECASE)
    
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


def _generate_kokoro_tts(structural_chunks: list, voice_tone: str, filepath: Path, has_outro: bool = False) -> Optional[dict]:
    """
    Generate speech using Kokoro TTS (local, free, GPU-accelerated).
    
    Uses structural chunk splitting for proper pause handling:
      - 0.80s silence between structural chunks (story transitions)
      - No batching needed within chunks — Kokoro handles full text natively
    
    Args:
        structural_chunks: List of cleaned text chunks, split at story boundaries
        voice_tone: Voice style string
        filepath: Output path for the final MP3
    """
    global _kokoro_pipeline
    
    try:
        from kokoro import KPipeline, KModel
        import torch
        import soundfile as sf
    except ImportError:
        print("  [KOKORO] kokoro or soundfile not installed — skipping")
        return None
    
    if not USE_KOKORO:
        return None
    
    try:
        voice_name = KOKORO_VOICE_MAP.get(voice_tone, "am_adam")
        speed = KOKORO_SPEED_MAP.get(voice_tone, 1.0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if _kokoro_pipeline is None:
            print(f"  [KOKORO] Loading pipeline (device={device})...")
            model = KModel()
            _kokoro_pipeline = KPipeline(lang_code='a', model=model, device=device)
            print("  [KOKORO] Pipeline ready")
        
        audio_per_chunk = []
        sr = None
        
        for i, chunk_text in enumerate(structural_chunks):
            if not chunk_text or not chunk_text.strip():
                audio_per_chunk.append(None)
                continue
            
            # Apply slower speed for outro chunk (melancholy feel)
            is_outro_chunk = has_outro and (i == len(structural_chunks) - 1)
            chunk_speed = 0.85 if is_outro_chunk else speed
            
            chunk_label = f"chunk {i+1}/{len(structural_chunks)}"
            if is_outro_chunk:
                chunk_label += " (outro)"
            print(f"  [KOKORO] Generating {chunk_label} ({len(chunk_text.split())} words)...")
            
            chunk_segments = []
            for _, _, audio in _kokoro_pipeline(chunk_text, voice=voice_name, speed=chunk_speed):
                if audio is not None:
                    chunk_segments.append(np.array(audio, dtype=np.float32))
            
            if sr is None and hasattr(_kokoro_pipeline, 'model') and hasattr(_kokoro_pipeline.model, 'sample_rate'):
                sr = _kokoro_pipeline.model.sample_rate
            elif sr is None:
                sr = 24000
            
            audio_per_chunk.append(chunk_segments if chunk_segments else None)
            print(f"  [KOKORO] {chunk_label}: {len(chunk_segments)} segments generated")
        
        valid_chunks = [c for c in audio_per_chunk if c is not None]
        if not valid_chunks:
            print("  [KOKORO] No audio produced")
            return None
        
        STRUCTURAL_SILENCE = 0.80
        silence_samples = int(sr * STRUCTURAL_SILENCE)
        fade_len = min(50, silence_samples // 2)
        structural_silence = np.zeros(silence_samples, dtype=np.float32)
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
        structural_silence[:fade_len] *= fade_in
        structural_silence[-fade_len:] *= fade_out
        
        CLOSING_SILENCE = 0.30
        closing_silence_samples = int(sr * CLOSING_SILENCE)
        closing_fade_len = min(50, closing_silence_samples // 2)
        closing_silence = np.zeros(closing_silence_samples, dtype=np.float32)
        closing_fade_in = np.linspace(0, 1, closing_fade_len, dtype=np.float32)
        closing_fade_out = np.linspace(1, 0, closing_fade_len, dtype=np.float32)
        closing_silence[:closing_fade_len] *= closing_fade_in
        closing_silence[-closing_fade_len:] *= closing_fade_out
        
        assembled_parts = []
        for i, chunk_segments in enumerate(audio_per_chunk):
            if chunk_segments is None:
                continue
            for seg in chunk_segments:
                assembled_parts.append(seg)
            if i < len(audio_per_chunk) - 1:
                if i == len(audio_per_chunk) - 2:
                    assembled_parts.append(closing_silence)
                else:
                    assembled_parts.append(structural_silence)
        
        assembled = np.concatenate(assembled_parts, axis=0)
        
        wav_path = filepath.with_suffix('.wav')
        sf.write(str(wav_path), assembled, sr)
        
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        wav_clip = AudioFileClip(str(wav_path))
        wav_clip.write_audiofile(str(filepath), verbose=False, logger=None)
        wav_clip.close()
        wav_path.unlink(missing_ok=True)
        
        clip = AudioFileClip(str(filepath))
        duration = clip.duration
        clip.close()
        
        full_clean_text = ' '.join(c for c in structural_chunks if c and c.strip())
        word_count = len(full_clean_text.split())
        file_size = filepath.stat().st_size
        
        print(f"  [KOKORO] Generated {duration:.1f}s of speech ({len(structural_chunks)} chunks)")
        
        return {
            "success": True,
            "filename": filepath.name,
            "path": str(filepath),
            "voice": f"kokoro_{voice_name}",
            "voice_tone": voice_tone,
            "text_length": len(full_clean_text),
            "word_count": word_count,
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
        _kokoro_pipeline = None
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


def _generate_elevenlabs_tts(structural_chunks: list, voice_tone: str, filepath: Path, has_outro: bool = False) -> Optional[dict]:
    """
    Generate speech using ElevenLabs API (highest quality natural voice).
    
    Uses two-phase batch splitting:
      1. Text is pre-split into structural chunks at story separator boundaries (....)
      2. Each chunk is batched normally (1-3 sentences per API call)
    
    Audio assembly uses two silence tiers:
      - 0.80s between structural chunks (story transitions)
      - 0.06s between intra-chunk batches (imperceptible micro-breath)
    
    Args:
        structural_chunks: List of cleaned text chunks, split at story boundaries
        voice_tone: Voice style string
        filepath: Output path for the final MP3
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
        
        all_batches = []
        chunk_boundaries = []
        
        for chunk_idx, chunk_text in enumerate(structural_chunks):
            if not chunk_text or not chunk_text.strip():
                continue
            
            sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
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
            
            # Merge last batch if it's a micro-fragment (< 4 words) into previous
            if len(batches) >= 2:
                last_wc = len(batches[-1].split())
                if last_wc < 4:
                    batches[-2] = f"{batches[-2]} {batches[-1]}"
                    batches.pop()
                    print(f"  [ELEVENLABS] Merged trailing micro-batch ({last_wc} words) into previous")
            
            start_idx = len(all_batches)
            all_batches.extend(batches)
            end_idx = len(all_batches)
            chunk_boundaries.append((start_idx, end_idx))
        
        if not all_batches:
            return None
        
        print(f"  [ELEVENLABS] Generating in {len(all_batches)} batches across {len(chunk_boundaries)} structural chunks (voice: {voice_id[:8]}...)")
        
        # Determine last chunk boundary for outro detection
        last_chunk_start_idx = 0
        last_chunk_end_idx = 0
        if has_outro and chunk_boundaries:
            last_chunk_start_idx, last_chunk_end_idx = chunk_boundaries[-1]

        audio_segments = []
        for i, batch_text in enumerate(all_batches):
            batch_voice_settings = dict(ELEVENLABS_SETTINGS)
            if i == 0:
                batch_voice_settings["stability"] = 0.55
            # Apply melancholy settings for outro batches (last structural chunk)
            if has_outro and i >= last_chunk_start_idx:
                batch_voice_settings = _apply_outro_tts_settings(batch_voice_settings, is_outro=True)
            payload = {
                "text": batch_text,
                "model_id": ELEVENLABS_MODEL,
                "voice_settings": batch_voice_settings,
            }
            
            resp = http_requests.post(url, json=payload, headers=headers, timeout=60)
            
            if resp.status_code != 200:
                print(f"  [ELEVENLABS] Batch {i+1}/{len(all_batches)} failed: HTTP {resp.status_code}")
                return None
            
            temp_path = filepath.with_suffix(f'.batch_{i}.mp3')
            temp_path.write_bytes(resp.content)
            audio_segments.append(str(temp_path))
        
        if not audio_segments:
            return None
        
        structural_after_indices = set()
        for chunk_idx, (start_idx, end_idx) in enumerate(chunk_boundaries):
            if chunk_idx < len(chunk_boundaries) - 1:
                structural_after_indices.add(end_idx - 1)
        
        STRUCTURAL_SILENCE = 0.80
        MICRO_BREATH = 0.06
        
        if len(audio_segments) == 1:
            Path(audio_segments[0]).rename(filepath)
        else:
            import soundfile as sf
            batch_arrays = []
            for p in audio_segments:
                arr, sr = sf.read(str(p))
                batch_arrays.append(arr)
            
            def _make_silence(duration_samples, sr, n_channels):
                fade_len = min(50, duration_samples // 2)
                fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
                fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
                if n_channels == 1:
                    sil = np.zeros(duration_samples, dtype=np.float32)
                    sil[:fade_len] *= fade_in
                    sil[-fade_len:] *= fade_out
                else:
                    sil = np.zeros((duration_samples, n_channels), dtype=np.float32)
                    sil[:fade_len] *= fade_in[:, None]
                    sil[-fade_len:] *= fade_out[:, None]
                return sil
            
            CLOSING_SILENCE = 0.30
            n_ch = 1 if batch_arrays[0].ndim == 1 else batch_arrays[0].shape[1]
            structural_sil = _make_silence(int(sr * STRUCTURAL_SILENCE), sr, n_ch)
            closing_sil = _make_silence(int(sr * CLOSING_SILENCE), sr, n_ch)
            micro_sil = _make_silence(int(sr * MICRO_BREATH), sr, n_ch)
            
            parts = []
            struct_count = 0
            closing_count = 0
            micro_count = 0
            for i, arr in enumerate(batch_arrays):
                parts.append(arr)
                if i < len(batch_arrays) - 1:
                    if i == len(batch_arrays) - 2 and (len(batch_arrays) - 2) in structural_after_indices:
                        parts.append(closing_sil)
                        closing_count += 1
                    elif i in structural_after_indices:
                        parts.append(structural_sil)
                        struct_count += 1
                    else:
                        parts.append(micro_sil)
                        micro_count += 1
            
            full_audio = np.concatenate(parts, axis=0)
            wav_path = filepath.with_suffix('.wav')
            sf.write(str(wav_path), full_audio, sr)
            
            from moviepy.audio.io.AudioFileClip import AudioFileClip as _AC
            wav_clip = _AC(str(wav_path))
            wav_clip.write_audiofile(str(filepath), verbose=False, logger=None)
            wav_clip.close()
            wav_path.unlink(missing_ok=True)
            
            print(f"  [ELEVENLABS] Silences: {struct_count}x{STRUCTURAL_SILENCE}s (story) + {closing_count}x{CLOSING_SILENCE}s (closing) + {micro_count}x{MICRO_BREATH}s (breath)")
            
            for p in audio_segments:
                Path(p).unlink(missing_ok=True)
        
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        clip = AudioFileClip(str(filepath))
        duration = clip.duration
        clip.close()
        
        file_size = filepath.stat().st_size
        full_clean_text = ' '.join(c for c in structural_chunks if c and c.strip())
        word_count = len(full_clean_text.split())
        
        print(f"  [ELEVENLABS] Generated {duration:.1f}s of natural speech ({len(all_batches)} batches, {len(chunk_boundaries)} chunks)")
        
        return {
            "success": True,
            "filename": filepath.name,
            "path": str(filepath),
            "voice": f"elevenlabs_{voice_id[:8]}",
            "voice_tone": voice_tone,
            "text_length": len(full_clean_text),
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
    Generate high-fidelity voiceover using ElevenLabs TTS (primary) or Edge TTS (fallback).
    
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

    # Detect outro segment for melancholy TTS treatment
    _, outro_text = _detect_outro_segment(text)
    has_outro = bool(outro_text and len(outro_text.strip()) > 5)

    # Determine target engine
    target_engine = "elevenlabs"
    
    # Split raw text at story separator boundaries (....) before text cleaning.
    # This identifies structural pauses that need longer silence (0.8s)
    # vs intra-story batch boundaries that only need a micro-breath (0.06s).
    raw_chunks = re.split(r'\.{4,}', text)
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
    
    # Apply text cleaning to each structural chunk independently
    structural_chunks = [_add_natural_pacing(c, engine=target_engine) for c in raw_chunks]
    
    # Full clean text for word timestamp alignment
    clean_text = ' '.join(c for c in structural_chunks if c)
    
    filename = f"voiceover_{voice_tone}_{hash(text) % 10000}.mp3"
    filepath = OUTPUT_DIR / filename
    
    # === PRIMARY: Kokoro (local, free, GPU-accelerated) ===
    result = _generate_kokoro_tts(structural_chunks, voice_tone, filepath, has_outro=has_outro)
    
    if result and result.get('success'):
        mastered = _apply_audio_mastering(Path(result['path']))
        if mastered:
            print(f"  [TTS] Audio mastering applied (EQ + compression)")
            result['audio_mastered'] = True
            result['file_size_bytes'] = Path(result['path']).stat().st_size
        
        # Apply subtle reverb tail for melancholy outro feel
        if has_outro:
            reverb_path = _apply_outro_reverb(result['path'])
            if reverb_path != result['path']:
                result['path'] = reverb_path
                result['audio_mastered'] = True
                print(f"  [TTS] Outro reverb applied (melancholy tail)")
        
        result['word_timestamps'] = _get_faster_whisper_timestamps(result['path'], clean_text)
        return result
    
    # === PREMIUM: ElevenLabs (cloud, highest quality) ===
    print(f"  [TTS] Kokoro unavailable, trying ElevenLabs...")
    result = _generate_elevenlabs_tts(structural_chunks, voice_tone, filepath, has_outro=has_outro)
    
    if result and result.get('success'):
        mastered = _apply_audio_mastering(Path(result['path']))
        if mastered:
            print(f"  [TTS] Audio mastering applied (EQ + compression)")
            result['audio_mastered'] = True
            result['file_size_bytes'] = Path(result['path']).stat().st_size
        
        # Apply subtle reverb tail for melancholy outro feel
        if has_outro:
            reverb_path = _apply_outro_reverb(result['path'])
            if reverb_path != result['path']:
                result['path'] = reverb_path
                result['audio_mastered'] = True
                print(f"  [TTS] Outro reverb applied (melancholy tail)")
        
        result['word_timestamps'] = _get_faster_whisper_timestamps(result['path'], clean_text)
        return result
    
    # === FALLBACK: Edge TTS (cloud, free) ===
    print(f"  [TTS] ElevenLabs unavailable, trying Edge TTS fallback...")
    result = _generate_edge_tts(clean_text, voice_tone, filepath)
    
    if result and result.get('success'):
        # Apply audio mastering
        mastered = _apply_audio_mastering(Path(result['path']))
        if mastered:
            print(f"  [TTS] Audio mastering applied")
            result['audio_mastered'] = True
            result['file_size_bytes'] = Path(result['path']).stat().st_size
        
        # Apply subtle reverb tail for melancholy outro feel
        if has_outro:
            reverb_path = _apply_outro_reverb(result['path'])
            if reverb_path != result['path']:
                result['path'] = reverb_path
                print(f"  [TTS] Outro reverb applied (melancholy tail)")
        
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


def _check_cudnn_available() -> bool:
    """Return True if CUDA cuDNN shared library is loadable.
    
    CTranslate2 (faster-whisper's engine) needs libcudnn_ops_infer.so.8 to run
    with device='cuda'.  If it's missing the process segfaults (uncatchable).
    We detect it early here so we can safely fall back to CPU.
    """
    try:
        import ctypes
        import ctypes.util
        return ctypes.util.find_library("cudnn_ops_infer") is not None
    except Exception:
        return False


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
        _cudnn_ok = _check_cudnn_available()
        _cuda_available = torch.cuda.is_available()
        
        if _cuda_available and not _cudnn_ok:
            print(f"  [TTS] CUDA available but cuDNN missing — forcing CPU (install: pip install nvidia-cudnn-cu12)")
            device = "cpu"
            compute_type = "int8"
        elif _cuda_available:
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"
        
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
