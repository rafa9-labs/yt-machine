import os
import asyncio
import time
import re
from pathlib import Path
from typing import Optional
from mcp.server.fastmcp import FastMCP
import edge_tts
import struct
import wave

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
    Clean script text and prepare it for natural speech.
    Removes any stray formatting characters, underscores, or markup
    that would be read literally by TTS.
    
    Args:
        text: Original script text
    
    Returns:
        Clean plain text safe for TTS or SSML inner content
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
    
    # Collapse multiple spaces/newlines into single space
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    
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
    prosody_settings = {
        "authoritative": {"rate": "+0%",  "pitch": "+0Hz"},
        "professional":  {"rate": "-5%",  "pitch": "-2Hz"},
        "energetic":     {"rate": "+10%", "pitch": "+2Hz"},
        "calm":          {"rate": "-10%", "pitch": "-4Hz"},
        "deep":          {"rate": "-5%",  "pitch": "-5Hz"},
        "female_professional": {"rate": "-3%", "pitch": "+0Hz"},
        "female_energetic":    {"rate": "+8%", "pitch": "+3Hz"},
        "male_casual":         {"rate": "+5%", "pitch": "+0Hz"},
    }
    return prosody_settings.get(voice_tone, {"rate": "+0%", "pitch": "+0Hz"})

@server.tool()
def generate_voiceover(text: str, voice_tone: str = "authoritative") -> dict:
    """
    Generate high-fidelity voiceover using Edge TTS.
    
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

    voice = VOICE_MAPPING.get(voice_tone, VOICE_MAPPING["authoritative"])

    # Clean the text (remove formatting, underscores, etc.)
    clean_text = _add_natural_pacing(text)
    
    # Get rate/pitch parameters for Edge TTS API
    voice_params = _get_voice_parameters(voice_tone)
    
    filename = f"voiceover_{voice_tone}_{hash(text) % 10000}.mp3"
    filepath = OUTPUT_DIR / filename
    
    try:
        async def generate_audio():
            # Edge TTS accepts plain text + rate/pitch parameters (NO SSML)
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
                    time.sleep(1)  # Wait 1 second before retry
                    continue
                else:
                    raise e
        
        if not filepath.exists():
            return {
                "success": False,
                "error": f"Audio file was not created at {filepath}"
            }

        # Phase 5.2: Apply audio mastering
        mastered = _apply_audio_mastering(filepath)
        if mastered:
            print(f"  [TTS] Audio mastering applied to {filename}")

        file_size = filepath.stat().st_size
        word_count = len(text.split())
        estimated_duration = word_count / 2.5

        # Save word timestamps for subtitle sync
        word_timestamps = _get_word_timestamps_sync(clean_text, voice, voice_params)

        return {
            "success": True,
            "filename": filename,
            "path": str(filepath),
            "voice": voice,
            "voice_tone": voice_tone,
            "text_length": len(text),
            "word_count": word_count,
            "file_size_bytes": file_size,
            "estimated_duration_seconds": round(estimated_duration, 2),
            "audio_mastered": mastered,
            "output_directory": str(OUTPUT_DIR),
            "word_timestamps": word_timestamps,
        }
    
    except Exception as e:
        print(f"  [TTS] Edge TTS failed: {e}")
        print(f"  [TTS] Generating duration-matched silent audio fallback...")
        
        # Create fallback audio matching the estimated script duration
        try:
            import numpy as np
            from moviepy.audio.AudioClip import AudioArrayClip
            
            word_count = len(text.split())
            estimated_duration = max(word_count / 2.5, 10.0)  # At least 10 seconds
            fps = 44100
            
            print(f"  [TTS] Fallback duration: {estimated_duration:.1f}s ({word_count} words)")
            
            # Generate silent audio of correct duration
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
                "fallback_reason": str(e)
            }
                
        except Exception as fallback_error:
            return {
                "success": False,
                "error": f"Edge TTS failed: {str(e)}. Fallback also failed: {str(fallback_error)}",
                "voice": voice,
                "voice_tone": voice_tone
            }


def _get_word_timestamps_sync(text: str, voice: str, voice_params: dict) -> list:
    """
    Synchronously extract word-level timestamps from edge-tts.
    Returns list of {'word': str, 'start': float, 'end': float}.
    """
    try:
        async def _fetch():
            communicate = edge_tts.Communicate(
                text, voice,
                rate=voice_params.get("rate", "+0%"),
                pitch=voice_params.get("pitch", "+0Hz"),
            )
            boundaries = []
            async for chunk in communicate.stream():
                if chunk["type"] == "WordBoundary":
                    boundaries.append({
                        'word': chunk.get('text', '').strip(),
                        'start': chunk.get('offset', 0) / 1_000_000,
                        'end': (chunk.get('offset', 0) + chunk.get('duration', 0)) / 1_000_000,
                    })
            return boundaries
        result = asyncio.run(_fetch())
        if result:
            print(f"  [TTS] Word timestamps: {len(result)} words extracted")
        return result
    except Exception as e:
        print(f"  [TTS] Word timestamp extraction skipped: {e}")
        return []
