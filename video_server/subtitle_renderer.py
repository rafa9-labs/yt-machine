"""
Subtitle Renderer — Word-by-word synced subtitles for split-screen video.
Renders a semi-transparent subtitle band at the center split line.
Uses edge-tts word boundary timestamps for precise sync.
"""

import re
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    from moviepy.editor import ImageClip, CompositeVideoClip
except ImportError:
    from moviepy import ImageClip, CompositeVideoClip

# Default subtitle styling — optimized for 1080px wide video on phone
SUBTITLE_STYLE = {
    'font_size': 52,
    'font_name': 'Arial-Bold',
    'text_color': (255, 255, 255),
    'outline_color': (0, 0, 0),
    'outline_width': 4,
    'bg_color': (0, 0, 0, 180),       # Semi-transparent black
    'band_height': 120,
    'max_chars_per_line': 32,
    'padding_x': 30,
    'padding_y': 15,
}


def get_word_timestamps(text: str, voice: str = "en-US-GuyNeural",
                        rate: str = "+0%", pitch: str = "+0Hz") -> List[Dict]:
    """
    Get word-level timestamps from edge-tts.
    Returns list of dicts: [{'word': str, 'start': float, 'end': float}, ...]
    """
    import edge_tts

    async def _fetch():
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        word_boundaries = []
        async for chunk in communicate.stream():
            if chunk["type"] == "WordBoundary":
                word_boundaries.append({
                    'word': chunk.get('text', '').strip(),
                    'start': chunk.get('offset', 0) / 10_000_000,
                    'end': (chunk.get('offset', 0) + chunk.get('duration', 0)) / 10_000_000,
                })
        return word_boundaries

    try:
        return asyncio.run(_fetch())
    except Exception as e:
        print(f"  [SUB] Word timestamp extraction failed: {e}")
        return _estimate_word_timestamps(text)


def _estimate_word_timestamps(text: str, words_per_sec: float = 2.5) -> List[Dict]:
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


def _group_words_into_phrases(word_timestamps: List[Dict],
                               max_chars: int = 32,
                               max_gap: float = 0.5) -> List[Dict]:
    """
    Group words into subtitle phrases (2 lines max).
    Groups words that are close together into single subtitle cards.
    """
    if not word_timestamps:
        return []

    phrases = []
    current_words = [word_timestamps[0]]
    current_len = len(word_timestamps[0]['word'])

    for i in range(1, len(word_timestamps)):
        w = word_timestamps[i]
        gap = w['start'] - current_words[-1]['end']
        new_len = current_len + 1 + len(w['word'])  # +1 for space

        if new_len <= max_chars and gap <= max_gap:
            current_words.append(w)
            current_len = new_len
        else:
            # Finalize current phrase
            phrases.append({
                'text': ' '.join(w['word'] for w in current_words),
                'start': current_words[0]['start'],
                'end': current_words[-1]['end'],
            })
            current_words = [w]
            current_len = len(w['word'])

    # Don't forget the last phrase
    if current_words:
        phrases.append({
            'text': ' '.join(w['word'] for w in current_words),
            'start': current_words[0]['start'],
            'end': current_words[-1]['end'],
        })

    return phrases


def _render_subtitle_frame(text: str, width: int, band_height: int,
                            style: dict = None) -> np.ndarray:
    """
    Render a single subtitle frame as RGB numpy array.
    Uses proper alpha blending for semi-transparent background.
    """
    s = {**SUBTITLE_STYLE, **(style or {})}

    # Create RGBA image for the subtitle band (transparent background)
    img = Image.new('RGBA', (width, band_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw semi-transparent background band
    draw.rectangle([0, 0, width, band_height], fill=s['bg_color'])

    # Load font — try multiple options for cross-platform compatibility
    font = None
    font_candidates = [
        (s['font_name'], s['font_size']),
        ('arial.ttf', s['font_size']),
        ('Arial.ttf', s['font_size']),
        ('C:/Windows/Fonts/arialbd.ttf', s['font_size']),
        ('C:/Windows/Fonts/arial.ttf', s['font_size']),
        ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', s['font_size']),
    ]
    for font_name, font_size in font_candidates:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except (IOError, OSError):
            continue
    
    if font is None:
        print(f"  [SUB] WARNING: No TrueType font found, using default (may look bad)")
        font = ImageFont.load_default()

    # Word-wrap text if needed
    lines = _wrap_text(text, font, width - 2 * s['padding_x'], draw)

    # Calculate text position (centered in band)
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])

    line_spacing = 6
    total_text_h = sum(line_heights) + line_spacing * (len(lines) - 1)
    y_offset = (band_height - total_text_h) // 2

    # Draw text with outline (stroke) for readability
    for line_idx, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (width - text_w) // 2

        # Draw outline (stroke) — circle around each character position
        ow = s['outline_width']
        if ow > 0:
            for dx in range(-ow, ow + 1):
                for dy in range(-ow, ow + 1):
                    if dx * dx + dy * dy <= ow * ow:
                        draw.text((x + dx, y_offset + dy), line,
                                  font=font, fill=s['outline_color'] + (255,))

        # Draw main text on top
        draw.text((x, y_offset), line, font=font,
                  fill=s['text_color'] + (255,))

        y_offset += line_heights[line_idx] + line_spacing

    # Convert RGBA → RGB by alpha compositing onto a black background
    background = Image.new('RGBA', (width, band_height), (0, 0, 0, 255))
    composite = Image.alpha_composite(background, img)
    
    return np.array(composite.convert('RGB'))


def _wrap_text(text: str, font, max_width: int, draw) -> List[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines if lines else [text]


def create_subtitle_clips(script_text: str, word_timestamps: List[Dict],
                           video_width: int, video_height: int,
                           band_y_position: int,
                           style: dict = None) -> list:
    """
    Create moviepy subtitle clips positioned at the split line.

    Args:
        script_text: Full script text (for reference)
        word_timestamps: Word-level timing data from get_word_timestamps()
        video_width: Final video width (1080)
        video_height: Final video height (1920)
        band_y_position: Y position for the subtitle band (split line)
        style: Optional style overrides

    Returns:
        List of ImageClip objects with timing, composited on video
    """
    s = {**SUBTITLE_STYLE, **(style or {})}
    band_h = s['band_height']

    # Group words into phrases
    phrases = _group_words_into_phrases(word_timestamps,
                                          max_chars=s['max_chars_per_line'])

    if not phrases:
        return []

    clips = []
    for phrase in phrases:
        # Render the subtitle frame
        frame = _render_subtitle_frame(phrase['text'], video_width, band_h, s)

        # Create clip from numpy array
        clip = ImageClip(frame).set_duration(phrase['end'] - phrase['start'])
        clip = clip.set_start(phrase['start'])
        clip = clip.set_position((0, band_y_position))

        clips.append(clip)

    print(f"  [SUB] Created {len(clips)} subtitle clips "
          f"({len(word_timestamps)} words → {len(phrases)} phrases)")

    return clips