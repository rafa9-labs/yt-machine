"""
Subtitle Renderer - Karaoke-style word-by-word subtitles.
Shows 2 content words at a time: current word highlighted yellow, previous white.
Filler words (articles, prepositions, conjunctions) are skipped.
"""

import re
from pathlib import Path
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    from moviepy.editor import ImageClip
except ImportError:
    from moviepy import ImageClip

# Filler words to skip
SKIP_WORDS = {
    'a', 'an', 'the',
    'from', 'to', 'in', 'on', 'at', 'of', 'for', 'with', 'by', 'up', 'out',
    'into', 'onto', 'upon', 'over', 'under', 'about', 'after', 'before',
    'between', 'through', 'during', 'without', 'within', 'along', 'across',
    'and', 'but', 'or', 'so', 'yet', 'nor', 'as', 'than',
    'it', 'he', 'she', 'they', 'we', 'i', 'you', 'me', 'him', 'her', 'us',
    'them', 'my', 'your', 'his', 'its', 'our', 'their', 'this', 'that',
    'these', 'those', 'who', 'whom', 'which', 'what',
    'is', 'are', 'was', 'were', 'be', 'been', 'am', 'do', 'does', 'did',
    'has', 'have', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
    'shall', 'can', 'must',
    'not', 'no', 'if', 'then', 'when', 'how', 'why', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own',
    'same', 'also', 'just', 'very', 'even', 'still', 'already', 'now',
    "don't", "doesn't", "didn't", "wasn't", "weren't", "won't", "wouldn't",
    "couldn't", "shouldn't", "isn't", "aren't",
}

SUBTITLE_STYLE = {
    'font_size': 80,
    'font_name': 'Arial-Bold',
    'highlight_color': (255, 215, 0),   # Yellow for current word
    'previous_color': (255, 255, 255),  # White for previous word
    'outline_color': (0, 0, 0),
    'outline_width': 5,
    'bg_color': (0, 0, 0, 180),
    'band_height': 110,
    'padding_x': 40,
    'lead_in_seconds': 0.3,
}


def _is_content_word(word: str) -> bool:
    """Check if word is a content word (not a filler)."""
    clean = word.strip('.,;:!?—–').lower()
    if '-' in clean:
        return True
    if clean.endswith("'s"):
        base = clean[:-2]
        return base not in SKIP_WORDS and len(base) > 1
    return clean not in SKIP_WORDS


def _clean_display(word: str) -> str:
    """Clean word for display: strip punctuation, uppercase."""
    return word.strip('.,;:!?—–').upper()


def _load_font(style: dict):
    """Load a bold TrueType font."""
    sz = style['font_size']
    for name in [style.get('font_name', 'Arial-Bold'), 'C:/Windows/Fonts/arialbd.ttf',
                 'arialbd.ttf', 'C:/Windows/Fonts/arial.ttf', 'arial.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']:
        try:
            return ImageFont.truetype(name, sz)
        except (IOError, OSError):
            continue
    print("  [SUB] WARNING: No TrueType font found")
    return ImageFont.load_default()


def _build_karaoke_pairs(word_timestamps: List[Dict]) -> List[Dict]:
    """
    Build karaoke pairs from word timestamps.
    Each pair: previous_word (white), current_word (yellow), start, end.
    Filler words don't trigger a new pair.
    """
    if not word_timestamps:
        return []

    pairs = []
    prev_content = None
    last_end = 0.0

    for wt in word_timestamps:
        if not _is_content_word(wt['word']):
            continue

        curr = _clean_display(wt['word'])
        prev = _clean_display(prev_content) if prev_content else None

        pair = {
            'previous_word': prev,
            'current_word': curr,
            'start': wt['start'],
            'end': None,
        }

        if pairs:
            pairs[-1]['end'] = wt['start']

        pairs.append(pair)
        prev_content = wt['word']
        last_end = wt['end']

    if pairs:
        pairs[-1]['end'] = last_end + 0.5

    return pairs


def _draw_outlined(draw, x, y, text, font, fill, outline, ow):
    """Draw text with circular outline."""
    if ow > 0:
        for dx in range(-ow, ow + 1):
            for dy in range(-ow, ow + 1):
                if dx * dx + dy * dy <= ow * ow:
                    draw.text((x + dx, y + dy), text, font=font,
                              fill=outline + (255,))
    draw.text((x, y), text, font=font, fill=fill + (255,))


def _render_karaoke_frame(prev_word, curr_word, width, band_h, font, style) -> np.ndarray:
    """Render frame: 2 words centered, current=yellow, previous=white."""
    bg = style['bg_color']
    hi = style['highlight_color']
    pw = style['previous_color']
    ol = style['outline_color']
    ow = style['outline_width']
    gap = 30

    img = Image.new('RGBA', (width, band_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, band_h], fill=bg)

    # Collect words to draw
    parts = []
    if prev_word:
        parts.append(('prev', prev_word))
    if curr_word:
        parts.append(('curr', curr_word))

    if not parts:
        bg_img = Image.new('RGBA', (width, band_h), (0, 0, 0, 255))
        return np.array(Image.alpha_composite(bg_img, img).convert('RGB'))

    # Measure widths
    widths = []
    for _, w in parts:
        bb = draw.textbbox((0, 0), w, font=font)
        widths.append(bb[2] - bb[0])

    total_w = sum(widths) + gap * (len(parts) - 1)
    bb_h = draw.textbbox((0, 0), parts[0][1], font=font)
    text_h = bb_h[3] - bb_h[1]
    y = (band_h - text_h) // 2 - bb_h[1]
    x = (width - total_w) // 2

    for idx, (kind, word) in enumerate(parts):
        color = hi if kind == 'curr' else pw
        _draw_outlined(draw, x, y, word, font, color, ol, ow)
        x += widths[idx] + gap

    bg_img = Image.new('RGBA', (width, band_h), (0, 0, 0, 255))
    return np.array(Image.alpha_composite(bg_img, img).convert('RGB'))


def create_subtitle_clips(script_text: str, word_timestamps: List[Dict],
                           video_width: int, video_height: int,
                           band_y_position: int,
                           style: dict = None) -> list:
    """
    Create karaoke subtitle clips: 2 content words at a time.
    Current word highlighted yellow, previous word white.
    """
    s = {**SUBTITLE_STYLE, **(style or {})}
    band_h = s['band_height']
    font = _load_font(s)

    pairs = _build_karaoke_pairs(word_timestamps)
    if not pairs:
        return []

    lead_in = s.get('lead_in_seconds', 0.3)

    # Pre-calculate adjusted starts (lead-in) without overlaps
    adj_starts = [max(0, p['start'] - lead_in) for p in pairs]

    clips = []
    for i, pair in enumerate(pairs):
        clip_start = adj_starts[i]
        clip_end = adj_starts[i + 1] if i + 1 < len(adj_starts) else pair['end']
        duration = clip_end - clip_start
        if duration <= 0:
            continue

        frame = _render_karaoke_frame(
            pair['previous_word'], pair['current_word'],
            video_width, band_h, font, s
        )

        clip = ImageClip(frame).set_duration(duration)
        clip = clip.set_start(clip_start)
        clip = clip.set_position((0, band_y_position))
        clips.append(clip)

    total_covered = sum(c.duration for c in clips)
    content_count = sum(1 for w in word_timestamps if _is_content_word(w['word']))
    print(f"  [SUB] Karaoke: {len(clips)} clips "
          f"({len(word_timestamps)} words, {content_count} content words, "
          f"{total_covered:.1f}s coverage)")

    return clips