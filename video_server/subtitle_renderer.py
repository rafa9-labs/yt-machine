"""
Subtitle Renderer - 5-word animated phrase subtitles with karaoke highlight.
Shows 5 words at a time with yellow highlight moving through them.
No background band — outline-only text rendered directly over the video.
Uses script words (not whisper transcription) for correct display.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    from moviepy.editor import ImageClip, VideoClip
except ImportError:
    from moviepy import ImageClip, VideoClip

SUBTITLE_STYLE = {
    'font_size': 64,
    'font_name': 'Arial-Bold',
    'highlight_color': (255, 215, 0),   # Yellow for current word
    'previous_color': (255, 255, 255),  # White for previous words
    'upcoming_color': (180, 180, 180),  # Gray for upcoming words
    'outline_color': (0, 0, 0),
    'outline_width': 5,
    'band_height': 120,      # Vertical space allocated for subtitle rendering
    'padding_x': 50,
    'lead_in_seconds': 0.3,
    'time_offset': 0.0,      # No offset — raw whisper timestamps
}

TITLE_STYLE = {
    'font_size': 64,                # Same as subtitles
    'font_name': 'Arial-Bold',      # Same font
    'color': (255, 215, 0),         # Static gold/yellow (like subtitle highlight)
    'outline_color': (0, 0, 0),     # Same black outline
    'outline_width': 5,             # Same outline width as subtitles
    'max_words': 8,
    'fade_in_seconds': 0.8,
    'display_seconds': 5.0,
    'y_position': 80,               # From top
}


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


def align_whisper_to_script(
    whisper_words: List[Dict],
    script_text: str
) -> List[Dict]:
    """
    Align whisper word timestamps to the original script words.
    
    Strategy: Use whisper for TIMING (when words were spoken) and script
    for WORDS (what to display). Uses anchor-based interpolation to prevent
    drift when whisper mishears or drops words.
    
    Args:
        whisper_words: [{'word': str, 'start': float, 'end': float}] from whisper
        script_text: Original script text that was spoken
    
    Returns:
        List of {'word': str, 'start': float, 'end': float} with correct words
    """
    script_words = script_text.split()
    n_script = len(script_words)
    n_whisper = len(whisper_words)
    
    if not script_words:
        return whisper_words
    if not whisper_words:
        return _estimate_from_script(script_text, 30.0)
    
    total_duration = whisper_words[-1]['end']
    
    # Step 1: Find anchor points where whisper and script words match
    whisper_cleaned = [w['word'].strip('.,;:!?—–').lower() for w in whisper_words]
    script_cleaned = [w.strip('.,;:!?—–').lower() for w in script_words]
    
    anchors = []  # List of (script_idx, whisper_idx) for matched words
    
    wi = 0
    for si, sword in enumerate(script_cleaned):
        # Search ahead in whisper words for a match (window of ±3)
        search_start = max(0, wi - 1)
        search_end = min(n_whisper, wi + 4)
        
        best_match_wi = -1
        for twi in range(search_start, search_end):
            wword = whisper_cleaned[twi]
            if sword == wword or _fuzzy_match(sword, wword):
                best_match_wi = twi
                break
        
        if best_match_wi >= 0:
            anchors.append((si, best_match_wi))
            wi = best_match_wi + 1
    
    # Step 2: Build aligned words using interpolation between anchors
    # Start with proportional baseline (ensures ALL words have timing)
    base_duration = total_duration / n_script if n_script > 0 else 0.4
    aligned = [
        {
            'word': script_words[i],
            'start': i * base_duration,
            'end': (i + 1) * base_duration,
        }
        for i in range(n_script)
    ]
    
    # Step 3: Override with precise timing from anchor matches
    if len(anchors) >= 2:
        # We have enough anchors to interpolate between them
        for ai in range(len(anchors)):
            si, twi = anchors[ai]
            aligned[si] = {
                'word': script_words[si],
                'start': whisper_words[twi]['start'],
                'end': whisper_words[twi]['end'],
            }
        
        # Interpolate between consecutive anchors
        for ai in range(len(anchors) - 1):
            si1, twi1 = anchors[ai]
            si2, twi2 = anchors[ai + 1]
            
            t_start = whisper_words[twi1]['end']
            t_end = whisper_words[twi2]['start']
            n_between = si2 - si1 - 1
            
            if n_between > 0 and t_end > t_start:
                seg_duration = (t_end - t_start) / n_between
                for j in range(n_between):
                    idx = si1 + 1 + j
                    aligned[idx] = {
                        'word': script_words[idx],
                        'start': t_start + j * seg_duration,
                        'end': t_start + (j + 1) * seg_duration,
                    }
        
        # Interpolate before first anchor
        si0, twi0 = anchors[0]
        if si0 > 0:
            t_end = whisper_words[twi0]['start']
            seg_duration = t_end / si0 if si0 > 0 else 0.3
            for j in range(si0):
                aligned[j] = {
                    'word': script_words[j],
                    'start': j * seg_duration,
                    'end': (j + 1) * seg_duration,
                }
        
        # Interpolate after last anchor
        si_last, twi_last = anchors[-1]
        if si_last < n_script - 1:
            t_start = whisper_words[twi_last]['end']
            remaining = n_script - si_last - 1
            remaining_time = total_duration - t_start
            seg_duration = remaining_time / remaining if remaining > 0 else 0.3
            for j in range(remaining):
                idx = si_last + 1 + j
                aligned[idx] = {
                    'word': script_words[idx],
                    'start': t_start + j * seg_duration,
                    'end': t_start + (j + 1) * seg_duration,
                }
    
    elif len(anchors) == 1:
        # Only one anchor — use it but distribute rest proportionally
        si0, twi0 = anchors[0]
        aligned[si0] = {
            'word': script_words[si0],
            'start': whisper_words[twi0]['start'],
            'end': whisper_words[twi0]['end'],
        }
    
    # Ensure no overlapping timestamps and minimum word duration
    for i in range(len(aligned)):
        if aligned[i]['end'] <= aligned[i]['start']:
            aligned[i]['end'] = aligned[i]['start'] + 0.2
        # Ensure chronological order
        if i > 0 and aligned[i]['start'] < aligned[i-1]['end']:
            aligned[i]['start'] = aligned[i-1]['end']
            aligned[i]['end'] = max(aligned[i]['end'], aligned[i]['start'] + 0.15)
    
    anchor_pct = len(anchors) / n_script * 100 if n_script > 0 else 0
    print(f"  [SUB] Alignment: {len(anchors)}/{n_script} anchors ({anchor_pct:.0f}% matched), {n_whisper} whisper words")
    
    return aligned


def _fuzzy_match(a: str, b: str) -> bool:
    """Check if two words are similar enough to be considered a match."""
    if len(a) < 3 or len(b) < 3:
        return False
    # Check if one contains the other (handles contractions, possessives)
    if a in b or b in a:
        return True
    # Check edit distance for short words
    if abs(len(a) - len(b)) > 2:
        return False
    matches = sum(1 for ca, cb in zip(a, b) if ca == cb)
    return matches >= min(len(a), len(b)) * 0.7


def _estimate_from_script(script_text: str, total_duration: float) -> List[Dict]:
    """Estimate timestamps evenly from script text."""
    words = script_text.split()
    n = len(words)
    if n == 0:
        return []
    word_dur = total_duration / n
    return [
        {'word': w, 'start': i * word_dur, 'end': (i + 1) * word_dur}
        for i, w in enumerate(words)
    ]


def _split_into_phrases(words: List[Dict], max_words: int = 5, min_words: int = 4) -> List[Tuple[int, int, float, float]]:
    """
    Split words into phrase chunks (start_idx, end_idx, phrase_start, phrase_end).
    Respects sentence boundaries and natural pauses.
    Enforces min_words per phrase — merges short phrases to avoid showing 1-2 words alone.
    """
    n_words = len(words)
    if n_words == 0:
        return []
    
    # If total words <= min_words, just return one phrase
    if n_words <= min_words:
        return [(0, n_words, words[0]['start'], words[-1]['end'])]
    
    # Phase 1: Initial split respecting sentence boundaries
    raw_phrases = []
    i = 0
    
    while i < n_words:
        end_idx = min(i + max_words, n_words)
        
        # Respect sentence boundaries
        for j in range(i, min(i + max_words, n_words)):
            word = words[j]['word'].strip()
            if any(word.endswith(punct) for punct in ['.', '?', '!']):
                end_idx = j + 1
                break
            if word.endswith(',') and j > i and j - i >= 3:
                end_idx = j + 1
                break
        
        raw_phrases.append((i, end_idx))
        i = end_idx
    
    # Phase 2: Merge short phrases (fewer than min_words) into neighbors
    merged = []
    for start_idx, end_idx in raw_phrases:
        phrase_len = end_idx - start_idx
        
        if phrase_len < min_words and merged:
            # Merge into previous phrase
            prev_start, prev_end = merged[-1]
            combined_len = end_idx - prev_start
            if combined_len <= max_words + 2:  # Allow slight overflow for merging
                merged[-1] = (prev_start, end_idx)
                continue
        
        merged.append((start_idx, end_idx))
    
    # Phase 3: Check if last phrase is too short — merge with previous
    if len(merged) >= 2:
        last_start, last_end = merged[-1]
        if last_end - last_start < min_words:
            prev_start, prev_end = merged[-2]
            merged[-2] = (prev_start, last_end)
            merged.pop()
    
    # Build final phrases with timing
    phrases = []
    for start_idx, end_idx in merged:
        phrase_words = words[start_idx:end_idx]
        if phrase_words:
            phrase_start = phrase_words[0]['start']
            phrase_end = phrase_words[-1]['end']
            phrases.append((start_idx, end_idx, phrase_start, phrase_end))
    
    return phrases


def _find_active_word_idx(words: List[Dict], time: float) -> int:
    """Find word being spoken at given time. Keeps most recent word highlighted during gaps."""
    best_idx = -1
    best_end = -float('inf')
    
    for idx, word in enumerate(words):
        if word['start'] <= time and word['end'] >= time:
            return idx
        elif word['end'] <= time and word['end'] > best_end:
            best_idx = idx
            best_end = word['end']
    
    return best_idx


def _draw_outlined(draw, x, y, text, font, fill, outline, ow):
    """Draw text with circular outline. Accepts 3-element (RGB) or 4-element (RGBA) colors."""
    # Ensure colors are 4-element RGBA
    if len(outline) == 3:
        outline_4 = outline + (255,)
    else:
        outline_4 = outline
    if len(fill) == 3:
        fill_4 = fill + (255,)
    else:
        fill_4 = fill
    
    if ow > 0:
        for dx in range(-ow, ow + 1):
            for dy in range(-ow, ow + 1):
                if dx * dx + dy * dy <= ow * ow:
                    draw.text((x + dx, y + dy), text, font=font,
                              fill=outline_4)
    draw.text((x, y), text, font=font, fill=fill_4)


def _clean_display(word: str) -> str:
    """Clean word for display: strip trailing punctuation, uppercase."""
    return word.strip('.,;:!?—–').upper()


def _render_phrase_frame(words: List[Dict], phrase_start_idx: int, phrase_end_idx: int,
                      time: float, width: int, band_h: int, font, style) -> np.ndarray:
    """
    Render transparent subtitle frame for a phrase at given time.
    No background — just outlined text.
    """
    hi = style['highlight_color']
    pw = style['previous_color']
    uc = style['upcoming_color']
    ol = style['outline_color']
    ow = style['outline_width']
    gap = 20

    # Transparent image (RGBA)
    img = Image.new('RGBA', (width, band_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    phrase_words = words[phrase_start_idx:phrase_end_idx]
    if not phrase_words:
        return np.array(img)  # Keep RGBA for transparency

    # Find active word (with time offset to compensate for whisper delay)
    adjusted_time = time + style.get('time_offset', 0)
    global_active_idx = _find_active_word_idx(words, adjusted_time)
    relative_active_idx = global_active_idx - phrase_start_idx if global_active_idx >= phrase_start_idx else -1

    # Prepare words with colors
    word_list = []
    for idx, wt in enumerate(phrase_words):
        word = _clean_display(wt['word'])
        if idx < relative_active_idx:
            color = pw
        elif idx == relative_active_idx:
            color = hi
        else:
            color = uc
        word_list.append((word, color))

    # Measure widths
    widths = []
    for word, _ in word_list:
        bb = draw.textbbox((0, 0), word, font=font)
        widths.append(bb[2] - bb[0])

    total_w = sum(widths) + gap * (len(word_list) - 1)
    bb_h = draw.textbbox((0, 0), word_list[0][0], font=font)
    text_h = bb_h[3] - bb_h[1]

    # Handle line wrapping
    max_width = width - 100
    if total_w > max_width:
        mid = len(word_list) // 2
        line1_words = word_list[:mid]
        line2_words = word_list[mid:]

        def _render_line(line_words, start_x, line_y):
            line_widths = []
            for word, _ in line_words:
                bb = draw.textbbox((0, 0), word, font=font)
                line_widths.append(bb[2] - bb[0])
            line_total = sum(line_widths) + gap * (len(line_words) - 1)
            x = start_x
            for idx, (word, color) in enumerate(line_words):
                _draw_outlined(draw, x, line_y, word, font, color, ol, ow)
                x += line_widths[idx] + gap

        total_w1 = sum([draw.textbbox((0, 0), w, font=font)[2] for w, _ in line1_words]) + gap * (len(line1_words) - 1)
        x1 = (width - total_w1) // 2
        y1 = (band_h - text_h * 2 - gap) // 2 - bb_h[1]
        _render_line(line1_words, x1, y1)

        total_w2 = sum([draw.textbbox((0, 0), w, font=font)[2] for w, _ in line2_words]) + gap * (len(line2_words) - 1)
        x2 = (width - total_w2) // 2
        y2 = y1 + text_h + gap
        _render_line(line2_words, x2, y2)
    else:
        y = (band_h - text_h) // 2 - bb_h[1]
        x = (width - total_w) // 2
        for idx, (word, color) in enumerate(word_list):
            _draw_outlined(draw, x, y, word, font, color, ol, ow)
            x += widths[idx] + gap

    return np.array(img)  # Keep RGBA — never convert to RGB (that creates black bg)


def _render_title_frame(title_text: str, width: int, font, style, alpha: float = 1.0) -> np.ndarray:
    """Render title text at top of frame with outline, no background."""
    color = style['color']
    ol = style['outline_color']
    ow = style['outline_width']
    
    img = Image.new('RGBA', (width, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Word wrap title
    words = title_text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bb = draw.textbbox((0, 0), test, font=font)
        if bb[2] - bb[0] > width - 100:
            if current:
                lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    
    y = style['y_position']
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        x = (width - tw) // 2
        
        # Apply alpha
        a = int(255 * min(1.0, max(0.0, alpha)))
        c = color + (a,)
        o = ol + (a,)
        _draw_outlined(draw, x, y - bb[1], line, font, c, o, ow)
        y += th + 10
    
    return np.array(img)  # Keep RGBA for transparency


def create_subtitle_clips(script_text: str, word_timestamps: List[Dict],
                       video_width: int, video_height: int,
                       band_y_position: int,
                       style: dict = None) -> list:
    """
    Create 5-word animated phrase subtitle clips.
    No background band — transparent overlay with outlined text.
    Uses script words aligned to whisper timestamps.
    """
    s = {**SUBTITLE_STYLE, **(style or {})}
    band_h = s['band_height']
    font = _load_font(s)

    if not word_timestamps:
        return []

    # Align whisper timestamps to original script words
    if script_text and script_text.strip():
        aligned_words = align_whisper_to_script(word_timestamps, script_text)
        print(f"  [SUB] Word alignment: {len(word_timestamps)} whisper → {len(aligned_words)} script words")
    else:
        aligned_words = word_timestamps

    phrases = _split_into_phrases(aligned_words, max_words=5)
    if not phrases:
        return []

    lead_in = s.get('lead_in_seconds', 0.3)

    clips = []
    for i, (start_idx, end_idx, phrase_start, phrase_end) in enumerate(phrases):
        clip_start = max(0, phrase_start - lead_in) if i == 0 else max(phrases[i-1][3], phrase_start - lead_in)
        clip_end = phrase_end
        duration = clip_end - clip_start
        
        if duration <= 0:
            continue

        def make_frame(t, si=start_idx, ei=end_idx, cs=clip_start):
            frame = _render_phrase_frame(
                aligned_words, si, ei,
                cs + t, video_width, band_h, font, s
            )
            return frame[:, :, :3]  # RGB only — alpha handled by mask

        clip = VideoClip(make_frame, duration=duration)
        clip = clip.set_start(clip_start)
        clip = clip.set_position((0, band_y_position))
        # Create mask from alpha channel (ismask=True required by moviepy)
        mask_clip = VideoClip(
            lambda t, si=start_idx, ei=end_idx, cs=clip_start: _make_alpha_mask(
                t, cs, si, ei, aligned_words, video_width, band_h, font, s),
            ismask=True, duration=duration
        )
        clip = clip.set_mask(mask_clip)
        clips.append(clip)

    total_covered = sum(c.duration for c in clips)
    print(f"  [SUB] Phrase subtitles: {len(clips)} clips ({len(aligned_words)} words, {len(phrases)} phrases, {total_covered:.1f}s coverage)")

    return clips


def _make_alpha_mask(t, cs, si, ei, aligned_words, video_width, band_h, font, style):
    """Create alpha mask for subtitle transparency."""
    frame = _render_phrase_frame(aligned_words, si, ei, cs + t, video_width, band_h, font, style)
    # Extract alpha channel from RGBA
    if frame.shape[2] == 4:
        alpha = frame[:, :, 3] / 255.0
    else:
        alpha = np.ones((frame.shape[0], frame.shape[1]))
    return alpha


def create_title_clip(title_text: str, video_width: int, video_height: int,
                     duration: float = None, style: dict = None) -> Optional[VideoClip]:
    """
    Create a title overlay clip that fades in and stays at the top.
    Shows first few words of the hook as a title.
    """
    s = {**TITLE_STYLE, **(style or {})}
    font = _load_font(s)
    
    if not title_text or not title_text.strip():
        return None
    
    # Truncate to max words
    words = title_text.strip().split()
    if len(words) > s['max_words']:
        title_text = ' '.join(words[:s['max_words']])
    
    display_dur = duration or s['display_seconds']
    fade_in = s['fade_in_seconds']
    
    def make_frame(t):
        # Fade in during first fade_in seconds
        if t < fade_in:
            alpha = t / fade_in
        else:
            alpha = 1.0
        frame = _render_title_frame(title_text, video_width, font, s, alpha)
        return frame[:, :, :3]  # RGB only — alpha handled by mask
    
    def make_mask(t):
        if t < fade_in:
            alpha = t / fade_in
        else:
            alpha = 1.0
        # Create a full alpha mask
        frame = _render_title_frame(title_text, video_width, font, s, alpha)
        if frame.shape[2] == 4:
            return frame[:, :, 3] / 255.0
        return np.ones((frame.shape[0], frame.shape[1]))
    
    clip = VideoClip(make_frame, duration=display_dur)
    clip = clip.set_position((0, 0))
    
    # Set mask with ismask=True for proper transparency
    mask_clip = VideoClip(make_mask, ismask=True, duration=display_dur)
    clip = clip.set_mask(mask_clip)
    
    return clip
