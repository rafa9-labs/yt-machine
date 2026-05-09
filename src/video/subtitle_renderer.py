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
    'highlight_color': (255, 215, 0),
    'previous_color': (255, 255, 255),
    'upcoming_color': (180, 180, 180),
    'outline_color': (0, 0, 0),
    'outline_width': 5,
    'band_height': 120,
    'padding_x': 50,
    'lead_in_seconds': 0.05,
}

TITLE_STYLE = {
    'font_size': 48,                # Smaller than subtitles (64) — non-intrusive
    'font_name': 'Arial-Bold',      # Same font family
    'color': (255, 215, 0),         # All yellow (matches subtitle highlight)
    'outline_color': (0, 0, 0),     # Same black outline
    'outline_width': 3,             # Thinner outline (subtitles use 5)
    'max_words': 10,
    'fade_in_seconds': 0.0,         # No fade — static the whole video
    'display_seconds': 999.0,       # Persist entire video (overridden by caller)
    'y_position': 20,               # Very top — minimal intrusion
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
    
    # ── GAP-AWARE REDISTRIBUTION ──
    # Detect large pauses (>1.5s gap between consecutive words) and redistribute
    # word timing within each segment independently. This prevents closing words
    # from being stretched over silence gaps (causes sped-up subtitle mismatch).
    PAUSE_THRESHOLD = 0.3  # seconds — catches structural pauses (story breaks) and audio gaps
    gap_count = 0
    for i in range(1, len(aligned)):
        gap = aligned[i]['start'] - aligned[i-1]['end']
        if gap > PAUSE_THRESHOLD:
            gap_count += 1
    
    if gap_count > 0:
        # Split into segments at pause boundaries
        segments = []
        seg_start = 0
        for i in range(1, len(aligned)):
            gap = aligned[i]['start'] - aligned[i-1]['end']
            if gap > PAUSE_THRESHOLD:
                segments.append((seg_start, i))
                seg_start = i
        segments.append((seg_start, len(aligned)))
        
        # Redistribute timing within each segment
        for seg_s, seg_e in segments:
            seg_words = aligned[seg_s:seg_e]
            n_seg = len(seg_words)
            if n_seg < 2:
                continue
            
            # Total time span of this segment (including intra-word gaps)
            seg_time_start = seg_words[0]['start']
            seg_time_end = seg_words[-1]['end']
            seg_total = seg_time_end - seg_time_start
            
            if seg_total <= 0:
                continue
            
            # Redistribute evenly within segment
            word_dur = seg_total / n_seg
            for j in range(n_seg):
                aligned[seg_s + j]['start'] = seg_time_start + j * word_dur
                aligned[seg_s + j]['end'] = seg_time_start + (j + 1) * word_dur
        
        print(f"  [SUB] Gap-aware redistribution: {gap_count} pauses detected, {len(segments)} segments")
    
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
    if a in b or b in a:
        return True
    if abs(len(a) - len(b)) > 2:
        return False
    matches = sum(1 for ca, cb in zip(a, b) if ca == cb)
    if matches >= min(len(a), len(b)) * 0.6:
        return True
    return _phonetic_match(a, b)


def _phonetic_match(a: str, b: str) -> bool:
    """Simple phonetic matching for proper nouns and accented words."""
    vowel_map = str.maketrans('aeiou', 'aaaaa')
    a_phonetic = a.translate(vowel_map)
    b_phonetic = b.translate(vowel_map)
    if a_phonetic == b_phonetic:
        return True
    a_cons = ''.join(c for c in a if c not in 'aeiou')
    b_cons = ''.join(c for c in b if c not in 'aeiou')
    if a_cons and b_cons and (a_cons == b_cons or a_cons in b_cons or b_cons in a_cons):
        return True
    return False


def _compute_drift(anchors, whisper_words, n_script, total_duration) -> float:
    """Compute per-video whisper drift by comparing anchor positions to expected positions."""
    if len(anchors) < 3:
        return 0.0
    drifts = []
    for si, twi in anchors:
        expected_time = (si / n_script) * total_duration if n_script > 0 else 0
        actual_time = whisper_words[twi]['start']
        drifts.append(actual_time - expected_time)
    drifts.sort()
    median_drift = drifts[len(drifts) // 2]
    return max(-0.4, min(0.4, median_drift))


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


def _measure_phrase_width(words: List[Dict], start_idx: int, end_idx: int, font, gap: int = 20) -> int:
    """Measure the total pixel width of a phrase rendered with the given font."""
    tmp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp_img)
    total = 0
    for i in range(start_idx, end_idx):
        word = _clean_display(words[i]['word'])
        if word:
            bb = tmp_draw.textbbox((0, 0), word, font=font)
            total += (bb[2] - bb[0]) + gap
    return max(0, total - gap)  # Remove trailing gap


def _split_into_phrases(words: List[Dict], max_words: int = 7, min_words: int = 2,
                        max_pixel_width: int = 900) -> List[Tuple[int, int, float, float]]:
    """
    Split words into phrase chunks using PIXEL WIDTH instead of word count.
    Ensures no phrase overflows the screen horizontally.
    Also respects sentence boundaries and natural pauses.
    """
    n_words = len(words)
    if n_words == 0:
        return []
    
    if n_words <= min_words:
        return [(0, n_words, words[0]['start'], words[-1]['end'])]
    
    # Load font for width measurement
    font = _load_font(SUBTITLE_STYLE)
    
    # Phase 1: Initial split by pixel width + sentence boundaries
    raw_phrases = []
    i = 0
    
    while i < n_words:
        best_end = i + 1  # At minimum, take one word
        
        # Greedily add words until we exceed max_pixel_width or max_words
        for j in range(i, min(i + max_words, n_words)):
            phrase_width = _measure_phrase_width(words, i, j + 1, font)
            
            if phrase_width > max_pixel_width:
                # If even a single word is too wide, take it anyway (can't split a word)
                if j == i:
                    best_end = j + 1
                break
            best_end = j + 1
            
            # Respect sentence boundaries — break AFTER punctuation
            word = words[j]['word'].strip()
            if any(word.endswith(punct) for punct in ['.', '?', '!']) and j > i:
                best_end = j + 1
                break
            if word.endswith(',') and j > i and j - i >= 2:
                best_end = j + 1
                break
        
        raw_phrases.append((i, best_end))
        i = best_end
    
    # Phase 2: Merge short phrases (fewer than min_words) into neighbors
    merged = []
    for start_idx, end_idx in raw_phrases:
        phrase_len = end_idx - start_idx
        
        if phrase_len < min_words and merged:
            prev_start, prev_end = merged[-1]
            # Check if merging would still fit in pixel width
            combined_width = _measure_phrase_width(words, prev_start, end_idx, font)
            if combined_width <= max_pixel_width + 100:  # Allow slight overflow for merges
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
    
    # Phase 4: Merge any phrase with < 0.8s display duration into previous
    MIN_DISPLAY_SECONDS = 0.8
    _p = len(phrases) - 1
    while _p > 0:
        _start_idx, _end_idx, _p_start, _p_end = phrases[_p]
        if _p_end - _p_start < MIN_DISPLAY_SECONDS:
            prev_start, prev_end, prev_ps, prev_pe = phrases[_p - 1]
            prev_end = _end_idx
            prev_pe = _p_end
            phrases[_p - 1] = (prev_start, prev_end, prev_ps, prev_pe)
            phrases.pop(_p)
        _p -= 1
    
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


def _clean_script_for_subtitles(script_text: str) -> str:
    """
    Clean script text before subtitle alignment.
    Removes TTS pause markers (...), stray quotes, and normalizes whitespace.
    Prevents phantom words and lonely single-character display issues.
    """
    # Remove glitch/stage markers from old personality
    text = re.sub(r'\*\[.*?\]\*', '', script_text)
    # Remove [STORY N] markers from curated text
    text = re.sub(r'\[STORY\s*\d+\]\s*', '', text)
    # Remove ellipsis pause markers injected by TTS
    text = re.sub(r'\.{2,}', ' ', text)
    # Remove stray double quotes (TTS injects ... "phrase" ... for dramatic timing)
    text = text.replace('"', '')
    # Remove stray smart quotes
    text = text.replace('"', '').replace('"', '').replace(''', "'").replace(''', "'")
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _clean_display(word: str) -> str:
    """Clean word for display: keep sentence punctuation (., !, ,), strip other noise, uppercase."""
    strip_chars = ';:—–""""\''
    cleaned = word.strip(strip_chars)
    if cleaned.startswith('-'):
        cleaned = cleaned[1:]
    if cleaned.endswith('-'):
        cleaned = cleaned[:-1]
    if not cleaned or cleaned in ('"', "'", '"', '"', ''', ''', '-'):
        return ''
    return cleaned.upper()
def _render_phrase_frame(words: List[Dict], phrase_start_idx: int, phrase_end_idx: int,
                      time: float, width: int, band_h: int, font, style,
                      drift: float = 0.0) -> np.ndarray:
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
    adjusted_time = time + drift
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


def _parse_ass_time(time_str: str) -> float:
    """Parse ASS timestamp H:MM:SS.CC to seconds."""
    parts = time_str.strip().split(':')
    h = int(parts[0])
    m = int(parts[1])
    s_parts = parts[2].split('.')
    s_val = int(s_parts[0])
    cs = int(s_parts[1]) if len(s_parts) > 1 else 0
    return h * 3600 + m * 60 + s_val + cs / 100.0


def _clamp_ass_overlaps(lines: List[str]) -> List[str]:
    """
    Post-process ASS Dialogue lines to eliminate centisecond rounding overlaps.

    ASS timestamps have centisecond (0.01s) precision. When whisper word gaps
    are < 10ms, rounding can cause adjacent events to share the same centisecond
    boundary, making the renderer show both simultaneously for that frame.

    This function parses all Dialogue events, sorts by start time, and clamps
    each event's end time to be strictly less than the next event's start time.

    MIN_EVENT_CS enforces a minimum event duration of 4 centiseconds (0.04s),
    which is ~1.2 frames at 30fps — guaranteeing the subtitle is visible for
    at least one full frame rather than being rendered for 0 or sub-frame duration.
    """
    import re
    dialogue_re = re.compile(
        r'^(Dialogue:\s*\d+),(\d+:\d{2}:\d{2}\.\d{2}),(\d+:\d{2}:\d{2}\.\d{2}),(.*)$'
    )
    MIN_EVENT_CS = 4  # 0.04s minimum — at least one frame at 30fps

    parsed = []
    non_dialogue = []
    for line in lines:
        m = dialogue_re.match(line)
        if m:
            prefix = m.group(1)
            start_ts = m.group(2)
            end_ts = m.group(3)
            rest = m.group(4)
            start_s = _parse_ass_time(start_ts)
            end_s = _parse_ass_time(end_ts)
            parsed.append((start_s, end_s, prefix, start_ts, end_ts, rest))
        else:
            non_dialogue.append(line)

    if not parsed:
        return lines

    parsed.sort(key=lambda x: (x[0], x[1]))

    clamped = []
    for i in range(len(parsed)):
        start_s, end_s, prefix, start_ts, end_ts, rest = parsed[i]
        if i < len(parsed) - 1:
            next_start = parsed[i + 1][0]
            if end_s >= next_start:
                end_s = next_start - 0.01
                if end_s <= start_s:
                    end_s = start_s + MIN_EVENT_CS / 100.0
        if end_s <= start_s:
            end_s = start_s + MIN_EVENT_CS / 100.0
        h = int(end_s // 3600)
        m = int((end_s % 3600) // 60)
        s_val = int(end_s % 60)
        cs = int((end_s % 1) * 100)
        clamped_end = f"{h}:{m:02d}:{s_val:02d}.{cs:02d}"
        clamped.append(f"{prefix},{start_ts},{clamped_end},{rest}")

    return non_dialogue + clamped


def generate_ass_subtitles(
    script_text: str,
    word_timestamps: List[Dict],
    video_width: int,
    video_height: int,
    band_y_position: int,
    style: dict = None,
    hook_text: str = None,
) -> str:
    """
    Generate ASS (Advanced SubStation Alpha) subtitle content for ffmpeg burn-in.
    
    Uses karaoke \\k tags for word-by-word highlight effect:
    - Current word: highlighted in gold
    - Previous/upcoming words: lighter colors
    
    This avoids the slow moviepy per-frame Python rendering entirely.
    """
    import re

    s = {**SUBTITLE_STYLE, **(style or {})}
    
    if not word_timestamps:
        return ""

    SUBTITLE_DELAY = 0.04
    delayed_ts = [{'word': w['word'], 'start': w['start'] + SUBTITLE_DELAY, 'end': w['end'] + SUBTITLE_DELAY}
                  for w in word_timestamps]

    clean_script = _clean_script_for_subtitles(script_text) if script_text else script_text

    if clean_script and clean_script.strip():
        aligned_words = align_whisper_to_script(delayed_ts, clean_script)
    else:
        aligned_words = delayed_ts

    aligned_words = [w for w in aligned_words if _clean_display(w['word'])]

    if not aligned_words:
        return ""

    phrases = _split_into_phrases(aligned_words, max_words=5)
    if not phrases:
        return ""

    highlight_color = s.get('highlight_color', (255, 215, 0))
    previous_color = s.get('previous_color', (255, 255, 255))
    upcoming_color = s.get('upcoming_color', (180, 180, 180))
    outline_color = s.get('outline_color', (0, 0, 0))
    outline_width = s.get('outline_width', 5)
    font_size = s.get('font_size', 64)
    font_name = s.get('font_name', 'Arial-Bold')

    def _ass_color(rgb):
        return f"&H{rgb[2]:02X}{rgb[1]:02X}{rgb[0]:02X}"

    def _format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s_val = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s_val:02d}.{cs:02d}"

    hi_ass = _ass_color(highlight_color)
    prev_ass = _ass_color(previous_color)
    upc_ass = _ass_color(upcoming_color)
    ol_ass = _ass_color(outline_color)

    ass_font = "Arial"
    if "bold" in font_name.lower() or "Bold" in font_name:
        ass_font = "Arial"

    sub_y = band_y_position + 10

    lines = [
        "[Script Info]",
        "Title: The Mask Daily News",
        "ScriptType: v4.00+",
        f"PlayResX: {video_width}",
        f"PlayResY: {video_height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{ass_font},{font_size},{prev_ass},{prev_ass},{ol_ass},&H80000000,-1,0,0,0,100,100,0,0,1,{outline_width},2,2,30,30,10,1",
        f"Style: Title,{ass_font},48,{hi_ass},{hi_ass},{ol_ass},&H80000000,-1,0,0,0,100,100,0,0,1,3,2,8,30,30,10,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    if hook_text and hook_text.strip():
        title_words = hook_text.strip().split()
        if len(title_words) > 10:
            title_display = " ".join(title_words[:10])
        else:
            title_display = " ".join(title_words)
        title_y = 20
        lines.append(
            f"Dialogue: 0,0:00:00.00,9:59:59.99,Title,,0,0,0,,"
            f"{{\\an8}}{{\\pos({video_width // 2},{title_y})}}{title_display}"
        )

    TAIL_PAD = 0.01

    for phrase_idx, (start_idx, end_idx, phrase_start, phrase_end) in enumerate(phrases):
        lead_in = s.get('lead_in_seconds', 0.05)
        prev_phrase_end = (phrases[phrase_idx - 1][3] + TAIL_PAD + 0.01) if phrase_idx > 0 else 0
        sub_start = max(prev_phrase_end, phrase_start - lead_in)
        sub_end = phrase_end + TAIL_PAD

        if sub_end <= sub_start:
            continue

        displayable_indices = []
        for i in range(start_idx, end_idx):
            if _clean_display(aligned_words[i]['word']):
                displayable_indices.append(i)

        if not displayable_indices:
            continue

        n_displayable = len(displayable_indices)

        # Build a sequence of (start_time, end_time, word_global_idx) for each displayable word
        word_spans = []
        for di, word_abs in enumerate(displayable_indices):
            w_start = aligned_words[word_abs]['start']
            if di < n_displayable - 1:
                next_abs = displayable_indices[di + 1]
                w_end = aligned_words[next_abs]['start']
            else:
                w_end = sub_end
            if w_end <= w_start:
                w_end = w_start + 0.15
            word_spans.append((w_start, w_end, word_abs))

        if not word_spans:
            continue

        # Use a SINGLE Dialogue line per phrase with dynamic color overrides.
        # This eliminates flickering caused by per-word Dialogue line replacement.
        # We split the phrase into time segments where the active word changes, 
        # creating one Dialogue per segment but with smart overlap handling.
        # 
        # Actually, the most flicker-free approach is to use a single Dialogue 
        # spanning the full phrase with {\k} karaoke timing for each word,
        # and color overrides that change per word.
        #
        # ASS \k tag: {\k<centiseconds>}<text> colors the text up to that duration,
        # then moves to next {\k} block. We use one {\k} per word.
        #
        # Color scheme for karaoke: previously-spoken words in white, 
        # currently-speaking word in gold, upcoming words in grey.
        # Since \k only gives 2 states (before-karaoke and after-karaoke per block),
        # we use {\c} overrides before each word to set the desired color.

        phrase_duration_cs = int((sub_end - sub_start) * 100)
        if phrase_duration_cs <= 0:
            phrase_duration_cs = 1

        # Build the full-phrase text with per-word color and timing
        # We create timed override segments:
        # For each moment the active-word changes, we output a new Dialogue 
        # BUT with continuous display (same phrase text, different highlight).
        # 
        # To avoid flicker entirely, we use ONE Dialogue per phrase with {\k} tags.
        # The trick: use {\1c&H<color>} override before each word, and {\k<cs>}
        # timing to control when the karaoke highlight moves to the next word.
        #
        # For 3-color karaoke (white/gold/grey), we need to set each word's 
        # color explicitly with {\1c} overrides since \k only provides 2 states.

        # Simpler approach: one Dialogue per phrase, with inline {\c} color tags 
        # and {\k} timing. The entire phrase is visible at once, and each word
        # has its color set by an override block.
        #
        # Format: {\k<cs_before_first_word>}{\c&H<grey>}\1c&H<grey>}word1 
        #         {\k<cs_word1_duration>}{\c&H<gold>\1c&H<gold>}word2 
        #         {\k<cs_word2_duration>}{\c&H<grey>\1c&H<grey>}word3 ...
        #
        # This keeps the phrase on screen continuously — no flicker.

        # Build karaoke text: each word preceded by its duration tag and color
        karaoke_parts = []
        for di, (w_start, w_end, word_abs) in enumerate(word_spans):
            word = _clean_display(aligned_words[word_abs]['word'])
            if not word:
                continue
            duration_cs = max(int((w_end - w_start) * 100), 1)

            if di == 0:
                # First word: gold (highlighted immediately)
                # Use a small {\k0} to start immediately, then {\k<duration>} for this word
                karaoke_parts.append(f"{{\\c{hi_ass}\\1c{hi_ass}}}{word}")
            else:
                # All other words: grey initially, will be overridden by timed segments
                karaoke_parts.append(f"{{\\c{upc_ass}\\1c{upc_ass}}}{word}")

        karaoke_text = " ".join(karaoke_parts)

        if not karaoke_text.strip():
            continue

        # Now build the timed override sequence using multiple Dialogues 
        # that each show the FULL phrase text but with different words highlighted.
        # To minimize flicker, we use ASS "{\pos}" positioning and ensure 
        # consecutive Dialogues overlap by 1 frame so there's no gap.
        #
        # KEY INSIGHT: libass renders overlapping Dialogue events by showing 
        # the LATER one on top. So we DON'T want overlaps for the same style — 
        # we want seamless transitions.
        #
        # The FLICKER-FREE approach: ONE Dialogue per phrase, with {\t()} 
        # animated override tags to change colors at specific times.

        # Build animated overrides using {\t(<start>,<end>,\c&H<color>\1c&H<color>)} 
        # tags within a single Dialogue line.
        # {\t()} changes a style at a specific time within the Dialogue's duration.

        # Collect color transition events: (time_offset_ms, word_index, new_color)
        transition_events = []
        for di, (w_start, w_end, word_abs) in enumerate(word_spans):
            word = _clean_display(aligned_words[word_abs]['word'])
            if not word:
                continue
            # When this word becomes active (highlight)
            start_offset_ms = int((w_start - sub_start) * 1000)
            transition_events.append((start_offset_ms, di, 'highlight'))
            # When this word stops being active (becomes previous)
            end_offset_ms = int((w_end - sub_start) * 1000)
            transition_events.append((end_offset_ms, di, 'previous'))

        # Build the single Dialogue line with embedded {\t()} animated color overrides
        # per word. Each word starts grey, transitions to gold when active, then white.
        animated_parts = []
        for di, (w_start, w_end, word_abs) in enumerate(word_spans):
            word = _clean_display(aligned_words[word_abs]['word'])
            if not word:
                continue
            start_offset_ms = max(int((w_start - sub_start) * 1000), 0)
            end_offset_ms = max(int((w_end - sub_start) * 1000), 0)

            # Start grey, transition to gold at start_offset, then to white at end_offset
            animated_parts.append(
                f"{{\\1c{upc_ass}\\c{upc_ass}\\t({start_offset_ms},{start_offset_ms},\\1c{hi_ass}\\c{hi_ass})\\t({end_offset_ms},{end_offset_ms},\\1c{prev_ass}\\c{prev_ass}))}}"
                f"{word}"
            )

        animated_text = " ".join(animated_parts)

        if not animated_text.strip():
            continue

        lines.append(
            f"Dialogue: 0,{_format_time(sub_start)},{_format_time(sub_end)},Default,,0,0,0,,"
            f"{{\\an8}}{{\\pos({video_width // 2},{sub_y})}}{animated_text}"
        )

    lines = _clamp_ass_overlaps(lines)
    content = "\n".join(lines)

    if not content or "Dialogue:" not in content:
        print(f"  [ASS] WARNING: generate_ass_subtitles produced {len(lines)} lines but no Dialogue events — subtitles will be missing")
        return ""

    return content


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

    # Apply subtitle delay to compensate for whisper's early start predictions.
    # Structural batch splitting now places silences at story boundaries,
    # making whisper timestamps more accurate. A small residual offset remains.
    SUBTITLE_DELAY = 0.04
    wt = [{'word': w['word'], 'start': w['start'] + SUBTITLE_DELAY, 'end': w['end'] + SUBTITLE_DELAY} for w in word_timestamps]
    word_timestamps = wt

    # Clean script text — remove TTS pause markers, stray quotes, etc.
    clean_script = _clean_script_for_subtitles(script_text) if script_text else script_text

    # Align whisper timestamps to original script words
    if clean_script and clean_script.strip():
        aligned_words = align_whisper_to_script(word_timestamps, clean_script)
        print(f"  [SUB] Word alignment: {len(word_timestamps)} whisper -> {len(aligned_words)} script words")
    else:
        aligned_words = word_timestamps

    # Filter out any words that render as empty after cleaning
    aligned_words = [w for w in aligned_words if _clean_display(w['word'])]

    drift = 0.0

    phrases = _split_into_phrases(aligned_words, max_words=5)
    if not phrases:
        return []

    lead_in = s.get('lead_in_seconds', 0.3)

    clips = []
    for i, (start_idx, end_idx, phrase_start, phrase_end) in enumerate(phrases):
        prev_clip_end = (phrases[i-1][3] + 0.01) if i > 0 else 0
        clip_start = max(prev_clip_end, phrase_start - lead_in)
        clip_end = phrase_end + 0.01
        duration = clip_end - clip_start
        
        if duration <= 0:
            continue

        def make_frame(t, si=start_idx, ei=end_idx, cs=clip_start, dr=drift):
            frame = _render_phrase_frame(
                aligned_words, si, ei,
                cs + t, video_width, band_h, font, s, drift=dr
            )
            return frame[:, :, :3]

        clip = VideoClip(make_frame, duration=duration)
        clip = clip.set_start(clip_start)
        clip = clip.set_position((0, band_y_position))
        mask_clip = VideoClip(
            lambda t, si=start_idx, ei=end_idx, cs=clip_start, dr=drift: _make_alpha_mask(
                t, cs, si, ei, aligned_words, video_width, band_h, font, s, dr),
            ismask=True, duration=duration
        )
        clip = clip.set_mask(mask_clip)
        clips.append(clip)

    total_covered = sum(c.duration for c in clips)
    print(f"  [SUB] Phrase subtitles: {len(clips)} clips ({len(aligned_words)} words, {len(phrases)} phrases, {total_covered:.1f}s coverage)")

    return clips


def _make_alpha_mask(t, cs, si, ei, aligned_words, video_width, band_h, font, style, drift=0.0):
    """Create alpha mask for subtitle transparency."""
    frame = _render_phrase_frame(aligned_words, si, ei, cs + t, video_width, band_h, font, style, drift=drift)
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
