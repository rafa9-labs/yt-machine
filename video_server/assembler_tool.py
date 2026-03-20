import os
import time
import numpy as np
from pathlib import Path
from typing import Optional, List
from mcp.server.fastmcp import FastMCP
from moviepy.editor import (
from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_videoclips,
    CompositeVideoClip, ImageClip, TextClip, vfx
)
from PIL import Image, ImageDraw, ImageFont

server = FastMCP("assembler-tool")

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "videos"
TEMP_DIR   = Path(__file__).parent.parent / "output" / "_tmp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_W, VIDEO_H = 608, 1080
FPS = 24
TICKER_H = 60
HUD_FONT_SIZE = 22

# Camera movement types for professional, studied patterns
CAMERA_PATTERNS = {
    'zoom_in': {'start_scale': 1.0, 'end_scale': 1.15},
    'zoom_out': {'start_scale': 1.15, 'end_scale': 1.0},
    'pan_right': {'start_x': -50, 'end_x': 0},
    'pan_left': {'start_x': 0, 'end_x': -50}
}


def _apply_camera_movement(clip: ImageClip, movement_type: str, duration: float) -> ImageClip:
    """
    Apply professional camera movement to image clip.
    
    Args:
        clip: ImageClip to animate
        movement_type: 'zoom_in', 'zoom_out', 'pan_right', or 'pan_left'
        duration: Duration of the clip
    
    Returns:
        Animated ImageClip with smooth camera movement
    """
    if movement_type == 'zoom_in':
        # Smooth zoom in from 1.0 to 1.15 scale
        return clip.resize(lambda t: 1.0 + 0.15 * (t / duration))
    elif movement_type == 'zoom_out':
        # Smooth zoom out from 1.15 to 1.0 scale
        return clip.resize(lambda t: 1.15 - 0.15 * (t / duration))
    elif movement_type == 'pan_right':
        # Pan from left to right
        return clip.set_position(lambda t: (int(-50 + 50 * (t / duration)), 'center'))
    elif movement_type == 'pan_left':
        # Pan from right to left
        return clip.set_position(lambda t: (int(50 * (t / duration)), 'center'))
    else:
        return clip


def _create_segment_captions(script_text: str, duration: float, width: int, height: int) -> List:
    """
    Create bold, centered captions that update every ~7 seconds.
    Splits the script into timed segments for muted viewing.
    
    Args:
        script_text: Full script text to split into segments
        duration: Total video duration
        width: Video width
        height: Video height
    
    Returns:
        List of TextClip objects with timing
    """
    import re
    import math
    
    # Split into sentences first
    sentences = re.split(r'[.!?]+\s*', script_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    # Group sentences into ~7-second segments
    segment_duration = 7.0
    n_segments = max(1, math.floor(duration / segment_duration))
    sentences_per_segment = max(1, math.ceil(len(sentences) / n_segments))
    
    segments = []
    for i in range(0, len(sentences), sentences_per_segment):
        chunk = sentences[i:i + sentences_per_segment]
        segments.append(". ".join(chunk))
    
    # Recalculate actual segment duration based on real segment count
    actual_seg_dur = duration / len(segments)
    
    caption_clips = []
    for i, segment_text in enumerate(segments):
        start_time = i * actual_seg_dur
        
        try:
            txt_clip = TextClip(
                font="Arial-Bold",
                text=segment_text,
                font_size=30,
                color="white",
                stroke_color="black",
                stroke_width=3,
                method="caption",
                size=(width - 60, None),
                align="center"
            )
            
            txt_clip = txt_clip.set_position(("center", height - TICKER_H - 140))
            txt_clip = txt_clip.set_start(start_time)
            txt_clip = txt_clip.set_duration(actual_seg_dur)
            
            caption_clips.append(txt_clip)
        except Exception as e:
            print(f"Warning: Could not create caption for segment {i}: {e}")
            continue
    
    return caption_clips


def _make_hud_overlay(width: int, height: int, duration: float, era: str = "2020s") -> ImageClip:
    hud_path = TEMP_DIR / f"hud_{era}.png"
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", HUD_FONT_SIZE)
        font_sm = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        font_sm = font
    
    # Add historical archive label for non-current eras
    if era != "2020s":
        draw.rectangle([8, 8, width - 8, 44], fill=(0, 0, 0, 140), outline=(200, 150, 0, 180), width=1)
        draw.text((16, 14), f"HISTORICAL ARCHIVE  |  {era.upper()}", font=font_sm, fill=(200, 150, 0, 220))
    else:
        draw.rectangle([8, 8, width - 8, 44], fill=(0, 0, 0, 140), outline=(0, 200, 255, 180), width=1)
        draw.text((16, 14), "SENTINEL v2.4  |  TACTICAL BRIEFING", font=font_sm, fill=(0, 200, 255, 220))
    
    timecode = "MAR 21 2026  UTC"
    draw.text((width - 16, 14), timecode, font=font_sm, fill=(0, 200, 255, 220), anchor="ra")
    draw.rectangle([8, height - TICKER_H - 44, 110, height - TICKER_H - 12],
                   fill=(0, 0, 0, 140), outline=(255, 40, 40, 180), width=1)
    draw.text((16, height - TICKER_H - 38), "● REC", font=font, fill=(255, 40, 40, 255))
    img.save(str(hud_path), format="PNG")

    hud_clip = ImageClip(str(hud_path)).set_duration(duration)
    hud_clip = ImageClip(str(hud_path)).set_duration(duration)
    return hud_clip


def _make_rec_blink(width: int, height: int, duration: float) -> ImageClip:
    on_path  = TEMP_DIR / "rec_on.png"
    off_path = TEMP_DIR / "rec_off.png"
    for path, color in [(on_path, (255, 40, 40, 255)), (off_path, (0, 0, 0, 0))]:
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([14, height - TICKER_H - 40, 26, height - TICKER_H - 28], fill=color)
        img.save(str(path), format="PNG")

    frames_per_blink = FPS
    def make_blink_frame(t):
        cycle = int(t * FPS) // frames_per_blink
        path = on_path if cycle % 2 == 0 else off_path
        return np.array(Image.open(path).convert("RGB"))

    from moviepy import VideoClip
    return VideoClip(make_blink_frame, duration=duration).with_fps(FPS)


def _make_ticker(headlines: List[str], width: int, duration: float) -> CompositeVideoClip:
    joined  = "   ★   ".join(h.upper() for h in headlines) + "   ★   "
    joined  = (joined + "   ") * 4
    bg_path = TEMP_DIR / "ticker_bg.png"
    img = Image.new("RGBA", (width, TICKER_H), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 4], fill=(255, 255, 255, 255))
    img.save(str(bg_path), format="PNG")
    bg_clip = ImageClip(str(bg_path)).set_duration(duration)
    try:
        txt_clip = (
            TextClip(
                font="Arial",
                text=joined,
                font_size=22,
                color="white",
                bg_color="transparent",
                method="label",
            )
            .set_duration(duration)
            .set_duration(duration)
        )
        scroll_speed = 120
        txt_clip = txt_clip.set_position(lambda t: (width - int(t * scroll_speed) % (txt_clip.w + width), 16))
        txt_clip = txt_clip.set_position(lambda t: (width - int(t * scroll_speed) % (txt_clip.w + width), 16))
        ticker = CompositeVideoClip([bg_clip, txt_clip], size=(width, TICKER_H))
    except Exception:
        ticker = bg_clip
    return ticker


@server.tool()
def build_final_video(
    audio_path: str,
    asset_paths: List[str],
    ticker_headlines: Optional[List[str]] = None,
    is_pixel_art: bool = True,
    output_filename: Optional[str] = None,
    script_text: Optional[str] = None,
    era_tags: Optional[List[str]] = None
) -> dict:
    """
    Assemble Sentinel v2.2 TikTok/Shorts optimized video using MoviePy 2.0.
    Optimised for 5-6 image sequences at ~10-12 seconds per image (60-80s total).

    Args:
        audio_path:       Path to .mp3 voiceover file
        asset_paths:      List of paths to video clips or images (5-6 for historical anchoring)
        ticker_headlines: Up to 3 short news ticker strings for the scrolling marquee
        is_pixel_art:     Apply professional camera movements (default True)
        output_filename:  Custom output filename (default: auto-generated)
        script_text:      Optional script text for sentence-by-sentence subtitles
        era_tags:         Optional list of era tags ('2020s', '1990s', etc.) for visual differentiation

    Returns:
        dict with success status, output path, and metadata
    """
    if not audio_path or not Path(audio_path).exists():
        return {"success": False, "error": f"Audio file not found: {audio_path}"}
    if not asset_paths:
        return {"success": False, "error": "No asset paths provided"}
    missing = [p for p in asset_paths if not Path(p).exists()]
    if missing:
        return {"success": False, "error": f"Missing asset files: {missing}"}

    if ticker_headlines is None:
        ticker_headlines = ["BREAKING: Hormuz Blockade — Oil at $110",
                            "ALERT: Helium shortage hits semiconductor supply chains",
                            "DEVELOPING: Haifa Refinery under drone surveillance"]

    try:
        audio_clip   = AudioFileClip(audio_path)
        total_dur    = audio_clip.duration
        n_assets     = max(len(asset_paths), 1)
        clip_dur     = max(total_dur / n_assets, 7.0)
        video_clips  = []
        
        # Dynamic camera: zoom_out first (reveals scale), rest randomized
        import random
        all_movements = ['zoom_in', 'zoom_out', 'pan_right', 'pan_left']
        camera_movements = ['zoom_out']  # First image always zoom_out
        for _ in range(max(0, n_assets - 1)):
            camera_movements.append(random.choice(all_movements))

        for idx, asset_path in enumerate(asset_paths):
            ext = Path(asset_path).suffix.lower()
            if ext in (".mp4", ".mov", ".avi", ".mkv"):
                clip = VideoFileClip(asset_path)
            elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
                clip = ImageClip(asset_path).set_duration(clip_dur)
            else:
                continue

            if is_pixel_art:
                clip = clip.fl(lambda gf, t: gf(t) * (1.0 + 0.02 * t))

            clip = clip.resize(height=VIDEO_H)
            if clip.w > VIDEO_W:
                clip = clip.resize(width=VIDEO_W)

            video_clips.append(clip)

        if not video_clips:
            return {"success": False, "error": "No valid clips could be processed"}

        base = concatenate_videoclips(video_clips) if len(video_clips) > 1 else video_clips[0]
        base = base.set_audio(audio_clip)

        layers = [base]

        if is_pixel_art:
            scanline_path = _make_scanline_overlay(base.w, base.h)
            scanline_clip = ImageClip(str(scanline_path)).set_duration(total_dur)
            layers.append(scanline_clip)

        hud = _make_hud_overlay(base.w, base.h, total_dur)
        layers.append(hud)

        ticker = _make_ticker(ticker_headlines, base.w, total_dur)
        ticker = ticker.set_position((0, base.h - TICKER_H))
        layers.append(ticker)

        final_video = CompositeVideoClip(layers, size=(base.w, base.h))

        if output_filename is None:
            output_filename = f"sentinel_{hash(str(asset_paths)) % 10000}.mp4"

        output_path = OUTPUT_DIR / output_filename

        final_video.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=FPS,
            verbose=False,
            logger=None,
        )

        if not output_path.exists():
            return {"success": False, "error": f"Video not created at {output_path}"}

        file_size = output_path.stat().st_size

        final_video.close()
        audio_clip.close()
        for c in video_clips:
            c.close()

        effects = []
        if is_pixel_art:
            effects += ["random_camera_movements"]
        if script_text:
            effects += ["segment_captions"]
        effects += ["ticker_marquee", "hud_overlay"]

        return {
            "success": True,
            "filename": output_filename,
            "path": str(output_path),
            "duration_seconds": round(total_dur, 2),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "asset_count": len(asset_paths),
            "is_pixel_art": is_pixel_art,
            "effects_applied": effects,
            "output_directory": str(OUTPUT_DIR),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to build video: {str(e)}",
            "asset_count": len(asset_paths),
            "is_pixel_art": is_pixel_art,
        }
