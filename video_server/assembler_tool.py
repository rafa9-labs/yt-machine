import os
import time
import numpy as np
from pathlib import Path
from typing import Optional, List
from mcp.server.fastmcp import FastMCP
from moviepy import (
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


def _make_scanline_overlay(width: int, height: int) -> Path:
    overlay_path = TEMP_DIR / "scanline_overlay.png"
    if overlay_path.exists():
        return overlay_path
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for y in range(0, height, 2):
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, 38))
    img.save(str(overlay_path), format="PNG")
    return overlay_path


def _make_hud_overlay(width: int, height: int, duration: float) -> ImageClip:
    hud_path = TEMP_DIR / "hud_base.png"
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", HUD_FONT_SIZE)
        font_sm = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        font_sm = font
    draw.rectangle([8, 8, width - 8, 44], fill=(0, 0, 0, 140), outline=(0, 200, 255, 180), width=1)
    draw.text((16, 14), "SENTINEL v2.1  |  TACTICAL BRIEFING", font=font_sm, fill=(0, 200, 255, 220))
    timecode = "MAR 20 2026  UTC"
    draw.text((width - 16, 14), timecode, font=font_sm, fill=(0, 200, 255, 220), anchor="ra")
    draw.rectangle([8, height - TICKER_H - 44, 110, height - TICKER_H - 12],
                   fill=(0, 0, 0, 140), outline=(255, 40, 40, 180), width=1)
    draw.text((16, height - TICKER_H - 38), "● REC", font=font, fill=(255, 40, 40, 255))
    img.save(str(hud_path), format="PNG")

    hud_clip = ImageClip(str(hud_path)).with_duration(duration)
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
    img = Image.new("RGBA", (width, TICKER_H), (0, 0, 0, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 4], fill=(0, 200, 255, 255))
    img.save(str(bg_path), format="PNG")
    bg_clip = ImageClip(str(bg_path)).with_duration(duration)
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
            .with_duration(duration)
        )
        scroll_speed = 120
        txt_clip = txt_clip.with_position(lambda t: (width - int(t * scroll_speed) % (txt_clip.w + width), 16))
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
    output_filename: Optional[str] = None
) -> dict:
    """
    Assemble Sentinel Tactical Briefing video using MoviePy 2.0.
    Optimised for 6-image sequences at ~7-8 seconds per image (45s total).

    Args:
        audio_path:       Path to .mp3 voiceover file
        asset_paths:      List of paths to video clips or images (ideally 6)
        ticker_headlines: Up to 3 short news ticker strings for the scrolling marquee
        is_pixel_art:     Apply Ken Burns zoom + CRT scanline effects (default True)
        output_filename:  Custom output filename (default: auto-generated)

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

        for asset_path in asset_paths:
            ext = Path(asset_path).suffix.lower()
            if ext in (".mp4", ".mov", ".avi", ".mkv"):
                clip = VideoFileClip(asset_path)
            elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
                clip = ImageClip(asset_path).with_duration(clip_dur)
            else:
                continue

            if is_pixel_art:
                clip = clip.with_effects([vfx.Resize(lambda t: 1.0 + 0.02 * t)])

            clip = clip.resized(height=VIDEO_H)
            if clip.w > VIDEO_W:
                clip = clip.resized(width=VIDEO_W)

            video_clips.append(clip)

        if not video_clips:
            return {"success": False, "error": "No valid clips could be processed"}

        base = concatenate_videoclips(video_clips) if len(video_clips) > 1 else video_clips[0]
        base = base.with_audio(audio_clip)

        layers = [base]

        if is_pixel_art:
            scanline_path = _make_scanline_overlay(base.w, base.h)
            scanline_clip = ImageClip(str(scanline_path)).with_duration(total_dur)
            layers.append(scanline_clip)

        hud = _make_hud_overlay(base.w, base.h, total_dur)
        layers.append(hud)

        ticker = _make_ticker(ticker_headlines, base.w, total_dur)
        ticker = ticker.with_position((0, base.h - TICKER_H))
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
            effects += ["ken_burns_zoom", "crt_scanlines"]
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
