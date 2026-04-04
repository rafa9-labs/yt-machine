"""
Split-Screen Video Assembler — Scene images top 60% + avatar bottom 40%.
Composites a 1080x1920 vertical video for TikTok/Reels/Shorts.
Layout (Option A — 60/40 split):
  - SCENE AREA (1080x1152): Full scene images with animated zoom/pan (top 60%)
  - TITLE OVERLAY: Hook text at the top with fade-in
  - SUBTITLES: Karaoke-style outlined text near the bottom of the scene area
  - AVATAR AREA (1080x768): Looping avatar animation (bottom 40%)
"""

import os
import math
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from PIL import Image

try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, ImageClip,
        CompositeVideoClip, concatenate_videoclips, vfx
    )
except ImportError:
    from moviepy import (
        VideoFileClip, AudioFileClip, ImageClip,
        CompositeVideoClip, concatenate_videoclips
    )

from .subtitle_renderer import create_subtitle_clips, create_title_clip

# Layout constants — Option A: 60% image / 40% avatar
VIDEO_W = 1080
VIDEO_H = 1920
TOP_H = 1152       # Scene image area (60% of screen)
BOTTOM_H = 768     # Avatar area (40% of screen)
FPS = 30

# Avatar asset path
AVATAR_PATH = Path(__file__).parent.parent / "assets" / "avatar" / "avatar_loop.mp4"


def _resize_image_fullscreen(img_path: str) -> ImageClip:
    """
    Load image and fit it into the VISIBLE top area (1080x1152).
    
    Native scene images are generated at 1088×1152 (flux requires mult of 16).
    This crops just 4px per side — virtually no content lost.
    Legacy images (square, portrait, etc.) use full cover-crop fallback.
    """
    img = Image.open(img_path)
    img_w, img_h = img.size

    # Target: fit image into the visible top area (1080x1152)
    visible_w = VIDEO_W   # 1080
    visible_h = TOP_H     # 1152

    # Check if image is already native scene size (1088×1152) — just crop 4px per side
    if img_w >= visible_w and img_h >= visible_h and abs(img_w - visible_w) <= 16 and abs(img_h - visible_h) <= 16:
        # Nearly native — crop the tiny excess (e.g., 1088→1080 = 4px per side)
        left = (img_w - visible_w) // 2
        top = (img_h - visible_h) // 2
        img = img.crop((left, top, left + visible_w, top + visible_h))
        print(f"  [SPLIT] Native scene image {img_w}×{img_h} → crop {left}px sides, {top}px top/bot")
    else:
        # Legacy image — cover-crop to fill visible area
        target_ratio = visible_w / visible_h  # ~0.9375
        img_ratio = img_w / img_h

        if img_ratio > target_ratio:
            new_w = int(img_h * target_ratio)
            left = (img_w - new_w) // 2
            img = img.crop((left, 0, left + new_w, img_h))
        else:
            new_h = int(img_w / target_ratio)
            top = (img_h - new_h) // 2
            img = img.crop((0, top, img_w, top + new_h))

        # Resize to visible area dimensions
        img = img.resize((visible_w, visible_h), Image.LANCZOS)
        print(f"  [SPLIT] Legacy image {img_w}×{img_h} → cover-crop to {visible_w}×{visible_h}")

    # Create full-frame canvas (1080x1920) and paste image at top
    canvas = Image.new('RGB', (VIDEO_W, VIDEO_H), (10, 5, 25))
    canvas.paste(img, (0, 0))

    # Save to temp file for moviepy
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    canvas.save(tmp.name, 'PNG')
    tmp.close()

    return ImageClip(tmp.name)


def _apply_zoom_out(clip: ImageClip, duration: float) -> ImageClip:
    """
    Animated zoom-out: starts tight (cropped center) and gradually reveals full image.
    Subtle 5% zoom — enough for life without losing image content.
    """
    def zoom_out_transform(get_frame, t):
        frame = get_frame(t)
        h, w = frame.shape[:2]

        # Start at 5% crop (subtle), end at 0% crop (full frame)
        progress = t / duration if duration > 0 else 1.0
        crop_pct = 0.05 * (1.0 - progress)

        crop_h = int(h * crop_pct)
        crop_w = int(w * crop_pct)

        if crop_h > 0 and crop_w > 0:
            cropped = frame[crop_h:h-crop_h, crop_w:w-crop_w]
            pil_img = Image.fromarray(cropped).resize((w, h), Image.LANCZOS)
            return np.array(pil_img)
        return frame

    return clip.fl(zoom_out_transform)


def _apply_pan_top_to_bottom(clip: ImageClip, duration: float) -> ImageClip:
    """
    Animated pan from top to bottom — no wrapping.
    Shifts the image down and fills the exposed top with the background color.
    """
    bg_color = np.array([10, 5, 25], dtype=np.uint8)  # Dark navy background

    def pan_transform(get_frame, t):
        frame = get_frame(t)
        h, w = frame.shape[:2]

        progress = t / duration if duration > 0 else 1.0

        # Pan range: scroll ~5% of frame height (subtle, no wrapping)
        pan_range = int(h * 0.05)
        offset = int(pan_range * progress)

        if offset > 0:
            result = np.empty_like(frame)
            # Shift image down by offset
            result[offset:, :] = frame[:h-offset, :]
            # Fill exposed top with background color (no wrapping!)
            result[:offset, :] = bg_color
            return result
        return frame

    return clip.fl(pan_transform)


def _apply_scene_effect(clip: ImageClip, scene_idx: int, duration: float) -> ImageClip:
    """
    Apply alternating effects to scene clips:
    - Even indices: zoom out
    - Odd indices: pan top to bottom
    """
    if scene_idx % 2 == 0:
        return _apply_zoom_out(clip, duration)
    else:
        return _apply_pan_top_to_bottom(clip, duration)


def _calculate_scene_durations(total_duration: float, num_scenes: int) -> List[float]:
    """Calculate varied scene durations for natural pacing."""
    if num_scenes == 0:
        return []
    if num_scenes == 1:
        return [total_duration]

    weights = []
    for i in range(num_scenes):
        if i == 0:
            weights.append(1.2)
        elif i == num_scenes - 1:
            weights.append(1.15)
        elif i == num_scenes // 2:
            weights.append(0.9)
        else:
            weights.append(0.95 + (i % 2) * 0.1)

    total_weight = sum(weights)
    durations = [total_duration * (w / total_weight) for w in weights]
    return durations


def _prepare_avatar_bottom(avatar_path: str, total_duration: float) -> VideoFileClip:
    """
    Load the avatar video, crop/resize to FILL 1080x768 bottom area, and loop.
    """
    avatar = VideoFileClip(avatar_path)
    av_w, av_h = avatar.size
    avatar_duration = avatar.duration

    scale_factor = BOTTOM_H / av_h
    scaled_w = int(av_w * scale_factor)
    scaled_h = BOTTOM_H

    avatar_scaled = avatar.resize((scaled_w, scaled_h))

    if scaled_w > VIDEO_W:
        crop_x = (scaled_w - VIDEO_W) // 2
        avatar_cropped = avatar_scaled.crop(
            x1=crop_x, y1=0, x2=crop_x + VIDEO_W, y2=scaled_h
        )
    else:
        avatar_cropped = avatar_scaled

    avatar_resized = avatar_cropped

    loops_needed = math.ceil(total_duration / avatar_duration)

    if loops_needed > 1:
        clips = [avatar_resized]
        for _ in range(loops_needed - 1):
            clips.append(avatar_resized.copy())
        avatar_looped = concatenate_videoclips(clips)
    else:
        avatar_looped = avatar_resized

    avatar_final = avatar_looped.subclip(0, total_duration)

    return avatar_final


def build_split_video(
    audio_path: str,
    image_paths: List[str],
    avatar_path: str = None,
    output_path: str = None,
    script_text: str = None,
    word_timestamps: list = None,
    hook_text: str = None,
) -> dict:
    """
    Build a video: scene background top 60%, avatar bottom 40%, title at top,
    karaoke subtitles near bottom of scene area.

    Args:
        audio_path: Path to voiceover MP3
        image_paths: List of scene image paths (4-6 images)
        avatar_path: Path to avatar loop video
        output_path: Output MP4 path
        script_text: Full narration text (for subtitle word alignment)
        word_timestamps: Word timing data from whisper
        hook_text: Hook text for title overlay at top

    Returns:
        dict with success status, output path, and metadata
    """
    if not audio_path or not Path(audio_path).exists():
        return {"success": False, "error": f"Audio not found: {audio_path}"}
    if not image_paths:
        return {"success": False, "error": "No image paths provided"}

    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
        return {"success": False, "error": f"Missing images: {missing}"}

    if not avatar_path:
        avatar_path = str(AVATAR_PATH)
    if not Path(avatar_path).exists():
        return {"success": False, "error": f"Avatar video not found: {avatar_path}"}

    try:
        audio = AudioFileClip(audio_path)
        total_dur = audio.duration
        print(f"  [SPLIT] Audio duration: {total_dur:.1f}s")

        # ── SCENE BACKGROUND: Images fitted to visible top area ──
        num_scenes = len(image_paths)
        scene_durations = _calculate_scene_durations(total_dur, num_scenes)

        scene_clips = []
        for idx, img_path in enumerate(image_paths):
            dur = scene_durations[idx]
            clip = _resize_image_fullscreen(img_path).set_duration(dur)
            clip = _apply_scene_effect(clip, idx, dur)
            scene_clips.append(clip)

        background = concatenate_videoclips(scene_clips, method="compose")
        if background.size != [VIDEO_W, VIDEO_H]:
            background = background.resize((VIDEO_W, VIDEO_H))

        print(f"  [SPLIT] Background: {num_scenes} scenes (60/40 layout)")

        # ── BOTTOM AREA: Avatar loop (40% of screen) ──
        bottom_half = _prepare_avatar_bottom(avatar_path, total_dur)
        bottom_half = bottom_half.set_position((0, TOP_H))
        print(f"  [SPLIT] Avatar: {BOTTOM_H}px at y={TOP_H}")

        # ── COMPOSITE: Stack all layers ──
        layers = [background, bottom_half]

        # ── TITLE OVERLAY: Hook text at top ──
        if hook_text:
            title_clip = create_title_clip(
                hook_text, VIDEO_W, VIDEO_H,
                duration=min(5.0, total_dur)
            )
            if title_clip:
                layers.append(title_clip)
                print(f"  [SPLIT] Title overlay: \"{hook_text[:50]}...\"")

        # ── SUBTITLES: Near bottom of scene area (above avatar) ──
        subtitle_clips = []
        if word_timestamps and len(word_timestamps) > 0:
            # Position subtitles just above the avatar zone
            subtitle_y = TOP_H - 140  # 140px above avatar boundary
            subtitle_clips = create_subtitle_clips(
                script_text=script_text or "",
                word_timestamps=word_timestamps,
                video_width=VIDEO_W,
                video_height=VIDEO_H,
                band_y_position=subtitle_y,
            )
            layers.extend(subtitle_clips)
            print(f"  [SPLIT] Subtitles: {len(subtitle_clips)} clips at y={subtitle_y}")
        else:
            print(f"  [SPLIT] No subtitle data — skipping subtitles")

        final = CompositeVideoClip(layers, size=(VIDEO_W, VIDEO_H))
        final = final.set_audio(audio)

        # ── EXPORT ──
        if not output_path:
            output_dir = Path(__file__).parent.parent / "output" / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"split_{int(total_dur)}s.mp4")

        print(f"  [SPLIT] Exporting to {output_path}...")
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=FPS,
            verbose=False,
            logger=None,
        )

        out_path = Path(output_path)
        if not out_path.exists():
            return {"success": False, "error": "Output file not created"}

        file_size = out_path.stat().st_size

        # Cleanup
        final.close()
        audio.close()
        background.close()
        bottom_half.close()
        for c in scene_clips:
            c.close()

        return {
            "success": True,
            "path": str(out_path),
            "duration_seconds": round(total_dur, 2),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "resolution": f"{VIDEO_W}x{VIDEO_H}",
            "fps": FPS,
            "scenes": num_scenes,
            "subtitles": len(subtitle_clips),
            "effects_applied": ["full_screen_bg", "animated_zoom_pan", "avatar_loop", "title_overlay", "karaoke_subtitles"],
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Split video build failed: {str(e)}"}