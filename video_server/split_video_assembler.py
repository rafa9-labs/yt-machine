"""
Split-Screen Video Assembler — Top: scene images, Bottom: avatar animation.
Composites a 1080×1920 vertical video for TikTok/Reels/Shorts.
Layout:
  - TOP HALF (1080×960): Pixel art scene images with pan/zoom
  - SUBTITLE BAND (~80px): Word-synced subtitles at the center
  - BOTTOM HALF (1080×960): Looping avatar animation on pixel background
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

from .subtitle_renderer import create_subtitle_clips

# Layout constants
VIDEO_W = 1080
VIDEO_H = 1920
TOP_H = 960        # Top half height (scene images)
BOTTOM_H = 960     # Bottom half height (avatar)
SUBTITLE_H = 80    # Subtitle band height
FPS = 30

# Avatar asset path
AVATAR_PATH = Path(__file__).parent.parent / "assets" / "avatar" / "avatar_loop.mp4"


def _resize_image_to_top(img_path: str) -> ImageClip:
    """Load an image and resize/crop to fill 1080×960 (top half)."""
    img = Image.open(img_path)
    img_w, img_h = img.size

    # Target aspect ratio
    target_ratio = VIDEO_W / TOP_H  # 1.125
    img_ratio = img_w / img_h

    if img_ratio > target_ratio:
        # Image is wider — crop sides
        new_w = int(img_h * target_ratio)
        left = (img_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, img_h))
    else:
        # Image is taller — crop top/bottom
        new_h = int(img_w / target_ratio)
        top = (img_h - new_h) // 2
        img = img.crop((0, top, img_w, top + new_h))

    img = img.resize((VIDEO_W, TOP_H), Image.LANCZOS)

    # Save to temp file for moviepy
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(tmp.name, 'PNG')
    tmp.close()

    return ImageClip(tmp.name)


def _apply_zoom_effect(clip: ImageClip, zoom_in: bool = True) -> ImageClip:
    """Apply subtle Ken Burns zoom to a still image clip."""
    def zoom_transform(pic):
        h, w = pic.shape[:2]
        if zoom_in:
            crop_h = int(h * 0.04)
            crop_w = int(w * 0.04)
        else:
            crop_h = int(h * 0.01)
            crop_w = int(w * 0.01)

        cropped = pic[crop_h:h-crop_h, crop_w:w-crop_w]
        pil = Image.fromarray(cropped).resize((w, h), Image.LANCZOS)
        return np.array(pil)

    return clip.fl_image(zoom_transform)


def _calculate_scene_durations(total_duration: float, num_scenes: int) -> List[float]:
    """Calculate varied scene durations for natural pacing."""
    if num_scenes == 0:
        return []
    if num_scenes == 1:
        return [total_duration]

    base = total_duration / num_scenes
    weights = []
    for i in range(num_scenes):
        if i == 0:
            weights.append(1.2)     # Hook: longer
        elif i == num_scenes - 1:
            weights.append(1.15)    # Conclusion: longer
        elif i == num_scenes // 2:
            weights.append(0.9)     # Middle pivot: shorter
        else:
            weights.append(0.95 + (i % 2) * 0.1)

    total_weight = sum(weights)
    durations = [total_duration * (w / total_weight) for w in weights]
    return durations


def _prepare_avatar_bottom(avatar_path: str, total_duration: float) -> VideoFileClip:
    """
    Load the avatar video, crop/resize to 1080×960, and loop to match duration.

    Avatar input: 1292×720 (landscape).
    Target: 1080×960 (portrait half).

    Strategy: Add letterbox bars top+bottom to make it portrait,
    then scale to 1080×960.
    """
    avatar = VideoFileClip(avatar_path)
    av_w, av_h = avatar.size  # 1292, 720
    avatar_duration = avatar.duration  # ~6.11s

    # Target ratio for bottom half: 1080/960 = 1.125
    # Avatar ratio: 1292/720 = 1.794
    # Strategy: pad with black top+bottom to reach 1.125 ratio
    # New height needed: 1292 / 1.125 = 1148px → pad top (1148-720)/2 = 214px each side

    target_h_for_ratio = int(av_w / (VIDEO_W / BOTTOM_H))  # 1292 / 1.125 = 1149
    pad_top = (target_h_for_ratio - av_h) // 2
    pad_bottom = target_h_for_ratio - av_h - pad_top

    # Use moviepy margin to add padding
    avatar_padded = avatar.margin(top=pad_top, bottom=pad_bottom, color=(0, 0, 0))

    # Resize to target
    avatar_resized = avatar_padded.resize((VIDEO_W, BOTTOM_H))

    # Loop to fill total duration
    loops_needed = math.ceil(total_duration / avatar_duration)

    if loops_needed > 1:
        # Create looped clip by concatenating copies
        clips = [avatar_resized]
        for _ in range(loops_needed - 1):
            clips.append(avatar_resized.copy())
        avatar_looped = concatenate_videoclips(clips)
    else:
        avatar_looped = avatar_resized

    # Trim to exact duration
    avatar_final = avatar_looped.subclip(0, total_duration)

    return avatar_final


def build_split_video(
    audio_path: str,
    image_paths: List[str],
    avatar_path: str = None,
    output_path: str = None,
    script_text: str = None,
    word_timestamps: list = None,
) -> dict:
    """
    Build a split-screen video: scenes on top, avatar on bottom, subtitles in middle.

    Args:
        audio_path: Path to voiceover MP3
        image_paths: List of scene image paths (4-6 images)
        avatar_path: Path to avatar loop video (default: assets/avatar/avatar_loop.mp4)
        output_path: Output MP4 path
        script_text: Full narration text (for subtitle rendering)
        word_timestamps: Word timing data from subtitle_renderer.get_word_timestamps()

    Returns:
        dict with success status, output path, and metadata
    """
    # Validate inputs
    if not audio_path or not Path(audio_path).exists():
        return {"success": False, "error": f"Audio not found: {audio_path}"}
    if not image_paths:
        return {"success": False, "error": "No image paths provided"}

    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
        return {"success": False, "error": f"Missing images: {missing}"}

    # Resolve avatar path
    if not avatar_path:
        avatar_path = str(AVATAR_PATH)
    if not Path(avatar_path).exists():
        return {"success": False, "error": f"Avatar video not found: {avatar_path}"}

    try:
        # Load audio to get total duration
        audio = AudioFileClip(audio_path)
        total_dur = audio.duration
        print(f"  [SPLIT] Audio duration: {total_dur:.1f}s")

        # ── TOP HALF: Scene images with pan/zoom ──
        num_scenes = len(image_paths)
        scene_durations = _calculate_scene_durations(total_dur, num_scenes)

        scene_clips = []
        for idx, img_path in enumerate(image_paths):
            dur = scene_durations[idx]
            clip = _resize_image_to_top(img_path).set_duration(dur)

            # Alternate zoom in/out for visual variety
            zoom_in = (idx % 2 == 0)
            clip = _apply_zoom_effect(clip, zoom_in=zoom_in)

            scene_clips.append(clip)

        # Concatenate scenes → top half video
        top_half = concatenate_videoclips(scene_clips, method="compose")
        # Ensure correct size
        if top_half.size != [VIDEO_W, TOP_H]:
            top_half = top_half.resize((VIDEO_W, TOP_H))

        print(f"  [SPLIT] Top half: {num_scenes} scenes, {VIDEO_W}×{TOP_H}")

        # ── BOTTOM HALF: Avatar loop ──
        bottom_half = _prepare_avatar_bottom(avatar_path, total_dur)
        print(f"  [SPLIT] Bottom half: avatar looped to {total_dur:.1f}s, {VIDEO_W}×{BOTTOM_H}")

        # ── COMPOSITE: Stack top and bottom ──
        # Position top half at y=0, bottom half at y=TOP_H
        bottom_half = bottom_half.set_position((0, TOP_H))
        top_half = top_half.set_position((0, 0))

        layers = [top_half, bottom_half]

        # ── SUBTITLES: At the split line ──
        subtitle_clips = []
        if word_timestamps and len(word_timestamps) > 0:
            band_y = TOP_H - SUBTITLE_H // 2  # Center on split line
            subtitle_clips = create_subtitle_clips(
                script_text=script_text or "",
                word_timestamps=word_timestamps,
                video_width=VIDEO_W,
                video_height=VIDEO_H,
                band_y_position=band_y,
            )
            layers.extend(subtitle_clips)
            print(f"  [SPLIT] Subtitles: {len(subtitle_clips)} clips at y={band_y}")
        else:
            print(f"  [SPLIT] No subtitle data — skipping subtitles")

        # Composite all layers
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

        # Verify output
        out_path = Path(output_path)
        if not out_path.exists():
            return {"success": False, "error": "Output file not created"}

        file_size = out_path.stat().st_size

        # Cleanup
        final.close()
        audio.close()
        top_half.close()
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
            "effects_applied": ["split_screen", "camera_movements", "avatar_loop", "subtitles"],
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Split video build failed: {str(e)}"}