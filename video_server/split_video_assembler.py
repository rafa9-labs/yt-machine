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
        # Legacy image — aspect-ratio-safe resize (no stretching!)
        # Use thumbnail to fit within visible area, then center-paste
        target_ratio = visible_w / visible_h
        img_ratio = img_w / img_h

        if img_ratio > target_ratio:
            # Image is wider — fit to width, may letterbox top/bottom
            new_w = visible_w
            new_h = int(visible_w / img_ratio)
        else:
            # Image is taller — fit to height, may pillarbox left/right
            new_h = visible_h
            new_w = int(visible_h * img_ratio)

        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        # Center-paste onto visible-area-sized canvas (bg color fills gaps)
        canvas_top = Image.new('RGB', (visible_w, visible_h), (10, 5, 25))
        paste_x = (visible_w - new_w) // 2
        paste_y = (visible_h - new_h) // 2
        canvas_top.paste(img_resized, (paste_x, paste_y))
        img = canvas_top
        print(f"  [SPLIT] Legacy image {img_w}×{img_h} → fit to {new_w}×{new_h} (no stretch)")

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


def _create_solid_color_image(width: int, height: int, color: tuple) -> str:
    """
    Create a solid color image and return its temp file path.
    Used as a background layer to prevent black frames in CompositeVideoClip.
    """
    img = Image.new('RGB', (width, height), color)
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(tmp.name, 'PNG')
    tmp.close()
    return tmp.name


def build_split_video(
    audio_path: str,
    image_paths: List[str],
    avatar_path: str = None,
    output_path: str = None,
    script_text: str = None,
    word_timestamps: list = None,
    hook_text: str = None,
    scene_timestamps: List[Dict] = None,
) -> dict:
    """
    Build a video: scene background top 60%, avatar bottom 40%, title at top,
    karaoke subtitles near bottom of scene area.

    Args:
        audio_path: Path to voiceover MP3
        image_paths: List of scene image paths (6 images for 3 stories)
        avatar_path: Path to avatar loop video
        output_path: Output MP4 path
        script_text: Full narration text (for subtitle word alignment)
        word_timestamps: Word timing data from whisper
        hook_text: Hook text for title overlay at top
        scene_timestamps: Optional list of {"start": float, "end": float} per image.
                          If provided, images switch at exact content-synced times.
                          If None, falls back to weighted duration splitting.

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

    # Validate all images can be opened (catches corrupt/truncated files)
    for img_p in image_paths:
        try:
            test_img = Image.open(img_p)
            test_img.verify()
            test_img.close()
        except Exception as e:
            return {"success": False, "error": f"Corrupt image {img_p}: {e}"}

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

        if scene_timestamps and len(scene_timestamps) == num_scenes:
            # ── SYNCED MODE: Image switches match narration content ──
            print(f"  [SPLIT] Using SYNCED scene timestamps (content-driven)")
            for i, ts in enumerate(scene_timestamps):
                print(f"    Image {i}: {ts['start']:.2f}s → {ts['end']:.2f}s ({ts['end']-ts['start']:.2f}s)")

            scene_clips = []
            for idx, img_path in enumerate(image_paths):
                ts = scene_timestamps[idx]
                start = ts['start']
                end = ts['end']
                dur = end - start
                if dur <= 0:
                    dur = 0.5  # minimum duration
                clip = _resize_image_fullscreen(img_path).set_duration(dur).set_start(start)
                clip = _apply_scene_effect(clip, idx, dur)
                scene_clips.append(clip)

            # Ensure last image persists to the very end of the video
            if scene_clips:
                last_ts = scene_timestamps[-1]
                needed_dur = total_dur - last_ts['start']
                if needed_dur > 0:
                    scene_clips[-1] = (
                        _resize_image_fullscreen(image_paths[-1])
                        .set_duration(needed_dur)
                        .set_start(last_ts['start'])
                    )
                    scene_clips[-1] = _apply_scene_effect(scene_clips[-1], len(image_paths) - 1, needed_dur)
                    print(f"  [SPLIT] Last image extended: {last_ts['start']:.2f}s → {total_dur:.2f}s ({needed_dur:.2f}s)")

            # Solid background color layer — safety net for any sub-frame gaps
            bg_color = (10, 5, 25)  # Dark navy, matches canvas fill
            bg_layer = ImageClip(
                _create_solid_color_image(VIDEO_W, VIDEO_H, bg_color)
            ).set_duration(total_dur)

            # Use CompositeVideoClip: solid bg + scene clips on top
            background = CompositeVideoClip([bg_layer] + scene_clips, size=(VIDEO_W, VIDEO_H))
            background = background.set_duration(total_dur)
            print(f"  [SPLIT] Background: {num_scenes} SYNCED scenes (content-timed) + solid bg layer")
        else:
            # ── FALLBACK: Weighted duration splitting (legacy) ──
            if scene_timestamps:
                print(f"  [SPLIT] ⚠️ scene_timestamps length mismatch ({len(scene_timestamps)} vs {num_scenes} images), using fallback")
            scene_durations = _calculate_scene_durations(total_dur, num_scenes)

            scene_clips = []
            for idx, img_path in enumerate(image_paths):
                dur = scene_durations[idx]
                clip = _resize_image_fullscreen(img_path).set_duration(dur)
                clip = _apply_scene_effect(clip, idx, dur)
                scene_clips.append(clip)

            background = concatenate_videoclips(scene_clips, method="compose")
            print(f"  [SPLIT] Background: {num_scenes} scenes (weighted fallback)")

        if background.size != [VIDEO_W, VIDEO_H]:
            background = background.resize((VIDEO_W, VIDEO_H))

        # ── BOTTOM AREA: Avatar loop (40% of screen) ──
        bottom_half = _prepare_avatar_bottom(avatar_path, total_dur)
        bottom_half = bottom_half.set_position((0, TOP_H))
        print(f"  [SPLIT] Avatar: {BOTTOM_H}px at y={TOP_H}")

        # ── COMPOSITE: Stack all layers ──
        layers = [background, bottom_half]

        # ── TITLE OVERLAY: Persistent headline at top (entire video) ──
        if hook_text:
            title_clip = create_title_clip(
                hook_text, VIDEO_W, VIDEO_H,
                duration=total_dur  # Persist entire video
            )
            if title_clip:
                layers.append(title_clip)
                print(f"  [SPLIT] Title overlay (persistent): \"{hook_text[:60]}...\"")

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
            ffmpeg_params=[
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',          # Max player compatibility (fixes 0xC00D36C4)
                '-profile:v', 'high',            # H.264 High profile (broad support)
                '-level', '4.0',                 # Level 4.0 (1080p30 safe)
                '-crf', '20',                    # Quality (lower=better, 18-28 range)
                '-bf', '2',                      # B-frames for efficiency
            ],
            preset='medium',
            threads=4,
            verbose=False,
            logger=None,
        )

        out_path = Path(output_path)
        if not out_path.exists():
            return {"success": False, "error": "Output file not created"}

        file_size = out_path.stat().st_size
        print(f"  [SPLIT] Export complete: {file_size / (1024*1024):.1f}MB")

        # Cleanup moviepy objects
        final.close()
        audio.close()
        background.close()
        bottom_half.close()
        for c in scene_clips:
            c.close()

        # Cleanup MoviePy temp files (audio mux leftovers)
        try:
            import glob
            temp_pattern = str(out_path).replace('.mp4', 'TEMP_MPY_wvf_snd.mp4')
            for tmp_file in glob.glob(temp_pattern):
                os.remove(tmp_file)
                print(f"  [SPLIT] Cleaned temp file: {tmp_file}")
        except Exception:
            pass

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