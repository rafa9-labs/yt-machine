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
import random
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from PIL import Image

try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, ImageClip,
        CompositeVideoClip, CompositeAudioClip, concatenate_videoclips, vfx
    )
except ImportError:
    from moviepy import (
        VideoFileClip, AudioFileClip, ImageClip,
        CompositeVideoClip, CompositeAudioClip, concatenate_videoclips
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

# Background music path
MUSIC_PATH = Path(__file__).parent.parent / "assets" / "avatar" / "music" / "news-yt.mp3"


def _find_ffmpeg() -> str:
    """Find ffmpeg executable — check imageio_ffmpeg bundled binary first, then system PATH."""
    # 1. imageio_ffmpeg bundled binary (used by moviepy)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, Exception):
        pass

    # 2. System PATH
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    return None


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
    20% zoom — noticeable cinematic movement that reveals the full scene.
    """
    def zoom_out_transform(get_frame, t):
        frame = get_frame(t)
        h, w = frame.shape[:2]

        # Start at 20% crop (cinematic), end at 0% crop (full frame)
        progress = t / duration if duration > 0 else 1.0
        crop_pct = 0.20 * (1.0 - progress)

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

        # Pan range: scroll ~20% of frame height (cinematic movement)
        pan_range = int(h * 0.20)
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


def _apply_breathing_pulse(clip: ImageClip, duration: float) -> ImageClip:
    """
    Subtle 1-2% scale oscillation (sin wave) — gives images a 'living' feel.
    Oscillates between 1.0x and 1.02x centered on the image.
    Completely contained within image bounds — no overflow.
    """
    def breathing_transform(get_frame, t):
        frame = get_frame(t)
        h, w = frame.shape[:2]
        scale = 1.0 + 0.02 * math.sin(2 * math.pi * 0.5 * t)
        new_w = int(w / scale)
        new_h = int(h / scale)
        crop_x = (w - new_w) // 2
        crop_y = (h - new_h) // 2
        cropped = frame[crop_y:crop_y + new_h, crop_x:crop_x + new_w]
        pil_img = Image.fromarray(cropped).resize((w, h), Image.LANCZOS)
        return np.array(pil_img)

    return clip.fl(breathing_transform)


def _apply_pixel_shimmer(clip: ImageClip, duration: float, density: float = 0.002) -> ImageClip:
    """
    Random 1-2px 'sparkle' overlay at low opacity — CRT/retro pixel-art effect.
    Randomly brightens a tiny fraction of pixels each frame for a subtle alive feeling.
    """
    rng = random.Random(42)

    def shimmer_transform(get_frame, t):
        frame = get_frame(t).copy()
        h, w = frame.shape[:2]
        n_pixels = int(h * w * density)
        for _ in range(n_pixels):
            y = rng.randint(0, h - 1)
            x = rng.randint(0, w - 1)
            boost = rng.randint(15, 40)
            frame[y, x] = np.clip(frame[y, x].astype(np.int16) + boost, 0, 255).astype(np.uint8)
        return frame

    return clip.fl(shimmer_transform)


def _apply_vignette_breath(clip: ImageClip, duration: float) -> ImageClip:
    """
    Subtle brightness pulse at edges — simulates ambient lighting breathing.
    Oscillates edge darkening at 0.3Hz, barely perceptible but adds depth.
    """
    def vignette_transform(get_frame, t):
        frame = get_frame(t).copy()
        h, w = frame.shape[:2]
        pulse = 0.85 + 0.15 * math.sin(2 * math.pi * 0.3 * t)
        edge_h = h // 10
        edge_w = w // 10
        for i in range(edge_h):
            factor = pulse * (i / edge_h)
            frame[i, :] = (frame[i, :].astype(np.float32) * factor).astype(np.uint8)
            frame[h - 1 - i, :] = (frame[h - 1 - i, :].astype(np.float32) * factor).astype(np.uint8)
        for i in range(edge_w):
            factor = pulse * (i / edge_w)
            frame[:, i] = (frame[:, i].astype(np.float32) * factor).astype(np.uint8)
            frame[:, w - 1 - i] = (frame[:, w - 1 - i].astype(np.float32) * factor).astype(np.uint8)
        return frame

    return clip.fl(vignette_transform)


def _apply_scene_effect(clip: ImageClip, scene_idx: int, duration: float) -> ImageClip:
    """
    Apply layered effects to scene clips:
    - Motion effect (zoom/pan) varies by scene index
    - Subtle ambient effect (breathing/shimmer/vignette) layered on top
    """
    motion_effects = [_apply_zoom_out, _apply_pan_top_to_bottom]
    ambient_effects = [_apply_breathing_pulse, _apply_pixel_shimmer, _apply_vignette_breath]

    motion_fn = motion_effects[scene_idx % len(motion_effects)]
    ambient_fn = ambient_effects[scene_idx % len(ambient_effects)]

    clip = motion_fn(clip, duration)
    clip = ambient_fn(clip, duration)
    return clip


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
    Trims 0.05s from the end to avoid moviepy's known MP4 duration rounding bug
    (ffmpeg reports slightly longer duration than actual readable frames).
    """
    avatar = VideoFileClip(avatar_path)
    av_w, av_h = avatar.size
    avatar_duration = avatar.duration - 0.05  # Safety trim to prevent last-frame read error

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
    avatar_trimmed = avatar_resized.subclip(0, avatar_duration)

    loops_needed = math.ceil(total_duration / avatar_duration)

    if loops_needed > 1:
        clips = [avatar_trimmed]
        for _ in range(loops_needed - 1):
            clips.append(avatar_trimmed.copy())
        avatar_looped = concatenate_videoclips(clips)
    else:
        avatar_looped = avatar_trimmed

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


def _create_hook_card(hook_text: str, width: int, height: int, duration: float = 1.5) -> Optional[ImageClip]:
    """
    Create a bold 1.5-second text-card overlay — a pattern interrupt for the first seconds.
    Uses PIL to render text on a solid dark background, then wraps as a MoviePy clip.
    
    The Orientation Response (Sokolov, 1963): the brain reflexively attends to
    novel visual stimuli. A bold text card before the greeting creates a 'wait, what?'
    moment that hooks the viewer before they swipe away.
    """
    if not hook_text or len(hook_text.strip()) < 5:
        return None
    
    try:
        from PIL import ImageDraw, ImageFont
        
        # Dark navy background matching the video aesthetic
        img = Image.new('RGB', (width, height), (10, 5, 25))
        draw = ImageDraw.Draw(img)
        
        # Try to load a bold font, fall back to default
        font_size = 52
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Word-wrap the text to fit width
        max_width = width - 120  # 60px padding each side
        words = hook_text.strip().upper().split()
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] > max_width and current_line:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)
        
        # Limit to 3 lines max
        lines = lines[:3]
        
        # Calculate text block height
        line_height = font_size + 12
        total_text_height = len(lines) * line_height
        y_start = (height - total_text_height) // 2
        
        # Draw each line centered
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            y = y_start + i * line_height
            
            # White text with subtle shadow
            draw.text((x + 2, y + 2), line, fill=(30, 20, 50), font=font)
            draw.text((x, y), line, fill=(255, 255, 255), font=font)
        
        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(tmp.name, 'PNG')
        tmp.close()
        
        clip = ImageClip(tmp.name).set_duration(duration).set_start(0)
        # Fade out the hook card in the last 0.3s
        clip = clip.crossfadeout(0.3)
        
        print(f"  [SPLIT] Hook card: \"{hook_text[:50]}...\" ({duration:.1f}s)")
        return clip
    
    except Exception as e:
        print(f"  [SPLIT] ⚠️ Hook card creation failed: {e}")
        return None


def build_split_video(
    audio_path: str,
    image_paths: List[str],
    avatar_path: str = None,
    output_path: str = None,
    script_text: str = None,
    word_timestamps: list = None,
    hook_text: str = None,
    scene_timestamps: List[Dict] = None,
    hook_card_text: str = None,
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

        # ── HOOK CARD: 1.5s pattern-interrupt text card at video start ──
        if hook_card_text:
            hook_clip = _create_hook_card(hook_card_text, VIDEO_W, VIDEO_H, duration=1.5)
            if hook_clip:
                layers.append(hook_clip)
                print(f"  [SPLIT] Hook card added (1.5s pattern interrupt)")

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

        # ── BACKGROUND MUSIC: news-yt.mp3 (broadcast standard -20dB) ──
        # Broadcast standard: music under speech = -20dB to -25dB (EBU R128 / ATSC A/85)
        # -20dB ≈ 0.10 linear. Anything louder increases cognitive load (Cocktail Party Effect).
        MUSIC_DIM_DURATION = 10.0  # seconds — music dims from 10% → 5% in last 10s
        VIDEO_FADE_DURATION = 0.8  # seconds — final video fade to black
        MUSIC_BASE_VOLUME = 0.10  # -20dB broadcast background standard (~10% of voice)
        mixed_audio = audio  # Default: voice only

        if MUSIC_PATH.exists():
            try:
                music = AudioFileClip(str(MUSIC_PATH))
                # Trim music to video length
                if music.duration > total_dur:
                    music = music.subclip(0, total_dur)
                
                # Set music to broadcast background level (-20dB ≈ 0.10)
                music_loud = music.volumex(MUSIC_BASE_VOLUME)
                
                # Apply 10-second fadeout: dims from 10% → 5%
                music_loud = music_loud.audio_fadeout(MUSIC_DIM_DURATION)
                
                # Mix voice (100%) + music (10% → 5%)
                mixed_audio = CompositeAudioClip([audio, music_loud.set_duration(total_dur)])
                music.close()
                print(f"  [SPLIT] ♪ news-yt.mp3 at {MUSIC_BASE_VOLUME:.0%} (-20dB), {MUSIC_DIM_DURATION:.0f}s dim to ~5%")
            except Exception as music_err:
                print(f"  [SPLIT] ⚠️ Music mixing failed, using voice only: {music_err}")
                mixed_audio = audio
        else:
            print(f"  [SPLIT] No news-yt.mp3 found, skipping background music")

        # ── PRE-RENDER AUDIO TO FILE (avoids moviepy ffmpeg subprocess crash) ──
        # moviepy's ffmpeg audio reader subprocess dies on Windows during
        # CompositeAudioClip rendering. Solution: use ffmpeg CLI directly to mix,
        # then open the simple pre-mixed file for the video export.
        audio_tmp = tempfile.NamedTemporaryFile(suffix='_mixed.wav', delete=False)
        audio_tmp_path = audio_tmp.name
        audio_tmp.close()

        # Close moviepy audio objects to release their ffmpeg processes
        try:
            mixed_audio.close()
        except:
            pass
        try:
            audio.close()
        except:
            pass
        try:
            if 'music' in dir():
                music.close()
        except:
            pass

        pre_rendered_ok = False
        if MUSIC_PATH.exists():
            try:
                # Use ffmpeg CLI directly to mix voice + music with 10s dim
                ffmpeg_exe = _find_ffmpeg()
                if ffmpeg_exe:
                    import subprocess as sp
                    dim_start = max(0, total_dur - MUSIC_DIM_DURATION)
                    # Volume: base 10% (-20dB broadcast standard), dims to 0% over last 10s via afade
                    # Uses simple afade instead of complex if() expression — Windows ffmpeg 7.1
                    # doesn't support the \, comma escaping needed by if() in filter_complex.
                    mix_cmd = [
                        ffmpeg_exe, '-y',
                        '-i', audio_path,                           # Voice (100%)
                        '-i', str(MUSIC_PATH),                      # Music (10% → fade out)
                        '-filter_complex',
                        # Music: trim to video length, set 10% volume,
                        # highpass=80 removes sub-bass rumble,
                        # 1s safety fade-in as belt-and-suspenders (source file already pre-processed).
                        # Only fade-out over last 10s for smooth ending.
                        f'[1:a]atrim=0:{total_dur:.3f},asetpts=PTS-STARTPTS,'
                        f'highpass=f=80,'
                        f'afade=t=in:st=0:d=1,'
                        f'volume=0.10,afade=t=out:st={dim_start:.3f}:d={MUSIC_DIM_DURATION:.1f}[music];'
                        # Mix voice (100%) + dimmed music (10%→0%), no normalization
                        # normalize=0 keeps voice at full volume (amix default divides by input count)
                        f'[0:a][music]amix=inputs=2:duration=longest:normalize=0,'
                        f'afade=t=out:st={total_dur - VIDEO_FADE_DURATION:.3f}:d={VIDEO_FADE_DURATION}[out]',
                        '-map', '[out]',
                        '-ar', '44100', '-ac', '2',
                        audio_tmp_path
                    ]
                    result = sp.run(mix_cmd, capture_output=True, timeout=60)
                    if result.returncode == 0 and Path(audio_tmp_path).exists():
                        pre_rendered_audio = AudioFileClip(audio_tmp_path)
                        final = final.set_audio(pre_rendered_audio)
                        pre_rendered_ok = True
                        print(f"  [SPLIT] Pre-mixed audio via ffmpeg CLI ({Path(audio_tmp_path).stat().st_size/1024:.0f}KB)")
                    else:
                        print(f"  [SPLIT] ffmpeg mix failed (rc={result.returncode}): {result.stderr[:200]}")
            except Exception as e:
                print(f"  [SPLIT] ffmpeg CLI mix error: {e}")

        if not pre_rendered_ok:
            # Fallback: just use voice audio directly (no music)
            try:
                voice_audio = AudioFileClip(audio_path)
                voice_audio = voice_audio.subclip(0, total_dur)
                final = final.set_audio(voice_audio)
                print(f"  [SPLIT] Using voice-only audio (no music)")
            except Exception as e2:
                print(f"  [SPLIT] ⚠️ Even voice-only audio failed: {e2}")
                # Last resort: export without audio, then mux with ffmpeg
                final = final.set_audio(None)

        final = final.fadeout(VIDEO_FADE_DURATION)
        print(f"  [SPLIT] Applied {VIDEO_FADE_DURATION}s video fade-out")

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

        # ── POST-EXPORT REMUX: Fix Windows MP4 container corruption ──
        # moviepy's two-pass audio mux on Windows can produce a broken MP4 container
        # (0xC00D36C4 in Windows Media Player). Remux with ffmpeg fixes it.
        try:
            ffmpeg_exe = _find_ffmpeg()
            if ffmpeg_exe:
                remux_path = out_path.with_suffix('.remux.mp4')
                remux_cmd = [
                    ffmpeg_exe, '-y',
                    '-i', str(out_path),
                    '-c', 'copy',                    # No re-encoding — just fix container
                    '-movflags', '+faststart',        # Move moov atom to start
                    '-map', '0:v',                   # All video streams
                    '-map', '0:a',                   # All audio streams
                    str(remux_path)
                ]
                import subprocess as sp
                remux_result = sp.run(remux_cmd, capture_output=True, timeout=60)
                if remux_result.returncode == 0 and remux_path.exists():
                    # Replace original with remuxed version
                    remux_path.replace(out_path)
                    print(f"  [SPLIT] ✅ Remuxed MP4 container (Windows compatibility fix)")
                else:
                    print(f"  [SPLIT] ⚠️ Remux failed (rc={remux_result.returncode}), keeping original")
                    try:
                        remux_path.unlink()
                    except:
                        pass
        except Exception as e:
            print(f"  [SPLIT] ⚠️ Remux error: {e}")

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