"""
Split-Screen Video Assembler — Scene images top 60% + avatar bottom 40%.
Composites a 1080x1920 vertical video for TikTok/Reels/Shorts.

PERFORMANCE: Scene zoom/pan effects rendered via ffmpeg-native zoompan filter.
Moviepy handles only lightweight compositing (layering, subtitles, avatar).
Expected render time: ~1-2 min (down from 15-20 min with frame-by-frame Python).

Layout (Option A — 60/40 split):
  - SCENE AREA (1080x1152): Full scene images with ffmpeg zoom/pan (top 60%)
  - TITLE OVERLAY: Hook text at the top with fade-in
  - SUBTITLES: Karaoke-style outlined text near the bottom of the scene area
  - AVATAR AREA (1080x768): Looping avatar animation (bottom 40%)
"""

import os
import math
import subprocess
import tempfile
import time as _time
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

from .subtitle_renderer import create_subtitle_clips, create_title_clip, generate_ass_subtitles

# Windows fix: moviepy's FFMPEG_VideoReader.__del__ calls proc.terminate()
# on already-closed handles, raising OSError [WinError 6]. Suppress it.
try:
    from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
    _orig_close = FFMPEG_VideoReader.close
    def _safe_close(self):
        try:
            _orig_close(self)
        except (OSError, AttributeError):
            pass
    FFMPEG_VideoReader.close = _safe_close
    FFMPEG_VideoReader.__del__ = lambda self: None
except (ImportError, AttributeError):
    pass

VIDEO_W = 1080
VIDEO_H = 1920
TOP_H = 1152
BOTTOM_H = 768
FPS = 30

SCENE_ZOOM_PROFILES = {
    0: {'name': 'HOOK',      'zoom_start': 1.08, 'zoom_end': 1.02, 'pan_x': 0.0},
    1: {'name': 'MECHANISM', 'zoom_start': 1.06, 'zoom_end': 1.01, 'pan_x': 0.0},
    2: {'name': 'TRUTH',     'zoom_start': 1.01, 'zoom_end': 1.07, 'pan_x': 0.0},
    3: {'name': 'FALLOUT',   'zoom_start': 1.05, 'zoom_end': 1.01, 'pan_x': 0.0},
}


def _get_scene_profile(scene_idx: int) -> dict:
    pos = scene_idx % 4
    return SCENE_ZOOM_PROFILES.get(pos, SCENE_ZOOM_PROFILES[0])


AVATAR_PATH = Path(__file__).parent.parent.parent / "assets" / "avatar" / "avatar_loop.mp4"
MUSIC_PATH = Path(__file__).parent.parent.parent / "assets" / "avatar" / "music" / "news_background_sound.mp3"


def _find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, Exception):
        pass
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    return None


def _validate_mp4(path: str, ffmpeg_exe: str = None) -> dict:
    """
    Validate an MP4 file has both video and audio streams with a valid moov atom.
    Returns {'valid': bool, 'video_codec': str|None, 'audio_codec': str|None, 'error': str|None}
    """
    if not ffmpeg_exe:
        ffmpeg_exe = _find_ffmpeg()
    if not ffmpeg_exe:
        return {'valid': False, 'video_codec': None, 'audio_codec': None, 'error': 'ffmpeg not found'}
    if not Path(path).exists():
        return {'valid': False, 'video_codec': None, 'audio_codec': None, 'error': 'file not found'}
    try:
        result = subprocess.run(
            [ffmpeg_exe, '-i', str(path)],
            capture_output=True, text=True, timeout=30
        )
        combined = result.stderr + result.stdout
        if 'moov atom not found' in combined or 'Invalid data' in combined.lower():
            return {'valid': False, 'video_codec': None, 'audio_codec': None,
                    'error': 'moov atom not found / invalid data'}
        video_codec = None
        audio_codec = None
        for line in combined.split('\n'):
            stripped = line.strip()
            if 'Video:' in stripped:
                after_video = stripped.split('Video:')[1].strip()
                codec_name = after_video.split(',')[0].strip().split()[0]
                video_codec = codec_name
            if 'Audio:' in stripped:
                after_audio = stripped.split('Audio:')[1].strip()
                codec_name = after_audio.split(',')[0].strip().split()[0]
                audio_codec = codec_name
        if not video_codec:
            return {'valid': False, 'video_codec': None, 'audio_codec': audio_codec,
                    'error': f'no video stream found (stderr had {len(result.stderr)} chars)'}
        return {'valid': True, 'video_codec': video_codec, 'audio_codec': audio_codec, 'error': None}
    except Exception as e:
        return {'valid': False, 'video_codec': None, 'audio_codec': None, 'error': str(e)}


def _adaptive_crf(total_dur: float, base_crf: int = 20) -> int:
    if total_dur <= 90:
        return base_crf
    elif total_dur <= 120:
        return base_crf + 2
    else:
        return base_crf + 4

def _render_avatar_ffmpeg(
    avatar_path: str,
    total_duration: float,
    output_path: str,
    target_w: int = VIDEO_W,
    target_h: int = BOTTOM_H,
) -> bool:
    """
    Pre-render the looping avatar as a standalone MP4 via ffmpeg CLI.
    Scales and crops to target dimensions, then loops to fill total_duration.
    """
    ffmpeg_exe = _find_ffmpeg()
    if not ffmpeg_exe:
        return False

    cmd = [
        ffmpeg_exe, '-y',
        '-stream_loop', '-1',
        '-i', avatar_path,
        '-t', f'{total_duration:.3f}',
        '-vf', f'scale=-2:{target_h},crop={target_w}:{target_h}:(iw-{target_w})/2:0',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '22',
        '-r', str(FPS),
        '-an',
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
    return result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 1000


def _render_scene_opencv(
    img_path: str,
    duration: float,
    scene_idx: int,
    output_path: str,
    width: int = VIDEO_W,
    height: int = TOP_H,
) -> bool:
    """
    Render smooth zoom using OpenCV — frame-by-frame with ease-out-cubic easing.
    Falls back to _render_scene_ffmpeg if OpenCV is unavailable.

    Scene-type-aware zoom profiles:
      HOOK (idx%4==0):      gentle zoom-in  (1.08 → 1.02)
      MECHANISM (idx%4==1): gentle zoom-out  (1.06 → 1.01)
      TRUTH (idx%4==2):     gentle zoom-in   (1.01 → 1.07)
      FALLOUT (idx%4==3):   gentle zoom-out  (1.05 → 1.01)
    
    All zooms are center-based (pan_x=0) for smooth, cinematic Ken Burns feel.
    Uses ease-out-cubic easing for natural deceleration.
    """
    try:
        import cv2
    except ImportError:
        return _render_scene_ffmpeg_fallback(img_path, duration, scene_idx, output_path, width, height)

    img = cv2.imread(img_path)
    if img is None:
        return False

    img_h, img_w = img.shape[:2]
    target_w = width
    target_h = height

    scale_w = target_w / img_w
    scale_h = target_h / img_h
    scale = max(scale_w, scale_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    new_w += new_w % 2
    new_h += new_h % 2

    img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    crop_x = max(0, (new_w - target_w) // 2)
    crop_y = max(0, (new_h - target_h) // 2)

    total_frames = max(int(duration * FPS), 2)

    profile = _get_scene_profile(scene_idx)
    zoom_start = profile['zoom_start']
    zoom_end = profile['zoom_end']
    pan_x_factor = profile['pan_x']
    scene_name = profile['name']

    def ease_out_cubic(t):
        t = max(0.0, min(1.0, t))
        return 1.0 - (1.0 - t) ** 3

    tmp_path = output_path.replace('.mp4', '_raw.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_path, fourcc, FPS, (target_w, target_h))
    if not out.isOpened():
        return False

    for frame_num in range(total_frames):
        progress = frame_num / max(total_frames - 1, 1)
        eased = ease_out_cubic(progress)

        zoom = zoom_start + (zoom_end - zoom_start) * eased

        crop_w = int(target_w / zoom)
        crop_h = int(target_h / zoom)

        pan_offset_x = int(crop_w * pan_x_factor * eased)
        cx = max(0, min((new_w - crop_w) // 2 + pan_offset_x, new_w - crop_w))
        cy = max(0, min((new_h - crop_h) // 2, new_h - crop_h))

        cropped = img_scaled[cy:cy + crop_h, cx:cx + crop_w]
        frame = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        out.write(frame)

    out.release()

    ffmpeg_exe = _find_ffmpeg()
    if ffmpeg_exe:
        cmd = [
            ffmpeg_exe, '-y',
            '-i', tmp_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '18',
            '-vf', 'vignette=angle=0.2:mode=forward',
            '-r', str(FPS),
            '-movflags', '+faststart',
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=180, text=True)
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass
        return result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 1000
    else:
        import shutil
        shutil.move(tmp_path, output_path)
        return Path(output_path).exists() and Path(output_path).stat().st_size > 1000


def _render_scene_ffmpeg_fallback(
    img_path: str,
    duration: float,
    scene_idx: int,
    output_path: str,
    width: int = VIDEO_W,
    height: int = TOP_H,
) -> bool:
    """
    Fallback: render scene using ffmpeg zoompan with ease-out-cubic easing.
    Used when OpenCV is unavailable.
    Scene-type-aware zoom profiles (same as OpenCV path).
    """
    ffmpeg_exe = _find_ffmpeg()
    if not ffmpeg_exe:
        return False

    img = Image.open(img_path)
    img_w, img_h = img.size
    img.close()

    target_w = width
    target_h = height

    scale_w = target_w / img_w
    scale_h = target_h / img_h
    scale = max(scale_w, scale_h)
    scaled_w = int(img_w * scale)
    scaled_h = int(img_h * scale)
    if scaled_w % 2 == 1:
        scaled_w += 1
    if scaled_h % 2 == 1:
        scaled_h += 1

    crop_x = max(0, (scaled_w - target_w) // 2)
    crop_y = max(0, (scaled_h - target_h) // 2)

    profile = _get_scene_profile(scene_idx)
    zoom_start = profile['zoom_start']
    zoom_end = profile['zoom_end']
    pan_x_factor = profile['pan_x']
    scene_name = profile['name']

    total_frames = int(duration * FPS)
    if total_frames < 2:
        total_frames = 2

    safe_total = max(total_frames - 1, 1)
    zoom_range = zoom_end - zoom_start

    # Ease-out cubic: 1 - (1-t)^3 where t = (on-1)/(total_frames-1)
    # zoom = zoom_start + zoom_range * (1 - (1-t)^3)
    # In ffmpeg expr: 1-pow(1-(on-1)/N, 3)
    if zoom_range >= 0:
        zoom_expr = f"{zoom_start}+{zoom_range}*(1-pow(1-(on-1)/{safe_total},3))"
    else:
        zoom_expr = f"{zoom_start}-{abs(zoom_range)}*(1-pow(1-(on-1)/{safe_total},3))"

    if abs(pan_x_factor) > 0.001:
        pan_expr = f"iw/2-(iw/zoom/2)+{pan_x_factor}*ow*(on-1)/{safe_total}"
        x_expr = pan_expr
    else:
        x_expr = "iw/2-(iw/zoom/2)"

    y_expr = "ih/2-(ih/zoom/2)"

    vf = (
        f"[0:v]scale={scaled_w}:{scaled_h},"
        f"crop={target_w}:{target_h}:{crop_x}:{crop_y},"
        f"zoompan=z='{zoom_expr}'"
        f":x='{x_expr}'"
        f":y='{y_expr}'"
        f":d={total_frames}"
        f":s={target_w}x{target_h}"
        f":fps={FPS},"
        f"vignette=angle=0.2:mode=forward,"
        f"setpts=PTS-STARTPTS[v]"
    )

    cmd = [
        ffmpeg_exe, '-y',
        '-loop', '1',
        '-i', img_path,
        '-vf', vf,
        '-t', f'{duration:.3f}',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', '18',
        '-r', str(FPS),
        '-movflags', '+faststart',
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=180, text=True)
    return result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 1000


def _resize_image_fullscreen(img_path: str) -> ImageClip:
    img = Image.open(img_path)
    img_w, img_h = img.size

    visible_w = VIDEO_W
    visible_h = TOP_H

    if abs(img_w - 1088) <= 8 and abs(img_h - 1152) <= 8:
        crop_x = (img_w - visible_w) // 2
        cropped = img.crop((crop_x, 0, crop_x + visible_w, visible_h))
    else:
        scale_w = visible_w / img_w
        scale_h = visible_h / img_h
        scale = max(scale_w, scale_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        crop_x = max(0, (new_w - visible_w) // 2)
        crop_y = max(0, (new_h - visible_h) // 2)
        cropped = img_resized.crop((crop_x, crop_y, crop_x + visible_w, crop_y + visible_h))

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    cropped.save(tmp.name, 'PNG')
    tmp.close()
    img.close()
    return ImageClip(tmp.name)


def _calculate_scene_durations(total_duration: float, num_scenes: int) -> List[float]:
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
    avatar = VideoFileClip(avatar_path)
    av_w, av_h = avatar.size
    avatar_duration = avatar.duration - 0.05

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
    img = Image.new('RGB', (width, height), color)
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(tmp.name, 'PNG')
    tmp.close()
    return tmp.name


def _create_hook_card(hook_text: str, width: int, height: int, duration: float = 1.5) -> Optional[ImageClip]:
    if not hook_text or len(hook_text.strip()) < 5:
        return None

    try:
        from PIL import ImageDraw, ImageFont

        img = Image.new('RGB', (width, height), (10, 5, 25))
        draw = ImageDraw.Draw(img)

        font_size = 52
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

        max_width = width - 120
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

        lines = lines[:3]

        line_height = font_size + 12
        total_text_height = len(lines) * line_height
        y_start = (height - total_text_height) // 2

        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            y = y_start + i * line_height
            draw.text((x + 2, y + 2), line, fill=(30, 20, 50), font=font)
            draw.text((x, y), line, fill=(255, 255, 255), font=font)

        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(tmp.name, 'PNG')
        tmp.close()

        clip = ImageClip(tmp.name).set_duration(duration).set_start(0)
        clip = clip.crossfadeout(0.3)

        print(f"  [SPLIT] Hook card: \"{hook_text[:50]}...\" ({duration:.1f}s)")
        return clip

    except Exception as e:
        print(f"  [SPLIT] Hook card creation failed: {e}")
        return None


def _render_scenes_ffmpeg(
    image_paths: List[str],
    scene_timestamps: Optional[List[Dict]],
    total_dur: float,
) -> tuple:
    """
    Pre-render all scene clips via ffmpeg zoompan filter.

    Returns:
        (scene_clips, bg_layer, scene_video_paths) — clips for compositing,
        a solid bg layer, and temp file paths for cleanup.
    """
    num_scenes = len(image_paths)
    temp_files = []

    if scene_timestamps and len(scene_timestamps) == num_scenes:
        print(f"  [SPLIT] Using SYNCED scene timestamps (content-driven, ffmpeg zoompan)")
        timings = []
        for i, ts in enumerate(scene_timestamps):
            start = ts['start']
            end = ts['end']
            dur = end - start
            if dur <= 0:
                dur = 0.5
            timings.append((start, dur))
            print(f"    Scene {i}: {start:.2f}s for {dur:.2f}s")

        last_ts = scene_timestamps[-1]
        needed_dur = total_dur - last_ts['start']
        if needed_dur > 0:
            timings[-1] = (last_ts['start'], needed_dur)
            print(f"  [SPLIT] Last scene extended to {needed_dur:.2f}s")
    else:
        if scene_timestamps:
            print(f"  [SPLIT] scene_timestamps mismatch ({len(scene_timestamps)} vs {num_scenes}), using fallback")
        scene_durations = _calculate_scene_durations(total_dur, num_scenes)
        timings = []
        t = 0
        for i, dur in enumerate(scene_durations):
            timings.append((t, dur))
            t += dur

    scene_clips = []
    for idx, img_path in enumerate(image_paths):
        start, dur = timings[idx]

        tmp = tempfile.NamedTemporaryFile(suffix=f'_scene{idx}.mp4', delete=False)
        tmp_path = tmp.name
        tmp.close()
        temp_files.append(tmp_path)

        ok = _render_scene_opencv(img_path, dur, idx, tmp_path)
        if ok and Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 0:
            clip = VideoFileClip(tmp_path).set_start(start)
            scene_clips.append(clip)
            print(f"    Scene {idx}: ffmpeg rendered ({dur:.2f}s, {_get_scene_profile(idx)['name']})")
        else:
            print(f"    Scene {idx}: ffmpeg failed, fallback to static image")
            clip = _resize_image_fullscreen(img_path).set_duration(dur).set_start(start)
            scene_clips.append(clip)

    bg_color = (10, 5, 25)
    bg_layer = ImageClip(
        _create_solid_color_image(VIDEO_W, VIDEO_H, bg_color)
    ).set_duration(total_dur)

    return scene_clips, bg_layer, temp_files


def _pre_render_overlay_ffmpeg(
    layers: list,
    subtitle_clips: list,
    total_dur: float,
    video_w: int,
    video_h: int,
    fps: int,
    ffmpeg_exe: str,
    temp_files: list,
) -> Optional[dict]:
    """
    Pre-render base video (bg+avatar) and subtitle/title overlay as separate MP4s.
    
    This avoids moviepy's CompositeVideoClip with 60+ Python callback subtitle clips,
    which makes write_videofile extremely slow (10+ min for 2 min video).
    
    Instead:
    1. Render bg+avatar as a single MP4 via moviepy (only 2 layers — fast)
    2. If subtitle clips exist, render them as a single transparent WebM (VP9 alpha)
    3. Composite both via ffmpeg CLI overlay filter (pure C, very fast)
    
    Returns {'base_video': path, 'overlay_video': path_or_None} or None on failure.
    """
    
    try:
        # ── Step 1: Render base video (bg + avatar + hook card + title) ──
        base_layers = [l for l in layers if l is not None]
        base_clip = CompositeVideoClip(base_layers, size=(video_w, video_h))
        base_clip = base_clip.set_duration(total_dur)
        
        base_tmp = tempfile.NamedTemporaryFile(suffix='_base.mp4', delete=False)
        base_path = base_tmp.name
        base_tmp.close()
        temp_files.append(base_path)
        
        t0 = _time.time()
        print(f"  [PRE-RENDER] Base video (bg+avatar, {len(base_layers)} layers)...")
        base_clip.write_videofile(
            base_path,
            codec='libx264',
            audio=False,
            fps=fps,
            ffmpeg_params=['-pix_fmt', 'yuv420p', '-crf', '18'],
            preset='ultrafast',
            threads=8,
            verbose=False,
            logger=None,
        )
        base_elapsed = _time.time() - t0
        print(f"  [PRE-RENDER] Base video: {base_elapsed:.1f}s")
        
        if not Path(base_path).exists() or Path(base_path).stat().st_size < 1000:
            print(f"  [PRE-RENDER] Base video failed")
            return None
        
        # ── Step 2: Render subtitle/title overlay as transparent WebM ──
        overlay_path = None
        if subtitle_clips:
            overlay_tmp = tempfile.NamedTemporaryFile(suffix='_overlay.webm', delete=False)
            overlay_path = overlay_tmp.name
            overlay_tmp.close()
            temp_files.append(overlay_path)
            
            # Create overlay clip from subtitle clips only
            overlay_clip = CompositeVideoClip(subtitle_clips, size=(video_w, video_h))
            overlay_clip = overlay_clip.set_duration(total_dur)
            
            t0 = _time.time()
            print(f"  [PRE-RENDER] Overlay ({len(subtitle_clips)} subtitle clips)...")
            overlay_clip.write_videofile(
                overlay_path,
                codec='libvpx-vp9',
                audio=False,
                fps=fps,
                ffmpeg_params=[
                    '-pix_fmt', 'yuva420p',
                    '-crf', '30',
                    '-b:v', '0',
                ],
                preset='ultrafast',
                threads=8,
                verbose=False,
                logger=None,
            )
            overlay_elapsed = _time.time() - t0
            print(f"  [PRE-RENDER] Overlay: {overlay_elapsed:.1f}s")
            
            if not Path(overlay_path).exists() or Path(overlay_path).stat().st_size < 100:
                print(f"  [PRE-RENDER] Overlay failed, will composite without it")
                overlay_path = None
        
        return {
            'base_video': base_path,
            'overlay_video': overlay_path,
        }
        
    except Exception as e:
        print(f"  [PRE-RENDER] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def _assemble_pure_ffmpeg(
    audio_path: str,
    image_paths: List[str],
    avatar_path: str,
    output_path: str,
    total_dur: float,
    scene_timestamps: Optional[List[Dict]],
    hook_text: str = None,
    script_text: str = None,
    word_timestamps: list = None,
    subtitle_y: int = 0,
    music_path: str = None,
) -> bool:
    """
    Pure ffmpeg assembly — no moviepy compositing at all. Zero Python frame rendering.
    Much more memory-efficient for longer videos (80s+).
    
    Pipeline:
    1. Render each scene image as MP4 clip via ffmpeg zoompan (no moviepy)
    2. Render looping avatar via ffmpeg
    3. Concatenate scenes into top-half video via ffmpeg concat demuxer
    4. Stack top-half + avatar via ffmpeg overlay
    5. Burn ASS subtitles + fade via ffmpeg
    6. Mix voiceover + music audio
    7. Final mux with validation
    """
    ffmpeg_exe = _find_ffmpeg()
    if not ffmpeg_exe:
        return False

    bg_color = '0x0A0519'
    VIDEO_FADE_DURATION = 0.8
    MUSIC_DIM_DURATION = 10.0
    temp_files = []

    try:
        # ── 1. Render each scene as MP4 via ffmpeg zoompan (no moviepy) ──
        scene_video_paths = []
        if scene_timestamps and len(scene_timestamps) == len(image_paths):
            timings = []
            for i, ts in enumerate(scene_timestamps):
                start = ts['start']
                end = ts['end']
                dur = max(end - start, 0.5)
                timings.append((start, dur))
            last_ts = scene_timestamps[-1]
            needed_dur = total_dur - last_ts['start']
            if needed_dur > 0:
                timings[-1] = (last_ts['start'], needed_dur)
        else:
            scene_durations = _calculate_scene_durations(total_dur, len(image_paths))
            timings = []
            t = 0
            for i, dur in enumerate(scene_durations):
                timings.append((t, dur))
                t += dur

        for idx, img_path in enumerate(image_paths):
            start, dur = timings[idx]
            tmp = tempfile.NamedTemporaryFile(suffix=f'_scene{idx}.mp4', delete=False)
            tmp_path = tmp.name
            tmp.close()
            temp_files.append(tmp_path)

            ok = _render_scene_opencv(img_path, dur, idx, tmp_path)
            if ok and Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 0:
                scene_video_paths.append(tmp_path)
                print(f"    [PURE-FF] Scene {idx}: rendered ({dur:.2f}s)")
            else:
                print(f"    [PURE-FF] Scene {idx}: ffmpeg zoompan FAILED")
                return False

        # ── 2. Pre-render looping avatar via ffmpeg ──
        avatar_path_out = tempfile.mktemp(suffix='_avatar.mp4')
        temp_files.append(avatar_path_out)
        print(f"  [PURE-FF] Rendering avatar loop ({total_dur:.1f}s)...")
        if not _render_avatar_ffmpeg(avatar_path, total_dur, avatar_path_out):
            print(f"  [PURE-FF] Avatar render failed")
            return False

        # ── 3. Concatenate scene clips into top-half video via concat demuxer ──
        concat_list_path = tempfile.mktemp(suffix='_concat.txt')
        temp_files.append(concat_list_path)
        with open(concat_list_path, 'w', encoding='utf-8') as f:
            for svp in scene_video_paths:
                safe_path = svp.replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        scenes_concat_path = tempfile.mktemp(suffix='_scenes.mp4')
        temp_files.append(scenes_concat_path)
        concat_cmd = [
            ffmpeg_exe, '-y',
            '-f', 'concat', '-safe', '0',
            '-i', concat_list_path,
            '-c', 'copy',
            '-y',
            scenes_concat_path,
        ]
        print(f"  [PURE-FF] Concatenating {len(scene_video_paths)} scenes...")
        r = subprocess.run(concat_cmd, capture_output=True, timeout=120, text=True)
        if r.returncode != 0:
            print(f"  [PURE-FF] Scene concat failed: rc={r.returncode}")
            print(f"    stdout: {r.stdout[:300]}")
            print(f"    stderr: {r.stderr[:500]}")
            return False
        # Also check if file has content
        concat_path = Path(scenes_concat_path)
        if not concat_path.exists() or concat_path.stat().st_size < 1000:
            print(f"  [PURE-FF] Scene concat output missing or tiny ({concat_path.stat().st_size if concat_path.exists() else 0} bytes)")
            print(f"    stderr: {r.stderr[:500]}")
            return False

        # Validate concat result
        scenes_val = _validate_mp4(scenes_concat_path, ffmpeg_exe)
        if not scenes_val['valid']:
            print(f"  [PURE-FF] Scene concat validation failed: {scenes_val['error']}")
            return False
        print(f"  [PURE-FF] Scenes concatenated OK (v:{scenes_val['video_codec']})")

        # ── 4. Stack top-half (scenes) + avatar (bottom) into 1080x1920 ──
        stacked_path = tempfile.mktemp(suffix='_stacked.mp4')
        temp_files.append(stacked_path)

        stack_filter = (
            f'[0:v]pad={VIDEO_W}:{VIDEO_H}:0:0:color={bg_color}[padded];'
            f'[padded][1:v]overlay=0:{TOP_H}:format=auto[stacked]'
        )
        stack_cmd = [
            ffmpeg_exe, '-y',
            '-i', scenes_concat_path,
            '-i', avatar_path_out,
            '-filter_complex', stack_filter,
            '-map', '[stacked]',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'fast', '-crf', str(_adaptive_crf(total_dur)),
            '-profile:v', 'high', '-level', '4.0', '-bf', '2',
            '-t', f'{total_dur:.3f}',
            '-an',
            stacked_path,
        ]
        print(f"  [PURE-FF] Stacking scene area + avatar into {VIDEO_W}x{VIDEO_H}...")
        stack_timeout = max(300, int(total_dur * 3))
        r = subprocess.run(stack_cmd, capture_output=True, timeout=stack_timeout, text=True)
        if r.returncode != 0:
            print(f"  [PURE-FF] Stack failed: {r.stderr[:500]}")
            return False

        stack_val = _validate_mp4(stacked_path, ffmpeg_exe)
        if not stack_val['valid']:
            print(f"  [PURE-FF] Stack validation failed: {stack_val['error']}")
            return False
        print(f"  [PURE-FF] Stack OK ({Path(stacked_path).stat().st_size / (1024*1024):.1f}MB)")

        # ── 5. Generate ASS subtitles (always generate if we have timestamps or a title) ──
        ass_path = None
        if (word_timestamps and script_text) or hook_text:
            try:
                ass_content = generate_ass_subtitles(
                    script_text=script_text or '',
                    word_timestamps=word_timestamps or [],
                    video_width=VIDEO_W,
                    video_height=VIDEO_H,
                    band_y_position=subtitle_y,
                    hook_text=hook_text,
                    total_duration=total_dur,
                )
                if ass_content:
                    ass_path = tempfile.mktemp(suffix='_subs.ass')
                    temp_files.append(ass_path)
                    with open(ass_path, 'w', encoding='utf-8') as f:
                        f.write(ass_content)
                    print(f"  [PURE-FF] ASS generated ({len(ass_content)} chars, hook={bool(hook_text)}, subs={bool(word_timestamps)})")
                else:
                    print(f"  [PURE-FF] WARNING: ASS generation returned empty content")
                    log.warning("assembly.ass_empty", reason="generate_ass_subtitles returned empty string")
            except Exception as e:
                print(f"  [PURE-FF] ASS generation failed: {e}")
                ass_path = None

        # ── 6. Build video filter chain (subtitles + fade) ──
        video_filter = ''

        if ass_path and Path(ass_path).exists():
            safe_ass = ass_path.replace('\\', '/').replace(':', '\\:')
            video_filter = f"subtitles='{safe_ass}'"

        # Add fade-out at the end
        if video_filter:
            video_filter += f',fade=t=out:st={total_dur - VIDEO_FADE_DURATION:.3f}:d={VIDEO_FADE_DURATION}'
        else:
            video_filter = f'fade=t=out:st={total_dur - VIDEO_FADE_DURATION:.3f}:d={VIDEO_FADE_DURATION}'

        # ── 7. Mix audio (voiceover + music) ──
        mixed_audio_path = tempfile.mktemp(suffix='_audio.wav')
        temp_files.append(mixed_audio_path)

        music_ok = False
        if music_path and Path(music_path).exists():
            try:
                dim_start = max(0, total_dur - MUSIC_DIM_DURATION)
                mix_cmd = [
                    ffmpeg_exe, '-y',
                    '-i', audio_path,
                    '-i', music_path,
                    '-filter_complex',
                    f'[1:a]atrim=0:{total_dur:.3f},asetpts=PTS-STARTPTS,'
                    f'highpass=f=80,'
                    f'afade=t=in:st=0:d=1,'
                    f'volume=0.30,afade=t=out:st={dim_start:.3f}:d={MUSIC_DIM_DURATION:.1f}[music];'
                    f'[0:a][music]amix=inputs=2:duration=longest:normalize=0,'
                    f'afade=t=out:st={total_dur - VIDEO_FADE_DURATION:.3f}:d={VIDEO_FADE_DURATION}[out]',
                    '-map', '[out]',
                    '-ar', '44100', '-ac', '2',
                    mixed_audio_path,
                ]
                r = subprocess.run(mix_cmd, capture_output=True, timeout=60, text=True)
                if r.returncode == 0 and Path(mixed_audio_path).exists():
                    music_ok = True
                    print(f"  [PURE-FF] Audio mixed with music")
                else:
                    print(f"  [PURE-FF] Audio mix failed, using voice only")
            except Exception as e:
                print(f"  [PURE-FF] Audio mix error: {e}")

        if not music_ok:
            import shutil as _shutil
            _shutil.copy2(audio_path, mixed_audio_path)

        # ── 8. Final mux: video (with subs/fade) + audio ──
        final_cmd = [
            ffmpeg_exe, '-y',
            '-i', stacked_path,
            '-i', mixed_audio_path,
            '-vf', video_filter,
            '-map', '0:v', '-map', '1:a',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', str(_adaptive_crf(total_dur)),
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'high',
            '-level', '4.0',
            '-bf', '2',
            '-movflags', '+faststart',
            '-c:a', 'aac', '-b:a', '192k',
            '-ar', '44100', '-ac', '2',
            '-t', f'{total_dur:.3f}',
            str(output_path),
        ]

        timeout_s = max(300, int(total_dur * 3))
        print(f"  [PURE-FF] Final mux (timeout={timeout_s}s)...")
        r = subprocess.run(final_cmd, capture_output=True, timeout=timeout_s, text=True)
        if r.returncode != 0:
            print(f"  [PURE-FF] Final mux failed: {r.stderr[:500]}")
            return False

        out_path = Path(output_path)
        if not out_path.exists() or out_path.stat().st_size < 1000:
            print(f"  [PURE-FF] Output file missing or tiny")
            return False

        # ── 9. Validate output ──
        validation = _validate_mp4(str(out_path), ffmpeg_exe)
        if not validation['valid']:
            print(f"  [PURE-FF] Output validation FAILED: {validation['error']}")
            try:
                out_path.unlink()
            except:
                pass
            return False

        print(f"  [PURE-FF] Assembly complete: {out_path.stat().st_size / (1024*1024):.1f}MB "
              f"(v:{validation['video_codec']} a:{validation['audio_codec']})")
        return True

    except subprocess.TimeoutExpired:
        print(f"  [PURE-FF] Timed out during assembly")
        return False
    except Exception as e:
        print(f"  [PURE-FF] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        for tf in temp_files:
            try:
                if Path(tf).exists():
                    Path(tf).unlink()
            except:
                pass


def _create_title_image(text: str, width: int, height: int, output_path: str):
    """Create a small title PNG image for overlay via ffmpeg."""
    from PIL import ImageDraw, ImageFont
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_size = 48
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    max_width = width - 120
    words = text.strip().upper().split()
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
    lines = lines[:2]
    line_height = font_size + 8
    y_start = (height - len(lines) * line_height) // 2
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (width - tw) // 2
        y = y_start + i * line_height
        draw.text((x + 3, y + 3), line, fill=(0, 0, 0, 200), font=font)
        draw.text((x, y), line, fill=(255, 215, 0, 255), font=font)
    img.save(output_path, 'PNG')


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
    if not audio_path or not Path(audio_path).exists():
        return {"success": False, "error": f"Audio not found: {audio_path}"}
    if not image_paths:
        return {"success": False, "error": "No image paths provided"}

    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
        return {"success": False, "error": f"Missing images: {missing}"}

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

    temp_files = []

    try:
        audio = AudioFileClip(audio_path)
        total_dur = audio.duration
        print(f"  [SPLIT] Audio duration: {total_dur:.1f}s")

        # ── SCENE BACKGROUNDS: ffmpeg zoompan ──
        print(f"  [SPLIT] Rendering {len(image_paths)} scenes via ffmpeg zoompan...")
        render_start = _time.time()

        scene_clips, bg_layer, scene_temp = _render_scenes_ffmpeg(
            image_paths, scene_timestamps, total_dur,
        )
        temp_files.extend(scene_temp)

        render_elapsed = _time.time() - render_start
        print(f"  [SPLIT] Scene rendering: {render_elapsed:.1f}s ({len(image_paths)} scenes)")

        background = CompositeVideoClip([bg_layer] + scene_clips, size=(VIDEO_W, VIDEO_H))
        background = background.set_duration(total_dur)

        if background.size != [VIDEO_W, VIDEO_H]:
            background = background.resize((VIDEO_W, VIDEO_H))

        # ── BOTTOM AREA: Avatar loop ──
        bottom_half = _prepare_avatar_bottom(avatar_path, total_dur)
        bottom_half = bottom_half.set_position((0, TOP_H))
        print(f"  [SPLIT] Avatar: {BOTTOM_H}px at y={TOP_H}")

        # ── COMPOSITE (base layers only for ffmpeg pipeline) ──
        layers = [background, bottom_half]

        # ── HOOK CARD ──
        if hook_card_text:
            hook_clip = _create_hook_card(hook_card_text, VIDEO_W, VIDEO_H, duration=1.5)
            if hook_clip:
                layers.append(hook_clip)
                print(f"  [SPLIT] Hook card added (1.5s pattern interrupt)")

        # ── TITLE OVERLAY ──
        title_clip = None
        if hook_text:
            title_clip = create_title_clip(
                hook_text, VIDEO_W, VIDEO_H,
                duration=total_dur
            )
            if title_clip:
                layers.append(title_clip)
                print(f"  [SPLIT] Title overlay (persistent): \"{hook_text[:60]}...\"")

        # ── SUBTITLES ──
        subtitle_clips = []
        subtitle_y = TOP_H - 140
        if word_timestamps and len(word_timestamps) > 0:
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

        has_overlay = len(subtitle_clips) > 0 or title_clip is not None

        # ── EXPORT ──
        MUSIC_DIM_DURATION = 10.0
        VIDEO_FADE_DURATION = 0.8
        MUSIC_BASE_VOLUME = 0.30

        out_path = Path(output_path) if output_path else Path(__file__).parent.parent.parent / "output" / "videos" / f"split_{int(total_dur)}s.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_exe = _find_ffmpeg()
        export_ok = False

        # ── PATH 1: Pure ffmpeg assembly (most reliable, no moviepy compositing) ──
        if ffmpeg_exe:
            try:
                audio.close()
            except:
                pass

            print(f"  [SPLIT] Attempting pure-ffmpeg assembly (no moviepy frames)...")
            pure_ok = _assemble_pure_ffmpeg(
                audio_path=audio_path,
                image_paths=image_paths,
                avatar_path=str(AVATAR_PATH) if not avatar_path else avatar_path,
                output_path=str(out_path),
                total_dur=total_dur,
                scene_timestamps=scene_timestamps,
                hook_text=hook_text,
                script_text=script_text or '',
                word_timestamps=word_timestamps,
                subtitle_y=subtitle_y,
                music_path=str(MUSIC_PATH) if MUSIC_PATH.exists() else None,
            )
            if pure_ok:
                validation = _validate_mp4(str(out_path), ffmpeg_exe)
                if validation['valid']:
                    export_ok = True
                    print(f"  [SPLIT] Pure-ffmpeg assembly validated (v:{validation['video_codec']} a:{validation['audio_codec']})")
                else:
                    print(f"  [SPLIT] Pure-ffmpeg output invalid: {validation['error']}")
                    try:
                        out_path.unlink()
                    except:
                        pass
            else:
                print(f"  [SPLIT] Pure-ffmpeg assembly failed, falling back to overlay pipeline...")

        # ── PATH 2: moviepy overlay pipeline (original fast path) ──
        if not export_ok and has_overlay and ffmpeg_exe:
            print(f"  [SPLIT] Using fast ffmpeg pipeline (overlay pre-render)...")

            # Close audio file so ffmpeg can read it
            try:
                audio.close()
            except:
                pass

            overlay_ok = _pre_render_overlay_ffmpeg(
                layers=[background, bottom_half, title_clip],
                subtitle_clips=subtitle_clips,
                total_dur=total_dur,
                video_w=VIDEO_W,
                video_h=VIDEO_H,
                fps=FPS,
                ffmpeg_exe=ffmpeg_exe,
                temp_files=temp_files,
            )
            if overlay_ok:
                base_video = overlay_ok['base_video']
                overlay_video = overlay_ok['overlay_video']
                overlay_exists = overlay_video is not None

                # pre-mix audio
                audio_tmp = tempfile.NamedTemporaryFile(suffix='_mixed.wav', delete=False)
                audio_tmp_path = audio_tmp.name
                audio_tmp.close()
                temp_files.append(audio_tmp_path)

                pre_rendered_ok = False
                if MUSIC_PATH.exists():
                    try:
                        dim_start = max(0, total_dur - MUSIC_DIM_DURATION)
                        mix_cmd = [
                            ffmpeg_exe, '-y',
                            '-i', audio_path,
                            '-i', str(MUSIC_PATH),
                            '-filter_complex',
                            f'[1:a]atrim=0:{total_dur:.3f},asetpts=PTS-STARTPTS,'
                            f'highpass=f=80,'
                            f'afade=t=in:st=0:d=1,'
                            f'volume=0.30,afade=t=out:st={dim_start:.3f}:d={MUSIC_DIM_DURATION:.1f}[music];'
                            f'[0:a][music]amix=inputs=2:duration=longest:normalize=0,'
                            f'afade=t=out:st={total_dur - VIDEO_FADE_DURATION:.3f}:d={VIDEO_FADE_DURATION}[out]',
                            '-map', '[out]',
                            '-ar', '44100', '-ac', '2',
                            audio_tmp_path
                        ]
                        mix_result = subprocess.run(mix_cmd, capture_output=True, timeout=60)
                        if mix_result.returncode == 0 and Path(audio_tmp_path).exists():
                            pre_rendered_ok = True
                            print(f"  [SPLIT] Pre-mixed audio via ffmpeg CLI ({Path(audio_tmp_path).stat().st_size/1024:.0f}KB)")
                    except Exception as e:
                        print(f"  [SPLIT] ffmpeg audio mix error: {e}")

                final_audio = audio_tmp_path if pre_rendered_ok else audio_path

                # ffmpeg final composite
                final_cmd = [ffmpeg_exe, '-y']
                final_cmd += ['-i', base_video]
                if overlay_exists:
                    final_cmd += ['-i', overlay_video]
                final_cmd += ['-i', final_audio]

                n_inputs = 1 + (1 if overlay_exists else 0) + 1
                overlay_idx = 1 if overlay_exists else None
                audio_idx = (2 if overlay_exists else 1)

                filter_parts = []
                if overlay_exists:
                    filter_parts.append(f'[0:v][1:v]overlay=0:0:format=auto[bg]')
                    filter_parts.append(f'[bg]fade=t=out:st={total_dur - VIDEO_FADE_DURATION:.3f}:d={VIDEO_FADE_DURATION}[vout]')
                else:
                    filter_parts.append(f'[0:v]fade=t=out:st={total_dur - VIDEO_FADE_DURATION:.3f}:d={VIDEO_FADE_DURATION}[vout]')

                filter_str = ';'.join(filter_parts)

                final_cmd += [
                    '-filter_complex', filter_str,
                    '-map', '[vout]',
                    '-map', f'{audio_idx}:a',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', str(_adaptive_crf(total_dur)),
                    '-pix_fmt', 'yuv420p',
                    '-profile:v', 'high',
                    '-level', '4.0',
                    '-bf', '2',
                    '-movflags', '+faststart',
                    '-c:a', 'aac', '-b:a', '192k',
                    '-ar', '44100', '-ac', '2',
                    str(out_path)
                ]

                composite_timeout = max(300, int(total_dur * 5))
                print(f"  [SPLIT] ffmpeg final composite ({'2' if overlay_exists else '1'} video + 1 audio, timeout={composite_timeout}s)...")
                try:
                    final_result = subprocess.run(final_cmd, capture_output=True, timeout=composite_timeout, text=True)
                    if final_result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
                        validation = _validate_mp4(str(out_path), ffmpeg_exe)
                        if validation['valid']:
                            export_ok = True
                            print(f"  [SPLIT] ffmpeg composite OK ({out_path.stat().st_size / (1024*1024):.1f}MB, v:{validation['video_codec']} a:{validation['audio_codec']})")
                        else:
                            print(f"  [SPLIT] ffmpeg composite output invalid: {validation['error']}")
                            try:
                                out_path.unlink()
                            except:
                                pass
                    else:
                        print(f"  [SPLIT] ffmpeg composite FAILED rc={final_result.returncode}")
                        print(f"    stderr: {final_result.stderr[:500]}")
                except subprocess.TimeoutExpired:
                    print(f"  [SPLIT] ffmpeg composite timed out ({composite_timeout}s)")

                # cleanup pre-renders
                for p in [base_video, overlay_video]:
                    if p:
                        try:
                            Path(p).unlink()
                        except:
                            pass

        if not export_ok:
            print(f"  [SPLIT] Fast pipeline unavailable/failed, using moviepy fallback...")
            if has_overlay:
                final = CompositeVideoClip(layers, size=(VIDEO_W, VIDEO_H))
            else:
                final = CompositeVideoClip(layers, size=(VIDEO_W, VIDEO_H))

            mixed_audio = audio
            if MUSIC_PATH.exists():
                try:
                    music = AudioFileClip(str(MUSIC_PATH))
                    if music.duration > total_dur:
                        music = music.subclip(0, total_dur)
                    music_loud = music.volumex(MUSIC_BASE_VOLUME)
                    music_loud = music_loud.audio_fadeout(MUSIC_DIM_DURATION)
                    mixed_audio = CompositeAudioClip([audio, music_loud.set_duration(total_dur)])
                    music.close()
                    print(f"  [SPLIT] Music at {MUSIC_BASE_VOLUME:.0%} (-20dB), {MUSIC_DIM_DURATION:.0f}s dim")
                except Exception as music_err:
                    print(f"  [SPLIT] Music mixing failed, voice only: {music_err}")
                    mixed_audio = audio

            audio_tmp = tempfile.NamedTemporaryFile(suffix='_mixed.wav', delete=False)
            audio_tmp_path = audio_tmp.name
            audio_tmp.close()
            temp_files.append(audio_tmp_path)

            try:
                mixed_audio.close()
            except:
                pass
            try:
                audio.close()
            except:
                pass

            pre_rendered_ok = False
            if MUSIC_PATH.exists():
                try:
                    if ffmpeg_exe:
                        dim_start = max(0, total_dur - MUSIC_DIM_DURATION)
                        mix_cmd = [
                            ffmpeg_exe, '-y',
                            '-i', audio_path,
                            '-i', str(MUSIC_PATH),
                            '-filter_complex',
                            f'[1:a]atrim=0:{total_dur:.3f},asetpts=PTS-STARTPTS,'
                            f'highpass=f=80,'
                            f'afade=t=in:st=0:d=1,'
                            f'volume=0.30,afade=t=out:st={dim_start:.3f}:d={MUSIC_DIM_DURATION:.1f}[music];'
                            f'[0:a][music]amix=inputs=2:duration=longest:normalize=0,'
                            f'afade=t=out:st={total_dur - VIDEO_FADE_DURATION:.3f}:d={VIDEO_FADE_DURATION}[out]',
                            '-map', '[out]',
                            '-ar', '44100', '-ac', '2',
                            audio_tmp_path
                        ]
                        result = subprocess.run(mix_cmd, capture_output=True, timeout=60)
                        if result.returncode == 0 and Path(audio_tmp_path).exists():
                            pre_rendered_audio = AudioFileClip(audio_tmp_path)
                            final = final.set_audio(pre_rendered_audio)
                            pre_rendered_ok = True
                except Exception:
                    pass

            if not pre_rendered_ok:
                try:
                    voice_audio = AudioFileClip(audio_path)
                    voice_audio = voice_audio.subclip(0, total_dur)
                    final = final.set_audio(voice_audio)
                except Exception:
                    final = final.set_audio(None)

            final = final.fadeout(VIDEO_FADE_DURATION)

            print(f"  [SPLIT] moviepy export (fallback) to {out_path}...")
            final.write_videofile(
                str(out_path),
                codec="libx264",
                audio_codec="aac",
                fps=FPS,
                ffmpeg_params=[
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-crf', str(_adaptive_crf(total_dur)),
                    '-bf', '2',
                ],
                preset='ultrafast',
                threads=8,
                verbose=False,
                logger=None,
            )

            # Validate moviepy output
            if ffmpeg_exe and out_path.exists() and out_path.stat().st_size > 1000:
                moviepy_validation = _validate_mp4(str(out_path), ffmpeg_exe)
                if moviepy_validation['valid']:
                    export_ok = True
                    print(f"  [SPLIT] moviepy export validated (v:{moviepy_validation['video_codec']} a:{moviepy_validation['audio_codec']})")
                else:
                    print(f"  [SPLIT] moviepy output INVALID: {moviepy_validation['error']}")
                    try:
                        out_path.unlink()
                    except:
                        pass

            final.close()
            background.close()
            bottom_half.close()
            for c in scene_clips:
                c.close()

        if not out_path.exists() or out_path.stat().st_size < 1000:
            # Clean up orphaned TEMP_MPY files
            try:
                import glob as _glob
                temp_pattern = str(out_path).replace('.mp4', 'TEMP_MPY_wvf_snd.mp4')
                for tmp_file in _glob.glob(temp_pattern):
                    try:
                        os.remove(tmp_file)
                    except:
                        pass
            except:
                pass
            # Clean orphaned temp files from root directory
            try:
                import glob as _glob2
                for orphan in _glob2.glob('video_*TEMP_MPY_wvf_snd.mp4'):
                    try:
                        os.remove(orphan)
                    except:
                        pass
            except:
                pass
            return {"success": False, "error": f"Output file not created or invalid: {out_path}"}

        # Final validation
        if ffmpeg_exe:
            final_val = _validate_mp4(str(out_path), ffmpeg_exe)
            if not final_val['valid']:
                print(f"  [SPLIT] FINAL VALIDATION FAILED: {final_val['error']}")
                try:
                    out_path.unlink()
                except:
                    pass
                return {"success": False, "error": f"Output file invalid: {final_val['error']}"}
            print(f"  [SPLIT] FINAL VALIDATION OK: video={final_val['video_codec']} audio={final_val['audio_codec']}")

        file_size = out_path.stat().st_size
        print(f"  [SPLIT] Export complete: {file_size / (1024*1024):.1f}MB")

        for tf in temp_files:
            try:
                if Path(tf).exists():
                    Path(tf).unlink()
            except:
                pass

        try:
            import glob
            temp_pattern = str(out_path).replace('.mp4', 'TEMP_MPY_wvf_snd.mp4')
            for tmp_file in glob.glob(temp_pattern):
                try:
                    os.remove(tmp_file)
                except:
                    pass
            # Also clean root-level orphaned temp files
            for orphan in glob.glob('video_*TEMP_MPY_wvf_snd.mp4'):
                try:
                    os.remove(orphan)
                except:
                    pass
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
            "scenes": len(image_paths),
            "subtitles": len(subtitle_clips),
            "effects_applied": [
                "full_screen_bg",
                "ffmpeg_zoompan",
                "ffmpeg_vignette",
                "avatar_loop",
                "title_overlay",
                "karaoke_subtitles",
            ],
            "render_method": "ffmpeg_overlay_pipeline",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Split video build failed: {str(e)}"}
