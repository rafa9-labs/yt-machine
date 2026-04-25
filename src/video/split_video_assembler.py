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

AVATAR_PATH = Path(__file__).parent.parent.parent / "assets" / "avatar" / "avatar_loop.mp4"
MUSIC_PATH = Path(__file__).parent.parent.parent / "assets" / "avatar" / "music" / "news-yt.mp3"


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


def _render_scene_ffmpeg(
    img_path: str,
    duration: float,
    scene_idx: int,
    output_path: str,
    width: int = VIDEO_W,
    height: int = TOP_H,
) -> bool:
    """
    Render a single scene image into an MP4 clip using ffmpeg's zoompan filter.

    Effects alternate by scene index:
      - Even scenes (0,2,4): zoom OUT 1.20→1.0 from center (reveals full image)
      - Odd scenes (1,3,5):  zoom IN  1.0→1.20 from center (crops into center)

    The output frame size never changes — only the visible crop region shifts.
    No stretching, no panning — pure center-based zoom.
    
    Zoom is rendered at 2x oversampling (60 internal frames/sec → 30fps output)
    for maximum smoothness with the 20% zoom range.
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

    ZOOM_RANGE = 0.20
    INTERNAL_FPS = FPS * 2
    total_frames = int(duration * INTERNAL_FPS)
    if total_frames < 1:
        total_frames = 1

    safe_frames = max(total_frames - 1, 1)
    x_expr = "iw/2-(iw/zoom/2)"
    y_expr = "ih/2-(ih/zoom/2)"

    if scene_idx % 2 == 0:
        zoom_expr = f"if(eq(on\\,1)\\,{1.0 + ZOOM_RANGE}\\,max(1.0\\,zoom-{ZOOM_RANGE}/{safe_frames}))"
    else:
        zoom_expr = f"min({1.0 + ZOOM_RANGE}\\,1.0+{ZOOM_RANGE}*(on-1)/{safe_frames})"

    vf = (
        f"[0:v]scale={scaled_w}:{scaled_h},"
        f"crop={target_w}:{target_h}:{crop_x}:{crop_y},"
        f"zoompan=z='{zoom_expr}'"
        f":x='{x_expr}'"
        f":y='{y_expr}'"
        f":d={total_frames}"
        f":s={target_w}x{target_h}"
        f":fps={INTERNAL_FPS},"
        f"vignette=angle=0.3:mode=forward,"
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
        '-preset', 'slow',
        '-crf', '18',
        '-r', str(FPS),
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
    return result.returncode == 0


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

        ok = _render_scene_ffmpeg(img_path, dur, idx, tmp_path)
        if ok and Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 0:
            clip = VideoFileClip(tmp_path).set_start(start)
            scene_clips.append(clip)
            print(f"    Scene {idx}: ffmpeg rendered ({dur:.2f}s, zoom {'out' if idx % 2 == 0 else 'in'})")
        else:
            print(f"    Scene {idx}: ffmpeg failed, fallback to static image")
            clip = _resize_image_fullscreen(img_path).set_duration(dur).set_start(start)
            scene_clips.append(clip)

    bg_color = (10, 5, 25)
    bg_layer = ImageClip(
        _create_solid_color_image(VIDEO_W, VIDEO_H, bg_color)
    ).set_duration(total_dur)

    return scene_clips, bg_layer, temp_files


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
        import time as _time
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

        # ── COMPOSITE ──
        layers = [background, bottom_half]

        # ── HOOK CARD ──
        if hook_card_text:
            hook_clip = _create_hook_card(hook_card_text, VIDEO_W, VIDEO_H, duration=1.5)
            if hook_clip:
                layers.append(hook_clip)
                print(f"  [SPLIT] Hook card added (1.5s pattern interrupt)")

        # ── TITLE OVERLAY ──
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
        if word_timestamps and len(word_timestamps) > 0:
            subtitle_y = TOP_H - 140
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

        # ── BACKGROUND MUSIC ──
        MUSIC_DIM_DURATION = 10.0
        VIDEO_FADE_DURATION = 0.8
        MUSIC_BASE_VOLUME = 0.10
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
        else:
            print(f"  [SPLIT] No news-yt.mp3 found, skipping background music")

        # ── PRE-RENDER AUDIO ──
        audio_tmp = tempfile.NamedTemporaryFile(suffix='_mixed.wav', delete=False)
        audio_tmp_path = audio_tmp.name
        audio_tmp.close()

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
                ffmpeg_exe = _find_ffmpeg()
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
                        f'volume=0.10,afade=t=out:st={dim_start:.3f}:d={MUSIC_DIM_DURATION:.1f}[music];'
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
                        print(f"  [SPLIT] Pre-mixed audio via ffmpeg CLI ({Path(audio_tmp_path).stat().st_size/1024:.0f}KB)")
                    else:
                        print(f"  [SPLIT] ffmpeg mix failed (rc={result.returncode}): {result.stderr[:200]}")
            except Exception as e:
                print(f"  [SPLIT] ffmpeg CLI mix error: {e}")

        if not pre_rendered_ok:
            try:
                voice_audio = AudioFileClip(audio_path)
                voice_audio = voice_audio.subclip(0, total_dur)
                final = final.set_audio(voice_audio)
                print(f"  [SPLIT] Using voice-only audio (no music)")
            except Exception as e2:
                print(f"  [SPLIT] Voice-only audio failed: {e2}")
                final = final.set_audio(None)

        final = final.fadeout(VIDEO_FADE_DURATION)
        print(f"  [SPLIT] Applied {VIDEO_FADE_DURATION}s video fade-out")

        # ── EXPORT ──
        if not output_path:
            output_dir = Path(__file__).parent.parent.parent / "output" / "videos"
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
                '-pix_fmt', 'yuv420p',
                '-profile:v', 'high',
                '-level', '4.0',
                '-crf', '20',
                '-bf', '2',
            ],
            preset='medium',
            threads=4,
            verbose=False,
            logger=None,
        )

        out_path = Path(output_path)
        if not out_path.exists():
            return {"success": False, "error": "Output file not created"}

        # ── POST-EXPORT REMUX ──
        try:
            ffmpeg_exe = _find_ffmpeg()
            if ffmpeg_exe:
                remux_path = out_path.with_suffix('.remux.mp4')
                remux_cmd = [
                    ffmpeg_exe, '-y',
                    '-i', str(out_path),
                    '-c', 'copy',
                    '-movflags', '+faststart',
                    '-map', '0:v',
                    '-map', '0:a',
                    str(remux_path)
                ]
                remux_result = subprocess.run(remux_cmd, capture_output=True, timeout=60)
                if remux_result.returncode == 0 and remux_path.exists():
                    remux_path.replace(out_path)
                    print(f"  [SPLIT] Remuxed MP4 container (Windows compatibility fix)")
                else:
                    print(f"  [SPLIT] Remux failed (rc={remux_result.returncode}), keeping original")
                    try:
                        remux_path.unlink()
                    except:
                        pass
        except Exception as e:
            print(f"  [SPLIT] Remux error: {e}")

        file_size = out_path.stat().st_size
        print(f"  [SPLIT] Export complete: {file_size / (1024*1024):.1f}MB")

        # Cleanup
        final.close()
        background.close()
        bottom_half.close()
        for c in scene_clips:
            c.close()

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
            "render_method": "ffmpeg_native",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Split video build failed: {str(e)}"}
