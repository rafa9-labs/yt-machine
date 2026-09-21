"""Delivery transcoding for finished videos.

WHY THIS EXISTS
===============
The assembler produces a lossless master. That is deliberate and must stay:
the pixel-art scenes are quantised to a 32-colour palette, and yuv420p chroma
subsampling invents hundreds of intermediate colours at every pixel edge
(measured: 32 -> 4,696 colours on a real frame), which visibly softens the art.

The problem is that "lossless" and "deliverable" are different requirements.
A 1080x1920 CRF-0 master runs ~38 Mbit/s, so a two-image run is ~400 MB and a
full run approaches 500 MB. Telegram's Bot API refuses anything over 50 MB,
and the TikTok and Instagram publishers read the entire file into memory, so a
lossless master is expensive for them as well.

The resolution is two artifacts with distinct contracts:

  master    lossless 4:4:4, archival, never re-encoded  (~400 MB)
  delivery  yuv420p H.264 High, size-bounded for upload (~20-40 MB)

Only the delivery copy is compressed. The master keeps the palette exactly, so
the archival record is not degraded by a delivery constraint.

SIZE DISCIPLINE
===============
A hard byte ceiling plus an unknown duration means the video bitrate must be
derived, not guessed:

    video_bitrate = (max_bytes * 8 / duration) - audio_bitrate - container_slack

The encode then caps that bitrate (``-maxrate`` / ``-bufsize``) on top of a
CRF, so quality stays constant on simple frames and cannot exceed the ceiling
on complex ones. Two-pass VBR would also work but doubles encode time and
needs a stats file; capped CRF reaches the same ceiling in one pass.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# Container/stream overhead allowed for before declaring a size budget.
_CONTAINER_SLACK = 0.98

DEFAULT_AUDIO_BITRATE_KBPS = 192
DEFAULT_CRF = 20
DEFAULT_PRESET = "fast"


def find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg executable.

    System PATH first, then imageio-ffmpeg (bundled with moviepy), which is
    present on installs that never added ffmpeg to PATH. Returns None rather
    than raising so callers can degrade.
    """
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).is_file():
            return bundled
    except Exception:
        pass
    return None


def probe_video(path: Path, ffmpeg_exe: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return stream facts for a video, or None when unreadable.

    Uses ffprobe when available and falls back to parsing the ffmpeg banner
    (the same technique the assembler's validator uses), so video metadata is
    still obtainable on installs without ffprobe.
    """
    candidate = Path(path)
    if not candidate.is_file():
        return None

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,pix_fmt,profile,width,height,bit_rate",
                    "-show_entries", "format=duration,size,bit_rate",
                    "-of", "json",
                    str(candidate),
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                payload = json.loads(result.stdout or "{}")
                streams = payload.get("streams") or [{}]
                container = payload.get("format") or {}
                stream = streams[0] if streams else {}
                return {
                    "duration_seconds": _to_float(container.get("duration")),
                    "size_bytes": _to_int(container.get("size"))
                    or candidate.stat().st_size,
                    "video_codec": stream.get("codec_name"),
                    "pix_fmt": stream.get("pix_fmt"),
                    "profile": stream.get("profile"),
                    "width": _to_int(stream.get("width")),
                    "height": _to_int(stream.get("height")),
                    "video_bitrate_kbps": _bits_to_kbps(stream.get("bit_rate")),
                    "container_bitrate_kbps": _bits_to_kbps(container.get("bit_rate")),
                }
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            pass

    executable = ffmpeg_exe or find_ffmpeg()
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "-i", str(candidate)],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    combined = result.stderr + result.stdout
    duration = None
    for token in combined.split():
        if token.startswith("time=") and ":" in token:
            duration = _parse_timestamp(token.split("=", 1)[1])
    if "Duration:" in combined:
        duration = _parse_timestamp(combined.split("Duration:")[1].split(",")[0].strip())
    if not duration:
        return None
    return {
        "duration_seconds": duration,
        "size_bytes": candidate.stat().st_size,
        "video_codec": None,
        "pix_fmt": "yuv420p" if "yuv420p" in combined else None,
        "profile": None,
        "width": None,
        "height": None,
        "video_bitrate_kbps": None,
        "container_bitrate_kbps": None,
    }


def video_bitrate_budget_kbps(
    duration_seconds: float,
    max_bytes: int,
    audio_bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS,
) -> Optional[int]:
    """Largest video bitrate that keeps a file under ``max_bytes``.

    Returns None when the duration is unusable, or when the budget left for
    video after audio is too small to produce a watchable result — a caller
    should treat that as "this cannot fit", not as "use a tiny bitrate".
    """
    if duration_seconds is None or duration_seconds <= 0 or max_bytes <= 0:
        return None
    total_kbps = (max_bytes * 8 / duration_seconds) / 1000
    video_kbps = int(total_kbps * _CONTAINER_SLACK) - int(audio_bitrate_kbps)
    if video_kbps < 200:
        return None
    return video_kbps


def transcode_for_delivery(
    master_path: Path,
    delivery_path: Path,
    *,
    max_bytes: int,
    audio_bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    """Produce a size-bounded upload copy of ``master_path``.

    Returns ``{"success", "path", "size_mb", "bitrate_kbps", "attempts"}`` on
    success, or ``{"success": False, "error": ...}``. The master is only ever
    read. On failure the partial delivery file is removed so a stale artifact
    is never mistaken for a usable one.

    Retry ladder: the first attempt uses capped CRF. If the container still
    exceeds the ceiling (rare, but possible when the encoder undershoots the
    cap on a very complex frame), one retry runs at a lower CRF ceiling. After
    that the delivery is reported as genuinely impossible for this budget.
    """
    source = Path(master_path)
    destination = Path(delivery_path)

    if not source.is_file():
        return {"success": False, "error": f"master not found: {source}"}

    executable = find_ffmpeg()
    if not executable:
        return {"success": False, "error": "ffmpeg not found"}

    probed = probe_video(source, executable)
    if not probed or not probed.get("duration_seconds"):
        return {"success": False, "error": "cannot read master duration"}

    duration = float(probed["duration_seconds"])
    budget_kbps = video_bitrate_budget_kbps(
        duration, max_bytes, audio_bitrate_kbps=audio_bitrate_kbps
    )
    if budget_kbps is None:
        return {
            "success": False,
            "error": (
                f"duration {duration:.1f}s leaves no usable video bitrate under "
                f"{max_bytes / (1024 * 1024):.1f}MB"
            ),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    last_error = ""

    for attempt_crf in _crf_ladder(crf, max_attempts):
        attempts += 1
        command = _build_delivery_command(
            executable, source, destination,
            video_kbps=budget_kbps,
            audio_bitrate_kbps=audio_bitrate_kbps,
            crf=attempt_crf,
            preset=preset,
        )
        try:
            result = subprocess.run(
                command, capture_output=True, text=True,
                timeout=max(300, int(duration * 6)),
            )
        except subprocess.TimeoutExpired:
            last_error = "ffmpeg timed out"
            continue

        if result.returncode != 0 or not destination.is_file():
            last_error = (result.stderr or "").strip()[-400:] or "ffmpeg failed"
            continue

        size = destination.stat().st_size
        if size <= max_bytes:
            return {
                "success": True,
                "path": str(destination),
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2),
                "bitrate_kbps": budget_kbps,
                "crf": attempt_crf,
                "attempts": attempts,
                "duration_seconds": round(duration, 2),
            }
        last_error = (
            f"still {size / (1024 * 1024):.1f}MB after attempt {attempts} "
            f"(limit {max_bytes / (1024 * 1024):.1f}MB)"
        )

    destination.unlink(missing_ok=True)
    return {
        "success": False,
        "error": last_error or "delivery encode failed",
        "attempts": attempts,
        "budget_kbps": budget_kbps,
    }


def _crf_ladder(crf: int, max_attempts: int) -> List[int]:
    """CRF values to try, tightening until the size ceiling is met."""
    ladder = []
    current = int(crf)
    for _ in range(max(1, max_attempts)):
        ladder.append(current)
        current = min(current + 4, 51)
    return ladder


def _build_delivery_command(
    ffmpeg_exe: str,
    source: Path,
    destination: Path,
    *,
    video_kbps: int,
    audio_bitrate_kbps: int,
    crf: int,
    preset: str,
) -> List[str]:
    """Build the delivery encode command.

    ``yuv420p`` and profile ``high`` are chosen for universal player and
    platform support — this is the copy that leaves the machine. ``maxrate``
    with ``bufsize`` caps the bitrate so the byte budget holds, while the CRF
    keeps quality constant on the simple frames that dominate pixel art.
    """
    return [
        ffmpeg_exe, "-y",
        "-i", str(source),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.0",
        "-preset", preset,
        "-crf", str(crf),
        "-maxrate", f"{video_kbps}k",
        "-bufsize", f"{video_kbps * 2}k",
        "-c:a", "aac",
        "-b:a", f"{audio_bitrate_kbps}k",
        "-ar", "44100",
        "-ac", "2",
        "-movflags", "+faststart",
        str(destination),
    ]


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bits_to_kbps(value: Any) -> Optional[int]:
    bits = _to_int(value)
    return int(bits / 1000) if bits else None


def _parse_timestamp(text: str) -> Optional[float]:
    try:
        hours, minutes, seconds = text.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (ValueError, AttributeError):
        return None
