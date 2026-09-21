"""Tests for the delivery transcoding stage.

The master is lossless and can be hundreds of megabytes; the delivery copy is
what providers actually receive. These tests pin the two properties that make
that split safe: the byte ceiling is respected, and the master is never
modified.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from src.video.media_export import (
    find_ffmpeg,
    probe_video,
    transcode_for_delivery,
    video_bitrate_budget_kbps,
)

ffmpeg_required = pytest.mark.skipif(
    find_ffmpeg() is None, reason="ffmpeg is required for delivery encoding"
)


# ══════════════════════════════════════════════════════════════
# Bitrate budget arithmetic (no ffmpeg needed)
# ══════════════════════════════════════════════════════════════
class TestBitrateBudget:
    def test_budget_scales_inversely_with_duration(self):
        short = video_bitrate_budget_kbps(60, 50 * 1024 * 1024)
        long = video_bitrate_budget_kbps(300, 50 * 1024 * 1024)
        assert short > long

    def test_budget_reserves_room_for_audio(self):
        """A 50 MB / 60 s budget must leave a sane video bitrate."""
        kbps = video_bitrate_budget_kbps(
            60, 50 * 1024 * 1024, audio_bitrate_kbps=192
        )
        # 50MB over 60s is ~6.9 Mbit/s total; audio takes 192k.
        assert 6000 < kbps < 7000

    def test_matches_the_measured_durations(self):
        """Sanity against real run lengths: 108 s and 130 s."""
        for duration in (108, 130):
            kbps = video_bitrate_budget_kbps(duration, 50 * 1024 * 1024)
            assert kbps is not None
            assert kbps > 2500, f"{duration}s budget too small: {kbps}"

    def test_returns_none_for_impossible_budget(self):
        """A two-hour video cannot fit in 50 MB; say so, do not emit 1 kbps."""
        assert video_bitrate_budget_kbps(7200, 50 * 1024 * 1024) is None

    @pytest.mark.parametrize("duration", [0, -1, None])
    def test_returns_none_for_invalid_duration(self, duration):
        assert video_bitrate_budget_kbps(duration, 50 * 1024 * 1024) is None

    def test_returns_none_for_non_positive_max_bytes(self):
        assert video_bitrate_budget_kbps(60, 0) is None


# ══════════════════════════════════════════════════════════════
# Probing
# ══════════════════════════════════════════════════════════════
class TestProbe:
    def test_probe_returns_none_for_missing_file(self, tmp_path):
        assert probe_video(tmp_path / "nope.mp4") is None

    def test_probe_returns_none_for_garbage(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"definitely not a video")
        assert probe_video(bad) is None

    @ffmpeg_required
    def test_probe_reads_real_stream_facts(self, tmp_path):
        master = _make_master(tmp_path)
        probed = probe_video(master)
        assert probed is not None
        assert probed["duration_seconds"] == pytest.approx(2.0, abs=0.3)
        assert probed["width"] == 320
        assert probed["height"] == 240
        assert probed["pix_fmt"] == "yuv444p"


# ══════════════════════════════════════════════════════════════
# Delivery transcode
# ══════════════════════════════════════════════════════════════
class TestTranscode:
    @ffmpeg_required
    def test_produces_a_fitting_yuv420p_copy(self, tmp_path):
        master = _make_master(tmp_path)
        delivery = tmp_path / "out" / "delivery.mp4"

        result = transcode_for_delivery(
            master, delivery, max_bytes=2 * 1024 * 1024
        )

        assert result["success"] is True, result.get("error")
        assert Path(result["path"]).is_file()
        assert result["size_bytes"] <= 2 * 1024 * 1024
        probed = probe_video(delivery)
        assert probed["pix_fmt"] == "yuv420p"
        assert probed["profile"] == "High"
        assert probed["video_codec"] == "h264"

    @ffmpeg_required
    def test_master_is_never_modified(self, tmp_path):
        master = _make_master(tmp_path)
        before_bytes = master.read_bytes()
        before_mtime = master.stat().st_mtime

        transcode_for_delivery(
            master, tmp_path / "delivery.mp4", max_bytes=2 * 1024 * 1024
        )

        assert master.read_bytes() == before_bytes
        assert master.stat().st_mtime == before_mtime

    @ffmpeg_required
    def test_palette_is_preserved_up_to_the_chroma_conversion(self, tmp_path):
        """The delivery copy is where yuv420p damage is expected and accepted.

        This documents the trade-off: the master keeps 32 colours, the
        delivery copy does not. If a future change makes the master lossy,
        this test's master half fails.
        """
        master = _make_master(tmp_path)
        assert _colour_count(master) <= 32

        delivery = tmp_path / "delivery.mp4"
        result = transcode_for_delivery(
            master, delivery, max_bytes=4 * 1024 * 1024, crf=18
        )
        assert result["success"] is True, result.get("error")
        assert _colour_count(delivery) > 32

    @ffmpeg_required
    def test_fails_cleanly_when_budget_cannot_fit(self, tmp_path):
        master = _make_master(tmp_path, duration=4.0)
        delivery = tmp_path / "delivery.mp4"

        result = transcode_for_delivery(master, delivery, max_bytes=1024)

        assert result["success"] is False
        assert "no usable video bitrate" in result["error"]
        assert not delivery.exists(), "a failed delivery must not leave an artifact"

    def test_fails_when_master_missing(self, tmp_path):
        result = transcode_for_delivery(
            tmp_path / "absent.mp4", tmp_path / "out.mp4",
            max_bytes=10 * 1024 * 1024,
        )
        assert result["success"] is False
        assert "master not found" in result["error"]


# ══════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════
def _make_master(tmp_path: Path, duration: float = 2.0) -> Path:
    """A tiny lossless 4:4:4 clip from a 32-colour frame, like the real master."""
    image = Image.new("RGB", (320, 240))
    pixels = image.load()
    palette = [(i * 37 % 256, i * 61 % 256, i * 91 % 256) for i in range(32)]
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = palette[((x // 20) + (y // 20)) % len(palette)]
    source = tmp_path / "frame.png"
    image.save(source)

    master = tmp_path / "master.mp4"
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-loglevel", "error",
            "-loop", "1", "-i", str(source),
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "0",
            "-r", "10", str(master),
        ],
        check=True, capture_output=True, timeout=120,
    )
    return master


def _colour_count(video: Path) -> int:
    frame = video.parent / f"{video.stem}_probe.png"
    subprocess.run(
        [
            find_ffmpeg(), "-y", "-loglevel", "error",
            "-i", str(video), "-frames:v", "1", str(frame),
        ],
        check=True, capture_output=True, timeout=60,
    )
    with Image.open(frame) as img:
        rgb = img.convert("RGB")
        getter = getattr(rgb, "get_flattened_data", None) or rgb.getdata
        return len(set(getter()))
