"""Tests for MP4 validation semantics.

``_validate_mp4`` gates two artifact classes with different contracts:

  * intermediate video-only files (concatenated scenes, stacked composite),
    which are audio-less by construction;
  * final deliverables, where the narration and music are the entire
    soundtrack and a missing audio stream means a broken artifact.

These tests pin both halves, and specifically guard the MoviePy fallback
failure class: that path can call ``set_audio(None)`` and write a silent
file, which previously validated as clean.
"""

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from src.video.split_video_assembler import _find_ffmpeg, _validate_mp4

ffmpeg_required = pytest.mark.skipif(
    _find_ffmpeg() is None, reason="ffmpeg is required to build validation fixtures"
)


# ══════════════════════════════════════════════════════════════
# fixtures built in tmp_path — never production files
# ══════════════════════════════════════════════════════════════
def _frame(path: Path) -> Path:
    image = path / "frame.png"
    Image.new("RGB", (64, 64), (10, 5, 25)).save(image)
    return image


def _run(cmd) -> None:
    subprocess.run(
        [_find_ffmpeg(), "-y", "-loglevel", "error", *cmd],
        check=True, capture_output=True, timeout=120,
    )


def _video_only(path: Path) -> Path:
    """A silent artifact — what a scene clip looks like, and what the MoviePy
    fallback produces when AudioFileClip fails."""
    out = path / "video_only.mp4"
    _run(["-loop", "1", "-i", str(_frame(path)), "-t", "1",
          "-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "0",
          "-r", "10", "-an", str(out)])
    return out


def _video_and_audio(path: Path) -> Path:
    """A normal deliverable: video plus an audio track."""
    out = path / "with_audio.mp4"
    _run(["-f", "lavfi", "-i", "sine=frequency=440:duration=1",
          "-loop", "1", "-i", str(_frame(path)), "-t", "1",
          "-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "0",
          "-r", "10", "-c:a", "aac", "-shortest", str(out)])
    return out


# ══════════════════════════════════════════════════════════════
# the four pass/fail combinations
# ══════════════════════════════════════════════════════════════
@ffmpeg_required
class TestRequirementMatrix:
    def test_video_only_permissive_is_valid(self, tmp_path):
        """Intermediates must keep working: they are audio-less by design."""
        result = _validate_mp4(str(_video_only(tmp_path)))
        assert result["valid"] is True
        assert result["video_codec"] == "h264"
        assert result["audio_codec"] is None

    def test_video_only_strict_is_invalid(self, tmp_path):
        result = _validate_mp4(str(_video_only(tmp_path)), require_audio=True)
        assert result["valid"] is False

    def test_video_and_audio_permissive_is_valid(self, tmp_path):
        result = _validate_mp4(str(_video_and_audio(tmp_path)))
        assert result["valid"] is True
        assert result["audio_codec"] == "aac"

    def test_video_and_audio_strict_is_valid(self, tmp_path):
        result = _validate_mp4(str(_video_and_audio(tmp_path)), require_audio=True)
        assert result["valid"] is True
        assert result["video_codec"] == "h264"
        assert result["audio_codec"] == "aac"


# ══════════════════════════════════════════════════════════════
# failure modes
# ══════════════════════════════════════════════════════════════
class TestFailures:
    def test_missing_file_is_invalid_both_modes(self, tmp_path):
        absent = tmp_path / "nope.mp4"
        assert _validate_mp4(str(absent))["valid"] is False
        assert _validate_mp4(str(absent), require_audio=True)["valid"] is False
        assert _validate_mp4(str(absent))["error"] == "file not found"

    def test_corrupted_file_is_invalid_both_modes(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"this is definitely not an mp4 container")
        for strict in (False, True):
            result = _validate_mp4(str(bad), require_audio=strict)
            assert result["valid"] is False, f"require_audio={strict}"
            assert result["error"]

    def test_truncated_file_is_invalid(self, tmp_path):
        """A real MP4 cut short loses its moov atom."""
        good = tmp_path / "good.mp4"
        good.write_bytes(b"\x00" * 4096)
        assert _validate_mp4(str(good))["valid"] is False


# ══════════════════════════════════════════════════════════════
# the distinct failure reason
# ══════════════════════════════════════════════════════════════
@ffmpeg_required
class TestFailureReason:
    def test_missing_audio_has_a_distinct_reason(self, tmp_path):
        """The message must name the actual problem, not report a generic
        invalid file — an operator needs to know audio is what is missing."""
        result = _validate_mp4(str(_video_only(tmp_path)), require_audio=True)
        assert result["error"] == "no audio stream found (require_audio=True)"

    def test_missing_video_reason_is_unchanged(self, tmp_path):
        """An audio-only file still reports the video problem."""
        audio_only = tmp_path / "audio_only.mp4"
        _run(["-f", "lavfi", "-i", "sine=frequency=440:duration=1",
              "-c:a", "aac", str(audio_only)])
        result = _validate_mp4(str(audio_only))
        assert result["valid"] is False
        assert "no video stream found" in result["error"]

    def test_video_problem_is_checked_before_audio(self, tmp_path):
        """With no video, the video error wins even in strict mode."""
        audio_only = tmp_path / "audio_only.mp4"
        _run(["-f", "lavfi", "-i", "sine=frequency=440:duration=1",
              "-c:a", "aac", str(audio_only)])
        result = _validate_mp4(str(audio_only), require_audio=True)
        assert "no video stream found" in result["error"]


# ══════════════════════════════════════════════════════════════
# metadata fidelity
# ══════════════════════════════════════════════════════════════
@ffmpeg_required
class TestMetadata:
    def test_codec_metadata_reported_in_both_modes(self, tmp_path):
        path = str(_video_and_audio(tmp_path))
        permissive = _validate_mp4(path)
        strict = _validate_mp4(path, require_audio=True)
        for result in (permissive, strict):
            assert result["video_codec"] == "h264"
            assert result["audio_codec"] == "aac"
            assert result["error"] is None

    def test_audio_codec_still_reported_when_not_required(self, tmp_path):
        """Detection is independent of the requirement, per the contract."""
        path = str(_video_and_audio(tmp_path))
        assert _validate_mp4(path)["audio_codec"] == "aac"
        assert _validate_mp4(str(_video_only(tmp_path)))["audio_codec"] is None


# ══════════════════════════════════════════════════════════════
# regression: the MoviePy silent-output failure class
# ══════════════════════════════════════════════════════════════
@ffmpeg_required
def test_moviepy_silent_output_is_rejected_as_final(tmp_path):
    """Regression test for the defect this change fixes.

    ``build_split_video``'s MoviePy fallback catches an AudioFileClip failure
    and calls ``set_audio(None)``, writing a final video with no audio. Before
    ``require_audio`` existed, that file validated as clean and shipped.

    This reproduces the artifact shape (a valid video container with no audio
    stream) and asserts the final-artifact contract rejects it.
    """
    silent_final = _video_only(tmp_path)

    # It is a genuinely playable file — this is not caught by "corrupt input".
    permissive = _validate_mp4(str(silent_final))
    assert permissive["valid"] is True, "fixture must be a real, playable MP4"
    assert permissive["audio_codec"] is None

    # The final-artifact contract must reject it.
    strict = _validate_mp4(str(silent_final), require_audio=True)
    assert strict["valid"] is False
    assert "no audio" in strict["error"]
