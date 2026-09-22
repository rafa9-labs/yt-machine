"""Tests for validated pipeline run-size configuration."""

import pytest

from src.video.pipeline_config import DEFAULT_NUM_IMAGES, resolve_image_limit


def test_image_limit_defaults_to_full_production_count():
    assert resolve_image_limit(None) == DEFAULT_NUM_IMAGES
    assert resolve_image_limit("") == DEFAULT_NUM_IMAGES


@pytest.mark.parametrize("raw, expected", [("1", 1), ("4", 4), ("8", 8)])
def test_image_limit_accepts_supported_counts(raw, expected):
    assert resolve_image_limit(raw) == expected


@pytest.mark.parametrize("raw", ["0", "9", "four", "2.5"])
def test_image_limit_rejects_invalid_counts(raw):
    with pytest.raises(ValueError, match="YT_IMAGE_LIMIT"):
        resolve_image_limit(raw)


# ══════════════════════════════════════════════════════════════
# Retention configuration
# ══════════════════════════════════════════════════════════════
from src.video.pipeline_config import (  # noqa: E402
    DEFAULT_RETENTION_DAYS,
    DEFAULT_RETENTION_KEEP_LAST,
    DEFAULT_RETENTION_MODE,
    resolve_retention_days,
    resolve_retention_enabled,
    resolve_retention_keep_last,
    resolve_retention_mode,
)


class TestRetentionDefaults:
    def test_defaults_are_report_only(self):
        """The default must never delete: deletion requires an explicit act."""
        assert resolve_retention_mode(None) == "report"
        assert resolve_retention_mode("") == "report"
        assert DEFAULT_RETENTION_MODE == "report"
        assert resolve_retention_enabled(None) is True

    def test_defaults(self):
        assert resolve_retention_days(None) == 30
        assert resolve_retention_keep_last(None) == 3
        assert DEFAULT_RETENTION_DAYS == 30
        assert DEFAULT_RETENTION_KEEP_LAST == 3


class TestRetentionValidation:
    @pytest.mark.parametrize("raw, expected", [("0", 0), ("7", 7), ("365", 365)])
    def test_days_accepts_valid(self, raw, expected):
        assert resolve_retention_days(raw) == expected

    @pytest.mark.parametrize("raw", ["-1", "abc", "2.5", "99999"])
    def test_days_rejects_invalid(self, raw):
        with pytest.raises(ValueError, match="YT_RETENTION_DAYS"):
            resolve_retention_days(raw)

    @pytest.mark.parametrize("raw, expected", [("0", 0), ("10", 10)])
    def test_keep_last_accepts_valid(self, raw, expected):
        assert resolve_retention_keep_last(raw) == expected

    @pytest.mark.parametrize("raw", ["-1", "x", "99999"])
    def test_keep_last_rejects_invalid(self, raw):
        with pytest.raises(ValueError, match="YT_RETENTION_KEEP_LAST"):
            resolve_retention_keep_last(raw)

    @pytest.mark.parametrize("raw", ["full", "delivery_only", "off", "report", "FULL"])
    def test_mode_accepts_known_values(self, raw):
        assert resolve_retention_mode(raw) in ("full", "delivery_only", "off", "report")

    @pytest.mark.parametrize("raw", ["delete", "yes", "rm -rf"])
    def test_mode_rejects_unknown(self, raw):
        with pytest.raises(ValueError, match="YT_RETENTION_MODE"):
            resolve_retention_mode(raw)

    @pytest.mark.parametrize("raw, expected", [
        ("true", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("0", False), ("no", False), ("disabled", False),
    ])
    def test_enabled_toggle(self, raw, expected):
        assert resolve_retention_enabled(raw) is expected
