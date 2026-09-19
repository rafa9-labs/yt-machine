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
