"""Tests for the Qwen generation profile contract."""

import json
from pathlib import Path

import pytest

from src.video.generation_profile import (
    GenerationProfileError,
    load_generation_profile,
    validate_generation_profile,
)


def test_active_profile_is_qwen_pixel_scene():
    profile = load_generation_profile()
    assert profile["name"] == "qwen_pixel_scene"
    assert profile["provider"] == "mlxgen"
    assert "qwen" in profile["model_id"].lower()
    assert (profile["width"], profile["height"]) == (768, 768)
    assert profile["steps"] == 20
    assert profile["guidance"] == pytest.approx(4.0)
    assert profile["seed_pool"] == [42, 137, 891, 2048, 7341]
    assert profile["zoom"] == "disabled"


def test_profile_expands_home_in_lora_path():
    profile = load_generation_profile()
    assert "$HOME" not in profile["lora"]["path"]
    assert str(Path.home()) in profile["lora"]["path"]


def test_invalid_provider_is_rejected():
    with pytest.raises(GenerationProfileError, match="provider"):
        validate_generation_profile({
            "provider": "fal_ai",
            "model_id": "qwen-image",
            "width": 768,
            "height": 768,
            "steps": 20,
            "guidance": 4.0,
            "seed_pool": [42],
            "postprocess": {
                "logical_size": [192, 192],
                "output_size": [768, 768],
                "colors": 32,
            },
            "zoom": "disabled",
        })


def test_fractional_grid_is_rejected():
    profile = load_generation_profile()
    profile["postprocess"]["output_size"] = [770, 768]
    with pytest.raises(GenerationProfileError, match="integer multiple"):
        validate_generation_profile(profile)


def test_unknown_profile_is_rejected(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"active_profile": "missing", "profiles": {}}))
    with pytest.raises(GenerationProfileError, match="Unknown generation profile"):
        load_generation_profile(path=path)
