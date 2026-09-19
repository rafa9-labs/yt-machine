"""Tests for the image generation profile contract.

The profile is validated structurally here, with no model required. Whether
the model a profile names is actually installed is a separate concern, covered
by tests/test_model_interchangeability.py.
"""

import json
from pathlib import Path

import pytest

from src.video.generation_profile import (
    GenerationProfileError,
    load_generation_profile,
    model_family,
    model_match_key,
    validate_generation_profile,
)


def test_active_profile_is_qwen_pixel_scene():
    profile = load_generation_profile()
    assert profile["name"] == "qwen_pixel_scene"
    assert profile["provider"] == "mlxgen"
    assert "qwen" in model_match_key(profile).lower()
    assert model_family(profile) == "qwen-image"
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
            "model": {"match": "qwen-image"},
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


def test_profile_without_a_model_is_rejected():
    """A profile must name a model, in either the new or legacy form."""
    with pytest.raises(GenerationProfileError, match="must name a model"):
        validate_generation_profile({
            "provider": "mlxgen",
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


def test_legacy_model_id_form_still_validates():
    """Profiles written before the model block existed keep loading."""
    validate_generation_profile({
        "provider": "mlxgen",
        "model_id": "AbstractFramework/qwen-image-2512-4bit",
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


def test_declared_dimension_multiple_is_enforced():
    """A profile declares its model's dimension constraint so typos fail here.

    This runs without a registry lookup, which is what lets the editor and the
    test suite work on a machine with no image model installed.
    """
    profile = load_generation_profile()
    profile["width"] = 100  # not a multiple of 16
    with pytest.raises(GenerationProfileError, match="multiple of 16"):
        validate_generation_profile(profile)


def test_profile_without_dimension_multiple_accepts_any_positive_size():
    """Models that report no constraint are not given an invented one."""
    profile = load_generation_profile()
    profile.pop("dimension_multiple", None)
    profile["width"] = 772  # not a multiple of 16
    validate_generation_profile(profile)


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


def test_both_shipped_profiles_validate():
    """Every profile in the shipped file must be structurally valid."""
    from src.video.generation_profile import describe_profiles

    for entry in describe_profiles():
        assert entry["valid"], f"{entry['name']}: {entry['error']}"
