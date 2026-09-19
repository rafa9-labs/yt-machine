"""
Tests for image-model interchangeability.
=========================================

The image model used to be locked to a single hardcoded checkpoint. It is now
resolved from whatever the active generation profile names, so switching models
is a profile change rather than an edit in two files.

These tests cover the three things that make that safe:

  1. Structural validation stays registry-free, so the editor and the suite
     work on a machine with no image model installed.
  2. Resolution fails loudly and actionably when a profile names a model that
     is not present.
  3. Both shipped profiles resolve, and the model that gets used is the one the
     profile names — not whichever checkpoint happens to be largest.

Run: .venv/bin/python -m pytest tests/test_model_interchangeability.py -v
"""

from pathlib import Path

import pytest
from src.models.registry import (
    CAP_IMAGE,
    PROVIDER_MLXGEN,
    clear_mlxgen_cache,
    list_mlxgen_models,
    resolve_mlxgen_model,
    specs_for_capability,
    discover_all,
)
from src.video.generation_profile import (
    GenerationProfileError,
    load_generation_profile,
    model_display_name,
    model_family,
    model_match_key,
    resolve_profile_model,
    validate_generation_profile,
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Discovery is cached per process; start each test from a cold cache."""
    clear_mlxgen_cache()
    yield
    clear_mlxgen_cache()


# ── Structural validation must not need the registry ────────────────────────


class TestStructuralValidationIsOffline:
    def test_profile_naming_an_uninstalled_model_validates(self):
        """The editor must be able to describe a model that is not present.

        Validation answers "is this profile well-formed", not "is this model
        installed". Those are different questions asked at different times.
        """
        validate_generation_profile({
            "provider": "mlxgen",
            "model": {"match": "a-model-that-is-not-installed", "family": "test"},
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

    def test_both_shipped_profiles_are_structurally_valid(self):
        from src.video.generation_profile import describe_profiles

        summaries = describe_profiles()
        assert len(summaries) >= 2, "expected the Qwen and FLUX profiles"
        for entry in summaries:
            assert entry["valid"], f"{entry['name']}: {entry['error']}"


# ── Model identity is read from the profile, not a constant ─────────────────


class TestModelIdentity:
    def test_match_key_reads_the_model_block(self):
        assert model_match_key(
            {"model": {"match": "qwen-image-2512-4bit"}}
        ) == "qwen-image-2512-4bit"

    def test_match_key_accepts_the_legacy_model_id_form(self):
        """Profiles written before the model block existed keep working."""
        assert model_match_key(
            {"model_id": "AbstractFramework/qwen-image-2512-4bit"}
        ) == "qwen-image-2512-4bit"

    def test_family_is_reported_for_provenance(self):
        assert model_family({"model": {"family": "qwen-image"}}) == "qwen-image"

    def test_display_name_shortens_a_huggingface_cache_path(self):
        """A snapshot hash tells an operator nothing; the repo name does."""
        label = model_display_name({
            "resolved_model_path": (
                "/home/u/.cache/huggingface/hub/"
                "models--AbstractFramework--qwen-image-2512-4bit/"
                "snapshots/423f1f5bf708c6e11eb78881ef9738422cea0814"
            )
        })
        assert label == "AbstractFramework/qwen-image-2512-4bit"

    def test_display_name_keeps_a_local_folder_name(self):
        label = model_display_name({
            "resolved_model_path": "/models/flux2-klein-base-9b-uncensored-8bit"
        })
        assert label == "flux2-klein-base-9b-uncensored-8bit"


# ── Resolution against discovered models ────────────────────────────────────


class TestResolution:
    def test_active_profile_resolves_to_a_real_checkpoint(self):
        result = resolve_profile_model(load_generation_profile())
        assert result["path"]
        assert Path(result["path"]).exists()

    def test_flux_profile_resolves_to_the_flux_checkpoint(self):
        profile = load_generation_profile("flux_klein_pixel_scene")
        result = resolve_profile_model(profile)
        assert "flux" in result["path"].lower()

    def test_qwen_profile_resolves_to_the_qwen_checkpoint(self):
        profile = load_generation_profile("qwen_pixel_scene")
        result = resolve_profile_model(profile)
        assert "qwen" in result["path"].lower()

    def test_each_profile_resolves_to_its_own_model(self):
        """A profile must get the model it names, not merely any model.

        This is the regression the old hardcoded lock prevented by refusing to
        run at all; capability-based filtering has to prevent it properly.
        """
        qwen = resolve_profile_model(load_generation_profile("qwen_pixel_scene"))
        flux = resolve_profile_model(load_generation_profile("flux_klein_pixel_scene"))
        assert qwen["path"] != flux["path"]

    def test_unknown_model_fails_and_lists_what_is_available(self):
        profile = load_generation_profile()
        profile.pop("model", None)
        profile["model"] = {"match": "definitely-not-installed"}

        with pytest.raises(GenerationProfileError) as excinfo:
            resolve_profile_model(profile)

        message = str(excinfo.value)
        assert "definitely-not-installed" in message
        # The message must be actionable, naming what IS present.
        for spec in list_mlxgen_models():
            assert spec.id in message

    def test_profile_without_a_model_is_rejected(self):
        with pytest.raises(GenerationProfileError, match="does not name a model"):
            resolve_profile_model({"provider": "mlxgen"})


# ── Resolution is cached, because it runs per image ─────────────────────────


class TestResolutionCaching:
    def test_second_resolution_does_not_re_discover(self):
        """generate_pixel_art resolves once per image; discovery must not repeat."""
        from unittest.mock import patch

        import src.models.registry as registry

        resolve_mlxgen_model("qwen-image-2512-4bit")  # warm

        with patch.object(registry, "discover_all",
                          side_effect=AssertionError("discovery was called again")):
            repeat = resolve_mlxgen_model("qwen-image-2512-4bit")

        assert repeat is not None

    def test_clearing_the_cache_forces_rediscovery(self):
        first = resolve_mlxgen_model("qwen-image-2512-4bit")
        clear_mlxgen_cache()
        second = resolve_mlxgen_model("qwen-image-2512-4bit")
        assert first.id == second.id


# ── Only local MLX-Gen image models are eligible ────────────────────────────


class TestProviderFiltering:
    def test_resolver_only_returns_mlxgen_specs(self):
        for spec in list_mlxgen_models():
            assert spec.provider == PROVIDER_MLXGEN
            assert spec.supports(CAP_IMAGE)

    def test_image_specs_are_not_text_models(self):
        """A text checkpoint must never resolve as the image model."""
        discovered = discover_all()
        for spec in specs_for_capability(discovered, CAP_IMAGE):
            if spec.provider == PROVIDER_MLXGEN:
                assert not spec.id.lower().endswith(".gguf")


# ── The FLUX profile's model-specific differences are recorded ──────────────


class TestFluxProfileDifferences:
    def test_flux_profile_declares_no_negative_prompt(self):
        """FLUX.2 Klein has no CFG branch, so a negative prompt is meaningless.

        The profile says so explicitly rather than relying on the provider to
        silently drop a flag the model would reject.
        """
        profile = load_generation_profile("flux_klein_pixel_scene")
        assert profile.get("negative_prompt", "unset") is None

    def test_flux_profile_declares_no_lora(self):
        """No FLUX-family adapter is installed; recording that is honest."""
        profile = load_generation_profile("flux_klein_pixel_scene")
        assert not profile.get("lora")

    def test_flux_profile_uses_fewer_steps_than_qwen(self):
        """8 steps is what this step-distilled checkpoint needs."""
        assert (load_generation_profile("flux_klein_pixel_scene")["steps"]
                < load_generation_profile("qwen_pixel_scene")["steps"])

    def test_flux_profile_omits_a_dimension_constraint(self):
        """The model reports none, so none is invented."""
        flux = load_generation_profile("flux_klein_pixel_scene")
        assert "dimension_multiple" not in flux

    def test_qwen_profile_declares_its_dimension_constraint(self):
        qwen = load_generation_profile("qwen_pixel_scene")
        assert qwen["dimension_multiple"] == 16
