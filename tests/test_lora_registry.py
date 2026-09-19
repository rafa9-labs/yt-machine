"""Tests for adapter discovery and base-family compatibility checks."""

import json

from src.video.lora_registry import (
    LoraSpec,
    describe_lora,
    discover_loras,
    lora_matches_model,
    normalize_family,
)


def test_normalize_family_handles_qwen_and_flux_variants():
    assert normalize_family("qwen_image") == "qwen-image"
    assert normalize_family("black-forest-labs/FLUX.2-Klein-9B") == "flux2-klein"
    assert normalize_family("mlx-gen") == "mlx-gen"


def test_describe_lora_reads_metadata_without_loading_a_base_model(monkeypatch, tmp_path):
    import src.video.lora_registry as registry

    path = tmp_path / "redmond.safetensors"
    path.write_bytes(b"not a real tensor file")

    class FakeHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def metadata(self):
            return {
                "ss_base_model_version": "qwen_image",
                "ss_network_dim": "32",
                "ss_tag_frequency": json.dumps({"1_Pixel Art, PixArFK": {"x": 1}}),
            }

        def keys(self):
            return ["diffusion_model.block.lora_A.weight"]

        def get_tensor(self, _key):
            raise AssertionError("metadata rank should avoid reading tensor data")

    monkeypatch.setattr(registry, "safe_open", lambda *_args, **_kwargs: FakeHandle())

    spec = describe_lora(path)

    assert spec.name == "redmond"
    assert spec.base_family == "qwen-image"
    assert spec.rank == 32
    assert spec.tensor_count == 1
    assert spec.trigger == "Pixel Art, PixArFK"
    assert spec.error is None


def test_discover_loras_only_returns_safetensors(tmp_path):
    (tmp_path / "adapter.safetensors").write_bytes(b"invalid")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    specs = discover_loras([tmp_path])

    assert [spec.name for spec in specs] == ["adapter"]


def test_lora_matches_model_rejects_wrong_family():
    spec = LoraSpec(
        path="/tmp/flux.safetensors",
        name="flux",
        base_family="flux2-klein",
    )

    compatible, reason = lora_matches_model(spec, "qwen-image")

    assert compatible is False
    assert "flux2-klein" in reason
    assert "qwen-image" in reason


def test_lora_without_family_is_allowed_with_warning():
    spec = LoraSpec(path="/tmp/unknown.safetensors", name="unknown")

    compatible, reason = lora_matches_model(spec, "qwen-image")

    assert compatible is True
    assert "does not declare" in reason
