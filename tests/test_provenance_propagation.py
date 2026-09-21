"""Tests for provenance propagation into the project folder.

The generator writes a sidecar next to the scratch PNG in output/images/.
This file covers the step that carries it into the per-project folder, which
is the artifact that gets archived — the gap that made provenance invisible
even though it was being written correctly.
"""

import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.video import pixel_art_tool
from src.video.postprocess import read_provenance, write_provenance
from src.video.provenance import find_secrets


# ══════════════════════════════════════════════════════════════
# Propagation contract
# ══════════════════════════════════════════════════════════════
class TestPropagationContract:
    """The invariants the pipeline's copy step must satisfy."""

    def test_sidecar_follows_the_renamed_image(self, tmp_path):
        """Simulates: scratch PNG + sidecar -> renamed project image + sidecar."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        source_image = scratch / "scene_raw_hash123.png"
        Image.new("RGB", (8, 8), (1, 2, 3)).save(source_image)
        source_sidecar = write_provenance(
            source_image,
            {"model": "qwen-image-2512-4bit", "sampling": {"seed": 42}},
        )
        assert source_sidecar.exists()

        project_images = tmp_path / "project" / "images"
        project_images.mkdir(parents=True)
        dest_image = project_images / f"story_1_part1_{source_image.name}"

        # This is what tools/generate_complete_video.py now does.
        payload = read_provenance(source_sidecar) or {}
        payload["project_id"] = 123
        payload["scene"] = "story_1_part1"
        payload["processed_output"] = str(dest_image)
        dest_sidecar = write_provenance(dest_image, payload)

        assert dest_sidecar.parent == project_images
        assert dest_sidecar.name == (
            "story_1_part1_scene_raw_hash123.provenance.json"
        )
        reloaded = read_provenance(dest_image)
        assert reloaded["scene"] == "story_1_part1"
        assert reloaded["project_id"] == 123
        assert reloaded["model"] == "qwen-image-2512-4bit"
        assert reloaded["processed_output"] == str(dest_image)

    def test_read_provenance_handles_missing_source(self, tmp_path):
        """A generator that wrote no sidecar must not break the copy step."""
        assert read_provenance(tmp_path / "never_written.png") is None
        payload = read_provenance(tmp_path / "never_written.png") or {}
        assert payload == {}

    def test_propagated_payload_stays_secret_free(self, tmp_path):
        dest = tmp_path / "images" / "scene.png"
        dest.parent.mkdir(parents=True)
        write_provenance(
            dest,
            {
                "model": "qwen",
                "api_key": "must-not-persist",
                "lora": {"name": "adapter", "path": "/models/loras/x.safetensors"},
            },
        )
        raw = (tmp_path / "images" / "scene.provenance.json").read_text()
        # The secret value must not survive. The api_key *name* remains as a
        # redaction marker, which is intentional: the record stays honest
        # about what the caller tried to include.
        assert "must-not-persist" not in raw
        persisted = json.loads(raw)
        assert persisted["api_key"] == "[REDACTED]"
        assert find_secrets(persisted) == ["api_key"]
        assert persisted["lora"]["path"] == "/models/loras/x.safetensors"


# ══════════════════════════════════════════════════════════════
# The provenance flag
# ══════════════════════════════════════════════════════════════
def _profile(provenance=True):
    profile = {
        "name": "qwen_pixel_scene",
        "provider": "mlxgen",
        "model": {"match": "qwen-image-2512-4bit", "family": "qwen-image"},
        "width": 64,
        "height": 64,
        "dimension_multiple": 16,
        "steps": 20,
        "guidance": 4.0,
        "lora": {"trigger": "Pixel Art"},
        "seed_pool": [42],
        "postprocess": {
            "logical_size": [192, 192],
            "output_size": [768, 768],
            "colors": 32,
        },
        "zoom": "disabled",
    }
    if provenance is not None:
        profile["provenance"] = provenance
    return profile


class _Provider:
    lora_paths = []

    def generate(self, **kwargs):
        Image.new("RGB", (kwargs["width"], kwargs["height"])).save(
            kwargs["output_path"]
        )
        return {"success": True, "provider": "mlxgen"}


def _run_generate(tmp_path, profile):
    with patch.object(pixel_art_tool, "OUTPUT_DIR", tmp_path), \
         patch.object(pixel_art_tool, "_MLXGEN_PROVIDER", _Provider()), \
         patch(
             "src.video.generation_profile.load_generation_profile",
             return_value=profile,
         ), \
         patch(
             "src.video.generation_profile.resolve_profile_model",
             side_effect=lambda p, **kw: {"spec": None, "path": "/m", "id": "/m"},
         ):
        return pixel_art_tool.generate_pixel_art("A radar tower.")


def test_provenance_true_writes_sidecar(tmp_path):
    result = _run_generate(tmp_path, _profile(True))
    assert result["success"] is True
    assert result["provenance_path"]
    assert Path(result["provenance_path"]).exists()


def test_provenance_false_writes_no_sidecar(tmp_path):
    result = _run_generate(tmp_path, _profile(False))
    assert result["success"] is True
    assert result["provenance_path"] is None
    assert not list(Path(tmp_path).glob("*.provenance.json"))


def test_provenance_absent_defaults_to_writing(tmp_path):
    """A profile predating the flag keeps the documented default."""
    result = _run_generate(tmp_path, _profile(None))
    assert result["success"] is True
    assert result["provenance_path"]
    assert Path(result["provenance_path"]).exists()
