"""Tests for the exposed Qwen-only image generation tool."""

from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.video import pixel_art_tool


def _profile():
    return {
        "name": "qwen_pixel_scene",
        "provider": "mlxgen",
        "model_id": "AbstractFramework/qwen-image-2512-4bit",
        "width": 768,
        "height": 768,
        "steps": 20,
        "guidance": 4.0,
        "lora": {"trigger": "Pixel Art"},
        "seed_pool": [42, 137, 891, 2048, 7341],
        "postprocess": {
            "logical_size": [192, 192],
            "output_size": [768, 768],
            "colors": 32,
        },
        "zoom": "disabled",
        "provenance": True,
    }


class FakeQwenProvider:
    lora_paths = []

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        image = Image.new("RGB", (kwargs["width"], kwargs["height"]))
        pixels = image.load()
        palette = [(i * 17 % 256, i * 43 % 256, i * 71 % 256) for i in range(64)]
        for y in range(image.height):
            for x in range(image.width):
                pixels[x, y] = palette[((x // 4) + (y // 4)) % len(palette)]
        image.save(kwargs["output_path"])
        return {"success": True, "provider": "mlxgen"}


def test_qwen_tool_generates_processed_asset_and_provenance(tmp_path):
    provider = FakeQwenProvider()
    with patch.object(pixel_art_tool, "OUTPUT_DIR", tmp_path), \
         patch.object(pixel_art_tool, "_MLXGEN_PROVIDER", provider), \
         patch(
             "src.video.generation_profile.load_generation_profile",
             return_value=_profile(),
         ):
        result = pixel_art_tool.generate_pixel_art(
            "A radar tower in the desert at dusk.",
            script_text="A radar tower tracks activity in the desert.",
            seed=891,
        )

    assert result["success"] is True
    assert result["source"] == "qwen"
    assert result["provider"] == "mlxgen"
    assert result["steps"] == 20
    assert result["guidance"] == 4.0
    assert result["seed"] == 891
    assert Path(result["raw_path"]).exists()
    assert Path(result["path"]).exists()
    assert Path(result["provenance_path"]).exists()
    assert result["postprocess"]["colors_written"] <= 32
    assert provider.calls[0]["width"] == 768
    assert provider.calls[0]["height"] == 768
    assert provider.calls[0]["negative_prompt"]
    assert "tactical map layout" not in provider.calls[0]["prompt"].lower()


def test_qwen_tool_fails_closed_without_provider():
    with patch.object(pixel_art_tool, "_MLXGEN_PROVIDER", None):
        result = pixel_art_tool.generate_pixel_art("A radar tower.")
    assert result["success"] is False
    assert "provider is not configured" in result["error"]


def test_qwen_tool_rejects_reference_images():
    result = pixel_art_tool.generate_pixel_art(
        "A radar tower.", reference_image="reference.png"
    )
    assert result["success"] is False
    assert "text-to-image only" in result["error"]
