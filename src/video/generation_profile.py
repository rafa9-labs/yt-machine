"""Load and validate the active image-generation profile.

The model role profile answers *which* model is used. This profile answers
*how* that model is sampled and how its output is prepared for video.
Keeping the two concerns separate makes every generated image reproducible.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "config"
    / "generation_profiles.json"
)
DEFAULT_PROFILE_NAME = "qwen_pixel_scene"
QWEN_IMAGE_MODEL_ID = "AbstractFramework/qwen-image-2512-4bit"


class GenerationProfileError(RuntimeError):
    """Raised when the selected generation profile is unusable."""


def profile_path(path: Optional[Path] = None) -> Path:
    """Return the configured generation-profile file path."""
    if path is not None:
        return Path(path)
    override = os.getenv("YT_GENERATION_PROFILES_PATH")
    return Path(override).expanduser() if override else DEFAULT_PROFILE_PATH


def _expand_profile_paths(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Expand shell-style paths without changing the JSON source."""
    result = copy.deepcopy(profile)
    lora = result.get("lora") or {}
    if lora.get("path"):
        lora["path"] = os.path.expandvars(os.path.expanduser(str(lora["path"])))
    result["lora"] = lora
    return result


def validate_generation_profile(profile: Dict[str, Any]) -> None:
    """Validate the invariants required by the Qwen production path."""
    required = ("provider", "model_id", "width", "height", "steps",
                "guidance", "seed_pool", "postprocess")
    missing = [key for key in required if key not in profile]
    if missing:
        raise GenerationProfileError(
            "Generation profile missing required fields: " + ", ".join(missing)
        )

    if profile["provider"] != "mlxgen":
        raise GenerationProfileError(
            f"Unsupported image provider {profile['provider']!r}; Qwen requires mlxgen"
        )
    if profile["model_id"] != QWEN_IMAGE_MODEL_ID:
        raise GenerationProfileError(
            f"Image model {profile['model_id']!r} is not the selected Qwen model "
            f"{QWEN_IMAGE_MODEL_ID!r}"
        )

    for key in ("width", "height", "steps"):
        try:
            value = int(profile[key])
        except (TypeError, ValueError) as exc:
            raise GenerationProfileError(f"{key} must be an integer") from exc
        if value <= 0:
            raise GenerationProfileError(f"{key} must be positive")

    try:
        guidance = float(profile["guidance"])
    except (TypeError, ValueError) as exc:
        raise GenerationProfileError("guidance must be numeric") from exc
    if guidance <= 0:
        raise GenerationProfileError("guidance must be positive")

    seeds = profile["seed_pool"]
    if not isinstance(seeds, list) or not seeds:
        raise GenerationProfileError("seed_pool must be a non-empty list")
    if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise GenerationProfileError("seed_pool must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise GenerationProfileError("seed_pool must not contain duplicates")

    postprocess = profile["postprocess"]
    if not isinstance(postprocess, dict):
        raise GenerationProfileError("postprocess must be an object")
    logical_size = postprocess.get("logical_size")
    output_size = postprocess.get("output_size")
    colors = postprocess.get("colors")
    if not (
        isinstance(logical_size, list)
        and len(logical_size) == 2
        and all(isinstance(value, int) and value > 0 for value in logical_size)
    ):
        raise GenerationProfileError("postprocess.logical_size must be [width, height]")
    if not (
        isinstance(output_size, list)
        and len(output_size) == 2
        and all(isinstance(value, int) and value > 0 for value in output_size)
    ):
        raise GenerationProfileError("postprocess.output_size must be [width, height]")
    if output_size[0] % logical_size[0] or output_size[1] % logical_size[1]:
        raise GenerationProfileError(
            "postprocess.output_size must be an integer multiple of logical_size"
        )
    if not isinstance(colors, int) or not 2 <= colors <= 256:
        raise GenerationProfileError("postprocess.colors must be between 2 and 256")

    lora = profile.get("lora") or {}
    if lora:
        if not lora.get("path"):
            raise GenerationProfileError("lora.path cannot be empty")
        try:
            scale = float(lora.get("scale", 1.0))
        except (TypeError, ValueError) as exc:
            raise GenerationProfileError("lora.scale must be numeric") from exc
        if scale < 0:
            raise GenerationProfileError("lora.scale must not be negative")

    if profile.get("zoom") != "disabled":
        raise GenerationProfileError(
            "Pixel-art generation profiles must disable continuous zoom"
        )


def load_generation_profile(
    name: Optional[str] = None,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load one named profile, expanding local asset paths."""
    source = profile_path(path)
    if not source.exists():
        raise GenerationProfileError(f"Generation profile not found at {source}")

    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationProfileError(f"Generation profile is unreadable: {exc}") from exc

    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, dict):
        raise GenerationProfileError("Generation profile file must contain a profiles object")

    requested = name or os.getenv("YT_GENERATION_PROFILE") or data.get("active_profile")
    requested = requested or DEFAULT_PROFILE_NAME
    if requested not in profiles:
        available = ", ".join(sorted(profiles)) or "none"
        raise GenerationProfileError(
            f"Unknown generation profile {requested!r}; available: {available}"
        )

    selected = _expand_profile_paths(profiles[requested])
    selected["name"] = requested
    validate_generation_profile(selected)
    return selected
