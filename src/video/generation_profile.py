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

# NOTE: there is deliberately no hardcoded model constant. A profile names its
# model via `model.match`, resolved against discovered MLX-Gen checkpoints by
# resolve_profile_model(). Pinning one model id here would reintroduce the lock
# that made swapping models impossible without editing code.

# Fields a caller may override at runtime, and the coercion to apply.
# Deliberately narrow: these are the sampling knobs that are safe to vary for
# a single experiment. Structure (postprocess grid, zoom policy) is excluded
# because changing it silently alters the output contract.
OVERRIDABLE_FIELDS = {
    "width": int,
    "height": int,
    "steps": int,
    "guidance": float,
    "lora.scale": float,
}


class GenerationProfileError(RuntimeError):
    """Raised when the selected generation profile is unusable."""


# Process-wide overrides applied on every load.
#
# WHY A MODULE-LEVEL SETTER rather than a parameter: the profile is loaded at
# two independent call sites in the same process — the pipeline (for validation
# and LoRA setup) and pixel_art_tool (per image). Threading an argument through
# both would change a public signature for what is a run-scoped experiment.
#
# Overrides are NEVER written to disk, so an ad-hoc experiment cannot change
# the scheduled daily run.
_OVERRIDES: Dict[str, Any] = {}


def set_overrides(overrides: Optional[Dict[str, Any]] = None) -> None:
    """Set process-wide profile overrides. Pass None or {} to clear."""
    _OVERRIDES.clear()
    if overrides:
        _OVERRIDES.update(overrides)


def get_overrides() -> Dict[str, Any]:
    """Return a copy of the active overrides."""
    return dict(_OVERRIDES)


def parse_override(raw: str) -> tuple:
    """Parse 'key=value' into a validated (key, coerced_value) pair.

    Raises GenerationProfileError for an unknown key or an uncoercible value so
    a typo is reported immediately rather than silently ignored.
    """
    if "=" not in raw:
        raise GenerationProfileError(
            f"Override {raw!r} must be in key=value form"
        )
    key, _, value = raw.partition("=")
    key = key.strip()
    value = value.strip()

    if key not in OVERRIDABLE_FIELDS:
        allowed = ", ".join(sorted(OVERRIDABLE_FIELDS))
        raise GenerationProfileError(
            f"Cannot override {key!r}; allowed overrides: {allowed}"
        )
    try:
        coerced = OVERRIDABLE_FIELDS[key](value)
    except (TypeError, ValueError) as exc:
        raise GenerationProfileError(
            f"Override {key}={value!r} is not a valid "
            f"{OVERRIDABLE_FIELDS[key].__name__}"
        ) from exc
    return key, coerced


def _apply_overrides(profile: Dict[str, Any]) -> None:
    """Apply active overrides in place, supporting one level of nesting."""
    for key, value in _OVERRIDES.items():
        if "." in key:
            section, _, field = key.partition(".")
            target = profile.get(section)
            if not isinstance(target, dict):
                raise GenerationProfileError(
                    f"Cannot override {key}: profile has no {section!r} object"
                )
            target[field] = value
        else:
            profile[key] = value


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


def model_match_key(profile: Dict[str, Any]) -> str:
    """Return the substring used to resolve this profile's model.

    Accepts both the structured form and the legacy bare ``model_id``, so
    profiles written before the model block existed keep loading.
    """
    model = profile.get("model")
    if isinstance(model, dict):
        return str(model.get("match") or "").strip()
    # Legacy form: model_id held a repo id (qwen-image-2512-4bit) or a path.
    legacy = str(profile.get("model_id") or "").strip()
    return legacy.rsplit("/", 1)[-1] if legacy else ""


def model_family(profile: Dict[str, Any]) -> str:
    """Declared model family, for provenance and log messages."""
    model = profile.get("model")
    if isinstance(model, dict) and model.get("family"):
        return str(model["family"])
    return str(profile.get("family") or "")


def model_display_name(profile: Dict[str, Any]) -> str:
    """Human-readable model name for provenance and logs.

    A HuggingFace cache path resolves to a snapshot hash, which tells an
    operator nothing, so the repo directory is used instead when the path is
    an HF cache layout. Local checkpoint folders keep their own name.
    """
    resolved = str(profile.get("resolved_model_path") or "")
    if resolved:
        path = Path(resolved)
        # .../hub/models--<org>--<name>/snapshots/<hash> -> "<org>/<name>"
        parts = path.parts
        for index, part in enumerate(parts):
            if part.startswith("models--"):
                repo = part.replace("models--", "", 1).replace("--", "/")
                return repo
        return path.name
    return model_match_key(profile)


def validate_generation_profile(profile: Dict[str, Any]) -> None:
    """Validate a generation profile STRUCTURALLY.

    Deliberately does not touch the model registry. This runs in the
    interactive editor and in unit tests, both of which must work on a machine
    where no image model has been downloaded yet — a profile is a description,
    and whether the model it names is present is a separate question answered
    by resolve_profile_model() at pipeline start.
    """
    required = ("provider", "width", "height", "steps",
                "guidance", "seed_pool", "postprocess")
    missing = [key for key in required if key not in profile]
    if missing:
        raise GenerationProfileError(
            "Generation profile missing required fields: " + ", ".join(missing)
        )

    if profile["provider"] != "mlxgen":
        raise GenerationProfileError(
            f"Unsupported image provider {profile['provider']!r}; "
            "local image generation requires mlxgen"
        )

    if not model_match_key(profile):
        raise GenerationProfileError(
            "Profile must name a model, via model.match or model_id"
        )

    for key in ("width", "height", "steps"):
        try:
            value = int(profile[key])
        except (TypeError, ValueError) as exc:
            raise GenerationProfileError(f"{key} must be an integer") from exc
        if value <= 0:
            raise GenerationProfileError(f"{key} must be positive")

    # Dimension constraints are model-specific: mlxgen reports a
    # dimension_multiple per route that differs between families. The profile
    # declares the constraint its model requires so a typo is caught here,
    # without a registry lookup (this runs in the editor and in tests, where
    # no model need be installed). resolve_profile_model() additionally
    # verifies the declared value against the model's own capability report.
    declared_multiple = profile.get("dimension_multiple")
    if declared_multiple is not None:
        try:
            declared_multiple = int(declared_multiple)
        except (TypeError, ValueError) as exc:
            raise GenerationProfileError(
                "dimension_multiple must be an integer"
            ) from exc
        if declared_multiple <= 0:
            raise GenerationProfileError("dimension_multiple must be positive")
        for key in ("width", "height"):
            value = int(profile[key])
            if value % declared_multiple:
                raise GenerationProfileError(
                    f"{key} must be a multiple of {declared_multiple} "
                    f"(got {value})"
                )

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


def resolve_profile_model(
    profile: Dict[str, Any],
    *,
    probe_capabilities: bool = False,
) -> Dict[str, Any]:
    """Resolve a profile's model to a discovered MLX-Gen checkpoint.

    Returns a dict describing the resolution so callers do not need to know
    about the registry:

        {"spec": ModelSpec, "path": str, "id": str}

    Raises GenerationProfileError when nothing matches, naming the models that
    were actually found so the message is actionable.

    ``probe_capabilities`` additionally checks the model's own dimension
    constraint when the runtime can report one. It is off by default because
    probing launches the mlxgen CLI, which is far too slow to do per image.
    """
    from src.models.registry import (
        PROVIDER_MLXGEN,
        list_mlxgen_models,
        resolve_mlxgen_model,
    )

    match = model_match_key(profile)
    if not match:
        raise GenerationProfileError(
            "Profile does not name a model (model.match or model_id)"
        )

    spec = resolve_mlxgen_model(match)
    if spec is None:
        available = list_mlxgen_models()
        if available:
            names = "\n    ".join(s.id for s in available)
            hint = f"\n  Discovered MLX-Gen models:\n    {names}"
        else:
            hint = ("\n  No MLX-Gen models were discovered. Check YT_MODEL_ROOTS "
                    "or download a checkpoint.")
        raise GenerationProfileError(
            f"No MLX-Gen image model matches {match!r}.{hint}"
        )

    if spec.provider != PROVIDER_MLXGEN:
        raise GenerationProfileError(
            f"Resolved {spec.id!r} is provider {spec.provider!r}, not mlxgen"
        )

    if probe_capabilities:
        _enforce_dimension_multiple(profile, spec)

    return {"spec": spec, "path": spec.path or spec.id, "id": spec.id}


def _enforce_dimension_multiple(profile: Dict[str, Any], spec) -> None:
    """Reject dimensions the model cannot produce.

    mlxgen reports ``dimension_multiple`` per route. Some families require a
    multiple (Qwen's latent route reports 16) and others report none, in which
    case no constraint is applied rather than inventing one.
    """
    from src.video.mlxgen_provider import MLXGenImageProvider

    executable = (spec.metadata or {}).get("executable") or os.getenv("MLXGEN_BIN")
    provider = MLXGenImageProvider(model_path=spec.path or spec.id,
                                   executable=executable)
    caps = provider.capabilities()
    multiple = caps.get("dimension_multiple") if caps else None
    if not multiple:
        return
    multiple = int(multiple)
    for key in ("width", "height"):
        value = int(profile[key])
        if value % multiple:
            raise GenerationProfileError(
                f"{key} must be a multiple of {multiple} for this model "
                f"(got {value})"
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
    _apply_overrides(selected)
    validate_generation_profile(selected)
    return selected


def describe_profiles(path: Optional[Path] = None) -> list:
    """Return a summary of every profile in the file, for CLI listing.

    Validation failures are reported per profile rather than raised, so one
    malformed profile does not hide the others from an operator.
    """
    source = profile_path(path)
    if not source.exists():
        raise GenerationProfileError(f"Generation profile not found at {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationProfileError(
            f"Generation profile is unreadable: {exc}"
        ) from exc

    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, dict):
        raise GenerationProfileError(
            "Generation profile file must contain a profiles object"
        )

    active = data.get("active_profile")
    summaries = []
    for name, raw in sorted(profiles.items()):
        entry = {"name": name, "active": name == active, "valid": True,
                 "error": None}
        try:
            selected = _expand_profile_paths(raw)
            validate_generation_profile(selected)
            lora = selected.get("lora") or {}
            entry.update({
                "provider": selected.get("provider"),
                "model_id": model_match_key(selected),
                "model_family": model_family(selected),
                "width": selected.get("width"),
                "height": selected.get("height"),
                "steps": selected.get("steps"),
                "guidance": selected.get("guidance"),
                "lora_name": lora.get("name"),
                "lora_scale": lora.get("scale"),
                "zoom": selected.get("zoom"),
            })
        except GenerationProfileError as exc:
            entry["valid"] = False
            entry["error"] = str(exc)
        summaries.append(entry)
    return summaries
