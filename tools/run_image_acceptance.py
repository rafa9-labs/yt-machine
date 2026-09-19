"""Generate a real-script image acceptance corpus.

This intentionally calls the same ``generate_pixel_art`` path as the video
pipeline. The provider's cache name is prompt-based, so each scene/seed result
is copied immediately into a unique acceptance directory.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def parse_seeds(raw: str) -> list[int]:
    """Parse a comma-separated seed list and reject malformed input."""
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one seed is required")
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"seeds must be comma-separated integers: {raw!r}") from exc


def select_scenes(payload: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """Select the first real visual scenes from a saved pipeline script."""
    scenes = payload.get("all_visual_scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("script_segments.json has no all_visual_scenes list")
    if count < 1:
        raise ValueError("scene count must be positive")
    if len(scenes) < count:
        raise ValueError(f"script contains {len(scenes)} visual scenes; {count} required")

    selected = []
    for index, raw_scene in enumerate(scenes[:count], 1):
        if isinstance(raw_scene, str):
            scene = {"scene": f"scene_{index}", "description": raw_scene}
        elif isinstance(raw_scene, dict):
            scene = dict(raw_scene)
        else:
            raise ValueError(f"visual scene {index} is not an object or string")
        if not str(scene.get("description", "")).strip():
            raise ValueError(f"visual scene {index} has no description")
        scene.setdefault("scene", f"scene_{index}")
        selected.append(scene)
    return selected


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return name[:60] or "scene"


@dataclass(frozen=True)
class LoraVariant:
    """One adapter choice in a single-profile or comparison run."""

    label: str
    path: str | None
    scale: float | None
    spec: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "scale": self.scale,
            "adapter": self.spec.to_dict() if self.spec is not None else None,
        }


def parse_lora_values(raw_values: Iterable[str]) -> list[str]:
    """Parse repeatable ``--lora`` values, accepting comma-separated paths."""
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    values: list[str] = []
    for raw in raw_values:
        values.extend(part.strip() for part in str(raw).split(",") if part.strip())
    return values


def _lora_variant_from_value(profile: dict[str, Any], raw: str) -> LoraVariant:
    from src.video.generation_profile import model_family
    from src.video.lora_registry import describe_lora, lora_matches_model

    if raw.strip().lower() in {"none", "off", "disabled"}:
        return LoraVariant(label="none", path=None, scale=None)

    spec = describe_lora(Path(raw).expanduser())
    if spec.error:
        raise ValueError(f"LoRA {raw!r} cannot be used: {spec.error}")
    compatible, reason = lora_matches_model(spec, model_family(profile))
    if not compatible:
        raise ValueError(f"LoRA {raw!r} is incompatible: {reason}")

    active_lora = profile.get("lora") or {}
    scale = float(active_lora.get("scale", 1.0))
    return LoraVariant(
        label=_safe_name(spec.name),
        path=str(spec.path),
        scale=scale,
        spec=spec,
    )


def build_lora_variants(
    profile: dict[str, Any],
    requested: Iterable[str] = (),
    *,
    compare: bool = False,
) -> list[LoraVariant]:
    """Resolve requested adapters without changing the saved profile."""
    values = parse_lora_values(requested)
    if not values:
        active_lora = profile.get("lora") or {}
        active_path = active_lora.get("path")
        values = [str(active_path)] if active_path else ["none"]

    if compare and len(values) < 2:
        raise ValueError("--compare requires at least two --lora choices")
    if not compare and len(values) > 1:
        raise ValueError("pass --compare when running more than one LoRA choice")

    variants: list[LoraVariant] = []
    seen: set[str] = set()
    label_counts: dict[str, int] = {}
    for raw in values:
        variant = _lora_variant_from_value(profile, raw)
        key = variant.path or "__none__"
        if key in seen:
            raise ValueError(f"duplicate LoRA choice: {raw}")
        seen.add(key)
        label_counts[variant.label] = label_counts.get(variant.label, 0) + 1
        if label_counts[variant.label] > 1:
            variant = LoraVariant(
                label=f"{variant.label}-{label_counts[variant.label]}",
                path=variant.path,
                scale=variant.scale,
                spec=variant.spec,
            )
        variants.append(variant)
    return variants


def _variant_profile(profile: dict[str, Any], variant: LoraVariant) -> dict[str, Any]:
    """Create an in-memory profile with one adapter choice applied."""
    selected = copy.deepcopy(profile)
    selected.pop("name", None)
    if variant.path is None:
        selected["lora"] = None
        return selected

    previous = selected.get("lora") or {}
    adapter = variant.spec
    selected["lora"] = {
        "name": adapter.name if adapter is not None else variant.label,
        "path": variant.path,
        "scale": variant.scale if variant.scale is not None else 1.0,
        "trigger": (
            adapter.trigger if adapter is not None and adapter.trigger
            else previous.get("trigger", "Pixel Art")
        ),
    }
    if adapter is not None:
        selected["lora"].update({
            "base_family": adapter.base_family,
            "rank": adapter.rank,
        })
    return selected


@contextlib.contextmanager
def _use_profile_file(path: Path) -> Iterator[None]:
    """Temporarily point profile loading at a comparison-only JSON file."""
    previous = os.environ.get("YT_GENERATION_PROFILES_PATH")
    os.environ["YT_GENERATION_PROFILES_PATH"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("YT_GENERATION_PROFILES_PATH", None)
        else:
            os.environ["YT_GENERATION_PROFILES_PATH"] = previous


def _copy_result(
    result: dict[str, Any],
    target: Path,
    scene_index: int,
    scene_name: str,
    seed: int,
    source_script: Path,
    variant: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Copy one production result and attach acceptance metadata."""
    from PIL import Image
    from src.video.postprocess import count_colors, write_provenance

    source = Path(result["path"])
    shutil.copy2(source, target)

    raw_target = target.with_name(f"{target.stem}__raw.png")
    raw_source = result.get("raw_path")
    if raw_source and Path(raw_source).exists():
        shutil.copy2(raw_source, raw_target)

    provenance = result.get("provenance_path")
    if provenance and Path(provenance).exists():
        payload = json.loads(Path(provenance).read_text(encoding="utf-8"))
    else:
        payload = {"production_result": result}
    payload["acceptance"] = {
        "scene_index": scene_index,
        "scene": scene_name,
        "seed": seed,
        "source_script": str(source_script),
        "variant": variant,
    }
    provenance_target = write_provenance(target, payload)

    with Image.open(target) as image:
        size = list(image.size)
        colors = count_colors(image)

    return {
        "scene_index": scene_index,
        "scene": scene_name,
        "seed": seed,
        "path": str(target),
        "raw_path": str(raw_target) if raw_target.exists() else None,
        "provenance_path": str(provenance_target),
        "size": size,
        "colors": colors,
        "postprocess": result.get("postprocess"),
        "provider": result.get("provider"),
        "model": result.get("model"),
        "generation_profile": result.get("generation_profile"),
        "steps": result.get("steps"),
        "guidance": result.get("guidance"),
        "lora_scale": (result.get("postprocess") or {}).get("lora_scale"),
        "variant": variant,
    }


def run(args: argparse.Namespace) -> int:
    from src.video.generation_profile import (
        load_generation_profile,
        model_match_key,
        resolve_profile_model,
        set_overrides,
    )
    from src.video.mlxgen_provider import MLXGenImageProvider
    from src.video.pixel_art_tool import generate_pixel_art, set_mlxgen_provider

    source_script = Path(args.script).expanduser().resolve()
    if not source_script.exists():
        raise ValueError(f"script file not found: {source_script}")
    payload = json.loads(source_script.read_text(encoding="utf-8"))
    scenes = select_scenes(payload, args.scenes)
    seeds = parse_seeds(args.seeds)
    compare = bool(getattr(args, "compare", False))

    if args.guidance is not None:
        set_overrides({"guidance": args.guidance})
    profile = load_generation_profile()

    # The acceptance run uses whichever model the active generation profile
    # names, so it can validate any profile rather than only Qwen.
    resolved = resolve_profile_model(profile)
    if resolved["spec"] is None or resolved["spec"].provider != "mlxgen":
        raise ValueError("the active generation profile does not resolve to an "
                         "MLX-Gen image model")

    executable = (resolved["spec"].metadata or {}).get("executable") or None
    provider = MLXGenImageProvider(
        model_path=resolved["path"],
        **({"executable": executable} if executable else {}),
    )
    if not provider.available():
        raise ValueError(f"MLX-Gen unavailable: {provider.missing_reason()}")

    variants = build_lora_variants(
        profile,
        getattr(args, "lora", []),
        compare=compare,
    )

    # Default the corpus directory to the profile that produced it, so runs
    # against different models do not overwrite each other.
    output_arg = args.output or f"output/acceptance/{profile['name']}"
    output_dir = Path(output_arg).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if compare:
        output_dir = output_dir / "compare"
    output_dir.mkdir(parents=True, exist_ok=True)
    script_text_path = source_script.with_name("script.txt")
    script_text = script_text_path.read_text(encoding="utf-8") if script_text_path.exists() else ""

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    report = output_dir / (
        "comparison_report.json" if compare else "acceptance_report.json"
    )
    status = "running"
    error = None
    try:
        with tempfile.TemporaryDirectory(prefix="yt-machine-lora-") as temp_dir:
            temp_root = Path(temp_dir)
            for variant_index, variant in enumerate(variants, 1):
                variant_dir = output_dir / variant.label if compare else output_dir
                variant_dir.mkdir(parents=True, exist_ok=True)
                variant_profile = _variant_profile(profile, variant)
                profile_file = temp_root / f"profile_{variant_index}.json"
                profile_file.write_text(
                    json.dumps({
                        "version": 1,
                        "active_profile": profile["name"],
                        "profiles": {profile["name"]: variant_profile},
                    }, indent=2),
                    encoding="utf-8",
                )

                with _use_profile_file(profile_file):
                    if variant.path:
                        provider.set_loras([variant.path], [variant.scale or 1.0])
                    else:
                        provider.set_loras([], [])
                    set_mlxgen_provider(provider)

                    total = len(scenes) * len(seeds)
                    for scene_index, scene in enumerate(scenes, 1):
                        scene_name = str(scene.get("scene") or f"scene_{scene_index}")
                        prompt = str(scene["description"]).strip()
                        for seed in seeds:
                            completed = len(
                                [
                                    record
                                    for record in records
                                    if record["variant"]["label"] == variant.label
                                ]
                            ) + 1
                            print(
                                f"[{variant.label} {completed}/{total}] "
                                f"{scene_name} seed={seed}",
                                flush=True,
                            )
                            image_started = time.monotonic()
                            result = generate_pixel_art(
                                prompt, script_text=script_text, seed=seed
                            )
                            if not result.get("success"):
                                raise RuntimeError(
                                    f"generation failed for {variant.label} / "
                                    f"{scene_name} seed {seed}: "
                                    f"{result.get('error', 'unknown error')}"
                                )
                            if result.get("detected_failure"):
                                raise RuntimeError(
                                    f"quality check failed for {variant.label} / "
                                    f"{scene_name} seed {seed}: "
                                    f"{result['detected_failure']}"
                                )
                            target = variant_dir / (
                                f"{scene_index:02d}_{_safe_name(scene_name)}_seed{seed}.png"
                            )
                            record = _copy_result(
                                result,
                                target,
                                scene_index,
                                scene_name,
                                seed,
                                source_script,
                                variant=variant.to_dict(),
                            )
                            record["lora_scale"] = variant.scale
                            record["duration_s"] = round(
                                time.monotonic() - image_started, 2
                            )
                            records.append(record)
                            print(
                                f"  OK {record['size'][0]}x{record['size'][1]}, "
                                f"{record['colors']} colors, {record['duration_s']}s",
                                flush=True,
                            )
        status = "complete"
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        report.write_text(
            json.dumps(
                {
                    "status": status,
                    "error": error,
                    "source_script": str(source_script),
                    "profile": profile["name"],
                    "model": model_match_key(profile),
                    "scenes_requested": len(scenes),
                    "seeds": seeds,
                    "guidance_override": args.guidance,
                    "compare": compare,
                    "variants": [variant.to_dict() for variant in variants],
                    "expected_images": len(scenes) * len(seeds) * len(variants),
                    "completed_images": len(records),
                    "duration_s": round(time.monotonic() - started, 2),
                    "images": records,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, help="real script_segments.json")
    parser.add_argument(
        "--output", default=None,
        help="corpus output directory (default: output/acceptance/<active profile>)",
    )
    parser.add_argument("--scenes", type=int, default=5, help="number of real scenes")
    parser.add_argument(
        "--seeds", default="42,137,891", help="comma-separated seeds (default: 3)"
    )
    parser.add_argument(
        "--guidance", type=float, default=None,
        help="optional one-run guidance override; baseline uses the profile value",
    )
    parser.add_argument(
        "--lora", action="append", default=[], metavar="PATH|none",
        help=("adapter choice; repeat or comma-separate for --compare. "
              "Use 'none' for the base model"),
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="render every --lora choice with identical scenes and seeds",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
