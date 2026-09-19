"""Generate the real-script Qwen image acceptance corpus.

This intentionally calls the same ``generate_pixel_art`` path as the video
pipeline. The provider's cache name is prompt-based, so each scene/seed result
is copied immediately into a unique acceptance directory.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

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


def _copy_result(
    result: dict[str, Any],
    target: Path,
    scene_index: int,
    scene_name: str,
    seed: int,
    source_script: Path,
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
    }


def run(args: argparse.Namespace) -> int:
    from src.models.profile import ModelProfile
    from src.video.generation_profile import (
        QWEN_IMAGE_MODEL_ID,
        load_generation_profile,
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

    if args.guidance is not None:
        set_overrides({"guidance": args.guidance})
    profile = load_generation_profile()
    model_profile = ModelProfile.load()
    image_model = model_profile.image
    if image_model is None or image_model.provider != "mlxgen":
        raise ValueError("config/model_profile.json does not select an MLX-Gen image model")
    if profile["model_id"] != QWEN_IMAGE_MODEL_ID:
        raise ValueError(f"acceptance requires {QWEN_IMAGE_MODEL_ID}, got {profile['model_id']}")

    executable = (image_model.metadata or {}).get("executable") or None
    provider = MLXGenImageProvider(
        model_path=image_model.path or image_model.id,
        **({"executable": executable} if executable else {}),
    )
    if not provider.available():
        raise ValueError(f"MLX-Gen unavailable: {provider.missing_reason()}")

    lora = profile.get("lora") or {}
    lora_path = lora.get("path")
    if lora_path:
        if not Path(lora_path).exists():
            raise ValueError(f"configured Qwen LoRA is missing: {lora_path}")
        provider.set_loras([lora_path], [float(lora.get("scale", 1.0))])
    set_mlxgen_provider(provider)

    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    script_text_path = source_script.with_name("script.txt")
    script_text = script_text_path.read_text(encoding="utf-8") if script_text_path.exists() else ""

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    report = output_dir / "acceptance_report.json"
    status = "running"
    error = None
    try:
        total = len(scenes) * len(seeds)
        for scene_index, scene in enumerate(scenes, 1):
            scene_name = str(scene.get("scene") or f"scene_{scene_index}")
            prompt = str(scene["description"]).strip()
            for seed in seeds:
                completed = len(records) + 1
                print(f"[{completed}/{total}] {scene_name} seed={seed}", flush=True)
                image_started = time.monotonic()
                result = generate_pixel_art(prompt, script_text=script_text, seed=seed)
                if not result.get("success"):
                    raise RuntimeError(
                        f"generation failed for {scene_name} seed {seed}: "
                        f"{result.get('error', 'unknown error')}"
                    )
                if result.get("detected_failure"):
                    raise RuntimeError(
                        f"quality check failed for {scene_name} seed {seed}: "
                        f"{result['detected_failure']}"
                    )
                target = output_dir / (
                    f"{scene_index:02d}_{_safe_name(scene_name)}_seed{seed}.png"
                )
                record = _copy_result(
                    result, target, scene_index, scene_name, seed, source_script
                )
                record["lora_scale"] = float(
                    (profile.get("lora") or {}).get("scale", 1.0)
                )
                record["duration_s"] = round(time.monotonic() - image_started, 2)
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
                    "model": profile["model_id"],
                    "scenes_requested": len(scenes),
                    "seeds": seeds,
                    "guidance_override": args.guidance,
                    "expected_images": len(scenes) * len(seeds),
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
        "--output", default="output/acceptance/qwen-2512", help="corpus output directory"
    )
    parser.add_argument("--scenes", type=int, default=5, help="number of real scenes")
    parser.add_argument(
        "--seeds", default="42,137,891", help="comma-separated seeds (default: 3)"
    )
    parser.add_argument(
        "--guidance", type=float, default=None,
        help="optional one-run guidance override; baseline uses the profile value",
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
