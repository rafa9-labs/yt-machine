"""Deterministic pixel-art output preparation.

The diffusion model is allowed to produce a detailed RGB image. The final
scene asset is deliberately reduced to a logical pixel grid and a bounded
palette before it reaches the video assembler.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image


def _resampling(name: str):
    """Resolve Pillow's enum while remaining compatible with older Pillow."""
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, name)


def process_pixel_art(
    image: Image.Image,
    logical_size: Tuple[int, int] = (192, 192),
    output_size: Tuple[int, int] = (768, 768),
    colors: int = 32,
) -> Image.Image:
    """Return an RGB image with a fixed logical grid and bounded palette.

    BOX is used while reducing detail into the logical grid. Palette
    quantization happens at that small size, then NEAREST restores the output
    dimensions without inventing intermediate colours.
    """
    logical_w, logical_h = (int(logical_size[0]), int(logical_size[1]))
    output_w, output_h = (int(output_size[0]), int(output_size[1]))
    colors = int(colors)
    if logical_w <= 0 or logical_h <= 0:
        raise ValueError("logical_size must contain positive dimensions")
    if output_w <= 0 or output_h <= 0:
        raise ValueError("output_size must contain positive dimensions")
    if output_w % logical_w or output_h % logical_h:
        raise ValueError("output_size must be an integer multiple of logical_size")
    if not 2 <= colors <= 256:
        raise ValueError("colors must be between 2 and 256")

    rgb = image.convert("RGB")
    logical = rgb.resize((logical_w, logical_h), _resampling("BOX"))
    palette = logical.quantize(
        colors=colors,
        method=getattr(getattr(Image, "Quantize", Image), "MEDIANCUT"),
        dither=getattr(getattr(Image, "Dither", Image), "NONE"),
    ).convert("RGB")
    return palette.resize((output_w, output_h), _resampling("NEAREST"))


def count_colors(image: Image.Image) -> int:
    """Count unique RGB colours in an image.

    ``Image.getdata`` is deprecated (removal in Pillow 14) in favour of
    ``get_flattened_data``; fall back for Pillow versions that predate it.
    """
    rgb = image.convert("RGB")
    getter = getattr(rgb, "get_flattened_data", None) or rgb.getdata
    return len(set(getter()))


def process_pixel_art_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Process one PNG and return auditable output metadata."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path
    config = (profile or {}).get("postprocess", profile or {})
    logical_size = tuple(config.get("logical_size", [192, 192]))
    output_size = tuple(config.get("output_size", [768, 768]))
    colors = int(config.get("colors", 32))

    with Image.open(input_path) as source:
        source_size = list(source.size)
        processed = process_pixel_art(
            source,
            logical_size=logical_size,
            output_size=output_size,
            colors=colors,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save(output_path, format="PNG", optimize=False)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "source_size": source_size,
        "logical_size": list(logical_size),
        "output_size": list(output_size),
        "colors_requested": colors,
        "colors_written": count_colors(processed),
        "resampling": "box-nearest",
    }


def write_provenance(path: Path, payload: Dict[str, Any]) -> Path:
    """Write a JSON sidecar next to a generated image.

    The payload is passed through the provenance secret guard first: a sidecar
    is a shareable artifact, so a credential that reached this function is
    redacted rather than persisted. See src/video/provenance.py for why this
    is a value check as well as a key-name check.
    """
    from src.video.provenance import redact_secrets

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(".provenance.json")
    safe_payload = redact_secrets(payload)
    sidecar.write_text(
        json.dumps(safe_payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return sidecar


def read_provenance(path: Path) -> Optional[Dict[str, Any]]:
    """Read a provenance sidecar for a generated image, or None.

    Accepts either the image path or the sidecar path. Returns None when no
    sidecar exists or it is unreadable, so a caller can treat provenance as
    optional rather than guarding every call.
    """
    candidate = Path(path)
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".provenance.json")
    if not candidate.is_file():
        return None
    try:
        loaded = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None
