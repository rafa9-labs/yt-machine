"""Discover local LoRA adapters and inspect their safetensors metadata.

The image model is selected by a generation profile, while a LoRA is a
separate adapter that must target the same model family.  This module keeps
that check cheap and metadata-only: it never loads model weights or runs
inference.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - exercised by minimal installations
    safe_open = None


REPO_ROOT = Path(__file__).resolve().parents[2]
_RANK_KEYS = ("ss_network_dim", "network_dim", "rank")


@dataclass(frozen=True)
class LoraSpec:
    """Metadata-only description of one local adapter file."""

    path: Path
    name: str
    base_family: Optional[str] = None
    rank: Optional[int] = None
    tensor_count: Optional[int] = None
    trigger: Optional[str] = None
    size_bytes: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for reports and TUI output."""
        return {
            "path": str(self.path),
            "name": self.name,
            "base_family": self.base_family,
            "rank": self.rank,
            "tensor_count": self.tensor_count,
            "trigger": self.trigger,
            "size_bytes": self.size_bytes,
            "error": self.error,
        }


def normalize_family(value: Optional[str]) -> Optional[str]:
    """Normalize common Qwen/Flux family spellings to profile family names."""
    if not value:
        return None
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    if not text:
        return None
    if "qwen" in text and "image" in text:
        return "qwen-image"
    if "flux" in text and "klein" in text:
        return "flux2-klein"
    if "flux" in text:
        return "flux"
    return text


def _metadata_value(metadata: dict[str, str], *keys: str) -> Optional[str]:
    lowered = {str(key).lower(): value for key, value in metadata.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value not in (None, ""):
            return str(value)
    return None


def _parse_rank(metadata: dict[str, str]) -> Optional[int]:
    value = _metadata_value(metadata, *_RANK_KEYS)
    if value is None:
        return None
    try:
        rank = int(float(value))
    except (TypeError, ValueError):
        return None
    return rank if rank > 0 else None


def _infer_rank(handle: Any, keys: list[str]) -> Optional[int]:
    """Infer rank from the first LoRA-A tensor when metadata omits it."""
    for key in keys:
        lowered = key.lower()
        if "lora_a" not in lowered and "lora_down" not in lowered:
            continue
        try:
            shape = tuple(handle.get_tensor(key).shape)
        except Exception:
            return None
        if shape:
            return int(shape[0])
    return None


def _parse_trigger(metadata: dict[str, str]) -> Optional[str]:
    value = _metadata_value(
        metadata,
        "ss_trigger_words",
        "trigger_word",
        "trigger_words",
    )
    if value:
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            parsed = value
        if isinstance(parsed, list):
            words = [str(item).strip() for item in parsed if str(item).strip()]
            return ", ".join(words) or None
        if isinstance(parsed, str) and parsed.strip():
            return parsed.strip()

    tag_frequency = _metadata_value(metadata, "ss_tag_frequency")
    if tag_frequency:
        try:
            parsed = json.loads(tag_frequency)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict) and parsed:
            first = str(next(iter(parsed))).strip()
            return re.sub(r"^\d+_", "", first) or None
    return None


def describe_lora(path: str | Path) -> LoraSpec:
    """Read one adapter's metadata without loading its base model."""
    resolved = Path(path).expanduser()
    name = resolved.stem
    try:
        size_bytes = resolved.stat().st_size
    except OSError:
        size_bytes = 0

    if not resolved.is_file():
        return LoraSpec(resolved, name, size_bytes=size_bytes, error="file not found")
    if safe_open is None:
        return LoraSpec(
            resolved,
            name,
            size_bytes=size_bytes,
            error="safetensors is not installed",
        )

    try:
        with safe_open(str(resolved), framework="pt") as handle:
            metadata = {str(k): str(v) for k, v in (handle.metadata() or {}).items()}
            keys = list(handle.keys())
            rank = _parse_rank(metadata) or _infer_rank(handle, keys)
            tensor_count = len(keys)
        base = normalize_family(
            _metadata_value(
                metadata,
                "ss_base_model_version",
                "base_model",
                "ss_network_module",
                "network_module",
                "modelspec.architecture",
            )
        )
        return LoraSpec(
            path=resolved,
            name=name,
            base_family=base,
            rank=rank,
            tensor_count=tensor_count,
            trigger=_parse_trigger(metadata),
            size_bytes=size_bytes,
            metadata=metadata,
        )
    except Exception as exc:
        return LoraSpec(
            resolved,
            name,
            size_bytes=size_bytes,
            error=f"could not read metadata: {type(exc).__name__}: {exc}",
        )


def default_lora_roots() -> list[Path]:
    """Return configured and conventional directories containing adapters."""
    roots: list[Path] = []
    configured = os.getenv("YT_LORA_ROOTS", "").strip()
    if configured:
        roots.extend(Path(item).expanduser() for item in configured.split(os.pathsep) if item)

    roots.extend([
        REPO_ROOT / "output" / "lora",
        Path.home() / "AI" / "FluxSprites" / "loras",
        Path.home() / "models" / "loras",
    ])

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.expanduser().resolve(strict=False))
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _adapter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() == ".safetensors":
            yield root
        return
    if not root.is_dir():
        return
    try:
        yield from sorted(root.rglob("*.safetensors"))
    except OSError:
        return


def discover_loras(roots: Optional[Iterable[str | Path]] = None) -> list[LoraSpec]:
    """Discover local ``.safetensors`` adapters in bounded configured roots."""
    scan_roots = list(roots) if roots is not None else default_lora_roots()
    found: list[LoraSpec] = []
    seen: set[str] = set()
    for raw_root in scan_roots:
        root = Path(raw_root).expanduser()
        for path in _adapter_files(root):
            key = str(path.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            found.append(describe_lora(path))
    return sorted(found, key=lambda spec: (spec.name.lower(), str(spec.path).lower()))


def lora_matches_model(
    lora: LoraSpec,
    model_family: Optional[str],
) -> tuple[bool, str]:
    """Return whether an adapter is safe to offer for a profile."""
    if lora.error:
        return False, lora.error

    target = normalize_family(model_family)
    if target in (None, "mlx-gen"):
        return True, "profile model family is not specific enough to check"
    if lora.base_family is None:
        return True, "adapter does not declare a base family"
    if lora.base_family == target:
        return True, f"base family matches ({target})"
    return False, f"adapter targets {lora.base_family}; profile targets {target}"
