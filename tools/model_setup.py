#!/usr/bin/env python3
"""
Model Setup — interactive role selection for local models.
===========================================================

Discovers every model on this machine, then walks the user through
choosing one model per pipeline role:

    text       required   the LLM for analysis / scriptwriting
    image      required   the image generator
    vision     optional   VQA checks (skipped if none)
    embedding  optional   vector memory (skipped if none)

Discovery never loads a model, so this is safe to run at any time — even
while another pipeline is running. Starting the model happens later, in
the pipeline's own phase manager.

USAGE:
    .venv/bin/python tools/model_setup.py            # interactive
    .venv/bin/python tools/model_setup.py --show     # print current profile
    .venv/bin/python tools/model_setup.py --auto     # non-interactive best pick
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.profile import (
    ModelProfile,
    REQUIRED_ROLES,
    ROLE_CAPABILITY,
    ROLE_EMBEDDING,
    ROLE_IMAGE,
    ROLE_TEXT,
    ROLE_VISION,
    profile_path,
)
from src.models.registry import (
    PROVIDER_GGUF,
    PROVIDER_LLAMACPP,
    PROVIDER_MLXGEN,
    PROVIDER_OLLAMA,
    ModelSpec,
    discover_all,
    specs_for_capability,
)

ROLE_DESCRIPTIONS = {
    ROLE_TEXT: "Text model — analysis, scriptwriting, prompts",
    ROLE_IMAGE: "Image model — pixel art scene generation",
    ROLE_VISION: "Vision model — optional image QA",
    ROLE_EMBEDDING: "Embedding model — optional vector memory",
}

# Preference order when guessing (most specific/capable first).
_PROVIDER_RANK = {
    PROVIDER_LLAMACPP: 0,
    PROVIDER_OLLAMA: 1,
    PROVIDER_GGUF: 2,
    PROVIDER_MLXGEN: 0,
}

# Reserved headroom (GB) that must remain after a model loads, so macOS
# stays responsive. 4 GB is calibrated against measured runs on a 32 GiB
# machine (see src/models/memory.py for the numbers).
_RESERVE_GB = 4.0


def _total_memory_gb() -> float:
    try:
        from src.models import memory
        return memory.snapshot().total_gb or 32.0
    except Exception:
        return 32.0


def _sort_specs(specs: List[ModelSpec], role: str = ROLE_TEXT) -> List[ModelSpec]:
    """Rank candidates for a role.

    Text models: prefer a dedicated server (llama.cpp) over Ollama-file
    scanning, then the largest model that still fits in memory.

    Image models: prefer models that actually fit, then the largest such
    model — a 49 GB unquantized checkpoint is useless on a 32 GiB Mac and
    must never be auto-selected.
    """
    limit = max(_total_memory_gb() - _RESERVE_GB, 4.0)

    def fits(spec: ModelSpec) -> bool:
        est = spec.estimated_memory_gb
        return est <= limit if est else True

    def key(spec: ModelSpec):
        provider_rank = _PROVIDER_RANK.get(spec.provider, 9)
        if role == ROLE_IMAGE:
            # Fitting models first, then largest (best quality).
            return (0 if fits(spec) else 1, provider_rank, -spec.size_bytes, spec.id)
        # Text: provider first, then largest that fits.
        return (provider_rank, 0 if fits(spec) else 1, -spec.size_bytes, spec.id)

    return sorted(specs, key=key)


def print_profile(profile: ModelProfile) -> None:
    print(f"\nProfile: {profile_path()}")
    for role in (ROLE_TEXT, ROLE_IMAGE, ROLE_VISION, ROLE_EMBEDDING):
        spec = profile.get(role)
        if spec:
            print(f"  {role:<10} {spec.describe()}")
            if spec.endpoint:
                print(f"             endpoint: {spec.endpoint}")
            if spec.path:
                print(f"             path:     {spec.path}")
        else:
            marker = "required" if role in REQUIRED_ROLES else "optional"
            print(f"  {role:<10} (none — {marker})")


def _pick_number(prompt: str, count: int, allow_skip: bool = False) -> Optional[int]:
    """Read a 1-based selection index. Returns None when skipped."""
    while True:
        raw = input(prompt).strip()
        if not raw:
            continue
        if allow_skip and raw.lower() in ("s", "skip", "n", "none"):
            return None
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= count:
                return index - 1
        print(f"  Enter a number between 1 and {count}" + (" (or 'skip')" if allow_skip else ""))


def choose_role(
    role: str,
    candidates: List[ModelSpec],
    current: Optional[ModelSpec],
    interactive: bool = True,
) -> Optional[ModelSpec]:
    """Select one model for a role. Returns None when skipped."""
    optional = role not in REQUIRED_ROLES

    if not candidates:
        print(f"\n  {ROLE_DESCRIPTIONS[role]}")
        print(f"    No models discovered for this role.")
        return None

    if not interactive:
        # Auto mode: keep current if still present, else best available.
        if current:
            for spec in candidates:
                if spec.id == current.id and spec.provider == current.provider:
                    return spec
        return candidates[0]

    print(f"\n  {ROLE_DESCRIPTIONS[role]}")
    print("  " + "-" * 58)
    for i, spec in enumerate(candidates, 1):
        current_marker = " ← current" if current and spec.id == current.id else ""
        print(f"   [{i}] {spec.id}")
        print(f"       provider={spec.provider} caps={','.join(spec.capabilities)} "
              f"size={spec.size_gb}GB quant={spec.quantization or 'n/a'}{current_marker}")
        if spec.endpoint:
            print(f"       endpoint: {spec.endpoint}")
        if spec.path:
            print(f"       path: {spec.path}")

    suffix = " [Enter = keep current]" if current else ("" if not optional else " ['skip' to omit]")
    selection = _pick_number(f"  Select {role} model (1-{len(candidates)}){suffix}: ",
                             len(candidates), allow_skip=optional)

    if selection is None:
        if current:
            print(f"  Keeping current {role} model.")
            return current
        return None

    return candidates[selection]


def _register_llamacpp_server(spec: ModelSpec) -> ModelSpec:
    """Give a GGUF spec a managed llama.cpp endpoint.

    The selected GGUF is served by the runtime's managed llama-server on a
    fixed local port. Recording the endpoint here keeps the provider adapter
    construction trivial later.
    """
    spec.provider = PROVIDER_LLAMACPP
    spec.endpoint = os.getenv("LLAMACPP_ENDPOINT", "http://127.0.0.1:8080")
    spec.managed = True
    spec.metadata = dict(spec.metadata or {})
    spec.metadata["served_model_name"] = Path(spec.path or spec.id).name
    return spec


def build_profile(discovered: Dict[str, List[ModelSpec]], interactive: bool = True,
                  existing: Optional[ModelProfile] = None) -> ModelProfile:
    profile = existing or ModelProfile()

    for role in (ROLE_TEXT, ROLE_IMAGE, ROLE_VISION, ROLE_EMBEDDING):
        capability = ROLE_CAPABILITY[role]
        candidates = specs_for_capability(discovered, capability)

        # Text models can come from GGUF files. Promote them to managed
        # llama.cpp specs so the pipeline knows it must launch the server.
        if role == ROLE_TEXT:
            promoted: List[ModelSpec] = []
            for spec in candidates:
                if spec.provider == PROVIDER_GGUF:
                    promoted.append(_register_llamacpp_server(spec))
                else:
                    promoted.append(spec)
            candidates = promoted

        candidates = _sort_specs(candidates, role=role)
        current = profile.get(role)
        chosen = choose_role(role, candidates, current, interactive=interactive)
        setattr(profile, role, chosen)

    profile.preferences = {
        "reserved_memory_gb": _RESERVE_GB,
        # Measured safe on a 32 GiB Mac with an 18.5 GB Q4_K_M model:
        # 32k context fits only when the KV cache is quantized.
        "context_length": int(os.getenv("LLAMACPP_CTX", "32768")),
        # The selected Qwen build thinks before answering; the pipeline
        # prompts expect direct output.
        "reasoning": os.getenv("LLAMACPP_REASONING", "off"),
        "sequential_phases": True,
        "one_job_at_a_time": True,
    }
    return profile


def confirm(profile: ModelProfile) -> bool:
    print("\n" + "=" * 60)
    print("  SELECTED PROFILE")
    print("=" * 60)
    print_profile(profile)
    print("=" * 60)
    try:
        answer = input("\nSave this profile? [Y/n]: ").strip().lower()
    except EOFError:
        return True
    return answer in ("", "y", "yes")


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure local model roles")
    parser.add_argument("--show", action="store_true", help="Print the current profile and exit")
    parser.add_argument("--auto", action="store_true",
                        help="Non-interactive: keep current choices or take best available")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--llamacpp-url", default=os.getenv("LLAMACPP_ENDPOINT", "http://127.0.0.1:8080"))
    parser.add_argument("--path", default=None, help="Override profile output path")
    args = parser.parse_args()

    if args.show:
        profile = ModelProfile.load_or_none(Path(args.path) if args.path else None)
        if profile is None:
            print(f"No profile found at {profile_path()}")
            return 1
        print_profile(profile)
        return 0

    print("=" * 60)
    print("  YT-MACHINE MODEL SETUP")
    print("=" * 60)
    print("Discovering local models (read-only, nothing is loaded)...")

    discovered = discover_all(
        ollama_url=args.ollama_url,
        llamacpp_url=args.llamacpp_url,
        verbose=True,
    )

    total = sum(len(v) for v in discovered.values())
    if total == 0:
        print("\nNo models discovered.")
        print("  - Start Ollama (ollama serve) if you use Ollama models")
        print("  - Set YT_MODEL_ROOTS to directories containing .gguf files")
        print("  - Set MLXGEN_BIN / place MLX-Gen models under ~/AI")
        return 1

    existing = ModelProfile.load_or_none(Path(args.path) if args.path else None)
    profile = build_profile(discovered, interactive=not args.auto, existing=existing)

    if not profile.text:
        print("\nERROR: A text model is required.")
        return 1

    if args.auto or confirm(profile):
        saved = profile.save(Path(args.path) if args.path else None)
        print(f"\nSaved: {saved}")
        return 0

    print("\nAborted — profile not saved.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
