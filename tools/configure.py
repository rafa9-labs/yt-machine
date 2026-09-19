#!/usr/bin/env python3
"""
Configure — interactive editor for image generation profiles.
==============================================================

WHAT THIS EDITS
───────────────
`config/generation_profiles.json` decides *how* the selected image model is
sampled and post-processed: resolution, diffusion steps, guidance, LoRA,
seed pool, and the pixel-grid reduction. Model *selection* is a separate
concern owned by `tools/model_setup.py`; this tool delegates to it rather than
duplicating model discovery.

WHY A PROFILE AND NOT MORE FLAGS
────────────────────────────────
A full run takes ~80 minutes, most of it image generation. Every image must be
reproducible later, so sampling parameters belong in a versioned file, not in
shell history. `--set` on the pipeline covers quick experiments; this tool is
for durable changes.

SAFETY
──────
Every save is validated by `validate_generation_profile()` before it touches
disk, so this tool cannot write a profile the pipeline would reject. Writes are
atomic (tmp file + os.replace), matching the model-profile writer.

USAGE
─────
    python tools/configure.py            # interactive
    python tools/configure.py --show     # print profiles and exit
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.video.generation_profile import (  # noqa: E402
    DEFAULT_PROFILE_NAME,
    GenerationProfileError,
    describe_profiles,
    profile_path,
    validate_generation_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DISABLE_LORA = object()
_MANUAL_LORA = object()
_KEEP_LORA = object()

try:
    import questionary
    from questionary import Style
except ImportError:  # pragma: no cover - dependency guard
    print("questionary is required for the interactive editor.")
    print("Install it with:")
    print("  uv pip install --python .venv/bin/python 'questionary>=2.0.0'")
    sys.exit(1)


# Reserved for future themed prompts; kept minimal and neutral on purpose.
QSTYLE = Style([
    ("qmark", "fg:#00b8d4 bold"),
    ("question", "bold"),
    ("pointer", "fg:#00b8d4 bold"),
    ("highlighted", "fg:#00b8d4 bold"),
    ("selected", "fg:#8bc34a"),
])


# ── Persistence ──────────────────────────────────────────────────────────

def load_file(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the whole profiles file, not just the active profile."""
    source = profile_path(path)
    if not source.exists():
        raise GenerationProfileError(f"Generation profile not found at {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationProfileError(f"Generation profile is unreadable: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
        raise GenerationProfileError("File must contain a top-level 'profiles' object")
    return data


def save_file(data: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Validate the active profile, then write atomically."""
    source = profile_path(path)
    active = data.get("active_profile")
    profiles = data.get("profiles") or {}
    if active in profiles:
        validate_generation_profile(copy.deepcopy(profiles[active]))

    source.parent.mkdir(parents=True, exist_ok=True)
    tmp = source.with_suffix(source.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, source)
    return source


# ── Display ──────────────────────────────────────────────────────────────

def print_profiles(path: Optional[Path] = None) -> None:
    print(f"\nGeneration profiles: {profile_path(path)}\n")
    try:
        summaries = describe_profiles(path)
    except GenerationProfileError as exc:
        print(f"  ERROR: {exc}\n")
        return
    for entry in summaries:
        marker = "*" if entry["active"] else " "
        if not entry["valid"]:
            print(f" {marker} {entry['name']}  [INVALID] {entry['error']}")
            continue
        print(f" {marker} {entry['name']}")
        print(f"      model    : {entry['model_id']} ({entry['provider']})")
        print(f"      family   : {entry.get('model_family') or '(unspecified)'}")
        print(f"      sampling : {entry['width']}x{entry['height']}, "
              f"{entry['steps']} steps, guidance {entry['guidance']}")
        print(f"      lora     : {entry['lora_name']} (scale {entry['lora_scale']})")
        print(f"      zoom     : {entry['zoom']}")
    print("\n  (* = active profile)\n")


def _fixed_settings() -> None:
    """Explain the values that are not configurable, and why."""
    print("\n" + "-" * 62)
    print("  Fixed by the script contract (not configurable here)")
    print("-" * 62)
    print("  stories per video   2      enforced by the synthesis prompt")
    print("  beats per story     4      hook / mechanism / real talk / fallout")
    print("  images per video    8      2 x 4, one image per beat")
    print("  zoom                off    continuous rescaling destroys the")
    print("                             logical pixel grid this profile keeps")
    print()
    print("  Changing stories or beats means changing the script format,")
    print("  not just a config value: the system prompt, the timeline builder,")
    print("  and the visual-prompt generator all encode the 2x4 structure.")
    print("-" * 62 + "\n")


# ── Editing ──────────────────────────────────────────────────────────────

def _ask_int(message: str, current: int, *, minimum: Optional[int] = None,
             multiple_of: Optional[int] = None) -> int:
    def _validate(raw: str):
        raw = (raw or "").strip()
        if not raw:
            return "Please enter a value"
        try:
            value = int(raw)
        except ValueError:
            return "Must be a whole number"
        if minimum is not None and value < minimum:
            return f"Must be at least {minimum}"
        if multiple_of and value % multiple_of:
            return f"Must be a multiple of {multiple_of}"
        return True

    answer = questionary.text(
        message, default=str(current), validate=_validate, style=QSTYLE
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return int(answer.strip())


def _ask_float(message: str, current: float, *, minimum: Optional[float] = None) -> float:
    def _validate(raw: str):
        raw = (raw or "").strip()
        if not raw:
            return "Please enter a value"
        try:
            value = float(raw)
        except ValueError:
            return "Must be a number"
        if minimum is not None and value < minimum:
            return f"Must be at least {minimum}"
        return True

    answer = questionary.text(
        message, default=str(current), validate=_validate, style=QSTYLE
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return float(answer.strip())


def _ask_text(message: str, current: str, *, required: bool = False) -> str:
    def _validate(raw: str):
        if required and not (raw or "").strip():
            return "This field cannot be empty"
        return True

    answer = questionary.text(
        message, default=current or "", validate=_validate, style=QSTYLE
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer.strip()


def _edit_model_block(profile: Dict[str, Any]) -> None:
    """Choose which discovered image model this profile targets.

    Switching models is the point of this field: the profile owns the model, so
    selecting a different checkpoint here is all that is required. Discovery is
    read-only and never loads weights, so this is safe to run at any time.
    """
    from src.models.registry import list_mlxgen_models

    from src.video.generation_profile import model_family, model_match_key
    from src.video.lora_registry import normalize_family

    current = model_match_key(profile)
    try:
        specs = list_mlxgen_models()
    except Exception as exc:  # discovery must never break the editor
        print(f"  Could not list models ({exc}); keeping {current!r}")
        return

    if not specs:
        print("  No MLX-Gen models discovered. Check YT_MODEL_ROOTS.")
        return

    # Present the full path because an HF cache and a local folder can hold
    # checkpoints with similar names, and the path is what actually runs.
    choices = []
    for spec in specs:
        label = spec.path or spec.id
        marker = "  (current)" if current and current.lower() in label.lower() else ""
        choices.append(questionary.Choice(f"{label}{marker}", value=spec.id))
    choices.append(questionary.Choice("Keep current", value=None))

    selected = questionary.select(
        f"Image model (currently {current!r})", choices=choices, style=QSTYLE
    ).ask()
    if selected is None:
        return

    label = selected
    selected_spec = next((spec for spec in specs if spec.id == selected), None)
    family = normalize_family(label)
    if selected_spec:
        family = normalize_family(selected_spec.id) or normalize_family(selected_spec.path)
    profile["model"] = {"match": label, "family": family or model_family(profile) or ""}
    profile.pop("model_id", None)  # superseded by the model block
    print(f"  Model set to {label}")


def _lora_label(spec, current_path: str, model_family_name: str) -> str:
    """Build a compact choice label without hiding compatibility information."""
    marker = "  (current)" if str(spec.path) == current_path else ""
    rank = f"rank {spec.rank}" if spec.rank else "rank ?"
    family = spec.base_family or "family ?"
    label = f"{spec.name}  [{rank}; {family}]{marker}"
    if spec.error:
        return f"{label}  [unreadable: {spec.error}]"

    from src.video.lora_registry import lora_matches_model

    compatible, reason = lora_matches_model(spec, model_family_name)
    if not compatible:
        return f"{label}  [incompatible: {reason}]"
    return label


def _select_lora(profile: Dict[str, Any]):
    """Return a discovered adapter, or a sentinel for disable/keep/manual."""
    from src.video.generation_profile import model_family
    from src.video.lora_registry import (
        describe_lora,
        discover_loras,
        lora_matches_model,
    )

    current = profile.get("lora") or {}
    current_path = str(current.get("path") or "")
    current_path = str(Path(current_path).expanduser()) if current_path else ""
    family = model_family(profile)
    specs = discover_loras()
    choices = [questionary.Choice("Disable LoRA", value=_DISABLE_LORA)]
    discovered_paths = set()

    for spec in specs:
        path_key = str(spec.path)
        discovered_paths.add(path_key)
        compatible, reason = lora_matches_model(spec, family)
        disabled = spec.error or (not compatible and reason)
        choices.append(
            questionary.Choice(
                _lora_label(spec, current_path, family),
                value=spec,
                disabled=disabled or None,
            )
        )

    if current_path and current_path not in discovered_paths:
        choices.append(
            questionary.Choice(
                f"Keep current (not discovered): {Path(current_path).name}",
                value=_KEEP_LORA,
            )
        )

    choices.extend([
        questionary.Separator(),
        questionary.Choice("Enter a local adapter path", value=_MANUAL_LORA),
    ])
    selected = questionary.select(
        f"Style LoRA for {family or 'the selected model'}",
        choices=choices,
        style=QSTYLE,
    ).ask()
    if selected is None:
        raise KeyboardInterrupt
    if selected in (_DISABLE_LORA, _KEEP_LORA):
        return selected
    if selected is _MANUAL_LORA:
        path = _ask_text("LoRA path", current_path, required=True)
        spec = describe_lora(path)
        if spec.error:
            raise GenerationProfileError(f"LoRA cannot be used: {spec.error}")
        compatible, reason = lora_matches_model(spec, family)
        if not compatible:
            raise GenerationProfileError(f"LoRA is incompatible: {reason}")
        return spec
    return selected


def _edit_lora(profile: Dict[str, Any]) -> None:
    """Choose an adapter and edit its scale/trigger metadata."""
    selected = _select_lora(profile)
    if selected is _DISABLE_LORA:
        profile["lora"] = None
        return
    if selected is not _KEEP_LORA:
        previous = profile.get("lora") or {}
        profile["lora"] = {
            "name": selected.name,
            "path": str(selected.path),
            "scale": previous.get("scale", 0.8),
            "trigger": previous.get("trigger") or selected.trigger or "Pixel Art",
            "base_family": selected.base_family,
            "rank": selected.rank,
        }

    lora = profile.get("lora") or {}
    lora["scale"] = _ask_float(
        "LoRA scale", float(lora.get("scale", 0.8)), minimum=0.0
    )
    lora["trigger"] = _ask_text(
        "LoRA trigger words", lora.get("trigger", "Pixel Art")
    )
    profile["lora"] = lora


def edit_profile(data: Dict[str, Any], name: str) -> bool:
    """Edit one profile in place. Returns True if anything changed."""
    profile = data["profiles"][name]
    post = profile.setdefault("postprocess", {})

    print(f"\nEditing profile: {name}")
    print("  (Ctrl-C to abandon and return to the menu)\n")

    try:
        if questionary.confirm("Change the model this profile targets?",
                               default=False, style=QSTYLE).ask():
            _edit_model_block(profile)

        if not questionary.confirm("Edit sampling and resolution?", default=True,
                                   style=QSTYLE).ask():
            return False

        # The dimension constraint belongs to the model, not to every profile:
        # Qwen's latent route needs multiples of 16, other families report no
        # requirement. Prompt from the profile's declared value so the editor
        # cannot create a profile its own validator would reject.
        _declared = profile.get("dimension_multiple")
        _align = int(_declared) if _declared else 1

        def _size(message: str, current: int) -> int:
            if _align <= 1:
                return _ask_int(message, current, minimum=1)
            return _ask_int(message, current, minimum=_align, multiple_of=_align)

        hint = f" (multiple of {_align})" if _align > 1 else ""
        profile["width"] = _size(f"Render width{hint}", int(profile.get("width", 768)))
        profile["height"] = _size(f"Render height{hint}", int(profile.get("height", 768)))
        profile["steps"] = _ask_int(
            "Diffusion steps (more = slower, diminishing returns)",
            int(profile.get("steps", 20)), minimum=1)
        profile["guidance"] = _ask_float(
            "Guidance scale", float(profile.get("guidance", 4.0)), minimum=0.1)

        print()
        _edit_lora(profile)

        seeds_raw = _ask_text(
            "Seed pool (comma-separated integers)",
            ", ".join(str(s) for s in profile.get("seed_pool", [])))
        seeds = [int(part.strip()) for part in seeds_raw.split(",") if part.strip()]
        if not seeds:
            raise GenerationProfileError("seed_pool cannot be empty")
        if len(set(seeds)) != len(seeds):
            raise GenerationProfileError("seed_pool must not contain duplicates")
        profile["seed_pool"] = seeds

        if not questionary.confirm("Edit pixel-grid post-processing?", default=False,
                                   style=QSTYLE).ask():
            _final_validate(profile, name)
            return True

        logical = post.get("logical_size", [192, 192])
        output = post.get("output_size", [profile["width"], profile["height"]])
        logical_w = _ask_int("Logical grid width", int(logical[0]),
                             minimum=16, multiple_of=16)
        logical_h = _ask_int("Logical grid height", int(logical[1]),
                             minimum=16, multiple_of=16)
        post["logical_size"] = [logical_w, logical_h]

        # output_size must stay an integer multiple of the logical grid, or
        # nearest-neighbour upscaling invents uneven pixel sizes.
        factor = max(1, round(profile["width"] / logical_w))
        output_w = _ask_int(f"Output width (multiple of {logical_w})",
                            int(output[0]), minimum=logical_w, multiple_of=logical_w)
        output_h = _ask_int(f"Output height (multiple of {logical_h})",
                            int(output[1]), minimum=logical_h, multiple_of=logical_h)
        post["output_size"] = [output_w, output_h]
        post["colors"] = _ask_int("Palette colours (2-256)",
                                  int(post.get("colors", 32)), minimum=2)
        if post["colors"] > 256:
            raise GenerationProfileError("postprocess.colors must be between 2 and 256")

        downsample = questionary.select(
            "Downsample filter", choices=["box", "nearest", "bilinear", "lanczos"],
            default=post.get("downsample", "box"), style=QSTYLE).ask()
        if downsample is None:
            raise KeyboardInterrupt
        post["downsample"] = downsample

        upscale = questionary.select(
            "Upscale filter", choices=["nearest", "box"],
            default=post.get("upscale", "nearest"), style=QSTYLE).ask()
        if upscale is None:
            raise KeyboardInterrupt
        post["upscale"] = upscale

    except KeyboardInterrupt:
        print("\n  Edit cancelled — no changes saved.")
        return False
    except (GenerationProfileError, ValueError) as exc:
        print(f"\n  Invalid value: {exc}")
        print("  No changes saved.")
        return False

    _final_validate(profile, name)
    return True


def _final_validate(profile: Dict[str, Any], name: str) -> None:
    """Validate before the caller commits; raises with a specific message."""
    try:
        validate_generation_profile(copy.deepcopy(profile))
    except GenerationProfileError as exc:
        raise GenerationProfileError(f"{name}: {exc}") from exc


# ── Menu actions ─────────────────────────────────────────────────────────

def action_switch(data: Dict[str, Any]) -> bool:
    names = sorted(data["profiles"])
    current = data.get("active_profile")
    choice = questionary.select(
        "Activate which profile?",
        choices=[questionary.Choice(n, value=n, checked=(n == current)) for n in names],
        style=QSTYLE,
    ).ask()
    if choice is None or choice == current:
        return False
    data["active_profile"] = choice
    print(f"  Active profile set to {choice}")
    return True


def action_duplicate(data: Dict[str, Any]) -> bool:
    names = sorted(data["profiles"])
    source = questionary.select("Duplicate which profile?", choices=names, style=QSTYLE).ask()
    if source is None:
        return False

    def _validate(raw: str):
        name = (raw or "").strip()
        if not name:
            return "Please enter a name"
        if name in data["profiles"]:
            return "That name already exists"
        if not all(c.isalnum() or c in "-_" for c in name):
            return "Use letters, digits, hyphen or underscore only"
        return True

    new_name = questionary.text("New profile name", validate=_validate, style=QSTYLE).ask()
    if new_name is None:
        return False
    new_name = new_name.strip()

    data["profiles"][new_name] = copy.deepcopy(data["profiles"][source])
    print(f"  Created {new_name} from {source} (edit it to change settings)")
    return True


def action_configure_models() -> bool:
    """Delegate model selection rather than duplicating discovery logic."""
    script = Path(__file__).resolve().parent / "model_setup.py"
    print(f"\n  Launching {script.name}...\n")
    try:
        subprocess.run([sys.executable, str(script)], check=False)
    except OSError as exc:
        print(f"  Could not launch model_setup.py: {exc}")
    print()
    return False


def action_train_lora() -> bool:
    """Offer the repository's CUDA-only local trainer, never a cloud backend."""
    from src.video.lora_training import check_local_training

    status = check_local_training()
    print(f"\n  {status.menu_label()}")
    if not status.available:
        print("  Local training was not started. Use a CUDA machine with the")
        print("  dependencies installed, then select the generated adapter here.\n")
        return False

    try:
        data_raw = _ask_text("Training data directory", "training_data", required=True)
        data_dir = Path(data_raw).expanduser()
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir
        data_dir = data_dir.resolve()
        if not data_dir.is_dir():
            print(f"  Training data directory not found: {data_dir}")
            return False

        steps = _ask_int("Training steps", 1200, minimum=1)
        rank = _ask_int("LoRA rank", 16, minimum=1)
        learning_rate = _ask_float("Learning rate", 0.0001, minimum=0.0)
        output_raw = _ask_text("Output directory", "output/lora", required=True)
        output_dir = Path(output_raw).expanduser()
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

        if not questionary.confirm(
            f"Start local LoRA training in {data_dir}?",
            default=False,
            style=QSTYLE,
        ).ask():
            return False

        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "train_lora_local.py"),
            str(data_dir),
            "--steps", str(steps),
            "--rank", str(rank),
            "--lr", str(learning_rate),
            "--output", str(output_dir),
        ]
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    except (KeyboardInterrupt, GenerationProfileError, ValueError) as exc:
        print(f"\n  Training cancelled: {exc}\n")
        return False
    except OSError as exc:
        print(f"\n  Could not start local training: {exc}\n")
        return False

    if result.returncode == 0:
        print("\n  Local training finished. Reopen profile editing to select the new adapter.\n")
    else:
        print(f"\n  Local training exited with status {result.returncode}.\n")
    return False


def action_delete(data: Dict[str, Any]) -> bool:
    names = sorted(data["profiles"])
    if len(names) <= 1:
        print("  Cannot delete the only profile.")
        return False
    target = questionary.select("Delete which profile?", choices=names, style=QSTYLE).ask()
    if target is None:
        return False
    if not questionary.confirm(
        f"Delete {target}? This cannot be undone.", default=False, style=QSTYLE
    ).ask():
        return False
    del data["profiles"][target]
    if data.get("active_profile") == target:
        data["active_profile"] = sorted(data["profiles"])[0]
        print(f"  Active profile moved to {data['active_profile']}")
    print(f"  Deleted {target}")
    return True


# ── Entry point ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive editor for image generation profiles")
    parser.add_argument("--show", action="store_true",
                        help="Print current profiles and exit")
    parser.add_argument("--path", default=None,
                        help="Override the generation profile file path")
    args = parser.parse_args()

    path = Path(args.path).expanduser() if args.path else None

    if args.show:
        print_profiles(path)
        return 0

    try:
        data = load_file(path)
    except GenerationProfileError as exc:
        print(f"ERROR: {exc}")
        return 1

    while True:
        try:
            active = data.get("active_profile", "(none)")
            choices: List[Any] = []

            names = sorted(data["profiles"])
            for name in names:
                label = f"{'* ' if name == active else '  '}{name}"
                choices.append(questionary.Choice(label, value=("edit", name)))

            from src.video.lora_training import check_local_training

            training_status = check_local_training()
            choices.extend([
                questionary.Separator(),
                questionary.Choice("Switch active profile", value=("switch", None)),
                questionary.Choice("Duplicate a profile", value=("duplicate", None)),
                questionary.Choice("Delete a profile", value=("delete", None)),
                questionary.Separator(),
                questionary.Choice("Configure models (model_setup.py)",
                                  value=("models", None)),
                questionary.Choice(training_status.menu_label(),
                                  value=("train", None)),
                questionary.Choice("Show resolved configuration", value=("show", None)),
                questionary.Choice("Explain fixed settings", value=("fixed", None)),
                questionary.Separator(),
                questionary.Choice("Save and exit", value=("save", None)),
                questionary.Choice("Exit without saving", value=("quit", None)),
            ])

            print(f"\nYT Machine — Configuration   [{profile_path(path)}]")
            print(f"  active: {active}\n")

            action = questionary.select(
                "What would you like to do?", choices=choices, style=QSTYLE
            ).ask()

            if action is None:
                return 0

            kind, value = action
            changed = False

            if kind == "edit":
                changed = edit_profile(data, value)
            elif kind == "switch":
                changed = action_switch(data)
            elif kind == "duplicate":
                changed = action_duplicate(data)
            elif kind == "delete":
                changed = action_delete(data)
            elif kind == "models":
                action_configure_models()
            elif kind == "train":
                action_train_lora()
            elif kind == "show":
                print_profiles(path)
            elif kind == "fixed":
                _fixed_settings()
            elif kind == "save":
                try:
                    saved = save_file(data, path)
                except GenerationProfileError as exc:
                    print(f"\n  Cannot save — validation failed: {exc}\n")
                    continue
                print(f"\n  Saved to {saved}\n")
                return 0
            elif kind == "quit":
                print("\n  Exited without saving.\n")
                return 0

            if changed:
                print("  (unsaved change — use 'Save and exit' when done)")

        except KeyboardInterrupt:
            print("\n\n  Interrupted — exiting without saving.\n")
            return 130


if __name__ == "__main__":
    sys.exit(main())
