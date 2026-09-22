"""Storage lifecycle: report and prune accumulated pipeline output.

WHY THIS EXISTS
===============
Two directories grow without bound and nothing reclaims them:

  output/projects/   one folder per run. A completed run holds a lossless
                     master plus a delivery copy (~380-500 MB combined); a
                     failed run can leave a partial or empty folder. At one
                     run per day that is roughly 140-180 GB/year.
  output/images/     per-image scratch: the processed PNG, the pre-postprocess
                     raw PNG, and the provenance sidecar. Recent files are a
                     curation source (tools/collect_best_images.py feeds LoRA
                     training from them), so they must not be deleted on sight.

DESIGN
======
Planning and mutation are separate functions, and the default is to plan:

    plan_retention(...)   pure. Inspects, decides, reports. Never writes.
    apply_retention(...)  executes a plan produced by plan_retention.

The split exists because deleting generated output is irreversible and the
safety rules are subtle (see below). A caller that only wants to know what
*would* happen cannot accidentally cause it to happen, and the destructive
path is testable with an injected ``now``.

SAFETY RULES, and why each exists
=================================
REPORT-ONLY BY DEFAULT
    Nothing is deleted because the pipeline ran. Deletion requires an explicit
    CLI flag or an explicit config opt-in.

NEVER DELETE THE NEWEST PUBLISHABLE PROJECT
    ``publish_video.find_latest_video``, ``automate.find_latest_video`` and the
    server's ``/latest`` all resolve the newest project. Removing it would
    break publishing until the next run completes. This protection holds even
    when ``keep_last=0``.

KEEP_LAST FLOOR
    Independent of age, the most recent N projects are retained, so a burst of
    runs cannot evict everything inside the window.

SCRATCH IS AGE-BASED, NOT IMMEDIATE
    Images are kept for the window so recent ones remain available for LoRA
    curation. A PNG's provenance sidecar is pruned with its PNG so a sidecar
    never outlives the image it describes, and is never deleted while the
    image remains.

DELIVERY-ONLY MODE
    Drops the large lossless master while keeping the compact delivery copy,
    plus the manifest, scripts, images and provenance that explain the run.
    This reclaims most of the bytes without losing the publishable artifact.
    It is opt-in: the master is the archival record and must not be discarded
    by a default.

WHAT IS NOT CLEANED
    Ad-hoc files in output/images/ that no code created (probe videos, benchmark
    clips) are left alone unless they exceed the age window. The pipeline only
    claims ownership of what it wrote.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Extensions the pipeline itself writes into the scratch directory.
_SCRATCH_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# Artifacts that make a project worth keeping intact.
_MASTER_SUFFIX = ".master.mp4"
_PROTECTED_PROJECT_FILES = ("manifest.json", "script.txt", "voiceover.mp3")

VALID_MODES = ("off", "report", "full", "delivery_only")


class RetentionConfigError(ValueError):
    """Raised when retention is configured with an unusable value."""


def _project_has_video(project: Path) -> bool:
    return any(project.glob("*.mp4"))


def _project_video_priority(project: Path) -> bool:
    """True when the project holds a delivery copy (the publishable artifact)."""
    return any(
        p.name.endswith(".mp4") and not p.name.endswith(_MASTER_SUFFIX)
        for p in project.glob("*.mp4")
    )


def _dir_size(path: Path) -> int:
    """Total bytes under ``path``, tolerating files that vanish mid-walk."""
    total = 0
    try:
        entries = list(path.rglob("*"))
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def directory_usage(root: Path) -> Dict[str, Any]:
    """Bytes and project count under ``root``. Read-only."""
    root = Path(root)
    if not root.is_dir():
        return {"root": str(root), "bytes": 0, "projects": 0, "exists": False}
    projects_dir = root / "projects"
    projects = (
        [p for p in projects_dir.iterdir() if p.is_dir()]
        if projects_dir.is_dir() else []
    )
    return {
        "root": str(root),
        "bytes": _dir_size(root),
        "projects": len(projects),
        "exists": True,
    }


def plan_retention(
    root: Path,
    *,
    max_age_days: int = 30,
    keep_last: int = 3,
    mode: str = "report",
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Decide what retention would remove. Performs no writes.

    Returns a plan describing every candidate with its reason and size, the
    bytes reclaimable, and the projects explicitly protected. ``apply_retention``
    consumes this structure.
    """
    if mode not in VALID_MODES:
        raise RetentionConfigError(
            f"retention mode must be one of {', '.join(VALID_MODES)}; got {mode!r}"
        )
    if max_age_days < 0:
        raise RetentionConfigError("max_age_days must not be negative")
    if keep_last < 0:
        raise RetentionConfigError("keep_last must not be negative")

    root = Path(root)
    reference = time.time() if now is None else float(now)
    cutoff = reference - (max_age_days * 86400)

    plan: Dict[str, Any] = {
        "root": str(root),
        "mode": mode,
        "max_age_days": max_age_days,
        "keep_last": keep_last,
        "projects": [],
        "scratch": [],
        "protected": [],
        "reclaimable_bytes": 0,
    }

    if mode == "off" or not root.is_dir():
        return plan

    # ── projects ──────────────────────────────────────────────────────
    projects_dir = root / "projects"
    if projects_dir.is_dir():
        projects = sorted(
            (p for p in projects_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )

        # Publishing discovery (``publish_video.find_latest_video``,
        # ``automate.find_latest_video``, the server's ``/latest``) walks every
        # project and returns the NEWEST one holding an MP4. Protecting that one
        # is therefore sufficient for discovery to keep working: a run can never
        # leave publishing with nothing to publish.
        #
        # Older publishable projects are still prunable — discovery degrades to
        # the next one, which was verified by deleting projects in order until
        # none remained. This protection holds even at keep_last=0.
        newest_publishable = next(
            (p for p in projects if _project_has_video(p)), None
        )

        for index, project in enumerate(projects):
            age = reference - project.stat().st_mtime
            within_keep_last = index < keep_last

            if project == newest_publishable:
                plan["protected"].append(
                    {"path": str(project),
                     "reason": "newest publishable project (publishing "
                               "discovery resolves it)"}
                )
                continue
            if within_keep_last:
                plan["protected"].append(
                    {"path": str(project), "reason": f"within keep_last={keep_last}"}
                )
                continue
            if age < (max_age_days * 86400):
                plan["protected"].append(
                    {"path": str(project), "reason": "inside age window"}
                )
                continue

            # Eligible. In delivery_only mode keep everything except the master.
            masters = [
                p for p in project.glob(f"*{_MASTER_SUFFIX}") if p.is_file()
            ]
            if mode == "delivery_only":
                reclaim = sum(p.stat().st_size for p in masters if p.exists())
                if not masters or reclaim == 0:
                    plan["protected"].append(
                        {"path": str(project), "reason": "no master to reclaim"}
                    )
                    continue
                plan["projects"].append({
                    "path": str(project),
                    "action": "remove_master",
                    "bytes": reclaim,
                    "age_days": round(age / 86400, 1),
                    "targets": [str(p) for p in masters],
                })
            else:  # mode == "full"
                plan["projects"].append({
                    "path": str(project),
                    "action": "remove_project",
                    "bytes": _dir_size(project),
                    "age_days": round(age / 86400, 1),
                    "targets": [str(project)],
                })
            plan["reclaimable_bytes"] += plan["projects"][-1]["bytes"]

    # ── scratch images ────────────────────────────────────────────────
    #
    # Age-based only. Recent images stay available for LoRA curation, and the
    # pipeline only claims files matching the names it writes.
    scratch_dirs = [root / "images"]
    for scratch in scratch_dirs:
        if not scratch.is_dir():
            continue
        for entry in sorted(scratch.iterdir()):
            try:
                if not entry.is_file() or entry.is_symlink():
                    continue
                if not entry.name.endswith(_SCRATCH_SUFFIXES):
                    continue
                stat = entry.stat()
            except OSError:
                continue
            if stat.st_mtime >= cutoff:
                continue

            # A sidecar is pruned with its image, never before it.
            companions = []
            if entry.suffix.lower() == ".png":
                sidecar = entry.with_suffix(".provenance.json")
                if sidecar.is_file():
                    companions.append(sidecar)

            size = stat.st_size + sum(
                c.stat().st_size for c in companions if c.exists()
            )
            plan["scratch"].append({
                "path": str(entry),
                "bytes": size,
                "age_days": round((reference - stat.st_mtime) / 86400, 1),
                "companions": [str(c) for c in companions],
            })
            plan["reclaimable_bytes"] += size

    return plan


def apply_retention(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a plan from :func:`plan_retention`. Returns what actually happened.

    Files may disappear between planning and applying, so every removal is
    individually guarded and missing targets are reported rather than raising.
    Paths are resolved and checked to be inside the plan's root: a symlink or
    crafted entry cannot redirect a deletion outside the retention tree.
    """
    root = Path(plan["root"]).resolve()
    result = {
        "removed_projects": [],
        "removed_masters": [],
        "removed_scratch": [],
        "missing": [],
        "skipped_outside_root": [],
        "freed_bytes": 0,
    }

    def _inside_root(target: Path) -> bool:
        try:
            resolved = target.resolve()
        except OSError:
            return False
        return resolved == root or root in resolved.parents

    for item in plan.get("projects", []):
        project = Path(item["path"])
        if not _inside_root(project):
            result["skipped_outside_root"].append(str(project))
            continue

        if item["action"] == "remove_project":
            if not project.exists():
                result["missing"].append(str(project))
                continue
            size = _dir_size(project)
            try:
                shutil.rmtree(project)
            except OSError:
                result["missing"].append(str(project))
                continue
            result["removed_projects"].append({"path": str(project), "bytes": size})
            result["freed_bytes"] += size

        elif item["action"] == "remove_master":
            for target in item.get("targets", []):
                master = Path(target)
                if not _inside_root(master):
                    result["skipped_outside_root"].append(str(master))
                    continue
                if not master.exists():
                    result["missing"].append(str(master))
                    continue
                try:
                    size = master.stat().st_size
                    master.unlink()
                except OSError:
                    result["missing"].append(str(master))
                    continue
                result["removed_masters"].append({"path": str(master), "bytes": size})
                result["freed_bytes"] += size

    for item in plan.get("scratch", []):
        targets = [Path(item["path"])] + [Path(c) for c in item.get("companions", [])]
        for target in targets:
            if not _inside_root(target):
                result["skipped_outside_root"].append(str(target))
                continue
            if not target.exists():
                result["missing"].append(str(target))
                continue
            try:
                size = target.stat().st_size
                target.unlink()
            except OSError:
                result["missing"].append(str(target))
                continue
            result["removed_scratch"].append({"path": str(target), "bytes": size})
            result["freed_bytes"] += size

    return result


def summarise_plan(plan: Dict[str, Any]) -> str:
    """One-line human summary of a plan, for run logs and CLI output."""
    projects = len(plan.get("projects", []))
    scratch = len(plan.get("scratch", []))
    mb = plan.get("reclaimable_bytes", 0) / (1024 * 1024)
    protected = len(plan.get("protected", []))
    if plan.get("mode") == "off":
        return "retention: disabled"
    if not projects and not scratch:
        return (f"retention[{plan['mode']}]: nothing to reclaim "
                f"({protected} protected)")
    verb = "would reclaim" if plan.get("mode") == "report" else "reclaimable"
    return (
        f"retention[{plan['mode']}]: {projects} project(s), {scratch} scratch "
        f"file(s) {verb} {mb:.1f}MB ({protected} protected)"
    )


def discard_folder_if_empty(folder: Path) -> bool:
    """Remove ``folder`` only when it is completely empty.

    Used by the pipeline to clean up the project directory when startup aborts
    before anything was written to it — the pre-pipeline resource check calls
    ``sys.exit(1)``, and the memory guard raises, both after the folder is
    created. Five empty ``output/projects/video_<id>/`` folders had accumulated
    this way.

    Deliberately conservative: a non-empty folder means work happened, so it is
    left alone. An OSError (permissions, a race) is reported as False rather
    than raised, because startup cleanup must never itself break startup.

    Returns True only when a directory was actually removed.
    """
    candidate = Path(folder)
    try:
        if candidate.is_dir() and not any(candidate.iterdir()):
            candidate.rmdir()
            return True
    except OSError:
        return False
    return False
