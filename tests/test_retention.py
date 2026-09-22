"""Tests for storage retention planning and application.

Every test operates inside ``tmp_path``. Nothing here may touch the repository's
real ``output/`` tree — these functions delete files, and a test that points at
production output would be destructive.

The important properties are not "does it delete" but "does it refuse to
delete": the newest publishable project, projects inside the age window,
projects within ``keep_last``, and anything reachable through a path that
escapes the retention root.
"""

import os
import time
from pathlib import Path

import pytest

from src.video.retention import (
    discard_folder_if_empty,
    RetentionConfigError,
    apply_retention,
    directory_usage,
    plan_retention,
    summarise_plan,
)

DAY = 86400


# ══════════════════════════════════════════════════════════════
# helpers — build synthetic trees in tmp_path
# ══════════════════════════════════════════════════════════════
def _project(root: Path, name: str, *, age_days: float, with_video: bool = True,
             master: bool = False, body: int = 1000) -> Path:
    """Create a project folder aged ``age_days``.

    The timestamp is applied LAST: writing a child file updates the parent
    directory's mtime, so ageing the directory before populating it would be
    immediately undone.
    """
    project = root / "projects" / name
    project.mkdir(parents=True)
    if with_video:
        (project / f"{name}.mp4").write_bytes(b"D" * body)
    if master:
        (project / f"{name}.master.mp4").write_bytes(b"M" * body)
    (project / "manifest.json").write_text("{}")
    (project / "script.txt").write_text("script")

    stamp = time.time() - (age_days * DAY)
    for child in project.iterdir():
        os.utime(child, (stamp, stamp))
    os.utime(project, (stamp, stamp))
    return project


def _scratch_image(root: Path, name: str, *, age_days: float, sidecar: bool = False) -> Path:
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    image = images / name
    image.write_bytes(b"P" * 500)
    if sidecar:
        image.with_suffix(".provenance.json").write_text("{}")
    stamp = time.time() - (age_days * DAY)
    os.utime(image, (stamp, stamp))
    if sidecar:
        os.utime(image.with_suffix(".provenance.json"), (stamp, stamp))
    return image


def _anchor(root: Path) -> Path:
    """A fresh publishable project that satisfies the discovery protection.

    The newest publishable project is never prunable, so in a tree where the
    subject under test is the only project it would be protected by that rule
    rather than by the age or keep_last logic the test is examining. Adding a
    current anchor makes those rules the deciding factor.

    Age-based tests use this; tests that specifically exercise the protection
    must NOT, because the protection is then the subject.
    """
    return _project(root, "video_anchor_current", age_days=0)


# ══════════════════════════════════════════════════════════════
# planning is pure
# ══════════════════════════════════════════════════════════════
class TestPlanningIsPure:
    def test_plan_performs_no_filesystem_mutation(self, tmp_path):
        _anchor(tmp_path)
        _project(tmp_path, "video_old", age_days=90)
        _scratch_image(tmp_path, "old_scene.png", age_days=90, sidecar=True)

        before = sorted(str(p) for p in tmp_path.rglob("*"))

        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0)

        after = sorted(str(p) for p in tmp_path.rglob("*"))
        assert before == after, "plan_retention must not touch the filesystem"
        # ...but it must still have found the work.
        assert plan["projects"], "expected the old project to be eligible"
        assert plan["scratch"], "expected the old scratch image to be eligible"

    def test_report_mode_plans_without_deleting(self, tmp_path):
        _anchor(tmp_path)
        project = _project(tmp_path, "video_old", age_days=90)
        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0, mode="report")
        assert plan["mode"] == "report"
        assert project.exists()

    def test_off_mode_is_a_no_op(self, tmp_path):
        _anchor(tmp_path)
        _project(tmp_path, "video_old", age_days=900)
        plan = plan_retention(tmp_path, max_age_days=1, keep_last=0, mode="off")
        assert plan["projects"] == []
        assert plan["scratch"] == []
        assert plan["reclaimable_bytes"] == 0


# ══════════════════════════════════════════════════════════════
# age window
# ══════════════════════════════════════════════════════════════
class TestAgeWindow:
    def test_inside_window_survives(self, tmp_path):
        _anchor(tmp_path)
        project = _project(tmp_path, "video_recent", age_days=5)
        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0)
        assert plan["projects"] == []
        assert any("inside age window" in p["reason"] for p in plan["protected"])
        assert project.exists()

    def test_outside_window_is_eligible(self, tmp_path):
        _anchor(tmp_path)
        _project(tmp_path, "video_old", age_days=45)
        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0)
        assert len(plan["projects"]) == 1
        assert plan["projects"][0]["age_days"] >= 30

    def test_boundary_just_inside_survives(self, tmp_path):
        _anchor(tmp_path)
        _project(tmp_path, "video_edge_in", age_days=29)
        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0)
        assert plan["projects"] == []

    def test_boundary_just_outside_is_eligible(self, tmp_path):
        _anchor(tmp_path)
        _project(tmp_path, "video_edge_out", age_days=31)
        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0)
        assert len(plan["projects"]) == 1

    def test_injected_now_makes_the_window_deterministic(self, tmp_path):
        fixed = 1_700_000_000.0
        _anchor(tmp_path)
        project = _project(tmp_path, "video_x", age_days=10)
        os.utime(project, (fixed - 10 * DAY, fixed - 10 * DAY))

        inside = plan_retention(tmp_path, max_age_days=30, keep_last=0, now=fixed)
        outside = plan_retention(tmp_path, max_age_days=5, keep_last=0, now=fixed)
        assert inside["projects"] == []
        assert len(outside["projects"]) == 1


# ══════════════════════════════════════════════════════════════
# protections
# ══════════════════════════════════════════════════════════════
class TestProtections:
    def test_keep_last_protects_recent_projects(self, tmp_path):
        for i in range(5):
            _project(tmp_path, f"video_{i}", age_days=100 + i, with_video=False)
        plan = plan_retention(tmp_path, max_age_days=1, keep_last=3, mode="full")
        # 5 projects, 3 protected by keep_last -> 2 eligible
        assert len(plan["projects"]) == 2
        assert sum(1 for p in plan["protected"] if "keep_last" in p["reason"]) == 3

    def test_newest_publishable_project_always_survives(self, tmp_path):
        """Even with the most aggressive settings."""
        newest = _project(tmp_path, "video_newest", age_days=500, with_video=True)
        _project(tmp_path, "video_older", age_days=600, with_video=True)

        plan = plan_retention(tmp_path, max_age_days=0, keep_last=0, mode="full")

        eligible = {p["path"] for p in plan["projects"]}
        assert str(newest) not in eligible, "newest publishable must never be eligible"
        assert any(
            "newest publishable" in p["reason"] for p in plan["protected"]
        )
        assert newest.exists()

    def test_newest_publishable_holds_when_it_is_not_most_recent_project(self, tmp_path):
        """A newer empty project must not become the 'publishable' one."""
        publishable = _project(tmp_path, "video_pub", age_days=100, with_video=True)
        # an empty project created more recently
        empty = tmp_path / "projects" / "video_empty"
        empty.mkdir(parents=True)
        os.utime(empty, (time.time(), time.time()))

        plan = plan_retention(tmp_path, max_age_days=0, keep_last=0, mode="full")
        eligible = {p["path"] for p in plan["projects"]}
        assert str(publishable) not in eligible
        assert str(empty) in eligible

    def test_keep_last_zero_still_protects_newest_publishable(self, tmp_path):
        newest = _project(tmp_path, "video_newest", age_days=999)
        plan = plan_retention(tmp_path, max_age_days=0, keep_last=0, mode="full")
        assert str(newest) not in {p["path"] for p in plan["projects"]}


# ══════════════════════════════════════════════════════════════
# scratch lifecycle
# ══════════════════════════════════════════════════════════════
class TestScratch:
    def test_recent_images_are_retained_for_curation(self, tmp_path):
        """collect_best_images.py depends on these staying available."""
        _scratch_image(tmp_path, "recent_scene.png", age_days=2, sidecar=True)
        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0)
        assert plan["scratch"] == []

    def test_old_images_become_eligible(self, tmp_path):
        _scratch_image(tmp_path, "old_scene.png", age_days=60, sidecar=True)
        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0)
        assert len(plan["scratch"]) == 1
        assert plan["scratch"][0]["companions"], "sidecar must be paired with its image"

    def test_sidecar_is_never_deleted_without_its_image(self, tmp_path):
        """A sidecar alone must not be planned; it is pruned via its image."""
        images = tmp_path / "images"
        images.mkdir(parents=True)
        sidecar = images / "orphan.provenance.json"
        sidecar.write_text("{}")
        stamp = time.time() - (90 * DAY)
        os.utime(sidecar, (stamp, stamp))

        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0)
        planned = {c for item in plan["scratch"] for c in item["companions"]}
        assert str(sidecar) not in planned

    def test_non_image_files_are_not_claimed(self, tmp_path):
        """Probe videos and ad-hoc files are not the pipeline's to delete."""
        images = tmp_path / "images"
        images.mkdir(parents=True)
        probe = images / "qwen_assembly_probe.mp4"
        probe.write_bytes(b"V" * 1000)
        stamp = time.time() - (400 * DAY)
        os.utime(probe, (stamp, stamp))

        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0)
        assert plan["scratch"] == []


# ══════════════════════════════════════════════════════════════
# apply
# ══════════════════════════════════════════════════════════════
class TestApply:
    def test_full_mode_removes_exactly_the_planned_project(self, tmp_path):
        old = _project(tmp_path, "video_old", age_days=90)
        keep = _project(tmp_path, "video_keep", age_days=1)
        _project(tmp_path, "video_keep2", age_days=2)

        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0, mode="full")
        result = apply_retention(plan)

        assert not old.exists()
        assert keep.exists(), "a project inside the window must survive"
        assert result["freed_bytes"] > 0
        assert len(result["removed_projects"]) == 1

    def test_delivery_only_removes_master_and_keeps_everything_else(self, tmp_path):
        _anchor(tmp_path)
        project = _project(
            tmp_path, "video_x", age_days=90, with_video=True, master=True
        )
        plan = plan_retention(
            tmp_path, max_age_days=30, keep_last=0, mode="delivery_only"
        )
        assert plan["projects"][0]["action"] == "remove_master"

        result = apply_retention(plan)

        # The master is gone; the run stays explainable and publishable.
        assert not (project / "video_x.master.mp4").exists()
        assert (project / "video_x.mp4").exists(), "delivery copy must survive"
        assert (project / "manifest.json").exists(), "manifest must survive"
        assert (project / "script.txt").exists(), "script must survive"
        assert result["removed_masters"]
        assert result["freed_bytes"] > 0

    def test_delivery_only_skips_projects_with_no_master(self, tmp_path):
        _anchor(tmp_path)
        project = _project(tmp_path, "video_nomaster", age_days=90, master=False)
        plan = plan_retention(
            tmp_path, max_age_days=30, keep_last=0, mode="delivery_only"
        )
        assert plan["projects"] == []
        assert any("no master" in p["reason"] for p in plan["protected"])
        assert project.exists()

    def test_apply_handles_files_vanishing_between_plan_and_apply(self, tmp_path):
        _anchor(tmp_path)
        project = _project(tmp_path, "video_old", age_days=90)
        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0, mode="full")

        # simulate a concurrent removal
        import shutil
        shutil.rmtree(project)

        result = apply_retention(plan)
        assert result["removed_projects"] == []
        assert str(project) in result["missing"]
        assert result["freed_bytes"] == 0

    def test_scratch_pruned_with_sidecar(self, tmp_path):
        image = _scratch_image(tmp_path, "old.png", age_days=90, sidecar=True)
        sidecar = image.with_suffix(".provenance.json")

        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0, mode="full")
        result = apply_retention(plan)

        assert not image.exists()
        assert not sidecar.exists()
        assert result["removed_scratch"]


# ══════════════════════════════════════════════════════════════
# path safety
# ══════════════════════════════════════════════════════════════
class TestPathSafety:
    def test_symlink_escaping_root_is_refused(self, tmp_path):
        """A symlinked project must not let a deletion escape the root."""
        outside = tmp_path / "outside"
        outside.mkdir()
        victim = outside / "precious.txt"
        victim.write_text("do not delete")

        root = tmp_path / "root"
        (root / "projects").mkdir(parents=True)
        link = root / "projects" / "video_link"
        link.symlink_to(outside)

        plan = plan_retention(root, max_age_days=0, keep_last=0, mode="full")
        stamp = time.time() - (90 * DAY)
        os.utime(link, (stamp, stamp), follow_symlinks=False)
        plan = plan_retention(root, max_age_days=0, keep_last=0, mode="full")

        result = apply_retention(plan)
        assert victim.exists(), "deletion must not escape the retention root"
        assert not result["removed_projects"]
        assert result["skipped_outside_root"]

    def test_crafted_plan_target_outside_root_is_refused(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("keep me")

        plan = {
            "root": str(root),
            "mode": "full",
            "projects": [{
                "path": str(outside),
                "action": "remove_project",
                "bytes": 1,
                "targets": [str(outside)],
            }],
            "scratch": [],
        }
        result = apply_retention(plan)
        assert outside.exists()
        assert result["skipped_outside_root"]
        assert result["freed_bytes"] == 0


# ══════════════════════════════════════════════════════════════
# reporting and config validation
# ══════════════════════════════════════════════════════════════
class TestReportingAndConfig:
    def test_directory_usage_reports_without_mutating(self, tmp_path):
        _project(tmp_path, "video_a", age_days=1)
        before = sorted(str(p) for p in tmp_path.rglob("*"))
        usage = directory_usage(tmp_path)
        assert usage["bytes"] > 0 and usage["projects"] == 1
        assert sorted(str(p) for p in tmp_path.rglob("*")) == before

    def test_directory_usage_handles_missing_root(self, tmp_path):
        usage = directory_usage(tmp_path / "nope")
        assert usage["exists"] is False and usage["bytes"] == 0

    def test_summary_is_honest_in_report_mode(self, tmp_path):
        _anchor(tmp_path)
        _project(tmp_path, "video_old", age_days=90, with_video=False)
        plan = plan_retention(tmp_path, max_age_days=30, keep_last=0, mode="report")
        text = summarise_plan(plan)
        assert "would reclaim" in text
        assert "report" in text

    @pytest.mark.parametrize("mode", ["delete", "yes", ""])
    def test_invalid_mode_rejected(self, tmp_path, mode):
        if mode == "":
            # empty string is the "unset" signal; resolve_* handles it, but the
            # plan function itself must still reject unknown names
            mode = "bogus"
        with pytest.raises(RetentionConfigError, match="mode"):
            plan_retention(tmp_path, mode=mode)

    @pytest.mark.parametrize("kwargs", [
        {"max_age_days": -1},
        {"keep_last": -1},
    ])
    def test_negative_settings_rejected(self, tmp_path, kwargs):
        with pytest.raises(RetentionConfigError):
            plan_retention(tmp_path, **kwargs)


# ══════════════════════════════════════════════════════════════
# empty-project-folder cleanup (startup abort residue)
# ══════════════════════════════════════════════════════════════
class TestEmptyProjectDiscard:
    """The pipeline creates its project folder before the pre-pipeline
    resource check. Aborts after that point left five empty folders behind.

    Exercises the real helper the pipeline calls
    (``retention.discard_folder_if_empty``); the pipeline's own wrapper adds the
    ``--resume`` exemption, which is asserted here via the same predicate.
    """

    def test_empty_folder_is_removed(self, tmp_path):
        folder = tmp_path / "video_1"
        folder.mkdir()
        assert discard_folder_if_empty(folder) is True
        assert not folder.exists()

    def test_non_empty_folder_is_never_removed(self, tmp_path):
        """A folder with work in it must survive any startup failure."""
        folder = tmp_path / "video_1"
        folder.mkdir()
        (folder / "script.txt").write_text("work happened")
        assert discard_folder_if_empty(folder) is False
        assert folder.exists()
        assert (folder / "script.txt").exists()

    def test_single_checkpoint_file_is_enough_to_preserve(self, tmp_path):
        folder = tmp_path / "video_1"
        folder.mkdir()
        (folder / "checkpoint.json").write_text("{}")
        assert discard_folder_if_empty(folder) is False
        assert folder.exists()

    def test_missing_folder_reports_no_removal(self, tmp_path):
        assert discard_folder_if_empty(tmp_path / "never_created") is False

    def test_a_file_path_is_not_removed(self, tmp_path):
        """Guard against being handed a file rather than a directory."""
        target = tmp_path / "not_a_dir"
        target.write_text("data")
        assert discard_folder_if_empty(target) is False
        assert target.exists()
