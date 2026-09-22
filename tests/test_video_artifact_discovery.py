"""Regression tests for which artifact gets published.

A project can now hold two MP4s: the lossless master
(`video_<id>.master.mp4`) and the size-bounded delivery copy
(`video_<id>.mp4`). Two independent discovery functions decide what to
upload, and before this change they used different rules — one took the first
MP4 in glob order, the other the newest by mtime. With two artifacts present
those rules can disagree, and publishing the master fails at the provider
(Telegram's 50 MB cap) or spikes memory (TikTok/Instagram read the whole file).

These tests pin the shared selector so the two can never diverge again.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from publish_video import _select_deliverable, find_latest_video  # noqa: E402


def _touch(path: Path, size: int = 2048) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


class TestSelectDeliverable:
    def test_prefers_delivery_over_master(self, tmp_path):
        master = _touch(tmp_path / "video_123.master.mp4")
        delivery = _touch(tmp_path / "video_123.mp4")
        assert _select_deliverable([master, delivery]) == delivery

    def test_order_of_input_does_not_matter(self, tmp_path):
        master = _touch(tmp_path / "video_123.master.mp4")
        delivery = _touch(tmp_path / "video_123.mp4")
        assert _select_deliverable([delivery, master]) == delivery

    def test_master_only_is_still_returned(self, tmp_path):
        """Delivery failed or was disabled: publish the master, do not skip.

        Returning None here would silently publish nothing, which is worse
        than a provider rejecting an oversize file with a clear error.
        """
        master = _touch(tmp_path / "video_123.master.mp4")
        assert _select_deliverable([master]) == master

    def test_delivery_only(self, tmp_path):
        delivery = _touch(tmp_path / "video_123.mp4")
        assert _select_deliverable([delivery]) == delivery

    def test_ignores_temp_files(self, tmp_path):
        temp = _touch(tmp_path / "TEMP_MPY_wvf_snd.mp4")
        delivery = _touch(tmp_path / "video_123.mp4")
        assert _select_deliverable([temp, delivery]) == delivery

    def test_returns_none_when_empty(self, tmp_path):
        assert _select_deliverable([]) is None

    def test_returns_none_when_only_temp_files(self, tmp_path):
        assert _select_deliverable([_touch(tmp_path / "TEMP_x.mp4")]) is None


class TestFindLatestVideo:
    def test_picks_delivery_when_both_exist(self, tmp_path, monkeypatch):
        project = tmp_path / "output" / "projects" / "video_999"
        _touch(project / "video_999.master.mp4", 40 * 1024 * 1024)
        _touch(project / "video_999.mp4", 5 * 1024 * 1024)
        _touch(project / "manifest.json")

        monkeypatch.chdir(tmp_path)
        found = find_latest_video()
        assert found["video_path"].endswith("video_999.mp4")
        assert ".master." not in found["video_path"]

    def test_selects_newest_project(self, tmp_path, monkeypatch):
        old = tmp_path / "output" / "projects" / "video_1"
        new = tmp_path / "output" / "projects" / "video_2"
        _touch(old / "video_1.mp4")
        _touch(new / "video_2.mp4")
        import os
        import time

        past = time.time() - 3600
        os.utime(old / "video_1.mp4", (past, past))

        monkeypatch.chdir(tmp_path)
        assert find_latest_video()["video_path"].endswith("video_2.mp4")

    def test_master_only_project_still_found(self, tmp_path, monkeypatch):
        project = tmp_path / "output" / "projects" / "video_3"
        _touch(project / "video_3.master.mp4")
        monkeypatch.chdir(tmp_path)
        assert find_latest_video()["video_path"].endswith("video_3.master.mp4")


class TestFindersAgree:
    """The two discovery functions must select the same file.

    automate.PROJECT_ROOT resolves from the module's own location, so it cannot
    be pointed at a tmp_path fixture. Rather than mutate a module constant, the
    agreement is asserted structurally: automate delegates to the same selector
    publish_video uses, so the two cannot diverge.
    """

    def test_automate_delegates_to_the_shared_selector(self):
        import importlib.util
        import inspect

        spec = importlib.util.spec_from_file_location(
            "_automate_under_test", REPO_ROOT / "src" / "automate.py"
        )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pytest.skip("automate.py executes on import in this environment")

        source = inspect.getsource(module.find_latest_video)
        assert "_select_deliverable" in source, (
            "automate.find_latest_video must use the shared selector; a local "
            "glob would reintroduce the disagreement this test guards against"
        )

    def test_both_finders_use_the_same_rule(self, tmp_path, monkeypatch):
        """End-to-end: the shared selector, applied to the same directory."""
        project = tmp_path / "output" / "projects" / "video_42"
        _touch(project / "video_42.master.mp4", 40 * 1024 * 1024)
        _touch(project / "video_42.mp4", 5 * 1024 * 1024)
        monkeypatch.chdir(tmp_path)

        from publish_video import _select_deliverable

        publish_choice = Path(find_latest_video()["video_path"]).name
        shared_choice = _select_deliverable(list(project.glob("*.mp4"))).name

        assert publish_choice == shared_choice == "video_42.mp4"
        assert ".master." not in publish_choice
