"""
Tests for macOS automation plumbing — schedule generation, wake handling,
and the publisher's explicit-path metadata lookup.

No network, no launchd, no pmset: launchctl/subprocess calls are patched.

Run: .venv/bin/python -m pytest tests/test_automation_macos.py -v
"""

import json
import os
import plistlib
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def automate():
    import importlib

    import src.automate as mod

    importlib.reload(mod)
    return mod


# ── Time parsing ────────────────────────────────────────────────────────────


class TestParseHHMM:
    def test_plain_time(self, automate):
        assert automate._parse_hhmm("06:00") == (6, 0)

    def test_with_seconds(self, automate):
        assert automate._parse_hhmm("05:50:00") == (5, 50)

    def test_invalid_falls_back_to_default(self, automate):
        with mock.patch.object(automate, "RUN_TIME", "06:00"):
            assert automate._parse_hhmm("nonsense") == (6, 0)

    def test_out_of_range_falls_back(self, automate):
        with mock.patch.object(automate, "RUN_TIME", "06:00"):
            assert automate._parse_hhmm("99:99") == (6, 0)


# ── Plist rendering ─────────────────────────────────────────────────────────


class TestPlistXml:
    def test_valid_plist_and_schedule(self, automate):
        xml = automate._plist_xml("06:00")
        parsed = plistlib.loads(xml.encode("utf-8"))

        assert parsed["Label"] == automate.LAUNCHD_LABEL
        assert parsed["StartCalendarInterval"] == {"Hour": 6, "Minute": 0}
        assert parsed["RunAtLoad"] is False

    def test_uses_wrapper_script_not_python_directly(self, automate):
        parsed = plistlib.loads(automate._plist_xml("07:30").encode("utf-8"))
        args = parsed["ProgramArguments"]

        assert args[0] == "/bin/bash"
        assert args[1].endswith("tools/run_daily.sh")

    def test_homebrew_bin_on_path(self, automate):
        parsed = plistlib.loads(automate._plist_xml("06:00").encode("utf-8"))
        env_path = parsed["EnvironmentVariables"]["PATH"]

        assert "/opt/homebrew/bin" in env_path

    def test_working_directory_is_project_root(self, automate):
        parsed = plistlib.loads(automate._plist_xml("06:00").encode("utf-8"))
        assert parsed["WorkingDirectory"] == str(automate.PROJECT_ROOT)

    def test_logs_go_to_output_logs(self, automate):
        parsed = plistlib.loads(automate._plist_xml("06:00").encode("utf-8"))
        assert "output/logs" in parsed["StandardOutPath"]
        assert "output/logs" in parsed["StandardErrorPath"]


# ── schedule_task / remove_scheduled_task ───────────────────────────────────


class TestScheduleTask:
    def test_install_writes_plist_and_bootstraps(self, automate, tmp_path):
        plist = tmp_path / "agent.plist"
        calls = []

        def fake_launchctl(*args):
            calls.append(args)
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(automate, "LAUNCHD_PLIST", plist), \
             mock.patch.object(automate, "_launchctl", side_effect=fake_launchctl), \
             mock.patch.object(automate, "LOG_DIR", tmp_path), \
             mock.patch.object(automate, "remove_scheduled_task", return_value=True):
            assert automate.schedule_task("06:00") is True

        assert plist.exists()
        parsed = plistlib.loads(plist.read_bytes())
        assert parsed["StartCalendarInterval"] == {"Hour": 6, "Minute": 0}
        assert any("bootstrap" in c for c in calls)

    def test_bootstrap_failure_is_reported(self, automate, tmp_path):
        plist = tmp_path / "agent.plist"

        def failing_launchctl(*args):
            return mock.Mock(returncode=1, stdout="", stderr="boom")

        with mock.patch.object(automate, "LAUNCHD_PLIST", plist), \
             mock.patch.object(automate, "_launchctl", side_effect=failing_launchctl), \
             mock.patch.object(automate, "LOG_DIR", tmp_path), \
             mock.patch.object(automate, "remove_scheduled_task", return_value=True):
            assert automate.schedule_task("06:00") is False

    def test_remove_is_idempotent(self, automate, tmp_path):
        missing = tmp_path / "nope.plist"

        with mock.patch.object(automate, "LAUNCHD_PLIST", missing), \
             mock.patch.object(automate, "_launchctl",
                               return_value=mock.Mock(returncode=1, stdout="", stderr="")):
            assert automate.remove_scheduled_task() is True


# ── Wake handling ───────────────────────────────────────────────────────────


class TestWakeHandling:
    """The wake phase must not spawn a real pipeline.

    run_full_automation calls run_pipeline, which launches the ~80 minute
    generation as a subprocess. These tests mock it out — an earlier version
    of this file started a real run and had to be killed.
    """

    def _run(self, automate, wol_mac, online=True, wol_ok=True):
        with mock.patch.object(automate, "WOL_MAC", wol_mac), \
             mock.patch.object(automate, "is_pc_online", return_value=online), \
             mock.patch.object(automate, "wake_pc", return_value=wol_ok) as wol, \
             mock.patch.object(automate, "wait_for_pc", return_value=True), \
             mock.patch.object(automate, "run_pipeline",
                               return_value={"success": True, "elapsed_seconds": 1}), \
             mock.patch.object(automate, "find_latest_video", return_value=None), \
             mock.patch.object(automate, "send_telegram_notification", return_value=False):
            report = automate.run_full_automation(
                wake=True, publish=False, platforms=None, skip_images=True
            )
        return report, wol

    def test_local_mode_does_not_send_wol(self, automate):
        report, wol = self._run(automate, "")

        wol.assert_not_called()
        assert report["wake"] == {"status": "local"}

    def test_wol_configured_still_uses_remote_flow(self, automate):
        report, wol = self._run(automate, "AA:BB:CC:DD:EE:FF", online=True)

        wol.assert_not_called()
        assert report["wake"] == {"status": "already_online"}

    def test_wol_sent_when_remote_is_offline(self, automate):
        report, wol = self._run(automate, "AA:BB:CC:DD:EE:FF", online=False)

        wol.assert_called_once()
        assert report["wake"] == {"status": "online"}

    def test_install_wake_requires_darwin(self, automate):
        with mock.patch.object(sys, "platform", "linux"):
            assert automate.install_wake_schedule() is False


# ── Retry classification ────────────────────────────────────────────────────


class TestRetryClassification:
    def test_missing_credentials_not_retryable(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import publish_video

        assert publish_video._is_retryable({"error": "Missing client secrets"}) is False
        assert publish_video._is_retryable({"error": "Missing credentials"}) is False
        assert publish_video._is_retryable({"error": "TikTok credentials not configured"}) is False

    def test_transient_errors_are_retryable(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import publish_video

        assert publish_video._is_retryable({"error": "503 Service Unavailable"}) is True
        assert publish_video._is_retryable({"error": "Connection reset by peer"}) is True
        assert publish_video._is_retryable({}) is True

    def test_no_backoff_sleep_for_config_errors(self, tmp_path):
        """A missing credential must not burn 90s of retry backoff."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import publish_video

        project = tmp_path / "video_1"
        project.mkdir()
        video = project / "v.mp4"
        video.write_bytes(b"x")

        calls = []

        def failing(v, m, d=False):
            calls.append(1)
            return {"platform": "youtube", "status": "error", "error": "Missing client secrets"}

        with mock.patch.object(publish_video, "LEDGER_PATH", tmp_path / "l.json"), \
             mock.patch.object(publish_video, "publish_youtube", side_effect=failing), \
             mock.patch.object(publish_video.time, "sleep") as sleeper:
            publish_video.publish_video(str(video), ["youtube"])

        assert len(calls) == 1, "config errors must not be retried"
        sleeper.assert_not_called()


# ── Pipeline timeout ────────────────────────────────────────────────────────


class TestPipelineTimeout:
    def test_default_timeout_exceeds_typical_run(self, automate):
        """The shipped default must cover a full run.

        Reads the default from source rather than the module attribute:
        ``automate.PIPELINE_TIMEOUT`` resolves the environment first, and a
        local .env may legitimately raise it.
        """
        import re

        src = (Path(__file__).resolve().parent.parent
               / "src" / "automate.py").read_text(encoding="utf-8")
        match = re.search(
            r'PIPELINE_TIMEOUT\s*=\s*int\(os\.getenv\("PIPELINE_TIMEOUT",\s*"(\d+)"\)',
            src,
        )
        assert match, "could not find the pipeline timeout default"
        default = int(match.group(1))

        assert default >= 5400, (
            f"default PIPELINE_TIMEOUT={default}s is too low for a run that "
            "generates 8 images sequentially"
        )

    def test_timeout_defaults_match_between_supervisors(self, automate):
        """automate.py and server.py supervise the same pipeline.

        Compares the two declaration defaults, not the environment-resolved
        values: a local .env may raise one without the other being wrong, and
        the invariant under test is that the shipped defaults agree.

        Parsed as text because src.server pulls in FastAPI and the full
        application graph, which is far too heavy for a unit test.
        """
        import re

        repo = Path(__file__).resolve().parent.parent

        automate_src = (repo / "src" / "automate.py").read_text(encoding="utf-8")
        automate_match = re.search(
            r'PIPELINE_TIMEOUT\s*=\s*int\(os\.getenv\("PIPELINE_TIMEOUT",\s*"(\d+)"\)',
            automate_src,
        )
        assert automate_match, "could not find automate.py's timeout default"

        server_src = (repo / "src" / "server.py").read_text(encoding="utf-8")
        server_match = re.search(
            r'_PIPELINE_TIMEOUT_S\s*=\s*int\(os\.getenv\("PIPELINE_TIMEOUT",\s*"(\d+)"\)',
            server_src,
        )
        assert server_match, "could not find the server's pipeline timeout default"

        assert int(server_match.group(1)) == int(automate_match.group(1)), (
            "the shipped pipeline timeout defaults must agree; "
            f"automate.py={automate_match.group(1)} "
            f"server.py={server_match.group(1)}"
        )


# ── Power configuration ─────────────────────────────────────────────────────


class TestPowerConfig:
    def test_configure_power_sets_ac_only(self, automate):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(automate.subprocess, "run", side_effect=fake_run):
            assert automate.configure_power(20) is True

        flat = [" ".join(c) for c in calls]
        assert any("-c sleep 20" in c for c in flat)
        assert any("-c displaysleep 10" in c for c in flat)
        # Battery settings must never be touched by the automation.
        assert not any("-b " in c for c in flat)

    def test_configure_power_reports_sudo_prompt(self, automate):
        def fake_run(cmd, **kwargs):
            return mock.Mock(returncode=1, stdout="", stderr="password is required")

        with mock.patch.object(automate.subprocess, "run", side_effect=fake_run):
            assert automate.configure_power(20) is False


# ── sys.path wiring ─────────────────────────────────────────────────────────


class TestImportPaths:
    def test_automate_can_import_tools_and_publish(self):
        import src.automate  # noqa: F401
        from src.automate import PROJECT_ROOT

        assert str(PROJECT_ROOT) in sys.path
        assert str(PROJECT_ROOT / "src") in sys.path

        import publish_video  # noqa: F401
        import tools.telegram_sender  # noqa: F401


# ── Publisher explicit-path metadata ────────────────────────────────────────


class TestExplicitVideoPathMetadata:
    def test_project_dir_derived_from_video_path(self, tmp_path):
        """An explicit --video path must still pick up platform_metadata.json.

        Regression: project_dir was hardcoded None, so the upload silently
        used the generic placeholder title instead of the generated one.
        """
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import publish_video

        project = tmp_path / "video_123"
        project.mkdir()
        video = project / "video_123.mp4"
        video.write_bytes(b"fake")
        (project / "platform_metadata.json").write_text(json.dumps({
            "youtube": {"title": "Generated Title From Pipeline"},
            "common_hashtags": ["#geopolitics"],
        }), encoding="utf-8")

        captured = {}

        def fake_publish(platform, video_path, metadata, dry_run=False):
            captured["metadata"] = metadata
            return {"platform": platform, "status": "dry_run"}

        with mock.patch.dict(publish_video.__dict__, {
            "publish_youtube": lambda v, m, d=False: fake_publish("youtube", v, m, d),
        }), mock.patch("publish_video.find_latest_video") as finder:
            results = publish_video.publish_video(
                video_path=str(video), platforms=["youtube"], dry_run=True
            )

        finder.assert_not_called()
        assert results[0]["status"] == "dry_run"
        assert captured["metadata"]["title"] == "Generated Title From Pipeline"

    def test_missing_video_returns_empty(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import publish_video

        results = publish_video.publish_video(
            video_path=str(tmp_path / "missing.mp4"), platforms=["youtube"]
        )
        assert results == []


# ── Publish idempotency (ledger) ────────────────────────────────────────────


class TestPublishLedger:
    def _make_video(self, tmp_path):
        project = tmp_path / "video_999"
        project.mkdir()
        video = project / "video_999.mp4"
        video.write_bytes(b"x" * 128)
        (project / "platform_metadata.json").write_text(
            json.dumps({"youtube": {"title": "Ledger Test"}}), encoding="utf-8"
        )
        return video

    def test_second_run_skips_already_published_platform(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import publish_video

        video = self._make_video(tmp_path)
        ledger_file = tmp_path / "ledger.json"
        calls = []

        def fake_youtube(v, m, d=False):
            calls.append(v)
            return {"platform": "youtube", "status": "published",
                    "url": "https://youtu.be/abc", "video_id": "abc"}

        with mock.patch.object(publish_video, "LEDGER_PATH", ledger_file), \
             mock.patch.object(publish_video, "publish_youtube", side_effect=fake_youtube):
            first = publish_video.publish_video(str(video), ["youtube"])
            second = publish_video.publish_video(str(video), ["youtube"])

        assert first[0]["status"] == "published"
        assert second[0]["status"] == "already_published"
        assert len(calls) == 1, "upload must happen exactly once"

    def test_force_overrides_ledger(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import publish_video

        video = self._make_video(tmp_path)
        ledger_file = tmp_path / "ledger.json"
        calls = []

        def fake_youtube(v, m, d=False):
            calls.append(v)
            return {"platform": "youtube", "status": "published", "url": "u", "video_id": "i"}

        with mock.patch.object(publish_video, "LEDGER_PATH", ledger_file), \
             mock.patch.object(publish_video, "publish_youtube", side_effect=fake_youtube):
            publish_video.publish_video(str(video), ["youtube"])
            forced = publish_video.publish_video(str(video), ["youtube"], force=True)

        assert forced[0]["status"] == "published"
        assert len(calls) == 2

    def test_failures_are_not_recorded(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import publish_video

        video = self._make_video(tmp_path)
        ledger_file = tmp_path / "ledger.json"

        def failing_youtube(v, m, d=False):
            return {"platform": "youtube", "status": "error", "error": "nope"}

        with mock.patch.object(publish_video, "LEDGER_PATH", ledger_file), \
             mock.patch.object(publish_video, "publish_youtube", side_effect=failing_youtube), \
             mock.patch.object(publish_video.time, "sleep"):
            publish_video.publish_video(str(video), ["youtube"])

        ledger = json.loads(ledger_file.read_text()) if ledger_file.exists() else {}
        assert not any("youtube" in v for v in ledger.values())

    def test_dry_run_does_not_write_ledger(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import publish_video

        video = self._make_video(tmp_path)
        ledger_file = tmp_path / "ledger.json"

        def fake_dry(v, m, d=False):
            return {"platform": "youtube", "status": "dry_run"}

        with mock.patch.object(publish_video, "LEDGER_PATH", ledger_file), \
             mock.patch.object(publish_video, "publish_youtube", side_effect=fake_dry):
            publish_video.publish_video(str(video), ["youtube"], dry_run=True)

        assert not ledger_file.exists()

    def test_fingerprint_changes_when_video_changes(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import publish_video

        video = self._make_video(tmp_path)
        first = publish_video._video_fingerprint(str(video))
        video.write_bytes(b"y" * 256)
        second = publish_video._video_fingerprint(str(video))

        assert first != second


# ── TikTok token helper ─────────────────────────────────────────────────────


class TestTiktokToken:
    def test_expiry_detection(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import tools.tiktok_auth as auth

        token_file = tmp_path / "tiktok.json"
        with mock.patch.object(auth, "token_path", return_value=token_file):
            from datetime import datetime, timedelta

            token_file.write_text(json.dumps({
                "access_token": "abc",
                "refresh_token": "def",
                "expires_at": (datetime.now() + timedelta(hours=2)).isoformat(),
            }), encoding="utf-8")
            assert auth.token_is_expired() is False

            token_file.write_text(json.dumps({
                "access_token": "abc",
                "refresh_token": "def",
                "expires_at": (datetime.now() - timedelta(minutes=1)).isoformat(),
            }), encoding="utf-8")
            assert auth.token_is_expired() is True

    def test_env_token_used_when_no_file(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import tools.tiktok_auth as auth

        with mock.patch.object(auth, "token_path", return_value=tmp_path / "none.json"), \
             mock.patch.dict(os.environ, {"TIKTOK_ACCESS_TOKEN": "from-env"}):
            assert auth.get_access_token() == "from-env"

    def test_refresh_without_credentials_fails_cleanly(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import tools.tiktok_auth as auth

        clean_env = {
            k: v for k, v in os.environ.items()
            if not k.startswith("TIKTOK_")
        }
        with mock.patch.object(auth, "token_path", return_value=tmp_path / "none.json"), \
             mock.patch.dict(os.environ, clean_env, clear=True):
            result = auth.refresh_access_token()

        assert result["success"] is False
        assert "Missing refresh credentials" in result["error"]
