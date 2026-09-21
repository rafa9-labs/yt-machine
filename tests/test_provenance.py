"""Tests for provenance records and their secret guard.

Provenance sidecars are written next to generated images and copied into the
project folder that gets archived and shared, so the guard that keeps
credentials out is treated here as a correctness requirement, not a nicety.
"""

import json
import subprocess
from pathlib import Path

import pytest

from src.video.postprocess import read_provenance, write_provenance
from src.video.provenance import (
    ProvenanceSecretError,
    assert_no_secrets,
    build_run_provenance,
    find_secrets,
    git_commit,
    redact_secrets,
    sha256_file,
)


# ══════════════════════════════════════════════════════════════
# 1. write / read round-trip and naming
# ══════════════════════════════════════════════════════════════
class TestProvenanceSidecar:
    def test_round_trip(self, tmp_path):
        image = tmp_path / "scene.png"
        image.write_bytes(b"not-a-real-png")
        payload = {"model": "qwen-image-2512-4bit", "sampling": {"seed": 137}}

        sidecar = write_provenance(image, payload)
        assert sidecar.name == "scene.provenance.json"
        assert read_provenance(image) == payload

    def test_sidecar_lands_next_to_destination_not_source(self, tmp_path):
        """The pipeline renames images; the sidecar must follow the new name."""
        source = tmp_path / "scratch" / "raw.png"
        source.parent.mkdir()
        source.write_bytes(b"x")
        dest = tmp_path / "project" / "images" / "story_1_part1_raw.png"
        dest.parent.mkdir(parents=True)

        sidecar = write_provenance(dest, {"scene": "story_1_part1"})
        assert sidecar.parent == dest.parent
        assert sidecar.name == "story_1_part1_raw.provenance.json"
        assert not (source.parent / "raw.provenance.json").exists()

    def test_read_accepts_sidecar_path_directly(self, tmp_path):
        image = tmp_path / "scene.png"
        image.write_bytes(b"x")
        write_provenance(image, {"a": 1})
        assert read_provenance(tmp_path / "scene.provenance.json") == {"a": 1}

    def test_read_returns_none_when_absent(self, tmp_path):
        assert read_provenance(tmp_path / "missing.png") is None

    def test_read_returns_none_for_corrupt_json(self, tmp_path):
        sidecar = tmp_path / "scene.provenance.json"
        sidecar.write_text("{not json")
        assert read_provenance(tmp_path / "scene.png") is None

    def test_read_returns_none_for_non_object_json(self, tmp_path):
        sidecar = tmp_path / "scene.provenance.json"
        sidecar.write_text("[1, 2, 3]")
        assert read_provenance(tmp_path / "scene.png") is None


# ══════════════════════════════════════════════════════════════
# 2. secret guard — the enforcement gate
# ══════════════════════════════════════════════════════════════
class TestSecretGuard:
    def test_clean_payload_has_no_findings(self):
        payload = {
            "model": "qwen-image-2512-4bit",
            "provider": "mlxgen",
            "prompt": "a radar tower at dawn",
            "sampling": {"steps": 20, "seed": 137},
            "lora": {"name": "Master-Pixel-Art", "scale": 0.7},
        }
        assert find_secrets(payload) == []
        assert_no_secrets(payload)

    @pytest.mark.parametrize(
        "key",
        ["api_key", "API_KEY", "FAL_KEY", "password", "secret", "token",
         "access_token", "authorization", "client_secret", "private_key"],
    )
    def test_secret_shaped_keys_are_detected(self, key):
        with pytest.raises(ProvenanceSecretError):
            assert_no_secrets({"nested": {key: "anything"}})

    @pytest.mark.parametrize(
        "value",
        [
            "ghp_" + "a" * 30,
            "hf_" + "b" * 30,
            "sk-" + "c" * 30,
            "BSA" + "d" * 30,
            "AKIA" + "E" * 16,
            "123456789:AA" + "f" * 35,
            "-----BEGIN RSA PRIVATE KEY-----",
        ],
    )
    def test_secret_shaped_values_are_detected_under_innocent_keys(self, value):
        """A credential pasted under 'note' must still be caught."""
        with pytest.raises(ProvenanceSecretError):
            assert_no_secrets({"note": value})

    def test_detection_walks_lists(self):
        with pytest.raises(ProvenanceSecretError):
            assert_no_secrets({"items": [{"ok": 1}, {"api_key": "x"}]})

    def test_redaction_removes_key_and_value_secrets(self):
        payload = {
            "model": "qwen",
            "api_key": "should-not-survive",
            "note": "ghp_" + "a" * 30,
            "nested": {"token": "also-secret"},
        }
        safe = redact_secrets(payload)
        assert safe["model"] == "qwen"
        assert safe["api_key"] == "[REDACTED]"
        assert safe["note"] == "[REDACTED]"
        assert safe["nested"]["token"] == "[REDACTED]"

    def test_write_provenance_redacts_before_persisting(self, tmp_path):
        """The writer is the last line of defence: nothing leaks to disk."""
        image = tmp_path / "scene.png"
        image.write_bytes(b"x")
        sidecar = write_provenance(
            image,
            {"model": "qwen", "api_key": "leak-me", "note": "hf_" + "z" * 30},
        )
        raw = sidecar.read_text(encoding="utf-8")
        assert "leak-me" not in raw
        assert "hf_" + "z" * 30 not in raw
        assert "[REDACTED]" in raw

    def test_redaction_does_not_mutate_input(self):
        payload = {"api_key": "keep"}
        redact_secrets(payload)
        assert payload["api_key"] == "keep"


# ══════════════════════════════════════════════════════════════
# 3. reproducibility helpers
# ══════════════════════════════════════════════════════════════
class TestHelpers:
    def test_sha256_matches_known_digest(self, tmp_path):
        target = tmp_path / "f.bin"
        target.write_bytes(b"abc")
        assert sha256_file(target) == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_sha256_returns_none_for_missing_file(self, tmp_path):
        assert sha256_file(tmp_path / "nope.bin") is None

    def test_git_commit_returns_none_outside_a_repo(self, tmp_path):
        """Provenance must not fail a run that has no repository."""
        assert git_commit(tmp_path) is None

    def test_git_commit_returns_sha_in_a_repo(self, tmp_path):
        if not _git_available():
            pytest.skip("git is required")
        _init_repo(tmp_path)
        commit = git_commit(tmp_path)
        assert commit is not None
        assert len(commit) == 40


# ══════════════════════════════════════════════════════════════
# 4. run-level record
# ══════════════════════════════════════════════════════════════
class TestRunProvenance:
    def test_records_identifiers_without_secrets(self):
        record = build_run_provenance(
            project_id=1790015012,
            profile={
                "name": "qwen_pixel_scene",
                "provider": "mlxgen",
                "model": {"match": "qwen-image-2512-4bit"},
                "steps": 20,
                "guidance": 4.0,
                "lora": {"name": "Master-Pixel-Art"},
            },
            tts_result={
                "engine": "kokoro",
                "voice": "kokoro_am_adam",
                "estimated_duration_seconds": 108.0,
            },
            llm_model="Qwen3.8-27B-Q4_K_M",
            assembly={"duration_seconds": 108.0, "resolution": "1080x1920"},
            status="complete",
        )
        assert record["kind"] == "run"
        assert record["project_id"] == 1790015012
        assert record["status"] == "complete"
        assert record["llm_model"] == "Qwen3.8-27B-Q4_K_M"
        assert record["generation_profile"]["model"] == "qwen-image-2512-4bit"
        assert record["tts"]["engine"] == "kokoro"
        assert record["assembly"]["resolution"] == "1080x1920"
        assert_no_secrets(record)

    def test_marks_incomplete_runs(self):
        record = build_run_provenance(project_id=1, status="incomplete")
        assert record["status"] == "incomplete"

    def test_legacy_model_id_form_is_recorded(self):
        """Profiles written before the model block still yield a model name."""
        record = build_run_provenance(
            project_id=1, profile={"model_id": "org/qwen-image-2512-4bit"}
        )
        assert record["generation_profile"]["model"] == "org/qwen-image-2512-4bit"


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=5)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _init_repo(path: Path) -> None:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
    }
    import os

    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "--allow-empty", "-m", "init"],
    ):
        subprocess.run(cmd, cwd=str(path), env={**os.environ, **env},
                       capture_output=True, timeout=15, check=False)
