"""
Tests for the CLI configuration surface and the interactive configure tool.

These cover the two things that can silently corrupt a run: an override that
does not reach the image provider, and a profile save that writes something the
pipeline would later reject.

No interactive prompts are exercised — the menu is driven by questionary, which
is not scriptable here. Instead the pure functions behind each action are
tested directly.

Run: .venv/bin/python -m pytest tests/test_configure_tool.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Override parsing ────────────────────────────────────────────────────────

class TestOverrideParsing:
    def test_parses_simple_and_nested_keys(self):
        from src.video.generation_profile import parse_override

        assert parse_override("steps=12") == ("steps", 12)
        assert parse_override("width=512") == ("width", 512)
        key, value = parse_override("lora.scale=0.55")
        assert key == "lora.scale"
        assert value == pytest.approx(0.55)

    def test_rejects_unknown_key(self):
        from src.video.generation_profile import GenerationProfileError, parse_override

        with pytest.raises(GenerationProfileError, match="Cannot override"):
            parse_override("bogus=1")

    def test_rejects_uncoercible_value(self):
        from src.video.generation_profile import GenerationProfileError, parse_override

        with pytest.raises(GenerationProfileError, match="not a valid int"):
            parse_override("steps=abc")

    def test_rejects_missing_equals(self):
        from src.video.generation_profile import GenerationProfileError, parse_override

        with pytest.raises(GenerationProfileError, match="key=value"):
            parse_override("steps")


# ── Overrides reach every call site ─────────────────────────────────────────

class TestOverridesApplyEverywhere:
    """The profile is loaded at two independent sites in the same process.

    If an override only applied to one, the pipeline would validate one
    configuration and generate images with another.
    """

    def teardown_method(self):
        from src.video.generation_profile import set_overrides

        set_overrides({})

    def test_override_is_visible_to_every_load(self):
        from src.video.generation_profile import load_generation_profile, set_overrides

        set_overrides({"steps": 12, "width": 512})

        first = load_generation_profile()          # pipeline call site
        second = load_generation_profile()         # pixel_art_tool call site

        assert first["steps"] == second["steps"] == 12
        assert first["width"] == second["width"] == 512

    def test_clearing_overrides_restores_file_values(self):
        from src.video.generation_profile import (
            load_generation_profile,
            set_overrides,
        )

        baseline = load_generation_profile()["steps"]
        set_overrides({"steps": 3})
        assert load_generation_profile()["steps"] == 3

        set_overrides({})
        assert load_generation_profile()["steps"] == baseline

    def test_nested_override_applies_to_lora(self):
        from src.video.generation_profile import load_generation_profile, set_overrides

        set_overrides({"lora.scale": 0.15})
        assert load_generation_profile()["lora"]["scale"] == pytest.approx(0.15)

    def test_overrides_are_not_persisted(self, tmp_path):
        """An experiment must not be able to alter the scheduled daily run."""
        from src.video.generation_profile import (
            load_generation_profile,
            profile_path,
            set_overrides,
        )

        source = profile_path()
        before = source.read_bytes()

        set_overrides({"steps": 7})
        load_generation_profile()

        assert source.read_bytes() == before, "overrides must never write to disk"


# ── Validation now covers override inputs ───────────────────────────────────

class TestOverrideValidation:
    def teardown_method(self):
        from src.video.generation_profile import set_overrides

        set_overrides({})

    def test_width_must_be_multiple_of_16(self):
        from src.video.generation_profile import (
            GenerationProfileError,
            load_generation_profile,
            set_overrides,
        )

        set_overrides({"width": 100})
        with pytest.raises(GenerationProfileError, match="multiple of 16"):
            load_generation_profile()

    def test_non_positive_steps_rejected(self):
        from src.video.generation_profile import (
            GenerationProfileError,
            load_generation_profile,
            set_overrides,
        )

        set_overrides({"steps": 0})
        with pytest.raises(GenerationProfileError, match="positive"):
            load_generation_profile()


# ── configure.py persistence ────────────────────────────────────────────────

class TestConfigurePersistence:
    def _fixture_file(self, tmp_path) -> Path:
        import copy

        from src.video.generation_profile import load_generation_profile

        profile = copy.deepcopy(load_generation_profile())
        profile.pop("name", None)
        data = {
            "version": 1,
            "active_profile": "test_profile",
            "profiles": {"test_profile": profile},
        }
        path = tmp_path / "generation_profiles.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def test_round_trip_save_and_load(self, tmp_path):
        import tools.configure as configure

        path = self._fixture_file(tmp_path)

        data = configure.load_file(path)
        data["profiles"]["test_profile"]["steps"] = 11
        configure.save_file(data, path)

        reloaded = configure.load_file(path)
        assert reloaded["profiles"]["test_profile"]["steps"] == 11

    def test_save_rejects_invalid_profile(self, tmp_path):
        """The writer must refuse to persist a profile the pipeline would reject."""
        import tools.configure as configure
        from src.video.generation_profile import GenerationProfileError

        path = self._fixture_file(tmp_path)
        data = configure.load_file(path)

        data["profiles"]["test_profile"]["width"] = 100  # not a multiple of 16

        with pytest.raises(GenerationProfileError):
            configure.save_file(data, path)

        # The file on disk is unchanged.
        on_disk = configure.load_file(path)
        assert on_disk["profiles"]["test_profile"]["width"] == 768

    def test_save_is_atomic_no_tmp_left_behind(self, tmp_path):
        import tools.configure as configure

        path = self._fixture_file(tmp_path)
        data = configure.load_file(path)
        configure.save_file(data, path)

        leftovers = list(tmp_path.glob("*.tmp"))
        assert not leftovers, f"temporary files left behind: {leftovers}"

    def test_load_rejects_missing_profiles_object(self, tmp_path):
        import tools.configure as configure
        from src.video.generation_profile import GenerationProfileError

        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"version": 1}), encoding="utf-8")

        with pytest.raises(GenerationProfileError, match="profiles"):
            configure.load_file(path)

    def test_switch_active_profile(self, tmp_path):
        import copy

        import tools.configure as configure

        path = self._fixture_file(tmp_path)
        data = configure.load_file(path)
        data["profiles"]["second"] = copy.deepcopy(data["profiles"]["test_profile"])
        configure.save_file(data, path)

        reloaded = configure.load_file(path)
        assert set(reloaded["profiles"]) == {"test_profile", "second"}


# ── CLI surface ─────────────────────────────────────────────────────────────

class TestPipelineCliFlags:
    """The pipeline flags are exercised as a subprocess, as a user would."""

    def _run(self, *args):
        import subprocess

        repo_root = Path(__file__).resolve().parent.parent
        return subprocess.run(
            [sys.executable, "tools/generate_complete_video.py", *args],
            capture_output=True, text=True, cwd=repo_root, timeout=180,
        )

    def test_list_profiles_exits_zero_and_names_profile(self):
        result = self._run("--list-profiles")
        assert result.returncode == 0, result.stderr[-400:]
        assert "qwen_pixel_scene" in result.stdout

    def test_print_config_shows_models_and_sampling(self):
        result = self._run("--print-config")
        assert result.returncode == 0, result.stderr[-400:]
        assert "Resolved configuration" in result.stdout
        assert "qwen-image" in result.stdout.lower()
        assert "sampling" in result.stdout

    def test_print_config_reports_overrides(self):
        result = self._run("--set", "steps=9", "--print-config")
        assert result.returncode == 0, result.stderr[-400:]
        assert "steps" in result.stdout
        assert "overrides" in result.stdout.lower()

    def test_invalid_override_exits_two(self):
        result = self._run("--set", "bogus=1", "--print-config")
        assert result.returncode == 2
        assert "Cannot override" in result.stdout

    def test_unknown_profile_exits_two(self):
        result = self._run("--generation-profile", "does_not_exist", "--print-config")
        assert result.returncode == 2
        assert "Unknown generation profile" in result.stdout


# ── Fixed-value documentation ───────────────────────────────────────────────

class TestFixedSettings:
    def test_fixed_settings_mentions_structural_values(self, capsys):
        import tools.configure as configure

        configure._fixed_settings()
        output = capsys.readouterr().out

        assert "stories per video" in output
        assert "beats per story" in output
        assert "images per video" in output
        assert "zoom" in output
