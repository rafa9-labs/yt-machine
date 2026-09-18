"""
Tests for the structured SceneSpec prompt builder.
Run: .venv/bin/python -m pytest tests/test_scene_spec.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from src.video.scene_spec import (
    STYLE_PIXEL_SCENE,
    TEXT_CONTEXTUAL,
    TEXT_EXACT,
    TEXT_OMIT,
    SceneSpec,
    classify_text,
    from_visual_scene,
)


# ════════════════════════════════════════════════════════════════════
# Text policy
# ════════════════════════════════════════════════════════════════════

class TestTextPolicy:
    """Class C (exact values) must never be delegated to the model.

    The previous sanitizer rewrote every literal into generic phrasing,
    which deleted meaning and could make the replacement the focal point
    ('$33,000,000,000' -> 'a large red indicator').
    """

    @pytest.mark.parametrize("value", [
        "$33,000,000,000",
        "€1,200",
        "33,000,000",
        "47%",
        "33000",
    ])
    def test_exact_values_are_deferred(self, value):
        r = classify_text(value)
        assert r.disposition == TEXT_EXACT, f"{value} should be deferred, got {r.disposition}"

    def test_currency_is_exact(self):
        assert classify_text("$5").disposition == TEXT_EXACT

    def test_percentage_is_exact(self):
        assert classify_text("12%").disposition == TEXT_EXACT

    @pytest.mark.parametrize("value", ["EMPTY", "OIL SPIKE", "REC", "STOP"])
    def test_short_labels_are_contextual(self, value):
        r = classify_text(value)
        assert r.disposition == TEXT_CONTEXTUAL

    def test_long_text_is_omitted(self):
        r = classify_text("a long descriptive sentence about something else entirely")
        assert r.disposition == TEXT_OMIT

    def test_empty_is_omitted(self):
        assert classify_text("").disposition == TEXT_OMIT
        assert classify_text("   ").disposition == TEXT_OMIT

    def test_every_requirement_has_a_rationale(self):
        for v in ["$1,000", "EMPTY", "a very long sentence that cannot be rendered well"]:
            assert classify_text(v).rationale


class TestSceneSpecTextRouting:
    def test_exact_text_excluded_from_prompt(self):
        spec = SceneSpec(subject="A receipt", style="pixel art")
        spec.add_text("$33,000,000,000")
        prompt = spec.to_prompt()
        assert "33,000,000,000" not in prompt
        assert "$33,000,000,000" in spec.deferred_text

    def test_contextual_text_included_in_prompt(self):
        spec = SceneSpec(subject="A crate", style="pixel art")
        spec.add_text("EMPTY")
        assert "EMPTY" in spec.to_prompt()
        assert "EMPTY" in spec.renderable_text

    def test_omitted_text_never_appears(self):
        spec = SceneSpec(subject="A room", style="pixel art")
        spec.add_text("a very long descriptive sentence that should not be rendered")
        assert "very long descriptive" not in spec.to_prompt()


# ════════════════════════════════════════════════════════════════════
# Prompt assembly
# ════════════════════════════════════════════════════════════════════

class TestPromptAssembly:
    def _spec(self):
        return SceneSpec(
            subject="An inspector holding a clipboard",
            action="stands beside two damaged hangars",
            environment="at a desert military base",
            camera="Isometric three-quarter view",
            hierarchy="one primary focal subject",
            props=["supply crates"],
            style=STYLE_PIXEL_SCENE,
            lora_trigger="Pixel Art",
        )

    def test_trigger_leads(self):
        assert self._spec().to_prompt().startswith("Pixel Art.")

    def test_subject_precedes_style(self):
        p = self._spec().to_prompt()
        assert p.lower().index("inspector") < p.lower().index("crisp game-art")

    def test_no_double_periods(self):
        assert ".." not in self._spec().to_prompt()

    def test_environment_attaches_grammatically(self):
        """'hangars. at a desert base' is broken English and hurts output."""
        p = self._spec().to_prompt()
        assert "hangars at a desert military base" in p
        assert "hangars. at" not in p

    def test_style_tail_is_capitalised(self):
        p = self._spec().to_prompt()
        assert "Crisp game-art" in p
        assert p.endswith(".")

    def test_prompt_is_concise(self):
        """Legacy prompts ran ~150 words of triple-styled text."""
        assert len(self._spec().to_prompt().split()) < 90

    def test_no_repeated_pixel_terminology(self):
        """One style tail; do not restate the medium repeatedly."""
        p = self._spec().to_prompt().lower()
        assert p.count("pixel art") <= 2

    def test_empty_fields_do_not_create_artifacts(self):
        spec = SceneSpec(subject="A lone tower", lora_trigger="Pixel Art")
        p = spec.to_prompt()
        assert ".." not in p
        assert "  " not in p

    def test_props_are_deduplicated(self):
        spec = SceneSpec(
            subject="A base", props=["crates", "Crates", "a radio"],
            style="pixel art",
        )
        p = spec.to_prompt().lower()
        assert p.count("crates") == 1

    def test_result_is_single_line(self):
        assert "\n" not in self._spec().to_prompt()

    def test_no_trigger_when_not_supplied(self):
        spec = SceneSpec(subject="A tower", style="pixel art")
        assert not spec.to_prompt().startswith(".")


# ════════════════════════════════════════════════════════════════════
# Pipeline integration
# ════════════════════════════════════════════════════════════════════

class TestFromVisualScene:
    """The pipeline feeds prose descriptions; they must route safely."""

    def test_boilerplate_prefix_stripped(self):
        spec = from_visual_scene({
            "description": "16-bit isometric pixel art scene: A radar tower in the desert."
        })
        assert "16-bit isometric pixel art scene:" not in spec.to_prompt()
        assert "radar tower" in spec.to_prompt()

    def test_quoted_literals_lifted_into_text_policy(self):
        spec = from_visual_scene({
            "description": "A crate marked 'EMPTY' beside a sign reading '$1,000,000'."
        })
        dispositions = {t.text: t.disposition for t in spec.text_requirements}
        assert "EMPTY" in dispositions
        assert "$1,000,000" in dispositions
        assert dispositions["$1,000,000"] == TEXT_EXACT

    def test_exact_values_not_in_assembled_prompt(self):
        spec = from_visual_scene({
            "description": "A display showing '33,000,000' at a base."
        })
        assert "33,000,000" not in spec.to_prompt()
        assert "33,000,000" in spec.deferred_text

    def test_no_category_infographic_injection(self):
        """The regression that caused repeated compositions."""
        spec = from_visual_scene({"description": "Two officials at a table."})
        prompt = spec.to_prompt().lower()
        for banned in ("tactical map layout", "territory indicators",
                       "resource flow arrows", "data tension visualization"):
            assert banned not in prompt

    def test_handles_empty_description(self):
        spec = from_visual_scene({"description": ""})
        assert spec.to_prompt() is not None  # must not raise

    def test_to_dict_round_trips(self):
        import json
        spec = from_visual_scene({
            "description": "A crate marked 'EMPTY'."
        }, trigger="Pixel Art")
        d = spec.to_dict()
        assert json.dumps(d)  # serialisable
        assert d["assembled_prompt"] == spec.to_prompt()
        assert "EMPTY" in d["deferred_text"] or "EMPTY" in str(d["text_requirements"])
