"""
Tests for local (MLX-Gen) prompt construction.

These cover the three defects that produced low-quality images in the first
end-to-end run, so they cannot silently regress:

  1. Style-block stacking — three overlapping style clauses plus a palette
     drowned the scene description.
  2. Text-render requests — scenes asked for literal characters, which FLUX
     renders as scrambled glyphs.
  3. Inert fal.ai LoRA trigger words — injected into local prompts where no
     LoRA is loaded.

Run: .venv/bin/python -m pytest tests/test_local_prompt_building.py -v
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from src.video.pixel_art_tool import (
    LOCAL_MAX_SCENE_WORDS,
    LOCAL_MAX_STYLE_WORDS,
    _strip_lora_triggers,
    _trim_to_words,
    build_local_generation_prompt,
    sanitize_text_requests,
)


# ════════════════════════════════════════════════════════════════════
# Word budgeting
# ════════════════════════════════════════════════════════════════════

class TestTrimToWords:
    def test_short_text_unchanged(self):
        assert _trim_to_words("a short scene", 10) == "a short scene"

    def test_prefers_sentence_boundary(self):
        text = "First sentence here. Second sentence follows. Third one is cut."
        # A 6-word budget fits sentence one (3) + sentence two (3) exactly.
        assert _trim_to_words(text, 6) == "First sentence here. Second sentence follows."
        # A tighter budget keeps only the first complete sentence.
        assert _trim_to_words(text, 5) == "First sentence here."

    def test_oversized_single_sentence_is_hard_trimmed(self):
        """One long sentence must not bypass the budget just because no
        sentence boundary fits."""
        text = " ".join(["word"] * 100) + "."
        assert len(_trim_to_words(text, 20).split()) == 20

    def test_hard_trims_when_no_sentence_fits(self):
        text = "one two three four five six seven eight"
        result = _trim_to_words(text, 4)
        assert len(result.split()) <= 4
        assert result.endswith(".")

    def test_empty_text(self):
        assert _trim_to_words("", 5) == ""


# ════════════════════════════════════════════════════════════════════
# Text-request sanitization
# ════════════════════════════════════════════════════════════════════

class TestSanitizeTextRequests:
    def test_quoted_number_with_digits(self):
        text = "a counter displays '33,000,000,000' in red pixelated digits"
        cleaned, replaced = sanitize_text_requests(text)
        assert "'33,000,000,000'" not in cleaned
        assert "digits" not in cleaned.lower()
        assert replaced

    def test_quoted_label_on_crates(self):
        cleaned, replaced = sanitize_text_requests("crates marked with red 'EMPTY' tags")
        assert "'EMPTY'" not in cleaned
        assert "symbol" in cleaned
        assert replaced

    def test_text_fragments_becomes_data_pulses(self):
        cleaned, replaced = sanitize_text_requests("a stream with pixelated text fragments")
        assert "text fragments" not in cleaned.lower()
        assert "glowing data pulses" in cleaned
        assert replaced

    def test_words_phrase_removed(self):
        cleaned, _ = sanitize_text_requests("a screen shows the words 'OIL SPIKE' in green")
        assert "'OIL SPIKE'" not in cleaned
        assert "the words" not in cleaned.lower()

    def test_single_quoted_letter(self):
        cleaned, _ = sanitize_text_requests("clipboard with red 'X' marks")
        assert "'X'" not in cleaned
        assert "\\1" not in cleaned  # no unresolved backreference

    def test_digital_counter(self):
        cleaned, _ = sanitize_text_requests("a large digital counter on the left")
        assert "counter" not in cleaned.lower()
        assert "indicator gauge" in cleaned

    def test_clean_text_untouched(self):
        text = "A data chart shows stockpiles dropping to near-zero levels."
        cleaned, replaced = sanitize_text_requests(text)
        assert cleaned == text
        assert replaced == []

    def test_no_double_articles_or_duplicate_words(self):
        """Substitutions must not create 'a large indicator gauge showing a
        large red graphic markers' style output."""
        text = "a large digital counter on the left displays '33,000,000,000' in red pixelated digits"
        cleaned, _ = sanitize_text_requests(text)
        assert "  " not in cleaned
        # No repeated adjacent words
        words = cleaned.lower().split()
        for a, b in zip(words, words[1:]):
            assert a != b, f"duplicate word in: {cleaned}"

    def test_no_unresolved_backreferences(self):
        """A regex backreference must never leak as literal text."""
        samples = [
            "clipboard with red 'X' marks",
            "green 'GO' sign",
            "a white 'STOP' label",
        ]
        for sample in samples:
            cleaned, _ = sanitize_text_requests(sample)
            assert "\\1" not in cleaned
            assert "\\g" not in cleaned

    def test_idempotent(self):
        """Running the sanitizer twice must not corrupt the text."""
        text = "a counter displays '99' in red digits and crates marked 'FULL'"
        once, _ = sanitize_text_requests(text)
        twice, _ = sanitize_text_requests(once)
        assert once == twice


# ════════════════════════════════════════════════════════════════════
# LoRA trigger stripping
# ════════════════════════════════════════════════════════════════════

class TestStripLoraTriggers:
    def test_removes_retro_pixel_trigger(self):
        cleaned = _strip_lora_triggers("Retro Pixel, a desert base at dusk")
        assert "retro pixel" not in cleaned.lower()
        assert "desert base" in cleaned

    def test_removes_trigger_without_comma(self):
        cleaned = _strip_lora_triggers("Retro Pixel a desert base")
        assert "retro pixel" not in cleaned.lower()

    def test_preserves_normal_text(self):
        text = "two figures at a table with a glowing map"
        assert _strip_lora_triggers(text) == text

    def test_handles_case_variation(self):
        cleaned = _strip_lora_triggers("retro pixel: a trading floor")
        assert "retro pixel" not in cleaned.lower()


# ════════════════════════════════════════════════════════════════════
# Prompt builder
# ════════════════════════════════════════════════════════════════════

class TestBuildLocalGenerationPrompt:
    def test_scene_comes_first(self):
        """Scene-first ordering keeps composition grounded.

        When a long style block led, the model produced stylised images
        with no focal subject.
        """
        prompt = build_local_generation_prompt(
            "Two figures sit at a table with a glowing map of the Gulf."
        )
        assert prompt.lower().startswith("two figures")

    def test_style_tail_is_short(self):
        prompt = build_local_generation_prompt("A radar tower in the desert.")
        style_part = prompt.split(".", 1)[1]
        assert len(style_part.split()) <= LOCAL_MAX_STYLE_WORDS

    def test_scene_is_capped(self):
        long_scene = " ".join(["word"] * 200) + "."
        prompt = build_local_generation_prompt(long_scene, style_suffix="chunky pixels")
        # Style tail plus the capped scene
        assert len(prompt.split()) <= LOCAL_MAX_SCENE_WORDS + LOCAL_MAX_STYLE_WORDS + 2

    def test_boilerplate_prefix_removed(self):
        prompt = build_local_generation_prompt(
            "16-bit isometric pixel art scene: A radar tower in the desert."
        )
        assert "16-bit isometric pixel art scene:" not in prompt
        assert "radar tower" in prompt

    def test_no_triple_style_repetition(self):
        """The regression that caused the quality drop: the same style
        concept stated three times plus a palette."""
        prompt = build_local_generation_prompt(
            "A radar tower in the desert.",
            style_suffix="16-bit isometric pixel art, isometric perspective, vibrant colors",
        )
        lowered = prompt.lower()
        assert lowered.count("isometric pixel art") <= 1
        assert "color palette:" not in lowered

    def test_explicit_style_override_respected(self):
        prompt = build_local_generation_prompt(
            "A radar tower.", style_suffix="chunky pixels, flat colors"
        )
        assert "chunky pixels" in prompt

    def test_empty_description_falls_back_to_style(self):
        prompt = build_local_generation_prompt("")
        assert prompt  # never an empty prompt

    def test_result_is_single_line(self):
        prompt = build_local_generation_prompt("A radar tower in the desert.")
        assert "\n" not in prompt


# ════════════════════════════════════════════════════════════════════
# End-to-end prompt quality
# ════════════════════════════════════════════════════════════════════

class TestRealScenePrompts:
    """The prompt actually sent must be much shorter than the fal prompt
    and must never contain literal text requests."""

    SCENE = (
        "16-bit isometric pixel art scene: A pixelated U.S. military base in "
        "the Gulf shows damaged buildings with smoke rising, while a large "
        "digital counter on the left displays '33,000,000,000' in red "
        "pixelated digits. Munitions crates are stacked but marked with red "
        "'EMPTY' tags."
    )

    def test_text_requests_gone(self):
        cleaned, _ = sanitize_text_requests(self.SCENE)
        cleaned = _strip_lora_triggers(cleaned)
        prompt = build_local_generation_prompt(cleaned)
        for token in ("'33,000,000,000'", "'EMPTY'", "counter", "digits"):
            assert token not in prompt, f"leaked {token!r} in: {prompt}"

    def test_prompt_is_far_shorter_than_legacy(self):
        """The legacy prompt ran ~150 words; local must stay lean."""
        cleaned, _ = sanitize_text_requests(self.SCENE)
        prompt = build_local_generation_prompt(_strip_lora_triggers(cleaned))
        assert len(prompt.split()) < 90, prompt

    def test_prompt_has_scene_subject(self):
        cleaned, _ = sanitize_text_requests(self.SCENE)
        prompt = build_local_generation_prompt(_strip_lora_triggers(cleaned))
        assert "military base" in prompt.lower()
        assert "smoke" in prompt.lower()
