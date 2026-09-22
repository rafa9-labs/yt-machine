"""Satisfiability tests for the script-length enforcement.

The defect these guard against: the compression instruction told the model to
trim ``part_1``/``part_2`` while leaving ``real_talk``, ``fallout`` and ``segue``
"as-is". On the reference run those locked fields were 106 words and the minimum
trimmed parts were 80, so the best achievable total was 186 against a 170
ceiling. **No compliant script existed**, which is why all three retries failed
and every real run shipped at 296-325 words instead of the intended 150-175.

Nothing failed loudly: the model simply could not satisfy an impossible
instruction, and the pipeline used "best available". These tests assert the
instruction is satisfiable *before* anyone spends an LLM call on it.
"""

from pathlib import Path

import pytest

from src.video.pipeline_config import (
    BEAT_WORD_RANGES,
    DEFAULT_NUM_STORIES,
    MAX_WORDS,
    MIN_WORDS,
    beat_budget_bounds,
)


class TestBudgetIsSatisfiable:
    def test_per_beat_ranges_can_produce_a_compliant_total(self):
        """There must exist a script satisfying both the beats and the band."""
        bounds = beat_budget_bounds()
        overlapping = max(bounds["lowest"], MIN_WORDS) <= min(
            bounds["highest"], MAX_WORDS
        )
        assert overlapping, (
            f"no script can satisfy both the beat ranges "
            f"({bounds['lowest']}-{bounds['highest']}) and the band "
            f"({MIN_WORDS}-{MAX_WORDS})"
        )

    def test_every_field_is_trimmable(self):
        """No field may be exempt from trimming.

        The original instruction locked three fields, which is what made the
        target unreachable. A field with no upper bound is an unsatisfiable
        constraint waiting to happen.
        """
        for field, (lo, hi) in BEAT_WORD_RANGES.items():
            assert hi >= lo, f"{field} has an inverted range"
            assert hi < 100, (
                f"{field} allows {hi} words, which cannot be reconciled with a "
                f"{MAX_WORDS}-word total across {DEFAULT_NUM_STORIES} stories"
            )

    def test_trimming_all_beats_reaches_the_floor(self):
        """If the model trims every beat to its minimum, that must be <= MIN."""
        assert beat_budget_bounds()["lowest"] <= MIN_WORDS

    def test_maximum_beats_still_fit_the_ceiling(self):
        """If the model writes to every maximum, that must be >= MIN too."""
        bounds = beat_budget_bounds()
        assert bounds["highest"] >= MIN_WORDS
        assert bounds["highest"] >= MAX_WORDS


class TestCompressionInstructionIsSatisfiable:
    """Read the actual instruction and prove it asks for something possible."""

    @staticmethod
    def _instruction() -> str:
        source = (
            Path(__file__).parent.parent / "src" / "brain" / "llm_interface.py"
        ).read_text(encoding="utf-8")
        # The compression prompt is the f-string block that names every beat.
        marker = "COMPRESS every field to fit"
        start = source.index(marker)
        return source[start:source.index('Return ONLY the corrected JSON with all', start)]

    def test_instruction_names_every_beat(self):
        """The fix: every field must be trimmable, none exempt.

        The old instruction listed only part_1/part_2 and said the rest were
        "keep as-is" — the source of the impossibility.
        """
        instruction = self._instruction()
        for field in BEAT_WORD_RANGES:
            assert field in instruction, (
                f"compression instruction omits {field}; a field it cannot "
                "touch is a constraint it cannot satisfy"
            )

    def test_instruction_does_not_exempt_any_field(self):
        instruction = self._instruction().lower()
        assert "keep as-is" not in instruction, (
            "compression instruction exempts fields from trimming, which "
            "previously made the target unreachable"
        )

    def test_instruction_binds_every_field_to_the_shared_constants(self):
        """The instruction's numbers must come from the code, not be retyped.

        The block is an f-string, so the ranges appear as
        ``{_BEAT_LO['field']}-{_BEAT_HI['field']}``. Binding to the constants is
        what prevents the prompt text and the enforcement from drifting apart
        again — they were independently hardcoded at three different values.
        """
        instruction = self._instruction()
        for field in BEAT_WORD_RANGES:
            assert f"_BEAT_LO['{field}']" in instruction, (
                f"instruction does not bind {field}'s lower bound to the constant"
            )
            assert f"_BEAT_HI['{field}']" in instruction, (
                f"instruction does not bind {field}'s upper bound to the constant"
            )
        assert "{MIN_WORDS}-{MAX_WORDS}" in instruction

    def test_bound_constants_are_imported(self):
        """The names the instruction interpolates must actually exist."""
        source = (
            Path(__file__).parent.parent / "src" / "brain" / "llm_interface.py"
        ).read_text(encoding="utf-8")
        assert "_BEAT_LO = {" in source
        assert "_BEAT_HI = {" in source
        assert "BEAT_WORD_RANGES," in source, "constants are not imported"

    def test_instruction_target_is_achievable_by_its_own_ranges(self):
        """The strongest form: the stated per-field ranges satisfy the target.

        Computes the sum the instruction describes and asserts it can land
        inside the band it names. This is the assertion that would have failed
        on the original instruction.
        """
        bounds = beat_budget_bounds()
        assert bounds["lowest"] <= MAX_WORDS, (
            "following the instruction's own ranges, the minimum achievable "
            f"total ({bounds['lowest']}) exceeds the ceiling ({MAX_WORDS})"
        )
        assert bounds["highest"] >= MIN_WORDS, (
            "following the instruction's own ranges, the maximum achievable "
            f"total ({bounds['highest']}) falls short of the floor ({MIN_WORDS})"
        )
