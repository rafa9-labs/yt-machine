"""Tests for validated pipeline run-size configuration."""

import pytest

from src.video.pipeline_config import DEFAULT_NUM_IMAGES, resolve_image_limit


def test_image_limit_defaults_to_full_production_count():
    assert resolve_image_limit(None) == DEFAULT_NUM_IMAGES
    assert resolve_image_limit("") == DEFAULT_NUM_IMAGES


@pytest.mark.parametrize("raw, expected", [("1", 1), ("4", 4), ("8", 8)])
def test_image_limit_accepts_supported_counts(raw, expected):
    assert resolve_image_limit(raw) == expected


@pytest.mark.parametrize("raw", ["0", "9", "four", "2.5"])
def test_image_limit_rejects_invalid_counts(raw):
    with pytest.raises(ValueError, match="YT_IMAGE_LIMIT"):
        resolve_image_limit(raw)


# ══════════════════════════════════════════════════════════════
# Retention configuration
# ══════════════════════════════════════════════════════════════
from src.video.pipeline_config import (  # noqa: E402
    DEFAULT_RETENTION_DAYS,
    DEFAULT_RETENTION_KEEP_LAST,
    DEFAULT_RETENTION_MODE,
    resolve_retention_days,
    resolve_retention_enabled,
    resolve_retention_keep_last,
    resolve_retention_mode,
)


class TestRetentionDefaults:
    def test_defaults_are_report_only(self):
        """The default must never delete: deletion requires an explicit act."""
        assert resolve_retention_mode(None) == "report"
        assert resolve_retention_mode("") == "report"
        assert DEFAULT_RETENTION_MODE == "report"
        assert resolve_retention_enabled(None) is True

    def test_defaults(self):
        assert resolve_retention_days(None) == 30
        assert resolve_retention_keep_last(None) == 3
        assert DEFAULT_RETENTION_DAYS == 30
        assert DEFAULT_RETENTION_KEEP_LAST == 3


class TestRetentionValidation:
    @pytest.mark.parametrize("raw, expected", [("0", 0), ("7", 7), ("365", 365)])
    def test_days_accepts_valid(self, raw, expected):
        assert resolve_retention_days(raw) == expected

    @pytest.mark.parametrize("raw", ["-1", "abc", "2.5", "99999"])
    def test_days_rejects_invalid(self, raw):
        with pytest.raises(ValueError, match="YT_RETENTION_DAYS"):
            resolve_retention_days(raw)

    @pytest.mark.parametrize("raw, expected", [("0", 0), ("10", 10)])
    def test_keep_last_accepts_valid(self, raw, expected):
        assert resolve_retention_keep_last(raw) == expected

    @pytest.mark.parametrize("raw", ["-1", "x", "99999"])
    def test_keep_last_rejects_invalid(self, raw):
        with pytest.raises(ValueError, match="YT_RETENTION_KEEP_LAST"):
            resolve_retention_keep_last(raw)

    @pytest.mark.parametrize("raw", ["full", "delivery_only", "off", "report", "FULL"])
    def test_mode_accepts_known_values(self, raw):
        assert resolve_retention_mode(raw) in ("full", "delivery_only", "off", "report")

    @pytest.mark.parametrize("raw", ["delete", "yes", "rm -rf"])
    def test_mode_rejects_unknown(self, raw):
        with pytest.raises(ValueError, match="YT_RETENTION_MODE"):
            resolve_retention_mode(raw)

    @pytest.mark.parametrize("raw, expected", [
        ("true", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("0", False), ("no", False), ("disabled", False),
    ])
    def test_enabled_toggle(self, raw, expected):
        assert resolve_retention_enabled(raw) is expected


# ══════════════════════════════════════════════════════════════
# Script word budget
# ══════════════════════════════════════════════════════════════
from src.video.pipeline_config import (  # noqa: E402
    BEAT_WORD_RANGES,
    MAX_WORDS,
    MIN_WORDS,
    TARGET_VIDEO_SECONDS,
    WORDS_PER_SECOND,
    beat_budget_bounds,
    count_narrated_words,
)


class TestWordBudget:
    def test_band_derives_from_the_target_duration(self):
        """Changing the duration must move the band — that is the whole point."""
        assert MIN_WORDS == int(TARGET_VIDEO_SECONDS[0] * WORDS_PER_SECOND)
        assert MAX_WORDS == int(TARGET_VIDEO_SECONDS[1] * WORDS_PER_SECOND)

    def test_band_matches_the_stated_target(self):
        assert TARGET_VIDEO_SECONDS == (60, 70)
        assert (MIN_WORDS, MAX_WORDS) == (150, 175)

    def test_beat_ranges_can_reach_the_global_band(self):
        """Satisfiability at the configuration level.

        The defect this guards against: the per-beat limits summed to a range
        that could not reach the global band, so no compliant script existed.
        """
        b = beat_budget_bounds()
        assert b["lowest"] <= MIN_WORDS, (
            f"beats floor {b['lowest']} is above the band minimum {MIN_WORDS}"
        )
        assert b["highest"] >= MAX_WORDS, (
            f"beats ceiling {b['highest']} is below the band maximum {MAX_WORDS}"
        )

    def test_beat_ranges_are_ordered(self):
        for field, (lo, hi) in BEAT_WORD_RANGES.items():
            assert 0 < lo <= hi, f"{field}: {lo}-{hi} is not a valid range"

    def test_segue_is_shorter_than_the_narrative_beats(self):
        """A 2-4s bridge should not be budgeted like a 10s beat."""
        segue_hi = BEAT_WORD_RANGES["segue"][1]
        for field in ("part_1_narration", "part_2_narration", "real_talk", "fallout"):
            assert BEAT_WORD_RANGES[field][0] >= segue_hi, (
                f"{field} minimum should exceed the segue maximum"
            )


class TestCountNarratedWords:
    @staticmethod
    def _script(**overrides):
        script = {
            "greeting": "",
            "intro_hook": "",
            "stories": [
                {
                    "part_1_narration": "one two three four five",
                    "part_2_narration": "six seven eight nine ten",
                    "real_talk": "eleven twelve",
                    "fallout": "thirteen fourteen",
                    "segue": "fifteen sixteen",
                }
            ],
            "closing": "seventeen eighteen nineteen",
        }
        script.update(overrides)
        return script

    def test_counts_all_spoken_fields(self):
        # 5 + 5 + 2 + 2 + 2(segue) + 3(closing) = 19
        assert count_narrated_words(self._script()) == 19

    def test_includes_the_closing(self):
        """The closing is spoken; excluding it is what caused the 305-vs-325 gap."""
        with_closing = self._script()
        without = self._script(closing="")
        assert (
            count_narrated_words(with_closing)
            - count_narrated_words(without)
            == 3
        )

    def test_timeline_separators_are_not_counted(self):
        """Separators ('....') are pauses, not words.

        The counter prefers the segment timeline when present, since that is
        the authoritative narration order `full_text` derives from. Separator
        segments must contribute zero words.
        """
        script = self._script()
        script["segment_timeline"] = [
            {"text": st, "label": f"seg{idx}"}
            for idx, st in enumerate(
                ["one two three", "....", "four five six", "......"]
            )
        ]
        assert count_narrated_words(script) == 6

    def test_timeline_takes_precedence_over_story_fields(self):
        """The timeline is authoritative when both are present.

        They normally agree (both derive from the same narration), but the
        timeline is what is actually rendered, so it wins.
        """
        script = self._script()
        script["segment_timeline"] = [{"text": "alpha beta gamma"}]
        assert count_narrated_words(script) == 3

    def test_falls_back_to_story_fields_without_a_timeline(self):
        """The synthesizer counts before the timeline exists."""
        script = self._script()
        script.pop("segment_timeline", None)
        assert count_narrated_words(script) == 19

    def test_handles_missing_and_malformed_input(self):
        assert count_narrated_words(None) == 0
        assert count_narrated_words({}) == 0
        assert count_narrated_words({"stories": None}) == 0
        assert count_narrated_words({"stories": ["not-a-dict"]}) == 0
        assert count_narrated_words({"segment_timeline": []}) == 0
        assert count_narrated_words({"segment_timeline": ["not-a-dict"]}) == 0

    def test_empty_fields_do_not_inflate_the_count(self):
        script = self._script(stories=[{
            "part_1_narration": "", "part_2_narration": "",
            "real_talk": "", "fallout": "", "segue": "",
        }], closing="")
        assert count_narrated_words(script) == 0

    def test_multi_story_sums(self):
        story = {
            "part_1_narration": "a b c", "part_2_narration": "d e",
            "real_talk": "f", "fallout": "g", "segue": "",
        }
        script = {"stories": [story, story], "closing": "", "greeting": ""}
        assert count_narrated_words(script) == 14
