"""
Unit tests for YT-Machine — no API calls, no network, fast.
Run: python -m pytest tests/test_unit.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from video_server.subtitle_renderer import (
    _clean_script_for_subtitles, _fuzzy_match, _split_into_phrases,
    _estimate_from_script, _clean_display
)
from video_server.split_video_assembler import _calculate_scene_durations


# ═══════════════════════════════════════════
# Timestamp gap bridging
# ═══════════════════════════════════════════
class TestBridgeTimestampGaps:
    def _make_times(self, pairs):
        return [{'start': s, 'end': e} for s, e in pairs]

    def test_no_gaps_unchanged(self):
        from pipeline_utils import bridge_timestamp_gaps
        times = self._make_times([(0, 10), (10, 20), (20, 30)])
        result = bridge_timestamp_gaps(times, 30.0)
        # Should be continuous
        for i in range(len(result) - 1):
            assert result[i]['end'] >= result[i+1]['start'] - 0.1

    def test_big_gap_bridged(self):
        from pipeline_utils import bridge_timestamp_gaps
        times = self._make_times([(0, 7.28), (22.0, 29.0), (29.0, 30.0)])
        result = bridge_timestamp_gaps(times, 30.0)
        # No gap should exist between image 0 and 1
        gap = result[1]['start'] - result[0]['end']
        assert gap <= 0.1, f"Gap of {gap:.2f}s remains"

    def test_none_values_filled(self):
        from pipeline_utils import bridge_timestamp_gaps
        times = [{'start': None, 'end': None} for _ in range(6)]
        result = bridge_timestamp_gaps(times, 90.0)
        assert all(t['start'] is not None for t in result)
        assert all(t['end'] is not None for t in result)
        assert result[0]['start'] == 0
        assert result[-1]['end'] >= 90.0

    def test_minimum_duration_enforced(self):
        from pipeline_utils import bridge_timestamp_gaps
        times = self._make_times([(0, 0.3), (0.3, 30), (30, 60), (60, 90)])
        result = bridge_timestamp_gaps(times, 90.0)
        for t in result:
            dur = t['end'] - t['start']
            assert dur >= 0.9, f"Duration {dur:.2f}s is below minimum"

    def test_total_duration_preserved(self):
        from pipeline_utils import bridge_timestamp_gaps
        times = [{'start': None, 'end': None} for _ in range(6)]
        result = bridge_timestamp_gaps(times, 80.0)
        total = sum(t['end'] - t['start'] for t in result)
        assert abs(total - 80.0) < 1.0, f"Total {total:.1f}s != 80.0s"


# ═══════════════════════════════════════════
# Subtitle helpers
# ═══════════════════════════════════════════
class TestCleanScript:
    def test_removes_ellipsis(self):
        assert '...' not in _clean_script_for_subtitles("Hello... world... test")

    def test_removes_quotes(self):
        cleaned = _clean_script_for_subtitles('He said "hello"')
        assert '"' not in cleaned

    def test_normalizes_whitespace(self):
        cleaned = _clean_script_for_subtitles("hello   world")
        assert cleaned == "hello world"

    def test_empty_string(self):
        assert _clean_script_for_subtitles("") == ""


class TestFuzzyMatch:
    def test_exact_match(self):
        assert _fuzzy_match("hello", "hello") is True

    def test_substring(self):
        assert _fuzzy_match("run", "running") is True

    def test_short_words(self):
        assert _fuzzy_match("ab", "cd") is False

    def test_different_words(self):
        assert _fuzzy_match("apple", "quantum") is False


class TestCleanDisplay:
    def test_strips_punctuation(self):
        assert _clean_display("hello.") == "HELLO"

    def test_empty_after_strip(self):
        assert _clean_display('"') == ''

    def test_normal_word(self):
        assert _clean_display("World") == "WORLD"


class TestEstimateFromScript:
    def test_basic(self):
        words = _estimate_from_script("one two three", 3.0)
        assert len(words) == 3
        assert words[-1]['end'] == 3.0

    def test_empty(self):
        assert _estimate_from_script("", 3.0) == []


class TestSplitIntoPhrases:
    def _make_words(self, text, dur_per_word=0.3):
        words = text.split()
        return [{'word': w, 'start': i*dur_per_word, 'end': (i+1)*dur_per_word}
                for i, w in enumerate(words)]

    def test_empty(self):
        assert _split_into_phrases([]) == []

    def test_single_word(self):
        words = self._make_words("hello")
        phrases = _split_into_phrases(words)
        assert len(phrases) >= 1

    def test_many_words(self):
        words = self._make_words(" ".join(f"word{i}" for i in range(50)))
        phrases = _split_into_phrases(words)
        assert len(phrases) > 1
        # All words covered
        covered = set()
        for s, e, _, _ in phrases:
            covered.update(range(s, e))
        assert len(covered) == 50


# ═══════════════════════════════════════════
# Scene duration calculation
# ═══════════════════════════════════════════
class TestSceneDurations:
    def test_zero_scenes(self):
        assert _calculate_scene_durations(90, 0) == []

    def test_one_scene(self):
        result = _calculate_scene_durations(90, 1)
        assert result == [90]

    def test_six_scenes_sum_to_total(self):
        result = _calculate_scene_durations(90, 6)
        assert abs(sum(result) - 90) < 0.01

    def test_all_positive(self):
        result = _calculate_scene_durations(90, 6)
        assert all(d > 0 for d in result)


# ═══════════════════════════════════════════
# Fallback prompt builder
# ═══════════════════════════════════════════
class TestBuildFallbackPrompt:
    def test_empty_text(self):
        from pipeline_utils import build_fallback_prompt
        result = build_fallback_prompt("", 0, 0, [])
        assert len(result) >= 20

    def test_with_locations(self):
        from pipeline_utils import build_fallback_prompt
        result = build_fallback_prompt("Russia attacks Ukraine forces", 0, 0, [])
        assert len(result) >= 20

    def test_with_numbers(self):
        from pipeline_utils import build_fallback_prompt
        result = build_fallback_prompt("Deployed 5,000 troops to the region", 0, 0, [])
        assert "5,000" in result or "units" in result

    def test_part_idx_affects_composition(self):
        from pipeline_utils import build_fallback_prompt
        r0 = build_fallback_prompt("Russia deployed missiles", 0, 0, [])
        r1 = build_fallback_prompt("Russia deployed missiles", 0, 1, [])
        # Part 0 should mention establishing, part 1 close-up
        assert r0 != r1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])