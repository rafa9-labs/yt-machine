"""Behavioural test for the word-budget enforcement, without an LLM.

``synthesize_multi_news_script`` is the only place the length band is enforced,
and it needs a model. This drives the real function with a stub ``generate``
that returns a deliberately over-long script, so the compression path executes
for real and the counting, trimming and re-checking are all the production code.

WHY THIS EXISTS, beyond the arithmetic tests: the first version of the fix used
structlog-style keyword arguments on a stdlib logger. That raises TypeError at
call time, so every unit test passed while the over-budget branch would crash in
production. Only driving the real path with a stub exposed it. This test is the
regression guard for that.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from src.video.pipeline_config import (  # noqa: E402
    BEAT_WORD_RANGES,
    MAX_WORDS,
    MIN_WORDS,
    count_narrated_words,
)


def _story(lengths: dict) -> dict:
    story = {
        "part_1_narration": "", "part_2_narration": "", "real_talk": "",
        "fallout": "", "segue": "",
        "part_1_visual": "v", "part_2_visual": "v",
        "real_talk_visual": "v", "fallout_visual": "v",
    }
    for field, count in lengths.items():
        story[field] = " ".join([field[:2]] * count)
    return story


# Mirrors the reference run that exposed the defect: ~323 narrated words.
OVER_LONG = {
    "greeting": "", "intro_hook": "",
    "stories": [
        _story({"part_1_narration": 50, "part_2_narration": 52, "real_talk": 13,
                "fallout": 34, "segue": 17}),
        _story({"part_1_narration": 48, "part_2_narration": 49, "real_talk": 13,
                "fallout": 29, "segue": 0}),
    ],
    "closing": " ".join(["cl"] * 18),
}

# What a compliant response looks like, at each beat's midpoint.
_MID = {f: (lo + hi) // 2 for f, (lo, hi) in BEAT_WORD_RANGES.items()}
COMPLIANT = {
    "greeting": "", "intro_hook": "",
    "stories": [
        _story({**_MID}),
        _story({**_MID, "segue": 0}),
    ],
    "closing": " ".join(["cl"] * 16),
}


def _analysis() -> list:
    return [
        {"topic": "Test topic one", "impact_score": 8, "key_facts": ["f1", "f2"],
         "angle": "a1", "second_order_consequence": "c1"},
        {"topic": "Test topic two", "impact_score": 7, "key_facts": ["f3"],
         "angle": "a2", "second_order_consequence": "c2"},
    ]


@pytest.fixture()
def llm_with_stub():
    from src.brain.llm_interface import LLMInterface

    interface = LLMInterface(config_path=str(REPO / "config" / "system_prompts.json"))
    calls = {"compression": 0}

    def fake_generate(prompt, **kwargs):
        if "COMPRESS" in prompt or "TOO LONG" in prompt:
            calls["compression"] += 1
            return json.dumps(COMPLIANT)
        return json.dumps(OVER_LONG)

    interface.generate = fake_generate
    return interface, calls


class TestCompressionEngages:
    def test_over_long_script_is_compressed_into_the_band(self, llm_with_stub):
        """The behaviour the fix was for: 323 words in, 150-175 out."""
        interface, calls = llm_with_stub

        result = interface.synthesize_multi_news_script(_analysis(), num_stories=2)

        assert result is not None, "synthesis returned nothing"
        assert calls["compression"] > 0, (
            "compression never engaged on a 323-word script; the ceiling is "
            f"{MAX_WORDS}"
        )
        final = count_narrated_words(result)
        assert MIN_WORDS <= final <= MAX_WORDS, (
            f"final script is {final} words, outside the {MIN_WORDS}-{MAX_WORDS} band"
        )

    def test_the_over_budget_branch_runs_without_raising(self, llm_with_stub):
        """Regression for the logging defect.

        The over-budget branch logs a warning. With structlog-style kwargs on a
        stdlib logger it raised TypeError — after the script was already
        produced, so the run failed late and confusingly.

        The stub always returns the over-long script, including in response to
        compression requests. The attempt limit is therefore exhausted and the
        "could not compress" warning fires, which is the call that used to
        raise. MAX_WORDS is a local inside the function, so the branch is
        reached by refusing to comply rather than by patching the constant.
        """
        from src.brain.llm_interface import LLMInterface

        interface = LLMInterface(
            config_path=str(REPO / "config" / "system_prompts.json")
        )

        def always_over_long(prompt, **kwargs):
            return json.dumps(OVER_LONG)

        interface.generate = always_over_long

        # Must not raise. The script ships over budget, which is the documented
        # advisory behaviour, and the warning is logged rather than thrown.
        result = interface.synthesize_multi_news_script(_analysis(), num_stories=2)
        assert result is not None, "synthesis returned nothing"
        assert count_narrated_words(result) > MAX_WORDS, (
            "the stub never complies, so the result should remain over budget"
        )

    def test_compliant_script_is_not_compressed(self, llm_with_stub):
        """No needless LLM spend when the script is already in band."""
        from src.brain.llm_interface import LLMInterface

        interface = LLMInterface(
            config_path=str(REPO / "config" / "system_prompts.json")
        )
        calls = {"compression": 0}

        def fake_generate(prompt, **kwargs):
            if "COMPRESS" in prompt or "TOO LONG" in prompt:
                calls["compression"] += 1
            return json.dumps(COMPLIANT)

        interface.generate = fake_generate
        result = interface.synthesize_multi_news_script(_analysis(), num_stories=2)

        assert result is not None
        assert calls["compression"] == 0, (
            "a compliant script triggered compression"
        )
