"""Unit tests for LLM response parsing — thinking tokens and JSON extraction.

These cover the pure, deterministic helpers that every LLM step depends on.
They run without a model, network, or database.

Ported from the former tests/test_pipeline_models.py, which was a live-service
script excluded from collection (it required a running Gemma model). The
config-routing and live-call sections were dropped because they tested
hardcoded model names and an Ollama instance, not behaviour.

Run: python -m pytest tests/test_llm_parsing.py -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def _strip(text: str) -> str:
    from src.brain.llm_interface import LLMInterface

    return LLMInterface._strip_thinking_tokens(text)


def _extract(text: str):
    from src.brain.llm_interface import LLMInterface

    return LLMInterface(config_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "system_prompts.json",
    ))._extract_json(text)


# ══════════════════════════════════════════════════════════════
# 1. _strip_thinking_tokens
# ══════════════════════════════════════════════════════════════
class TestStripThinkingTokens:
    def test_strips_think_wrapper(self):
        raw = ('<think\nLet me analyze this...\nSome reasoning here.\n'
               '</think\n\n{"topic": "test", "impact_score": 7}')
        out = _strip(raw)
        assert '"topic"' in out
        assert "<think" not in out

    def test_handles_unclosed_think_tag(self):
        out = _strip('<think\nReasoning about stuff...\n{"topic": "test"}')
        assert '"topic"' in out
        assert "<think" not in out

    def test_strips_channel_thought_tokens(self):
        raw = ('<|channel>thought<channel|>analysis here'
               '<|channel>output<channel|>{"topic": "test"}')
        out = _strip(raw)
        assert '"topic"' in out
        assert "channel" not in out

    def test_clean_json_passes_through(self):
        raw = '{"topic": "Iran sanctions", "impact_score": 8}'
        assert _strip(raw) == raw

    def test_strips_markdown_fence(self):
        out = _strip('```json\n{"topic": "test"}\n```')
        assert out.startswith('{"topic"')
        assert "```" not in out

    def test_drops_leading_prose_before_json(self):
        out = _strip('Here is the result:\n{"topic": "test"}')
        assert out.startswith('{"topic"')

    def test_empty_input_is_safe(self):
        assert _strip("") == ""


# ══════════════════════════════════════════════════════════════
# 2. _extract_json
# ══════════════════════════════════════════════════════════════
class TestExtractJson:
    def test_parses_plain_json(self):
        out = _extract('{"topic": "test", "impact_score": 5}')
        assert out == {"topic": "test", "impact_score": 5}

    def test_parses_json_wrapped_in_thinking(self):
        raw = '<think\nAnalyzing...\n</think\n\n{"topic": "Iran", "impact_score": 8}'
        out = _extract(raw)
        assert out is not None
        assert out["topic"] == "Iran"

    def test_parses_json_in_markdown_block(self):
        out = _extract('```json\n{"topic": "test", "impact_score": 5}\n```')
        assert out is not None
        assert out["topic"] == "test"

    def test_fixes_trailing_commas(self):
        out = _extract('{"topic": "test", "impact_score": 8,}')
        assert out is not None
        assert out["topic"] == "test"

    def test_auto_closes_incomplete_json(self):
        raw = '{"topic": "test", "impact_score": 8, "key_facts": ["fact1", "fact2"'
        out = _extract(raw)
        assert out is not None
        assert out["topic"] == "test"

    def test_returns_none_when_no_json_present(self):
        assert _extract("I cannot answer that.") is None

    def test_returns_none_for_empty_input(self):
        assert _extract("") is None

    def test_extracts_first_complete_object_from_prose(self):
        raw = 'Sure! Here you go: {"topic": "a"} and some trailing words'
        out = _extract(raw)
        assert out == {"topic": "a"}
