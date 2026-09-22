"""Contract tests for the LLM prompt <-> Pydantic schema boundary.

LangChain parses the analysis response into ``NewsAnalysis`` with a
``PydanticOutputParser``, which **silently discards any key the model does not
declare**. The prompt is free to ask for whatever it likes; the schema decides
what survives.

That mismatch hid three fields for months: ``shift_vector``,
``pixel_art_prompts`` and ``ticker_headlines`` were requested on every analysis
call, and every one was dropped before the pipeline saw it. Nothing failed
loudly — the call succeeded, the script was produced, and the values simply were
not there.

These tests assert the boundary in both directions so a future prompt edit
cannot quietly reintroduce a field the model will not carry.
"""

import json
import re
from pathlib import Path

import pytest

from src.models.schemas import NewsAnalysis

CONFIG = Path(__file__).parent.parent / "config" / "system_prompts.json"


# Fields the prompt asked for historically and that were dropped. Removing them
# from the prompt was the fix; reintroducing one without adding it to the schema
# would silently restore the bug.
RETIRED_ANALYSIS_FIELDS = ("shift_vector", "pixel_art_prompts", "ticker_headlines")


def _analysis_prompt() -> str:
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    return data["prompts"]["news_processor"]["system_prompt"]


def _requested_keys(prompt: str) -> list:
    """Keys named in the prompt's JSON template."""
    template = re.search(r"\{.*\}", prompt, re.S)
    if not template:
        pytest.fail("analysis prompt no longer contains a JSON template")
    return re.findall(r'"(\w+)":', template.group(0))


class TestPromptSchemaAgreement:
    def test_prompt_only_requests_keys_the_schema_carries(self):
        """Every requested key must survive parsing.

        A requested-but-undeclared key is silently dropped; this is the check
        that would have caught the original defect.
        """
        missing = [k for k in _requested_keys(_analysis_prompt())
                   if k not in NewsAnalysis.model_fields]
        assert not missing, (
            "the analysis prompt requests keys NewsAnalysis does not declare, "
            f"so the parser will discard them: {missing}"
        )

    def test_retired_fields_are_not_requested(self):
        prompt = _analysis_prompt()
        present = [f for f in RETIRED_ANALYSIS_FIELDS if f in prompt]
        assert not present, (
            f"retired analysis fields requested again: {present}. Either the "
            "field is genuinely needed (add it to NewsAnalysis and give it a "
            "consumer) or it should stay removed."
        )

    def test_prompt_tells_the_model_not_to_add_extra_keys(self):
        """Undirected models add fields; the prompt must forbid that."""
        prompt = _analysis_prompt().lower()
        assert "do not add keys" in prompt or "not listed above" in prompt

    def test_schema_fields_are_reachable_or_intentional(self):
        """Sanity: the declared fields are the ones the pipeline reads."""
        declared = set(NewsAnalysis.model_fields)
        for field in ("topic", "impact_score", "key_facts", "angle",
                      "second_order_consequence"):
            assert field in declared, f"{field} is read by the pipeline but undeclared"


class TestRetiredFieldsStayRetired:
    # Modules verified as unreachable from the pipeline (no importer anywhere).
    # They may still mention retired fields; they cannot execute. If one of
    # these is ever wired back in, the dead-code inventory in PIPELINE.md §13
    # is the place to revisit — and this exemption list must shrink with it.
    UNREACHABLE_MODULES = {
        "src/collector/prompt_generator.py",
        "src/collector/prompt_validator.py",
        "src/collector/visual_extractor.py",
        "src/collector/salience_extractor.py",
        "src/collector/historical_analyzer.py",
        "src/collector/script_parser.py",
        "src/collector/action_mapping.py",
        "src/collector/historical_equipment_db.py",
    }

    def test_no_reachable_code_reads_the_retired_analysis_fields(self):
        """Retired fields must not be read by anything the pipeline can call.

        The analysis prompt no longer produces these keys, so any live reader
        would silently receive None. Unreachable modules are exempt and listed
        explicitly above.
        """
        repo = Path(__file__).parent.parent
        offenders = []
        for path in list((repo / "tools").rglob("*.py")) + list(
            (repo / "src").rglob("*.py")
        ):
            rel = path.relative_to(repo).as_posix()
            if rel in self.UNREACHABLE_MODULES:
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for field in RETIRED_ANALYSIS_FIELDS:
                    if f"'{field}'" in line or f'"{field}"' in line:
                        offenders.append(f"{rel}:{lineno} {stripped[:60]}")
        assert not offenders, (
            "retired analysis fields are read by reachable code: "
            + ", ".join(offenders)
        )

    def test_exempt_modules_are_genuinely_unreachable(self):
        """Keep the exemption honest: nothing may import these."""
        repo = Path(__file__).parent.parent
        imported = []
        for path in list((repo / "tools").rglob("*.py")) + list(
            (repo / "src").rglob("*.py")
        ):
            rel = path.relative_to(repo).as_posix()
            if rel in self.UNREACHABLE_MODULES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for module in self.UNREACHABLE_MODULES:
                stem = module.rsplit("/", 1)[-1].removesuffix(".py")
                if re.search(rf"(from|import)\s+[\w.\s]*\b{stem}\b", text):
                    imported.append(f"{rel} imports {stem}")
        assert not imported, (
            "modules listed as unreachable now have importers, so the "
            "retired-field exemption is no longer valid: " + ", ".join(imported)
        )


class TestPromptBeatBudgetAgreement:
    def test_prompt_word_targets_match_the_code_budget(self):
        """The prompt's numbers must be the ones the code enforces.

        Three budgets once disagreed (enforced 130-170, requested 150-170,
        per-segment 112-197). They now derive from pipeline_config; this test
        fails if someone edits the prompt text without the constants.
        """
        from src.video.pipeline_config import BEAT_WORD_RANGES, MIN_WORDS, MAX_WORDS

        prompt = json.loads(CONFIG.read_text(encoding="utf-8"))[
            "prompts"
        ]["multi_news_synthesizer"]["system_prompt"]

        assert f"{MIN_WORDS}-{MAX_WORDS} words" in prompt, (
            f"prompt does not state the enforced band {MIN_WORDS}-{MAX_WORDS}"
        )
        for field, (lo, hi) in BEAT_WORD_RANGES.items():
            assert f"{lo}-{hi} words" in prompt, (
                f"prompt does not state the {field} budget {lo}-{hi}"
            )
