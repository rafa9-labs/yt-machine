# Changelog

## 2026-04-04 — LLM Verification & Correction System Analysis

### What was done
- Conducted a comprehensive analysis of the entire LLM-based script verification and correction pipeline
- Documented the multi-stage verification architecture across all relevant files

### Architecture Analysis Summary

The system uses a **multi-stage pipeline** with **two layers of verification**:

#### Layer 1: Programmatic Validation (Rule-Based)
- **`redfish/prompt_validator.py`** — Validates image prompts for quality (specificity, equipment nomenclature, real locations, action verbs, pixel art style, token weighting, visual grounding)
- **`redfish/geopolitical_validator.py`** — Validates geopolitical accuracy (country representation, equipment-country combinations, hallucination prevention, required visual elements)
- **`redfish/geopolitical_accuracy.py`** — Core database of country visual specs, equipment mappings, hallucination prevention rules, and accuracy scoring
- **`redfish/script_parser.py`** — Extracts visual concepts from script segments (actions, subjects, settings, moods, eras) for image prompt generation

#### Layer 2: LLM-Based Verification & Correction
- **`brain/llm_interface.py`** — Core LLM wrapper using Ollama (localhost:11434):
  - `process_news()` — Analyzes news articles for viral potential
  - `debate_skeptic()` — Skeptic agent challenges the narrative
  - `debate_explainer()` — Explainer agent responds to critique
  - `synthesize_script()` — Generates 6-segment script with **missing-segment recovery** (re-queries LLM for truncated segments)
  - `synthesize_multi_news_script()` — Generates 3-news Masker personality script with **closing validation** (`_validate_closing`)
  - `curate_script()` — Second-pass LLM speech coach that transforms written text to natural spoken language
  - `_extract_json()` — Robust JSON extraction with trailing comma removal, brace counting, and incomplete JSON repair

#### Key Verification Mechanisms:
1. **Missing-segment recovery** — If LLM truncates output (missing segments), a recovery prompt re-queries for just the missing parts
2. **Closing validation** — Ensures every script ends with subscribe/like CTA + "I'm Masker" + "see you tomorrow"
3. **Script curation sanity check** — If curated script is <50% of original length, original is used instead
4. **Geopolitical accuracy scoring** — 0-100 score with penalties for invalid country-equipment combos, hallucination risks, missing required elements
5. **Debate engine** — Multi-agent Skeptic vs Explainer debate to stress-test narratives before synthesis

### Files Analyzed
- `AGENT_CONTEXT.md` — Project single source of truth
- `brain/llm_interface.py` — LLM interface with all verification logic
- `config/system_prompts.json` — LLM personas and configuration
- `redfish/prompt_generator.py` — Script-to-image prompt construction
- `redfish/script_parser.py` — Visual concept extraction from scripts
- `redfish/prompt_validator.py` — Prompt quality and relevance validation
- `redfish/geopolitical_validator.py` — Geopolitical accuracy validator class
- `redfish/geopolitical_accuracy.py` — Country/equipment databases and rules
- `redfish/debate_engine.py` — Multi-agent debate pipeline
- `redfish/visual_extractor.py` — Dual-layer (LLM + spaCy) entity extraction
- `generate_complete_video.py` — Main pipeline entry point
- `_update_prompts.py` — Prompt update helper script