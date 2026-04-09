# Changelog

## 2026-04-08 — Prosody & Punchline Delivery Fix

### Problem
TTS narrator wasn't pausing before punchlines — dramatic reveals, contrasts, and key lines ran together without the natural beat that gives them impact. Root cause: 3-dot pauses (`...`) were invisible to both Kokoro and Edge TTS engines.

### What was done
- **Fixed `_add_natural_pacing()` in `tts_tool.py`**: `...` (3 dots) now converts to `. , ` (period + comma = ~500ms TTS pause). Previously only `....` (4+ dots) was converted — 3-dot dramatic pauses passed through unconverted and were ignored by TTS.
- **Updated curator system prompt** (`config/system_prompts.json`): Instructed curator to use **periods** before punchlines instead of `...`. Example: `"Classic leverage play. Disguised as safety."` — TTS respects period sentence boundaries as real pauses.
- **Updated `curate_script()` inline prompt** (`brain/llm_interface.py`): Same period-based punchline rules applied to the user-level curation prompt.
- **Added "PUNCHLINE DELIVERY" section** to curator prompt: Explicit rules for ending setup sentences with periods, starting punchlines fresh, and adding pauses after rhetorical questions.

### Technical Details
| Pause Marker | Before | After (TTS hears) | Effect |
|---|---|---|---|
| `....` | `. , , ` | `. , , ` (unchanged) | ~800ms story separator |
| `...` | *(ignored)* | `. , ` | ~500ms dramatic pause |
| Period `.` | Natural pause | Natural pause | ~300-400ms sentence break |
| `—` | `, ` | `, ` (unchanged) | ~150ms beat |

### Black Frame Fix
- **Added solid background layer** in `split_video_assembler.py` synced mode: `CompositeVideoClip` now has a dark navy (10,5,25) solid color image underneath all scene clips. This prevents black frames from appearing when scene clips don't perfectly cover the full video duration (gaps between clips, before first clip, or after last clip).

### Files Changed
- `video_server/tts_tool.py` — Added `...` → `. , ` conversion in `_add_natural_pacing()`
- `config/system_prompts.json` — Curator prompt: period-based pauses, punchline delivery rules
- `brain/llm_interface.py` — `curate_script()` inline prompt updated to match
- `video_server/split_video_assembler.py` — Solid bg layer + `_create_solid_color_image()` helper to eliminate black frames

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

## 2026-04-08 — Pipeline Reliability & Test Infrastructure

### What was done
- **Fixed black frame gaps**: Timestamp bridging logic ensures no gaps between scene images
- **Image generation retry**: Failed images retry with fallback prompt; if still fails, creates placeholder
- **Corrupt image detection**: Assembler validates all images before processing
- **Extracted `bridge_timestamp_gaps()`**: Testable pure function for timestamp gap handling
- **Moved test files to `tests/`**: Cleaned root directory (was scattered with 6 test scripts)
- **Added unit test suite** (`tests/test_unit.py`): 25 tests covering timestamp logic, subtitles, scene durations, fallback prompts
- **Added post-export QA validator** (`tests/validate_project.py`): 8 checks on generated projects
- **Updated CHANGELOG.md**: This entry

### Test Structure
- `tests/test_unit.py` — Unit tests (no API calls, fast)
- `tests/validate_project.py` — Post-export project validator
- `tests/test_video_rebuild.py` — Integration: rebuild video from existing assets
- `tests/test_improvements.py` — Integration: TTS + assembler improvements
- `tests/test_timeline.py` — Integration: timeline sync testing
- `tests/test_option_a_layout.py` — Integration: 60/40 layout testing
