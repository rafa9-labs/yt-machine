# Architecture Decisions

> **Companion document to [`PIPELINE.md`](./PIPELINE.md).**
> `PIPELINE.md` describes *what the pipeline is*. This document records *why it
> became that* — the decisions, the alternatives that were rejected, and the
> failures that forced each change.
>
> **Coverage:** commit `54a25d2` (2026-03-20) → working tree at `9f13110` +
> uncommitted work (2026-09-17). 117 commits, 3 major eras.

A note on method: every decision below is reconstructed from commit history,
commit bodies, in-code rationale comments (`WHY`/`RATIONALE` blocks), and
measured artifacts. Where a decision was *not* documented at the time, the
reconstruction is marked **Inferred**. Where a decision was later reversed, the
reversal is documented alongside it — an architecture record that only shows
the winning path is not usable.

---

## Table of contents

1. [The three eras](#1-the-three-eras)
2. [Timeline](#2-timeline)
3. [Foundational decisions (Era 1)](#3-foundational-decisions-era-1)
4. [Productionization decisions (Era 2)](#4-productionization-decisions-era-2)
5. [Local-first decisions (Era 3)](#5-local-first-decisions-era-3)
6. [The reliability campaign (2026-05)](#6-the-reliability-campaign-2026-05)
7. [Platform migration: Windows to macOS](#7-platform-migration-windows-to-macos)
8. [Model provider abstraction (uncommitted)](#8-model-provider-abstraction-uncommitted)
9. [The image model migration in flight](#9-the-image-model-migration-in-flight)
10. [Why the pipeline is composed this way](#10-why-the-pipeline-is-composed-this-way)
11. [Decisions that were reversed](#11-decisions-that-were-reversed)
12. [Anti-patterns this codebase learned to avoid](#12-anti-patterns-this-codebase-learned-to-avoid)
13. [Open decisions](#13-open-decisions)

---

## 1. The three eras

| Era | Window | Commits | Defining question | Answer |
|---|---|---|---|---|
| **1 — Prototype** | 2026-03-20 → 04-09 | ~30 | *Can an agentic pipeline make a watchable video at all?* | Yes, with a 6-act SDXL + stock-footage design |
| **2 — Productionization** | 2026-04-14 → 04-30 | ~30 | *Can it be reliable, typed, observable, and deployable?* | Pydantic + Postgres + LangChain + FastAPI + Docker, then local models |
| **3 — Local-first & hardening** | 2026-05-01 → present | ~57 | *Can it run unattended on one 32 GB Mac, every day, without a human?* | Yes — via phase-based memory management, deterministic enforcement, and fail-closed providers |

The eras overlap and the numbering is retrospective, but the shift in
*question* is real. Era 1 optimised for capability. Era 2 optimised for
engineering quality. Era 3 optimised for **unattended reliability under a hard
resource constraint** — which turned out to be a different and much harder
problem than either of the first two.

---

## 2. Timeline

```text
2026-03-20  ██ Genesis: multi-agent design (DeepSeek R1 + Ollama)
            │  open-viking memory, debate engine, Pexels stock footage
            │  v2.2 "Budget King" → SDXL Lightning + PixelArtRedmond LoRA, ~$0.02/video
            │  v2.3 → grounded military pixel art, 60–80 s scripts
            │  v2.4 → military pixel art generation
2026-03-21  │  Category rotation system (14 geopolitical categories)
2026-03-22  │  "Production ready" — retention research, TTS tag bugs, FAL 422
2026-03-23  │  Image-text correlation (Phase 1–3)
            │  Country-specific visual accuracy system
2026-03-24  │  Custom LoRA training pipeline
2026-03-25  │  6 improvements: num_ctx=8192, geo accuracy, phrasal verbs, stemming
2026-03-28  │  Image-to-image reference guidance
2026-03-29  │  Pipeline failure fixes (script extraction, TTS 403)
2026-03-31  │  ▓▓ SPLIT-SCREEN becomes default format
2026-04-01  │  Karaoke subtitles, continuous coverage, 0.3 s pre-roll
2026-04-04  │  ▓▓ OPTION A 60/40 layout + Kokoro + curation
            │  Comedian delivery (intro_hook, punchlines, CTA)
2026-04-08  │  Curation hallucination fixes
2026-04-09  │  Audio mastering + voice directing + story balancing
────────────┼──────────────────────────────────────────────────────────
2026-04-14  │  ▓▓ PHASE 1–5 BUILD: Pydantic+PG, LangChain, Playwright, pgvector
2026-04-15  │  ▓▓ PHASE 6–8: Docker, FastAPI, v2 orchestrator, structlog
2026-04-17  │  Priority-1 bugfixes
2026-04-20  │  flux/dev + personality revert + Qwen3 architecture notes
2026-04-24  │  ▓▓ STRUCTURAL REFACTOR: everything into src/, secrets untracked
2026-04-26  │  Local Ollama vision curation; soundfile MP3 workaround
2026-04-28  │  Script evaluator + visual QA modules
2026-04-29  │  ▓▓ SPRINTS 5–12: 2-story format, 3 img/story, local FLUX+Kokoro
2026-04-30  │  ▓▓ SINGLE-MODEL Gemma 4 26B migration + VRAM management
────────────┼──────────────────────────────────────────────────────────
2026-05-03  │  ▓▓ VRAM orchestrator finalised; 3 img/story → 4
2026-05-05  │  ▓▓ GGUF quantisation + VRAM orchestrator
            │  ffmpeg DSP mastering replaces naive limiter; last-5s outro reverb
2026-05-06  │  CLIP 77-token fix; geographic anchor rewrite; automation script
2026-05-07  │  Visual prompt hallucination prevention
2026-05-08  │  ▓▓ Script pipeline overhaul: script fixer, formatting-only curation
            │  Trademark greeting enforced
2026-05-09  │  ▓▓ SINGLE TIMELINE SOURCE — dedup + subtitle desync fix
            │  Marker gate replaces quarter-split fallback
2026-05-11  │  Title persistence fix; algosafe content overhaul
            │  intro_hook REMOVED entirely (added 04-04, killed 05-11)
2026-05-12  │  Music to −22 dB
2026-08-30  │  README simplification
────────────┼──────────────────────────────────────────────────────────
2026-09-17  │  ▓▓ WORKING TREE (uncommitted):
            │  • Model provider abstraction (registry/profile/runtime/providers)
            │  • MLX-Gen image provider for Apple Silicon
            │  • macOS launchd + pmset automation
            │  • YouTube/TikTok publishing + token refresh
            │  • Qwen-Image 2512 benchmark (Redmond + Prithiv LoRAs)
```

▓ = structural pivot

---

## 3. Foundational decisions (Era 1)

### ADR-001 — Local LLM, not a hosted API

**Decision:** Run the reasoning model locally on the operator's own machine.

**Alternatives rejected:**
- OpenAI/Anthropic for all steps
- Self-hosted cloud GPU (RunPod, Vast.ai)

**Rationale:** The genesis README specifies "local LLMs" as a core premise, with
`ollama pull deepseek-r1:latest` in the install instructions. The content is
geopolitical and adversarial; the volume is high (daily runs); and the original
author was targeting a machine that already existed. Cost per run approaches
zero and no request leaves the machine.

**Consequences that only became visible later:**
1. Content-filter avoidance — cloud models refuse or degrade on conflict
   imagery and casualty reporting. This becomes an explicit rule in Era 3:
   `llm_interface.py:431` states news analysis has **no cloud fallback** because
   "news analysis often contains sensitive geopolitical content".
2. It makes **model capability a hard scheduling constraint** — see ADR-014.
3. It makes model *serving* a problem the codebase must own — see §8.

**Status:** Held, and hardened. The one exception is script synthesis
(ADR-016).

---

### ADR-002 — Multi-agent debate → single-pass synthesis

**Decision:** Abandon the Skeptic-vs-Explainer debate architecture in favour of
one model producing a complete structured script.

**Alternatives rejected:** Keeping the two-agent debate as a quality mechanism.

**Rationale (from the former `src/brain/chains/debate.py` header):** The debate
engine required two sequential LLM round-trips per story, each returning JSON
that had to be parsed and merged. On a local model, this doubled latency and
introduced two more JSON-parse failure points for a quality gain that was never
measured.

**Evidence of the decision:** the `DebateChain` module, the
`src/collector/debate_engine.py` driver, and `tests/test_langchain_chains.py`
were **removed** once nothing in the pipeline imported them. The
`LLMInterface.debate_skeptic` / `debate_explainer` methods and their prompt
entries remain, exercised only by tests — they are not on any pipeline path
(see ADR-037).

**Status:** Held. The removal was carried out rather than left as a standing
candidate.

---

### ADR-003 — Cheapest viable image generation (v2.2 "Budget King")

**Decision:** Use `fal-ai/stable-diffusion-xl-lightning` with a pixel-art LoRA
at ~$0.003/image.

**Alternatives rejected:** FLUX-class models (more expensive per image at the
time); no image generation (stock footage only).

**Rationale (from commit `cbe6447`):** Explicitly a cost decision — the commit
message states "Ultra-cheap SDXL workflow" and "Cost: ~$0.02 per complete
video". At this stage the goal was proving the pipeline could produce a video,
not maximum visual quality.

**Status:** **Superseded.** Replaced by FLUX.1-dev (ADR-008), then local
generation (ADR-009), then GGUF (ADR-010), then MLX-Gen (ADR-021). The cost
optimisation was correct for its era and wrong for the next one.

---

### ADR-004 — Six-act storytelling → two stories × four beats

**Decision (initial):** Structure each video as a 6-act arc: HOOK / SCALE /
TENSION / PIVOT / ESCALATION / RESOLVE.

**Decision (revised, 2026-04-29):** Restructure to **2 stories × 4 beats**:
`part1` (hook) / `part2` (mechanism) / `real_talk` (truth) / `fallout`.

**Alternatives rejected:** 1 long story; 3 stories; keeping 6 acts.

**Rationale for the revision (Sprints 7–10, `f3f7964` → `564b6cf`):**
- Six acts forced a single narrative to fill ~90 seconds. In practice the
  model padded middle acts, and retention research (commit `1612bc8`,
  "Enhance script synthesis with retention best practices") concluded that two
  distinct stories sustain attention better than one stretched arc.
- Four beats per story map cleanly onto four visual scenes
  (`IMAGES_PER_STORY`), giving one image per narrative beat.
- `NUM_STORIES` was made a constant rather than a hardcoded value, enabling the
  current `NUM_STORIES=2, IMAGES_PER_STORY=4` configuration (8 images).

**Status:** Held. The 2×4 structure is the current architecture and the visual
prompt system depends on its exact naming (`story_N_{part1,part2,real_talk,fallout}`).

---

### ADR-005 — Option A layout: 60/40 split screen

**Decision:** 1080×1920 vertical, with generated scenes on top (1152 px, 60 %)
and a looping avatar on the bottom (768 px, 40 %).

**Alternatives rejected:**
- Full-screen scenes with overlaid subtitles (as used in the earlier
  `video_server/assembler_tool.py`)
- Full-screen avatar (talking-head format)
- 50/50 split

**Rationale (commit `519f6c1`, `57e0e98`):**
- **Vertical 9:16 is non-negotiable** for Shorts/TikTok.
- **Persistent visual anchor:** the avatar loop gives the frame a stable
  human element while the scene changes, which reads as "a broadcast" rather
  than "a slideshow".
- **Subtitle real estate:** the split line at y=1152 creates a natural band for
  captions without covering the scene or the avatar. Subtitles sit at
  `TOP_H − 140 = 1012`.
- **60/40 specifically:** the scene needs enough height to compose an isometric
  scene with foreground/background separation; the avatar needs enough to read
  as a figure. 50/50 gave the scene too little vertical room, and full-screen
  scenes removed the human anchor.

**Status:** Held. `assets/avatar/avatar_loop.mp4` is a pre-rendered asset and
the layout constants live in `config/image_style.json` → `split_layout`, so the
geometry is configuration rather than buried code.

---

### ADR-006 — Karaoke subtitles as a retention mechanism

**Decision:** Word-synchronised subtitles with per-word colour animation
(grey → gold → white), rendered via ASS format, 5-word phrases.

**Alternatives rejected:**
- Sentence-by-sentence captions (used in v2.3, "7-second segment captions")
- Static subtitles
- Word-by-word with no phrase grouping

**Rationale (commits `076b0e9`, `eb45fe9`, `6170124`, `70d395c`):**
Each commit body records a specific defect in the previous iteration:
- `6170124`: "subtitles appear 0.3s BEFORE first word + bigger bolder font" —
  subtitles lagging the audio breaks the sync illusion
- `eb45fe9`: "continuous subtitle coverage no gaps between phrases" — gaps
  between phrases read as missing captions
- `70d395c`: "min 4 words per subtitle phrase" — shorter phrases flashed too
  briefly to read
- `85709b1`: "min 3 words per subtitle" — relaxed after phrases got too long

**Why ASS specifically:** the format supports `{\t()}` colour interpolation
across a single dialogue event. Emitting one event per word caused visible
flicker; one event per phrase with internal animation does not. This is a
technical constraint, not a style preference.

**Status:** Held. The 5-word grouping and 0.04 s delay are current constants.

---

## 4. Productionization decisions (Era 2)

Era 2 is the "Phase 1–8" build: eight sequenced commits over 2026-04-14 →
04-15. The `requirements.txt` file preserves the rationale for each phase in
`WHY:` comments — an unusually good practice that made this document possible.

### ADR-007 — Typed contracts at every boundary (Phase 1)

**Decision:** Pydantic v2 models for all data crossing a module boundary; a
PostgreSQL + pgvector layer as the relational store.

**Alternatives rejected:** Continuing with raw dicts and JSON files.

**Rationale (commit `bdd9c4d` + `requirements.txt`):** "replaces raw dicts with
type-enforced models". Eleven models were introduced covering the full pipeline
(`RSSArticle`, `NewsAnalysis`, `VideoScript`, `VideoProject`, `GenerateRequest`,
…). Articles failing validation are skipped rather than crashing the run.

**Consequence — the JSON/PG dual-write pattern.** From the start, PostgreSQL
was additive: JSON files remained authoritative, and every Postgres write is
wrapped so a failure only warns. This is why the pipeline still completes
successfully today with Postgres not running. The decision proved correct
under exactly the scenario it was designed for.

**Status:** Held. The Pydantic layer is load-bearing; Postgres has drifted into
being optional (see ADR-030).

---

### ADR-008 — FLUX.1-dev over SDXL

**Decision:** Move image generation to `fal-ai/flux/dev`.

**Rationale (commit `71c9d76`):** SDXL-Lightning at 512×512 could not compose
multi-object isometric scenes. The pixel-art aesthetic needs foreground/background
separation, multiple structures, and coherent lighting across the frame — all
of which SDXL handles poorly and FLUX-class models handle well. The per-image
cost increase was accepted.

**Status:** Superseded by local generation, but the *aesthetic standard* set
here (compositional coherence over cost) persists and is why the current
pipeline uses a 9B FLUX.2-class model rather than a smaller one.

---

### ADR-009 — Local FLUX + Kokoro as primary engines

**Decision:** Try local GPU generation first; fall back to fal.ai only if the
local path is unavailable. Same pattern for TTS: Kokoro first, cloud second.

**Alternatives rejected:** Cloud-only; local-only.

**Rationale (commit `1127424`):**
- `USE_LOCAL_FLUX` with default `auto` — "try if CUDA available"
- `USE_KOKORO` with default `auto`
- TTS chain became: Kokoro → ElevenLabs → Edge TTS → silent
- Detection and fallback rather than a hard configuration switch, because the
  same codebase had to run on machines with and without a capable GPU

**Consequence — this is where the "degrade, don't fail" philosophy starts.**
The four-engine TTS chain and the multi-path assembly chain both trace to this
decision.

**Status:** Held, and extended. On the current Apple Silicon machine, MLX-Gen
replaces the CUDA path (ADR-021) and Kokoro remains primary.

---

### ADR-010 — GGUF quantisation for local FLUX

**Decision:** Support GGUF-quantised FLUX checkpoints (Q2_K through F16) with
CPU offload, loaded via `FluxTransformer2DModel.from_single_file()`.

**Rationale (commit `a607634`):**
- Bitsandbytes quantisation required cuDNN, which crashed in this environment
  (commit `c9dcd79`: "cuDNN crash guard")
- GGUF avoids the bitsandbytes dependency entirely
- On a 24 GB GPU, an F16 FLUX.1-dev does not fit; a Q4/Q8 GGUF does

**Notable implementation detail:** with GGUF weights, LoRA cannot be fused
(weights are read-only), so the scale is applied at inference via
`joint_attention_kwargs`. This distinction is preserved in the code.

**Status:** Retained for the CUDA path. Superseded on Apple Silicon by MLX-Gen
(ADR-021), where 8-bit MLX quantisation is native.

---

### ADR-011 — LangChain for structured output (Phase 2)

**Decision:** Route LLM calls through LangChain chains with
`PydanticOutputParser`.

**Alternatives rejected:** Continuing with raw `requests.post` + a custom JSON
extractor.

**Rationale (`requirements.txt`, Phase 2 comment):** The original code had a
"`_extract_json()` hack (60 lines of brace-counting)". LangChain's parser
"auto-retries on parse failure" and integrates with Pydantic.

**Consequence — the fallback still exists, and that is deliberate.** The
pipeline imports LangChain first and falls back to `llm_interface.py` raw calls:

```python
try:
    from src.brain.langchain_interface import LangChainInterface
except Exception:
    _USE_LANGCHAIN = False
```

Every LangChain call site in the pipeline has a raw-path twin. This looked like
redundancy when written; it became essential when the model changed from a 4B
model to a 27B one, where the parsing characteristics differ.

**Status:** Held. Notably, only the *curation* chain is actually used by the
pipeline today (ADR-024). The `debate.py` and `news_analysis.py` chains were
unused (ADR-002) and have been removed.

---

### ADR-012 — Async scraping with a two-tier extraction strategy (Phase 3)

**Decision:** `aiohttp` for parallel RSS fetching; `trafilatura` for static
article text; Playwright Chromium only for JavaScript-rendered sites.

**Alternatives rejected:**
- Sequential `requests` (the original)
- Playwright for everything
- trafilatura for everything

**Rationale (commit `aeb656f`, `async_scraper.py` header — verbatim reasoning):**

> PROBLEM 1: SYNCHRONOUS FETCHING
> Old code: `for feed in feeds: fetch(feed)` → 8 feeds × ~2s = 16s total
> New code: `await gather(*[fetch(f) for f in feeds])` → 8 feeds in parallel = ~2s total
>
> PROBLEM 2: JAVASCRIPT-RENDERED SITES
> `trafilatura` sends a raw HTTP GET and sees the empty template. Playwright
> launches a real browser, waits for `networkidle` (all network requests finish),
> then extracts the text.

And the explicit anti-over-engineering note:

> We do NOT use Playwright for RSS — that would be launching a browser to parse
> XML, which is massive overkill. Playwright is ONLY for full-article extraction.

**The two-tier design is the important part:** trafilatura handles the static
majority instantly; Playwright is paid for only when trafilatura returns an
empty shell. This keeps the common case at ~2 s rather than paying a ~500 ms
browser launch per article.

**Status:** Held. `RSScraper.get_full_article_text` (the sync path used inside
the pipeline's analysis loop) uses trafilatura only — a small inconsistency
with the async scraper's two-tier approach.

---

### ADR-013 — FastAPI replacing Flask (Phase 4)

**Decision:** Rewrite the HTTP API surface on FastAPI + uvicorn.

**Alternatives rejected:** Keeping Flask.

**Rationale (commit `f96ec02`, plus an extensive educational block at the top
of `src/server.py`):**
- Pydantic request validation reusing the Phase 1 models (a malformed request
  returns 422 instead of being silently coerced)
- Auto-generated Swagger at `/docs` — this is how n8n discovers the endpoints
- `BackgroundTasks` replaces manual `threading.Thread`
- Uvicorn is a production ASGI server; Werkzeug is a dev server

**Consequence:** The API is a *secondary* interface. The primary interface is
the CLI. The API exists so n8n and the Docker deployment can drive the same
pipeline — but the launchd daily job never touches it. This is why the pipeline
works with the server not running.

**Status:** Held.

---

### ADR-014 — Single large model instead of two small models

**Decision:** Replace the Qwen3 4B (fast tasks) + Gemma 4 (creative tasks)
dual-model routing with a **single Gemma 4 26B A4B** model for all tasks.

**Alternatives rejected:** Keeping task-specific model routing.

**Rationale (commit `87f27bb`):** Task routing meant two resident models or a
model swap per task. Both are expensive: two models consume memory
simultaneously, and swapping costs a full load cycle per transition. One
capable model handling everything removes an entire class of scheduling and
memory problems.

**The trade-off accepted:** a 26B model is slower per call than a 4B model, so
per-call timeouts had to be introduced in the same commit ("hard/idle per-call
timeouts to prevent Ollama context bloat"). The timeout configuration in
`config/system_prompts.json` still records this lineage — `script_synthesizer.hard`
was later raised from 300 s to 1500 s when the model moved to 27B on Apple
Silicon.

**Status:** Held and extended. The current text model is
Qwen3.8-27B, and the same single-model premise applies. ADR-015 formalises the
general principle.

---

## 5. Local-first decisions (Era 3)

### ADR-015 — Sequential phase model lifecycle (the central constraint)

**Decision:** Enforce strict phase ordering — `idle → text → image → post →
idle` — with exactly one heavy model resident at any time, and verify memory
reclamation at every transition.

**Alternatives rejected:**
- **Both models resident** — impossible: 17 GB + 17 GB on a 32 GB machine
- **Load/unload per call** — a 20 GB model load takes 1–2 min; doing this per
  image would add ~10 min per run for no benefit
- **Reduce model sizes** — a smaller text model reintroduced the JSON
  truncation failures (ADR-014); a smaller image model degraded composition
  (ADR-008)

**Rationale — the constraint that forces the whole design:**

On Apple Silicon there is no VRAM/RAM split. `src/models/memory.py:6` states it
directly:

> Loading Qwen (~20 GiB) and Flux (~10+ GiB) would exceed a 32 GiB machine.

And `tools/generate_complete_video.py:1340` marks the transition:

> This is the critical memory barrier: Qwen must be fully stopped and its
> memory returned before the image model is allowed to load.

**Three mechanisms make it work:**

1. **Phase gating** — `ModelRuntime` owns the transitions; entering a phase
   guarantees the other heavy model is stopped.
2. **Verified reclamation** — `stop_text_model()` samples anonymous memory
   before the kill and requires ≥80 % of the model's estimated footprint
   returned within 90 s. Terminating a process is not the same as reclaiming
   its memory, especially on unified memory.
3. **Anonymous-memory accounting** — the guard measures `Anonymous pages +
   wired`, not "free" memory. macOS uses spare RAM as file cache, so `free` is
   near-zero on a healthy machine and would make every load appear unsafe.

**Why subprocess-per-image for the image model:** a child process exits and its
memory returns to the OS deterministically. A resident image model would have
to be evicted by force and its reclamation verified — exactly the fragile step
the runtime already has to do for the text model. Exiting is simpler and
strictly safer: a hung generation cannot poison the pipeline.

**Status:** Held. This is the defining architectural decision of Era 3 and the
reason the codebase has a `ModelRuntime` at all.

---

### ADR-016 — Script synthesis is the one cloud-permitted step

**Decision:** `multi_news_synthesizer` tries OpenAI `gpt-5-mini` first, then
falls back to the local model. Every other LLM step is local-only.

**Alternatives rejected:** Fully local; fully cloud.

**Rationale:**
- Script synthesis is the **most failure-prone call in the pipeline**: longest
  output (a complete 2-story structured script), strictest structure (nested
  story objects with four labelled beats each), and the one whose failure kills
  the run outright (`exit(1)` after 3 attempts).
- It is also the **least sensitive**: the input is already-summarised analysis
  of public reporting, not source material or adversarial content.
- News analysis, by contrast, is explicitly local-only
  (`llm_interface.py:431`) because raw geopolitical reporting may trigger cloud
  content filters.

**Status:** Held. The key is set in `.env`; when absent the local path runs.

---

### ADR-017 — Vector dedup disabled due to VRAM contention

**Decision:** Disable the pgvector embedding-based deduplication step in the
main pipeline.

**Alternatives rejected:**
- Keeping it enabled
- Running embeddings on GPU
- Running a second embedding model alongside the text model

**Rationale (`generate_complete_video.py:784`):**

```python
log.info("dedup.skipped", reason="vector_dedup_disabled_vram_contention")
```

Embedding every topic required calling a *second* Ollama model
(`nomic-embed-text`) while the 18 GB text model was resident. On 32 GB this
pushed into swap. ADR-015 forbids two heavy models at once — and while
`nomic-embed-text` is small, its KV cache and allocator overhead during the
text phase was enough to destabilise the run.

**The important part — the capability was not lost, it was relocated.** Semantic
dedup now happens later, inside `script_evaluator.py`, using a **CPU-only**
`all-MiniLM-L6-v2` SentenceTransformer. Same benefit, no GPU contention. This is
a recurring pattern in this codebase: when a step conflicts with the memory
constraint, move it off the GPU rather than removing it.

**Status:** Held. `roles.embedding` is `null` and `nomic-embed-text` is not
pulled, so the vector-memory path is inert.

---

### ADR-018 — Deterministic enforcement alongside LLM generation

**Decision:** Insert a purely deterministic Python step
(`script_enforcement`) between the LLM fixer and the LLM curation steps. It
enforces structure with no model involvement.

**Alternatives rejected:** Trusting the LLM to produce correct structure after
the fixer pass.

**Rationale:** The pipeline's structure requirements are mechanical:
≥2 stories, empty greeting, segue only on story 1, no 4+ word prefix
duplication, no fallout echo in the closing, the trademark closing present.
A model can get each of these right *most* of the time; a Python function gets
them right *always*. The enforcement chain runs in fixed order:

```
_enforce_greeting → _enforce_segues → _dedup_segue_overlap
→ _dedup_inter_story_phrases → _enforce_fallout → _ensure_greeting_in_fulltext
```

**Why this matters architecturally:** it means the pipeline has a **guaranteed
structural floor**. No matter what the model produces, the run either yields a
valid script or fails loudly — it never yields a structurally broken script
that then produces a broken video. For an unattended 06:00 job, that guarantee
is worth more than any single step's output quality.

**Status:** Held. This pattern was later extended to the publisher
(ADR-028) and the timeline (ADR-022).

---

## 6. The reliability campaign (2026-05)

Between 2026-05-05 and 2026-05-12 the commit subjects are almost entirely
`fix:` — a concentrated period of hardening an already-feature-complete
pipeline. Reading the commit bodies in sequence reveals a single systemic
problem and its resolution.

### ADR-019 — Kill the intro_hook

**Decision (2026-04-04):** Introduce an `intro_hook` — a comedian-style opening
line before the first story, plus per-story punchlines and a CTA closing.

**Decision reversed (2026-05-11):** Remove `intro_hook` entirely "across
codebase".

**Timeline of the reversal:**
- `02d25bc` (05-06): "shorten intro to one phrase ... remove intro_hook" (first
  attempt, incomplete)
- `f9dd595` (05-11): "remove intro_hook"
- `e329827` (05-11): "remove greeting and intro_hook completely across codebase"

**Rationale for the reversal (inferred, corroborated by three commits):** The
hook created more problems than it solved — it delayed the first story, it was
redundant with the persistent title overlay, and each attempted fix required
corresponding changes in the synthesis prompt, the fixer, the enforcement chain,
the curation prompt, and the timeline builder. A recurring theme: **every
optional script component multiplies across every downstream step.**

**Current state:** `greeting` is forced to `''` in synthesis
(`generate_complete_video.py:966`), the video opens directly into story 1, and
the `hook_card` in assembly is explicitly disabled
(`hook_card_text = None`). The title overlay is the only opening element.

**This is the most instructive reversal in the history.** It shows the real cost
of a feature in a pipeline where each script field is consumed by six
downstream stages.

---

### ADR-020 — Single timeline source

**Decision:** Build the segment timeline **once**, after script evaluation. No
earlier step may build or mutate it.

**The failure it fixed (commit `525b4b4`, verbatim):**

> Root cause: `segment_timeline` and `full_text` were being built 3 times
> (synthesis enforcement, curation `_reassemble_script`, evaluation
> `_rebuild_timeline`), with the evaluation step silently overwriting the
> curated `full_text`. This caused duplicated segways in the final audio and
> desynchronized subtitle timestamps.

**Alternatives rejected:** Keeping per-step timeline construction with dedup
guards at each site.

**Rationale:** Three construction sites meant three chances to differ, and the
last writer won silently. The fix was architectural rather than defensive:
remove premature timeline builds, remove the evaluation step's `_rebuild_timeline`
call, and make one authoritative build after all script mutation has stopped.

**Subsequent hardening in the same area:**
- `cd6604c`: prefix-overlap dedup in `_reassemble_script` + a closing echo
  whitelist + a curation fidelity validator
- `d2a484c`: prefix-overlap dedup added to the timeline rebuild, with a test
- `525b4b4`: dedup threshold lowered to 3 words when both strings start with the
  same bridging conjunction (And/But/While/Meanwhile/Now/Speaking)

**Corroborating detail:** The same commit also fixed an in-place mutation of
`word_timestamps` in `subtitle_renderer.py` (creates a copy before applying the
delay), increased the minimum ASS event duration from 0.01 s to 0.04 s (one
frame at 30 fps), and added an explicit warning when ASS generation returns
empty instead of silently dropping subtitles.

**Status:** Held. `build_timeline` runs after evaluation and is the single
source of truth. The `PipelinePhase` enum exists specifically to make
regressions detectable — `_advance_phase()` raises if anything tries to go
backwards.

**This is the single most important correctness decision in the codebase.**

---

### ADR-021 — Reject unmarked curator output rather than guess

**Decision:** If the curation LLM ignores the `[HOOK]/[MECHANISM]/[REAL_TALK]/[FALLOUT]`
markers, reject its output entirely and fall back to the original narration.
Remove the previous quarter-split fallback.

**Rationale (commit `276c0c6`, verbatim):**

> The quarter-split fallback silently assigns text to wrong fields, causing
> fallout sections to drop and text to be misaligned.

**The mechanism:** `_parse_curated_structures` now requires at least
`expected_count × 2` markers before it will parse. Below the gate, it returns
`None`, which forces `_reassemble_script` to keep the original narration.

**Why this is the right call:** The quarter-split produced output that *looked*
successful — a complete script with all fields populated — but with text in the
wrong beats. That is worse than a visible failure, because it silently
degrades the video while the run reports success. The commit removed 10 lines
of fallback logic and added 4 tests asserting the gate.

**Generalised principle:** *A silent wrong answer is worse than a loud failure.*
This appears three times in the codebase (here, ADR-023, ADR-029).

---

### ADR-022 — Prevent visual prompt hallucination with context injection

**Decision:** Inject the article title and analysis topic directly into visual
prompt generation, and add an explicit anti-hallucination rule set.

**Rationale (commit `2113001`):** The visual prompt generator was inventing
geographic locations not present in the story — e.g. a Kenya/Ukraine story
acquiring Tokyo or the Andes. The fix introduced prompt rules 13–14 (verify
every location appears in the narration or STORY CONTEXT) plus a
`_validate_visual_relevance()` fallback.

**Follow-on work in the same area:**
- `cad400f`, `f7fc79a`: geographic anchor simplified to plain country names —
  earlier versions added regional descriptors which the model then embellished
- `4fae978`: CLIP 77-token truncation fix — content was being placed after
  token 77, where CLIP cannot see it, causing the encoder to truncate the
  scene description and the model to invent from the style tail alone

**Status:** Held. The "never invent locations" rules are in the current
`visual_prompt_generator` system prompt.

---

### ADR-023 — Fail closed on local image generation

**Decision:** When a managed local image provider fails, return an explicit
failure. Do **not** silently fall back to fal.ai.

**Alternatives rejected:** Automatic cloud fallback (the previous behaviour).

**Rationale (`pixel_art_tool.py:2204`):**

```python
# Fail closed: report the failure, do not fall back to a cloud model.
return mlx_result
```

And at the pipeline level (`generate_complete_video.py:1360`), an unavailable
MLX-Gen provider exits with code 4 rather than routing elsewhere.

**Why reject a working fallback:** A cloud fallback produces a video that looks
subtly different (different model, different style, different cost) while
reporting success. The operator cannot tell from the output which model
generated it. An explicit failure is diagnosable; a silent model substitution
is not. The same principle governs `TextProvider`, which raises `ProviderError`
rather than switching backends.

**Status:** Held. `providers.py` header states it explicitly: "we never quietly
swap in a different model."

---

### ADR-024 — Curation is formatting-only, never rewriting

**Decision:** The curation step may reformat for pacing. It may not rewrite
content.

**Rationale (commit `308c07e`, "formatting-only curation"):** Curation was
found to hallucinate and to drop sections. Constraining it to formatting
preserves the verified content from the analysis and synthesis steps, where all
the factual grounding lives.

**Consequence — this is why `_validate_curation_fidelity` exists.** Every
curated story body is compared against the original; a story failing fidelity
is reverted. Curation is the only LLM step with a fidelity gate, precisely
because it is the only one trusted to touch already-validated content.

**Status:** Held. Combined with ADR-021's marker gate, curation is now the most
heavily guarded step in the pipeline.

---

### ADR-025 — Audio mastering via ffmpeg DSP chain

**Decision:** Replace the custom numpy-based mastering with an ffmpeg filter
chain:

```
highpass=f=80:t=0.7071
equalizer=f=4000:t=q:w=1.2:g=2
acompressor=threshold=0.125:ratio=2:attack=10:release=100:knee=4:makeup=1
loudnorm=I=-16:LRA=11:TP=-1.5
```

**Alternatives rejected:** The previous numpy implementation; no mastering.

**Rationale (commit `ad66350`, "replace crude mastering with ffmpeg DSP chain"):**
The numpy version had already produced a bug (`f193f1f`: "audio mastering numpy
rfftfreq shape mismatch"), and it lacked the perceptual reasoning that
purpose-built DSP filters encode:
- **highpass at 80 Hz** — removes rumble that wastes headroom on phone speakers
- **EQ at 4 kHz (+2 dB)** — presence/ intelligibility region for speech
- **compressor** — evens out TTS level variation between chunks
- **loudnorm I=−16, TP=−1.5** — broadcast-standard integrated loudness with
  true-peak headroom so platform transcoding does not clip

**Same commit also scoped the outro reverb** to the last 5 seconds
(`aecho=0.8:0.88:60:0.4`). Applying it to the whole track had been an
over-application; melancholy treatment belongs only on the closing.

**Status:** Held, unchanged.

---

### ADR-026 — Music bed at −22 dB

**Decision:** Mix the music bed at `volume=0.08` (−22 dB) with a 10 s tail fade.

**Rationale — a measured decision, not a taste one.** Commit history:
- `f9dd595` (05-11): "raise music volume"
- `02f0eb2` (05-12): "drop music volume to **−22 dB (0.08)** for **standard
  background level**"

The final commit message describes the level as *standard*, i.e. the result of
comparing against reference broadcasts rather than an arbitrary adjustment.
0.08 was chosen because louder settings masked narration intelligibility.

**A related reversal in the same area:** the music *track* was changed to
`news_background_sound.mp3` (`e5da148`) and then reverted to `news-yt.mp3`
(`e329827`) the same day. The original asset remains backed up as
`news-yt-original-backup.mp3`.

**Status:** Held.

---

### ADR-027 — Title persistence

**Decision:** The title overlay persists for the entire video and is always
emitted into the ASS file, even when no word timestamps exist.

**The bug it fixed (commit `e5da148`, verbatim):**

> prevent `_clamp_ass_overlaps` from clamping Title events to subtitle start,
> use `total_dur` instead of hardcoded `9:59:59.99`, always include title in
> ASS output even without `word_timestamps`

**Rationale:** Two independent defects: an overlap-clamping function treating
the title as a subtitle and truncating it, and a hardcoded end time. Both
resulted in the title vanishing partway through — a silent visual regression.

**Status:** Held. `title_end_seconds = total_duration`, emitted at
`\pos(540,20)`.

---

## 7. Platform migration: Windows to macOS

### ADR-028 — launchd + pmset replaces Task Scheduler + WSL

**Decision:** Move daily automation from Windows Task Scheduler driving WSL2 to
macOS launchd driving the local pipeline, with `pmset` providing the hardware
wake.

**Alternatives rejected:**
- **cron** — cannot fire a missed job after sleep, and has no wake primitive
- **Keeping the WSL path** — the machine changed; the work now happens locally
- **A resident daemon** — unnecessary; the job is daily and short-lived

**Rationale:**

| Requirement | Windows | macOS |
|---|---|---|
| Scheduled trigger | `schtasks` | launchd `StartCalendarInterval` |
| Fire missed jobs after sleep | Task Scheduler setting | native to `StartCalendarInterval` |
| Hardware wake | BIOS wake-on-alarm | `pmset repeat wakeorpoweron` |
| Keep awake during run | power plan | `caffeinate -s` |
| Sleep after run | idle timer | `pmset -c sleep 20` |

The critical capability is that **launchd fires a missed job when the machine
wakes**, which cron cannot do and which `StartCalendarInterval` (rather than
`StartInterval`) provides.

**Non-obvious failure modes discovered during implementation:**
1. **launchd's PATH is minimal** — `/usr/bin:/bin:/usr/sbin:/sbin`, with no
   `/opt/homebrew/bin`. Without an explicit export, `ffmpeg`, `llama-server`,
   `mlx_lm.server`, and `ollama` are all "command not found".
2. **A stale-lock test wedges the job forever** — a bare `[ -f "$LOCK" ]`
   check treats a lock from a crashed run as permanent. The guard must delegate
   to `PipelineLock._is_stale()` (checks pid liveness and age).
3. **`sudo pmset` cannot be automated** — power schedules live in the SMC and
   require root. The tool prints the exact command and instructs manual
   execution rather than hanging on a password prompt inside launchd.

**Status:** Implemented and installed; wake/power steps require the operator's
one-time `sudo`.

---

### ADR-029 — Publish idempotency via a ledger

**Decision:** Maintain `output/publish_logs/.published.json` mapping a video
fingerprint (`path|size|mtime`) to per-platform results. Skip platforms that
already succeeded. Record only successes.

**Rationale:** Two independent paths can publish the same video: the scheduled
launchd run and a manual `--publish` invocation. Without a ledger, the second
run double-posts. A publish is externally visible and effectively irreversible,
so this is a case where the cost of a duplicate exceeds the cost of a missed
publish (the missed one retries on the next run).

**Design details that matter:**
- Failures are **not** recorded, so a failed platform retries next run
- `dry_run` never writes to the ledger
- `--force` overrides
- The fingerprint is `path|size|mtime`, not a content hash — hashing a 20 MB
  MP4 on every run is wasted I/O, and any re-encode changes size and mtime anyway

**Status:** Implemented and tested.

---

### ADR-030 — Postgres is optional, JSON is authoritative

**Decision:** Keep PostgreSQL in the architecture but treat every write as
best-effort. The pipeline must complete with Postgres unreachable.

**Rationale:** This was implicit from Phase 1 (JSON always written, Postgres
additive) and became explicit when the current machine stopped running Docker.
The measured behaviour: a full successful run with
`postgres.save_failed` warnings at every step, and a complete
`manifest.json`/`checkpoint.json` on disk.

**Assessment:** The design is correct for the constraint (no external service
may gate a 06:00 unattended run), but it has drifted — the API surface's
`/status` and `/latest` endpoints read in-memory state and JSON files, not the
database, so Postgres currently provides no functioning capability. It is
scaffolding for a future multi-run analytics layer.

**Status:** Accepted drift. Either the database gains a consumer or it should be
removed from the default deployment.

---

## 8. Model provider abstraction (uncommitted)

This is the largest in-flight architectural change. It introduces five new
modules under `src/models/` plus a new image provider.

### ADR-031 — Discovery/selection/serving split

**Decision:** Separate three concerns that were previously entangled:

| Concern | Module | Responsibility |
|---|---|---|
| **Discovery** | `registry.py` | Find every model on the machine; normalise into `ModelSpec` |
| **Selection** | `profile.py` + `model_setup.py` | Persist a role→model mapping |
| **Serving** | `runtime.py` + `providers.py` | Launch/adopt servers, enforce memory, adapt APIs |

**Rationale (`registry.py` header, verbatim):**

> Before: model names were hard-coded in `config/system_prompts.json` and every
> module built its own endpoint. Now discovery happens once, the user selects
> roles in `tools/model_setup.py`, and every provider adapter consumes the same
> `ModelSpec`.

**Why this was necessary:** The pipeline moved from one known model
(Gemma 4 via Ollama at a fixed URL) to a machine-local selection of models
served by three different runtimes (llama.cpp, Ollama, MLX). Hard-coded
endpoints could not express that.

**The provider abstraction (ADR-032) is what makes the swap possible without
touching pipeline code.**

---

### ADR-032 — One interface, multiple adapters, no silent fallback

**Decision:** Every text model is reached through a `TextProvider`; concrete
adapters translate to a specific server API. Providers **raise** on failure
rather than substituting.

**Rationale (`providers.py` header, verbatim):**

> Before, `LLMInterface` hard-coded Ollama's endpoint format. The confirmed Qwen
> GGUF actually runs on llama-server (OpenAI-compatible). Now the active profile
> selects the adapter and the rest of the code is unchanged.

> **NO SILENT FALLBACK:** Providers raise `ProviderError` when the server is
> unreachable or returns nothing usable. Callers decide whether to retry or
> abort — we never quietly swap in a different model.

**Adapters:**

| Adapter | Protocol | Endpoint |
|---|---|---|
| `OllamaTextProvider` | NDJSON streaming | `/api/generate`, `/api/chat` |
| `OpenAITextProvider` | SSE streaming | `/v1/chat/completions` |

Both enforce two wall-clock limits while streaming: `hard_timeout` (total
duration) and `idle_timeout` (max gap between tokens). When exceeded, the HTTP
response is closed immediately. This is the *real* timeout mechanism — the
pipeline's `_run_with_timeout` only abandons a thread, which Python cannot kill.

---

### ADR-033 — Profile file separate from prompt file

**Decision:** Model selection lives in `config/model_profile.json`
(machine-written, gitignored). Prompts live in `config/system_prompts.json`
(hand-edited, tracked).

**Rationale (`profile.py` header, verbatim):**

> `config/system_prompts.json` holds prompts (hand-edited). The profile is
> machine-written by `tools/model_setup.py`. Keeping them separate means
> re-running setup never clobbers prompt engineering.

**Supporting details:**
- Overridable via `YT_MODEL_PROFILE=/path/to/profile.json`
- **No secrets stored** — `metadata` may carry environment-variable *names*,
  never values
- Atomic writes (tmp file + `os.replace`) so an interrupted setup never leaves a
  corrupt profile
- Required roles: `text` and `image`. Optional: `vision`, `embedding`

---

### ADR-034 — Conservative capability probing and memory estimation

**Decision:** Two matching rules that keep the runtime safe on unknown models:

**1. Capability probing before sending flags** (`mlxgen_provider.py`):
`supports_negative_prompt`, `supports_guidance`, and `supports_lora` are probed
live via `<executable> capabilities --model <path>`, and a flag is only sent
when the capability is truthy.

**Rationale:** Sending an unsupported flag makes `mlxgen` exit(2) **before
loading weights** — an instant failure that is easily mistaken for a generation
failure. The concrete trap is `--negative-prompt`, which step-distilled FLUX.2
Klein rejects because it has no CFG branch.

**2. Memory estimates from measured calibration, not theory** (`registry.py`):

| Model class | Rule | Measured basis |
|---|---|---|
| llama.cpp GGUF | `max(disk × 1.0 + 1, 4)` GB | Weights stay resident in Metal; calibrated from an 18.4 GB measured anonymous load |
| MLX-Gen safetensors | `max(disk × 0.65, 6)` GB | Safetensors remain file-backed and evictable |

The reserve is **4.0 GB**, calibrated: an 18.4 GB load left 4.7 GB headroom
with `swap 0.0 GB`; a 6 GB reserve made the same load impossible.

**Discovery never loads a model** — it reads metadata and filesystem state only.
This means `model_setup.py` is safe to run while the pipeline is active.

---

## 9. The image model migration in flight

### ADR-035 — Migrate from FLUX.2 Klein to Qwen-Image 2512

**Status:** In progress as of 2026-09-17. The pipeline still runs FLUX.2 Klein;
a Qwen-Image benchmark is executing on the machine.

**Decision:** Evaluate `AbstractFramework/qwen-image-2512-4bit` plus one of two
pixel-art LoRAs (Redmond, Prithiv) as a replacement for the current FLUX.2 Klein
checkpoint.

**Rationale — the capability gap:**

FLUX.2 Klein is step-distilled: it has **no CFG branch**, therefore
`supports_negative_prompt: false`. Every quality problem that would normally be
suppressed with a negative prompt must be fought in the positive prompt
instead. The pipeline carries three explicit workarounds for this:

| Workaround | Location | Purpose |
|---|---|---|
| `sanitize_text_requests()` | `pixel_art_tool.py` | Rewrites any prompt requesting literal text (the model renders it as scrambled glyphs) |
| `_strip_lora_triggers()` | `pixel_art_tool.py` | Removes fal.ai trigger words that are inert locally and pull toward sprite-sheet imagery |
| `LOCAL_MAX_STYLE_WORDS = 14` | `pixel_art_tool.py` | Caps the style tail — measured: a 153-word triple-suffix prompt produced garbled pseudo-text and disconnected fragments while a short tail produced a clean image |

Qwen-Image supports negative prompts (`supports_negative_prompt: true`),
so `NEGATIVE_PROMPT` from `config/image_style.json` — currently discarded on
the local path — would become effective.

**The benchmark methodology is itself an architectural decision:**

**ADR-035a — Provenance-tracked benchmarking.** `gen.sh` wraps
`mlxgen generate` so every image carries a `.provenance.json` sidecar
containing model, LoRA path, scale, prompt, seed, steps, guidance, wall time,
exit code, byte size, the runtime's LoRA application report, and a post-run
memory snapshot.

> **Rationale (from the script header, verbatim):** The selection decision has
> to be auditable. Reproducing an image later requires more than the PNG — it
> needs the exact LoRA file, scale, seed and step count.

**ADR-035b — LoRA compatibility must be validated manually.**
`phase5-lora-base-validation.json` records structural validation of both
candidates because **mlxgen's `LoRACompatibility` gate skips validation for
absolute local paths**. Validation checked base-model family
(`ss_base_model_version: qwen_image`), block coverage (0..59 = 60 blocks),
tensor count (1680), and rank (32 vs 64). Both were accepted; the common
failure mode being guarded against is grabbing a FLUX.1 adapter by filename.

**ADR-035c — Sampling parameters are deferred, not guessed.**
`sampling-rationale.json` documents the initial choice (guidance 4.0 from the
official model card, steps 20 deliberately below the card's 50) and explicitly
defers step tuning to a later phase.

**The measured problem:** at 20 steps, Qwen-Image takes **572–607 s per image**
on an M1 Pro. Eight scenes would be ~80 minutes of image generation alone,
before any LLM or assembly time. FLUX.2 Klein at 8 steps takes ~5.6 min/image
(339 s, measured from the last real run).

**Benchmark state as of 2026-09-17 18:17 — INCOMPLETE.** `phase7-screen.sh` was
designed as a 12-image comparison (3 conditions × 2 scenes × 2 seeds) and
produced **3 of 12** before halting. The stopping condition was environmental,
not a quality verdict: the run aborted on `Battery below 5% threshold: 3%`.
Only seed 42, scene A completed for all three conditions; scene B and seed 891
were never attempted.

The harness is resumable — `phase7-screen.sh` skips any tag whose PNG already
exists — so the remaining 9 images can be produced by re-running it on AC power.
**No conclusion about relative image quality can be drawn from 3 images of a
12-image screen**, and the Redmond-vs-Prithiv choice (open decision 2) is
blocked on that comparison.

**ADR-035d — Step count is the highest-leverage decision in the migration.**
The model card's 50 steps would make a run exceed three hours. Whether
Qwen-Image is viable at 8–12 steps is the open question the benchmark must
answer.

**Migration cost from the pipeline's perspective:** near zero. The provider
abstraction means the swap is a change to the generation profile
(`config/generation_profiles.json`: `model.match` and `lora.*`), resolved
per-image through the registry. No pipeline code changes. (At the time of
writing this ADR that role was split across `config/model_profile.json`
`roles.image` and a `MLXGEN_LORA_PATH` environment variable; both were later
consolidated into the generation profile.)

---

### ADR-036 — The pixel-art LoRA lineage

**Observation (not a decision, but the most persistent thread in the
repository):** the project has used a named pixel-art LoRA in every
image-generation era, and the *style adapter* has outlived every base model it
was attached to.

**The lineage — with an important correction to the obvious reading:**

| Era | Base model | LoRA | Trigger / scale | Source |
|---|---|---|---|---|
| 2026-03-20 | SDXL-Lightning | `PixelArtRedmond-Lite64` | — / 0.85 | `cbe6447`, `02731a9` |
| 2026-04→05 | FLUX.1-dev / FLUX.2 Klein | `prithivMLmods/Retro-Pixel-Flux-LoRA` | `Retro Pixel` / 0.85 | `config/image_style.json` (current) |
| 2026-09-17 (eval) | Qwen-Image 2512 | **both** Redmond and Prithiv | — / 1.0 | `phase5-lora-base-validation.json` |

**The correction:** it is tempting to describe this as "Redmond since day one,"
and the two Redmond LoRAs (`qwen-redmond/`, and an HF probe for
`PIXELART-REDMOND-FLUXKLEIN9B`) make that reading plausible. **It is not what
happened.** Redmond was used in the SDXL era; the FLUX eras switched to a
different artist (`prithivMLmods`); the current Qwen benchmark is evaluating
both because the Redmond Qwen adapter became available. Two independent
pixel-art authors have now been used, and the Qwen evaluation is choosing
between them on rendered output rather than on lineage.

**Why this is recorded as an ADR:** because the *stability* being demonstrated
is not "one LoRA is best" — it is that **the style is a swappable adapter over
a swappable base model**. That is the premise the provider abstraction (§8) was
built on, and it has now held across four base-model generations.

**A note on the `.redmond_path` benchmark harness vs. the FLUX Klein variant.**
`artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B` has a HuggingFace cache entry on
this machine containing **only `refs/main` — a resolved commit SHA, no
`blobs/`, no `snapshots/`, no `.safetensors` anywhere on disk**. For contrast,
the Qwen-Image checkpoint downloaded in the same session is 16 GB with a full
`blobs/ snapshots/ trees/ refs/` layout.

The distinction matters because it records *intent*, not evaluation. Eight
repos acquired `refs/` entries inside a 90-second window on 2026-09-17
(16:44–16:45) — the Qwen image model, two Qwen LoRAs, and five FLUX Klein
variants. That is candidate enumeration: resolving repo IDs to pin commit SHAs
before deciding what to download. Only the Qwen checkpoint was then fetched.

**Consequence:** a candidate that has never been benchmarked —
`PIXELART-REDMOND-FLUXKLEIN9B`, an adapter explicitly tagged
`base_model:black-forest-labs/FLUX.2-klein-9B` — was enumerated alongside the
Qwen options and then not carried forward. Its compatibility case is strong on
paper: our running model derives from `FLUX.2-klein-9B`, the adapter targets
exactly that base, and the LoRA is ~563 MB against the 16 GB already spent on
Qwen. Measured from the last real run, the current FLUX.2 Klein path costs
**339 s / 5.6 min per image**, roughly half Qwen's 20-step time.

This is recorded as an **unevaluated candidate**, not a rejected one — there is
no evidence a comparison was made. See open decision 1b.

---

### ADR-037 — Verification is layered, and the layers fail differently

**Decision:** Keep validation split across two layers with distinct failure
semantics, rather than consolidating it into one gate.

| Layer | Mechanism | Failure mode | Where |
|---|---|---|---|
| 1 — Deterministic | Code checks on structured data | Rejects, or repairs from source text | `src/collector/prompt_validator.py`, `src/collector/geopolitical_validator.py`, `src/collector/geopolitical_accuracy.py`, `src/brain/script_evaluator.py`, `src/brain/llm_interface.py::_extract_json` |
| 2 — LLM recovery | A second model pass over unusable output | Retries, or substitutes a validated shape | missing-segment recovery, `_validate_closing` + CTA quarantine, `_validate_curation_fidelity` |

**Rationale:** the two layers contain different classes of defect. Layer 1
catches *malformed or ungrounded* output where the correct behaviour is to
reject (ADR-021) or to repair deterministically from the source text. Layer 2
handles output that is well-formed but wrong — a truncated story, a closing
that drifted into a call to action, curation that silently rewrote rather than
reformatted (ADR-024). Collapsing them would force one policy on both, and the
two policies are deliberately opposite: layer 1 must never guess, layer 2
exists because guessing from the *original* text is better than shipping a hole.

**Explicitly not part of this layer:** the Skeptic/Explainer debate. It was
abandoned (ADR-002, §11 row 1) and its modules (`src/collector/debate_engine.py`,
`src/brain/chains/debate.py`, `src/brain/chains/news_analysis.py`) were removed
as unreachable. The remaining `debate_skeptic` / `debate_explainer` prompt
entries and `LLMInterface` methods have no caller on any pipeline path.

**Status:** Held.

---

### ADR-039 — Provenance is a per-image artifact, propagated with its image

**Decision:** Every generated image carries a JSON sidecar named
`<image>.provenance.json`, written next to the image. The pipeline copies it
into the project folder alongside the image it describes, and the manifest
records both.

**Why this is an ADR and not an implementation detail.** The record is what
makes an image reproducible: which checkpoint, which adapter, which seed, and
which post-processing produced it. A record that exists only in the shared
scratch directory (`output/images/`) is not part of the deliverable — the
project folder is what gets archived, inspected, and attached to a bug report.
An artifact meant for audit but stored outside the audited artifact does not
function.

**What went wrong before this decision:** the sidecar was written correctly on
every successful generation, but the pipeline's copy step moved only the PNG.
Six sidecars existed on disk in the scratch directory while every project
folder held zero, so provenance looked unimplemented. It was implemented and
unreachable.

**The guard.** A sidecar is a shareable file, so `src/video/provenance.py`
applies a deny-list on every write: key names matching credential patterns
(`*_key`, `token`, `secret`, `password`, …) and values matching known provider
formats (`ghp_`, `hf_`, `sk-`, `AKIA`, PEM headers, …) are redacted. The check
is value-based as well as key-based, because a credential pasted under an
innocent key name is still a credential. `assert_no_secrets` is the strict
form and is asserted in tests, so a future caller that passes
`os.getenv("FAL_KEY")` fails CI rather than leaking into an artifact.

**Scope of a record.** Per-image sidecars hold the generation facts (model,
provider, profile, prompts, sampling, LoRA, post-processing). The run-level
record in `manifest.json` holds what belongs to the whole run: project id,
status, git commit, LLM identifier, TTS engine/voice, assembly settings, and
the provenance filenames. Partial and failed runs still produce the run-level
record with `status: incomplete`, because knowing what a failed run attempted
is the point of recording it.

**Alternatives rejected:**
- *One run-level record only* — loses per-image seeding, which is the field
  that makes a single bad frame reproducible.
- *Provenance in the database* — the JSON store is authoritative (ADR-030);
  Postgres is optional and may not be running.
- *Refuse to write on any secret rather than redact* — a run that has already
  produced images would lose its record entirely over one bad key name.

**Status:** Held.

---

## 10. Why the pipeline is composed this way

Beyond the individual decisions, the *shape* of the pipeline follows from four
structural principles. These were not written down at the time; they are
inferred from the accumulated patterns and are stated here so future changes can
be evaluated against them.

### Principle 1 — LLM steps alternate with deterministic steps

```
LLM      news_analysis
LLM      script_synthesis
LLM      script_fixer
───      script_enforcement        ← deterministic
LLM      visual_prompts
LLM      script_curation
LLM      script_evaluation
───      build_timeline            ← deterministic
───      MEMORY BARRIER
LLM      pixel_art (via mlxgen)
───      voice_generation (TTS)
───      video_assembly (ffmpeg)
───      platform_metadata         ← deterministic
```

Every LLM step is followed or bracketed by a deterministic one. The LLM
produces content; Python enforces structure. This is what makes an unattended
run safe: the pipeline's *guarantees* come from code, and its *quality* comes
from the model.

**Corollary:** when a step fails, the first question is whether it was an LLM
step (quality/failure mode) or a deterministic step (bug). The two have
completely different remediation paths.

### Principle 2 — Order is constrained by memory, not by logic

The logical order would put image generation adjacent to script synthesis,
since both consume the script. The actual order is forced: **all text-model work
must complete before any image-model work begins** (ADR-015). This is why the
memory barrier sits between `build_timeline` and `pixel_art` — and why the
visual prompts (an LLM step) must be generated *before* the barrier, even
though they are logically part of image generation.

Any future "move step X earlier/later" proposal must check the phase it lands in.

### Principle 3 — Degrade, but never silently

| Situation | Behaviour |
|---|---|
| Postgres unreachable | Warn, continue — JSON files are authoritative |
| Telegram unconfigured | Skip silently — no side effect on output |
| Embedding model absent | Skip dedup, log reason |
| TTS engine fails | Try the next of 4 engines |
| Assembly path fails | Try the next of 3 paths |
| **Image model fails** | **Fail closed** — no cloud substitution |
| **Text model unavailable** | **Fail closed** — `ProviderError` |
| **Curation returns unmarked output** | **Reject, fall back to original** |

The distinction: degradation is acceptable for **enrichment** (dedup, vision QA,
notifications). It is not acceptable for **provenance** — substituting a
different model changes what the artifact is, while failing to notify changes
nothing about it.

### Principle 4 — Every optional component costs six times

A script field like `intro_hook` must be produced by synthesis, preserved by the
fixer, enforced by the enforcement chain, respected by curation, included by
`_reassemble_script`, and mapped in the timeline builder. Abolishing it
(ADR-019) required three commits.

**Practical consequence:** new script fields should be treated as expensive.
Adding one is not a one-line prompt change; it is a change to six modules and
their tests.

---

## 11. Decisions that were reversed

Documented for the same reason failure modes are documented: the reversal is
often more informative than the original decision.

| # | Decision | Reversed | Why |
|---|---|---|---|
| 1 | Multi-agent debate (Skeptic vs Explainer) | Abandoned in favour of single-pass synthesis | Two round-trips per story, two more JSON parse points, unmeasured quality gain |
| 2 | Six-act narrative structure | Replaced by 2 stories × 4 beats | Padding in middle acts; retention research favoured two distinct stories |
| 3 | SDXL-Lightning images | Replaced by FLUX-class | Could not compose multi-object isometric scenes |
| 4 | Dual-model routing (Qwen 4B + Gemma creative) | Replaced by single large model | Two resident models or a swap per task; both memory-expensive |
| 5 | Cloud-first image/TTS | Replaced by local-first with cloud fallback | Cost, rate limits, content filters, and offline operation |
| 6 | Full-screen scenes | Replaced by 60/40 split screen | Needed a persistent human anchor and a subtitle band |
| 7 | Sentence/7-second captions | Replaced by 5-word karaoke phrases | Long captions cannot highlight; per-word events flickered |
| 8 | numpy audio mastering | Replaced by ffmpeg DSP chain | Had already caused a shape bug; filters encode perceptual reasoning |
| 9 | Vector dedup in-pipeline | Disabled (relocated to CPU post-script) | GPU contention with the 18 GB text model |
| 10 | `intro_hook` + trademark greeting | Removed entirely | Multiplied across 6 downstream modules; redundant with the title overlay |
| 11 | `news_background_sound.mp3` | Reverted to `news-yt.mp3` | Same-day revert; original retained as backup |
| 12 | Music at higher volume | Dropped to −22 dB (0.08) | Masked narration intelligibility |
| 13 | Quarter-split curation fallback | Removed entirely | Silently mis-assigned text to wrong beats |
| 14 | Per-step timeline construction | Single authoritative build | Three construction sites caused duplicated segues and subtitle desync |
| 15 | Windows Task Scheduler + WSL | Replaced by launchd + pmset | Machine changed; needed missed-job firing and a hardware wake |
| 16 | Hardcoded model names/endpoints | Replaced by registry + profile | Could not express 3 runtimes and machine-local model choice |
| 17 | 3 images per story | 4 images per story | 4 beats need 4 scenes (one per beat) |

---

## 12. Anti-patterns this codebase learned to avoid

Stated as rules, each derived from an observed failure.

| Anti-pattern | The failure that taught it | Rule |
|---|---|---|
| **Multiple sources of truth** | Timeline built 3× → duplicated segues, desynced subtitles (ADR-020) | Build a derived artifact exactly once, after all inputs stop changing |
| **Silent fallback that guesses** | Quarter-split assigned text to wrong beats (ADR-021) | If structured output is unusable, reject it — never infer structure |
| **Default `free` memory on macOS** | File cache makes `free` ≈ 0 always | Measure `anonymous + wired`, not free |
| **`os.kill(pid, 0)` for liveness** | Succeeds for unreaped zombie children | Poll the `Popen` handle for spawned processes |
| **`killpg` without checking own group** | Could silently kill the calling pipeline | Compare `getpgid(pid)` against `getpgrp()` first |
| **Bare lock-file existence test** | A crashed run wedges the daily job forever | Check pid liveness and lock age |
| **Relying on `cwd` for imports/`.env`** | Works manually, breaks under launchd | Resolve paths from `__file__`; insert roots into `sys.path` explicitly |
| **Retrying configuration errors** | Missing credential burned 90 s of backoff per run | Classify errors; retry only transient ones |
| **One timeout for a 3-hour workload** | 900 s ceiling killed every ~78 min run | Size the ceiling to the measured p95, and assert supervisors agree |
| **Assuming diffusion can render text** | Garbled glyphs that look like broken UI | Route numbers/text through visual quantities (gauges, stacks, charts) |
| **Chaining LoRAs across model families** | A FLUX.1 adapter on a FLUX.2 model | Validate base family and block coverage before use |
| **Sending flags the runtime rejects** | `mlxgen` exits(2) before loading weights | Probe capabilities; only send supported flags |

---

## 13. Open decisions

Unresolved as of 2026-09-17, with the trade-off stated.

| # | Question | Options | Notes |
|---|---|---|---|
| 1 | **Qwen-Image step count** | 8–12 steps (fast, quality unproven) vs 20–30 (quality, ~3 h runs) | The gating question for the migration. If 8–12 steps is viable the swap is a clear win; otherwise the run must be re-scheduled |
| 1b | **Unevaluated candidate: FLUX.2 Klein + Redmond Klein LoRA** | Download `PIXELART-REDMOND-FLUXKLEIN9B` (~563 MB) and screen it against Qwen-Image | Reconnaissance: the repo was enumerated at 16:45 on 2026-09-17 (refs-only, never fetched) but never benchmarked. It targets the base our model already derives from, and the current path is measured at 5.6 min/image vs Qwen's ~10 min. **A cheap test that could make the 16 GB Qwen download unnecessary** (ADR-036) |
| 2 | **Which LoRA** | Redmond (rank 32, 590 MB) vs Prithiv (rank 64, 1.18 GB) | Both validated structurally; the choice is visual and must be made by comparing rendered output |
| 3 | **`--resume` semantics** | Implement step-skipping vs remove the flag | It currently reuses the project folder but re-runs every step — the name promises more than it delivers |
| 4 | **Postgres** | Give it a consumer vs remove from default deployment | Currently provides no functioning capability (ADR-030) |
| 5 | **Dead code** | Remove vs retain `debate.py`, `news_analysis.py` chain, Pexels import | Removed `debate.py`, `news_analysis.py` and `debate_engine.py` (unreachable; ADR-037), plus `src/video/assembler_tool.py` (zero importers). The `llm_interface` debate methods and prompt entries remain, but now have no caller. The Pexels import (`tools/generate_complete_video.py`) remains in use |
| 6 | **Vision QA** | Keep `skip_vlm=True` vs enable the 4B vision model | Disabled by default; the 3.3 GB model would fit in the post phase |
| 7 | **Embedding role** | Pull `nomic-embed-text` vs stay disabled | Would re-enable cross-run topic memory (ADR-017) |
| 8 | **Image count** | 8 (2×4) vs more scenes | More scenes multiply the dominant cost (image generation) linearly |
| 9 | **Cloud script synthesis** | Keep `gpt-5-mini` primary vs local-only | The only cloud dependency in the reasoning path (ADR-016) |

---

## Appendix A — Commit index of structural changes

Commits that changed the architecture rather than fixing or tuning it.

| Commit | Date | Change |
|---|---|---|
| `54a25d2` | 2026-03-20 | Genesis: multi-agent + open-viking memory |
| `cbe6447` | 2026-03-20 | SDXL Lightning + 6-act structure |
| `932ae5e` | 2026-03-20 | Grounded military pixel art, 60–80 s target |
| `eddb71a` | 2026-03-21 | Category rotation system |
| `57e0e98` | 2026-03-31 | **Split-screen default format** |
| `076b0e9` | 2026-04-01 | Karaoke word-by-word subtitles |
| `519f6c1` | 2026-04-04 | **Option A 60/40 layout + Kokoro + curation** |
| `756da9d` | 2026-04-04 | Comedian delivery (`intro_hook`, punchlines, CTA) |
| `bdd9c4d` | 2026-04-14 | Phase 1: Pydantic + PostgreSQL |
| `86e8748` | 2026-04-14 | Phase 2: LangChain orchestration |
| `aeb656f` | 2026-04-14 | Phase 3: async Playwright scraper |
| `9c9d302` | 2026-04-14 | Phase 5: pgvector memory |
| `f96ec02` | 2026-04-15 | Phase 4: FastAPI replaces Flask |
| `35c2b52` | 2026-04-15 | Phase 6: Docker deployment |
| `d90e12c` | 2026-04-15 | Phase 7: v2 orchestrator + version toggle |
| `c66b909` | 2026-04-15 | Phase 8: structlog |
| `61cf2d4` | 2026-04-24 | **Structural refactor into `src/`, secrets untracked** |
| `ae94017` | 2026-04-28 | Script evaluator + visual QA modules |
| `f3f7964` | 2026-04-29 | **2-story format (Sprint 7)** |
| `f7940f9` | 2026-04-29 | Timeline rebuild, 3 images/story (Sprint 9) |
| `1127424` | 2026-04-29 | **Local FLUX + Kokoro primary** |
| `87f27bb` | 2026-04-30 | **Single-model Gemma 4 migration + VRAM management** |
| `a607634` | 2026-05-05 | GGUF quantisation + VRAM orchestrator |
| `ad66350` | 2026-05-05 | ffmpeg DSP mastering replaces numpy |
| `a2c9e6f` | 2026-05-05 | `num_ctx=32768`, JSON repair |
| `308c07e` | 2026-05-08 | **Script pipeline overhaul: fixer, formatting-only curation** |
| `525b4b4` | 2026-05-09 | **Single timeline source** |
| `276c0c6` | 2026-05-09 | Marker gate; quarter-split removed |
| `e5da148` | 2026-05-11 | Title persistence; algosafe overhaul |
| `e329827` | 2026-05-11 | **`intro_hook` + greeting removed entirely** |
| — | 2026-09-17 | **Uncommitted:** provider abstraction, MLX-Gen, macOS automation, publishing, Qwen benchmark |

---

## Appendix B — Data sources for this document

| Source | Used for |
|---|---|
| `git log` (117 commits, full bodies) | Timeline, decision dates, reversal evidence |
| In-code `WHY:`/`RATIONALE:` blocks | `requirements.txt`, `providers.py`, `registry.py`, `profile.py`, `runtime.py`, `async_scraper.py`, `ollama_mlx_bridge.py` |
| `~/AI/FluxSprites/benchmarks/qwen-selection/` | Migration rationale, provenance methodology, sampling decisions |
| `config/system_prompts.json` `model_config.note` | Timeout re-sizing rationale |
| `src/models/memory.py` + `registry.py` calibration comments | 4 GB reserve derivation, anonymous-memory rules |
| Genesis `README.md` (at `54a25d2`) | Original architecture and stated premise |
| Measured run artifacts (`output/logs/`, `output/projects/`) | Run duration, quality-gate behaviour |

**Reconstruction confidence:**

| Section | Confidence |
|---|---|
| §2–§7, §11, Appendix A | **High** — directly evidenced by commits and code |
| §10 (principles) | **Inferred** — patterns are clear but were never stated in-repo |
| ADR-035 step-count concern | **High for timing** (measured), **open for conclusion** |

---

*This document records decisions, not preferences. Where the codebase
contradicts a stated decision, the code is the current truth and the
discrepancy is listed in [`PIPELINE.md` §13](./PIPELINE.md#13-known-gaps-and-honest-caveats).*
