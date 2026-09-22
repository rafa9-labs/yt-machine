# YT Machine — Pipeline Reference

> **Last updated:** 2026-09-17
> **Branch:** `main` @ `9f13110` (+ uncommitted work: macOS automation, model-provider rewrite, Qwen image benchmarks)
> **Entry point:** `python tools/generate_complete_video.py` (2,084 lines) — invoked by `src/automate.py`, which is invoked by `tools/run_daily.sh` under launchd.

This document describes the complete path from a single command to a published
vertical video: every step, every model, every file, and — critically — **why
each technology is in the stack**. It reflects the current state of the machine,
including the in-progress FLUX.2 Klein → Qwen-Image migration (Redmond LoRA).

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [The one command](#2-the-one-command)
3. [Architecture at a glance](#3-architecture-at-a-glance)
4. [Tech stack and justification](#4-tech-stack-and-justification)
5. [Model runtime: one heavy model at a time](#5-model-runtime-one-heavy-model-at-a-time)
6. [Pipeline steps in detail](#6-pipeline-steps-in-detail)
7. [Video assembly internals](#7-video-assembly-internals)
8. [The image model migration (FLUX.2 Klein → Qwen-Image)](#8-the-image-model-migration-flux2-klein--qwen-image)
9. [Cloud APIs: primary vs fallback](#9-cloud-apis-primary-vs-fallback)
10. [Reliability machinery](#10-reliability-machinery)
11. [Scheduling and automation](#11-scheduling-and-automation)
12. [Configuration reference](#12-configuration-reference)
13. [Known gaps and honest caveats](#13-known-gaps-and-honest-caveats)
14. [Glossary](#14-glossary)

---

## 1. Executive summary

YT Machine converts a fresh news cycle into a ~100-second vertical (1080×1920)
news video with generative pixel-art visuals, a synthetic voiceover, animated
karaoke subtitles, a looping avatar, and platform-specific metadata — then posts
it to YouTube Shorts and TikTok.

Everything runs **locally on one 32 GB Apple Silicon Mac** (M1 Pro). There is no
GPU server, no cloud GPU, and no paid API in the primary path. The only network
calls are: RSS feeds (read), Hugging Face (model download, once), YouTube/TikTok
(publish), and Telegram (notify).

The design premise is a hard physical constraint: **32 GB of unified memory
cannot hold a 17 GB language model and a 17 GB image model at the same time.**
Every architectural decision downstream — the phase system, the subprocess-per-
image model, the lock file, the fail-closed image provider — exists to respect
that constraint.

**One run, end to end (measured 2026-09-15):** 78 minutes wall clock,
8 generated images, 296-word script, 107-second video, 20 MB output.

---

## 2. The one command

```bash
python src/automate.py --publish youtube,tiktok
```

This is the command the daily launchd job runs. It chains:

```
tools/run_daily.sh
  └─ python src/automate.py --publish youtube,tiktok
       ├─ [wake check]            skip locally (no WOL_MAC configured)
       ├─ [lock pre-check]        refuse if another pipeline holds the lock
       ├─ telegram: "started"
       ├─ subprocess: python tools/generate_complete_video.py
       │    ├─ acquire PipelineLock          ← hard gate, one job at a time
       │    ├─ ModelRuntime: start text phase (llama-server, Qwen 27B)
       │    ├─ steps 1–4.8  (news → script, all LLM work)
       │    ├─ ModelRuntime: stop text, verify memory reclaimed
       │    ├─ ModelRuntime: start image phase (mlxgen subprocess per image)
       │    ├─ step 5       (8 images, ~5.5 min each)
       │    ├─ ModelRuntime: end image phase → post
       │    ├─ step 7       (TTS: Kokoro)
       │    ├─ step 8       (assembly: ffmpeg)
       │    ├─ steps 9–10   (metadata, manifest, Telegram)
       │    └─ ModelRuntime: finish → release PipelineLock
       ├─ find latest video
       ├─ publish_video.publish_video(platforms=[youtube, tiktok])
       │    ├─ load platform_metadata.json (title, description, tags)
       │    ├─ ledger check  ← skip platforms already published for this file
       │    ├─ YouTube: OAuth2 resumable upload
       │    └─ TikTok: token refresh → Content Posting API
       └─ telegram: "published: <urls>"
```

For development without publishing:

```bash
python src/automate.py --generate            # generate only, no publish
python src/automate.py --generate --no-notify --skip-images   # fast wiring test
python tools/generate_complete_video.py --dry-run             # canned data, no models
```

---

## 3. Architecture at a glance

```
┌──────────────────────────────────────────────────────────────────────────┐
│ COLLECT                                                                   │
│   config/rss_feeds.json  (19 feeds: BBC, Reuters, AP, Al Jazeera, …)      │
│   aiohttp  →  parallel RSS fetch (8 feeds ≈ 2 s)                          │
│   feedparser → parse XML                                                  │
│   pydantic RSSArticle → validate every entry                              │
│   RSScraper.filter_viral_potential → keyword scoring + category rotation  │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ RESEARCH & DEVELOP  (text model: Qwen3.8-27B Q4_K_M via llama-server)     │
│   trafilatura / Playwright → full article text                            │
│   LangChain structured chain → NewsAnalysis (topic, impact, shift)        │
│   Script synthesis (2 stories × 4 beats)                                  │
│   Script fixer → enforcement (deterministic) → visual prompts             │
│   Curation (text chain) → evaluation (MiniLM + critic LLM)                 │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼  ← MEMORY BARRIER: text model stopped,
                                │     memory reclamation verified (≥80 %)
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ PRODUCE VISUALS  (image model: FLUX.2 Klein 9B 8-bit via mlxgen)          │
│   8 scenes (2 stories × hook / mechanism / truth / fallout)               │
│   subprocess per image → 8 steps, guidance 3.5, 768×816 → 1088×1152       │
│   QA gate: Laplacian sharpness + CLIP relevance + content scrub retries   │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼  ← image children reaped, memory settles
┌──────────────────────────────────────────────────────────────────────────┐
│ PRODUCE AUDIO & VIDEO                                                     │
│   Kokoro TTS → ffmpeg mastering (highpass/EQ/compress/loudnorm)           │
│   faster-whisper word timestamps → ASS karaoke subtitles                  │
│   ffmpeg: 60/40 split screen, Ken Burns, avatar loop, music bed, x264     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ PUBLISH & NOTIFY                                                          │
│   manifest.json + platform_metadata.json                                  │
│   YouTube Shorts (OAuth2) + TikTok (Content Posting API)                  │
│   publish ledger (idempotency) → Telegram summary                         │
│   launchd daily 06:00 · pmset wake 05:50 · AC idle sleep 20 m             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tech stack and justification

Each choice below exists for a specific, verifiable reason — usually a failure
that was observed and fixed. Where the reasoning is a trade-off rather than a
win, it is stated as such.

### 4.1 Language models

| Role | Selection | Provider | Size | Why this one |
|---|---|---|---|---|
| **Text** | `Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf` | `llamacpp` (managed `llama-server`) | 18.5 GB, Q4_K_M | Uncensored (geopolitical content passes content filters), 27B capacity for multi-step JSON reasoning, Q4_K_M is the largest quant that fits the 32 GB budget alongside a 4 GB reserve |
| **Image** | `flux2-klein-base-9b-uncensored-8bit` | `mlxgen` (mflux/MLX) | 17.9 GB, 8-bit MLX | Runs natively on Apple Silicon GPU via MLX; uncensored for conflict imagery; 8-bit keeps anonymous memory near 10.8 GB |
| **Vision (QA)** | `gemma3:4b` | `ollama` | 3.3 GB, Q4_K_M | Only 4B — cheap enough to run during the post phase for image QA; optional, skipped by default (`skip_vlm=True`) |
| **Embedding** | *unconfigured* (`null`) | — | — | Vector dedup is deliberately **disabled** — see below |

**Why Qwen over a smaller model:** the pipeline's hardest task is emitting
*valid JSON* containing two fully-structured stories with four labelled beats
each, under a 1500-token budget. Smaller models truncated JSON mid-string,
which needed a 60-line brace-counting repair hack to survive. A 27B model with
`num_ctx=32768` reduced that failure class to near zero.

**Why `llama-server` and not Ollama for text:** Ollama's default scheduler
manages its own memory and could not be evicted deterministically before the
image phase. `llama-server` is a child process the runtime owns: it can be
launched with exact flags, polled for readiness, and SIGTERM'd in a process
group with verified memory reclamation. That determinism is what makes the
phase system possible.

**Why vector dedup is disabled** (`generate_complete_video.py:784`):
`log.info("dedup.skipped", reason="vector_dedup_disabled_vram_contention")`.
Embedding every topic required calling a *second* Ollama model
(`nomic-embed-text`) while the 18 GB text model was resident. On 32 GB that
pushed into swap. Semantic dedup now happens later, inside
`script_evaluator.py`, using a **CPU-only** SentenceTransformer
(`all-MiniLM-L6-v2`) — same benefit, no GPU contention.

### 4.2 Video and audio

| Component | Technology | Why |
|---|---|---|
| Video compositing | **ffmpeg** (primary), **moviepy** (fallback) | ffmpeg's `zoompan`/`overlay`/`subtitles` filters render a 1080×1920 100 s video in ~90 s on CPU. moviepy spawns a Python process per clip and was 5–10× slower; it survives only as path 3 of a 3-path fallback chain |
| Subtitles | **ASS** (Advanced SubStation Alpha) | The only subtitle format that supports per-word animated colour transitions (`{\t()}`) without re-encoding per frame. Karaoke highlighting is written as one `Dialogue` line per 5-word phrase with timed colour interpolation |
| Word timestamps | **faster-whisper** (`base`, int8/CPU) | Gives word-level timing needed for the karaoke effect. Runs on CPU because the GPU is unloaded after the image phase. Falls back to calibrated even-distribution if unavailable |
| Video encode | **libx264 yuv444p CRF 0** (master) | Lossless 4:4:4 keeps the 32-colour pixel-art palette exact; a subsampled encode decodes it back to thousands of colours. The master is archival and is **not** what gets uploaded — see §7.8 for the size-bounded delivery copy |
| TTS (primary) | **Kokoro 82M** (local) | Free, local, GPU/CPU, no API key, no rate limit. Voice `am_adam` for "authoritative" |
| TTS (tier 2) | **ElevenLabs** `eleven_multilingual_v2` | Premium quality when `ELEVEN_LABS_KEY` is set. Not primary because it costs money and adds a network dependency to an otherwise offline pipeline |
| TTS (tier 3) | **Edge TTS** | Free Microsoft voices. Last resort before silence |
| Audio mastering | ffmpeg DSP chain | `highpass=f=80` → `equalizer=f=4000:g=2` → `acompressor` → `loudnorm I=-16 LRA=11 TP=-1.5`. Replaced an earlier crude limiter that pumped audibly |
| Music bed | `assets/avatar/music/news-yt.mp3` | Mixed at **0.08 (−22 dB)** with a 10 s fade-out. Chosen after measuring: −14 dB (0.2) masked narration |
| Avatar | `assets/avatar/avatar_loop.mp4` | Bottom 40 % of frame; a pre-rendered loop avoids generating a talking head per run |

### 4.3 Text processing and orchestration

| Component | Technology | Why |
|---|---|---|
| Article extraction | **trafilatura** (tier 1) → **Playwright Chromium** (tier 2) | Most sites serve static HTML where trafilatura is instant. Reuters/Foreign Policy render content via JavaScript and return an empty shell to trafilatura — Playwright launches a real browser and waits for `networkidle`. Two tiers avoid paying the ~500 ms browser cost on the easy majority |
| RSS fetch | **aiohttp** + **feedparser** | 19 feeds fetched in parallel (~2 s total) vs ~16 s sequential. `feedparser` because RSS is static XML — launching a browser to parse XML would be absurd |
| Structured LLM output | **LangChain** + **Pydantic** | `PydanticOutputParser` auto-retries when the model emits malformed JSON, replacing a hand-rolled brace-counting extractor. Pydantic gives field-level validation and a typed contract between steps |
| Parsing/validation | **pydantic v2** | `RSSArticle`, `NewsAnalysis` etc. Invalid articles are skipped rather than crashing the run |
| Logging | **structlog** | Machine-parseable key-value events (`step.start`, `step.complete` with `duration_s`) enable per-step timing analysis. Falls back to stdlib `logging` |
| Sentence similarity | **sentence-transformers** (`all-MiniLM-L6-v2`) + **scikit-learn** cosine | CPU-only duplicate-story detection at 0.90 threshold |

### 4.4 Persistence, publishing, automation

| Component | Technology | Why |
|---|---|---|
| Source of truth | **JSON files** in `output/projects/video_<ts>/` | Degrades gracefully: if Postgres is down, the run still succeeds. Files are inspectable, diffable, and resumable by hand |
| Relational store | **PostgreSQL 16 + pgvector** (Docker) | Progress tracking for the API/n8n surface and vector topic memory. **Currently not running** — the pipeline logs `postgres.save_failed` warnings and continues |
| API surface | **FastAPI** + **uvicorn** | `/generate`, `/status`, `/latest`, `/publish` with Pydantic validation and auto-generated `/docs`. Enables n8n orchestration |
| Workflow automation | **n8n** (Docker) | Optional external orchestrator; the launchd path does not depend on it |
| YouTube upload | **google-api-python-client** + `google-auth-oauthlib` | Official Data API v3, resumable upload, OAuth2 with a cached refresh token so unattended runs work |
| TikTok upload | TikTok **Content Posting API** via `requests` | Official Direct Post endpoint. Refresh token flow added because access tokens expire in ~24 h |
| Scheduling | **launchd** + **pmset** | launchd's `StartCalendarInterval` fires missed jobs when the Mac wakes (cron cannot). `pmset` is the only way to schedule a hardware wake |
| Wake/sleep control | **caffeinate**, **pmset** | `caffeinate -s` holds the machine awake for the ~80 min run; AC idle sleep (20 min) puts it back down afterwards |
| Notifications | Telegram Bot API via `requests` | Status + video delivery (50 MB hard limit; the delivery copy is sized to fit — §7.8) |

### 4.5 Why not the alternatives

| Rejected | Reason |
|---|---|
| Cloud GPU (RunPod etc.) | Cost per run + cold-start model download + upload/download the 20 MB video. The Mac is idle at 05:50 anyway |
| Stable Diffusion 1.5/XL for pixel art | 512×512 native, poor at isometric scene composition with multiple objects. FLUX-class models handle layout and coherence far better |
| ComfyUI | A GUI-first graph runner. The pipeline needs a scriptable, deterministic CLI with a documented exit code — not a workflow JSON with a web server |
| Ollama for the big text model | Its scheduler owns memory; determinism was impossible. (Still used for the 4B vision model, where determinism does not matter) |
| PIL/Pillow text rendering for subtitles | Per-frame CPU compositing was slower than ffmpeg's ASS renderer and produced worse text shaping |
| Streaming/SSE from the LLM | The pipeline consumes complete JSON objects, not token streams. Non-streaming with generous timeouts is simpler and matches the workload |

---

## 5. Model runtime: one heavy model at a time

### 5.1 The constraint

On a discrete-GPU system, system RAM and VRAM are separate pools. On Apple
Silicon, unified memory is **one pool**: a 17 GB text model plus a 17 GB image
model is a 34 GB request on a 32 GB machine. Result: swap, then a jetsam kill.

The codebase makes this constraint explicit in two places:

- `src/models/memory.py:6` — "Loading Qwen (~20 GiB) and Flux (~10+ GiB) would
  exceed a 32 GiB machine"
- `tools/generate_complete_video.py:1340` — "This is the critical memory
  barrier: Qwen must be fully stopped and its memory returned before the image
  model is allowed to load."

### 5.2 State machine

```
        ┌──────────────────────────────┐
        │            idle              │   nothing resident
        └──────────────┬───────────────┘
                       │ start_text_phase()
                       │  • reap image stragglers
                       │  • require_capacity(18.2 GB + 4 GB reserve)
                       │  • launch llama-server, poll /v1/models
                       ▼
        ┌──────────────────────────────┐
        │            text              │   Qwen 27B resident (~20 GB)
        │  steps 1 … 4.8, build_timeline│
        └──────────────┬───────────────┘
                       │ start_image_phase()
                       │  • stop_text_model()  ← SIGTERM process group
                       │  • verify ≥80 % anonymous memory reclaimed (90 s)
                       │  • wait_for_capacity(10.8 GB + 4 GB, 60 s)
                       ▼
        ┌──────────────────────────────┐
        │            image             │   mlxgen subprocess per image
        │  step 5: 8 scenes, sequential │   one at a time, reaped after each
        └──────────────┬───────────────┘
                       │ end_image_phase()
                       │  • reap stragglers
                       │  • wait for available ≥ reserve + 2 GB
                       ▼
        ┌──────────────────────────────┐
        │            post              │   neither heavy model resident
        │  step 7 TTS, step 8 assembly │   Kokoro small-model only
        └──────────────┬───────────────┘
                       │ finish()   (idempotent; atexit + SIGINT/SIGTERM)
                       ▼
        ┌──────────────────────────────┐
        │            idle              │   lock released
        └──────────────────────────────┘
```

### 5.3 Memory accounting

The guard uses **anonymous** memory, not "free" memory. On macOS, `free` is
near-zero at all times because the kernel uses spare RAM as file cache —
looking at `free` would make the machine appear permanently full.

```
safe_to_load  ⇔  (total_gb − anonymous_gb) ≥ needed_gb + reserved_gb
```

- `anonymous_gb` = `Anonymous pages + wired` (from `vm_stat`)
- `needed_gb` = **on-disk weights are not counted** — safetensors stay
  file-backed and are evictable; only what lands in anonymous memory matters
- `reserved_gb = 4.0` (calibrated: an 18.4 GB load left 4.7 GB headroom with
  `swap 0.0 GB`; a 6 GB reserve made the same load impossible)

Estimates (`registry.py:128-155`):

| Model | On disk | `estimated_anonymous_gb` | Rule |
|---|---|---|---|
| Qwen 27B Q4_K_M | 17.2 GiB | **18.2 GB** | `max(disk × 1.0 + 1, 4)` — llama.cpp/Metal keeps weights resident |
| FLUX.2 Klein 9B 8-bit | 16.6 GiB | **10.8 GB** | `max(disk × 0.65, 6)` — safetensors stay file-backed |

Two functions enforce it:
- `require_capacity()` — fails closed immediately before launching the text model
- `wait_for_capacity()` — polls every 3 s (60 s budget) before the image phase

### 5.4 Process lifecycle

**llama-server** is launched in its own process group
(`start_new_session=True`) so a timeout can kill the whole group — otherwise
worker threads survive and the next run OOMs.

**Adoption rule:** a server already listening on `:8080` is adopted *only if*
its command line contains the selected model's filename. Anything else raises
an error rather than killing a process the runtime did not start
(`runtime.py:450-456`).

**Verified teardown:** terminating the process is not the same as reclaiming
its memory. `stop_text_model()` samples anonymous memory before the kill and
requires ≥80 % of `estimated_anonymous_gb` to return within 90 s, warning
loudly below 50 % (`runtime.py:568-631`).

**Zombie trap:** `os.kill(pid, 0)` succeeds for an exited-but-unreaped child.
`ManagedProcess.alive` polls the `Popen` handle instead, which reaps the child
and reveals true exit status. Without this, shutdown burned the full SIGTERM
grace period plus the SIGKILL timeout on processes that were already gone
(measured: 120 s per phase transition instead of ~1 s).

**Image model is a subprocess per image.** `mlxgen generate` is invoked once
per scene and exits. Consequences: memory returns to the OS deterministically
on exit; a hung generation cannot poison the pipeline; the runtime can
*guarantee* no image model is resident after the image phase. Cost: model load
overhead per image (`mlxgen` handles caching internally via a 4 GiB MLX cache
limit).

---

## 6. Pipeline steps in detail

Step names below are the exact `step.start` / `step.complete` log keys.

### STEP 1 — `news_fetch`

| | |
|---|---|
| **Module** | `src/collector/async_scraper.py` (`AsyncScraper`), `src/collector/rss_scraper.py` (`RSScraper`) |
| **Input** | `config/rss_feeds.json` — 19 feeds (BBC, Reuters, AP, NPR, Al Jazeera, Foreign Policy, Stratfor, Defense One, Middle East Eye, The Diplomat, SCMP, …) |
| **Method** | `asyncio.run(scraper.scrape_all(max_age_hours=24))` → aiohttp parallel fetch → feedparser parse → pydantic validation |
| **Selection** | `RSScraper.filter_viral_potential(all_articles, top_n=10)` — keyword scoring (`+4` title / `+1` summary per geopolitical keyword), **+10 category rotation boost**, `+3` virality keywords, `+3` feed priority |
| **Dedup** | `_is_semantically_similar` — ≥3 shared words among the first 6 words of the title |
| **Output** | Top `NUM_STORIES=2` articles selected |
| **Fallback** | async fails → sync `RSScraper.scrape_all`; both fail → `exit(1)` |
| **Checkpoint** | `news_fetch` → `{article_titles: [...]}` |
| **Disabled** | vector dedup (`dedup.skipped`, VRAM contention — see §4.1) |

**Category rotation** is why the second story isn't always another Middle East
conflict: `output/category_rotation.json` tracks which category ran when, and
today's category gets a +10 score boost.

### STEP 2 — `news_analysis`

| | |
|---|---|
| **Module** | `src/brain/langchain_interface.py` (`build_structured_chain`), `src/brain/llm_interface.py` (`process_news`) |
| **Per article** | 1. fetch full text (`RSScraper.get_full_article_text`, 15 s timeout, falls back to RSS `summary`)<br>2. **primary:** LangChain structured chain → `NewsAnalysisModel`<br>3. **fallback:** raw `llm.process_news()` |
| **Model** | Qwen 27B, `num_ctx=32768`, timeout 420 s |
| **Extracts** | `topic`, `impact_score`, `shift_vector`, `category` |
| **Sorting** | ascending by `impact_score` — the **strongest story lands last** for retention |
| **Guard** | `< 2` analyses → `exit(1)` |
| **No cloud fallback** | explicit: geopolitical content may trigger cloud content filters (`llm_interface.py:431`) |
| **Checkpoint** | `news_analysis` → `{analysis_topics: [...]}` |

### STEP 3 — `trending_context`

Pure-Python n-gram frequency over all articles (`TrendingAnalyzer`), no LLM.
Failure is a warning only. **Note:** the result is currently not consumed
downstream — a vestigial step kept for future metadata use.

### STEP 4 — `script_synthesis`

| | |
|---|---|
| **Method** | `llm.synthesize_multi_news_script(analyses, 2)` under heartbeat (8 s, 1800 s timeout) |
| **Retries** | 3 outer attempts; inner JSON repair escalates token budget ×2, ×4 with a stricter system prompt |
| **LLM order** | 1. **OpenAI `gpt-5-mini`** if `OPENAI_API_KEY` is set (2× → 8× budget retry)<br>2. local Qwen |
| **Shape** | 2 stories × (`part1` hook, `part2` mechanism, `real_talk` truth, `fallout`) + segue + closing |
| **Post-processing** | greeting forced empty (video opens straight into story 1); `full_text` rebuilt if absent; `word_count`, `estimated_duration = words / 2.5` |
| **Output** | `script.txt`, `script_segments.json` |
| **Checkpoint** | `script_synthesis` → `{script_word_count}` |

**Why OpenAI is a fallback, not the primary:** script synthesis is the single
most failure-prone LLM call (longest output, strictest structure). The cloud
model produces better structured output for this one task. Everything else —
including all news analysis — stays local. The trade-off is accepted because
the synthesized script is not sensitive source material; it is a summary of
public reporting.

### STEP 4.05 — `script_fixer`

An LLM pass that repairs the script's structural integrity. Only accepted if
the result is a dict with `stories`; otherwise the original is kept silently.
Enforces: ≥2 stories, empty greeting, segue only on story 1, strips 4+ word
prefix overlaps, strips fallout echo from the closing, requires the trademark
closing, and runs a per-story content-fidelity check that rejects any rewrite
that loses substance.

### STEP 4.1 — `script_enforcement`

**Deterministic Python only — no LLM.** Runs in fixed order:

1. `_enforce_greeting` — 2. `_enforce_segues` — 3. `_dedup_segue_overlap` —
4. `_dedup_inter_story_phrases` — 5. `_enforce_fallout` —
6. `_ensure_greeting_in_fulltext`

This step is why the pipeline is not fully at the mercy of the model: every
run gets valid structure regardless of what the LLM produced.

### STEP 4.5 — `visual_prompts`

| | |
|---|---|
| **Method** | `llm.generate_visual_prompts(script, articles, analyses)` under heartbeat (8 s, 900 s) |
| **Requested** | exactly `stories × 4 = 8` scenes: `story_N_part1` (HOOK), `story_N_part2` (MECHANISM), `story_N_real_talk` (TRUTH), `story_N_fallout` (FALLOUT) |
| **Constraints** | 2–4 full sentences each; geography restricted to countries named in the narration or STORY CONTEXT; **never request rendered text/letters/numbers** — routed through charts, gauges, symbols instead |
| **Validation** | accepted only if `≥ NUM_IMAGES` (8) valid prompts |
| **Fallback** | `_build_fallback_prompt` (in `src/pipeline_utils.py`) derives prompts from narration text |
| **Prompt source** | `config/system_prompts.json` → `visual_prompt_generator` |

**The "never request text" rule is load-bearing.** Diffusion models cannot
render readable characters; asking for a label produces garbled pseudo-text
that looks like a broken UI. This is why the prompt instructs the model to
express numbers as visual quantities (stacked blocks, gauge position, rising
line) instead.

### STEP 4.7 — `script_curation`

| | |
|---|---|
| **Primary** | LangChain text chain (`CurationChain.curate_stories`) — text, not JSON, because output carries `[STORY N]` markers |
| **Fallback** | raw `llm.curate_script()` under heartbeat (8 s, 900 s) |
| **Scope** | **only story bodies** are sent. Segues, separators, and closing are preserved deterministically |
| **Validation** | `_validate_curation_fidelity` — rejects unmarked output; unmarked output no longer falls back to a naive quarter-split (that fallback was removed after it produced mangled scripts) |
| **Reassembly** | `_parse_curated_stories` → `_validate_curation_fidelity` → `_reassemble_script` |
| **Output** | `script_curated.txt` (only when changed) |
| **Checkpoint** | `script_curation` → `{curated_word_count}` |

### STEP 4.8 — `script_evaluation`

| | |
|---|---|
| **Module** | `src/brain/script_evaluator.py` |
| **Phase A** | Semantic dedup — `all-MiniLM-L6-v2` embeddings, cosine ≥ 0.90 flags near-duplicate stories; the weaker story is regenerated via the LLM |
| **Phase B** | Continuity critic — LLM returns `{issues:[{type, story_index, field, description, suggested_fix}]}`; each fix validated (≥5 words, no CTA patterns, ≥50 % of original length) |
| **CPU only** | deliberately avoids a GPU embedding model to prevent VRAM contention |
| **Checkpoint** | `script_evaluation` → `{stories}` |

### BUILD TIMELINE — `build_timeline` (no banner)

Deterministic mapping from script segments to image indices:

```
image_idx(story i, beat) = i*4 + {part1:0, part2:1, real_talk:2, fallout:3}
segue  → img_base + 3        (previous story's final image)
closing → (len(stories)-1)*4 + 3
```

Also strips segue↔part_1 prefix overlaps ≥3 words. **This is the single source
of truth for the timeline** — an earlier design built the timeline earlier and
experienced duplicated segues and dropped subtitles when later steps mutated
the script. Building it after evaluation fixed both.

### MEMORY BARRIER

```
stop text model → verify reclamation → start image phase
```

Fatal exits: `exit(4)` if the MLX-Gen provider is unavailable, `exit(5)` if the
image phase cannot start. LoRA attachment comes from the generation profile:

```python
lora = image_generation_profile.get("lora") or {}
lora_path = lora.get("path")
if lora_path and not Path(lora_path).exists():
    print(f"\nFATAL: Configured LoRA is missing — {lora_path}")
    sys.exit(4)
if lora_path:
    mlx_provider.set_loras([lora_path], [float(lora.get("scale", 1.0))])
```

A configured-but-missing adapter is a hard failure, not a silent downgrade.

### STEP 5 — `pixel_art`

| | |
|---|---|
| **Count** | 8 scenes (2 stories × 4 beats) |
| **Seeds** | `base_seed = project_id % 2^32`; per scene `base_seed + idx + attempt*100` |
| **Attempts** | up to 4 per scene |
| **Primary provider** | **MLX-Gen subprocess** (fail-closed — no cloud fallback) |
| **Sampling** | 8 steps, guidance 3.5 (env `MLXGEN_STEPS`, `MLXGEN_GUIDANCE`) |
| **Resolution** | render 768×816 → nearest-neighbour upscale to 1088×1152 |
| **QA gate** | `src/video/visual_qa.py::validate_image(skip_vlm=True)` — Laplacian variance ≥ 100.0 (sharpness) + CLIP ViT-L/14 similarity ≥ 0.22 (relevance). Imported at `generate_complete_video.py:353`, called at `:1555` |
| **Failure detection** | `_detect_failed_image` catches solid-colour / monochrome / flat-channel output |
| **Content scrub** | `_progressive_content_scrub` levels 1–3 on repeated failure; fallback to category-safe prompt templates |
| **Heartbeat** | `orchestrator.heartbeat(f"image_gen_{n}/8")` every 2 scenes |
| **Checkpoint** | `pixel_art` → `{image_count}` |
| **Exit on failure** | `exit(6)` when a managed local provider fails |

**Local prompt construction differs from the cloud prompt** and this is
deliberate. The fal.ai prompt layers three overlapping style blocks plus a
palette because its LoRA needs reinforcement. Measured against FLUX.2 Klein at
identical settings and seed, that 153-word prompt produced **garbled
pseudo-text and disconnected fragments**; a single concise style tail produced
a clean image. Hence:

- `build_local_generation_prompt()` — scene first, then a style tail trimmed to
  `LOCAL_MAX_STYLE_WORDS = 14`
- `sanitize_text_requests()` — rewrites any prompt requesting literal text
- `_strip_lora_triggers()` — removes fal.ai trigger words (inert tokens locally
  that pull toward sprite-sheet imagery)
- `clamp to LOCAL_MAX_SCENE_WORDS`

### STEP 7 — `voice_generation`

*(Step 6, stock video footage, was removed — `fetch_vertical_footage` is
imported but never called. All visuals are generated pixel art.)*

| | |
|---|---|
| **Module** | `src/video/tts_tool.py` |
| **Voice selection** | `select_voice_for_content()` maps keywords → tone: `military/conflict/war → authoritative`, `economic/sanctions → professional`, `breaking/urgent → energetic`, … |
| **Engine order** | 1. **Kokoro** (local, `am_adam`, speed 1.0)<br>2. **ElevenLabs** (`eleven_multilingual_v2`)<br>3. **Edge TTS**<br>4. silent audio (zeros) — prevents assembly from failing outright |
| **Pacing** | `STRUCTURAL_SILENCE = 0.80 s` at `....` boundaries, `CLOSING_SILENCE = 0.30 s`, outro chunk sped to 0.85 |
| **Mastering** | ffmpeg: `highpass=f=80:t=0.7071, equalizer=f=4000:t=q:w=1.2:g=2, acompressor=threshold=0.125:ratio=2:attack=10:release=100:knee=4:makeup=1, loudnorm=I=-16:LRA=11:TP=-1.5`, `-ar 44100` |
| **Outro** | `aecho=0.8:0.88:60:0.4` applied to the **last 5 seconds only** |
| **Timestamps** | faster-whisper `base`, `word_timestamps=True`, `beam_size=5`, CPU int8; fallback = calibrated even distribution |
| **Output** | `voiceover.mp3` |
| **Checkpoint** | `tts` → `{voice_duration}` |

**Docstring vs reality:** the `generate_voiceover` docstring still says
"ElevenLabs (primary) or Edge TTS (fallback)". The actual code order is
**Kokoro → ElevenLabs → Edge → silence**. Kokoro is primary because it is
local, free, and unlimited; ElevenLabs is the quality upgrade if you supply a
key.

### STEP 8 — `video_assembly`

See §7 for the full breakdown. Summary: fuzzy word-matching aligns each script
segment to the Whisper timestamps, per-story beat balancing (25/30/25/20 %),
minimum 2 s per image, 0.3 s pre-roll, then a three-path ffmpeg/moviepy
fallback chain with `_validate_mp4` gating every attempt.

### STEP 9 — `platform_metadata`

Pure Python (`PlatformMetadataGenerator`), **no LLM**. Writes
`platform_metadata.json` with per-platform `title`, `description`/`caption`,
`cta`, plus `common_hashtags`. This is the file the publisher reads for
uploads — which is why the publisher had to be fixed to derive `project_dir`
from an explicit `--video` argument (it was silently falling back to the
generic placeholder title).

### STEP 10 — `project_summary`

Writes `manifest.json` (articles, analyses, full script, platform metadata,
asset paths, TTS timestamps, project folder), updates
`output/category_rotation.json` and `output/video_history.json`, saves the
final Postgres status, stores topic vectors (best-effort), then delivers the
video to Telegram.

The manifest also carries a run-level `provenance` block: project id, status
(`complete` / `incomplete`), git commit, generation-profile identity, LLM
identifier, TTS engine and voice, assembly settings, and the list of per-image
provenance filenames. `assets.provenance` lists the sidecars copied into
`images/` — see §7.7.

**Telegram delivery:** `sendVideo` multipart, `parse_mode=HTML`,
`supports_streaming=true`, 120 s timeout. Caption = hook title (≤60 chars) +
date + first 3 lines of the YouTube description + up to 8 hashtags, truncated
to 1024 chars. Hard 50 MB limit; the delivery copy is encoded to fit (§7.8).

Finally: `llm.unload_model()` → `orchestrator.phase_cleanup()` →
`runtime.finish()` (idempotent: stop text model, reap stragglers, release
lock, phase → idle).

---

## 7. Video assembly internals

**Module:** `src/video/split_video_assembler.py::build_split_video`

### 7.1 Geometry

```
1080 × 1920  (9:16 vertical)
┌──────────────────────────┐
│                          │
│   SCENE (generated art)  │  TOP_H = 1152   (60 %)
│   Ken Burns zoom/pan     │
│                          │
│─── subtitles @ y=1012 ───│  subtitle band height 140
├──────────────────────────┤
│                          │
│   AVATAR LOOP (video)    │  BOTTOM_H = 768  (40 %)
│                          │
└──────────────────────────┘
FPS = 30 · background pad colour 0x0A0519
```

### 7.2 Camera motion — disabled for the pixel-art profile

`SCENE_ZOOM_PROFILES` is retained as a **naming table only**: the four beats
still carry their labels (`HOOK`, `MECHANISM`, `TRUTH`, `FALLOUT`) for logs and
callers, but every `zoom_start`/`zoom_end` is `1.0` and `pan_x` is `0.0`.

Continuous camera motion is intentionally disabled. Fractional resampling of a
background that was generated on a strict logical grid creates shimmer, and
that grid is what makes the pixel art read as pixel art. The scene still
changes every beat, and the beat change itself carries the rhythm.

The only zoom/pan implementation lived in `src/video/assembler_tool.py`, which
was removed — it had zero importers, and `build_split_video` never called it.

### 7.3 Subtitles (ASS)

- Title overlay: Arial 48, yellow, `\pos(540,20)`, Alignment 8, max 10 words,
  **persists for the entire video** (`title_end_seconds = total_duration`)
- Captions: Arial 64, black outline 5, shadow 2, Alignment 2, at y=1012
- 5-word phrases; animated colour per word via `{\t()}` — grey `(180,180,180)`
  upcoming → gold `(255,215,0)` active → white `(255,255,255)` spoken
- `SUBTITLE_DELAY = 0.04 s`, lead-in 0.05 s
- **One `Dialogue` line per phrase** with interpolation, not one per word —
  per-word lines caused visible flicker
- Word timing from `align_whisper_to_script` (anchor matching + interpolation),
  gap-aware redistribution at `PAUSE_THRESHOLD = 0.3 s`

### 7.4 Three-path fallback chain

1. **Pure ffmpeg** (`_assemble_pure_ffmpeg`) — scenes rendered by OpenCV
   (`_render_scene_opencv`) or ffmpeg `zoompan`; avatar loop rendered by
   ffmpeg; concat demuxer; pad+overlay stack; ASS burn-in via `subtitles='…'`;
   fade; audio mix. Final encode uses the master codec:
   `-c:v libx264 -pix_fmt yuv444p -crf 0 -preset fast -movflags +faststart
   -c:a aac -b:a 192k -ar 44100 -ac 2`
2. **moviepy overlay pre-render** — only if pure ffmpeg fails and overlays
   exist. Base render CRF 18; overlays as VP9 alpha WebM composited with
   ffmpeg `overlay=0:0`
3. **Full moviepy** — same master codec and audio chain

All three paths write the **master**: lossless 4:4:4, no size control, because
compressing here would defeat the palette guarantee that the master exists to
provide. Size discipline belongs to the delivery stage instead (§7.8).

Every path's output passes `_validate_mp4`, which rejects
`moov atom not found` / `Invalid data` and requires a video stream. An invalid
file is deleted and the next path runs. (Note: the docstring claims it checks
for an audio stream too; it currently only requires video — audio absence is
caught downstream by the publisher and player.)

### 7.5 Audio mix

```
[1:a]atrim=0:<total>,asetpts=PTS-STARTPTS,highpass=f=80,
     afade=t=in:st=0:d=1,volume=0.08,afade=t=out:st=<total-10>:d=10[music];
[0:a][music]amix=inputs=2:duration=longest:normalize=0,
     afade=t=out:st=<total-0.8>:d=0.8[out]
```

Music at **0.08 (−22 dB)** with a 10 s tail fade; the voiceover/dub fades out
over the final 0.8 s. `normalize=0` is important — the default would
re-normalise the mix and duck the narration.

### 7.6 Scene duration alignment

1. `_fuzzy_find_segment` matches each timeline segment to word timestamps
   (prefix match on first 4 chars, ≤3 skipped words, ≥2 matches required)
2. Proportional gap fill between matched anchors
3. Gap bridging — split points at 70 % of each gap
4. Per-story beat balancing: **hook 25 % / mechanism 30 % / truth 25 % /
   fallout 20 %**, minimum 10 % per image, minimum 2 s per image (stolen from a
   pair partner only if that partner is >4 s)
5. `PREROLL_OFFSET = 0.3 s` — the image appears slightly before the narration
   that describes it

(The code comment says "~1s"; the constant is 0.3.)

### 7.7 Provenance sidecars

Every generated image carries a JSON sidecar next to it, named
`<image>.provenance.json`. The generator writes it beside the scratch PNG in
`output/images/`; the pipeline copies it into the project folder with the
image it describes, so the record travels with the deliverable:

```
output/projects/video_<id>/images/
  story_1_part1_scene_….png
  story_1_part1_scene_….provenance.json
```

The sidecar records `model`, `provider`, `generation_profile`, both prompts
(`prompt` as sent, `original_prompt` before sanitisation), the `sampling` block
(steps, guidance, seed, width, height), the `lora` block (name, path, scale),
the `postprocess` block (input/output sizes, colours requested and written,
resampling), any rewritten text requests, deferred text, and the geopolitical
accuracy score. The pipeline adds `project_id`, `scene`, and
`processed_output` when copying.

**Secrets.** A sidecar is a shareable file, so `src/video/provenance.py`
redacts credential-shaped keys and values on every write. Key names matching
`*_key`, `token`, `secret`, `password`, `credential`, and similar are redacted,
and so are values matching known provider formats (`ghp_`, `hf_`, `sk-`,
`AKIA`, PEM headers, …). The value check exists because a credential pasted
under an innocent key name is still a credential.

**Lifecycle.** Sidecars are written per-image as each image is accepted, so a
partially failed run still records the images it did produce. The run-level
record in `manifest.json` is written at finalization with
`status: "complete"` or `"incomplete"`. Set `"provenance": false` in a
generation profile to disable sidecars; the key is validated (boolean only)
and defaults to enabled, including for profiles written before it existed.

---

### 7.8 Master and delivery artifacts

A run produces two artifacts with different contracts:

| Artifact | Name | Format | Purpose |
|---|---|---|---|
| Master | `video_<id>.master.mp4` | libx264 / yuv444p / CRF 0 | Archival. Keeps the 32-colour palette exactly |
| Delivery | `video_<id>.mp4` | libx264 / yuv420p / High | Upload. Size-bounded for providers |

**Why two files.** The palette guarantee and the size constraint are
irreconcilable: a lossless 1080×1920 master runs ~38 Mbit/s, so a full run is
~400–500 MB, while Telegram's Bot API refuses anything over 50 MB and the
TikTok and Instagram uploaders read the entire file into memory. Measured on a
real 32-colour frame, a yuv420p encode decodes back to 4,696 colours — so the
master must stay 4:4:4 and the delivery copy must accept that damage.

**When delivery runs.** Only when the master exceeds `YT_DELIVERY_MAX_MB`
(default 50). Below the ceiling no second file is written and the canonical
name keeps pointing at the master, so short runs behave exactly as they did
before this stage existed.

**How the size is guaranteed.** The video bitrate is derived, not guessed:

```
video_kbps = (max_bytes × 8 / duration) − audio_kbps   (with container slack)
```

The encode caps that bitrate (`-maxrate` / `-bufsize`) on top of a CRF, so
quality stays constant on the simple frames that dominate pixel art while the
byte ceiling holds. Two-pass VBR reaches the same ceiling but doubles encode
time and needs a stats file. If the ceiling cannot yield a usable bitrate
(a very long video in a small budget), the stage reports that explicitly
rather than emitting an unwatchable file, and if a single attempt overshoots,
one retry runs at a tighter CRF before reporting failure.

**Naming.** The delivery copy takes the canonical `video_<id>.mp4` and the
master is renamed to `.master.mp4`. Every consumer — manifest,
`publish_video.find_latest_video`, `automate.find_latest_video`, `server.py`,
Telegram — already resolves the canonical name, so nothing needed to change to
receive the upload copy. Both discovery functions share one selector that
prefers the delivery copy and excludes `.master.mp4`, and falls back to the
master if delivery failed or was disabled (publishing something beats
publishing nothing).

**Failure behaviour.** Delivery is a convenience; the master is the artifact.
A failed transcode restores the canonical name, logs a specific reason, and
never fails the run.

**Configuration:** `YT_DELIVERY_ENABLED` (default true),
`YT_DELIVERY_MAX_MB` (default 50), `YT_DELIVERY_CRF` (default 20).

---

## 8. The image model migration (FLUX.2 Klein → Qwen-Image)

> **Status: IN PROGRESS as of 2026-09-17.** The pipeline still runs FLUX.2
> Klein. A Qwen-Image 2512 benchmark is actively running on this machine.

### 8.1 Why migrate

FLUX.2 Klein is step-distilled: it has **no CFG branch**, therefore
`supports_negative_prompt: false`. Every quality problem that could normally be
suppressed with a negative prompt (garbled pseudo-text, unwanted text, sprite-
sheet composition) has to be fought in the positive prompt instead. The
pipeline carries several workarounds purely for this limitation:

- `sanitize_text_requests()` rewrites any prompt that asks for literal text
- `_strip_lora_triggers()` removes trigger words that are inert locally
- `LOCAL_MAX_STYLE_WORDS = 14` caps the style tail because long prompts made
  output worse, not better
- The "never request text/letters/numbers" section in the visual prompt system
  prompt exists because the model cannot follow it reliably

Qwen-Image supports negative prompts (`supports_negative_prompt: true`) and has
a larger, more current ecosystem of pixel-art LoRAs.

### 8.2 The benchmark methodology

A dedicated, provenance-tracked harness lives at
`~/AI/FluxSprites/benchmarks/qwen-selection/`:

**`gen.sh`** wraps `mlxgen generate` so every image carries a
`.provenance.json` sidecar with the exact model, LoRA path, scale, prompt,
seeded sampling parameters, wall time, exit code, byte size, the runtime's LoRA
application report, and a post-generation free-memory snapshot. **Rationale
(from the script itself):** the selection decision must be auditable —
reproducing an image later requires more than the PNG.

**`preflight.json`** records the environment before any generation:

```json
{
  "python": "Python 3.13.15",
  "mlx": "0.31.2",
  "mlx_gen": "0.37.0",
  "machine": "MacBookPro18,1",
  "cpu": "Apple M1 Pro",
  "memory_gb": 32.0,
  "disk_free_gb": "280",
  "memory_pressure_free_pct": "92%",
  "preserved_model": ".../flux2-klein-base-9b-uncensored-8bit",
  "preserved_model_size": "17G"
}
```

The existing FLUX.2 Klein checkpoint is explicitly **preserved** — the migration
is additive, not destructive.

**`qwen-capabilities.json`** is the live capability probe for
`AbstractFramework/qwen-image-2512-4bit`:

| Capability | Value |
|---|---|
| `public_task` | `text-to-image` (handler `qwen.generate`) |
| `supports_guidance` | `true` |
| `supports_negative_prompt` | **`true`** ← the migration's whole point |
| `supports_lora` | `true` (target role: `transformer`) |
| `lora_status` | `mapped-unvalidated` |
| `dimension_multiple` | 16 |

**`sampling-rationale.json`** documents the chosen defaults with sources: the
official Qwen-Image-2512 card uses `true_cfg_scale=4.0` with 50 steps; the
smoke test deliberately used 20 steps to prove the route executes without
paying the full cost on an unvalidated path. Step count is deferred to tuning.

### 8.3 LoRA candidacy validation

`phase5-lora-base-validation.json` records structural validation of both
candidate pixel-art LoRAs. This exists because **mlxgen's `LoRACompatibility`
gate skips validation for absolute local paths** — so validation was performed
manually and recorded:

| | Redmond | Prithiv |
|---|---|---|
| File | `[Qwen.Image]PixelArt_Redmond.safetensors` | `Qwen-Image-2512-Master-Pixel-Art-LoRA.safetensors` |
| Size | 590 MB | 1.18 GB |
| Declared base | `Qwen/Qwen-Image-2512` (via model tree; frontmatter omits it) | `Qwen/Qwen-Image-2512` (frontmatter) |
| `ss_base_model_version` | `qwen_image` | `qwen_image` |
| Block coverage | **0..59 (60) — matches** | **0..59 (60) — matches** |
| Rank | 32 | 64 |
| Tensors | 1680 | 1680 |
| Toolchain | ai-toolkit 0.7.22 | — |
| Verdict | **ACCEPT** | **ACCEPT** |

Neither is a FLUX.1 adapter — the common failure mode when grabbing a
pixel-art LoRA by name.

### 8.4 Smoke test results (same prompt, same seed 42, 768×768)

| Run | Steps | Guidance | Wall time | Output |
|---|---|---|---|---|
| Base (no LoRA) | 20 | 4.0 | **572 s** | 806 KB |
| + Redmond LoRA (scale 1.0) | 20 | 4.0 | **589 s** | 515 KB |
| + Prithiv LoRA (scale 1.0) | 20 | 4.0 | **607 s** | 669 KB |

All three applied cleanly: **1680/1680 keys matched, 840 layers**.

**Timing reality:** ~9.5–10 minutes per image at 20 steps on an M1 Pro. With
8 scenes, that is ~80 minutes of image generation alone. For comparison,
FLUX.2 Klein at 8 steps takes ~5.5 minutes per image. **Step-count tuning is
therefore the single highest-leverage performance decision in the migration** —
which is exactly why `sampling-rationale.json` defers it to a dedicated phase
rather than guessing here.

### 8.5 The `.redmond_path` / `.prithiv_path` convention

The benchmark harness stores the selected LoRA path in a dotfile
(`.redmond_path`, `.prithiv_path`) rather than hardcoding it in `gen.sh`. This
makes the LoRA under test a **runtime input** — the same script benchmarks any
adapter without editing, and the provenance sidecar records which one produced
each image.

### 8.6 What integrating Qwen means for the pipeline

The code is already structured for this swap. To move from FLUX.2 Klein to
Qwen-Image 2512:

1. `tools/model_setup.py` discovers MLX-Gen checkpoints by structural scan
   (`scan_mlxgen_models`) — the Qwen snapshot in the HF cache is already
   discoverable
2. the active generation profile's `model.match` names the checkpoint, resolved
   per-image through the registry
3. `lora.path` / `lora.scale` in the same profile attach the chosen adapter —
   **and only now does the LoRA actually do something**, because
   `_strip_lora_triggers()` stops stripping when `_MLXGEN_PROVIDER.lora_paths`
   is non-empty
4. Negative prompts become available again, so `NEGATIVE_PROMPT` from
   `config/image_style.json` starts being sent instead of dropped

No pipeline code change is required — the provider abstraction
(`MLXGenImageProvider`) handles both.

---

## 9. Cloud APIs: primary vs fallback

| API | Where | Primary? | Role |
|---|---|---|---|
| **fal.ai** | `pixel_art_tool.py` | ❌ | Image fallback on CUDA systems. `fal-ai/flux/dev`, then `fal-ai/flux/schnell`. Also used for LoRA *training*. Currently unreachable in the macOS path (`USE_LOCAL_FLUX=false`, no `FAL_KEY`) |
| **ElevenLabs** | `tts_tool.py` | ❌ | TTS tier 2. `eleven_multilingual_v2`, stability 0.35 / similarity 0.70 / style 0.65. Empty key → skipped |
| **OpenAI `gpt-5-mini`** | `llm_interface.py` | ⚠️ | Primary **only** for `multi_news_synthesizer`. Falls back to local Qwen |
| **ZhipuAI `glm-4v-flash`** | `image_curator.py` | ❌ | Vision fallback when Ollama vision is unreachable |
| **Pexels** | `pexels_tool.py` | ❌ | Stock footage. **Dead path** — `fetch_vertical_footage` is imported but never called |
| **Telegram** | `telegram_sender.py` | ✅ | Status + video delivery |
| **YouTube Data API v3** | `publish_video.py` | ✅ | Shorts upload |
| **TikTok Content Posting API** | `publish_video.py` | ✅ | Direct Post |

**The fail-closed principle.** The pipeline refuses to silently substitute a
cloud model for a locally managed one:

- `TextProvider` raises `ProviderError` rather than switching backends
- The MLX-Gen path returns an explicit failure — it does **not** fall back to
  fal.ai (`pixel_art_tool.py:2204`)
- `_is_retryable()` in the publisher treats missing credentials as
  non-retryable — no 90 s of pointless backoff

Rationale: an explicit failure is diagnosable. A silent model substitution
produces a video that looks subtly wrong and costs money per run.

---

## 10. Reliability machinery

### 10.1 Pipeline lock (cross-process)

`src/models/runtime.py::PipelineLock` — an `O_CREAT|O_EXCL` pidfile at
`/tmp/yt-machine-pipeline.lock` holding `{pid, job_id, started_at, host}`.

- **Stale takeover:** lock is stale if older than 6 h **or** the pid is dead
- **Release:** only unlinks when the recorded pid matches the current process
- **Pre-checked** by `run_daily.sh` and `automate.py` before spawning, so a
  scheduled run backs off cleanly instead of fighting for 20 GB of memory

**Critical detail:** the shell guard delegates to `PipelineLock` rather than
testing `[ -f "$LOCK" ]`. A bare existence test would treat a stale lock from a
crashed run as permanent, wedging the daily job forever.

### 10.2 Checkpoints

`checkpoint.json` per project accumulates `completed_steps`, `last_step`,
`last_updated`, plus step-specific data.

> **Honest caveat:** `--resume <folder>` reuses the project folder and project
> id and re-reads the previous metadata, but **does not skip completed steps**.
> The checkpoint is written and logged; it is not yet consulted to skip work.
> Every step re-runs from scratch. This is a real gap between the feature's
> name and its behaviour.

### 10.3 Timeout matrix

| Step | Env | Default | Retries |
|---|---|---|---|
| `news_analysis` | `LLM_TIMEOUT_ANALYSIS` | 420 s | LangChain → raw |
| `script_synthesis` | `LLM_TIMEOUT_SYNTHESIS` | 1800 s | 3 outer; JSON ×2, ×4 tokens |
| `script_fixer` | `LLM_TIMEOUT_FIXER` | 600 s | none (keep original) |
| `visual_prompts` | `LLM_TIMEOUT_VISUALS` | 900 s | none (fallback prompts) |
| `script_curation` | `LLM_TIMEOUT_CURATION` | 900 s | LangChain → raw |
| `script_evaluation` | `LLM_TIMEOUT_EVALUATION` | 600 s | none |
| `pixel_art` | — | — | 4 attempts + scrub levels 1–3 |
| TTS | — | — | 4 engines |
| assembly | `max(300, duration × 5)` | — | 3 paths |

**Whole-run ceiling:** `PIPELINE_TIMEOUT` — **14400 s (240 min)** in both
`src/automate.py` and `src/server.py`. The value has been raised twice: 900 s
to 7200 s, then to 14400 s, after the earlier ceilings were observed killing
real runs. A test asserts the two modules declare the same default.

The 78 minutes measured in §1 is a happy path; runs are dominated by image
generation and LLM latency, both of which vary with machine and model. Size
this above your own expected worst case.

Per-task timeouts are also declared in `config/system_prompts.json` →
`model_config.call_timeouts` (`dedup_comparison`, `news_processor`,
`script_synthesizer`, `script_curator`, `visual_prompt_generator`,
`script_evaluation`), each with a `hard` and an `idle` value. These were raised
substantially for the 27B local model — e.g. `script_synthesizer.hard` went
from 300 s to **1500 s** — because token generation on a 27B Q4 model is far
slower than the 4B models the original values targeted. The JSON `note` field
records this rationale, and `default_model: "local-profile"` with
`base_url: http://127.0.0.1:8080` confirms the file is now used only for
prompt tuning: model selection lives in `config/model_profile.json`.

**Note on thread timeouts:** `_run_with_timeout` abandons the worker thread on
timeout because Python cannot kill threads. The *real* timeout enforcement is
at the provider level (`hard_timeout` / `idle_timeout` in `providers.py`).

### 10.4 Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Scraper failure · VRAM too low · model health check failed · `<2` analyses · synthesis failed · missing voice/images |
| 2 | No model profile (`ModelProfile.load()` raised) |
| 3 | Pipeline lock held by another process |
| 4 | MLX-Gen image model unavailable |
| 5 | Could not enter image generation phase |
| 6 | Image generation failed under a managed local provider |
| 128 + signum | SIGINT/SIGTERM (handler runs `runtime.finish()` then `os._exit`) |

### 10.5 Publish idempotency

`output/publish_logs/.published.json` maps a video fingerprint
(`path|size|mtime`) to per-platform results. A platform that already succeeded
is skipped with status `already_published`. Only real successes are recorded —
failures must be retried on the next run. `--force` overrides. `dry_run` never
writes the ledger.

This exists because a manual run followed by the scheduled run would otherwise
post the same video twice.

---

## 11. Scheduling and automation

### 11.1 Daily timeline (macOS)

```
05:50  pmset wakes the Mac              sudo pmset repeat wakeorpoweron MTWRFSU 05:50:00
06:00  launchd fires the agent          StartCalendarInterval {Hour: 6, Minute: 0}
06:00  caffeinate -s holds the Mac awake
06:00  PipelineLock pre-check → pipeline starts
06:00  llama-server loads Qwen            (~1–2 min)
06:05  steps 1–4.8 run                    (~10–15 min)
06:20  memory barrier → image phase
06:20  step 5: 8 images                   (~45 min at ~5.5 min each)
07:05  step 7 TTS                         (~2 min)
07:07  step 8 assembly                    (~2 min)
07:10  step 9–10 metadata, manifest
07:12  publish to YouTube (+ TikTok)
07:15  Telegram summary
07:15  run ends → caffeinate releases
07:35  AC idle timer (20 min) elapses → Mac sleeps
```

**Measured total: ~78 minutes** on 2026-09-15.

### 11.2 Components

| Component | Path | Purpose |
|---|---|---|
| launchd agent | `~/Library/LaunchAgents/com.rafa9labs.ytmachine.plist` | Daily run at `RUN_TIME` |
| Wrapper | `tools/run_daily.sh` | PATH, venv, caffeinate, stale-lock check, publish flags |
| Master script | `src/automate.py` | Wake → generate → notify → publish |
| Wake schedule | `pmset` (root) | Hardware wake at `WAKE_TIME` |

**Why launchd, not cron:** `StartCalendarInterval` fires a missed job when the
machine wakes; cron silently skips anything scheduled during sleep and has no
wake primitive. `pmset` supplies the wake; launchd supplies the job.

**Why the wrapper sets PATH explicitly:** launchd children get
`/usr/bin:/bin:/usr/sbin:/sbin` — no `/opt/homebrew/bin`. Without the explicit
export, `ffmpeg`, `llama-server`, `mlx_lm.server`, and `ollama` are all
"command not found".

**Why caffeinate:** an idle sleep mid-pipeline would suspend an 80-minute
generation and could leave `llama-server` holding unified memory across the
sleep boundary.

### 11.3 Management commands

```bash
python src/automate.py --install-schedule 06:00   # install/replace launchd agent
python src/automate.py --install-wake             # print the sudo pmset command
python src/automate.py --configure-power 20       # print the sudo pmset sleep commands
python src/automate.py --show-schedule            # launchd state + pmset wake
python src/automate.py --show-power               # sleep timers
python src/automate.py --remove-schedule          # unload + delete agent
launchctl kickstart gui/$(id -u)/com.rafa9labs.ytmachine   # run now
```

### 11.4 Storage retention

`output/projects/` accumulates one folder per run (~380–500 MB: a lossless
master plus its delivery copy), which is roughly **140–180 GB/year** at one run
per day. `output/images/` accumulates per-image scratch. Neither was reclaimed
before ADR-040.

**Retention is report-only by default.** Every automation run prints what could
be reclaimed and deletes nothing:

```
retention[report]: 3 project(s), 12 scratch file(s) would reclaim 412.3MB (4 protected)
Output: 1.2GB across 9 project(s)
```

Deletion requires an explicit action:

```bash
# Preview — exact plan, nothing deleted
python src/automate.py --cleanup-dry-run
python src/automate.py --cleanup-dry-run delivery_only

# Apply
python src/automate.py --cleanup                    # full: remove whole projects
python src/automate.py --cleanup delivery_only      # masters only
```

| Mode | Effect | Reclaims |
|---|---|---|
| `report` | plan only (default) | nothing |
| `full` | remove whole old projects | all bytes of that project |
| `delivery_only` | remove only the lossless `.master.mp4` | ~93 % of the project, keeping the publishable delivery copy, manifest, scripts, images and provenance |
| `off` | disable retention entirely | nothing |

**Protections** — these hold at every setting, including `keep_last=0` and a
zero-day window:

- the **newest publishable project** is never removed; publishing discovery
  resolves it and would otherwise fail
- the most recent `YT_RETENTION_KEEP_LAST` projects (default 3)
- every project inside `YT_RETENTION_DAYS` (default 30)
- recent scratch images, so `tools/collect_best_images.py` keeps working as a
  LoRA curation source

`delivery_only` is never a default: the master is the archival record that
preserves the exact palette (ADR-038). Discarding it is an explicit choice.

---

## 12. Configuration reference

### 12.1 Files

| File | Contents | Gitignored |
|---|---|---|
| `config/model_profile.json` | Active model per role + memory preferences. Written by `tools/model_setup.py` | ✅ (machine-specific) |
| `config/system_prompts.json` | All LLM system prompts + per-task timeouts | ❌ |
| `config/image_style.json` | Style suffix, CLIP tag, negative prompt, brand colours, LoRA map, layout constants | ❌ |
| `config/rss_feeds.json` | 19 feeds + settings | ❌ |
| `.env` | API keys, scheduling, toggles | ✅ |
| `credentials/youtube_client_secrets.json` | Google OAuth client (you provide) | ✅ |
| `credentials/youtube_token.json` | Cached OAuth token (auto-created) | ✅ |
| `credentials/tiktok_token.json` | Access + refresh token (auto-managed) | ✅ |

### 12.2 Key `.env` variables

```bash
# Model serving
OLLAMA_HOST=http://localhost:11434
USE_LOCAL_FLUX=auto               # auto (default) enables when a CUDA GPU is present.
                                  # true/1/yes force it; false/0/no disable it.
TARGET_APP=Default                # resolution preset from image_style.json size_map

# Image backend executable. Machine-specific, so it stays in .env; an unset
# value fails the availability check rather than defaulting to a path.
MLXGEN_BIN=/usr/local/bin/mlxgen
MLXGEN_TIMEOUT=900

# Image model + sampling live in the generation profile, not in .env:
# change config/generation_profiles.json or run tools/configure.py.
#   model            profile.model.match
#   steps / guidance profile.steps / profile.guidance
#   LoRA path/scale  profile.lora.path / profile.lora.scale
#   provenance       profile.provenance (boolean, default true)

# Publishing
YOUTUBE_PRIVACY=public            # private | unlisted | public
TIKTOK_TOKEN_FILE=credentials/tiktok_token.json

# Scheduling
RUN_TIME=06:00
WAKE_TIME=05:50:00
PIPELINE_TIMEOUT=14400            # MUST exceed the expected run time

# Pipeline
LOG_FORMAT=console                # or json
LOG_LEVEL=INFO
```

`PUBLISH_PLATFORMS` (default `youtube,tiktok`) is read by `run_daily.sh` — drop
`tiktok` until the Content Posting API app is approved.

### 12.3 Model selection

```bash
python tools/model_setup.py           # interactive
python tools/model_setup.py --show    # print current profile
python tools/model_setup.py --auto    # best pick, no prompts
```

Discovery is read-only — it probes Ollama, `llama-server`, GGUF files, and
MLX-Gen folders without loading anything. Ranking prefers a dedicated server
(llamacpp) over Ollama-file scanning, then the largest model that fits
`total_memory − 4 GB`. The 4 GB reserve means a 49 GB checkpoint can never be
auto-selected.

---

## 13. Known gaps and honest caveats

Documented because a reference that only lists strengths is not usable for
debugging.

| # | Gap | Impact |
|---|---|---|
| 1 | **`--resume` does not resume.** Checkpoint is written and logged but never consulted to skip steps | Every resumed run re-runs everything from news fetch |
| 2 | **`trending_context` output is unused** downstream | Wasted computation, no behaviour change. Verified as expected: no action |
| 3 | **`llm_interface` debate methods have no caller.** `debate_skeptic` / `debate_explainer` and their prompt entries remain, but the chain modules that used them (`chains/debate.py`, `chains/news_analysis.py`, `collector/debate_engine.py`) were removed as dead code | Unused methods and two prompt entries; not on any pipeline path (ADR-002, ADR-037) |
| 4 | **`fetch_vertical_footage` is imported but never called** (step 6 removed) | Dead import; Pexels path untested |
| 5 | **`build_timeline` has no step banner**; visual-prompts banner shares the counter without its own step name | Cosmetic log inconsistency |
| 6 | **`PREROLL_OFFSET` comment says ~1 s; constant is 0.3 s** | Minor |
| 7 | **`_advance_phase` is in-memory only** — not persisted, not consulted on resume | Phase regression guards do not survive a restart |
| 8 | **Vector memory effectively disabled**: `nomic-embed-text` is not pulled and `roles.embedding` is `null` | Topic dedup across runs does not happen |
| 9 | **Postgres not running.** All `_save_to_postgres` calls log warnings | Progress tracking unavailable; JSON files remain authoritative |
| 10 | **Qwen-Image at 20 steps costs ~10 min/image** vs FLUX.2 Klein's ~5.5 min at 8 steps | 8 scenes would take ~80 min of image time alone — step tuning is mandatory before switching |
| 11 | **`lora_status: mapped-unvalidated`** for Qwen | Adapter compatibility was validated manually and recorded; mlxgen's own gate skips absolute local paths |
| 12 | **Telegram not configured** (empty token/chat id) — all notifications silently skipped | No run status until credentials are added |
| 13 | **Retired prompt fields are still requested by the analysis prompt** — resolved for `shift_vector`, `pixel_art_prompts`, `ticker_headlines`; the prompt now asks only for the five fields `NewsAnalysis` carries | Resolved in this revision (see below) |
| 14 | **Storage grows ~140-180 GB/year** with no automatic reclamation. Retention exists (`src/video/retention.py`, ADR-040) but defaults to **report-only** | Disk fills over ~2 years at one run/day unless `--cleanup` is run deliberately. Run `python src/automate.py --cleanup-dry-run` to see what is reclaimable |
| 15 | **Unreachable methods remain in `llm_interface`** — `extract_visual_elements` (39 lines), `warmup_model` (16), `_get_time_greeting` (10). All have zero callers and no removed-module dependencies | No runtime impact. Left out of the dead-code removal PR to keep it scoped; safe to delete in a later cleanup. Verified by AST reference scan |

### Word budget (resolved in this revision)

Gap 13 previously read "the synthesizer's word-count enforcement does not match
real output". The root cause was **three mutually inconsistent budgets**:

| Budget | Value | Source before the fix |
|---|---|---|
| Enforced | 130–170 | `llm_interface.MIN_WORDS` / `MAX_WORDS` |
| Prompt asked for | 150–170 | `system_prompts.json` |
| Per-segment limits | 112–197 | `SEGMENT_LIMITS`, advisory only |
| **Real output** | **296–325** | measured |

Two further defects compounded it: the counter excluded the closing while
`full_text` included it (305 counted vs 325 narrated), and the compression
instruction was **unsatisfiable** — it locked `fallout`/`segue`/`real_talk`
(106 words on the reference run) while demanding a total that those locked
fields alone exceeded, so no retry could ever succeed.

All budgets now derive from one place, `src/video/pipeline_config.py`:

```python
WORDS_PER_SECOND = 2.5            # the rate the pipeline already assumed
TARGET_VIDEO_SECONDS = (60, 70)
MIN_WORDS, MAX_WORDS = 150, 175
BEAT_WORD_RANGES = {...}          # per-beat ranges, sum proven compatible
```

`beat_budget_bounds()` asserts at import that the per-beat ranges can reach the
global band, so an inconsistency fails the first test run instead of producing
over-long scripts silently. `count_narrated_words()` is the single counter, used
by the enforcement, the `word_count` the manifest records, and the duration
estimate. The compression instruction now permits trimming every field.

### Retired prompt fields (resolved in this revision)

The analysis prompt requested three fields that `NewsAnalysis` does not carry,
so LangChain's parser discarded them on every run:

- `shift_vector` — its only reader was `historical_analyzer.py`, which is
  unreachable and whose fallback accepted the argument without using it
- `pixel_art_prompts` — superseded by the dedicated `visual_prompt_generator`
  step (ADR-022)
- `ticker_headlines` — no consumer since the ticker overlay was removed

The prompt, the manifest, and the dead analyzer's signature no longer reference
them. The prompt now states that every requested key must exist and extra keys
must not be added.

### Dead code removed (resolution of the earlier gap 15)

Eight collector modules (~3,200 lines) were deleted, along with the code they
kept alive. They were abandoned by a single commit, `8f44941` (2026-04-08,
"prevent curation hallucination + visual prompt alignment"), which replaced the
rule-based prompt pipeline with the LLM-based `generate_visual_prompts()` that
is still in use (ADR-022). The diff removed the imports and added the
replacement; the modules were left behind.

| Removed | Lines | Its job now lives in |
|---|---:|---|
| `script_parser.py` | 1,075 | `llm_interface.segment_timeline` |
| `prompt_generator.py` | 606 | `llm_interface.generate_visual_prompts()` |
| `prompt_validator.py` | 494 | `geopolitical_validator` (live) + `_score_prompt_specificity` |
| `visual_extractor.py` | 298 | `military_equipment_db` via `geopolitical_validator` (live) |
| `historical_equipment_db.py` | 264 | no successor — inert data |
| `action_mapping.py` | 191 | no successor |
| `historical_analyzer.py` | 190 | no successor (6-act era) |
| `salience_extractor.py` | 108 | no successor |
| `LLMInterface.synthesize_script` | 158 | `synthesize_multi_news_script` |
| `debate_skeptic` / `debate_explainer` | 42 | abandoned per ADR-002 |
| 4 orphaned config prompts | 28 | — |

**Safety, verified not assumed.** An AST walk of the import graph from all six
real entrypoints (`generate_complete_video.py`, `automate.py`, `server.py`,
`pipeline_api.py`, `rerun_video.py`, `publish_video.py`) reached none of them.
Removing the roots orphaned exactly zero further modules:
`military_equipment_db` and `geopolitical_accuracy` stay live via
`geopolitical_validator`, which `pixel_art_tool` imports.

**What survived deliberately.** `script_synthesizer` remains in
`model_config.call_timeouts` — that key is live for per-task timeout routing
(`task_name` indexes `call_timeouts`, never `prompts`). Only its `prompts`
entry was removed.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Beat** | One of four story parts: hook, mechanism, truth, fallout |
| **CLIP** | Contrastive Language-Image Pretraining — used here to score image↔prompt relevance |
| **CRF** | Constant Rate Factor — x264 quality knob (lower = better/larger) |
| **GGUF** | GPT-Generated Unified Format — llama.cpp's quantized model format |
| **Ken Burns** | Slow zoom/pan applied to a still image to imply motion |
| **Laplacian variance** | Sharpness metric — low variance means blurry or flat |
| **LoRA** | Low-Rank Adaptation — a small adapter that steers a base model's style |
| **MLX** | Apple's array framework for Apple Silicon GPU |
| **mlxgen / mflux** | CLI for running FLUX-class image models natively on MLX |
| **Quantization (Q4_K_M / 8-bit)** | Weight compression; trades small quality loss for large memory savings |
| **Reserve (4 GB)** | Headroom the runtime refuses to allocate, keeping macOS responsive |
| **Segue** | The narrative bridge between story 1 and story 2 |
| **Shift vector** | The `NewsAnalysis` field describing how a story changes the geopolitical picture |
| **Step (diffusion)** | One denoising iteration; more steps = slower, diminishing quality returns |
| **Trademark closing** | "good morning, good afternoon, and goodnight" — the channel's sign-off |
| **Unified memory** | Apple Silicon's shared CPU/GPU memory pool |

---

*Generated from direct code inspection of `main` @ `9f13110` plus the
uncommitted macOS-automation and model-provider work, and the Qwen-Image
benchmark artifacts in `~/AI/FluxSprites/benchmarks/qwen-selection/`.*
