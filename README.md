# YT Machine

An automated pipeline that turns global news feeds into short-form vertical
videos: it collects articles, writes a two-story script, generates pixel-art
visuals and a voiceover, composites a 1080×1920 video with `ffmpeg`, and can
publish to YouTube Shorts and TikTok on a daily schedule.

Runs locally on a single machine. The default configuration targets Apple
Silicon with a local LLM and Qwen-Image-2512 through MLX-Gen, so the pipeline
produces video without a hosted image-generation API. Optional cloud backends
(ElevenLabs and OpenAI) are used only when their API keys are present and are
never required for the primary path.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](requirements-macos.txt)

---

## Overview

Producing a news video by hand is a chain of separate tasks: reading sources,
picking a story, writing to a length, generating visuals, recording narration,
editing, and publishing. YT Machine automates that chain end to end as a staged
pipeline with a checkpointed, inspectable artifact at every step.

**The problem it addresses.** Short-form news production is repetitive and
latency-sensitive — the value of a story decays quickly. The pipeline is built
to run unattended: it wakes the machine, produces the video, publishes it, and
shuts down again.

**What makes it non-trivial.** The pipeline has to run a large language model
and a diffusion image model on a single machine. On Apple Silicon both compete
for one unified memory pool, so the system enforces a strict phase ordering and
verifies that a model's memory has actually been reclaimed before loading the
next one. That constraint shapes most of the architecture.

**Who it is for.** Developers interested in local-first generative media
pipelines, and anyone who wants a modifiable base for automated video
production rather than a hosted service.

## Pipeline

```mermaid
flowchart LR
    A[News feeds<br/>19 RSS sources] --> B[Collect<br/>fetch + extract articles]
    B --> C[Research<br/>score topics, analyze]
    C --> D[Script<br/>synthesize + enforce]
    D --> E[Visuals<br/>8 pixel-art scenes]
    E --> F[Voice<br/>TTS + mastering]
    F --> G[Assembly<br/>ffmpeg composite]
    G --> H[Video<br/>1080x1920 MP4]
    H --> I[Publish<br/>YouTube + TikTok]
```

Each stage writes its output to the project folder before the next one starts,
so a failed run leaves a readable artifact instead of a partial state. The
script stages are followed by deterministic validation steps that enforce
structural guarantees the language model cannot be relied on to produce.

## Features

- **Multi-source news collection** — 19 RSS feeds (BBC, Reuters, AP, Al Jazeera,
  Foreign Policy, SCMP, and others) fetched concurrently, with topic scoring and
  an 8-day category rotation.
- **Two-tier article extraction** — `trafilatura` for static pages, with a
  Playwright fallback for JavaScript-rendered sites.
- **Structured script generation** — two stories with four labelled beats each
  (hook, mechanism, real talk, fallout), produced by a local LLM through
  LangChain chains with Pydantic-validated output.
- **Deterministic script enforcement** — a Python-only pass that guarantees
  story count, segue placement, de-duplication, and the closing, regardless of
  what the model returned.
- **Local pixel-art generation** — eight Qwen-Image-2512 scenes per video through
  an MLX-Gen subprocess, with capability probing, deterministic 192×192/32-color
  post-processing, provenance sidecars, and a fail-closed policy.
- **Multi-engine TTS with fallbacks** — Kokoro (local) → ElevenLabs → Edge TTS
  → silent track, plus an `ffmpeg` mastering chain.
- **FFmpeg video composition** — 60/40 split-screen layout with static
  grid-preserving pixel scenes, karaoke subtitles in ASS format, avatar loop,
  and a music bed.
- **Memory-safe model lifecycle** — one heavy model resident at a time, with
  verified memory reclamation between phases and an inter-process lock.
- **Unattended daily automation on macOS** — `launchd` scheduling, `pmset` wake,
  `caffeinate` during the run, and idle sleep afterwards.
- **Publishing with idempotency** — YouTube Shorts via OAuth2 and TikTok via the
  Content Posting API, with a publish ledger that prevents double-posting.

## Quick Start

**Prerequisites**

- Python 3.12
- `ffmpeg` on `PATH`
- For the default local-model path: Apple Silicon macOS with a compatible
  text model and MLX-Gen image model on disk
- Optional: Docker, only if you want PostgreSQL and n8n

**Install**

```bash
git clone https://github.com/rafa9-labs/yt-machine.git
cd yt-machine

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-macos.txt
```

`requirements.txt` is the alternative set for the CUDA/WSL path.

**Configure**

```bash
cp config/.env.example .env
```

All keys are optional for a local-only run. Set them when you need the
corresponding capability — see [Configuration](#configuration).

Discover the models on your machine and select one per role:

```bash
python tools/model_setup.py --show    # list what was detected
python tools/model_setup.py           # interactive role selection
```

This writes `config/model_profile.json`, which the pipeline requires.

**Verify the wiring without loading any model**

```bash
python tools/generate_complete_video.py --dry-run
```

Dry-run runs the full pipeline against canned data with no model calls and no
network access. It writes `output/projects/video_<timestamp>/script.txt` and
exits. Use this to confirm the installation before committing to a real run.

**Generate a real video**

```bash
python tools/generate_complete_video.py --no-telegram
```

Output lands in `output/projects/video_<timestamp>/`:

```
video_<timestamp>.mp4   # 1080x1920 H.264, typically 100-130s
voiceover.mp3
script.txt
script_segments.json
platform_metadata.json
manifest.json
images/
checkpoint.json
```

A full run takes roughly 80 minutes on an M1 Pro: most of that is eight image
generations at about 5–6 minutes each.

## Configuration

Environment variables live in `.env`; `config/.env.example` documents all of
them. The settings that most affect behaviour:

| Variable | Purpose |
|---|---|
| `OLLAMA_HOST` | Endpoint for the Ollama-compatible text server |
| `YT_GENERATION_PROFILE` | Active validated image-generation profile; defaults to `qwen_pixel_scene` |
| `YT_GENERATION_PROFILES_PATH` | Optional override for the generation-profile JSON |
| `MLXGEN_BIN`, `MLXGEN_TIMEOUT` | MLX-Gen executable and per-image timeout |
| `PIPELINE_TIMEOUT` | Hard ceiling for one run in seconds |
| `YOUTUBE_PRIVACY` | `private`, `unlisted`, or `public` upload visibility |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Enables run notifications and delivery |
| `ELEVEN_LABS_KEY` | Optional TTS fallback |
| `POSTGRES_*` | Optional progress persistence; the pipeline runs without it |

Non-secret configuration is version-controlled:

| File | Contents |
|---|---|
| `config/model_profile.json` | Active model per role (generated; not committed) |
| `config/generation_profiles.json` | Qwen sampling, LoRA, seed, post-processing, and zoom policy |
| `config/system_prompts.json` | All LLM system prompts and per-task timeouts |
| `config/image_style.json` | Style suffix, negative prompt, palette, layout, LoRA map |
| `config/rss_feeds.json` | Feed list and collection settings |

## Architecture

```mermaid
flowchart TB
    subgraph entry["Entry points"]
        CLI["generate_complete_video.py"]
        AUTO["automate.py"]
        API["server.py"]
    end

    subgraph core["Orchestration"]
        PIPE["Pipeline runner"]
        RT["ModelRuntime<br/>+ PipelineLock"]
    end

    subgraph modules["Pipeline modules"]
        COLL["Collector"]
        BRAIN["Brain"]
        VIDEO["Video"]
    end

    subgraph providers["Model providers"]
        REG["Registry + Profile"]
        TEXT["Text adapters"]
        IMG["MLXGen adapter"]
    end

    subgraph external["External"]
        RSS["RSS feeds"]
        LLM["Local LLM server"]
        MIX["MLX image model"]
        PUB["YouTube / TikTok"]
        TG["Telegram"]
    end

    subgraph store["Storage"]
        OUT["output/projects/"]
        PG["PostgreSQL"]
    end

    CLI --> PIPE
    AUTO --> PIPE
    API --> PIPE
    PIPE --> RT
    PIPE --> COLL
    PIPE --> BRAIN
    PIPE --> VIDEO
    REG --> TEXT
    REG --> IMG
    COLL --> RSS
    BRAIN --> TEXT
    VIDEO --> IMG
    TEXT --> LLM
    IMG --> MIX
    PIPE --> OUT
    PIPE -.-> PG
    AUTO --> PUB
    AUTO --> TG
```

| Component | Responsibility |
|---|---|
| `tools/generate_complete_video.py` | Pipeline runner; owns step order, checkpoints, and phase transitions |
| `src/automate.py` | Scheduling, wake handling, publish orchestration, notifications |
| `src/collector/` | RSS fetching, article extraction, topic scoring, category rotation, platform metadata |
| `src/brain/` | LLM interfaces, LangChain chains, prompt curation, script evaluation, memory |
| `src/video/` | Pixel-art generation, visual QA, TTS, subtitles, video assembly |
| `src/models/` | Model discovery, profile persistence, provider adapters, runtime lifecycle |
| `src/db/` | PostgreSQL schema and persistence helpers (optional) |
| `src/server.py` | FastAPI surface for triggering runs and checking status |
| `src/publish_video.py` | YouTube and TikTok upload with a publish ledger |

The dotted edge marks a genuinely optional dependency: PostgreSQL is written to
on a best-effort basis and the pipeline completes without it.

## Runtime Flow

```mermaid
sequenceDiagram
    autonumber
    participant user as User / launchd
    participant wrapper as run_daily.sh
    participant auto as automate.py
    participant pipe as Pipeline
    participant runtime as ModelRuntime
    participant text as Text model
    participant image as Image model
    participant fs as output/projects/

    user->>wrapper: scheduled 06:00
    wrapper->>wrapper: check PipelineLock
    wrapper->>auto: --publish youtube,tiktok
    auto->>pipe: subprocess
    pipe->>runtime: acquire lock
    pipe->>runtime: start text phase
    runtime->>text: launch llama-server
    text-->>runtime: ready
    loop script stages
        pipe->>text: analyze / synthesize / curate
        text-->>pipe: structured script
    end
    pipe->>runtime: stop text, verify memory
    pipe->>runtime: start image phase
    loop 8 scenes
        pipe->>image: generate pixel art
        image-->>pipe: PNG
    end
    pipe->>pipe: TTS + ffmpeg assembly
    pipe->>fs: write video + manifest
    pipe->>runtime: finish, release lock
    auto->>auto: publish to platforms
    auto->>user: Telegram summary
```

The text model is stopped and its memory reclamation verified before the image
model loads, because both cannot be resident at once. The image model then runs
as one subprocess per scene, so its memory returns to the OS deterministically
between generations. The lock is released on every exit path, including
failures and signals.

## Project Structure

```
src/
  automate.py     # scheduling, wake handling, publish orchestration, notifications
  brain/          # LLM interfaces, chains, script evaluation, memory
  collector/      # RSS fetching, extraction, scoring, metadata
  video/          # image generation, QA, TTS, subtitles, assembly
  models/         # provider registry, profile, runtime, memory guards
  db/             # optional PostgreSQL layer
  server.py       # FastAPI surface
  publish_video.py
tools/            # pipeline runner, model setup, auth helpers, daily wrapper
config/           # prompts, image style, feeds, env example
tests/            # pytest suite plus standalone validation scripts
infra/            # Dockerfile and docker-compose for API + Postgres + n8n
assets/           # avatar loop and music bed used during assembly
output/           # generated projects, logs, publish ledger (gitignored)
```

## Customization

**Swap or add a text model.** Model selection is data, not code. Add GGUF files
under a scan root (`YT_MODEL_ROOTS`, defaulting to `~/AI` and `~/models`) or
serve a model through Ollama, then run `tools/model_setup.py`. To add a new
backend protocol, implement `TextProvider` in `src/models/providers.py`; see
`OllamaTextProvider` and `OpenAITextProvider` for the two existing shapes.

```python
# src/models/providers.py
class TextProvider:
    def generate(self, prompt, model, system_prompt=None, ...) -> str: ...
    def health(self, timeout: float = 5.0) -> bool: ...
```

**Change what the script sounds like.** The voice and structure live in
`config/system_prompts.json`. The pipeline treats the model's output as
untrusted: `script_enforcement` re-establishes structure afterwards, so a prompt
change cannot break the format.

**Change the visual style.** `config/image_style.json` holds the style suffix,
negative prompt, brand palette, per-category LoRA settings, and the split-screen
layout. Attach a LoRA with `MLXGEN_LORA_PATH`; it is applied only if the file
already exists on disk, so a run never downloads model weights mid-flight.

**Change the news sources.** Edit `config/rss_feeds.json`. Each feed carries a
category used by the rotation system and the scoring pass.

**Change the output package.** `src/video/split_video_assembler.py` defines the
layout constants (`VIDEO_W`, `VIDEO_H`, `TOP_H`, `BOTTOM_H`, `FPS`) and the
`ffmpeg` filter chain. Video encoding parameters, scene motion, and the audio
mix are all in that module.

**Add a platform.** `src/publish_video.py` maps platform names to publisher
functions. A new publisher returns a result dict with `platform` and `status`,
which is all the ledger and the summary need.

## Testing

Tests use `pytest`. Run the suite:

```bash
.venv/bin/python -m pytest \
  tests/test_automation_macos.py \
  tests/test_dedup.py \
  tests/test_image_pipeline.py \
  tests/test_local_prompt_building.py \
  tests/test_model_registry.py \
  tests/test_pipeline_progress.py \
  tests/test_runtime_lifecycle.py \
  tests/test_visual_prompts.py \
  tests/test_vram_budget.py \
  tests/test_vram_orchestrator.py
```

That set covers 461 tests and runs in under a minute without loading a model or
touching the network. The main areas:

| Area | File |
|---|---|
| Scheduling, launchd, publish idempotency, retry classification | `tests/test_automation_macos.py` |
| Provider adapters, streaming timeouts, pipeline lock | `tests/test_model_registry.py` |
| Model phase transitions and memory verification | `tests/test_runtime_lifecycle.py` |
| Script de-duplication and enforcement | `tests/test_dedup.py`, `tests/test_visual_prompts.py` |
| Local prompt construction and QA thresholds | `tests/test_local_prompt_building.py`, `tests/test_image_pipeline.py` |
| Video/VRAM budget calculations | `tests/test_vram_budget.py`, `tests/test_vram_orchestrator.py` |

Some test files under `tests/` are standalone validation scripts from earlier
iterations rather than `pytest` modules; they import older module paths and are
not collected by the command above.

## Development

```bash
# Validate the pipeline wiring end to end without models or network
python tools/generate_complete_video.py --dry-run

# Inspect which models were discovered and selected
python tools/model_setup.py --show

# Start the local LLM stack (MLX server + Ollama-compatible bridge)
tools/start_llm.sh start
tools/start_llm.sh status

# Run the API surface locally
python -m src.server --port 8000        # docs at /docs

# Preview a publish without uploading
python src/publish_video.py --platform youtube --dry-run
```

Optional infrastructure:

```bash
docker compose -f infra/docker-compose.yml up -d
```

This starts the FastAPI service, PostgreSQL with `pgvector`, and n8n. The
pipeline does not require any of them.

## Contributing

Contributions are welcome. The repository is a single Python codebase with no
build step.

1. Fork the repository and create a branch from `main`.
2. Keep changes focused — one concern per branch.
3. Add or update tests under `tests/`. The suite must stay green.
4. Run the test command above before opening a pull request.
5. Describe what changed and why in the pull request.

Two conventions worth knowing before editing the pipeline:

- **Steps must not reorder across the memory barrier.** All text-model work
  happens before any image-model work. See
  [PIPELINE.md](PIPELINE.md#5-model-runtime-one-heavy-model-at-a-time).
- **Fail closed, not silently.** If a configured local model fails, the pipeline
  reports it and exits rather than substituting a different model. Preserve that
  behaviour when adding providers or fallbacks.

Detailed references live in:

- [PIPELINE.md](PIPELINE.md) — every stage, model, and constant, with the
  reasoning behind each choice.
- [architecture_decisions.md](architecture_decisions.md) — decision records
  including options that were tried and reversed.
- [SETUP.md](SETUP.md) — environment setup, including the macOS automation and
  publishing walkthroughs.

## Roadmap

Known gaps in the current implementation, in rough priority order:

- **Vector memory is stored but not read.** Topic embeddings are written at the
  end of a run, but cross-run similarity checks are disabled to avoid loading a
  second model alongside the text model.
- **Resume does not skip completed steps.** `--resume` reuses the project folder
  and re-reads the checkpoint, but every step re-runs.
- **Legacy test scripts need modernising.** The standalone validation scripts
  under `tests/` predate the `src/` layout and are not collected by `pytest`.
- **Qwen image acceptance is in progress.** The active path is Qwen-Image-2512;
  the final 15-image quality set and end-to-end MP4 palette check are still
  pending.
- **TikTok publishing requires app approval.** The integration is implemented
  and refreshes tokens, but posting only works once the Content Posting API
  access is granted.

## License

[MIT](LICENSE).
