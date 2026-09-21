# Geopolitical Sentinel — Setup Guide

## Prerequisites

This pipeline runs on two kinds of hosts:

**macOS (Apple Silicon) — current supported setup**

- macOS 14+ on an M-series Mac with 32GB+ unified memory
- Homebrew (`ffmpeg`, `llama-server`, `mlx_lm.server`, `ollama`)
- ~60GB free disk space (models + output)
- Docker optional (PostgreSQL only; the pipeline degrades gracefully without it)

**Windows 11 + WSL2 (legacy)**

- Windows 11 with WSL2 (Ubuntu 22.04+)
- NVIDIA GPU with 12GB+ VRAM (RTX 3090/4090 recommended)
- 32GB+ system RAM
- Docker Desktop (for PostgreSQL)
- ~50GB free disk space (models + output)

## 1. Clone and Install

### macOS

```bash
git clone https://github.com/rafa9-labs/yt-machine.git
cd yt-machine

# Install dependencies into the uv-managed virtualenv
uv pip install --python .venv/bin/python -r requirements-macos.txt

# Install Playwright browser (for async scraping)
.venv/bin/playwright install chromium
```

### Windows / WSL2

```bash
# In WSL
cd ~
git clone https://github.com/rafa9-labs/yt-machine.git
cd yt-machine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install google-auth-oauthlib google-api-python-client

# Install Playwright browser (for async scraping)
playwright install chromium

# Install spacy model (for NER entity extraction)
python -m spacy download en_core_web_sm
```

## 2. WSL2 Memory Configuration (legacy — Windows only)

Skip this section entirely on macOS. It applies only to the legacy
Windows/WSL2 deployment documented in section 10.

Edit `C:\Users\<you>\.wslconfig` on Windows:

```ini
[wsl2]
memory=24GB
swap=8GB
```

Restart WSL: `wsl --shutdown` in PowerShell, then reopen WSL.

## 3. Environment Variables

Copy the example and fill in your values:

```bash
cp .env.example .env
nano .env
```

`.env` is git-ignored — never commit real keys. Almost everything is optional
for a local-only run; the two you are most likely to need are `OLLAMA_HOST`
(required) and `MLXGEN_BIN` (your local image backend).

### Required variables

The pipeline is local-first: only the local LLM endpoint is genuinely required.

| Variable | Description | Example |
|---|---|---|
| `OLLAMA_HOST` | Local LLM endpoint (required) | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model used for script work | `qwen3:14b` |
| `MLXGEN_BIN` | Path to your local image backend executable | `/usr/local/bin/mlxgen` |
| `YT_MODEL_ROOTS` | Where your downloaded models live | `~/AI/models` |

### Optional cloud keys

Set these only if you want the corresponding cloud capability. Leaving them
blank keeps the run entirely local.

| Variable | Description | Example |
|---|---|---|
| `ELEVEN_LABS_KEY` | ElevenLabs API key for TTS fallback | `sk_...` |
| `FAL_KEY` | fal.ai key for cloud image generation fallback | `key-...` |
| `PEXELS_API_KEY` | Pexels key for stock-footage fallback | `...` |
| `ZHIPUAI_API_KEY` | ZhipuAI GLM vision key for `tools/image_curator.py` | `...` |
| `HF_TOKEN` | HuggingFace token for gated model downloads | `hf_...` |

### Database (optional — JSON on disk is authoritative)

| Variable | Description | Example |
|---|---|---|
| `POSTGRES_HOST` | PostgreSQL host | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_USER` | PostgreSQL user | `yt_machine` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `your_password` |
| `POSTGRES_DB` | PostgreSQL database | `yt_machine` |

### Publishing variables (YouTube + TikTok)

| Variable | Description |
|---|---|
| `YOUTUBE_CLIENT_SECRETS_FILE` | Path to OAuth2 credentials JSON (see YouTube setup below) |
| `YOUTUBE_CREDENTIALS_FILE` | Path where OAuth token will be cached (auto-created) |
| `YOUTUBE_PRIVACY` | Upload privacy: `private` \| `unlisted` \| `public` (default `public`) |
| `TIKTOK_CLIENT_KEY` | TikTok Developer app client key |
| `TIKTOK_CLIENT_SECRET` | TikTok Developer app client secret |
| `TIKTOK_ACCESS_TOKEN` | TikTok access token (expires in ~24h; prefer the token file) |
| `TIKTOK_REFRESH_TOKEN` | Refresh token for unattended daily posting |
| `TIKTOK_REDIRECT_URI` | Redirect URI registered in the TikTok app (for `--authorize`) |
| `TIKTOK_TOKEN_FILE` | Token cache path (default `credentials/tiktok_token.json`) |

### macOS automation variables

| Variable | Description | Default |
|---|---|---|
| `RUN_TIME` | Daily launchd run time | `06:00` |
| `WAKE_TIME` | Daily pmset wake time | `05:50:00` |
| `LAUNCHD_LABEL` | launchd agent label | `com.rafa9labs.ytmachine` |
| `IDLE_SLEEP_MIN` | AC idle-sleep minutes for `--configure-power` | `20` |
| `PUBLISH_PLATFORMS` | Platforms the daily wrapper publishes to | `youtube,tiktok` |

### Notification variables

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Your chat ID (run `python -m tools.telegram_sender --get-chat-id`) |

### Automation variables

| Variable | Description | Default |
|---|---|---|
| `WOL_MAC` | Target PC MAC address for Wake-on-LAN | — |
| `PIPELINE_TIMEOUT` | Pipeline timeout in seconds | `14400` |
| `IDLE_SLEEP_MIN` | Minutes idle before the Mac sleeps again | `20` |
| `RUN_TIME` / `WAKE_TIME` | Daily launchd run time / pmset wake time | `06:00` / `05:50:00` |

### Optional toggles

| Variable | Description | Default |
|---|---|---|
| `USE_KOKORO` | Kokoro local TTS (`auto` = use when installed) | `auto` |
| `USE_LOCAL_FLUX` | Use local FLUX/CUDA path (`auto` = when a GPU is present) | `auto` |
| `LOCAL_FLUX_MIN_VRAM_GB` | Minimum free VRAM to use local FLUX | `14` |
| `LOCAL_FLUX_EVICT_OLLAMA` | Evict Ollama from GPU before loading FLUX | `true` |
| `SKIP_IMAGES` | Use placeholder images instead of generating (fast test runs) | `0` |

See `.env.example` for the complete list.

## 4. Text model setup

There is no single hardcoded model to pull: the pipeline discovers the models
on your machine and you select one per role. That selection is written to
`config/model_profile.json`, which the pipeline requires.

```bash
# See what was discovered (GGUF, MLX, Ollama, llama.cpp)
.venv/bin/python tools/model_setup.py --show

# Interactive role selection
.venv/bin/python tools/model_setup.py

# Or let it pick the best available non-interactively
.venv/bin/python tools/model_setup.py --auto
```

Discovery scans `YT_MODEL_ROOTS` (see `.env.example`); with it unset it looks
in `~/AI`, `~/models` and the HuggingFace cache.

If you use Ollama as the serving backend:

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Embedding model for cross-run topic memory (optional — needs Postgres)
ollama pull nomic-embed-text

# Verify what Ollama is serving
ollama list
```

## 5. PostgreSQL Setup (optional)

Postgres is **optional**. JSON on disk is authoritative for the pipeline; the
database is only needed for cross-run vector memory and the n8n dashboard.
Without it the pipeline runs normally and logs that persistence is disabled
(ADR-030).

```bash
# Start the container (compose maps 5432:5432)
docker compose -f infra/docker-compose.yml up -d postgres

# Wait for it to be healthy
docker compose -f infra/docker-compose.yml ps

# The pipeline auto-creates tables on first run via init_db()
```

If port 5432 is already taken by another PostgreSQL instance, change the host
mapping in `infra/docker-compose.yml` (the `ports:` entry) and set
`POSTGRES_PORT` in `.env` to match.

## 6. YouTube OAuth Setup

YouTube publishing requires a one-time OAuth2 authorization. This opens a browser window for consent, then caches the token for unattended use.

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (e.g., "Geopolitical Sentinel")
3. Enable the **YouTube Data API v3**:
   - Navigate to APIs & Services → Library
   - Search for "YouTube Data API v3"
   - Click Enable

### Step 2: Create OAuth2 Credentials

1. Go to APIs & Services → Credentials
2. Click "Create Credentials" → "OAuth client ID"
3. Application type: **Desktop app**
4. Name: `yt-machine-publisher`
5. Download the JSON file
6. Save it as `credentials/youtube_client_secrets.json` in the project root

### Step 3: One-Time Authorization

```bash
cd yt-machine
source .venv/bin/activate

# Run the publisher in dry-run mode to trigger the OAuth flow
python src/publish_video.py --platform youtube --dry-run
```

A browser window will open asking you to authorize the app. After consent, the token is cached at `credentials/youtube_token.json` for all future unattended runs.

**Important**: You must do this step once while you have physical access to the machine. After that, the cached token refreshes automatically.

## 7. TikTok API Setup

### Step 1: Create TikTok Developer App

1. Go to [TikTok Developer Portal](https://developers.tiktok.com/)
2. Create a new app
3. Apply for **Content Posting API** access
4. Wait for approval (may take several days)

### Step 2: Get Access Token

1. In your app dashboard, generate a Content Posting API access token
2. Add these to your `.env`:

```env
TIKTOK_CLIENT_KEY=your_client_key
TIKTOK_CLIENT_SECRET=your_client_secret
TIKTOK_ACCESS_TOKEN=your_access_token
```

### Step 3: Verify

```bash
python src/publish_video.py --platform tiktok --dry-run
```

## 8. Manual Run

```bash
cd yt-machine
source .venv/bin/activate

# Generate video only (no publish)
python tools/generate_complete_video.py

# Generate and publish to YouTube + TikTok
python src/automate.py --publish youtube,tiktok

# Generate only (skip WOL, skip publish)
python src/automate.py --generate

# Dry-run publish (test credentials without uploading)
python src/publish_video.py --dry-run
```

Output is saved to `output/projects/<project_id>/` with the final MP4 video, manifest, and metadata.

## 9. Automation — macOS (launchd + pmset)

The pipeline runs automatically every day: the Mac wakes from sleep, generates
the video, publishes to YouTube + TikTok, and sleeps again after the idle timer.

**End-to-end timeline**

```
05:50  pmset wakes the Mac
06:00  launchd fires tools/run_daily.sh
06:00  caffeinate holds the Mac awake for the run
06:00  pipeline generates the video (~80 min on an M1 Pro)
07:20  publish to YouTube (+ TikTok once approved)
07:25  Telegram notification with the published URLs
07:45  run ends, caffeinate releases, idle timer counts down
08:05  Mac sleeps
```

### Step 1 — Install the dependencies

YouTube publishing needs two packages that are not in the base requirements:

```bash
uv pip install --python .venv/bin/python google-auth-oauthlib google-api-python-client
```

### Step 2 — One-time YouTube consent

`--dry-run` does **not** trigger OAuth (it returns before the credential flow),
so use the dedicated helper while you are sitting at the Mac:

```bash
.venv/bin/python tools/youtube_auth.py
```

This opens a browser, asks for the upload permission, and caches the token at
`credentials/youtube_token.json`. The token refreshes itself from then on, so
unattended runs work. Verify at any time with:

```bash
.venv/bin/python tools/youtube_auth.py --check
```

### Step 3 — TikTok (optional, needs app approval)

TikTok access tokens expire in ~24 hours, which breaks a daily unattended job
unless the token is refreshed. The publisher checks
`credentials/tiktok_token.json` first and refreshes automatically before each
upload. Authorize once (requires an approved Content Posting API app):

```bash
.venv/bin/python tools/tiktok_auth.py --authorize
# approve in the browser, copy the `code` query param, then:
.venv/bin/python tools/tiktok_auth.py --authorize --code <CODE>
.venv/bin/python tools/tiktok_auth.py --check
```

Until the app is approved, drop `tiktok` from `PUBLISH_PLATFORMS` in
`tools/run_daily.sh` (or set `PUBLISH_PLATFORMS=youtube` in the environment).

### Step 4 — Install the launchd agent

```bash
python src/automate.py --install-schedule 06:00
```

This writes `~/Library/LaunchAgents/com.rafa9labs.ytmachine.plist` and loads it.
`StartCalendarInterval` (not `StartInterval`) is what makes launchd fire the job
at wall-clock time rather than counting from load.

### Step 5 — Schedule the wake (needs sudo)

```bash
sudo pmset repeat wakeorpoweron MTWRFSU 05:50:00
```

Or let the script print the exact command for you:

```bash
python src/automate.py --install-wake          # prints the sudo line
```

`pmset` schedules live in the power controller and require root, so this one
step must be run manually in a terminal.

### Step 6 — Configure the sleep-after-run cycle

This machine ships with AC idle sleep set to **never** (`sleep 0`), so it would
stay awake all day after the wake. Set a short AC idle timer:

```bash
python src/automate.py --configure-power 20    # prints the sudo lines
# or run directly:
sudo pmset -c sleep 20
sudo pmset -c displaysleep 10
```

Battery settings are deliberately left untouched.

### Verifying the setup

```bash
python src/automate.py --show-schedule   # launchd state + pmset wake schedule
python src/automate.py --show-power      # sleep timers

# Fire the job immediately without waiting for 06:00:
launchctl kickstart gui/$(id -u)/com.rafa9labs.ytmachine
tail -f output/logs/launchd.out.log
```

A dry run of publishing (no network):

```bash
.venv/bin/python src/publish_video.py --platform youtube,tiktok --dry-run
```

### Removing the automation

```bash
python src/automate.py --remove-schedule
sudo pmset repeat cancel
```

### Troubleshooting (macOS)

| Problem | Cause | Fix |
|---|---|---|
| Job never fires | Agent not loaded | `python src/automate.py --show-schedule` then re-install |
| Job starts but exits 0 immediately | Pipeline lock held | Another run is active; check `/tmp/yt-machine-pipeline.lock` |
| `ffmpeg: command not found` | launchd has a minimal PATH | `run_daily.sh` exports Homebrew's bin; verify it was not edited |
| YouTube 401 / no token | Consent never completed | `.venv/bin/python tools/youtube_auth.py` |
| YouTube upload is private | `YOUTUBE_PRIVACY=private` | Set `YOUTUBE_PRIVACY=public` in `.env` |
| TikTok 401 after day 1 | Static token expired | Run `tools/tiktok_auth.py --authorize` (daily refresh is automatic) |
| TikTok rejects the post | Unaudited app | Unaudited apps can only post `SELF_ONLY`; complete app review |
| Mac does not wake | pmset schedule missing | `sudo pmset repeat wakeorpoweron MTWRFSU 05:50:00` |
| Mac stays awake all day | AC idle sleep is 0 | `python src/automate.py --configure-power 20` |

## 10. Automation — legacy Windows/WSL2 (removed)

The original deployment scheduled the pipeline with Windows Task Scheduler and
WSL2. That path is **no longer implemented**: `src/automate.py` targets macOS
only (launchd + pmset, section 9), and no Windows task scripts remain in the
repository. The migration is recorded in ADR-028.

The Windows/WSL2 install path in sections 1–2 still works for running the
pipeline manually, but unattended daily scheduling requires macOS or your own
scheduler. If you are porting this to Windows, the pieces to reproduce are:

- a scheduler entry that runs `tools/run_daily.sh` (or the equivalent
  `python src/automate.py --generate --publish`)
- a wake mechanism that starts the machine before the run
- an idle-sleep timer so the host returns to sleep afterwards

## 11. Troubleshooting Common Errors

| Error | Cause | Fix |
|---|---|---|
| `Model not found` / LLM 404 | No model selected, or the server is not serving it | `.venv/bin/python tools/model_setup.py --show`, then select one |
| `MLXGEN_BIN is not set` | Local image backend not configured | Set `MLXGEN_BIN` in `.env` (see `.env.example`) |
| `CLIP 77-token truncation warning` | FLUX CLIP encoder truncates long prompts | Expected and harmless; scene content is front-loaded |
| `CUDA out of memory` | VRAM too low | Set `USE_LOCAL_FLUX=false` in `.env` or increase `LOCAL_FLUX_MIN_VRAM_GB` |
| `psycopg2.OperationalError: connection refused` | PostgreSQL not running (it is optional) | Start it, or leave it stopped — the pipeline runs without it |
| `Port 5432 already in use` | Another PostgreSQL instance | Change the host mapping in `infra/docker-compose.yml` and `POSTGRES_PORT` in `.env` |
| `FAL 401 Unauthorized` | Invalid or expired FAL_KEY | Check `.env` FAL_KEY value |
| `ElevenLabs 401` | Invalid API key | Check `.env` ELEVEN_LABS_KEY |
| JSON truncation from the LLM | Model hit context limit | Already mitigated with `num_ctx=32768`; retry with a shorter prompt |
| `nomic-embed-text not found` | Embedding model not pulled | `ollama pull nomic-embed-text` |
| `nvidia-cudnn-cu12 not installed` | cuDNN missing | `pip install nvidia-cudnn-cu12` (optional, speeds up faster-whisper) |

## 12. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     NEWS COLLECTION                          │
│  RSS feeds → async scraper → salience extractor → LLM      │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     SCRIPT GENERATION                        │
│  Local LLM (selected via model_profile.json) → LangChain    │
│  → script evaluator. 2 stories × 4 beats, no greeting,       │
│  CTA-free closing (ADR-019)                                  │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     MEDIA GENERATION                         │
│  TTS: Kokoro (primary) / ElevenLabs / Edge TTS              │
│  Images: MLX-Gen local (Qwen-Image + pixel LoRA)            │
│  Subtitles: faster-whisper word timestamps → ASS burn-in    │
│  Title: persistent overlay (from story topics)               │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     VIDEO ASSEMBLY                           │
│  Split-screen: scene (60%) + avatar (40%)                    │
│  Static grid-preserving scenes, ASS subtitles,               │
│  ffmpeg audio mastering                                      │
│  Output: 1080×1920 vertical MP4, typically 100-130s          │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     PUBLISHING                               │
│  YouTube Shorts (OAuth2) + TikTok (Content Posting API)     │
│  Telegram notification → launchd agent daily + pmset wake    │
└─────────────────────────────────────────────────────────────┘
```

## 13. Key Configuration Files

| File | Purpose |
|---|---|
| `.env` | All environment variables (gitignored — see `.env.example`) |
| `config/model_profile.json` | Selected model per role (gitignored; written by `tools/model_setup.py`) |
| `config/system_prompts.json` | LLM system prompts (Mask persona, visual generator, etc.) |
| `config/generation_profiles.json` | Image sampling profiles (Qwen / FLUX Klein) |
| `config/image_style.json` | Prompt style, palette, negative prompt, split layout |
| `infra/docker-compose.yml` | PostgreSQL + n8n containers (optional) |
| `src/automate.py` | Master automation (schedule/wake/power, generate, publish, notify) |
| `src/publish_video.py` | YouTube + TikTok + Instagram publisher |
| `tools/generate_complete_video.py` | Full pipeline entry point |
| `tools/run_daily.sh` | launchd wrapper for the daily run |