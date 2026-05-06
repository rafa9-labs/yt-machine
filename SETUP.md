# Geopolitical Sentinel — Setup Guide

## Prerequisites

- Windows 11 with WSL2 (Ubuntu 22.04+)
- NVIDIA GPU with 12GB+ VRAM (RTX 3090/4090 recommended)
- 32GB+ system RAM
- Docker Desktop (for PostgreSQL)
- ~50GB free disk space (models + output)

## 1. Clone and Install

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

## 2. WSL2 Memory Configuration

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
cp config/.env.example .env
nano .env
```

### Required variables

| Variable | Description | Example |
|---|---|---|
| `ELEVEN_LABS_KEY` | ElevenLabs API key for TTS | `sk_...` |
| `FAL_KEY` | fal.ai API key for cloud image generation fallback | `key-...` |
| `POSTGRES_HOST` | PostgreSQL host | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5433` |
| `POSTGRES_USER` | PostgreSQL user | `yt_machine` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `your_password` |
| `POSTGRES_DB` | PostgreSQL database | `yt_machine` |

### Publishing variables (YouTube + TikTok)

| Variable | Description |
|---|---|
| `YOUTUBE_CLIENT_SECRETS_FILE` | Path to OAuth2 credentials JSON (see YouTube setup below) |
| `YOUTUBE_CREDENTIALS_FILE` | Path where OAuth token will be cached (auto-created) |
| `TIKTOK_CLIENT_KEY` | TikTok Developer app client key |
| `TIKTOK_CLIENT_SECRET` | TikTok Developer app client secret |
| `TIKTOK_ACCESS_TOKEN` | TikTok content posting access token |

### Notification variables

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Your chat ID (run `python -m tools.telegram_sender --get-chat-id`) |

### Automation variables

| Variable | Description | Default |
|---|---|---|
| `WSL_USER` | Your WSL username (for Task Scheduler) | `rafa` |
| `WOL_MAC` | Target PC MAC address for Wake-on-LAN | — |
| `PIPELINE_TIMEOUT` | Pipeline timeout in seconds | `900` |

### Optional toggles

| Variable | Description | Default |
|---|---|---|
| `USE_KOKORO` | Use Kokoro TTS instead of ElevenLabs | `false` |
| `USE_LOCAL_FLUX` | Use local FLUX for image generation | `auto` |
| `LOCAL_FLUX_MIN_VRAM_GB` | Minimum free VRAM to use local FLUX | `14` |
| `LOCAL_FLUX_EVICT_OLLAMA` | Evict Ollama from GPU before loading FLUX | `true` |

## 4. Ollama Setup

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the required model
ollama pull hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:latest

# Pull embedding model for topic deduplication
ollama pull nomic-embed-text

# Verify model is available
ollama list
```

## 5. PostgreSQL Setup

```bash
# Start PostgreSQL container (port 5433 to avoid conflicts)
cd ~/yt-machine
docker compose -f infra/docker-compose.yml up -d postgres

# Wait for it to be healthy
docker compose -f infra/docker-compose.yml ps

# The pipeline auto-creates tables on first run via init_db()
```

Note: The default `docker-compose.yml` uses port 5432 internally but the `.env` maps it to 5433 on the host to avoid conflicts with other PostgreSQL instances. If you need to change this, edit `infra/docker-compose.yml` ports mapping.

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
cd ~/yt-machine
source venv/bin/activate

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
cd ~/yt-machine
source venv/bin/activate

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

## 9. Automation (Windows Task Scheduler)

The pipeline can run automatically every day, even when the PC is asleep.

### Step 1: Put PC to Sleep (not shut down)

Configure Windows power settings so the PC sleeps instead of shutting down. The Task Scheduler can wake from sleep.

### Step 2: Create Scheduled Task

From **Windows PowerShell** (not WSL):

```powershell
cd C:\Users\rafa\yt-machine
python src/automate.py --install-schedule "08:00"
```

This creates a Windows Task Scheduler task called `GeopoliticalSentinel_DailyVideo` that runs daily at 08:00.

### Step 3: Enable Wake-to-Run

1. Open **Task Scheduler** (`taskschd.msc`)
2. Find `GeopoliticalSentinel_DailyVideo`
3. Right-click → **Properties**
4. **Conditions** tab → check **"Wake the computer to run this task"**
5. **Settings** tab → check **"Run task as soon as possible after a scheduled start is missed"**
6. Click OK

### Step 4: Set Idle Sleep Timer

In Windows Power Settings, configure the PC to sleep after 30 minutes of idle. This way:
- 08:00 → Task Scheduler wakes PC
- 08:00-08:15 → Pipeline runs
- 08:15+ → Publish to YouTube + TikTok
- 08:15+ → Telegram notification sent
- ~08:30 → PC goes back to sleep after idle timeout

### Troubleshooting Automation

| Problem | Solution |
|---|---|
| Task doesn't wake PC | Check BIOS: enable "Wake on LAN" or "Wake on Alarm" |
| Task runs but pipeline fails | Check `output/logs/automate_YYYYMMDD.log` |
| WSL command not found | Ensure `wsl` is in Windows PATH; try `wsl ~ -e bash ...` |
| YouTube OAuth fails unattended | Run `--dry-run` once manually to cache the token |
| TikTok returns 401 | Token expired; regenerate from TikTok Developer Portal |

## 10. Troubleshooting Common Errors

| Error | Cause | Fix |
|---|---|---|
| `Ollama 404: model not found` | Model not pulled | `ollama pull hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:latest` |
| `CLIP 77-token truncation warning` | FLUX CLIP encoder truncates long prompts | Expected and harmless; scene content is front-loaded |
| `CUDA out of memory` | VRAM too low | Set `USE_LOCAL_FLUX=false` in `.env` or increase `LOCAL_FLUX_MIN_VRAM_GB` |
| `psycopg2.OperationalError: connection refused` | PostgreSQL not running | `docker compose -f infra/docker-compose.yml up -d postgres` |
| `Port 5433 already in use` | Another PostgreSQL on 5433 | Change `POSTGRES_PORT` in `.env` or stop the other instance |
| `FAL 401 Unauthorized` | Invalid or expired FAL_KEY | Check `.env` FAL_KEY value |
| `ElevenLabs 401` | Invalid API key | Check `.env` ELEVEN_LABS_KEY |
| JSON truncation from Ollama | Model hit context limit | Already mitigated with `num_ctx=32768`; retry with shorter prompt |
| `nomic-embed-text not found` | Embedding model not pulled | `ollama pull nomic-embed-text` |
| `nvidia-cudnn-cu12 not installed` | cuDNN missing | `pip install nvidia-cudnn-cu12` (optional, speeds up faster-whisper) |

## 11. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     NEWS COLLECTION                          │
│  RSS feeds → async scraper → salience extractor → LLM      │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     SCRIPT GENERATION                        │
│  Ollama (gemma-4-26B) → LangChain → script evaluator        │
│  2 stories × 4 segments each + greeting + closing            │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     MEDIA GENERATION                         │
│  TTS: ElevenLabs (primary) / Kokoro / Edge TTS              │
│  Images: Local FLUX GGUF (primary) / fal.ai (fallback)      │
│  Subtitles: faster-whisper word timestamps → ASS burn-in    │
│  Title: persistent overlay (4-8 words from story topics)     │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     VIDEO ASSEMBLY                           │
│  Split-screen: scene (60%) + avatar (40%)                   │
│  Ken Burns zoom, ASS subtitles, audio mastering              │
│  Output: 1080×1920 vertical MP4, ~65-70 seconds              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     PUBLISHING                               │
│  YouTube Shorts (OAuth2) + TikTok (Content Posting API)     │
│  Telegram notification → Windows Task Scheduler daily at 08:00│
└─────────────────────────────────────────────────────────────┘
```

## 12. Key Configuration Files

| File | Purpose |
|---|---|
| `.env` | All environment variables (gitignored) |
| `config/system_prompts.json` | LLM system prompts (Mask persona, visual generator, etc.) |
| `config/image_style.json` | FLUX prompt style, CLIP tags, color palette, negative prompt |
| `infra/docker-compose.yml` | PostgreSQL + n8n containers |
| `src/automate.py` | Master automation script (WOL + pipeline + publish + notify) |
| `src/publish_video.py` | YouTube + TikTok + Instagram publisher |
| `tools/generate_complete_video.py` | Full pipeline entry point |
| `tools/run_daily.sh` | Bash wrapper for Task Scheduler |