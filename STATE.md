# YT-Machine — Current State (2026-04-16)

> **Purpose**: Snapshot of where the project is right now. Read this after AGENT_CONTEXT.md.

---

## Last Session: What Was Done (2026-04-16)

### 1. Vector Dedup Import Fix ✅
- **Problem**: `generate_complete_video.py` imported `VideoTopicStore` and `TopicDeduplicator` — neither existed. Real names: `VectorStore`, `DeduplicationChecker`. This silently killed ALL semantic dedup for months.
- **Fix**: Fixed both imports + method calls (`is_duplicate()` → `check_topic()`). Also fixed `AsyncRSScraper` → `AsyncScraper` and `fetch_all_feeds()` → `scrape_all()`.
- **Impact**: Semantic dedup via pgvector now actually works. Ollama embeddings + cosine similarity prevent repeat stories across runs.
- **Files**: `generate_complete_video.py` (lines 202, 278-279, 281, 284-297, 1052-1063)

### 2. PostgreSQL Save Fix ✅
- **Problem**: `_save_to_postgres()` didn't include `topic` column → `null value in column "topic"` error on every run
- **Fix**: Added optional `topic` parameter; on "completed" step, extracts topic from first news analysis and includes it in INSERT.
- **Files**: `generate_complete_video.py` (lines 145-168, 1061)

### 3. LangChain Curly Braces Fix ✅
- **Problem**: 6 prompts in `system_prompts.json` had raw JSON examples with `{` and `}` — LangChain's `ChatPromptTemplate` interpreted them as template variables → KeyError
- **Fix**: Escaped all literal braces: `{` → `{{`, `}` → `}}` in `news_processor`, `debate_skeptic`, `debate_explainer`, `visual_prompt_generator`, `salience_extractor`, and `multi_news_synthesizer`
- **Files**: `config/system_prompts.json`

### 4. LLM-Based Intra-Batch Dedup ✅
- **Problem**: Story selection used a weak 5-word frozenset overlap check — near-identical stories with different wording slipped through
- **Fix**: Replaced with `_is_semantically_similar()` that uses the LLM to judge if two headlines cover the same story. Falls back to frozenset if LLM fails.
- **Files**: `generate_complete_video.py` (lines 248-272)

### 5. Subtitle Timing Overhaul ✅
- **Problem**: Fixed 0.2s offset, low anchor matching (70% threshold), mismatched pre-roll (1.0s scene vs 0.3s subtitle)
- **Fix**: 
  - Dynamic drift calculation per video (median of anchor position differences)
  - Phonetic matching for proper nouns (consonant skeleton comparison)
  - Lowered fuzzy match threshold from 70% to 60%
  - Unified pre-roll: scenes now use 0.3s (same as subtitle lead-in)
- **Files**: `video_server/subtitle_renderer.py`, `generate_complete_video.py`, `rebuild_video.py`

### 6. Image Motion Effects ✅
- **Problem**: Images had only zoom/pan — needed more life within their confined space
- **Fix**: Added 3 new ambient effects layered on top of motion effects:
  - **Breathing pulse**: 1-2% scale oscillation (sin wave, 0.5Hz)
  - **Pixel shimmer**: Random sparkle overlay (CRT/retro effect)
  - **Vignette breath**: Edge brightness pulse (0.3Hz ambient lighting)
  - Effects rotate across scenes (each scene gets a different motion + ambient combo)
- **Files**: `video_server/split_video_assembler.py`

---

## Known Issues / Remaining

### Non-Critical
- **Zoom rendering slow**: ~15-20 min for a 76s video on CPU. Could use ffmpeg-native zoom filters instead of moviepy frame-by-frame processing.
- **pgvector dependency**: Semantic dedup requires PostgreSQL with pgvector extension running + `topic_embeddings` table. If not available, dedup silently falls back to LLM-based intra-batch check.

### Not Yet Started
- Wake-on-LAN + automated Telegram delivery (tools exist, automation script not written)
- n8n/Docker scheduled publishing
- YouTube/TikTok/Instagram API publishing
- A/B testing hooks for retention
- PDF/book input for video generation
- Connected storytelling (narrative thread between 3 stories)

---

## Pipeline Output
Last successful run: `output/projects/video_1776278345/`
- 3 stories: Iran/China satellite intel, US-Iran ceasefire, Strait of Hormuz threat
- 6 pixel art images (1088×1152)
- 76s voiceover (ElevenLabs, 204 word timestamps)
- 14.6 MB video file
- Telegram delivery: configured but needs verification

---

## Environment Requirements
- **.env keys needed**: `ZHIPUAI_API_KEY`, `FAL_KEY`, `ELEVENLABS_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- **Ollama**: Running at localhost:11434 (fallback LLM + embeddings for dedup)
- **Python venv**: `./venv/` with requirements.txt installed
- **ffmpeg**: Bundled via imageio_ffmpeg

---

## MCP Server Status
All 8 MCP servers from Cline are configured in `.opencode.json`:
1. filesystem (C:\Users\rafa\Projects)
2. github
3. context7 (library docs)
4. sequential-thinking
5. brave-search
6. puppeteer (browser automation)
7. postgres (forex_ml database)
8. fetch (HTTP/YouTube transcripts)

**Note**: Brave search requires `BRAVE_API_KEY` env var. Postgres connection string is in the config.
