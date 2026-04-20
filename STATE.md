# YT-Machine — Current State (2026-04-20)

> **Purpose**: Snapshot of where the project is right now. Read this after AGENTS.md.

---

## Last Session: What Was Done (2026-04-20)

### 1. Image Generation: Switched to FAL Flux/Dev ✅
- **Before**: `fal-ai/flux-lora` at $0.07/image — expensive for 6 images/run
- **After**: `fal-ai/flux/dev` primary ($0.025/image), `fal-ai/flux/schnell` fallback. Pixel art prefix prepended to all prompts. 1MP cost cap. Skip destructive upscale when image >= target resolution.
- **Files**: `video_server/pixel_art_tool.py`, `config/image_style.json`, `.env.example`

### 2. Personality Revert: Removed "Self-Aware Algorithm" ✅
- **Problem**: A previous iteration introduced a "Masker v5 — Self-Aware Algorithm" personality with glitch markers (`*[stutter]*`, `*[system_warning]*`, `*[feed_interrupted]*`, `*[rebooting]*`, `*[signal_lost]*`) that bled into TTS, subtitles, and the final video.
- **Fix**: Fully reverted to original Masker comedian personality across ALL layers:
  - `brain/llm_interface.py` — greeting, segues, closing (Truman Show outro), intro_hook, synthesis prompt, segue validation keywords
  - `config/system_prompts.json` — `multi_news_synthesizer` (back to "comedian who does the news"), `script_curator` (back to "Speech Coach for ElevenLabs"), `visual_prompt_generator` (removed glitch instruction)
- **Files**: `brain/llm_interface.py`, `config/system_prompts.json`

### 3. Subtitle Fixes ✅
- **Overlap fix**: Phrase clips extended +0.15s for last-word highlight caused visual overlap. `clip_start` now accounts for previous phrase's extension.
- **Sync fix**: `_clean_script_for_subtitles()` now strips `*[marker]*` patterns before word alignment, preventing phantom words and timing drift.
- **TTS cleaning**: `tts_tool.py` strips `*[marker]*` before sending to ElevenLabs.
- **Files**: `video_server/subtitle_renderer.py`, `video_server/tts_tool.py`

### 4. Image Zoom Effect Fix ✅
- **Problem**: Zoom expressions were mathematically broken — ffmpeg's zoompan starts `zoom` at 1.0, but the expression tried to subtract from 1.0 and immediately hit the `max(1.0, ...)` floor. **Zero visible movement guaranteed.**
- **Fix**: Replaced with frame-counter-based (`on`) alternating zoom — no panning:
  - Even scenes (0,2,4): zoom OUT from 1.25→1.0 (starts cropped, reveals full image)
  - Odd scenes (1,3,5): zoom IN from 1.0→1.25 (starts full, zooms into center)
  - Pure center-based — no stretching, output frame size never changes, only crop region shifts
- **Files**: `video_server/split_video_assembler.py`

### 5. Closing Restored ✅
- Reverted to Truman Show-inspired outro: `"And these were the news for today. Subscribe, like, share. And in case I don't see you, good morning, good afternoon... and good night."`
- Old greeting restored: `"Good Morning/Afternoon/Evening! I'm Masker!"`
- **Files**: `brain/llm_interface.py`

---

## Known Issues / Remaining

### Needs Verification
- **Zoom effect**: Mathematically correct but needs a test run to confirm visible movement in output video
- **Subtitle sync**: Overlap and drift fixes need end-to-end test
- **Personality**: System prompts reverted but LLM may still occasionally use old patterns — monitor first few runs

### Not Yet Started
- n8n/Docker scheduled publishing
- YouTube/TikTok/Instagram API publishing
- A/B testing hooks for retention
- PDF/book input for video generation
- Connected storytelling (narrative thread between 3 stories)

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
