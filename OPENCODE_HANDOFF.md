# OpenCode Handoff Prompt — yt-machine

Copy everything below the line into OpenCode to continue work.

---

## START OF PROMPT

You are continuing work on the **yt-machine** project — an automated geopolitical news video generator. Read these files first (in order):

1. **`AGENT_CONTEXT.md`** — Channel identity, content rules, visual style, full architecture
2. **`STATE.md`** — Current state snapshot, what was done last session, known issues
3. **`OPENCODE.md`** — Project rules, code style, PowerShell quirks

### What This Project Does
Automates creation of 60-80 second vertical news videos with:
- RSS news scraping (16+ feeds) → LLM analysis → Script synthesis → Pixel art images → TTS voiceover → Video assembly → Telegram delivery
- 3 stories per video, 6 pixel art images, karaoke subtitles, zoom animation, avatar overlay
- Entry point: `python generate_complete_video.py`

### What Was Done Last Session (2026-04-15)
1. ✅ **Beep removal** — Fixed loud audio beep at video start (removed 500ms leading silence in `video_server/tts_tool.py`)
2. ✅ **Outro unification** — Fixed conflicting outro instructions in `config/system_prompts.json`. Now always uses: "... And these were the news for this {time_of_day}. Subscribe, like, share and in case if I don't see you: good morning, good afternoon, and good night!"
3. ✅ **Telegram import fix** — Fixed `send_video` → `send_video_to_telegram` in `generate_complete_video.py`
4. ✅ **Pipeline unification** — Merged v1 and v2 pipelines into single `generate_complete_video.py`. `generate_v2.py` is now a redirect.
5. ✅ **Zoom effect** — Added slow zoom-out animation (1.08x → 1.0x) on scene images in `video_server/split_video_assembler.py`
6. ✅ **OpenCode migration** — Created `.opencode.json`, `OPENCODE.md`, `STATE.md`

### What Needs To Be Done Next (Prioritized)

#### PRIORITY 1: Fix Non-Critical Issues (Quick Wins)
- [ ] **PostgreSQL save fails** — `null value in column "topic"` in `_save_to_postgres()`. The function in `generate_complete_video.py` needs to include a topic field when inserting. Low priority but clutters logs.
- [ ] **LangChain analysis chain** — Prompt template in `brain/langchain_interface.py` or `brain/chains/news_analysis.py` has unescaped curly braces (`{\n  "topic"}` should be `{{\n  "topic"}}`). Falls back to raw LLM fine but wastes a call each time.
- [ ] **Async scraper import** — `AsyncRSScraper` class name doesn't match what's in `redfish/async_scraper.py`. Quick fix — check the actual class name and update the import.
- [ ] **Vector dedup** — `VideoTopicStore` not found in `brain.memory.vector_store`. Check if it was renamed or needs to be implemented.

#### PRIORITY 2: Automation (The Original Goal)
The user wants full automation: Wake PC remotely → run pipeline → send to Telegram. Tools already exist:
- `tools/wake_pc.py` — Wake-on-LAN script (exists, needs testing)
- `tools/telegram_sender.py` — `send_video_to_telegram()` function (exists, working)
- `generate_complete_video.py` — Already calls Telegram at end of pipeline
- [ ] Create a master automation script that ties WOL + pipeline + Telegram together
- [ ] Set up scheduled triggering (cron/Task Scheduler/n8n)

#### PRIORITY 3: Performance
- [ ] Zoom rendering is very slow (~15-20 min per video). Could use ffmpeg-native zoom filters instead of moviepy frame-by-frame processing in `video_server/split_video_assembler.py`

#### PRIORITY 4: Future Features
- [ ] YouTube/TikTok/Instagram API publishing
- [ ] A/B testing hooks for retention optimization
- [ ] n8n Docker automation for scheduled publishing

### Key Environment Info
- **OS**: Windows 11, PowerShell 5.1 (NO `&&` for chaining — use `;`)
- **Python**: venv at `./venv/`
- **LLM**: Z.ai GLM-5 (primary) + Ollama localhost:11434 (fallback)
- **Image gen**: FAL.ai (flux-lora with pixel art LoRA)
- **TTS**: ElevenLabs (primary), Kokoro/Edge (fallback)
- **Video**: moviepy + ffmpeg (bundled via imageio_ffmpeg)
- **DB**: PostgreSQL (forex_ml database, optional — pipeline works without it)
- **.env keys**: ZHIPUAI_API_KEY, FAL_KEY, ELEVENLABS_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

### Quick Test Command
```powershell
python generate_complete_video.py
```
This runs the full pipeline end-to-end. Use `--skip-images` for faster testing without API calls. Use `--no-telegram` to skip Telegram delivery.

### Last Successful Output
`output/projects/video_1776278345/video_1776278345.mp4` — 14.6 MB, 76s, 6 pixel art scenes, 3 Iran/China stories.

## END OF PROMPT