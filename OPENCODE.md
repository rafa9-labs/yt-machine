# OpenCode Project Rules — yt-machine

## Environment Context
- Operating System: Windows 11
- Terminal: Windows PowerShell 5.1
- Python: 3.x (venv at `./venv/`)
- Project: Geopolitical news video generator (automated pipeline)

## Command Execution Rules
- **CRITICAL:** Do NOT use `&&` to chain commands — PowerShell 5.1 doesn't support it
- Use `;` to separate commands (e.g., `git add . ; git commit -m "update"`)
- Or run commands as separate sequential steps

## Project Architecture
- **Entry point**: `generate_complete_video.py` — runs the full unified pipeline
- **Config single source of truth**: `config/system_prompts.json` (LLM prompts), `config/image_style.json` (visuals)
- **Read `AGENT_CONTEXT.md` first** — it's the definitive project reference for any AI agent

## Code Style Rules
1. **Never hardcode style values** — always read from config files
2. **Preserve backward compatibility** — new parameters default to `None`/`False`
3. **structlog for logging** — use `from brain.log import get_logger`
4. **No print()** — use structured logging via `log.info()`, `log.warning()`, `log.error()`
5. **Type hints** on function signatures where practical
6. **Docstrings** on public functions

## Pipeline Rules
1. Test with `python generate_complete_video.py` — must complete end-to-end
2. Every code change must result in a runnable pipeline — no broken intermediate states
3. Brand palette and pixel-art style are non-negotiable — never change visual identity
4. Check `output/video_history.json` before claiming a feature works
5. Don't add Docker/n8n/publishing features unless explicitly asked
6. Don't create documentation files unless asked — keep repo clean
7. Commit to `feature/image-text-correlation` branch — active development branch

## Available MCP Tools
- **filesystem**: Read/write files in `C:\Users\rafa\Projects`
- **github**: GitHub API (issues, PRs, repos, code search)
- **context7**: Query library documentation (resolve → query)
- **sequential-thinking**: Complex multi-step reasoning
- **brave-search**: Web search (general + local)
- **puppeteer**: Browser automation (navigate, screenshot, click, fill)
- **postgres**: SQL queries on `forex_ml` database (read-only)
- **fetch**: HTTP requests (HTML, markdown, JSON, YouTube transcripts)

## Key Files Quick Reference
| File | Purpose |
|---|---|
| `AGENT_CONTEXT.md` | Single source of truth for project identity |
| `generate_complete_video.py` | Unified pipeline (v1+v2 merged) |
| `generate_v2.py` | Redirect → generate_complete_video.py |
| `server.py` | FastAPI server (pipeline API) |
| `brain/llm_interface.py` | LLM wrapper (GLM-5 + Ollama fallback) |
| `brain/langchain_interface.py` | LangChain structured chains |
| `video_server/tts_tool.py` | ElevenLabs/Kokoro/Edge TTS + audio mastering |
| `video_server/split_video_assembler.py` | Video assembly (zoom, subtitles, avatar) |
| `video_server/pixel_art_tool.py` | FAL.ai pixel art generation |
| `video_server/subtitle_renderer.py` | Karaoke subtitles with word alignment |
| `tools/telegram_sender.py` | Send videos to Telegram |
| `tools/wake_pc.py` | Wake-on-LAN for remote PC |
| `redfish/rss_scraper.py` | 16+ RSS feed scraper |
| `redfish/async_scraper.py` | Async parallel scraper |
| `config/system_prompts.json` | LLM personas, script structure |
| `config/image_style.json` | Visual style config (1080×1920) |