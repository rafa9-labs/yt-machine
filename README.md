# YT Machine

YT Machine is an agentic automation pipeline that moves from global news feeds to short-form video production using local language models, debate-driven ideation, voice, visuals, and video assembly.

## Pipeline

- **Collect**: fetch and extract articles from RSS sources
- **Research**: structure stories, score topics, and check memory for duplicates
- **Develop**: debate angles and produce a short-form script
- **Produce**: generate voiceover, visuals, and a rendered video
- **Automate**: schedule runs and send completion notifications

## Stack

Python · Ollama · LangChain · Pydantic · MoviePy · FFmpeg · Playwright

## Development

```bash
pip install -r requirements-macos.txt   # macOS / Apple Silicon
python tools/generate_complete_video.py --skip-images --no-telegram
```

## Daily automation (macOS)

The Mac wakes itself, generates a video, publishes to YouTube/TikTok, and
sleeps again:

```bash
python src/automate.py --install-schedule 06:00   # launchd agent
sudo pmset repeat wakeorpoweron MTWRFSU 05:50:00  # wake before the run
python src/automate.py --configure-power 20       # sleep after the run

python src/automate.py --show-schedule            # verify
```

One-time publishing credentials:

```bash
python tools/youtube_auth.py          # YouTube OAuth consent
python tools/tiktok_auth.py --check   # TikTok (after app approval)
```

See `SETUP.md` §9 for the full walkthrough and troubleshooting.

The public repository documents the pipeline and its operational stages.

## License

MIT
