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
pip install -r requirements.txt
python generate_complete_video.py --skip-images --no-telegram
```

The public repository documents the pipeline and its operational stages.

## License

MIT
