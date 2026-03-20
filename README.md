# YT Machine - Autonomous Video Generation System

An agentic AI system that autonomously creates viral short-form videos from global news feeds using local LLMs, debate-driven ideation, and automated video production.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    YT MACHINE PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Brain & Memory                                    │
│  ├─ Ollama (DeepSeek R1/V3)                                 │
│  ├─ Open Viking Memory System                               │
│  └─ System Prompt Configuration                             │
│                                                              │
│  Phase 2: Ideation Engine                                   │
│  ├─ Redfish (RSS Scraper)                                   │
│  ├─ Debate Agents (Skeptic vs Explainer)                    │
│  └─ Promptfoo Quality Assurance                             │
│                                                              │
│  Phase 3: Video MCP Server                                  │
│  ├─ Voiceover Generation (edge-tts)                         │
│  ├─ Stock Footage Fetcher (Pexels API)                      │
│  └─ Video Assembly (moviepy)                                │
│                                                              │
│  Phase 4: Automation & Publishing                           │
│  ├─ n8n Workflow Orchestration                              │
│  └─ Multi-Platform Publishing (YT/TikTok/IG)                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Current Status: Phase 1 & 2 Complete ✓

### Completed Components

#### Phase 1: Agentic Foundation & Memory ✓
1. **Open Viking Memory System**
   - Directory Structure: `/open-viking/{resources, skills, history}`
   - Memory Logger: Write video metadata with performance tracking
   - Memory Reader: Query past videos, check duplicates, analyze performance
   - Tested: All memory operations verified with sample data

2. **LLM Brain Interface**
   - Model: Llama 3.2 (via Ollama)
   - System Prompts: 4 specialized personas (News Processor, Skeptic, Explainer, Script Synthesizer)
   - LLM Wrapper: Ollama API integration with retry logic, streaming, and robust JSON parsing
   - Debate System: Multi-agent conversation framework

#### Phase 2: Ideation Engine & Testing ✓
1. **Redfish RSS Scraper**
   - 8 global news sources (BBC, TechCrunch, The Verge, Hacker News, Science Daily, NPR, etc.)
   - Fetches 80-100 articles per run
   - Viral scoring algorithm ranks by potential
   - Filters by publication date and keywords

2. **Debate Engine**
   - Full pipeline: News → Analysis → Debate → Script
   - Multi-agent debate (Skeptic vs Explainer)
   - Duplicate detection via memory integration
   - Outputs structured 45-second scripts
   - **Successfully tested with live news articles**

## Installation

### 1. Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Install Ollama
Download from: https://ollama.ai/download

### 3. Pull DeepSeek Model
```powershell
ollama pull deepseek-r1:latest
```

### 4. Verify Installation
```powershell
# Test memory system
cd open-viking
python test_memory.py

# Test LLM interface (requires Ollama running)
cd ..\brain
python test_llm.py
```

## Project Structure

```
yt-machine/
├── open-viking/              # Memory & state management
│   ├── resources/            # Raw materials (news, footage metadata)
│   ├── skills/               # Learned patterns & templates
│   ├── history/              # Video logs & analytics
│   │   └── videos.json       # Master video index
│   ├── memory_logger.py      # Write operations
│   ├── memory_reader.py      # Read/query operations
│   └── test_memory.py        # Memory system tests
│
├── brain/                    # LLM interface & reasoning
│   ├── llm_interface.py      # Ollama API wrapper
│   └── test_llm.py           # LLM integration tests
│
├── config/                   # Configuration files
│   ├── system_prompts.json   # LLM persona definitions
│   └── ollama_setup.md       # Installation guide
│
└── requirements.txt          # Python dependencies
```

## Usage Examples

### Memory System
```python
from open_viking.memory_logger import MemoryLogger
from open_viking.memory_reader import MemoryReader

# Log a new video
logger = MemoryLogger()
logger.log_video({
    "topic": "AI Breakthrough in Quantum Computing",
    "script": {...},
    "keywords": ["AI", "quantum", "tech"],
    "video_path": "/output/video_001.mp4"
})

# Check for duplicates
reader = MemoryReader()
duplicate = reader.check_topic_coverage("quantum computing", days=7)
if duplicate["duplicate_found"]:
    print(f"Already covered {duplicate['days_ago']} days ago")

# Get performance stats
stats = reader.get_performance_stats(days=30)
print(f"Avg views: {stats['avg_views']}")
print(f"Top keywords: {stats['top_keywords']}")
```

### LLM Interface
```python
from brain.llm_interface import LLMInterface

llm = LLMInterface()

# Process news article
analysis = llm.process_news(article_text)
# Returns: {topic, key_facts, angle, keywords, virality_score}

# Run debate
skeptic = llm.debate_skeptic(analysis)
explainer = llm.debate_explainer(analysis, skeptic)

# Generate final script
script = llm.synthesize_script(analysis, skeptic, explainer)
# Returns: {hook, body, twist, cta, word_count, estimated_duration}
```

## Next Steps (Phase 3: Video MCP Server)

- [ ] Build Python MCP server for video generation
- [ ] Implement `generate_voiceover(text)` using edge-tts
- [ ] Implement `fetch_stock_video(keyword)` using Pexels API
- [ ] Implement `assemble_short(audio, video)` using moviepy
- [ ] Test end-to-end video generation from script

## Phase 4: Automation & Publishing

- [ ] Set up n8n workflow orchestration
- [ ] Configure cron triggers for automated runs
- [ ] Integrate YouTube, TikTok, Instagram APIs
- [ ] Deploy full autonomous pipeline

## Configuration

### System Prompts
Edit `config/system_prompts.json` to customize:
- News processing behavior
- Debate persona characteristics
- Script structure and tone
- Model parameters (temperature, max_tokens)

### Ollama Settings
Default configuration:
- **Base URL**: `http://localhost:11434`
- **Model**: `deepseek-r1:latest`
- **Timeout**: 30 seconds
- **Retry Attempts**: 3

## Troubleshooting

### Ollama Connection Issues
```powershell
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
Stop-Process -Name "ollama" -Force
ollama serve
```

### Memory System Errors
- Ensure `open-viking/history/videos.json` exists
- Check file permissions for write access
- Verify JSON structure is valid

## License

MIT

## Contributing

This is an autonomous agent system. Contributions should focus on:
- Improving prompt quality
- Adding new debate personas
- Enhancing memory retrieval algorithms
- Optimizing video generation pipeline
