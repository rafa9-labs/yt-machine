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

---

## LoRA Training Pipeline (Phase 3)

Train a custom Flux LoRA on your RTX 3090 to lock in your channel's pixel art style permanently.

### One-Time Setup

**1. Add to `.env`**
```
FAL_KEY=your_fal_ai_key
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   # from https://huggingface.co/settings/tokens
```

**2. Install training dependencies**
```powershell
# PyTorch with CUDA 12.1 (match your driver version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Training stack
pip install diffusers peft accelerate bitsandbytes transformers safetensors huggingface_hub
```

**3. Configure accelerate (run once)**
```powershell
accelerate config
# Select: single GPU, bf16 mixed precision, no distributed training
```

### Training Workflow

**Step 1 — Generate diverse training images (~$2-3, ~20 min)**
```powershell
python tools/auto_generate_training_set.py --count 70
# Generates 70 images across 8 categories: military air/naval/ground,
# economic, diplomatic, geographic, human impact, historical, energy
# Saves to training_data/ with caption .txt files
# Resumable — safe to interrupt and re-run
```

**Step 2 — Train the LoRA locally on your 3090 (~1-3 hrs, free)**
```powershell
# Train and upload to HuggingFace Hub (recommended — permanent storage)
python tools/train_lora_local.py training_data/ --steps 1200 --upload-to-hub

# Train locally only (no upload)
python tools/train_lora_local.py training_data/ --steps 1200
```

**Step 3 — Done. Every future video automatically uses your LoRA.**

`config/custom_lora.json` is updated automatically after training.
`pixel_art_tool.py` reads it on startup — no code changes needed.

To revert to the default HuggingFace LoRA: `del config\custom_lora.json`

### Training Parameters (RTX 3090 24GB)

| Parameter | Default | Notes |
|---|---|---|
| `--steps` | 1200 | 800 = fast, 1500 = best quality |
| `--rank` | 16 | Higher = more expressive but more VRAM |
| Base model | FLUX.1-dev | ~24GB download, cached in HF cache |
| VRAM usage | ~18-22GB | Safe on 3090 24GB |
| Mixed precision | bf16 | RTX 3090 native support |

### What Gets Trained

The LoRA learns your **style**, not new factual knowledge:
- Exact navy/amber/cyan color palette (`#0A1628`, `#FFA500`, `#00D4FF`)
- Isometric pixel art perspective and framing
- Pixel density and hard-edge aesthetics
- Generalises across all visual categories (military, economic, diplomatic, etc.)

Trigger word: `sentinel_pixel` (prepended to every prompt automatically)

---

## License

MIT

## Contributing

This is an autonomous agent system. Contributions should focus on:
- Improving prompt quality
- Adding new debate personas
- Enhancing memory retrieval algorithms
- Optimizing video generation pipeline
