# YT Machine - Autonomous Video Generation System

An agentic AI system that autonomously creates viral short-form videos from global news feeds using local LLMs, debate-driven ideation, and automated video production.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    YT MACHINE PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage A: Web Scraping (Redfish)                            │
│  ├─ 8 RSS feeds (aiohttp parallel fetch, ~2s)               │
│  ├─ Full article extraction (trafilatura + Playwright)      │
│  └─ Viral scoring + category rotation                       │
│                                                              │
│  Stage B: LLM Pipeline (Fast-Slow Architecture)             │
│  ├─ Worker: Qwen3 4B abliterated (all JSON/text tasks)      │
│  │   ├─ news_processor (structured analysis)                 │
│  │   ├─ debate_skeptic / debate_explainer                    │
│  │   ├─ visual_prompt_generator                              │
│  │   ├─ script_curator                                       │
│  │   └─ salience_extractor                                   │
│  ├─ Brain: Qwen3 30B MoE abliterated (reasoning only)       │
│  │   └─ multi_news_synthesizer (3-story script synthesis)    │
│  └─ Fallback: Gemma 4 26B Heretic (safety net)              │
│                                                              │
│  Stage C: Video Production                                   │
│  ├─ Voiceover (edge-tts / Kokoro / ElevenLabs)              │
│  ├─ Pixel Art Images (FAL.ai FLUX + LoRA)                   │
│  ├─ Stock Footage (Pexels API)                               │
│  └─ Video Assembly (moviepy + ffmpeg)                        │
│                                                              │
│  Stage D: Automation & Publishing                           │
│  ├─ automate.py (single entry point)                        │
│  ├─ Telegram notifications                                   │
│  └─ Windows Task Scheduler integration                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Current Status: Stages A-D Operational

### Completed Components

#### Phase 1: Agentic Foundation & Memory ✓
1. **Open Viking Memory System**
   - Directory Structure: `/open-viking/{resources, skills, history}`
   - Memory Logger: Write video metadata with performance tracking
   - Memory Reader: Query past videos, check duplicates, analyze performance
   - Tested: All memory operations verified with sample data

2. **LLM Brain Interface (Fast-Slow Architecture)**
   - Worker: Qwen3 4B abliterated (fast JSON/text, ~80+ tok/s on RTX 3090)
   - Brain: Qwen3 30B-A3B MoE abliterated (reasoning, ~60+ tok/s on RTX 3090)
   - Fallback: Gemma 4 26B Heretic (safety net)
   - System Prompts: 7 specialized personas with per-task model routing
   - LLM Wrapper: Ollama API + LangChain with retry logic, streaming, and Pydantic output parsing
   - Debate System: Multi-agent conversation framework
   - Thinking token stripping: Handles Qwen3 `<think`, Gemma4 `<|channel|>`, DeepSeek-R1 patterns

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

### 3. Pull LLM Models (Fast-Slow Architecture)

The pipeline uses a two-model setup optimized for RTX 3090 24GB:

| Role | Model | VRAM | Purpose |
|---|---|---|---|
| **Worker** | `huihui_ai/qwen3-abliterated:4b` | ~4 GB | JSON output, news analysis, curation (fast) |
| **Brain** | `huihui_ai/qwen3-abliterated:30b-a3b` | ~19 GB | Script synthesis, reasoning (MoE) |
| **Fallback** | `hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF` | ~16 GB | Safety net if Qwen3 models fail |

All three models are abliterated/uncensored — no content restrictions on geopolitical news topics.

```powershell
# Pull the Worker (2.5 GB download — fast JSON/text tasks)
ollama pull huihui_ai/qwen3-abliterated:4b

# Pull the Brain (18 GB download — MoE reasoning model)
ollama pull huihui_ai/qwen3-abliterated:30b-a3b

# Pull the Fallback (16 GB download — already have it? skip this)
ollama pull hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:latest

# Verify all models are available
ollama list
```

**Why this setup?** The Brain is a Mixture-of-Experts (MoE) model — 30B total parameters but only ~3B active per token. This gives large-model intelligence at small-model speed. The Worker handles all structured tasks (JSON output, curation, dedup) at 80+ tok/s. Only script synthesis hits the Brain. Total VRAM: ~23 GB of 24 GB.

### 4. Install LangChain (Recommended)

LangChain adds auto-retry on JSON parse failures, Pydantic-validated output, and declarative model fallbacks. Without it, the pipeline uses raw HTTP requests with manual JSON parsing.

```powershell
pip install langchain langchain-ollama langchain-core
```

### 5. Verify Installation
```powershell
# Test model connectivity + benchmark all 3 model tiers
python test_ollama.py

# Run full pipeline validation (config, token stripping, live LLM calls)
python tests\test_pipeline_models.py
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
# Worker (4B) handles these — fast structured output
analysis = llm.process_news(article_text)
# Returns: {topic, key_facts, angle, keywords, impact_score, ...}

skeptic = llm.debate_skeptic(analysis)
explainer = llm.debate_explainer(analysis, skeptic)

# Brain (30B MoE) handles this — heavy reasoning
script = llm.synthesize_multi_news_script([analysis1, analysis2, analysis3])
# Returns: {greeting, stories: [{headline, part1, part2, segue}, ...], full_text, segment_timeline}
```

## Running the Pipeline

### Quick Reference

| Command | What it does |
|---|---|
| `python generate_complete_video.py` | **Full pipeline** — scrape, analyze, script, images, voice, video |
| `python generate_complete_video.py --skip-images` | **No images** — uses placeholders (faster, no FAL.ai API calls) |
| `python generate_complete_video.py --no-telegram` | **No Telegram** — skips delivery notification |
| `python generate_complete_video.py --resume output/projects/video_XXXXX` | **Resume** — picks up a failed run from where it stopped |

### Full Production Run

```powershell
# Complete pipeline: RSS → analysis → debate → script → images → voice → video
python generate_complete_video.py
# Output: output/projects/video_<timestamp>/<timestamp>.mp4
# Time: ~10-15 min total (LLM ~6 min, images ~5 min, video assembly ~2 min)
```

### Testing & Debugging Runs

```powershell
# Skip image generation — uses placeholder black frames instead
# Saves ~5 min and avoids FAL.ai API costs
python generate_complete_video.py --skip-images

# Skip Telegram notification (useful when testing repeatedly)
python generate_complete_video.py --no-telegram

# Combine both for fastest test cycle
python generate_complete_video.py --skip-images --no-telegram

# Resume a failed pipeline run (reuses existing analysis/script/images)
python generate_complete_video.py --resume output/projects/video_1776278345
```

### Automation via automate.py

```powershell
# Generate only (PC already on, no Wake-on-LAN)
python automate.py --generate

# Generate + publish to all platforms (YouTube, TikTok, Instagram)
python automate.py --publish

# Generate + publish to YouTube only
python automate.py --publish youtube

# Full flow: Wake PC via WOL → wait for boot → generate → Telegram notify
python automate.py

# Wake the PC only (don't generate)
python automate.py --wake-only

# Resume a failed run via automation (with Telegram notifications)
python automate.py --resume output/projects/video_1776278345
```

### Scheduling

```powershell
# Install daily scheduled task at 8:00 AM
python automate.py --schedule "08:00"

# Install with default time (08:00)
python automate.py --install-schedule

# Remove the scheduled task
python automate.py --remove-schedule
```

## Next Steps (Phase 3: Video MCP Server)

- [ ] Build Python MCP server for video generation
- [ ] Implement `generate_voiceover(text)` using edge-tts
- [ ] Implement `fetch_stock_video(keyword)` using Pexels API
- [ ] Implement `assemble_short(audio, video)` using moviepy
- [ ] Test end-to-end video generation from script

## Configuration

### System Prompts
Edit `config/system_prompts.json` to customize:
- News processing behavior
- Debate persona characteristics
- Script structure and tone
- Model parameters (temperature, max_tokens)

### Ollama Settings
Default configuration (in `config/system_prompts.json`):
- **Base URL**: `http://localhost:11434`
- **Default Model**: `huihui_ai/qwen3-abliterated:4b` (Worker)
- **Reasoning Model**: `huihui_ai/qwen3-abliterated:30b-a3b` (Brain — MoE)
- **Fallback Model**: `hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF` (Gemma 4 Heretic)
- **Context Window**: 16384 tokens (reduced from 32768 to fit both models in VRAM)
- **Timeout**: 300 seconds
- **Retry Attempts**: 3

### Model Routing
Task → Model mapping (configured in `config/system_prompts.json` → `task_models`):
- `multi_news_synthesizer` → Brain (30B MoE) — heavy reasoning
- `news_processor` → Worker (4B) — structured JSON
- `visual_prompt_generator` → Worker (4B) — structured JSON
- `script_curator` → Worker (4B) — text transformation
- `salience_extractor` → Worker (4B) — analysis
- `debate_skeptic` → Worker (4B) — debate
- `debate_explainer` → Worker (4B) — debate

## Troubleshooting

### Ollama Connection Issues
```powershell
# Check if Ollama is running
curl http://localhost:11434/api/tags

# List available models
ollama list

# Restart Ollama
Stop-Process -Name "ollama" -Force
ollama serve
```

### Model Issues
```powershell
# If a model fails to load, check VRAM usage
nvidia-smi

# Unload all models from VRAM (frees ~23 GB)
# Ollama auto-unloads after 5 min idle, but this forces it
ollama stop huihui_ai/qwen3-abliterated:30b-a3b
ollama stop huihui_ai/qwen3-abliterated:4b

# If both models don't fit simultaneously (24 GB limit):
# Option 1: Reduce num_ctx in config/system_prompts.json (16384 → 8192)
# Option 2: Only load one model at a time (Ollama auto-swaps but slower)
# Option 3: Use only the Worker (4B) for all tasks (fast but lower quality scripts)
```

### Thinking Token Issues
If LLM responses contain `<think...` or `<|channel|>` in the output:
- `_strip_thinking_tokens()` in `brain/llm_interface.py` handles these automatically
- If tokens leak through, the JSON extraction will still find the `{...}` block
- Run `python test_ollama.py` to verify token stripping works for your model

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
