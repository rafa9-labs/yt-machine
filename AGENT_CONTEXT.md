# YT-Machine Agent Context — Editorial & Creative Direction

> **Purpose**: This file is the single source of truth for any AI agent working on this project.
> Read this FIRST before making any changes. It defines what the channel is, what videos it makes,
> how the system works, and where it's going.

---

## 1. Channel Identity

**Name**: Geopolitical Sentinel  
**Format**: Short-form vertical video (60–80 seconds)  
**Platforms**: TikTok, YouTube Shorts, Instagram Reels  
**Voice**: Senior intelligence analyst — authoritative, contrarian, pattern-aware  
**Visual Signature**: True 16-bit isometric pixel art with a locked brand palette  
 
**Core Promise to the Audience**:  
"We show you the angle mainstream media buries. Every event connects to a pattern — we reveal it."
 
**What Makes This Channel Different**:
- Every video links a current event to 2–3 historical parallels (Deep Context)
- Pixel-art visuals create a distinct, recognizable aesthetic nobody else uses for news
- The tone is analytical, not sensational — intelligence briefing, not clickbait
- Second-order consequences are always surfaced: "Who actually benefits from the chaos?"
 
---

## 2. Content Categories (Rotation System)

Videos rotate across these categories to avoid audience fatigue. The system tracks rotation in `output/category_rotation.json`.
 
| Category | Description | Example Topics |
|---|---|---|
| **middle_east_conflict** | Iran, Israel, Hormuz, Yemen, Hezbollah, Gulf states | Houthi missile strikes, IRGC operations, Strait blockade |
| **great_power_competition** | US vs China vs Russia, NATO, QUAD, Taiwan | Ukraine talks, South China Sea, AUKUS |
| **economic_warfare** | Sanctions, trade wars, OPEC, currency, supply chains | Petrodollar shift, SWIFT weaponization, chip embargo |
| **regional_flashpoints** | Africa, Latin America, South/SE Asia, Korea | Sahel coups, Kashmir tension, North Korea tests |
| **technology_disruption** | AI, cyber, quantum, semiconductors, space | AI arms race, chip war, satellite warfare |
| **climate_geopolitics** | Energy transition, water, food, rare minerals | Lithium wars, Arctic routes, drought migration |
| **diplomatic_pivot** | Negotiations, alliances, backchannel deals | Summit outcomes, normalization, realignment |
| **kinetic_operations** | Active military operations, strikes, deployments | Airstrikes, naval exercises, special operations |
 
**Rule**: Never produce 3+ videos of the same category in a row. Check `output/video_history.json` before topic selection.
 
---

## 3. Video Structure (6 Segments)

Every video follows this narrative arc. Each segment gets one image.
 
| # | Segment | Duration | Purpose | Image Style |
|---|---|---|---|---|
| 1 | **hook** | 10–18 words | Pattern-interrupt opening — shocking number, question, or contrarian take | Dramatic close-up, high contrast, attention-grabbing |
| 2 | **historical_1** | ~12s | First historical parallel (e.g., 1990s) | Wide panoramic, sepia-amber, archival feel |
| 3 | **historical_2** | ~12s | Second historical parallel (different era) | Strategic overhead, cold war palette, olive drab |
| 4 | **modern_pivot** | ~12s | Return to 2026 — what changed, new players | Dynamic diagonal, modern HD, sharp clean lines |
| 5 | **consequence** | ~12s | Human impact — prices, shortages, civilian life | Ground-level, eye-height, emotional framing |
| 6 | **future_outlook** | ~12s | Where this leads — strategic implication | Wide strategic map view, golden hour, contemplative |
 
**Hook Rules** (most critical segment):
- NEVER start with a date or "Today..."
- Use ONE of: shocking number, provocative question, contrarian statement, action scene
- Maximum 18 words — short, punchy, one idea
- Example: "One hundred twelve dollars per barrel. Twenty one percent of global oil just stopped flowing."
 
---

## 4. Visual Style (Non-Negotiable)
 
### Brand Palette
| Color | Hex | Usage |
|---|---|---|
| Dark Navy Blue | `#0A1628` | Primary background |
| Amber Orange | `#FFA500` | Accent, highlights, fire, energy |
| Cyan Blue | `#00D4FF` | Technology, water, information |
| Slate Gray | `#4A5568` | Neutral elements, infrastructure |
 
### Pixel Art Rules
- **True 16-bit pixel art** — hard edges, limited palette, no anti-aliasing
- **Isometric perspective** — consistent across all images
- **SNES/retro aesthetic** — not modern low-poly, not voxel art
- **Detailed proportions** — military equipment, buildings, vehicles must be recognizable
- **No text in images** — no labels, HUD, speech bubbles, watermarks
 
### Image Generation Model
- **Primary**: `fal-ai/flux-pro/v1.1-ultra` (pixel-art optimized)
- **Fallback chain**: `fal-ai/flux-lora` → `fal-ai/flux/dev` → `fal-ai/flux/schnell`
- **LoRA**: `prithivMLmods/Retro-Pixel-Flux-LoRA` at scale 0.85
- **Trigger word**: "Retro Pixel"
- **Image-to-Image**: Supported via reference_image parameter for style consistency
- **Portrait 4:3** aspect ratio (vertical short-form)
 
### Prompt Hierarchy (Flux Best Practice)
Flux weights earlier tokens more heavily. Always structure prompts:
1. **SUBJECT** — what the image shows (highest weight)
2. **ACTION** — what's happening (dynamic verbs)
3. **ENVIRONMENT** — where (geography, setting)
4. **COMPOSITION** — camera angle, framing (from scene config)
5. **LIGHTING** — per-segment lighting (from scene config)
6. **MOOD** — atmosphere (lowest weight)
 
---

## 5. Script Writing Standards
 
### Narrative Science Principles
1. **Cause-and-effect chain** — every sentence causally connects to the next
2. **Personification** — "Tehran's energy minister just got the call" not "sanctions were lifted"
3. **Concrete specifics** — exact numbers, names, dates over vague language
4. **Historical pattern** — "This is the third time in forty years..."
5. **Surprise/contrast** — reveal second-order consequences that reframe everything
 
### What the Script Must Do
- Connect the current event to 2–3 historical parallels
- Surface the contrarian angle mainstream misses
- End with an open question (drives comments and rewatches)
- Stay between 60–80 seconds when read aloud
 
### What the Script Must NOT Do
- Start with dates or "Today..."
- Use generic filler ("In this video we'll explore...")
- Be a neutral Wikipedia summary — take a position
- Use more than 3 proper nouns per sentence
- Exceed 80 seconds — retention collapses after that
 
---

## 6. Content Quality Gates
 
### Image Generation
- **Specificity score** ≥ 35/100 (prompt must be concrete, not generic)
- **Geopolitical accuracy** validated (correct equipment per country, correct geography)
- **Visual diversity** enforced (each segment uses different camera/lighting/framing from config)
- **No military bias** — economic, diplomatic, civilian imagery equally weighted
 
### Script Generation
- **6 segments** matching the narrative structure
- **Historical anchoring** — at least 2 historical parallels
- **LLM context window** — 8192 tokens to prevent truncation
 
### Pipeline Checks
- **Duplicate detection** — 7-day window via `output/video_history.json`
- **Category rotation** — no 3+ same-category streak via `output/category_rotation.json`
- **Trending context** — current RSS trending terms injected into prompts for relevance
 
---

## 7. Technical Architecture Summary
 
```
generate_complete_video.py          ← Entry point: runs full pipeline
│
├── redfish/rss_scraper.py          ← Fetches 16+ RSS feeds, scores viral potential
├── redfish/debate_engine.py        ← Multi-agent debate: Skeptic vs Explainer
├── brain/llm_interface.py          ← Ollama LLM wrapper (DeepSeek R1 / Llama 3.2)
├── redfish/prompt_generator.py     ← Script-first visual prompt construction
├── redfish/script_parser.py        ← Extracts visual concepts from script text
├── redfish/trending_analyzer.py    ← Real-time trending term extraction from RSS
│
├── video_server/pixel_art_tool.py  ← FAL.ai image generation + I2I reference
├── video_server/tts_tool.py        ← Microsoft edge-tts voiceover
├── video_server/assembler_tool.py  ← moviepy video assembly with camera movements
│
├── config/system_prompts.json      ← LLM personas and script structure
├── config/image_style.json         ← Visual style single source of truth
└── output/projects/<id>/           ← Generated video, images, audio, manifest
```
 
### Key Dependencies
- **Ollama** running at localhost:11434 (local LLM — no API cost)
- **FAL_KEY** in .env (fal.ai image generation — ~$0.04/image)
- **edge-tts** (Microsoft TTS — free, no API key)
- **moviepy** (video assembly — local, free)
 
---

## 8. Current Development Priorities
 
### Active (feature/image-text-correlation branch)
- Pixel-art optimized model integration (flux-pro/v1.1-ultra)
- Image-to-Image reference pipeline for style consistency
- Accuracy refinement parameters (4 control modes)
- Script-to-image synchronization — images match narration
- Trending context injection into prompts
- Military bias removal from specificity scoring
 
### Known Issues to Address
- **Script relevance** is low (6–12%) — prompts don't closely match script segments
- **Low specificity fallback** still triggers generic enrichment too often
- **Category rotation** skews toward middle_east_conflict (see video_history.json)
- **Audio mastering** occasionally fails with array stacking error (non-blocking)
 
### Future (Not Started)
- n8n + Docker automation for scheduled publishing (infrastructure exists but not wired)
- YouTube/TikTok/Instagram API publishing integration
- A/B testing different hook styles for retention optimization
- Audience feedback loop into content selection
 
---

## 9. Rules for AI Agents Working on This Project
 
1. **Read `config/image_style.json` and `config/system_prompts.json` before touching visual or script code** — these are the single sources of truth
2. **Never hardcode style values** — always read from config files
3. **Preserve backward compatibility** — new parameters default to None/False
4. **Test with `python generate_complete_video.py`** — the pipeline must complete end-to-end
5. **Check `output/video_history.json`** before claiming a feature works — verify actual output
6. **Don't add Docker/n8n/publishing features** unless explicitly asked — focus is content quality
7. **Don't create documentation files** unless asked — keep the repo clean
8. **Commit to `feature/image-text-correlation` branch** — this is the active development branch
9. **Brand palette and pixel-art style are non-negotiable** — never change the visual identity
10. **Every code change must result in a runnable pipeline** — no broken intermediate states