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

## 3. Video Structure (3-News Multi-Story Format)

Every video covers **3 news stories** ordered by impact (least → most important, climax last).
Each story gets 2 images (setup + payoff). Total: 12 segments, 6 images, 75–90 seconds.

 | # | Segment | Duration | Purpose | Image |
 |---|---|---|---|---|
 | 1 | **intro** | ~5s | Time-of-day greeting + teaser of all 3 stories | img 0 |
 | 2 | **intro_pause** | ~1s | Beat pause — let the hook land | img 0 |
 | 3 | **story_1_part1** | ~8s | Story 1 setup (least important) | img 0 |
 | 4 | **story_1_part2** | ~8s | Story 1 payoff / punchline | img 1 |
 | 5 | **story_1_transition** | ~2s | Bridge to next story | img 1 |
 | 6 | **separator** | ~1s | .... pause | — |
 | 7–10 | **story_2** | ~20s | Same structure (medium importance) | img 2–3 |
 | 11 | **separator** | ~1s | .... pause | — |
 | 12–15 | **story_3** | ~20s | Same structure (most important — climax) | img 4–5 |
 | 16 | **pre_closing_pause** | ~1s | Beat pause before CTA | img 5 |
 | 17 | **closing** | ~5s | Subscribe/like CTA + "I'm Masker, see you tomorrow" | img 5 |
 
 **Intro Rules**:
 - Time-aware greeting: "Good Morning/Afternoon/Evening! I'm Masker!"
 - Tease all 3 stories in one punchy sentence
 - Example: "Tonight: Turkey discovers geography makes excellent blackmail, France denies US overflights, and Ukraine's energy war escalates."
 
 **Closing Rules** (mandatory):
 - Must include subscribe/like CTA
 - Must sign off as "I'm Masker"
 - Must reference "see you tomorrow" or similar
 
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
- **12 segments** (3 stories × 4 segments each: part1, part2, transition, separator)
- **GLM-5 via Z.ai** (primary) with Ollama fallback
- **LLM max_tokens** multiplied by 3× for GLM-5 (reasoning model hidden token budget)
- **4 LLM steps**: news analysis → script synthesis → script curation → visual prompts
 
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
├── brain/llm_interface.py          ← GLM-5 (Z.ai) primary + Ollama fallback LLM wrapper
├── redfish/prompt_generator.py     ← Script-first visual prompt construction
├── redfish/script_parser.py        ← Extracts visual concepts from script text
├── redfish/trending_analyzer.py    ← Real-time trending term extraction from RSS
│
├── video_server/pixel_art_tool.py          ← FAL.ai image generation + I2I reference
├── video_server/tts_tool.py                ← Kokoro TTS (primary) + Edge TTS (fallback) + audio mastering
├── video_server/split_video_assembler.py   ← Full-screen image bg + animated zoom/pan + avatar
├── video_server/subtitle_renderer.py       ← Karaoke subtitles with word alignment + title overlay
├── video_server/assembler_tool.py          ← Legacy assembler (not used in main pipeline)
│
├── config/system_prompts.json              ← LLM personas and script structure
├── config/image_style.json                 ← Visual style single source of truth (1080×1920)
└── output/projects/<id>/                   ← Generated video, images, audio, manifest
```

### Video Layout (1080×1920 vertical)
```
┌──────────────────────────┐
│  HOOK / TITLE TEXT       │  ← Fades in, top overlay, outlined
│                          │
│  Full-screen scene image │  ← Animated zoom-out or pan-top-to-bottom
│  with Ken Burns effects  │
│                          │
│  KARAOKE SUBTITLES       │  ← 5-word phrases, yellow highlight, outline only
│  (no black bar)          │     positioned above center
│                          │
│  AVATAR LOOP             │  ← Bottom half (1080×960), pixel art character
│  (bottom 50%)            │
└──────────────────────────┘
```

### Subtitle System
- **Word alignment**: Whisper timestamps mapped to original script words (not whisper transcription)
- **Phrase display**: 5 words at a time with yellow karaoke highlight
- **No background band**: Outlined text only (black outline, no rectangle)
- **Title overlay**: Hook text displayed at top with fade-in for first 5 seconds
 
### Key Dependencies
- **ZHIPUAI_API_KEY** in .env (GLM-5 via Z.ai — primary LLM for all script generation)
- **Ollama** running at localhost:11434 (fallback LLM — no API cost)
- **FAL_KEY** in .env (fal.ai image generation — ~$0.04/image)
- **Kokoro TTS** (primary voice — natural human-like speech, open source)
- **edge-tts** (fallback TTS — free, no API key)
- **faster-whisper** (word timestamps for subtitle sync)
- **soundfile + ffmpeg** (audio mastering — loudness normalization)
- **moviepy** (video assembly — local, free)
 
---

## 8. Current Development Priorities
 
### Completed
- ✅ GLM-5 (Z.ai) integration — primary LLM for all 4 pipeline steps
- ✅ Kokoro TTS integration — natural speech with number normalization and pause handling
- ✅ Audio mastering fix — soundfile + ffmpeg pipeline (no more array stacking errors)
- ✅ 3-news multi-story format — 12 segments, 6 images, intro/closing with CTA
- ✅ Script-to-image synchronization — dedicated visual prompts per narration segment
- ✅ Trending context injection — 40 terms extracted from RSS, injected into prompts
- ✅ Speech pipeline cleanup — number normalization, em-dash handling, story separator pauses
- ✅ `.env.example` for onboarding
 
### Known Issues to Address
- **Test files** scattered in project root (should be in `tests/` directory)
 
### Recently Fixed
- ✅ **Category rotation** expanded to all 8 categories (was only 4), tracks all 3 stories
- ✅ **Low specificity fallback** replaced with script-aware extraction (locations, actions, numbers from narration)
- ✅ **Dead code removed**: `prosody_processor.py` deleted (was 488 lines, unused in Kokoro pipeline)
- ✅ **File-based logging** added — all stdout mirrored to `output/logs/run_YYYYMMDD_HHMMSS.log`
 
### Future (Not Started)
- n8n + Docker automation for scheduled publishing
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