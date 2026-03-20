# Sentinel v2.1 — Framework Map

## Pipeline: Step-by-Step Flow

```
1. SCRAPE       redfish/rss_scraper.py
                └─ RSScraper.scrape_all(max_age_hours=48)
                └─ filter_viral_potential(top_n=20)
                └─ Hormuz keyword boost → select target article

2. BRAIN        brain/llm_interface.py  ←  config/system_prompts.json
                └─ process_news()        → analysis + 6 pixel_art_prompts + ticker_headlines
                └─ debate_skeptic()      → critique
                └─ debate_explainer()    → response
                └─ synthesize_script()   → hook / body / twist / cta

3. ASSETS       video_server/pixel_art_tool.py
                └─ generate_pixel_art(prompt) × 6
                   ├─ FAL_KEY set  → fal-ai/stable-diffusion-xl-lightning
                   │                  LoRA: PixelArtRedmond-Lite64 (scale 0.85)
                   │                  4 inference steps, portrait_4_3
                   └─ FAL_KEY unset → Pillow tactical-grid placeholder (1024×1792)

                video_server/tts_tool.py
                └─ generate_voiceover(script, voice_tone="authoritative")
                   └─ edge-tts → output/audio/*.mp3
                   └─ Fallback: silent WAV → MP3 if Edge TTS fails

4. ASSEMBLE     video_server/assembler_tool.py
                └─ build_final_video(audio, 6×images, ticker_headlines)
                   ├─ Ken Burns zoom effect (pixel art mode)
                   ├─ CRT scanline overlay (PIL)
                   ├─ HUD overlay: "SENTINEL v2.1 | TACTICAL BRIEFING" + REC indicator
                   └─ Scrolling ticker marquee (3 headlines)
                   → output/videos/hormuz_sentinel_<slug>_<timestamp>.mp4

5. MEMORY       open-viking/memory_logger.py
                └─ log_video(metadata) → open-viking/history/videos.json
```

## Cost Per Video

| Step | Tool | Cost |
|------|------|------|
| 6× Pixel Art | Fal.ai SDXL-Lightning | ~$0.018 |
| Voiceover | edge-tts (Microsoft) | Free |
| LLM | Ollama / Llama 3.2 (local) | Free |
| Stock footage | Not used in Sentinel mode | $0 |
| **Total** | | **~$0.018** |

## Configuration Files

- `config/system_prompts.json` — LLM personas, 6-Act structure, Neo-Pixel palette enforcement
- `config/rss_feeds.json` — 8 RSS sources
- `.env` — `FAL_KEY`, `PEXELS_API_KEY`

## Entry Point

```powershell
python run_hormuz_sentinel.py
```

## Framework Validation

- ✅ 6-Act Structure enforced in `news_processor` system prompt
- ✅ Neo-Pixel palette (Cyan #00FFFF / Orange #FF6600) hardcoded
- ✅ Fal.ai SDXL-Lightning + PixelArtRedmond LoRA configured
- ✅ Placeholder fallback generates 6 unique tactical grids
- ✅ MoviePy assembler includes HUD + Ticker + Scanlines
- ✅ Memory logging operational

## Pre-Launch Checklist

1. Ollama running: `ollama serve`
2. Model loaded: `ollama pull llama3.2:latest`
3. FAL_KEY in `.env` (optional for test run — placeholders will generate if unset)
4. MoviePy version matches import syntax (check `pip show moviepy`)
