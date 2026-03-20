import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from redfish.rss_scraper import RSScraper
from redfish.debate_engine import DebateEngine

_mem_spec = importlib.util.spec_from_file_location(
    "memory_logger", project_root / "open-viking" / "memory_logger.py"
)
_mem_mod = importlib.util.module_from_spec(_mem_spec)
_mem_spec.loader.exec_module(_mem_mod)
MemoryLogger = _mem_mod.MemoryLogger

from video_server.pixel_art_tool import generate_pixel_art
from video_server.tts_tool import generate_voiceover
from video_server.assembler_tool import build_final_video

HORMUZ_KEYWORDS = [
    "strait of hormuz", "hormuz blockade", "haifa refinery",
    "haifa strike", "irgc", "oil blockade", "hormuz shipping",
    "persian gulf oil", "oil $110", "brent crude"
]

def print_banner(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def score_hormuz_relevance(article: dict) -> int:
    combined = (article.get("title", "") + " " + article.get("summary", "")).lower()
    score = article.get("virality_score", 0)
    for kw in HORMUZ_KEYWORDS:
        if kw in combined:
            score += 5
    return score

def main():
    print_banner("SENTINEL v2.2 — BUDGET KING NEO-PIXEL (6-ACT)")
    print(f"  Timestamp : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Context   : Hormuz Blockade | $110+ Oil | Helium/Fertilizer Crisis")

    # ── STEP 1: Scrape ───────────────────────────────────────────
    print_banner("STEP 1/5 — Scraping RSS Feeds")
    scraper = RSScraper()
    articles = scraper.scrape_all(max_age_hours=48)
    print(f"  Total articles fetched: {len(articles)}")

    viral_candidates = scraper.filter_viral_potential(articles, top_n=20)

    hormuz_scored = sorted(
        viral_candidates,
        key=score_hormuz_relevance,
        reverse=True
    )

    if not hormuz_scored:
        print("  [WARN] No relevant articles found. Using top viral article instead.")
        target_article = viral_candidates[0] if viral_candidates else articles[0]
    else:
        target_article = hormuz_scored[0]

    print(f"  Selected  : {target_article['title'][:70]}")
    print(f"  Source    : {target_article.get('feed_name', 'Unknown')}")
    print(f"  Score     : {score_hormuz_relevance(target_article)}")

    # ── STEP 2: Script Generation ─────────────────────────────────
    print_banner("STEP 2/5 — Generating Contrarian Hook Script")
    engine = DebateEngine()
    result = engine.run_full_pipeline(target_article)

    if not result:
        print("  [ERR] Pipeline failed for selected article. Aborting.")
        sys.exit(1)

    analysis  = result["analysis"]
    script    = result["script"]

    print(f"  Topic     : {analysis.get('topic', 'N/A')}")
    print(f"  Vector    : {analysis.get('shift_vector', 'N/A')}")
    print(f"  Impact    : {analysis.get('impact_score', 0)}/10")
    print(f"  Hook      : {script.get('hook', '')[:80]}")

    pixel_art_prompts = analysis.get("pixel_art_prompts", [])
    if len(pixel_art_prompts) < 6:
        print("  [WARN] LLM did not return 6 pixel_art_prompts. Using fallback 6-act prompts.")
        pixel_art_prompts = [
            "HOOK: Strait of Hormuz aerial wide shot at dusk, IRGC patrol boats in formation, oil tankers halted, midnight cyan water, emergency orange horizon, pixel art",
            "SCALE: Satellite view of Persian Gulf, dozens of stranded supertankers, naval blockade perimeter, tactical grid overlay, neon cyan on black, pixel art",
            "TENSION: Close-up IRGC commander in radar room, face lit by screens showing missile tracks, amber and red lighting, gritty cyberpunk pixel art",
            "PIVOT: Washington DC Situation Room, anonymous figures around a lit table, oil price graph on wall at $110, cold blue lighting, pixel art",
            "ESCALATION: Haifa Refinery complex on fire, black smoke against orange sky, emergency vehicles pixel dots, chaos composition, pixel art",
            "RESOLVE: Empty Hormuz strait at dawn, single tanker moving through, uncertain cyan light, ambiguous pixel art landscape"
        ]

    ticker_headlines = analysis.get("ticker_headlines", [])
    if len(ticker_headlines) < 3:
        print("  [WARN] LLM did not return 3 ticker_headlines. Using fallback headlines.")
        ticker_headlines = [
            "BREAKING: Hormuz Blockade — Brent Crude breaches $110/barrel",
            "ALERT: Haifa Refinery complex under drone threat — Gulf output at 60%",
            "DEVELOPING: Helium crisis deepens as Qatar North Dome output falls 40%"
        ]
    print(f"  Ticker    : {ticker_headlines[0][:60]}")

    # ── STEP 3: Pixel Art Generation ─────────────────────────────
    print_banner("STEP 3/5 — Generating 6 Pixel Art Scenes (6-Act Structure)")
    image_paths = []

    for i, prompt in enumerate(pixel_art_prompts[:6], 1):
        print(f"\n  [Scene {i}] {prompt[:70]}...")
        result_img = generate_pixel_art(prompt)

        if result_img["success"]:
            image_paths.append(result_img["path"])
            source = result_img.get("source", "unknown")
            print(f"  [OK]  Saved ({source}): {result_img['filename']}")
        else:
            print(f"  [ERR] Scene {i} failed: {result_img.get('error')}")

    if not image_paths:
        print("  [ERR] No pixel art images generated. Aborting.")
        sys.exit(1)

    # ── STEP 4: Voiceover ─────────────────────────────────────────
    print_banner("STEP 4/5 — Generating Voiceover")

    hook  = script.get("hook", "")
    body  = script.get("body", "")
    twist = script.get("twist", "")
    cta   = script.get("cta", "")

    full_script = f"{hook}\n\n{body}\n\n{twist}\n\n{cta}"
    print(f"  Words     : {len(full_script.split())}")

    tts_result = generate_voiceover(full_script, voice_tone="authoritative")

    if not tts_result["success"]:
        print(f"  [ERR] TTS failed: {tts_result.get('error')}")
        sys.exit(1)

    audio_path = tts_result["path"]
    print(f"  [OK]  Audio  : {tts_result['filename']}")
    print(f"  [OK]  Voice  : {tts_result['voice']}")
    print(f"  [OK]  Duration: ~{tts_result['estimated_duration_seconds']}s")

    # ── STEP 5: Video Assembly ────────────────────────────────────
    print_banner("STEP 5/5 — Assembling Tactical Briefing Video")

    topic_slug = "".join(
        c if c.isalnum() else "_"
        for c in analysis.get("topic", "hormuz_sentinel")[:40]
    ).strip("_")
    output_filename = f"hormuz_sentinel_{topic_slug}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.mp4"

    video_result = build_final_video(
        audio_path=audio_path,
        asset_paths=image_paths,
        ticker_headlines=ticker_headlines,
        is_pixel_art=True,
        output_filename=output_filename
    )

    # ── Final Report ──────────────────────────────────────────────
    print_banner("SENTINEL v2.2 — RUN COMPLETE")

    if video_result["success"]:
        print(f"  Status    : SUCCESS")
        print(f"  Output    : {video_result['path']}")
        print(f"  Duration  : {video_result['duration_seconds']}s")
        print(f"  Size      : {video_result['file_size_mb']} MB")
        print(f"  Effects   : {', '.join(video_result.get('effects_applied', []))}")

        logger = MemoryLogger()
        logger.log_video({
            "topic": analysis.get("topic", ""),
            "script": script,
            "keywords": analysis.get("keywords", []),
            "video_path": video_result["path"],
            "duration": video_result["duration_seconds"],
            "source_url": target_article.get("link", ""),
            "status": "generated",
            "metadata": {
                "virality_score": analysis.get("impact_score", 0),
                "shift_vector": analysis.get("shift_vector", ""),
                "pixel_art": True,
                "run_timestamp": datetime.utcnow().isoformat()
            }
        })
        print(f"  Memory    : Logged to open-viking history")
        print(f"\n  FINAL VIDEO: {video_result['path']}")

    else:
        print(f"  Status    : FAILED")
        print(f"  Error     : {video_result.get('error')}")
        sys.exit(1)

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
