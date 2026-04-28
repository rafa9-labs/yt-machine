"""
════════════════════════════════════════════════════════════════════════════════
generate_complete_video.py — UNIFIED Pipeline (v1 + v2 merged)
════════════════════════════════════════════════════════════════════════════════

UNIFIED DESIGN:
  - One pipeline file. No v1/v2 toggle needed.
  - New features (async scraper, LangChain, pgvector) are opt-in with fallbacks.
  - structlog for logging (falls back to standard logging if unavailable).
  - PostgreSQL saves after each step (non-blocking, JSON files always written).
  - Telegram delivery after successful video creation.

STEPS:
  1.   Fetch news (AsyncRSScraper → sync fallback)
  1.5. Vector dedup check (optional, requires pgvector)
  2.   News analysis (LangChain → raw LLM fallback)
  3.   Trending context
  4.   Script synthesis
  4.5. Visual prompt generation
  4.7. Script curation (LangChain → raw LLM fallback)
  5.   Pixel art generation
  6.   Video footage (skipped — pixel art only)
  7.   Voice generation
  8.   Video assembly
  9.   Platform metadata
  10.  Project summary + Telegram delivery
════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════
# STRUCTURED LOGGING — replaces old _TeeWriter + print() hack
# Falls back to standard logging if structlog not available.
# ══════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.brain.log import get_logger
log = get_logger("pipeline")

# ── CONFIGURATION ──
SKIP_IMAGES = os.environ.get("SKIP_IMAGES", "0") == "1"

# ── ARGUMENT PARSING ──
import argparse
parser = argparse.ArgumentParser(description='YT-MACHINE Unified Video Pipeline')
parser.add_argument('--resume', type=str, default=None, metavar='PROJECT_FOLDER',
                    help='Resume a failed pipeline run from the project folder')
parser.add_argument('--skip-images', action='store_true', help='Use placeholder images (no API calls)')
parser.add_argument('--no-telegram', action='store_true', help='Skip Telegram delivery')
args = parser.parse_args()
if args.skip_images:
    SKIP_IMAGES = True

# ── RESUME LOGIC ──
checkpoint = None
if args.resume:
    resume_path = Path(args.resume)
    checkpoint_file = resume_path / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        log.info("pipeline.resume", path=str(resume_path),
                 completed=checkpoint.get('completed_steps', []))
    else:
        log.warning("pipeline.resume.no_checkpoint", path=str(resume_path))

log.info("pipeline.start", skip_images=SKIP_IMAGES, no_telegram=args.no_telegram)


# ── HELPER: Extract text from script segments ──
def _extract_segment_text(segment_data) -> str:
    if isinstance(segment_data, str):
        return segment_data
    if isinstance(segment_data, dict):
        return (segment_data.get('narration')
                or segment_data.get('text')
                or segment_data.get('content')
                or str(segment_data))
    return str(segment_data) if segment_data else ''


from src.pipeline_utils import bridge_timestamp_gaps, build_fallback_prompt as _build_fallback_prompt


# ══════════════════════════════════════════════════════════════════════════
# LLM INITIALIZATION — Try LangChain first, fallback to raw interface
# ══════════════════════════════════════════════════════════════════════════
_USE_LANGCHAIN = True

try:
    from src.brain.langchain_interface import LangChainInterface
    _langchain = LangChainInterface()
    log.info("llm.langchain.loaded")
except Exception as e:
    log.warning("llm.langchain.failed", error=str(e), fallback="raw_llm")
    _USE_LANGCHAIN = False

# Always load raw interface as fallback — it handles complex methods
from src.brain.llm_interface import LLMInterface
llm = LLMInterface()
log.info("llm.raw.loaded")

# Video server components
from src.video.pixel_art_tool import generate_pixel_art
from src.video.pexels_tool import fetch_vertical_footage
from src.video.tts_tool import generate_voiceover
from src.video.split_video_assembler import build_split_video


# ── CHECKPOINT HELPER ──
def _save_checkpoint(step_name, project_folder, data=None):
    cp_path = Path(project_folder) / "checkpoint.json"
    existing = {}
    if cp_path.exists():
        try:
            with open(cp_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except:
            pass
    completed = existing.get('completed_steps', [])
    if step_name not in completed:
        completed.append(step_name)
    existing['completed_steps'] = completed
    existing['last_step'] = step_name
    existing['last_updated'] = datetime.now().isoformat()
    if data:
        existing.update(data)
    with open(cp_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False, default=str)


# ── PostgreSQL SAVE HELPER (non-blocking) ──
_STATUS_MAP = {
    'news_fetch': 'scraped',
    'vector_dedup': 'scraped',
    'news_analysis': 'analyzed',
    'trending_context': 'analyzed',
    'script_synthesis': 'scripted',
    'script_curation': 'scripted',
    'tts': 'voiceover_generated',
    'image_generation': 'assembled',
    'video_assembly': 'assembled',
    'completed': 'published',
}


def _save_to_postgres(step_name: str, project_id: int, data: dict, topic: str = None):
    """Save pipeline step results to PostgreSQL. Errors logged, never raised."""
    try:
        from src.db.connection import get_connection
        conn = get_connection()
        cursor = conn.cursor()

        _topic = topic or data.get('topic', '') or 'pending'
        _status = _STATUS_MAP.get(step_name, 'scraped')
        _pid = str(project_id)
        cursor.execute("""
            INSERT INTO videos (project_id, status, topic, created_at, updated_at)
            VALUES (%s, %s, %s, NOW(), NOW())
            ON CONFLICT (project_id) DO UPDATE SET
                status = EXCLUDED.status,
                topic = CASE WHEN videos.topic IN ('pending', '') AND EXCLUDED.topic NOT IN ('pending', '')
                         THEN EXCLUDED.topic ELSE videos.topic END,
                updated_at = NOW()
        """, (_pid, _status, _topic))
        conn.commit()
        cursor.close()
        conn.close()
        log.debug("postgres.saved", step=step_name, status=_status, project_id=project_id)
    except Exception as e:
        log.warning("postgres.save_failed", step=step_name, error=str(e))


# Create unique project folder (or reuse from resume)
if checkpoint and args.resume:
    project_folder = Path(args.resume)
    project_id = int(project_folder.name.replace('video_', ''))
    log.info("project.reuse", folder=str(project_folder), project_id=project_id)
else:
    project_id = int(time.time())
    project_folder = Path(f"output/projects/video_{project_id}")
    project_folder.mkdir(parents=True, exist_ok=True)
    log.info("project.create", folder=str(project_folder), project_id=project_id)

log = log.bind(project_id=project_id)

greeting_label = ('Morning' if datetime.now().hour < 12
                  else 'Afternoon' if datetime.now().hour < 18
                  else 'Evening')
log.info("time_of_day", greeting=greeting_label, hour=datetime.now().strftime('%H:%M'))


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: FETCH LATEST NEWS — Async Scraper with sync fallback
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="news_fetch")
_step_start = time.time()

articles = []
all_articles = []
viral_articles = []

try:
    # ── Try async scraper first (parallel fetching + JS rendering) ──
    from src.collector.async_scraper import AsyncScraper

    async def _fetch_articles():
        async_scraper = AsyncScraper()
        results = await async_scraper.scrape_all(max_age_hours=24)
        return results

    all_articles = asyncio.run(_fetch_articles())
    if hasattr(all_articles, 'articles'):
        all_articles = [a.model_dump() if hasattr(a, 'model_dump') else a for a in all_articles.articles]
    log.info("scraper.async.success", articles=len(all_articles))

    # Use sync scraper for ranking (filter_viral_potential)
    from src.collector.rss_scraper import RSScraper
    ranker = RSScraper()
    viral_articles = ranker.filter_viral_potential(all_articles, top_n=10)

    _save_to_postgres("news_fetch", project_id, {
        "article_count": len(all_articles),
        "scraper": "async",
    })

except Exception as e:
    log.warning("scraper.async.failed", error=str(e), fallback="sync_scraper")
    try:
        from src.collector.rss_scraper import RSScraper
        scraper = RSScraper()
        all_articles = scraper.scrape_all(max_age_hours=24)
        viral_articles = scraper.filter_viral_potential(all_articles, top_n=10)
        log.info("scraper.sync.fallback", articles=len(all_articles))
    except Exception as e2:
        log.error("scraper.all_failed", error=str(e2))
        exit(1)

if len(viral_articles) < 3:
    log.warning("articles.few", count=len(viral_articles), minimum=3)

# ── Topic diversity selection (LLM-aware) ──
def _is_semantically_similar(title_a: str, title_b: str) -> bool:
    """Use LLM to check if two article titles cover the same underlying story."""
    try:
        prompt = (
            "Are these two news headlines covering the SAME event or essentially the same story? "
            "Answer ONLY 'yes' or 'no'.\n\n"
            f"Headline A: {title_a}\n"
            f"Headline B: {title_b}"
        )
        response = llm.generate(prompt, max_tokens=10, temperature=0.1)
        answer = response.strip().lower()
        return answer.startswith('yes')
    except Exception:
        words_a = frozenset(title_a.lower().split()[:5])
        words_b = frozenset(title_b.lower().split()[:5])
        return len(words_a & words_b) >= 3

selected = []
for article in viral_articles:
    title = article.get('title', '')
    is_duplicate = False
    for existing in selected:
        if _is_semantically_similar(title, existing.get('title', '')):
            is_duplicate = True
            log.info("dedup.intra_batch.skip", new=title[:60], similar_to=existing.get('title', '')[:60])
            break
    if not is_duplicate:
        selected.append(article)
    if len(selected) >= 3:
        break

while len(selected) < 3 and len(viral_articles) > len(selected):
    for a in viral_articles:
        if a not in selected:
            selected.append(a)
            break
    if len(selected) >= 3:
        break

articles = selected[:3]
for i, a in enumerate(articles, 1):
    log.info("article.selected", index=i, title=a['title'][:70])

_save_checkpoint("news_fetch", project_folder, {"article_titles": [a.get('title', '') for a in articles]})
_step_duration = time.time() - _step_start
log.info("step.complete", step="news_fetch", duration_s=round(_step_duration, 2), articles=len(articles))


# ══════════════════════════════════════════════════════════════════════════
# STEP 1.5: VECTOR DEDUP CHECK (optional — requires pgvector)
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="vector_dedup")
_step_start = time.time()

try:
    from src.brain.memory.vector_store import VectorStore
    from src.brain.memory.deduplication import DeduplicationChecker

    dedup = DeduplicationChecker(threshold=0.35)
    log.info("dedup.loaded", backend="pgvector")

    deduped_articles = []
    for article in articles:
        topic_text = article.get('title', '')
        result = dedup.check_topic(topic_text)
        if result.is_duplicate:
            log.info("dedup.skip_duplicate", topic=topic_text[:60], matched=result.matched_topic)
        else:
            deduped_articles.append(article)

    if len(deduped_articles) < 3:
        log.warning("dedup.few_unique", count=len(deduped_articles))
        for a in viral_articles:
            if a not in deduped_articles and len(deduped_articles) < 3:
                result = dedup.check_topic(a.get('title', ''))
                if not result.is_duplicate:
                    deduped_articles.append(a)

    articles = deduped_articles[:3]
    log.info("dedup.complete", unique_articles=len(articles))

except Exception as e:
    log.warning("dedup.failed", error=str(e), action="continuing_without_dedup")

_step_duration = time.time() - _step_start
log.info("step.complete", step="vector_dedup", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: NEWS ANALYSIS — LangChain with raw LLM fallback
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="news_analysis")
_step_start = time.time()

news_analyses = []
for i, article in enumerate(articles, 1):
    try:
        # Get full article text
        try:
            from src.collector.rss_scraper import RSScraper
            text_scraper = RSScraper()
            article_text = text_scraper.get_full_article_text(article)
        except:
            article_text = article.get('summary', article.get('title', ''))

        log.debug("analysis.article", index=i, chars=len(article_text))

        analysis = None

        # ── Try LangChain structured chain first ──
        if _USE_LANGCHAIN:
            try:
                from src.models.schemas import NewsAnalysis as NewsAnalysisModel
                chain = _langchain.build_structured_chain(NewsAnalysisModel, "news_processor")
                result = chain.invoke({"input_text": f"Analyze this news article:\n\n{article_text}"})
                analysis = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
                log.info("analysis.langchain.success", index=i)
            except Exception as e:
                log.warning("analysis.langchain.failed", index=i, error=str(e))

        # ── Fallback to raw LLM interface ──
        if not analysis:
            analysis = llm.process_news(article_text)

        if analysis:
            log.info("analysis.complete", index=i,
                     topic=analysis.get('topic', 'N/A')[:60],
                     impact=analysis.get('impact_score', 0))
            news_analyses.append(analysis)
        else:
            log.error("analysis.empty", index=i)
    except Exception as e:
        log.error("analysis.failed", index=i, error=str(e))

if len(news_analyses) < 2:
    log.error("analysis.insufficient", count=len(news_analyses), minimum=2)
    exit(1)

# Sort by impact (lowest first, most important last for retention)
paired = []
for i, analysis in enumerate(news_analyses):
    art_idx = min(i, len(articles) - 1)
    paired.append((articles[art_idx], analysis))

paired.sort(key=lambda x: x[1].get('impact_score', 0))
articles = [p[0] for p in paired]
news_analyses = [p[1] for p in paired]

log.info("analysis.all_complete", stories=len(news_analyses))
for i, a in enumerate(news_analyses):
    log.info("analysis.ranked", rank=i+1,
             impact=a.get('impact_score', '?'),
             topic=a.get('topic', 'N/A')[:50])

_save_checkpoint("news_analysis", project_folder,
                 {"analysis_topics": [a.get('topic', '') for a in news_analyses]})
_save_to_postgres("news_analysis", project_id, {
    "topics": [a.get('topic', '') for a in news_analyses],
    "impact_scores": [a.get('impact_score', 0) for a in news_analyses],
})
_step_duration = time.time() - _step_start
log.info("step.complete", step="news_analysis", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: TRENDING CONTEXT (optional boost)
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="trending_context")
_step_start = time.time()

trending_context = {}
try:
    from src.collector.trending_analyzer import TrendingAnalyzer
    trending_analyzer = TrendingAnalyzer()
    trending_context = trending_analyzer.analyze(all_articles, top_n=40)
    log.info("trending.complete", terms=len(trending_context))
except Exception as e:
    log.warning("trending.failed", error=str(e))

_step_duration = time.time() - _step_start
log.info("step.complete", step="trending_context", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: MULTI-NEWS SCRIPT SYNTHESIS (Masker personality)
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="script_synthesis")
_step_start = time.time()

try:
    log.info("script.generating", format="multi_news_3_stories")
    script = llm.synthesize_multi_news_script(news_analyses)

    if not script:
        log.error("script.synthesis_failed", reason="empty_result")
        exit(1)

    full_script = script.get('full_text', '')

    # Strip greeting and intro_hook — go straight to the first news story
    greeting = script.get('greeting', '')
    intro_hook = script.get('intro_hook', '')
    for prefix in [greeting, intro_hook]:
        if prefix and full_script.startswith(prefix):
            full_script = full_script[len(prefix):].strip()
            script['full_text'] = full_script
    script['greeting'] = ''
    script['intro_hook'] = ''

    if not full_script:
        parts = []
        for story in script.get('stories', []):
            parts.append(story.get('mini_hook', ''))
            parts.append(story.get('body', ''))
            p = story.get('punchline', '')
            if p:
                parts.append(p)
            t = story.get('transition', '')
            if t:
                parts.append(t)
        parts.append(script.get('closing', ''))
        full_script = ' '.join(filter(None, parts))
        script['full_text'] = full_script

    script['word_count'] = len(full_script.split())
    script['estimated_duration'] = int(len(full_script.split()) / 2.5)

    log.info("script.synthesized",
             duration_s=script.get('estimated_duration', 0),
             words=script.get('word_count', 0))
    for i, story in enumerate(script.get('stories', []), 1):
        log.debug("script.story", index=i,
                  hook=story.get('mini_hook', 'N/A')[:60],
                  punchline=story.get('punchline', 'N/A')[:60])

    # Save script files
    script_file = project_folder / "script.txt"
    script_file.write_text(full_script, encoding='utf-8')

    segments_file = project_folder / "script_segments.json"
    segments_file.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding='utf-8')

    _save_checkpoint("script_synthesis", project_folder,
                     {"script_word_count": script.get('word_count', 0)})
    _save_to_postgres("script_synthesis", project_id,
                      {"word_count": script.get('word_count', 0)})

except Exception as e:
    log.error("script.synthesis.exception", error=str(e))
    import traceback
    traceback.print_exc()
    exit(1)

_step_duration = time.time() - _step_start
log.info("step.complete", step="script_synthesis", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 4.5: DEDICATED VISUAL PROMPT GENERATION (BEFORE curation)
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="visual_prompts")
_step_start = time.time()

try:
    dedicated_visuals = llm.generate_visual_prompts(script)

    if dedicated_visuals and len(dedicated_visuals) >= 6:
        script['all_visual_scenes'] = dedicated_visuals
        log.info("visual_prompts.dedicated", count=len(dedicated_visuals))
    else:
        log.warning("visual_prompts.fallback", reason="insufficient_dedicated_prompts")

    segments_file = project_folder / "script_segments.json"
    segments_file.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding='utf-8')

except Exception as e:
    log.warning("visual_prompts.failed", error=str(e))

_step_duration = time.time() - _step_start
log.info("step.complete", step="visual_prompts", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 4.7: SCRIPT CURATION — LangChain with raw LLM fallback
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="script_curation")
_step_start = time.time()

try:
    original_text = full_script
    log.info("curation.original", words=len(original_text.split()))

    curated_text = None

    # ── Try LangChain curation chain first ──
    if _USE_LANGCHAIN:
        try:
            from src.brain.chains.curation import CurationChain
            curation_chain_obj = CurationChain()
            story_bodies = []
            for story in script.get('stories', []):
                p1 = story.get('part_1_narration', '')
                p2 = story.get('part_2_narration', '')
                rt = story.get('real_talk', '')
                story_bodies.append(f"{p1} {p2} {rt}".strip())

            body_text = "\n\n---\n\n".join(
                f"[STORY {i+1}]\n{body}" for i, body in enumerate(story_bodies)
            )

            chain_result = curation_chain_obj.curate_stories(story_bodies)
            if chain_result and isinstance(chain_result, str):
                curated_text = chain_result
                log.info("curation.langchain.success")
        except Exception as e:
            log.warning("curation.langchain.failed", error=str(e))

    # ── Fallback to raw LLM curation ──
    if not curated_text:
        curated_text = llm.curate_script(script)

    if curated_text and curated_text != original_text:
        script['full_text'] = curated_text
        full_script = curated_text
        script['word_count'] = len(curated_text.split())
        script['estimated_duration'] = int(len(curated_text.split()) / 2.5)

        curated_file = project_folder / "script_curated.txt"
        curated_file.write_text(curated_text, encoding='utf-8')

        log.info("curation.complete", words=script['word_count'])
        _save_checkpoint("script_curation", project_folder,
                         {"curated_word_count": len(curated_text.split())})
    else:
        log.warning("curation.unchanged", reason="curation_returned_original")

except Exception as e:
    log.warning("curation.failed", error=str(e))

_step_duration = time.time() - _step_start
log.info("step.complete", step="script_curation", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: PIXEL ART GENERATION (2 per story = 6 total)
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="pixel_art")
_step_start = time.time()

try:
    generated_images = []
    image_folder = project_folder / "images"
    image_folder.mkdir(exist_ok=True)

    if SKIP_IMAGES:
        from PIL import Image as PILImage
        scene_names = []
        for i in range(3):
            scene_names.append(f'story_{i+1}_part1')
            scene_names.append(f'story_{i+1}_part2')

        for scene_name in scene_names:
            placeholder = PILImage.new('RGB', (1088, 1152), (10, 5, 25))
            placeholder_path = image_folder / f"{scene_name}_placeholder.png"
            placeholder.save(str(placeholder_path))
            generated_images.append(str(placeholder_path))

        log.warning("pixel_art.skip_images", placeholders=len(generated_images))

    else:
        import shutil
        from PIL import Image as PILImage

        base_seed = project_id % (2 ** 32)
        log.info("pixel_art.batch_seed", seed=base_seed)

        all_visual_scenes = script.get('all_visual_scenes', [])

        # Build script-aware fallback prompts for missing scenes
        stories = script.get('stories', [])
        for s_idx in range(3):
            for p_idx in range(2):
                fallback_idx = s_idx * 2 + p_idx
                if fallback_idx >= len(all_visual_scenes):
                    story = stories[s_idx] if s_idx < len(stories) else {}
                    part_key = f'part_{p_idx+1}_narration'
                    part_text = story.get(part_key, story.get('body', ''))
                    fallback_desc = _build_fallback_prompt(part_text, s_idx, p_idx, news_analyses)
                    all_visual_scenes.append({
                        'scene': f'story_{s_idx+1}_part{p_idx+1}',
                        'description': fallback_desc
                    })
        all_visual_scenes = all_visual_scenes[:6]

        scene_names = []
        for i in range(3):
            scene_names.append(f'story_{i+1}_part1')
            scene_names.append(f'story_{i+1}_part2')

        for scene_idx, scene_name in enumerate(scene_names):
            log.debug("pixel_art.generating", scene=scene_name)

            scene_data = all_visual_scenes[scene_idx]
            prompt = scene_data.get('description', '')

            style_suffix = ('Retro Pixel, (true 16-bit pixel art:1.5), (retro SNES style:1.3), '
                            'isometric perspective, (hard pixel edges:1.2), limited color palette, '
                            'detailed proportions, flat colors, dramatic lighting')
            full_prompt = f"{prompt}, {style_suffix}" if prompt else style_suffix

            story_idx = scene_idx // 2
            story_text = ''
            if story_idx < len(script.get('stories', [])):
                story = script['stories'][story_idx]
                part_key = 'part_1_narration' if scene_idx % 2 == 0 else 'part_2_narration'
                story_text = story.get(part_key, '')
                rt = story.get('real_talk', '')
                if scene_idx % 2 == 1 and rt:
                    story_text = f"{story_text} {rt}".strip()

            scene_seed = base_seed + scene_idx
            art_result = generate_pixel_art(full_prompt, script_text=story_text, seed=scene_seed)

            if art_result.get('success'):
                src_path = Path(art_result.get('path'))
                dst_filename = f"{scene_name}_{src_path.name}"
                dst_path = image_folder / dst_filename
                shutil.copy2(src_path, dst_path)
                generated_images.append(str(dst_path))
                log.debug("pixel_art.success", file=dst_filename)
            else:
                log.warning("pixel_art.primary_failed", scene=scene_name, action="retry")
                fallback_desc = _build_fallback_prompt(story_text, story_idx, scene_idx % 2, news_analyses)
                fallback_prompt = f"{fallback_desc}, {style_suffix}"
                retry_seed = base_seed + scene_idx + 100
                art_result = generate_pixel_art(fallback_prompt, script_text=story_text, seed=retry_seed)

                if art_result.get('success'):
                    src_path = Path(art_result.get('path'))
                    dst_filename = f"{scene_name}_{src_path.name}"
                    dst_path = image_folder / dst_filename
                    shutil.copy2(src_path, dst_path)
                    generated_images.append(str(dst_path))
                    log.debug("pixel_art.retry_success", file=dst_filename)
                else:
                    placeholder = PILImage.new('RGB', (1088, 1152), (10, 5, 25))
                    placeholder_path = image_folder / f"{scene_name}_placeholder.png"
                    placeholder.save(str(placeholder_path))
                    generated_images.append(str(placeholder_path))
                    log.warning("pixel_art.placeholder", scene=scene_name)

        # Final validation: ensure exactly 6 images
        if len(generated_images) < 6:
            log.warning("pixel_art.few_images", count=len(generated_images), needed=6)
            while len(generated_images) < 6:
                generated_images.append(generated_images[-1])

        log.info("pixel_art.complete", images=len(generated_images))
        _save_checkpoint("pixel_art", project_folder, {"image_count": len(generated_images)})
        _save_to_postgres("pixel_art", project_id, {"image_count": len(generated_images)})

except Exception as e:
    log.error("pixel_art.failed", error=str(e))
    import traceback
    traceback.print_exc()
    generated_images = []

_step_duration = time.time() - _step_start
log.info("step.complete", step="pixel_art", duration_s=round(_step_duration, 2),
         images=len(generated_images))


# ══════════════════════════════════════════════════════════════════════════
# STEP 6: VIDEO FOOTAGE (SKIP — using pixel art only)
# ══════════════════════════════════════════════════════════════════════════
log.info("step.skip", step="video_footage", reason="using_pixel_art_only")
downloaded_files = []


# ══════════════════════════════════════════════════════════════════════════
# STEP 7: VOICE GENERATION
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="voice_generation")
_step_start = time.time()

voice_file = None
tts_result = {}
try:
    tts_result = generate_voiceover(full_script, "authoritative")

    if tts_result.get('success'):
        import shutil
        src_audio = Path(tts_result.get('path'))
        dst_audio = project_folder / "voiceover.mp3"
        shutil.copy2(src_audio, dst_audio)

        log.info("voice.complete", duration_s=tts_result.get('estimated_duration_seconds', 0))
        voice_file = str(dst_audio)
        _save_checkpoint("tts", project_folder,
                         {"voice_duration": tts_result.get('estimated_duration_seconds', 0)})
        _save_to_postgres("tts", project_id,
                          {"voice_duration": tts_result.get('estimated_duration_seconds', 0)})
    else:
        log.error("voice.failed", reason="tts_returned_failure")

except Exception as e:
    log.error("voice.exception", error=str(e))

_step_duration = time.time() - _step_start
log.info("step.complete", step="voice_generation", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 8: VIDEO ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="video_assembly")
_step_start = time.time()

if not voice_file or not generated_images:
    log.error("assembly.missing_assets", has_voice=bool(voice_file), has_images=bool(generated_images))
    exit(1)

try:
    video_filename = f"video_{project_id}.mp4"
    video_output_path = str(project_folder / video_filename)

    word_timestamps = tts_result.get('word_timestamps', [])

    if word_timestamps:
        total_dur = word_timestamps[-1].get('end', tts_result.get('estimated_duration_seconds', 90))
    else:
        total_dur = tts_result.get('estimated_duration_seconds', 90)
    log.info("assembly.audio_duration", duration_s=round(total_dur, 1))

    scene_timestamps = None
    segment_timeline = script.get('segment_timeline', [])

    if segment_timeline and word_timestamps:
        num_images = len(generated_images)
        image_times = [{'start': None, 'end': None} for _ in range(num_images)]

        # ── Strip dead segments (greeting/intro_hook removed from TTS) ──
        stripped_greeting = script.get('greeting', '')
        cleaned_timeline = []
        for seg in segment_timeline:
            seg_text = seg.get('text', '').strip()
            seg_label = seg.get('label', '')
            if seg_label == 'intro' and stripped_greeting and stripped_greeting[:20].lower() in seg_text.lower():
                continue
            if seg_label == 'intro_pause' and stripped_greeting:
                continue
            cleaned_timeline.append(seg)
        segment_timeline = cleaned_timeline
        log.debug("assembly.timeline_cleaned", active_segments=len(segment_timeline))

        def _fuzzy_find_segment(seg_text, word_timestamps):
            """Find the best matching position for segment text in word timestamps."""
            seg_words = seg_text.lower().split()
            seg_clean = [w.strip(".,!?;:'\"()-") for w in seg_words if len(w.strip(".,!?;:'\"()-")) > 1]
            if len(seg_clean) < 2:
                return None, None
            wt_clean = [wt.get('word', '').lower().strip(".,!?;:'\"()-") for wt in word_timestamps]
            best_match = None
            best_match_len = 0
            search_words = seg_clean[:8]
            for start_i in range(len(wt_clean)):
                match_count = 0
                seg_i = 0
                wt_i = start_i
                skips = 0
                while seg_i < len(search_words) and wt_i < len(wt_clean) and skips <= 3:
                    if wt_clean[wt_i] == search_words[seg_i] or (
                        len(wt_clean[wt_i]) > 2 and len(search_words[seg_i]) > 2 and
                        (wt_clean[wt_i].startswith(search_words[seg_i][:4]) or
                         search_words[seg_i].startswith(wt_clean[wt_i][:4]))
                    ):
                        match_count += 1
                        seg_i += 1
                        wt_i += 1
                        skips = 0
                    else:
                        skips += 1
                        wt_i += 1
                if match_count > best_match_len and match_count >= 2:
                    best_match_len = match_count
                    end_wt_i = min(start_i + len(seg_clean) + 2, len(word_timestamps) - 1)
                    best_match = (
                        word_timestamps[start_i].get('start', 0),
                        word_timestamps[end_wt_i].get('end', word_timestamps[start_i].get('start', 0) + 5)
                    )
            return best_match if best_match else (None, None)

        for seg in segment_timeline:
            img_idx = seg.get('image_idx', 0)
            if img_idx >= num_images:
                continue
            seg_text = seg.get('text', '').strip()
            if not seg_text or seg_text in ('....', '...', '..'):
                continue
            seg_start, seg_end = _fuzzy_find_segment(seg_text, word_timestamps)
            if seg_start is not None:
                if image_times[img_idx]['start'] is None or seg_start < image_times[img_idx]['start']:
                    image_times[img_idx]['start'] = seg_start
                if seg_end is not None and (image_times[img_idx]['end'] is None or seg_end > image_times[img_idx]['end']):
                    image_times[img_idx]['end'] = seg_end
            else:
                log.debug("assembly.no_match", image_idx=img_idx, text=seg_text[:40])

        # Fill gaps with proportional split
        for i, it in enumerate(image_times):
            if it['start'] is None:
                it['start'] = (total_dur / num_images) * i
            if it['end'] is None:
                if i + 1 < num_images and image_times[i + 1]['start'] is not None:
                    it['end'] = image_times[i + 1]['start']
                else:
                    it['end'] = (total_dur / num_images) * (i + 1)

        if image_times:
            image_times[-1]['end'] = max(image_times[-1]['end'], total_dur)
            image_times[0]['start'] = 0

        # Bridge gaps (no black frames)
        for i in range(len(image_times) - 1):
            gap = image_times[i + 1]['start'] - image_times[i]['end']
            if gap > 0.1:
                split_point = image_times[i]['end'] + gap * 0.7
                image_times[i]['end'] = split_point
                image_times[i + 1]['start'] = split_point

        # Per-story image balancing (35-65%)
        num_stories = num_images // 2
        for story_i in range(num_stories):
            img_a = story_i * 2
            img_b = story_i * 2 + 1
            if img_b >= num_images:
                break
            story_start = image_times[img_a]['start']
            story_end = image_times[img_b]['end']
            story_total = story_end - story_start
            if story_total <= 0:
                continue
            dur_a = image_times[img_a]['end'] - image_times[img_a]['start']
            ratio_a = dur_a / story_total if story_total > 0 else 0.5
            if not (0.35 <= ratio_a <= 0.65):
                target_split = story_start + story_total * 0.5
                min_img_dur = story_total * 0.35
                max_img_dur = story_total * 0.65
                target_split = max(story_start + min_img_dur,
                                   min(target_split, story_start + max_img_dur))
                image_times[img_a]['end'] = target_split
                image_times[img_b]['start'] = target_split
                new_dur_a = target_split - image_times[img_a]['start']
                new_ratio_a = new_dur_a / story_total
                log.debug("assembly.rebalanced", story=story_i+1,
                          img_a_dur=f"{new_dur_a:.1f}s", ratio=f"{new_ratio_a:.0%}")

        # Safety: ensure minimum 2s per image
        for i, it in enumerate(image_times):
            dur = it['end'] - it['start']
            if dur < 2.0:
                needed = 2.0 - dur
                pair_idx = i - 1 if i % 2 == 1 else i + 1
                if 0 <= pair_idx < num_images:
                    pair_dur = image_times[pair_idx]['end'] - image_times[pair_idx]['start']
                    if pair_dur > 4.0:
                        steal = min(needed, (pair_dur - 2.0) / 2)
                        if steal > 0:
                            if pair_idx < i:
                                image_times[pair_idx]['end'] -= steal
                                it['start'] -= steal
                            else:
                                image_times[pair_idx]['start'] += steal
                                it['end'] += steal

        # Pre-roll offset: images appear ~1s before narration
        PREROLL_OFFSET = 0.3
        for i in range(1, len(image_times)):
            new_start = max(0, image_times[i]['start'] - PREROLL_OFFSET)
            if i > 0 and new_start < image_times[i - 1]['start']:
                new_start = image_times[i - 1]['start']
            image_times[i]['start'] = new_start

        scene_timestamps = image_times
        for i, ts in enumerate(scene_timestamps):
            dur = ts['end'] - ts['start']
            log.debug("assembly.scene_timestamp", image=i,
                      start=f"{ts['start']:.2f}", end=f"{ts['end']:.2f}", duration=f"{dur:.2f}s")
    else:
        log.warning("assembly.no_timeline", reason="missing segment_timeline or word_timestamps")

    # Title from most important story
    last_analysis = news_analyses[-1] if news_analyses else {}
    hook_text = last_analysis.get('topic', '') or last_analysis.get('angle', '')
    if not hook_text:
        hooks = [s.get('part_1_narration', s.get('mini_hook', '')) for s in script.get('stories', [])]
        hook_text = hooks[-1] if hooks else ''
    log.info("assembly.hook_text", text=hook_text[:60])

    # Hook card: DISABLED — cut straight to content
    hook_card_text = None

    assembly_result = build_split_video(
        audio_path=voice_file,
        image_paths=generated_images,
        output_path=video_output_path,
        script_text=full_script,
        word_timestamps=word_timestamps,
        hook_text=hook_text,
        scene_timestamps=scene_timestamps,
        hook_card_text=hook_card_text,
    )

    if assembly_result.get('success'):
        final_video_path = assembly_result.get('path')
        log.info("assembly.complete",
                 video=final_video_path,
                 duration_s=assembly_result.get('duration_seconds'),
                 size_mb=assembly_result.get('file_size_mb'),
                 resolution=assembly_result.get('resolution'))
    else:
        log.error("assembly.failed", error=assembly_result.get('error'))
        final_video_path = None

    if final_video_path:
        _save_checkpoint("video_assembly", project_folder, {"final_video": final_video_path})
        _save_to_postgres("video_assembly", project_id, {
            "video_path": final_video_path,
            "duration": assembly_result.get('duration_seconds'),
        })

except Exception as e:
    log.error("assembly.exception", error=str(e))
    import traceback
    traceback.print_exc()
    final_video_path = None

_step_duration = time.time() - _step_start
log.info("step.complete", step="video_assembly", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 9: PLATFORM METADATA
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="platform_metadata")
_step_start = time.time()

platform_metadata = {}
try:
    from src.collector.platform_metadata_generator import PlatformMetadataGenerator
    metadata_gen = PlatformMetadataGenerator()
    platform_metadata = metadata_gen.generate_all_metadata(
        script, news_analyses[0] if news_analyses else {}, None
    )
    metadata_path = project_folder / 'platform_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(platform_metadata, f, indent=2, ensure_ascii=False)
    log.info("platform_metadata.complete")
except Exception as e:
    log.warning("platform_metadata.failed", error=str(e))

_step_duration = time.time() - _step_start
log.info("step.complete", step="platform_metadata", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 10: PROJECT SUMMARY + TELEGRAM DELIVERY
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="project_summary")

manifest = {
    'project_id': project_id,
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'format': 'multi_news_3',
    'pipeline_version': 'unified',
    'articles': [
        {
            'title': a.get('title'),
            'url': a.get('link'),
            'source': a.get('feed_name')
        }
        for a in articles
    ],
    'analyses': [
        {
            'topic': a.get('topic'),
            'impact_score': a.get('impact_score'),
            'shift_vector': a.get('shift_vector')
        }
        for a in news_analyses
    ],
    'script': script,
    'platform_metadata': platform_metadata,
    'assets': {
        'images': [str(Path(p).name) for p in generated_images],
        'voiceover': 'voiceover.mp3',
        'video': video_filename if final_video_path else None
    },
    'tts': {
        'word_timestamps': word_timestamps,
        'engine': tts_result.get('engine', 'unknown'),
        'estimated_duration_seconds': tts_result.get('estimated_duration_seconds', 0),
    },
    'project_folder': str(project_folder)
}

manifest_path = project_folder / "manifest.json"
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
log.info("manifest.saved", path=str(manifest_path))

# ── Category tracking ──
try:
    from src.collector.category_rotation import CategoryRotation
    rotation = CategoryRotation()
    for idx, article in enumerate(articles):
        matched_categories = rotation.detect_article_categories(article)
        category = matched_categories[0] if matched_categories else "other"
        region = rotation._detect_region_from_title(article.get('title', ''))
        analysis = news_analyses[idx] if idx < len(news_analyses) else {}
        rotation.track_video_generated(
            topic=analysis.get('topic', 'Unknown'),
            category=category,
            region=region,
            article_title=article.get('title', 'Unknown')
        )
        log.debug("category.tracked", story=idx+1, category=category, region=region)
except Exception as e:
    log.warning("category_tracking.failed", error=str(e))

# ── PostgreSQL final status ──
_video_topic = news_analyses[0].get('topic', '') if news_analyses else ''
_save_to_postgres("completed", project_id, {"manifest_path": str(manifest_path)}, topic=_video_topic or None)

# ── Store topic vectors for future dedup ──
try:
    from src.brain.memory.vector_store import VectorStore
    store = VectorStore()
    for i, analysis in enumerate(news_analyses):
        topic = analysis.get('topic', '')
        category = analysis.get('category', 'general')
        if topic:
            store.store_embedding_from_text(
                project_id=str(project_id),
                topic=topic,
                category=category,
            )
    log.info("vector_topics.stored", count=len(news_analyses))
except Exception as e:
    log.warning("vector_topics.failed", error=str(e))

# ── Telegram delivery ──
if final_video_path and not args.no_telegram:
    try:
        from tools.telegram_sender import send_video_to_telegram
        today_str = datetime.now().strftime('%b %d, %Y')
        caption = f"📹 The Mask Daily News — {today_str}"
        result = send_video_to_telegram(video_path=final_video_path, caption=caption)
        success = result.get("success", False)
        if success:
            log.info("telegram.sent", video=final_video_path)
        else:
            log.warning("telegram.send_failed")
    except Exception as e:
        log.warning("telegram.failed", error=str(e))


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETE
# ══════════════════════════════════════════════════════════════════════════
log.info("pipeline.complete",
         project_id=project_id,
         video=final_video_path or "FAILED",
         duration_s=assembly_result.get('duration_seconds') if final_video_path else 0,
         images=len(generated_images),
         words=script.get('word_count', 0))