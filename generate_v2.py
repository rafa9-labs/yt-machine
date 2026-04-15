"""
════════════════════════════════════════════════════════════════════════════════
generate_v2.py — Phase 7+8: Modern Stack Pipeline (with automatic fallback)
════════════════════════════════════════════════════════════════════════════════

SAFETY DESIGN:
  This file runs ALONGSIDE generate_complete_video.py (the old pipeline).
  It tries new modules first, but falls back to old ones on any failure.
  
  Toggle via .env:  PIPELINE_VERSION=v2   (this file)
                    PIPELINE_VERSION=v1   (old file — instant rollback)

  Every new module call is wrapped in try/except with old-module fallback.

WHAT'S NEW vs generate_complete_video.py:
  Step 1:   AsyncRSSScraper (parallel fetching + JS rendering)
  Step 1.5: Vector dedup check (pgvector semantic similarity)
  Step 2:   LangChain chains for news analysis (when available)
  Step 5+:  PostgreSQL save after each step (not just JSON files)
  Logging:  structlog — structured, leveled, timestamped logs (Phase 8)
  All else: Falls back to old generate_complete_video.py modules
════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════
# PHASE 8: STRUCTURED LOGGING — replaces _TeeWriter + print() hack
# ══════════════════════════════════════════════════════════════════════════
# WHY STRUCTLOG? The old pipeline used:
#   sys.stdout = _TeeWriter(sys.stdout, log_path)   ← monkey-patches stdout!
#   print("✅ Async scraper found 47 articles")      ← no level, no structure
#
# structlog gives you:
#   log.info("step.complete", step="news_fetch", duration_s=2.3, articles=47)
#   → Structured, leveled, timestamped, grep-able, machine-parseable
#
# SAFETY: If structlog fails to import, brain/log.py falls back to
# standard logging. Your pipeline NEVER crashes from a logging issue.

from brain.log import get_logger

log = get_logger("pipeline")

# ── CONFIGURATION ──
SKIP_IMAGES = os.environ.get("SKIP_IMAGES", "0") == "1"

# ── ARGUMENT PARSING ──
import argparse
parser = argparse.ArgumentParser(description='YT-MACHINE V2 Pipeline (Modern Stack)')
parser.add_argument('--resume', type=str, default=None, metavar='PROJECT_FOLDER',
                    help='Resume from project folder')
parser.add_argument('--skip-images', action='store_true', help='Use placeholder images')
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
        log.info("pipeline.resume", path=str(resume_path))
    else:
        log.warning("pipeline.resume.no_checkpoint", path=str(resume_path))

log.info("pipeline.start", pipeline_version="v2", skip_images=SKIP_IMAGES)


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


from pipeline_utils import bridge_timestamp_gaps, build_fallback_prompt as _build_fallback_prompt


# ══════════════════════════════════════════════════════════════════════════
# LLM INITIALIZATION — Try LangChain first, fallback to raw interface
# ══════════════════════════════════════════════════════════════════════════
_USE_LANGCHAIN = True

try:
    from brain.langchain_interface import LangChainInterface
    _langchain = LangChainInterface()
    log.info("llm.langchain.loaded")
except Exception as e:
    log.warning("llm.langchain.failed", error=str(e), fallback="raw_llm")
    _USE_LANGCHAIN = False

# Always load old interface as fallback — it handles complex methods
from brain.llm_interface import LLMInterface
llm = LLMInterface()
log.info("llm.raw.loaded")

# Video server components (unchanged)
from video_server.pixel_art_tool import generate_pixel_art
from video_server.pexels_tool import fetch_vertical_footage
from video_server.tts_tool import generate_voiceover
from video_server.split_video_assembler import build_split_video


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
    existing['pipeline_version'] = 'v2'
    if data:
        existing.update(data)
    with open(cp_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False, default=str)


# ── PostgreSQL SAVE HELPER ──
def _save_to_postgres(step_name: str, project_id: int, data: dict):
    """Save pipeline step results to PostgreSQL. Non-blocking — errors are logged, not raised."""
    try:
        from db.connection import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO videos (project_id, status, created_at, updated_at)
            VALUES (%s, %s, NOW(), NOW())
            ON CONFLICT (project_id) DO UPDATE SET
                status = EXCLUDED.status,
                updated_at = NOW()
        """, (project_id, step_name))
        
        cursor.execute("""
            UPDATE videos SET
                metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
            WHERE project_id = %s
        """, (json.dumps({step_name: data}, default=str), project_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        log.debug("postgres.saved", step=step_name, project_id=project_id)
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

# ── Bind project context to logger — all future logs include project_id ──
# WHY bind()? Every log line from this point includes project_id automatically.
# No need to pass it to every log.info() call manually.
log = log.bind(project_id=project_id)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: FETCH LATEST NEWS — Async Scraper (NEW) with old fallback
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="news_fetch")
_step_start = time.time()

articles = []
all_articles = []
try:
    from redfish.async_scraper import AsyncRSScraper
    
    async def _fetch_articles():
        async_scraper = AsyncRSScraper()
        results = await async_scraper.fetch_all_feeds(max_age_hours=24)
        return results
    
    all_articles = asyncio.run(_fetch_articles())
    log.info("scraper.async.success", articles=len(all_articles))
    
    from redfish.rss_scraper import RSScraper
    ranker = RSScraper()
    viral_articles = ranker.filter_viral_potential(all_articles, top_n=10)
    
    _save_to_postgres("news_fetch", project_id, {
        "article_count": len(all_articles),
        "scraper": "async_v2",
    })
    
except Exception as e:
    log.warning("scraper.async.failed", error=str(e), fallback="sync_scraper")
    try:
        from redfish.rss_scraper import RSScraper
        scraper = RSScraper()
        all_articles = scraper.scrape_all(max_age_hours=24)
        viral_articles = scraper.filter_viral_potential(all_articles, top_n=10)
        log.info("scraper.sync.fallback", articles=len(all_articles))
    except Exception as e2:
        log.error("scraper.all_failed", error=str(e2))
        exit(1)

_step_duration = time.time() - _step_start
log.info("step.complete", step="news_fetch", duration_s=round(_step_duration, 2), articles=len(all_articles))

if len(viral_articles) < 3:
    log.warning("articles.few", count=len(viral_articles), minimum=3)

# ── Topic diversity selection (same logic as old pipeline) ──
selected = []
seen_topic_words = []
for article in viral_articles:
    title_lower = article.get('title', '').lower()
    topic_words = frozenset(title_lower.split()[:5])
    is_duplicate = False
    for seen in seen_topic_words:
        overlap = len(topic_words & seen)
        if overlap >= 3:
            is_duplicate = True
            break
    if not is_duplicate:
        selected.append(article)
        seen_topic_words.append(topic_words)
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

_save_checkpoint("news_fetch", project_folder, {"article_titles": [a.get('title','') for a in articles]})


# ══════════════════════════════════════════════════════════════════════════
# STEP 1.5: VECTOR DEDUP CHECK (NEW — no equivalent in old pipeline)
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="vector_dedup")
_step_start = time.time()

_dedup_available = False
try:
    from brain.memory.vector_store import VideoTopicStore
    from brain.memory.deduplication import TopicDeduplicator
    
    dedup = TopicDeduplicator(similarity_threshold=0.35)
    _dedup_available = True
    log.info("dedup.loaded", backend="pgvector")
    
    deduped_articles = []
    for article in articles:
        topic_text = article.get('title', '')
        is_dup = dedup.is_duplicate(topic_text)
        if is_dup:
            log.info("dedup.skip_duplicate", topic=topic_text[:60])
        else:
            deduped_articles.append(article)
    
    if len(deduped_articles) < 3:
        log.warning("dedup.few_unique", count=len(deduped_articles))
        for a in viral_articles:
            if a not in deduped_articles and len(deduped_articles) < 3:
                topic_text = a.get('title', '')
                if not dedup.is_duplicate(topic_text):
                    deduped_articles.append(a)
    
    articles = deduped_articles[:3]
    log.info("dedup.complete", unique_articles=len(articles))
    
except Exception as e:
    log.warning("dedup.failed", error=str(e), action="continuing_without_dedup")

_step_duration = time.time() - _step_start
log.info("step.complete", step="vector_dedup", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: NEWS ANALYSIS — LangChain chain (NEW) with old fallback
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="news_analysis")
_step_start = time.time()

news_analyses = []
for i, article in enumerate(articles, 1):
    try:
        # Get full article text
        try:
            from redfish.rss_scraper import RSScraper
            text_scraper = RSScraper()
            article_text = text_scraper.get_full_article_text(article)
        except:
            article_text = article.get('summary', article.get('title', ''))
        
        log.debug("analysis.article", index=i, chars=len(article_text))
        
        analysis = None
        
        # ── Try LangChain structured chain first ──
        if _USE_LANGCHAIN:
            try:
                from models.schemas import NewsAnalysis as NewsAnalysisModel
                chain = _langchain.build_structured_chain(NewsAnalysisModel, "news_processor")
                result = chain.invoke({"input_text": f"Analyze this news article:\n\n{article_text}"})
                analysis = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
                log.info("analysis.langchain.success", index=i)
            except Exception as e:
                log.warning("analysis.langchain.failed", index=i, error=str(e))
        
        # ── Fallback to old raw LLM interface ──
        if not analysis:
            analysis = llm.process_news(article_text)
        
        if analysis:
            log.info("analysis.complete", index=i, topic=analysis.get('topic', 'N/A')[:60], impact=analysis.get('impact_score', 0))
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
    log.info("analysis.ranked", rank=i+1, impact=a.get('impact_score', '?'), topic=a.get('topic', 'N/A')[:50])

_step_duration = time.time() - _step_start
log.info("step.complete", step="news_analysis", duration_s=round(_step_duration, 2))

_save_checkpoint("news_analysis", project_folder, {"analysis_topics": [a.get('topic','') for a in news_analyses]})
_save_to_postgres("news_analysis", project_id, {
    "topics": [a.get('topic', '') for a in news_analyses],
    "impact_scores": [a.get('impact_score', 0) for a in news_analyses],
})


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: TRENDING CONTEXT (unchanged — uses old module)
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="trending_context")
_step_start = time.time()

trending_context = {}
try:
    from redfish.trending_analyzer import TrendingAnalyzer
    trending_analyzer = TrendingAnalyzer()
    trending_context = trending_analyzer.analyze(all_articles, top_n=40)
    log.info("trending.complete", terms=len(trending_context))
except Exception as e:
    log.warning("trending.failed", error=str(e))

_step_duration = time.time() - _step_start
log.info("step.complete", step="trending_context", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: SCRIPT SYNTHESIS (uses old LLM — complex 200-line method)
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
    
    # Strip greeting and intro_hook
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
    
    log.info("script.synthesized", duration_s=script.get('estimated_duration', 0), words=script.get('word_count', 0))
    for i, story in enumerate(script.get('stories', []), 1):
        log.debug("script.story", index=i, hook=story.get('mini_hook', 'N/A')[:60])
    
    # Save script files
    script_file = project_folder / "script.txt"
    script_file.write_text(full_script, encoding='utf-8')
    
    segments_file = project_folder / "script_segments.json"
    segments_file.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding='utf-8')
    
    _save_checkpoint("script_synthesis", project_folder, {"script_word_count": script.get('word_count', 0)})
    _save_to_postgres("script_synthesis", project_id, {"word_count": script.get('word_count', 0)})
    
except Exception as e:
    log.error("script.synthesis.exception", error=str(e))
    import traceback
    traceback.print_exc()
    exit(1)

_step_duration = time.time() - _step_start
log.info("step.complete", step="script_synthesis", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 4.5: VISUAL PROMPT GENERATION (unchanged)
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
# STEP 4.7: SCRIPT CURATION — Try LangChain chain (NEW), fallback to old
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="script_curation")
_step_start = time.time()

try:
    original_text = full_script
    log.info("curation.original", words=len(original_text.split()))
    
    curated_text = None
    if _USE_LANGCHAIN:
        try:
            from brain.chains.curation import create_curation_chain
            curation_chain = create_curation_chain()
            story_bodies = []
            for story in script.get('stories', []):
                p1 = story.get('part_1_narration', '')
                p2 = story.get('part_2_narration', '')
                story_bodies.append(f"{p1} {p2}".strip())
            
            body_text = "\n\n---\n\n".join(
                f"[STORY {i+1}]\n{body}" for i, body in enumerate(story_bodies)
            )
            
            chain_result = curation_chain.invoke({"input_text": body_text})
            if chain_result and isinstance(chain_result, str):
                curated_text = chain_result
                log.info("curation.langchain.success")
        except Exception as e:
            log.warning("curation.langchain.failed", error=str(e))
    
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
        _save_checkpoint("script_curation", project_folder, {"curated_word_count": len(curated_text.split())})
    else:
        log.warning("curation.unchanged", reason="curation_returned_original")
except Exception as e:
    log.warning("curation.failed", error=str(e))

_step_duration = time.time() - _step_start
log.info("step.complete", step="script_curation", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: PIXEL ART GENERATION (unchanged)
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
        base_seed = project_id % (2**32)
        all_visual_scenes = script.get('all_visual_scenes', [])
        
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

            scene_seed = base_seed + scene_idx
            art_result = generate_pixel_art(full_prompt, script_text=story_text, seed=scene_seed)
            
            if art_result.get('success'):
                import shutil
                src_path = Path(art_result.get('path'))
                dst_filename = f"{scene_name}_{src_path.name}"
                dst_path = image_folder / dst_filename
                shutil.copy2(src_path, dst_path)
                generated_images.append(str(dst_path))
                log.debug("pixel_art.success", file=dst_filename)
            else:
                log.warning("pixel_art.primary_failed", scene=scene_name, action="retry")
                import shutil
                from PIL import Image as PILImage
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

        if len(generated_images) < 6:
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
log.info("step.complete", step="pixel_art", duration_s=round(_step_duration, 2), images=len(generated_images))


# ══════════════════════════════════════════════════════════════════════════
# STEP 6: VIDEO FOOTAGE (SKIP — unchanged)
# ══════════════════════════════════════════════════════════════════════════
log.info("step.skip", step="video_footage", reason="using_pixel_art_only")
downloaded_files = []


# ══════════════════════════════════════════════════════════════════════════
# STEP 7: VOICE GENERATION (unchanged)
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
        _save_checkpoint("tts", project_folder, {"voice_duration": tts_result.get('estimated_duration_seconds', 0)})
        _save_to_postgres("tts", project_id, {"voice_duration": tts_result.get('estimated_duration_seconds', 0)})
    else:
        log.error("voice.failed", reason="tts_returned_failure")
except Exception as e:
    log.error("voice.exception", error=str(e))

_step_duration = time.time() - _step_start
log.info("step.complete", step="voice_generation", duration_s=round(_step_duration, 2))


# ══════════════════════════════════════════════════════════════════════════
# STEP 8: VIDEO ASSEMBLY (unchanged — complex timeline logic)
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
        
        def _fuzzy_find_segment(seg_text, word_timestamps):
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
        
        for i, it in enumerate(image_times):
            if it['start'] is None:
                it['start'] = (total_dur / num_images) * i
            if it['end'] is None:
                it['end'] = (total_dur / num_images) * (i + 1)
        
        if image_times:
            image_times[-1]['end'] = max(image_times[-1]['end'], total_dur)
            image_times[0]['start'] = 0
        
        for i in range(len(image_times) - 1):
            gap = image_times[i + 1]['start'] - image_times[i]['end']
            if gap > 0.1:
                split_point = image_times[i]['end'] + gap * 0.7
                image_times[i]['end'] = split_point
                image_times[i + 1]['start'] = split_point
        
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
                image_times[img_a]['end'] = target_split
                image_times[img_b]['start'] = target_split
        
        PREROLL_OFFSET = 1.0
        for i in range(1, len(image_times)):
            new_start = max(0, image_times[i]['start'] - PREROLL_OFFSET)
            if i > 0 and new_start < image_times[i-1]['start']:
                new_start = image_times[i-1]['start']
            image_times[i]['start'] = new_start
        
        scene_timestamps = image_times
    
    last_analysis = news_analyses[-1] if news_analyses else {}
    hook_text = last_analysis.get('topic', '') or last_analysis.get('angle', '')
    if not hook_text:
        hooks = [s.get('part_1_narration', s.get('mini_hook', '')) for s in script.get('stories', [])]
        hook_text = hooks[-1] if hooks else ''
    
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
        log.info("assembly.complete", video=final_video_path, duration_s=assembly_result.get('duration_seconds'), size_mb=assembly_result.get('file_size_mb'))
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
# STEP 9: PLATFORM METADATA (unchanged)
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="platform_metadata")
_step_start = time.time()

platform_metadata = {}
try:
    from redfish.platform_metadata_generator import PlatformMetadataGenerator
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
# STEP 10: PROJECT SUMMARY + PostgreSQL save + Vector topic tracking
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="project_summary")

manifest = {
    'project_id': project_id,
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'format': 'multi_news_3',
    'pipeline_version': 'v2',
    'articles': [
        {'title': a.get('title'), 'url': a.get('link'), 'source': a.get('feed_name')}
        for a in articles
    ],
    'analyses': [
        {'topic': a.get('topic'), 'impact_score': a.get('impact_score'), 'shift_vector': a.get('shift_vector')}
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

# ── Track video for category rotation (unchanged) ──
try:
    from redfish.category_rotation import CategoryTracker
    tracker = CategoryTracker()
    topic_list = [a.get('topic', '') for a in news_analyses]
    tracker.record_topics(topic_list)
    log.debug("category_rotation.recorded", topics=topic_list)
except Exception as e:
    log.warning("category_rotation.failed", error=str(e))

# ── Update PostgreSQL status to completed ──
_save_to_postgres("completed", project_id, {"manifest_path": str(manifest_path)})

# ── Store topic vectors for future dedup (Phase 5 integration) ──
try:
    from brain.memory.vector_store import VideoTopicStore
    store = VideoTopicStore()
    for i, analysis in enumerate(news_analyses):
        topic = analysis.get('topic', '')
        if topic:
            store.store_topic(
                topic_text=topic,
                metadata={"project_id": project_id, "impact_score": analysis.get('impact_score', 0)}
            )
    log.info("vector_topics.stored", count=len(news_analyses))
except Exception as e:
    log.warning("vector_topics.failed", error=str(e))


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETE
# ══════════════════════════════════════════════════════════════════════════
log.info("pipeline.complete", 
    project_id=project_id,
    video=final_video_path or "FAILED",
    duration_s=assembly_result.get('duration_seconds') if final_video_path else 0,
    images=len(generated_images),
    words=script.get('word_count', 0),
)