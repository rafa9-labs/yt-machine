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

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
os.environ['PYTHONUNBUFFERED'] = '1'

import sys
import json
import time
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Fix Unicode encoding + force line buffering for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
else:
    sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

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
parser.add_argument('--dry-run', action='store_true',
                    help='Run pipeline with canned data (no API calls) to validate wiring')
args = parser.parse_args()
if args.skip_images:
    SKIP_IMAGES = True
DRY_RUN = args.dry_run
if DRY_RUN:
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


# ── DRY-RUN CANNED DATA ──
if DRY_RUN:
    log.info("[DRY-RUN] MODE ACTIVE — using canned data, no API calls")

    _DRY_RUN_ARTICLES = [
        {'title': 'Iran launches missile strike on Israeli military base',
         'summary': 'Iran fired dozens of missiles at an Israeli airbase in a retaliatory strike.',
         'link': 'https://example.com/1', 'published': '2025-01-15', 'source': 'test'},
        {'title': 'Pentagon confirms new AI defense contract with SpaceX',
         'summary': 'The US Department of Defense signed a $2B AI defense contract with SpaceX.',
         'link': 'https://example.com/2', 'published': '2025-01-15', 'source': 'test'},
        {'title': 'Global oil prices spike after Middle East tensions escalate',
         'summary': 'Brent crude jumped 8% as supply route disruptions spread across the region.',
         'link': 'https://example.com/3', 'published': '2025-01-15', 'source': 'test'},
    ]

    _DRY_RUN_ANALYSES = [
        {
            'topic': 'Iran missile strike on Israeli military base',
            'summary': 'Iran fired dozens of missiles at an Israeli airbase in retaliation.',
            'impact_score': 9,
            'geopolitical_angle': 'escalation',
            'first_order_effect': 'Immediate military escalation in the Middle East.',
            'second_order_consequence': 'Allied nations forced to choose sides, diplomatic channels freeze.',
            'third_order_effect': 'Global energy markets react as shipping routes are threatened.',
            'visual_category': 'warfare',
            'key_entities': ['Iran', 'Israel', 'Missile Defense'],
            'locations': ['Iran', 'Israel', 'Persian Gulf'],
            'emotions': ['shock', 'fear', 'urgency'],
        },
        {
            'topic': 'Pentagon AI defense contract with SpaceX',
            'summary': 'The US Department of Defense signed a $2B AI defense contract with SpaceX.',
            'impact_score': 7,
            'geopolitical_angle': 'technology_race',
            'first_order_effect': 'US military gains AI-powered surveillance and defense capabilities.',
            'second_order_consequence': 'Rival nations accelerate their own AI military programs.',
            'third_order_effect': 'Civilian privacy concerns mount as military AI expands.',
            'visual_category': 'arms_defense',
            'key_entities': ['Pentagon', 'SpaceX', 'AI'],
            'locations': ['United States', 'Pentagon'],
            'emotions': ['surprise', 'intrigue', 'concern'],
        },
    ]

    _DRY_RUN_SCRIPT = {
        'greeting': "Ssssmokin'!",
        'intro_hook': "Hold onto your lobsters, folks!",
        'full_text': "Ssssmokin'! Hold onto your lobsters, folks! Iran launched missiles at Israel yesterday. "
                     "The Pentagon confirmed multiple impacts across the region. This changes the regional power "
                     "dynamics entirely. The consequences will ripple for decades. But wait, there is more. The "
                     "Pentagon signed a two billion dollar AI defense contract with SpaceX. Global markets "
                     "reacted with oil prices spiking immediately. This is what happens when geopolitics meets "
                     "economics. The ripple effects are just beginning. Stay curious, stay critical, and remember "
                     "— the truth is always stranger than fiction. Subscribe for more Masker!",
        'word_count': 80,
        'estimated_duration': 32,
        'stories': [
            {
                'part_1_narration': 'Iran launched missiles at Israel yesterday.',
                'part_2_narration': 'The Pentagon confirmed multiple impacts across the region.',
                'real_talk': 'This changes the regional power dynamics entirely.',
                'fallout': 'The consequences will ripple for decades.',
                'segue': "But wait, there is more!",
                'part_1_visual': '16-bit isometric pixel art scene: dramatic wide establishing shot, missile trails in foreground, military installation in midground, sunset lighting',
                'part_2_visual': '16-bit isometric pixel art scene: tactical close-up view of radar installation, impact zones visible in background, golden hour lighting',
                'real_talk_visual': '16-bit isometric pixel art scene: somber revealing scene, civilian perspective on left, consequences visible in background, cold blue lighting',
                'fallout_visual': '16-bit isometric pixel art scene: forward-looking consequence scene, domino effect visible, dark horizon on left, twilight atmosphere',
            },
            {
                'part_1_narration': 'The Pentagon signed a two billion dollar AI defense contract with SpaceX.',
                'part_2_narration': 'Global markets reacted with oil prices spiking immediately.',
                'real_talk': 'This is what happens when geopolitics meets economics.',
                'fallout': 'The ripple effects are just beginning.',
                'segue': '',
                'part_1_visual': '16-bit isometric pixel art scene: dramatic wide establishing shot, Pentagon building in foreground, contract signing scene in midground, strategic lighting',
                'part_2_visual': '16-bit isometric pixel art scene: tactical close-up view of trading screens, oil price charts visible, red indicators in background, dramatic lighting',
                'real_talk_visual': '16-bit isometric pixel art scene: somber revealing scene, economic impact visible on left, global consequences in background, cold lighting',
                'fallout_visual': '16-bit isometric pixel art scene: forward-looking consequence scene, cascade effect spreading, ripple patterns in foreground, twilight atmosphere',
            },
        ],
        'closing': 'Stay curious, stay critical, and remember — the truth is always stranger than fiction. Subscribe for more Masker!',
    }


def _run_with_timeout(func, timeout_seconds, step_name, *args, **kwargs):
    """Run a function with a hard wall-clock timeout. Returns (result, timed_out)."""
    result_container = [None]
    exception_container = [None]

    def worker():
        try:
            result_container[0] = func(*args, **kwargs)
        except Exception as e:
            exception_container[0] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        log.error("step.hard_timeout", step=step_name, timeout_s=timeout_seconds)
        return None, True

    if exception_container[0] is not None:
        raise exception_container[0]

    return result_container[0], False


def _run_with_heartbeat(func, label: str, heartbeat_interval: int = 8, timeout_seconds: int = 600, *args, **kwargs):
    """Run a function in a background thread, printing heartbeat dots while waiting.

    Prints a dot every `heartbeat_interval` seconds so the user knows the process
    isn't frozen. Also prints VRAM status every 3rd heartbeat.
    Enforces `timeout_seconds` — if the function doesn't complete in time, returns
    (None, True) and abandons the thread (it's daemon, so it won't block exit).

    Returns:
        (result, timed_out) — same as _run_with_timeout
    """
    result_container = [None]
    exception_container = [None]
    done_event = threading.Event()

    def worker():
        try:
            result_container[0] = func(*args, **kwargs)
        except Exception as e:
            exception_container[0] = e
        finally:
            done_event.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    elapsed = 0
    dots = 0
    while not done_event.is_set():
        waited = done_event.wait(timeout=heartbeat_interval)
        if waited:
            break
        elapsed += heartbeat_interval
        dots += 1
        print(".", end="", flush=True)
        if dots % 3 == 0:
            try:
                orchestrator.heartbeat(f"{label} ({elapsed}s)")
            except Exception:
                print(f" ({elapsed}s)", end="", flush=True)
        if elapsed >= timeout_seconds:
            break

    if not done_event.is_set():
        print(f" TIMEOUT ({timeout_seconds}s)", flush=True)
        log.error("heartbeat.hard_timeout", label=label, timeout_s=timeout_seconds)
        return None, True

    if exception_container[0] is not None:
        print(f" FAIL", flush=True)
        raise exception_container[0]

    print(f" done ({elapsed}s)", flush=True)
    return result_container[0], False


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
from src.video.pixel_art_tool import generate_pixel_art, _detect_failed_image, _progressive_content_scrub, _detect_visual_type, _CATEGORY_SAFE_PROMPTS
from src.video.pexels_tool import fetch_vertical_footage
from src.video.tts_tool import generate_voiceover
from src.video.split_video_assembler import build_split_video
from src.video.visual_qa import validate_image, adjust_prompt_for_retry
from src.video.model_orchestrator import ModelOrchestrator

# ── GPU MODEL ORCHESTRATOR ──
orchestrator = ModelOrchestrator()

# ── DATABASE INIT ──
try:
    from src.db.connection import init_db
    init_db()
except Exception as _db_err:
    log.warning("db.init_failed", error=str(_db_err))


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


# ── PIPELINE CONSTANTS ──
NUM_STORIES = 2
IMAGES_PER_STORY = 4
NUM_IMAGES = NUM_STORIES * IMAGES_PER_STORY  # = 8


# ── STEP BANNER ──
_PIPELINE_STEP = [0]

def _step_banner(title: str) -> None:
    """Print a visible step banner so the user knows where the pipeline is."""
    _PIPELINE_STEP[0] += 1
    step_num = _PIPELINE_STEP[0]
    orchestrator.heartbeat(f"step {step_num}: {title}" if step_num % 2 == 0 else "")
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'='*60}")


def _step_done(step_name: str) -> None:
    duration = round(time.time() - _step_start, 1)
    elapsed = round(time.time() - _PIPELINE_START, 1)
    print(f"  [{step_name}] Done in {duration}s (elapsed: {elapsed}s)")


_PIPELINE_START = time.time()


# ── PostgreSQL SAVE HELPER (non-blocking) ──
_STATUS_MAP = {
    'news_fetch': 'scraped',
    'vector_dedup': 'scraped',
    'news_analysis': 'analyzed',
    'trending_context': 'analyzed',
    'script_synthesis': 'scripted',
    'script_curation': 'scripted',
    'script_evaluation': 'scripted',
    'tts': 'voiceover_generated',
    'image_generation': 'assembled',
    'video_assembly': 'assembled',
    'completed': 'published',
}


def _save_to_postgres(step_name: str, project_id: int, data: dict, topic: str = None):
    """Save pipeline step results to PostgreSQL in a background thread.

    Never blocks the pipeline. DB errors are logged, never raised.
    Uses connect_timeout=5 so a dead DB doesn't hang the background thread.
    """
    def _do_save():
        try:
            from src.db.connection import get_connection
            conn = get_connection(connect_timeout=5)
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

    threading.Thread(target=_do_save, daemon=True).start()


def _get_adjacent_fallback(scene_idx, generated_images, image_folder, scene_name, PILImage):
    """Use an adjacent story image instead of a blank placeholder."""
    if generated_images:
        if scene_idx % 2 == 1:
            return Path(generated_images[-1])
        elif len(generated_images) > 0:
            return Path(generated_images[-1])
    placeholder = PILImage.new('RGB', (1088, 1152), (10, 5, 25))
    placeholder_path = image_folder / f"{scene_name}_placeholder.png"
    placeholder.save(str(placeholder_path))
    return placeholder_path


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
# DRY-RUN: Short-circuit the entire pipeline with canned data
# ══════════════════════════════════════════════════════════════════════════
if DRY_RUN:
    log.info("[DRY-RUN] ========== DRY-RUN MODE — no API calls ==========")

    # Inject canned data
    articles = _DRY_RUN_ARTICLES
    news_analyses = _DRY_RUN_ANALYSES[:NUM_STORIES]
    trending_context = {}
    script = dict(_DRY_RUN_SCRIPT)

    # Enforce structure (same as real pipeline)
    script = llm._enforce_greeting(script)
    script = llm._enforce_segues(script)
    script = llm._dedup_segue_overlap(script)
    script = llm._dedup_inter_story_phrases(script)
    script = llm._enforce_fallout(script, news_analyses)
    script = llm._ensure_greeting_in_fulltext(script)
    script['word_count'] = len(script['full_text'].split())
    script['estimated_duration'] = int(script['word_count'] / 2.5)

    # Build segment timeline (same as real pipeline)
    segment_timeline = []
    greeting_seg = script.get('greeting', '')
    if greeting_seg:
        segment_timeline.append({'text': greeting_seg, 'image_idx': -1, 'label': 'greeting'})
    intro_hook = script.get('intro_hook', '')
    if intro_hook:
        segment_timeline.append({'text': intro_hook, 'image_idx': -1, 'label': 'intro_hook'})
    segment_timeline.append({'text': '....', 'image_idx': -1, 'label': 'intro_pause', 'is_separator': True})
    for i, story in enumerate(script['stories']):
        img_base = i * 4
        for field, suffix, img_off in [
            ('part_1_narration', 'part1', 0),
            ('part_2_narration', 'part2', 1),
            ('real_talk', 'real_talk', 2),
            ('fallout', 'fallout', 3),
        ]:
            val = story.get(field, '')
            if val:
                segment_timeline.append({'text': val, 'image_idx': img_base + img_off, 'label': f'story_{i+1}_{suffix}'})
        segue = story.get('segue', story.get('transition', ''))
        if segue and i < len(script['stories']) - 1:
            segment_timeline.append({'text': segue, 'image_idx': img_base + 3, 'label': f'story_{i+1}_segue'})
        if i < len(script['stories']) - 1:
            segment_timeline.append({'text': '....', 'image_idx': img_base + 3, 'label': f'story_{i+1}_separator', 'is_separator': True})
    script['segment_timeline'] = segment_timeline

    # Save output files
    project_folder.mkdir(parents=True, exist_ok=True)
    (project_folder / "script.txt").write_text(script['full_text'], encoding='utf-8')
    (project_folder / "script_segments.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding='utf-8')
    _save_checkpoint("dry_run_complete", project_folder, {"word_count": script['word_count']})

    full_script = script['full_text']
    log.info("[DRY-RUN] Script synthesized", words=script['word_count'], duration_s=script['estimated_duration'])
    log.info("[DRY-RUN] Stories: %d, Timeline segments: %d" % (len(script['stories']), len(segment_timeline)))
    for seg in segment_timeline:
        log.info("  [%s] img#%d: \"%s\"" % (seg['label'], seg['image_idx'], seg['text'][:50]))

    log.info("[DRY-RUN] ========== DRY-RUN COMPLETE ==========")
    log.info("[DRY-RUN] Project folder: %s" % project_folder)
    log.info("[DRY-RUN] To view output: cat %s/script.txt" % project_folder)

    llm.unload_model()
    print(f"\n[DRY-RUN] Complete! Project: {project_folder}")
    print(f"[DRY-RUN] Script: {script['word_count']} words, ~{script['estimated_duration']}s")
    print(f"[DRY-RUN] Stories: {len(script['stories'])}, Segments: {len(segment_timeline)}")
    exit(0)


# ── GPU MODEL LIFECYCLE: Pre-pipeline sweep + VRAM check ──
if not orchestrator.phase_pre_pipeline():
    print("\nFATAL: Insufficient GPU VRAM to start the pipeline.")
    print("Close other GPU processes (Ollama, browser tabs, ComfyUI) and retry.")
    print("Or set USE_LOCAL_FLUX=false to use cloud API instead.")
    sys.exit(1)

# ── GPU MODEL LIFECYCLE: Transition to LLM phase ──
orchestrator.phase_llm()

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: FETCH LATEST NEWS — Async Scraper with sync fallback
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="news_fetch")
_step_banner("FETCH NEWS")
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
    print("  [SCRAPER] Ranking articles...", flush=True)
    from src.collector.rss_scraper import RSScraper
    ranker = RSScraper()
    viral_articles = ranker.filter_viral_potential(all_articles, top_n=10)
    print(f"  [SCRAPER] {len(viral_articles)} viral articles found", flush=True)

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

# ── Topic diversity selection (word overlap — no LLM calls) ──
def _is_semantically_similar(title_a: str, title_b: str) -> bool:
    """Fast word-overlap dedup. No LLM call — titles don't need semantic analysis."""
    words_a = frozenset(title_a.lower().split()[:6])
    words_b = frozenset(title_b.lower().split()[:6])
    return len(words_a & words_b) >= 3


def _build_video_title(analyses, script):
    """Build a 4-8 word video title from story topics. Extracts proper nouns."""
    import re
    entities = []
    for a in (analyses or []):
        topic = a.get('topic', '')
        proper_nouns = re.findall(r'\b[A-Z][a-z]{2,}\b', topic)
        entities.extend(proper_nouns[:3])
    seen = set()
    unique = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique.append(e)
    if len(unique) >= 2:
        return f"{unique[0]}, {unique[1]}, and the Shift"
    elif unique:
        return f"{unique[0]} — The Shift"
    else:
        last = analyses[-1] if analyses else {}
        return last.get('topic', 'Geopolitical Update')[:60]

selected = []
for i, article in enumerate(viral_articles):
    title = article.get('title', '')
    is_duplicate = False
    for existing in selected:
        if _is_semantically_similar(title, existing.get('title', '')):
            is_duplicate = True
            log.info("dedup.intra_batch.skip", new=title[:60], similar_to=existing.get('title', '')[:60])
            break
    if not is_duplicate:
        selected.append(article)
        print(f"  [DEDUP] Selected: {title[:65]}")
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
print(f"\n  [SCRAPER] {len(articles)} articles selected in {round(_step_duration, 1)}s")
for i, a in enumerate(articles, 1):
    print(f"    {i}. {a.get('title', 'N/A')[:70]}")
_step_done("FETCH NEWS")


# ══════════════════════════════════════════════════════════════════════════
# NOTE: Vector dedup (pgvector + nomic-embed-text) is DISABLED because
# it loads a SECOND Ollama model into VRAM during the LLM phase, causing
# VRAM contention on 24GB GPUs. Word-overlap dedup above is sufficient.
# ══════════════════════════════════════════════════════════════════════════
log.info("dedup.skipped", reason="vector_dedup_disabled_vram_contention")


# ══════════════════════════════════════════════════════════════════════════
# OLLAMA HEALTH CHECK — verify Ollama is responsive before first LLM call
# ══════════════════════════════════════════════════════════════════════════
if not orchestrator.check_ollama_health():
    print("\nFATAL: Ollama is not running or unresponsive.")
    print("Start it with: ollama serve")
    print("Or if using a remote server, check OLLAMA_HOST in .env")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: NEWS ANALYSIS — LangChain with raw LLM fallback
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="news_analysis")
_step_banner("NEWS ANALYSIS (LLM)")
_step_start = time.time()
print("  [LLM] Ollama model will load on first call (may take 30-90s)...", flush=True)

news_analyses = []
for i, article in enumerate(articles, 1):
    print(f"\n  [{i}/{len(articles)}] Analyzing: {article.get('title', 'N/A')[:70]}...")
    _analysis_start = time.time()
    try:
        # Get full article text
        print(f"  [{i}/{len(articles)}] Fetching article text...", flush=True)
        article_text = None
        try:
            from src.collector.rss_scraper import RSScraper
            text_scraper = RSScraper()
            article_text, _fetch_timed_out = _run_with_heartbeat(
                text_scraper.get_full_article_text, f"fetch_text_{i}", 5, 15,
                article
            )
            if _fetch_timed_out or not article_text:
                article_text = None
        except Exception:
            article_text = None

        if not article_text:
            article_text = article.get('summary', article.get('title', ''))

        log.debug("analysis.article", index=i, chars=len(article_text))

        analysis = None

        # ── Try LangChain structured chain first ──
        if _USE_LANGCHAIN:
            try:
                print(f"  [{i}/{len(articles)}]   LangChain analysis...", end="", flush=True)
                from src.models.schemas import NewsAnalysis as NewsAnalysisModel
                chain = _langchain.build_structured_chain(NewsAnalysisModel, "news_processor")
                result, _lc_timed_out = _run_with_heartbeat(
                    chain.invoke, f"analysis_{i}/{len(articles)}", 8, 120,
                    {"input_text": f"Analyze this news article:\n\n{article_text}"}
                )
                if not _lc_timed_out and result:
                    analysis = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
                    log.info("analysis.langchain.success", index=i)
                else:
                    print(f"  [{i}/{len(articles)}]   LangChain timed out", flush=True)
            except Exception as e:
                print(f"  [{i}/{len(articles)}]   LangChain error: {str(e)[:60]}", flush=True)
                log.warning("analysis.langchain.failed", index=i, error=str(e))

        # ── Fallback to raw LLM interface ──
        if not analysis:
            print(f"  [{i}/{len(articles)}]   Raw LLM fallback...", end="", flush=True)
            analysis, _proc_timed_out = _run_with_heartbeat(
                llm.process_news, f"analysis_{i}/{len(articles)}", 8, 120,
                article_text
            )
            if _proc_timed_out:
                analysis = None

        if analysis:
            log.info("analysis.complete", index=i,
                     topic=analysis.get('topic', 'N/A')[:60],
                     impact=analysis.get('impact_score', 0))
            news_analyses.append(analysis)
            _analysis_dur = round(time.time() - _analysis_start, 1)
            print(f"  [{i}/{len(articles)}] Done ({_analysis_dur}s) — impact={analysis.get('impact_score', '?')}: {analysis.get('topic', 'N/A')[:55]}")
            if i % 2 == 0:
                orchestrator.heartbeat(f"news_analysis_{i}/{len(articles)}")
        else:
            log.error("analysis.empty", index=i)
            print(f"  [{i}/{len(articles)}] FAILED — empty analysis result")
    except Exception as e:
        log.error("analysis.failed", index=i, error=str(e))
        print(f"  [{i}/{len(articles)}] ERROR — {str(e)[:80]}")

if len(news_analyses) < 2:
    log.error("analysis.insufficient", count=len(news_analyses), minimum=2)
    llm.unload_model()
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
_step_done("NEWS ANALYSIS")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: TRENDING CONTEXT (optional boost)
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="trending_context")
_step_banner("TRENDING CONTEXT")
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
_step_done("TRENDING CONTEXT")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: MULTI-NEWS SCRIPT SYNTHESIS (Masker personality)
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="script_synthesis")
_step_banner("SCRIPT SYNTHESIS (LLM)")
_step_start = time.time()

try:
    log.info("script.generating", format="multi_news_3_stories")
    print("\n  [LLM] Generating script (this may take 30-120s)...", flush=True)
    script, _synth_timed_out = _run_with_heartbeat(
        llm.synthesize_multi_news_script, "script_synthesis", 8, 600,
        news_analyses
    )

    if _synth_timed_out or not script:
        log.error("script.synthesis_failed", reason="timeout" if _synth_timed_out else "empty_result")
        llm.unload_model()
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
    print(f"  [LLM] Script done — {script.get('word_count', 0)} words, ~{script.get('estimated_duration', 0)}s")
    orchestrator.heartbeat("script_synthesis_done")
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
    llm.unload_model()
    exit(1)

_step_duration = time.time() - _step_start
log.info("step.complete", step="script_synthesis", duration_s=round(_step_duration, 2))
_step_done("SCRIPT SYNTHESIS")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4.1: ENFORCE SCRIPT STRUCTURE + BUILD SEGMENT TIMELINE
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="script_enforcement")
_step_banner("SCRIPT ENFORCEMENT")
_step_start = time.time()

try:
    script = llm._enforce_greeting(script)
    script = llm._enforce_segues(script)
    script = llm._dedup_segue_overlap(script)
    script = llm._dedup_inter_story_phrases(script)
    script = llm._enforce_fallout(script, news_analyses)
    script = llm._ensure_greeting_in_fulltext(script)
    script['word_count'] = len(script.get('full_text', full_script).split())
    script['estimated_duration'] = int(script['word_count'] / 2.5)
    full_script = script.get('full_text', full_script)

    # Build segment timeline for video assembly
    segment_timeline = []
    greeting_seg = script.get('greeting', '')
    if greeting_seg:
        segment_timeline.append({'text': greeting_seg, 'image_idx': -1, 'label': 'greeting'})
    intro_hook = script.get('intro_hook', '')
    if intro_hook:
        segment_timeline.append({'text': intro_hook, 'image_idx': -1, 'label': 'intro_hook'})
    segment_timeline.append({'text': '....', 'image_idx': -1, 'label': 'intro_pause', 'is_separator': True})
    for i, story in enumerate(script['stories']):
        img_base = i * IMAGES_PER_STORY
        for field, suffix, img_off in [
            ('part_1_narration', 'part1', 0),
            ('part_2_narration', 'part2', 1),
            ('real_talk', 'real_talk', 2),
            ('fallout', 'fallout', 3),
        ]:
            val = story.get(field, '')
            if val:
                segment_timeline.append({'text': val, 'image_idx': img_base + img_off, 'label': f'story_{i+1}_{suffix}'})
        segue = story.get('segue', story.get('transition', ''))
        if segue and i < len(script['stories']) - 1:
            segment_timeline.append({'text': segue, 'image_idx': img_base + 3, 'label': f'story_{i+1}_segue'})
        if i < len(script['stories']) - 1:
            segment_timeline.append({'text': '....', 'image_idx': img_base + 3, 'label': f'story_{i+1}_separator', 'is_separator': True})
    script['segment_timeline'] = segment_timeline

    # Save updated script
    script_file = project_folder / "script.txt"
    script_file.write_text(full_script, encoding='utf-8')
    segments_file = project_folder / "script_segments.json"
    segments_file.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding='utf-8')

    log.info("script_enforcement.complete",
             greeting=bool(script.get('greeting')),
             segues=sum(1 for s in script.get('stories', [])[:-1] if s.get('segue')),
             fallouts=sum(1 for s in script.get('stories', []) if s.get('fallout')),
             timeline_segments=len(segment_timeline))

except Exception as e:
    log.error("script_enforcement.failed", error=str(e))

_step_duration = time.time() - _step_start
log.info("step.complete", step="script_enforcement", duration_s=round(_step_duration, 2))
_step_done("SCRIPT ENFORCEMENT")

# ══════════════════════════════════════════════════════════════════════════
_step_banner("VISUAL PROMPTS (LLM)")
log.info("step.start", step="visual_prompts")
_step_start = time.time()

try:
    print("\n  [LLM] Generating visual prompts (this may take 15-60s)...", flush=True)
    dedicated_visuals, _visual_timed_out = _run_with_heartbeat(
        llm.generate_visual_prompts, "visual_prompts", 8, 300,
        script, articles, news_analyses
    )
    if _visual_timed_out:
        log.error("visual_prompts.timeout", timeout_s=300)
        dedicated_visuals = None

    if dedicated_visuals and len(dedicated_visuals) >= NUM_IMAGES:
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
_step_done("VISUAL PROMPTS")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4.7: SCRIPT CURATION — LangChain with raw LLM fallback
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="script_curation")
_step_banner("SCRIPT CURATION (LLM)")
_step_start = time.time()

try:
    print("\n  [LLM] Curating script (this may take 15-60s)...")
    original_text = full_script
    log.info("curation.original", words=len(original_text.split()))

    curated_text = None

    # ── Try LangChain curation chain first ──
    if _USE_LANGCHAIN:
        try:
            print("  [LLM] LangChain curation...", end="", flush=True)
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
            print(f"  [LLM] Curation error: {str(e)[:60]}", flush=True)
            log.warning("curation.langchain.failed", error=str(e))

    # ── Fallback to raw LLM curation ──
    if not curated_text:
        print("  [LLM] Curation fallback...", end="", flush=True)
        curated_text, _curate_timed_out = _run_with_heartbeat(
            llm.curate_script, "script_curation", 8, 300,
            script
        )
        if _curate_timed_out:
            log.error("curation.timeout", timeout_s=300)
            curated_text = None

    if curated_text and curated_text != original_text:
        # Reassemble full script with structural elements (segues, closing)
        # that curation strips out. This prevents losing segues/outro.
        curated_bodies = llm._parse_curated_stories(curated_text, len(script.get('stories', [])))
        if curated_bodies:
            reassembled = llm._reassemble_script(script, curated_bodies)
            script['full_text'] = reassembled
            full_script = reassembled
        else:
            script['full_text'] = curated_text
            full_script = curated_text
        script['word_count'] = len(full_script.split())
        script['estimated_duration'] = int(len(full_script.split()) / 2.5)

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
_step_done("SCRIPT CURATION")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4.8: SCRIPT EVALUATION — Semantic Dedup + Continuity Critic
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="script_evaluation")
_step_banner("SCRIPT EVALUATION")
_step_start = time.time()

try:
    print("  [LLM] Evaluating script...", end="", flush=True)
    from src.brain.script_evaluator import run_script_evaluation

    script, _eval_timed_out = _run_with_heartbeat(
        run_script_evaluation, "script_evaluation", 8, 120,
        script=script,
        news_analyses=news_analyses,
        llm_interface=llm,
        similarity_threshold=0.90,
    )
    if _eval_timed_out:
        log.error("script_evaluation.timeout", timeout_s=120)

    full_script = script.get('full_text', full_script)
    script_file = project_folder / "script.txt"
    script_file.write_text(full_script, encoding='utf-8')

    segments_file = project_folder / "script_segments.json"
    segments_file.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding='utf-8')

    _save_checkpoint("script_evaluation", project_folder,
                     {"stories": len(script.get('stories', []))})

except Exception as e:
    log.warning("script_evaluation.failed", error=str(e))

_step_duration = time.time() - _step_start
log.info("step.complete", step="script_evaluation", duration_s=round(_step_duration, 2))
_step_done("SCRIPT EVALUATION")


# ── GPU MODEL LIFECYCLE: Clean up LLM phase before loading FLUX ──
log.info("orchestrator.transition", phase="image_gen", note="Force cleanup, then evict Ollama + preload FLUX")
orchestrator.force_cleanup()
if not orchestrator.verify_clean_state():
    print("\nWARNING: GPU state not clean before image generation. Proceeding anyway...")
    log.warning("orchestrator.dirty_state_before_image_gen")

flux_preloaded = orchestrator.phase_image_generation()
if flux_preloaded:
    log.info("orchestrator.flux_preloaded", note="FLUX pipeline loaded and pinned for batch generation")
else:
    log.warning("orchestrator.flux_preload_failed", note="FLUX preload failed, will attempt per-image or fall back to cloud")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: PIXEL ART GENERATION (3 per story = 6 total)
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="pixel_art")
_step_banner("PIXEL ART GENERATION (GPU)")
_step_start = time.time()

try:
    generated_images = []
    image_folder = project_folder / "images"
    image_folder.mkdir(exist_ok=True)

    if SKIP_IMAGES:
        from PIL import Image as PILImage
        scene_names = []
        for i in range(NUM_STORIES):
            scene_names.append(f'story_{i+1}_part1')
            scene_names.append(f'story_{i+1}_part2')
            scene_names.append(f'story_{i+1}_real_talk')
            scene_names.append(f'story_{i+1}_fallout')

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
        scene_types = ['part1', 'part2', 'real_talk', 'fallout']
        for s_idx in range(NUM_STORIES):
            for p_idx in range(IMAGES_PER_STORY):
                fallback_idx = s_idx * IMAGES_PER_STORY + p_idx
                if fallback_idx >= len(all_visual_scenes):
                    story = stories[s_idx] if s_idx < len(stories) else {}
                    scene_type = scene_types[p_idx]
                    if scene_type == 'real_talk':
                        part_text = story.get('real_talk', story.get('part_2_narration', ''))
                    elif scene_type == 'fallout':
                        part_text = story.get('fallout', story.get('real_talk', ''))
                    else:
                        part_key = f'part_{p_idx+1}_narration'
                        part_text = story.get(part_key, story.get('body', ''))
                    fallback_desc = _build_fallback_prompt(part_text, s_idx, p_idx, news_analyses)
                    all_visual_scenes.append({
                        'scene': f'story_{s_idx+1}_{scene_type}',
                        'description': fallback_desc
                    })
        all_visual_scenes = all_visual_scenes[:NUM_IMAGES]

        scene_names = []
        for i in range(NUM_STORIES):
            scene_names.append(f'story_{i+1}_part1')
            scene_names.append(f'story_{i+1}_part2')
            scene_names.append(f'story_{i+1}_real_talk')
            scene_names.append(f'story_{i+1}_fallout')

        for scene_idx, scene_name in enumerate(scene_names):
            log.debug("pixel_art.generating", scene=scene_name)
            print(f"\n  [IMG {scene_idx+1}/{NUM_IMAGES}] {scene_name}...")

            scene_data = all_visual_scenes[scene_idx] if scene_idx < len(all_visual_scenes) else {}
            prompt = scene_data.get('description', '')

            # Pass only the raw scene description to generate_pixel_art.
            # Style suffix (from config/image_style.json) is applied internally by pixel_art_tool.
            full_prompt = prompt if prompt else ""

            story_idx = scene_idx // IMAGES_PER_STORY
            scene_in_story = scene_idx % IMAGES_PER_STORY
            story_text = ''
            if story_idx < len(script.get('stories', [])):
                story = script['stories'][story_idx]
                if scene_in_story == 0:
                    story_text = story.get('part_1_narration', story.get('body', ''))
                elif scene_in_story == 1:
                    story_text = story.get('part_2_narration', story.get('body', ''))
                elif scene_in_story == 2:
                    story_text = story.get('real_talk', story.get('part_2_narration', ''))
                elif scene_in_story == 3:
                    story_text = story.get('fallout', story.get('real_talk', ''))

            fallback_desc = _build_fallback_prompt(
                story_text, story_idx, scene_in_story, news_analyses
            )

            current_prompt = full_prompt
            accepted = False
            qa_attempts = []
            scrub_level = 0

            for attempt in range(4):
                seed = base_seed + scene_idx + (attempt * 100)
                log.debug("pixel_art.attempt", scene=scene_name, attempt=attempt + 1)
                art_result = generate_pixel_art(current_prompt, script_text=story_text, seed=seed)

                # Auto-detect failed generation (solid color, monochrome, etc.)
                if art_result.get('success') and art_result.get('detected_failure'):
                    log.warning("pixel_art.failed_detection", scene=scene_name, reason=art_result['detected_failure'])
                    scrub_level = min(scrub_level + 1, 3)
                    visual_type = _detect_visual_type(current_prompt)
                    scrubbed = _progressive_content_scrub(full_prompt, visual_type, level=scrub_level)
                    if scrubbed != full_prompt:
                        current_prompt = scrubbed
                    elif fallback_desc:
                        current_prompt = fallback_desc
                    if attempt < 3:
                        continue
                    else:
                        break

                if not art_result.get('success'):
                    log.warning("pixel_art.gen_failed", scene=scene_name, attempt=attempt + 1)
                    scrub_level = min(scrub_level + 1, 3)
                    if attempt < 3:
                        adjusted = adjust_prompt_for_retry(full_prompt, 'api_failure', attempt + 1)
                        if adjusted == full_prompt and fallback_desc:
                            adjusted = fallback_desc
                        current_prompt = adjusted
                        # On 4th attempt (last), use category-safe prompt
                        if attempt == 2:
                            visual_type = _detect_visual_type(full_prompt)
                            safe_prompt = _CATEGORY_SAFE_PROMPTS.get(visual_type, _CATEGORY_SAFE_PROMPTS.get('general', ''))
                            if safe_prompt:
                                current_prompt = safe_prompt
                        continue
                    else:
                        break

                src_path = Path(art_result.get('path'))
                dst_filename = f"{scene_name}_{src_path.name}"
                dst_path = image_folder / dst_filename
                shutil.copy2(src_path, dst_path)

                qa_result = validate_image(
                    str(dst_path), current_prompt,
                    skip_vlm=True,
                )
                qa_attempts.append(qa_result)

                if qa_result['pass']:
                    generated_images.append(str(dst_path))
                    print(f"  [IMG {scene_idx+1}/{NUM_IMAGES}] {scene_name} OK (attempt {attempt+1})")
                    log.info("pixel_art.accepted", scene=scene_name, attempt=attempt + 1,
                             reason=qa_result.get('reason', 'pass'))
                    accepted = True
                    break
                else:
                    print(f"  [IMG {scene_idx+1}/{NUM_IMAGES}] {scene_name} QA failed: {qa_result.get('reason', 'unknown')[:40]} (attempt {attempt+1})")
                    log.warning("pixel_art.qa_failed", scene=scene_name,
                                reason=qa_result.get('reason', 'unknown'), attempt=attempt + 1)
                    dst_path.unlink(missing_ok=True)
                    scrub_level = min(scrub_level + 1, 3)
                    if attempt < 3:
                        adjusted = adjust_prompt_for_retry(full_prompt, qa_result.get('reason', ''), attempt + 1)
                        if adjusted == full_prompt and fallback_desc:
                            adjusted = fallback_desc
                        # On last QA retry, use category-safe prompt
                        if attempt == 2:
                            visual_type = _detect_visual_type(full_prompt)
                            safe_prompt = _CATEGORY_SAFE_PROMPTS.get(visual_type, _CATEGORY_SAFE_PROMPTS.get('general', ''))
                            if safe_prompt:
                                current_prompt = safe_prompt
                        else:
                            current_prompt = adjusted

            if not accepted:
                fallback_path = _get_adjacent_fallback(
                    scene_idx, generated_images, image_folder, scene_name, PILImage
                )
                generated_images.append(str(fallback_path))
                print(f"  [IMG {scene_idx+1}/{NUM_IMAGES}] {scene_name} FALLBACK")
                log.warning("pixel_art.fallback", scene=scene_name,
                            source="adjacent_image" if 'placeholder' not in str(fallback_path) else "placeholder")

            if (scene_idx + 1) % 2 == 0:
                orchestrator.heartbeat(f"image_gen_{scene_idx+1}/{NUM_IMAGES}")

        # Final validation: ensure exactly NUM_IMAGES images
        if len(generated_images) < NUM_IMAGES:
            log.warning("pixel_art.few_images", count=len(generated_images), needed=NUM_IMAGES)
            while len(generated_images) < NUM_IMAGES:
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
_step_done("PIXEL ART")

# ── GPU MODEL LIFECYCLE: Flush FLUX after batch generation, then transition to TTS ──
orchestrator.phase_image_generation_done()
log.info("orchestrator.transition", phase="post_image", note="FLUX pipeline flushed, VRAM released")
orchestrator.force_cleanup()
orchestrator.phase_tts()
log.info("orchestrator.transition", phase="tts", note="VRAM cleaned, Kokoro will load on demand")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: VOICE GENERATION
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="voice_generation")
_step_banner("VOICE GENERATION (TTS)")
_step_start = time.time()

voice_file = None
tts_result = {}
try:
    print("\n  [TTS] Generating voiceover...")
    tts_result = generate_voiceover(full_script, "authoritative")

    if tts_result.get('success'):
        import shutil
        src_audio = Path(tts_result.get('path'))
        dst_audio = project_folder / "voiceover.mp3"
        shutil.copy2(src_audio, dst_audio)

        log.info("voice.complete", duration_s=tts_result.get('estimated_duration_seconds', 0))
        voice_file = str(dst_audio)
        print(f"  [TTS] Done — {tts_result.get('estimated_duration_seconds', 0):.1f}s audio")
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
_step_done("VOICE GENERATION")


# ── GPU MODEL LIFECYCLE: Video assembly needs no GPU models ──
orchestrator.phase_video_assembly()

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: VIDEO ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="video_assembly")
_step_banner("VIDEO ASSEMBLY")
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

        # Per-story image balancing (4 images per story: hook ~25%, mechanism ~30%, truth ~25%, fallout ~20%)
        num_stories_calc = num_images // IMAGES_PER_STORY
        for story_i in range(num_stories_calc):
            img_a = story_i * IMAGES_PER_STORY
            img_b = story_i * IMAGES_PER_STORY + 1
            img_c = story_i * IMAGES_PER_STORY + 2
            img_d = story_i * IMAGES_PER_STORY + 3
            if img_d >= num_images:
                break
            story_start = image_times[img_a]['start']
            story_end = image_times[img_d]['end']
            story_total = story_end - story_start
            if story_total <= 0:
                continue
            min_dur = story_total * 0.10
            # Target split: hook=25%, mechanism=30%, truth=25%, fallout=20%
            target_a_end = story_start + story_total * 0.25
            target_b_end = story_start + story_total * 0.55
            target_c_end = story_start + story_total * 0.80
            # Enforce minimum 10% per image
            target_a_end = max(story_start + min_dur, min(target_a_end, story_end - 3 * min_dur))
            target_b_end = max(target_a_end + min_dur, min(target_b_end, story_end - 2 * min_dur))
            target_c_end = max(target_b_end + min_dur, min(target_c_end, story_end - min_dur))
            image_times[img_a]['end'] = target_a_end
            image_times[img_b]['start'] = target_a_end
            image_times[img_b]['end'] = target_b_end
            image_times[img_c]['start'] = target_b_end
            image_times[img_c]['end'] = target_c_end
            image_times[img_d]['start'] = target_c_end

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

    # Build a clean 4-8 word video title from story topics
    hook_text = _build_video_title(news_analyses, script)
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
_step_done("VIDEO ASSEMBLY")


# ══════════════════════════════════════════════════════════════════════════
# STEP 9: PLATFORM METADATA
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="platform_metadata")
_step_banner("PLATFORM METADATA")
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
_step_done("PLATFORM METADATA")


# ══════════════════════════════════════════════════════════════════════════
# STEP 10: PROJECT SUMMARY + TELEGRAM DELIVERY
# ══════════════════════════════════════════════════════════════════════════
log.info("step.start", step="project_summary")
_step_banner("PROJECT SUMMARY")

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

# ── Store topic vectors for future dedup (deferred — non-blocking) ──
try:
    from src.brain.memory.vector_store import VectorStore
    from src.brain.memory.embedder import Embedder
    embedder = Embedder()
    store = VectorStore(embedder=embedder)
    for i, analysis in enumerate(news_analyses):
        topic = analysis.get('topic', '')
        category = analysis.get('category', 'general')
        if topic:
            try:
                vector = embedder.embed(topic)
                store.store_embedding(
                    project_id=str(project_id),
                    topic=topic,
                    category=category,
                    vector=vector,
                )
            except Exception as emb_e:
                log.warning("vector_topics.embed_failed", topic=topic[:40], error=str(emb_e))
    log.info("vector_topics.stored", count=len(news_analyses))
except Exception as e:
    log.warning("vector_topics.failed", error=str(e))

# ── Telegram delivery ──
if final_video_path and not args.no_telegram:
    try:
        from tools.telegram_sender import send_video_to_telegram
        today_str = datetime.now().strftime('%b %d, %Y')
        video_title = hook_text or 'Geopolitical Update'
        if isinstance(video_title, str) and len(video_title) > 60:
            video_title = video_title[:57] + '...'
        caption = f"<b>{video_title}</b>\n{today_str}"
        if platform_metadata:
            yt = platform_metadata.get('youtube', {})
            if yt.get('description'):
                desc_lines = yt['description'].split('\n')[:3]
                caption += '\n\n' + '\n'.join(desc_lines)
            common_tags = platform_metadata.get('common_hashtags', [])
            if common_tags:
                hashtag_str = ' '.join(f'#{t.replace(" ", "").replace("-", "")}' for t in common_tags[:8])
                caption += '\n\n' + hashtag_str
        if len(caption) > 1024:
            caption = caption[:1021] + '...'
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
try:
    llm.unload_model()
except Exception:
    pass

# ── GPU MODEL LIFECYCLE: Final cleanup ──
orchestrator.phase_cleanup()
log.info("orchestrator.cleanup", note="All GPU models evicted")
_step_done("PROJECT SUMMARY")

total_elapsed = round(time.time() - _PIPELINE_START, 1)

print(f"\n{'='*60}")
print(f"  PIPELINE COMPLETE")
print(f"  Elapsed: {total_elapsed}s")
print(f"  Project: {project_folder}")
orchestrator.heartbeat("pipeline_complete")

log.info("pipeline.complete",
         project_id=project_id,
         video=final_video_path or "FAILED",
         duration_s=assembly_result.get('duration_seconds') if final_video_path else 0,
         images=len(generated_images),
         words=script.get('word_count', 0))