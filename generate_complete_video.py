import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import json
import time
from datetime import datetime

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ── File-based logging: mirror all stdout to a log file ──
import io

class _TeeWriter:
    """Writes to both stdout and a log file simultaneously."""
    def __init__(self, terminal, log_path):
        self.terminal = terminal
        self.log_file = open(log_path, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def reconfigure(self, **kwargs):
        # Pass through reconfigure calls to terminal (for Windows Unicode fix)
        if hasattr(self.terminal, 'reconfigure'):
            self.terminal.reconfigure(**kwargs)

# Create logs directory
_log_dir = Path("output/logs")
_log_dir.mkdir(parents=True, exist_ok=True)
_log_path = _log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
sys.stdout = _TeeWriter(sys.stdout, str(_log_path))

print("🎬 YT-MACHINE COMPLETE VIDEO GENERATION (3-News Format)")
print("=" * 60)
print(f"📝 Log file: {_log_path}")


def _extract_segment_text(segment_data) -> str:
    """Extract plain text from a script segment that may be a string or dict."""
    if isinstance(segment_data, str):
        return segment_data
    if isinstance(segment_data, dict):
        return (segment_data.get('narration')
                or segment_data.get('text')
                or segment_data.get('content')
                or str(segment_data))
    return str(segment_data) if segment_data else ''


from pipeline_utils import bridge_timestamp_gaps, build_fallback_prompt as _build_fallback_prompt


# Import core components
from brain.llm_interface import LLMInterface
from video_server.pixel_art_tool import generate_pixel_art
from video_server.pexels_tool import fetch_vertical_footage
from video_server.tts_tool import generate_voiceover
from video_server.split_video_assembler import build_split_video

# Initialize components
llm = LLMInterface()

# Create unique project folder
project_id = int(time.time())
project_folder = Path(f"output/projects/video_{project_id}")
project_folder.mkdir(parents=True, exist_ok=True)

print(f"📁 Project folder: {project_folder}")
print(f"🕐 Time of day: {datetime.now().strftime('%H:%M')} → greeting will be {'Morning' if datetime.now().hour < 12 else 'Afternoon' if datetime.now().hour < 18 else 'Evening'}")

# ============================================================
# STEP 1: FETCH LATEST NEWS (Top 3)
# ============================================================
print("\n📰 STEP 1: FETCHING LATEST NEWS (Top 3)")
print("-" * 40)

articles = []
try:
    from redfish.rss_scraper import RSScraper
    scraper = RSScraper()
    
    print("Scraping RSS feeds...")
    all_articles = scraper.scrape_all(max_age_hours=24)
    print(f"Found {len(all_articles)} articles")
    
    # Get top 3 viral articles (diverse topics)
    viral_articles = scraper.filter_viral_potential(all_articles, top_n=10)
    
    if len(viral_articles) < 3:
        print(f"⚠️  Only found {len(viral_articles)} articles, need at least 3")
        # Try to use what we have
        if len(viral_articles) == 0:
            print("❌ No suitable articles found")
            exit(1)
    
    # Pick top 3 diverse articles (avoid same topic)
    selected = []
    seen_topic_words = []  # list of frozensets (hashable)
    for article in viral_articles:
        title_lower = article.get('title', '').lower()
        # Simple topic diversity check — skip if too similar to already selected
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
    
    # Fill remaining slots if not enough diverse articles
    while len(selected) < 3 and len(viral_articles) > len(selected):
        for a in viral_articles:
            if a not in selected:
                selected.append(a)
                break
        if len(selected) >= 3:
            break
    
    articles = selected[:3]
    
    for i, a in enumerate(articles, 1):
        print(f"  ✅ Story {i}: {a['title'][:70]}...")
    
except Exception as e:
    print(f"❌ News scraping failed: {e}")
    exit(1)

# ============================================================
# STEP 2: QUICK ANALYSIS (All 3 articles)
# ============================================================
print("\n🔍 STEP 2: QUICK NEWS ANALYSIS (3 articles)")
print("-" * 40)

news_analyses = []
for i, article in enumerate(articles, 1):
    try:
        print(f"\n  Analyzing story {i}...")
        article_text = scraper.get_full_article_text(article)
        print(f"    Article length: {len(article_text)} chars")
        analysis = llm.process_news(article_text)
        
        if analysis:
            print(f"    ✅ Topic: {analysis.get('topic', 'N/A')[:60]}")
            print(f"    ✅ Impact: {analysis.get('impact_score', 0)}/10")
            news_analyses.append(analysis)
        else:
            print(f"    ❌ Analysis failed for story {i}")
    except Exception as e:
        print(f"    ❌ Story {i} analysis failed: {e}")

if len(news_analyses) < 2:
    print(f"❌ Need at least 2 analyzed stories, got {len(news_analyses)}")
    exit(1)

# Sort by impact_score ascending — most important story LAST for maximum retention
paired = []
for i, analysis in enumerate(news_analyses):
    # Find matching article (same index, or skip)
    art_idx = min(i, len(articles) - 1)
    paired.append((articles[art_idx], analysis))

paired.sort(key=lambda x: x[1].get('impact_score', 0))  # lowest impact first
articles = [p[0] for p in paired]
news_analyses = [p[1] for p in paired]

print(f"\n✅ Analyzed {len(news_analyses)} stories successfully (sorted by impact, most important last)")
for i, a in enumerate(news_analyses):
    print(f"  Story {i+1}: impact={a.get('impact_score', '?')}/10 — {a.get('topic', 'N/A')[:50]}")

# ============================================================
# STEP 3: TRENDING CONTEXT (optional boost)
# ============================================================
print("\n🔬 STEP 3: TRENDING CONTEXT ANALYSIS")
print("-" * 40)

trending_context = {}
try:
    from redfish.trending_analyzer import TrendingAnalyzer
    trending_analyzer = TrendingAnalyzer()
    trending_context = trending_analyzer.analyze(all_articles, top_n=40)
    print(f"✅ Trending terms: {len(trending_context)}")
except Exception as e:
    print(f"⚠️  Trending analysis failed: {e}")

# ============================================================
# STEP 4: MULTI-NEWS SCRIPT SYNTHESIS (Masker personality)
# ============================================================
print("\n🎭 STEP 4: MULTI-NEWS SCRIPT SYNTHESIS (Masker)")
print("-" * 40)

try:
    print("Generating Masker personality script with 3 stories...")
    script = llm.synthesize_multi_news_script(news_analyses)
    
    if not script:
        print("❌ Multi-news script synthesis failed")
        exit(1)
    
    full_script = script.get('full_text', '')
    if not full_script:
        # Build from stories (including intro_hook and punchlines)
        parts = [script.get('greeting', ''), script.get('intro_hook', '')]
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
    
    print(f"✅ Script synthesized (~{script.get('estimated_duration', 0)}s)")
    print(f"  Greeting: {script.get('greeting', 'N/A')[:70]}...")
    print(f"  Intro hook: {script.get('intro_hook', 'N/A')[:70]}...")
    for i, story in enumerate(script.get('stories', []), 1):
        print(f"  Story {i} hook: {story.get('mini_hook', 'N/A')[:60]}...")
        print(f"  Story {i} punchline: {story.get('punchline', 'N/A')[:60]}...")
    print(f"  Closing: {script.get('closing', 'N/A')[:60]}...")
    print(f"  Words: {script.get('word_count', 0)}")
    
    # Save script files
    script_file = project_folder / "script.txt"
    script_file.write_text(full_script, encoding='utf-8')
    
    segments_file = project_folder / "script_segments.json"
    segments_file.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding='utf-8')
    
    print(f"  📄 Script saved: {script_file.name}, {segments_file.name}")
    
except Exception as e:
    print(f"❌ Script synthesis failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# STEP 4.5: DEDICATED VISUAL PROMPT GENERATION (BEFORE curation)
# Generate visual prompts from original narrations FIRST so images
# always match the content, regardless of later curation rephrasing.
# ============================================================
print("\n🖼️ STEP 4.5: DEDICATED VISUAL PROMPT GENERATION")
print("-" * 40)

try:
    dedicated_visuals = llm.generate_visual_prompts(script)
    
    if dedicated_visuals and len(dedicated_visuals) >= 6:
        # Override the script-synthesis visuals with dedicated ones
        script['all_visual_scenes'] = dedicated_visuals
        print(f"  ✅ Override: using {len(dedicated_visuals)} dedicated visual prompts (narration-grounded)")
        for v in dedicated_visuals:
            print(f"    [{v['scene']}] {v.get('description', '')[:60]}...")
    else:
        # Fallback: keep existing all_visual_scenes from script synthesis
        print(f"  ⚠️ Dedicated visual generation failed — using script-synthesis visuals")
    
    # Save updated script with (potentially) new visuals
    segments_file = project_folder / "script_segments.json"
    segments_file.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding='utf-8')
    
except Exception as e:
    print(f"  ⚠️ Visual prompt generation failed: {e}")
    print(f"  ⚠️ Continuing with script-synthesis visuals")

# ============================================================
# STEP 4.7: SCRIPT CURATION (LLM Speech Coach)
# AFTER visual prompts — only rephrases for TTS, doesn't change facts.
# Visual prompts already locked to original narrations above.
# ============================================================
print("\n🎙️ STEP 4.7: SCRIPT CURATION (Natural Delivery)")
print("-" * 40)

try:
    original_text = full_script
    print(f"  Original script: {len(original_text.split())} words")
    
    curated_text = llm.curate_script(original_text)
    
    if curated_text and curated_text != original_text:
        full_script = curated_text
        script['full_text'] = curated_text
        script['word_count'] = len(curated_text.split())
        script['estimated_duration'] = int(len(curated_text.split()) / 2.5)
        
        # Save curated script
        curated_file = project_folder / "script_curated.txt"
        curated_file.write_text(curated_text, encoding='utf-8')
        
        print(f"  ✅ Script curated: {len(curated_text.split())} words")
        print(f"  📄 Saved: {curated_file.name}")
    else:
        print(f"  ⚠️  Curation returned original, using as-is")
    
except Exception as e:
    print(f"  ⚠️  Curation failed: {e}, using original script")

# ============================================================
# STEP 5: PIXEL ART GENERATION (2 per story = 6 total)
# Uses visual prompts from Step 4.7 (dedicated) or script synthesis (fallback)
# ============================================================
print("\n🎨 STEP 5: PIXEL ART GENERATION (synced visuals)")
print("-" * 40)

try:
    generated_images = []
    image_folder = project_folder / "images"
    image_folder.mkdir(exist_ok=True)
    
    # Base seed for batch consistency
    base_seed = project_id % (2**32)
    print(f"  Batch seed: {base_seed}")
    
    # Extract visual prompts from script (new format: part_1_visual, part_2_visual)
    all_visual_scenes = script.get('all_visual_scenes', [])
    
    # Build script-aware fallback prompts for any missing scenes
    stories = script.get('stories', [])
    for s_idx in range(3):
        for p_idx in range(2):
            fallback_idx = s_idx * 2 + p_idx
            if fallback_idx >= len(all_visual_scenes):
                story = stories[s_idx] if s_idx < len(stories) else {}
                part_key = f'part_{p_idx+1}_narration'
                part_text = story.get(part_key, story.get('body', ''))
                # Extract concrete nouns/entities from narration for a specific prompt
                fallback_desc = _build_fallback_prompt(part_text, s_idx, p_idx, news_analyses)
                all_visual_scenes.append({
                    'scene': f'story_{s_idx+1}_part{p_idx+1}',
                    'description': fallback_desc
                })
    all_visual_scenes = all_visual_scenes[:6]
    
    # Scene names matching the timeline
    scene_names = []
    for i in range(3):
        scene_names.append(f'story_{i+1}_part1')
        scene_names.append(f'story_{i+1}_part2')
    
    for scene_idx, scene_name in enumerate(scene_names):
        print(f"\n  Generating {scene_name}...")
        
        # Get visual prompt
        scene_data = all_visual_scenes[scene_idx]
        prompt = scene_data.get('description', '')
        
        # Build full prompt with pixel art style
        style_suffix = ('Retro Pixel, (true 16-bit pixel art:1.5), (retro SNES style:1.3), '
                       'isometric perspective, (hard pixel edges:1.2), limited color palette, '
                       'detailed proportions, flat colors, dramatic lighting')
        full_prompt = f"{prompt}, {style_suffix}" if prompt else style_suffix
        
        # Get corresponding narration for relevance check
        story_idx = scene_idx // 2
        story_text = ''
        if story_idx < len(script.get('stories', [])):
            story = script['stories'][story_idx]
            part_key = 'part_1_narration' if scene_idx % 2 == 0 else 'part_2_narration'
            story_text = story.get(part_key, '')
        
        print(f"    Visual prompt: {full_prompt[:100]}...")
        print(f"    Narration: {story_text[:80]}...")
        
        # Generate image
        scene_seed = base_seed + scene_idx
        art_result = generate_pixel_art(full_prompt, script_text=story_text, seed=scene_seed)
        print(f"    Seed: {scene_seed}")
        
        if art_result.get('success'):
            src_path = Path(art_result.get('path'))
            dst_filename = f"{scene_name}_{src_path.name}"
            dst_path = image_folder / dst_filename
            
            import shutil
            shutil.copy2(src_path, dst_path)
            
            generated_images.append(str(dst_path))
            print(f"    ✅ {dst_filename}")
        else:
            print(f"    ❌ Failed, retrying with fallback prompt...")
            # Retry with simplified fallback prompt
            fallback_desc = _build_fallback_prompt(story_text, story_idx, scene_idx % 2, news_analyses)
            fallback_prompt = f"{fallback_desc}, {style_suffix}"
            retry_seed = base_seed + scene_idx + 100  # Different seed for retry
            art_result = generate_pixel_art(fallback_prompt, script_text=story_text, seed=retry_seed)
            
            if art_result.get('success'):
                src_path = Path(art_result.get('path'))
                dst_filename = f"{scene_name}_{src_path.name}"
                dst_path = image_folder / dst_filename
                shutil.copy2(src_path, dst_path)
                generated_images.append(str(dst_path))
                print(f"    ✅ Retry succeeded: {dst_filename}")
            else:
                # Create a solid-color placeholder so we never have missing images
                print(f"    ⚠️ Retry also failed — creating placeholder image")
                from PIL import Image as PILImage
                placeholder = PILImage.new('RGB', (1088, 1152), (10, 5, 25))
                placeholder_path = image_folder / f"{scene_name}_placeholder.png"
                placeholder.save(str(placeholder_path))
                generated_images.append(str(placeholder_path))
    
    # Final validation: ensure exactly 6 images
    if len(generated_images) < 6:
        print(f"  ⚠️ Only {len(generated_images)} images generated (need 6)")
        # Duplicate last image to fill gaps
        while len(generated_images) < 6:
            generated_images.append(generated_images[-1])
            print(f"    ⚠️ Duplicated last image to fill slot {len(generated_images)}")
    
    print(f"\n✅ Generated {len(generated_images)} synced images in {image_folder}")
    
except Exception as e:
    print(f"❌ Pixel art generation failed: {e}")
    import traceback
    traceback.print_exc()
    generated_images = []

# ============================================================
# STEP 6: VIDEO FOOTAGE (SKIP)
# ============================================================
print("\n🎥 STEP 6: VIDEO FOOTAGE (OPTIONAL)")
print("-" * 40)
print("⏭️  Skipping - using pixel art only")
downloaded_files = []

# ============================================================
# STEP 7: VOICE GENERATION
# ============================================================
print("\n🎤 STEP 7: VOICE GENERATION")
print("-" * 40)

try:
    print("Generating Masker voiceover...")
    
    tts_result = generate_voiceover(full_script, "authoritative")
    
    if tts_result.get('success'):
        src_audio = Path(tts_result.get('path'))
        dst_audio = project_folder / "voiceover.mp3"
        
        import shutil
        shutil.copy2(src_audio, dst_audio)
        
        print(f"✅ Voiceover: {dst_audio.name}")
        print(f"  Duration: ~{tts_result.get('estimated_duration_seconds', 0)}s")
        voice_file = str(dst_audio)
    else:
        print(f"❌ Voice generation failed")
        voice_file = None
    
except Exception as e:
    print(f"❌ Voice generation failed: {e}")
    voice_file = None

# ============================================================
# STEP 8: VIDEO ASSEMBLY
# ============================================================
print("\n🎬 STEP 8: VIDEO ASSEMBLY")
print("-" * 40)

if not voice_file or not generated_images:
    print("❌ Missing required assets (voice or images)")
    exit(1)

try:
    print("Assembling split-screen video...")
    
    video_filename = f"video_{project_id}.mp4"
    video_output_path = str(project_folder / video_filename)
    
    word_timestamps = tts_result.get('word_timestamps', [])
    
    # ── Build scene timestamps from segment timeline ──
    # Map each image to its start/end time based on which narration segments it covers
    scene_timestamps = None
    segment_timeline = script.get('segment_timeline', [])
    
    # Compute total audio duration from word_timestamps or TTS estimate
    if word_timestamps:
        total_dur = word_timestamps[-1].get('end', tts_result.get('estimated_duration_seconds', 90))
    else:
        total_dur = tts_result.get('estimated_duration_seconds', 90)
    print(f"  Audio duration: {total_dur:.1f}s")
    
    if segment_timeline and word_timestamps:
        # Build image-to-time mapping
        num_images = len(generated_images)
        image_times = [{'start': None, 'end': None} for _ in range(num_images)]
        
        for seg in segment_timeline:
            img_idx = seg.get('image_idx', 0)
            if img_idx >= num_images:
                continue
            
            seg_text = seg.get('text', '').strip()
            if not seg_text or seg_text == '....':
                continue
            
            # Find this segment's words in word_timestamps by matching text
            seg_words = seg_text.lower().split()
            if len(seg_words) < 2:
                continue
            
            # Find start time: first word of segment
            seg_start = None
            seg_end = None
            first_word = seg_words[0].strip('.,!?;:')
            
            for wi, wt in enumerate(word_timestamps):
                wt_word = wt.get('word', '').lower().strip('.,!?;:')
                if wt_word == first_word and wi + len(seg_words) - 1 < len(word_timestamps):
                    # Check if next few words match too
                    match = True
                    for j, sw in enumerate(seg_words[:5]):
                        if wi + j >= len(word_timestamps):
                            match = False
                            break
                        tw = word_timestamps[wi + j].get('word', '').lower().strip('.,!?;:')
                        if tw != sw.strip('.,!?;:'):
                            match = False
                            break
                    if match:
                        seg_start = wt.get('start', 0)
                        # End = last word of segment
                        end_idx = min(wi + len(seg_words) - 1, len(word_timestamps) - 1)
                        seg_end = word_timestamps[end_idx].get('end', seg_start + 5)
                        break
            
            if seg_start is not None:
                if image_times[img_idx]['start'] is None or seg_start < image_times[img_idx]['start']:
                    image_times[img_idx]['start'] = seg_start
                if seg_end is not None and (image_times[img_idx]['end'] is None or seg_end > image_times[img_idx]['end']):
                    image_times[img_idx]['end'] = seg_end
        
        # Fill gaps and validate
        for i, it in enumerate(image_times):
            if it['start'] is None:
                # Fallback: proportional split
                if word_timestamps:
                    it['start'] = (total_dur / num_images) * i
                else:
                    it['start'] = 0
            if it['end'] is None:
                if i + 1 < num_images and image_times[i + 1]['start'] is not None:
                    it['end'] = image_times[i + 1]['start']
                else:
                    it['end'] = (total_dur / num_images) * (i + 1)
        
        # Ensure last image ends at audio end
        if image_times:
            image_times[-1]['end'] = max(image_times[-1]['end'], total_dur)
            # Ensure first image starts at 0
            image_times[0]['start'] = 0
        
        # Bridge gaps: extend each image so there are NO black frames
        # If image[i] ends BEFORE image[i+1] starts, extend to cover the gap
        for i in range(len(image_times) - 1):
            current_end = image_times[i]['end']
            next_start = image_times[i + 1]['start']
            gap = next_start - current_end
            if gap > 0.1:  # More than 100ms gap = potential black frame
                # Split the gap: previous image extends 70%, next starts 30% earlier
                split_point = current_end + gap * 0.7
                image_times[i]['end'] = split_point
                image_times[i + 1]['start'] = split_point
                print(f"    ⚠️ Bridged {gap:.1f}s gap between image {i} and {i+1}")
        
        # Ensure minimum duration per image (at least 1 second)
        for i, it in enumerate(image_times):
            dur = it['end'] - it['start']
            if dur < 1.0:
                # Steal time from neighbors
                needed = 1.0 - dur
                if i > 0:
                    steal = min(needed / 2, (image_times[i-1]['end'] - image_times[i-1]['start'] - 1.0))
                    if steal > 0:
                        image_times[i-1]['end'] -= steal
                        it['start'] -= steal
                        needed -= steal
                if needed > 0 and i < len(image_times) - 1:
                    steal = min(needed, (image_times[i+1]['end'] - image_times[i+1]['start'] - 1.0))
                    if steal > 0:
                        image_times[i+1]['start'] += steal
                        it['end'] += steal
        
        scene_timestamps = image_times
        print(f"  🎯 Built scene timestamps from timeline ({len(segment_timeline)} segments)")
        for i, ts in enumerate(scene_timestamps):
            dur = ts['end'] - ts['start']
            print(f"    Image {i}: {ts['start']:.2f}s → {ts['end']:.2f}s ({dur:.2f}s)")
    else:
        print(f"  ⚠️ No segment_timeline or word_timestamps — using weighted fallback")
    
    # Title from LAST story's topic (most important — for maximum retention)
    last_analysis = news_analyses[-1] if news_analyses else {}
    hook_text = last_analysis.get('topic', '') or last_analysis.get('angle', '')
    if not hook_text:
        hooks = [s.get('part_1_narration', s.get('mini_hook', '')) for s in script.get('stories', [])]
        hook_text = hooks[-1] if hooks else ''
    print(f"  Title headline: \"{hook_text[:60]}\" (from most important story)")
    
    assembly_result = build_split_video(
        audio_path=voice_file,
        image_paths=generated_images,
        output_path=video_output_path,
        script_text=full_script,
        word_timestamps=word_timestamps,
        hook_text=hook_text,
        scene_timestamps=scene_timestamps,
    )
    
    if assembly_result.get('success'):
        final_video_path = assembly_result.get('path')
        print(f"✅ VIDEO CREATED: {final_video_path}")
        print(f"  Duration: {assembly_result.get('duration_seconds')}s")
        print(f"  Size: {assembly_result.get('file_size_mb')}MB")
        print(f"  Resolution: {assembly_result.get('resolution')}")
        print(f"  Effects: {', '.join(assembly_result.get('effects_applied', []))}")
    else:
        print(f"❌ Video assembly failed: {assembly_result.get('error')}")
        final_video_path = None
    
except Exception as e:
    print(f"❌ Video assembly failed: {e}")
    import traceback
    traceback.print_exc()
    final_video_path = None

# ============================================================
# STEP 9: PLATFORM METADATA
# ============================================================
print("\n📋 STEP 9: PLATFORM METADATA GENERATION")
print("-" * 40)

platform_metadata = {}
try:
    from redfish.platform_metadata_generator import PlatformMetadataGenerator
    
    metadata_gen = PlatformMetadataGenerator()
    # Use first analysis for metadata base
    platform_metadata = metadata_gen.generate_all_metadata(
        script, news_analyses[0] if news_analyses else {}, None
    )
    
    metadata_path = project_folder / 'platform_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(platform_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Platform metadata generated")
    print(f"  TikTok: {platform_metadata.get('tiktok', {}).get('caption', '')[:60]}...")
    print(f"  YouTube: {platform_metadata.get('youtube', {}).get('title', '')[:60]}...")
    
except Exception as e:
    print(f"⚠️  Platform metadata generation failed: {e}")

# ============================================================
# STEP 10: PROJECT SUMMARY
# ============================================================
print("\n📋 STEP 10: PROJECT SUMMARY")
print("-" * 40)

# Save project manifest
manifest = {
    'project_id': project_id,
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'format': 'multi_news_3',
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
    'project_folder': str(project_folder)
}

manifest_path = project_folder / "manifest.json"
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"✅ Manifest saved: {manifest_path}")

# Track video for diversity — log ALL 3 stories, not just the first
try:
    from redfish.category_rotation import CategoryRotation
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
        print(f"✅ Story {idx+1} tracked: {category} / {region}")
except Exception as e:
    print(f"⚠️  Video tracking failed: {e}")

print("\n🎉 COMPLETE!")
print("=" * 60)
print(f"📁 PROJECT FOLDER: {project_folder}")
print(f"📊 CONTENTS:")
print(f"  📹 Video: {video_filename if final_video_path else 'FAILED'}")
print(f"  📰 Stories: {len(news_analyses)} news items")
print(f"  🎨 Images: {len(generated_images)} files")
print(f"  🎤 Audio: voiceover.mp3")
print(f"  📋 Manifest: manifest.json")
print(f"\n🎬 FINAL VIDEO: {final_video_path if final_video_path else 'NOT CREATED'}")
print("=" * 60)