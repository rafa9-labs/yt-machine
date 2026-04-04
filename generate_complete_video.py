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

print("🎬 YT-MACHINE COMPLETE VIDEO GENERATION (3-News Format)")
print("=" * 60)


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

print(f"\n✅ Analyzed {len(news_analyses)} stories successfully")

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
        # Build from stories
        parts = [script.get('greeting', '')]
        for story in script.get('stories', []):
            parts.append(story.get('mini_hook', ''))
            parts.append(story.get('body', ''))
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
    for i, story in enumerate(script.get('stories', []), 1):
        print(f"  Story {i} hook: {story.get('mini_hook', 'N/A')[:60]}...")
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
# STEP 4.5: SCRIPT CURATION (LLM Speech Coach)
# ============================================================
print("\n🎙️ STEP 4.5: SCRIPT CURATION (Natural Delivery)")
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
# ============================================================
print("\n🎨 STEP 5: PIXEL ART GENERATION (2 per story)")
print("-" * 40)

try:
    from redfish.prompt_generator import VisualPromptGenerator
    from redfish.prompt_validator import calculate_prompt_relevance
    
    generated_images = []
    image_folder = project_folder / "images"
    image_folder.mkdir(exist_ok=True)
    
    # Base seed for batch consistency
    base_seed = project_id % (2**32)
    print(f"  Batch seed: {base_seed}")
    
    # Extract visual scene descriptions from script
    all_visual_scenes = script.get('all_visual_scenes', [])
    
    # If no visual scenes, create from story content
    if not all_visual_scenes or len(all_visual_scenes) < 6:
        print("  ⚠️  Building visual scenes from story content...")
        all_visual_scenes = []
        for i, story in enumerate(script.get('stories', []), 1):
            # Default scenes if LLM didn't provide them
            story_scenes = story.get('visual_scenes', [])
            if story_scenes:
                all_visual_scenes.extend(story_scenes)
            else:
                # Fallback: generate from story text
                hook_text = story.get('mini_hook', '')
                body_text = story.get('body', '')
                all_visual_scenes.append({
                    'scene': f'story_{i}_hook',
                    'description': hook_text[:100]
                })
                all_visual_scenes.append({
                    'scene': f'story_{i}_consequence',
                    'description': body_text[:100]
                })
    
    # Ensure we have exactly 6 scenes
    while len(all_visual_scenes) < 6:
        idx = len(all_visual_scenes) + 1
        all_visual_scenes.append({
            'scene': f'fallback_{idx}',
            'description': f'Geopolitical scene {idx}, pixel art style, dramatic lighting'
        })
    
    # Generate 6 images (2 per story)
    scene_names_for_images = []
    for i in range(3):
        scene_names_for_images.append(f'story_{i+1}_hook')
        scene_names_for_images.append(f'story_{i+1}_consequence')
    
    for scene_idx, scene_name in enumerate(scene_names_for_images):
        print(f"\n  Generating {scene_name}...")
        
        # Get visual scene description
        scene_data = all_visual_scenes[scene_idx] if scene_idx < len(all_visual_scenes) else None
        if scene_data:
            prompt = scene_data.get('description', '')
        else:
            prompt = f'Geopolitical pixel art scene {scene_idx + 1}'
        
        # Build full prompt with pixel art style
        style_suffix = ('Retro Pixel, (true 16-bit pixel art:1.5), (retro SNES style:1.3), '
                       'isometric perspective, (hard pixel edges:1.2), limited color palette, '
                       'detailed proportions, flat colors, dramatic lighting')
        full_prompt = f"{prompt}, {style_suffix}" if prompt else style_suffix
        
        # Get corresponding story text for relevance check
        story_idx = scene_idx // 2
        story_text = ''
        if story_idx < len(script.get('stories', [])):
            story = script['stories'][story_idx]
            story_text = f"{story.get('mini_hook', '')} {story.get('body', '')}"
        
        print(f"    Prompt: {full_prompt[:100]}...")
        
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
            print(f"    ❌ Failed")
    
    print(f"\n✅ Generated {len(generated_images)} images in {image_folder}")
    
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
    
    # Build hook text from all 3 story hooks for title overlay
    hooks = [s.get('mini_hook', '') for s in script.get('stories', [])]
    hook_text = script.get('greeting', ' | '.join(hooks))
    
    assembly_result = build_split_video(
        audio_path=voice_file,
        image_paths=generated_images,
        output_path=video_output_path,
        script_text=full_script,
        word_timestamps=word_timestamps,
        hook_text=hook_text,
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

# Track video for diversity
try:
    from redfish.category_rotation import CategoryRotation
    rotation = CategoryRotation()
    if articles:
        matched_categories = rotation.detect_article_categories(articles[0])
        category = matched_categories[0] if matched_categories else "other"
        region = rotation._detect_region_from_title(articles[0].get('title', ''))
        rotation.track_video_generated(
            topic=news_analyses[0].get('topic', 'Unknown') if news_analyses else 'Unknown',
            category=category,
            region=region,
            article_title=articles[0].get('title', 'Unknown')
        )
        print(f"✅ Video tracked: {category} / {region}")
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