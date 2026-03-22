import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import json
import time

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

print("🎬 YT-MACHINE COMPLETE VIDEO GENERATION")
print("="*60)

# Import core components
from brain.llm_interface import LLMInterface
from video_server.pixel_art_tool import generate_pixel_art
from video_server.pexels_tool import fetch_vertical_footage
from video_server.tts_tool import generate_voiceover
from video_server.assembler_tool import build_final_video

# Initialize components
llm = LLMInterface()

# Create unique project folder
project_id = int(time.time())
project_folder = Path(f"output/projects/video_{project_id}")
project_folder.mkdir(parents=True, exist_ok=True)

print(f"📁 Project folder: {project_folder}")

print("\n📰 STEP 1: FETCHING LATEST NEWS")
print("-" * 40)

try:
    from redfish.rss_scraper import RSScraper
    scraper = RSScraper()
    
    # Check if manual selection exists
    manual_selection_file = Path("output/manual_selection.json")
    
    if manual_selection_file.exists() and '--manual' in sys.argv:
        print("Using manually selected article...")
        with open(manual_selection_file, 'r', encoding='utf-8') as f:
            article = json.load(f)
        print(f"✅ Selected: {article['title'][:80]}...")
        # Delete the selection file after use
        manual_selection_file.unlink()
    else:
        print("Scraping RSS feeds...")
        articles = scraper.scrape_all(max_age_hours=24)
        print(f"Found {len(articles)} articles")
        
        viral_articles = scraper.filter_viral_potential(articles, top_n=5)
        
        if not viral_articles:
            print("❌ No suitable articles found")
            exit(1)
        
        article = viral_articles[0]
        print(f"✅ Selected: {article['title'][:80]}...")
    
except Exception as e:
    print(f"❌ News scraping failed: {e}")
    exit(1)

print("\n🔍 STEP 2: NEWS ANALYSIS")
print("-" * 40)

try:
    print("Fetching full article text...")
    article_text = scraper.get_full_article_text(article)
    print(f"  Article length: {len(article_text)} chars")
    news_analysis = llm.process_news(article_text)
    
    if not news_analysis:
        print("❌ News analysis failed")
        exit(1)
    
    print(f"✅ Topic: {news_analysis.get('topic', 'N/A')}")
    print(f"✅ Impact Score: {news_analysis.get('impact_score', 0)}/10")
    
except Exception as e:
    print(f"❌ News analysis failed: {e}")
    exit(1)

print("\n🔬 STEP 2.5: SALIENCE EXTRACTION")
print("-" * 40)

salience_data = None
try:
    from redfish.salience_extractor import SalienceExtractor
    
    salience_ext = SalienceExtractor(llm)
    salience_data = salience_ext.extract(article_text)
    
    if salience_data:
        print(f"✅ Conflict: {salience_data.get('conflict', 'N/A')[:60]}...")
        print(f"✅ Consequence chain: {len(salience_data.get('consequence_chain', []))} steps")
        print(f"✅ Emotional anchors: {len(salience_data.get('emotional_anchors', []))} found")
        print(f"✅ Visual subjects: {len(salience_data.get('key_visual_subjects', []))} found")
    else:
        print("⚠️  Salience extraction returned empty")
    
except Exception as e:
    print(f"⚠️  Salience extraction failed: {e}")

print("\n🔬 STEP 2.7: HISTORICAL PARALLEL ANALYSIS")
print("-" * 40)

historical_parallels = None
try:
    from redfish.historical_analyzer import HistoricalAnalyzer
    
    hist_analyzer = HistoricalAnalyzer(llm)
    historical_parallels = hist_analyzer.find_historical_parallels(article_text, news_analysis, max_parallels=3)
    
    if historical_parallels and 'parallels' in historical_parallels:
        print(f"✅ Found {len(historical_parallels['parallels'])} historical parallels:")
        for i, parallel in enumerate(historical_parallels['parallels'][:2], 1):
            print(f"  {i}. {parallel.get('event_name', 'N/A')} ({parallel.get('year', 'N/A')})")
        print(f"✅ Pattern: {historical_parallels.get('historical_pattern', 'N/A')[:60]}...")
    else:
        print("⚠️  Historical analysis returned empty")
    
except Exception as e:
    print(f"⚠️  Historical analysis failed: {e}")

print("\n🔬 STEP 2.6: VISUAL ELEMENT EXTRACTION")
print("-" * 40)

try:
    from redfish.visual_extractor import VisualElementExtractor
    from redfish.prompt_generator import VisualPromptGenerator
    from redfish.prompt_validator import calculate_prompt_relevance
    
    visual_extractor = VisualElementExtractor(llm)
    visual_elements = visual_extractor.extract_visual_elements(article_text)
    
    print(f"✅ Subjects: {len(visual_elements.get('primary_subjects', []))} items")
    print(f"✅ Settings: {len(visual_elements.get('settings', []))} found")
    print(f"✅ Actions: {len(visual_elements.get('actions', []))} verbs")
    
except Exception as e:
    print(f"⚠️  Visual extraction failed: {e}")
    visual_elements = {
        'primary_subjects': [],
        'settings': [],
        'actions': [],
        'mood': 'tense',
        'temporal_context': ''
    }

print("\n🎭 STEP 3: DEBATE GENERATION")
print("-" * 40)

try:
    skeptic = llm.debate_skeptic(news_analysis)
    explainer = llm.debate_explainer(news_analysis, skeptic or {})
    print("✅ Debate components generated")
    
except Exception as e:
    print(f"❌ Debate generation failed: {e}")
    exit(1)

print("\n📝 STEP 4: SCRIPT SYNTHESIS (LLM-Generated)")
print("-" * 40)

try:
    print("Synthesizing narrative script with historical anchoring...")
    script = llm.synthesize_script(news_analysis, skeptic, explainer, salience_data, historical_parallels)
    
    if not script:
        print("❌ Script synthesis failed")
        exit(1)
    
    # Build full_text from segments if not provided by LLM
    if 'full_text' not in script or not script['full_text']:
        segments = []
        # Support both 6-segment and 5-segment structures
        if 'historical_1' in script:
            segment_names = ['hook', 'historical_1', 'historical_2', 'modern_pivot', 'consequence', 'future_outlook']
        else:
            segment_names = ['hook', 'context', 'escalation', 'consequence', 'twist']
        
        for seg in segment_names:
            text = script.get(seg, '')
            if text:
                segments.append(text)
        script['full_text'] = ' '.join(segments)
    
    full_script = script['full_text']
    
    # Calculate word count if missing
    if 'word_count' not in script:
        script['word_count'] = len(full_script.split())
        script['estimated_duration'] = int(len(full_script.split()) / 2.5)
    
    print(f"✅ Script synthesized (~{script.get('estimated_duration', 0)}s)")
    print(f"  Hook: {script.get('hook', 'N/A')[:80]}...")
    print(f"  Words: {script.get('word_count', 0)}")
    
    # Save script files for debugging
    import json
    script_file = project_folder / "script.txt"
    script_file.write_text(full_script, encoding='utf-8')
    
    segments_file = project_folder / "script_segments.json"
    segments_data = {k: v for k, v in script.items() if k in ['hook', 'historical_1', 'historical_2', 'modern_pivot', 'consequence', 'future_outlook', 'context', 'escalation', 'twist']}
    segments_file.write_text(json.dumps(segments_data, indent=2, ensure_ascii=False), encoding='utf-8')
    
    print(f"  📄 Script saved: {script_file.name}, {segments_file.name}")
    
except Exception as e:
    print(f"❌ Script synthesis failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n🎨 STEP 5: PIXEL ART GENERATION")
print("-" * 40)

try:
    # Phase 4.5: Import script parser for action-specific prompts
    from redfish.script_parser import ScriptParser
    script_parser = ScriptParser()
    parsed_segments = script_parser.parse_all_segments(script)
    print(f"✅ Script parsed: {len(parsed_segments)} segments")
    for ps in parsed_segments:
        print(f"  [{ps['segment']}] action={ps['action']} | subject={ps['subject']} | setting={ps['setting']}")

    # Initialize prompt generator with extracted elements and script
    # ScriptParser is now embedded inside VisualPromptGenerator (Phase 4.2)
    prompt_generator = VisualPromptGenerator(news_analysis, visual_elements, script)
    
    # Generate prompts for all scenes (6 or 5 depending on script structure)
    scene_prompts = prompt_generator.generate_all_scenes()
    num_scenes = len(scene_prompts)
    
    print(f"Generating {num_scenes} scenes...")
    
    generated_images = []
    image_folder = project_folder / "images"
    image_folder.mkdir(exist_ok=True)
    
    # Determine scene names based on script structure
    if 'historical_1' in script:
        scene_names = ['hook', 'historical_1', 'historical_2', 'modern_pivot', 'consequence', 'future_outlook']
    else:
        scene_names = ['hook', 'context', 'escalation', 'consequence', 'twist']
    
    for scene_name in scene_names:
        print(f"  Generating {scene_name} scene...")
        
        # Get prompt and calculate relevance
        prompt = scene_prompts[scene_name]
        relevance = calculate_prompt_relevance(prompt, article_text)
        
        print(f"    Relevance: {relevance}%")
        
        # If relevance too low, regenerate with strict constraints
        if relevance < 50:
            print(f"    ⚠️  Low relevance, regenerating...")
            prompt = prompt_generator.regenerate_strict(scene_name)
            relevance = calculate_prompt_relevance(prompt, article_text)
            print(f"    New relevance: {relevance}%")
        
        # Generate image
        art_result = generate_pixel_art(prompt)
        
        if art_result.get('success'):
            # Move image to project folder
            src_path = Path(art_result.get('path'))
            dst_filename = f"{scene_name}_{src_path.name}"
            dst_path = image_folder / dst_filename
            
            import shutil
            shutil.copy2(src_path, dst_path)
            
            generated_images.append(str(dst_path))
            print(f"    ✅ {dst_filename}")
        else:
            print(f"    ❌ Failed")
    
    print(f"✅ Generated {len(generated_images)} images in {image_folder}")
    
except Exception as e:
    print(f"❌ Pixel art generation failed: {e}")
    import traceback
    traceback.print_exc()
    generated_images = []

print("\n🎥 STEP 6: VIDEO FOOTAGE (OPTIONAL)")
print("-" * 40)

# Skip video footage for now - using only pixel art
print("⏭️  Skipping - using pixel art only")
downloaded_files = []

print("\n🎤 STEP 7: VOICE GENERATION")
print("-" * 40)

try:
    print("Generating voiceover...")
    
    tts_result = generate_voiceover(full_script, "authoritative")
    
    if tts_result.get('success'):
        # Move audio to project folder
        src_audio = Path(tts_result.get('path'))
        dst_audio = project_folder / f"voiceover.mp3"
        
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

print("\n🎬 STEP 8: VIDEO ASSEMBLY")
print("-" * 40)

if not voice_file or not generated_images:
    print("❌ Missing required assets (voice or images)")
    exit(1)

try:
    print("Assembling final video...")
    
    # Create ticker headlines
    ticker_headlines = [
        news_analysis.get('topic', 'BREAKING NEWS')[:50],
        f"Impact Score: {news_analysis.get('impact_score', 0)}/10",
        "SENTINEL v2.1 | TACTICAL BRIEFING"
    ]
    
    video_filename = f"video_{project_id}.mp4"
    
    # Extract era tags from script visual_scenes if available
    era_tags = []
    if script and 'visual_scenes' in script:
        for scene in script['visual_scenes']:
            era_tags.append(scene.get('era', '2020s'))
    
    assembly_result = build_final_video(
        audio_path=voice_file,
        asset_paths=generated_images,
        ticker_headlines=ticker_headlines,
        is_pixel_art=True,
        output_filename=video_filename,
        script_text=full_script,
        era_tags=era_tags if era_tags else None
    )
    
    if assembly_result.get('success'):
        # Move final video to project folder
        src_video = Path(assembly_result.get('path'))
        dst_video = project_folder / video_filename
        
        import shutil
        shutil.copy2(src_video, dst_video)
        
        print(f"✅ VIDEO CREATED: {dst_video}")
        print(f"  Duration: {assembly_result.get('duration_seconds')}s")
        print(f"  Size: {assembly_result.get('file_size_mb')}MB")
        print(f"  Effects: {', '.join(assembly_result.get('effects_applied', []))}")
        
        final_video_path = str(dst_video)
    else:
        print(f"❌ Video assembly failed: {assembly_result.get('error')}")
        final_video_path = None
    
except Exception as e:
    print(f"❌ Video assembly failed: {e}")
    import traceback
    traceback.print_exc()
    final_video_path = None

print("\n📋 STEP 9: PLATFORM METADATA GENERATION")
print("-" * 40)

# Generate platform-optimized metadata
platform_metadata = {}
try:
    from redfish.platform_metadata_generator import PlatformMetadataGenerator
    
    metadata_gen = PlatformMetadataGenerator()
    platform_metadata = metadata_gen.generate_all_metadata(script, news_analysis, historical_parallels)
    
    # Save platform metadata to file
    metadata_path = project_folder / 'platform_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(platform_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Platform metadata generated:")
    print(f"  TikTok: {platform_metadata['tiktok']['caption'][:60]}...")
    print(f"  YouTube: {platform_metadata['youtube']['title'][:60]}...")
    print(f"  Hashtags: {len(platform_metadata['common_hashtags'])} generated")
    
except Exception as e:
    print(f"⚠️  Platform metadata generation failed: {e}")

print("\n📋 STEP 10: PROJECT SUMMARY")
print("-" * 40)

# Track video generation for diversity enforcement
try:
    from redfish.category_rotation import CategoryRotation
    rotation = CategoryRotation()
    
    # Detect category and region from article
    matched_categories = rotation.detect_article_categories(article)
    category = matched_categories[0] if matched_categories else "other"
    region = rotation._detect_region_from_title(article.get('title', ''))
    
    # Track the video
    rotation.track_video_generated(
        topic=news_analysis.get('topic', 'Unknown'),
        category=category,
        region=region,
        article_title=article.get('title', 'Unknown')
    )
    
    print(f"✅ Video tracked for diversity enforcement:")
    print(f"  Category: {category}")
    print(f"  Region: {region}")
    
except Exception as e:
    print(f"⚠️  Video tracking failed: {e}")

# Save project manifest
manifest = {
    'project_id': project_id,
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'article': {
        'title': article.get('title'),
        'url': article.get('link'),
        'source': article.get('feed_name')
    },
    'analysis': {
        'topic': news_analysis.get('topic'),
        'impact_score': news_analysis.get('impact_score'),
        'shift_vector': news_analysis.get('shift_vector')
    },
    'historical_parallels': historical_parallels,
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

print("\n🎉 COMPLETE!")
print("="*60)
print(f"📁 PROJECT FOLDER: {project_folder}")
print(f"📊 CONTENTS:")
print(f"  📹 Video: {video_filename if final_video_path else 'FAILED'}")
print(f"  🎨 Images: {len(generated_images)} files")
print(f"  🎤 Audio: voiceover.mp3")
print(f"  📋 Manifest: manifest.json")
print(f"\n🎬 FINAL VIDEO: {final_video_path if final_video_path else 'NOT CREATED'}")
print("="*60)
