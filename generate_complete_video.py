import os
from dotenv import load_dotenv
from pathlib import Path
import json
import time

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
    
    print("Scraping RSS feeds...")
    articles = scraper.scrape_all(max_age_hours=24)
    print(f"Found {len(articles)} articles")
    
    viral_articles = scraper.filter_viral_potential(articles, top_n=5)
    
    if not viral_articles:
        print("❌ No suitable articles found")
        exit(1)
    
    article = viral_articles[0]
    print(f"✅ Selected: {article['title'][:60]}...")
    
except Exception as e:
    print(f"❌ News scraping failed: {e}")
    exit(1)

print("\n🔍 STEP 2: NEWS ANALYSIS")
print("-" * 40)

try:
    article_text = scraper.get_article_text(article)
    news_analysis = llm.process_news(article_text)
    
    if not news_analysis:
        print("❌ News analysis failed")
        exit(1)
    
    print(f"✅ Topic: {news_analysis.get('topic', 'N/A')}")
    print(f"✅ Impact Score: {news_analysis.get('impact_score', 0)}/10")
    
except Exception as e:
    print(f"❌ News analysis failed: {e}")
    exit(1)

print("\n🎭 STEP 3: DEBATE GENERATION")
print("-" * 40)

try:
    skeptic = llm.debate_skeptic(news_analysis)
    explainer = llm.debate_explainer(news_analysis, skeptic or {})
    print("✅ Debate components generated")
    
except Exception as e:
    print(f"❌ Debate generation failed: {e}")
    exit(1)

print("\n📝 STEP 4: SCRIPT CREATION")
print("-" * 40)

# Create simple script
topic = news_analysis.get('topic', 'Breaking News')
hook = f"Breaking: {topic}"
body = f"Latest developments show {news_analysis.get('shift_vector', 'significant changes')}."
twist = f"But there's more to this story than headlines suggest."
cta = "Stay informed for real updates."

full_script = f"{hook} {body} {twist} {cta}"

simple_script = {
    'hook': hook,
    'body': body,
    'twist': twist,
    'cta': cta,
    'full_text': full_script,
    'word_count': len(full_script.split()),
    'estimated_duration': len(full_script.split()) / 2.5
}

print(f"✅ Script created (~{simple_script['estimated_duration']:.0f}s)")

print("\n🎨 STEP 5: PIXEL ART GENERATION (3 SCENES)")
print("-" * 40)

try:
    pixel_prompts = news_analysis.get('pixel_art_prompts', [])
    
    # ONLY generate 3 images: hook, body, twist
    scenes_to_generate = [
        ('hook', pixel_prompts[0] if len(pixel_prompts) > 0 else 'Breaking news scene'),
        ('body', pixel_prompts[2] if len(pixel_prompts) > 2 else 'Main story scene'),
        ('twist', pixel_prompts[4] if len(pixel_prompts) > 4 else 'Conclusion scene')
    ]
    
    generated_images = []
    image_folder = project_folder / "images"
    image_folder.mkdir(exist_ok=True)
    
    for scene_name, prompt in scenes_to_generate:
        print(f"  Generating {scene_name} scene...")
        
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
    
    assembly_result = build_final_video(
        audio_path=voice_file,
        asset_paths=generated_images,
        ticker_headlines=ticker_headlines,
        is_pixel_art=True,
        output_filename=video_filename
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

print("\n📋 STEP 9: PROJECT SUMMARY")
print("-" * 40)

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
    'script': simple_script,
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
