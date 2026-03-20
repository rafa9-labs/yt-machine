import os
from dotenv import load_dotenv
from pathlib import Path
import json
import time

load_dotenv()

print("🎬 YT-MACHINE VIDEO GENERATION DEMONSTRATION")
print("="*60)

# Import core components
from brain.llm_interface import LLMInterface
from video_server.pixel_art_tool import generate_pixel_art
from video_server.pexels_tool import fetch_vertical_footage
from video_server.tts_tool import generate_voiceover

# Initialize components
llm = LLMInterface()

print("\n📰 STEP 1: FETCHING LATEST NEWS")
print("-" * 40)

try:
    from redfish.rss_scraper import RSScraper
    scraper = RSScraper()
    
    print("Scraping RSS feeds for recent news...")
    articles = scraper.scrape_all(max_age_hours=24)
    print(f"Found {len(articles)} articles")
    
    # Filter for viral potential
    viral_articles = scraper.filter_viral_potential(articles, top_n=5)
    print(f"Top viral candidates: {len(viral_articles)}")
    
    if not viral_articles:
        print("❌ No suitable articles found")
        exit(1)
    
    # Select the best article
    article = viral_articles[0]
    print(f"Selected: {article['title'][:80]}...")
    print(f"Source: {article.get('feed_name', 'Unknown')}")
    
except Exception as e:
    print(f"❌ News scraping failed: {e}")
    exit(1)

print("\n🔍 STEP 2: NEWS ANALYSIS")
print("-" * 40)

try:
    article_text = scraper.get_article_text(article)
    print("Analyzing article with LLM...")
    
    news_analysis = llm.process_news(article_text)
    
    if not news_analysis:
        print("❌ News analysis failed")
        exit(1)
    
    print(f"✅ Topic: {news_analysis.get('topic', 'N/A')}")
    print(f"✅ Impact Score: {news_analysis.get('impact_score', 0)}/10")
    print(f"✅ Shift Vector: {news_analysis.get('shift_vector', 'N/A')}")
    print(f"✅ Generated {len(news_analysis.get('pixel_art_prompts', []))} pixel art prompts")
    
except Exception as e:
    print(f"❌ News analysis failed: {e}")
    exit(1)

print("\n🎭 STEP 3: DEBATE GENERATION")
print("-" * 40)

try:
    print("Generating skeptic response...")
    skeptic = llm.debate_skeptic(news_analysis)
    
    print("Generating explainer response...")
    explainer = llm.debate_explainer(news_analysis, skeptic or {})
    
    print("✅ Debate components generated")
    print(f"  Skeptic: {skeptic.get('critique', 'N/A')[:60]}...")
    print(f"  Explainer: {explainer.get('explanation', 'N/A')[:60]}...")
    
except Exception as e:
    print(f"❌ Debate generation failed: {e}")
    exit(1)

print("\n📝 STEP 4: SIMPLE SCRIPT CREATION")
print("-" * 40)

# Create a simple script from the analysis
topic = news_analysis.get('topic', 'Breaking News')
hook = f"Breaking: {topic} - Here's what you need to know."
body = f"Latest developments show {news_analysis.get('shift_vector', 'significant changes')}."
twist = f"But there's more to this story than headlines suggest."
cta = "Stay informed with our channel for real updates."

simple_script = {
    'hook': hook,
    'body': body,
    'twist': twist,
    'cta': cta,
    'full_text': f"{hook} {body} {twist} {cta}",
    'word_count': len(f"{hook} {body} {twist} {cta}".split()),
    'estimated_duration': len(f"{hook} {body} {twist} {cta}".split()) // 2.5
}

print(f"✅ Simple script created")
print(f"  Hook: {hook}")
print(f"  Body: {body}")
print(f"  Twist: {twist}")
print(f"  Duration: ~{simple_script['estimated_duration']} seconds")

print("\n🎨 STEP 5: PIXEL ART GENERATION")
print("-" * 40)

try:
    pixel_prompts = news_analysis.get('pixel_art_prompts', [])
    generated_images = []
    
    print(f"Generating {len(pixel_prompts)} pixel art scenes...")
    
    for i, prompt in enumerate(pixel_prompts[:2]):  # Limit to 2 for demo
        print(f"  Scene {i+1}: {prompt[:50]}...")
        
        art_result = generate_pixel_art(prompt)
        
        if art_result.get('success'):
            generated_images.append({
                'scene': i+1,
                'prompt': prompt,
                'filename': art_result.get('filename'),
                'path': art_result.get('path'),
                'source': art_result.get('source')
            })
            print(f"    ✅ Generated: {art_result.get('filename')} ({art_result.get('source')})")
        else:
            print(f"    ❌ Failed: {art_result.get('error', 'Unknown')}")
    
    print(f"✅ Generated {len(generated_images)} pixel art images")
    
except Exception as e:
    print(f"❌ Pixel art generation failed: {e}")
    generated_images = []

print("\n🎥 STEP 6: VIDEO FOOTAGE RETRIEVAL")
print("-" * 40)

try:
    # Extract keywords from the topic
    topic = news_analysis.get('topic', '').lower()
    keywords = ['technology', 'innovation', 'digital']  # Default keywords
    
    if 'iran' in topic or 'israel' in topic:
        keywords = ['conflict', 'military', 'middle east']
    elif 'oil' in topic or 'energy' in topic:
        keywords = ['oil', 'energy', 'industry']
    elif 'tech' in topic or 'ai' in topic:
        keywords = ['technology', 'innovation', 'digital']
    
    print(f"Fetching video footage for keywords: {keywords}")
    
    footage_result = fetch_vertical_footage(keywords, min_duration=5)
    
    if footage_result.get('success'):
        downloaded_files = footage_result.get('files', [])
        print(f"✅ Downloaded {len(downloaded_files)} video clips")
        for video in downloaded_files[:2]:  # Show first 2
            print(f"  📹 {video.get('filename')} ({video.get('duration')}s, {video.get('resolution')})")
    else:
        print(f"❌ Footage retrieval failed: {footage_result.get('error', 'Unknown')}")
        downloaded_files = []
    
except Exception as e:
    print(f"❌ Footage retrieval failed: {e}")
    downloaded_files = []

print("\n🎤 STEP 7: VOICE GENERATION")
print("-" * 40)

try:
    print("Generating voiceover...")
    print(f"Script text: {simple_script['full_text'][:100]}...")
    
    tts_result = generate_voiceover(simple_script['full_text'], "authoritative")
    
    if tts_result.get('success'):
        print(f"✅ Voiceover generated")
        print(f"  File: {tts_result.get('filename')}")
        print(f"  Duration: ~{tts_result.get('estimated_duration_seconds', 0)}s")
        print(f"  Voice: {tts_result.get('voice')}")
        voice_file = tts_result.get('path')
    else:
        print(f"❌ Voice generation failed: {tts_result.get('error', 'Unknown')}")
        voice_file = None
    
except Exception as e:
    print(f"❌ Voice generation failed: {e}")
    voice_file = None

print("\n📋 STEP 8: FINAL ASSEMBLY MANIFEST")
print("-" * 40)

# Create assembly manifest
assembly_data = {
    'metadata': {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'article_title': article.get('title'),
        'article_url': article.get('link'),
        'source_feed': article.get('feed_name'),
        'topic': news_analysis.get('topic'),
        'impact_score': news_analysis.get('impact_score'),
        'shift_vector': news_analysis.get('shift_vector'),
        'estimated_duration': simple_script['estimated_duration']
    },
    'script': simple_script,
    'assets': {
        'pixel_art': generated_images,
        'video_footage': downloaded_files,
        'voiceover': {
            'file': voice_file,
            'duration': tts_result.get('estimated_duration_seconds', 0) if tts_result else 0
        }
    },
    'debate': {
        'skeptic': skeptic,
        'explainer': explainer
    },
    'analysis': news_analysis
}

# Save assembly manifest
output_dir = Path("output/assembly")
output_dir.mkdir(parents=True, exist_ok=True)
manifest_path = output_dir / f"video_manifest_{int(time.time())}.json"

with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(assembly_data, f, indent=2, ensure_ascii=False)

print(f"✅ Assembly manifest saved: {manifest_path}")

print(f"\n🎊 GENERATION COMPLETE!")
print("="*60)
print(f"📊 SUMMARY:")
print(f"  📰 Article: {article.get('title')[:60]}...")
print(f"  🎨 Pixel Art: {len(generated_images)} images generated")
print(f"  🎥 Video Clips: {len(downloaded_files)} clips downloaded")
print(f"  🎤 Voiceover: {'✅ Generated' if voice_file else '❌ Failed'}")
print(f"  ⏱️  Duration: ~{simple_script['estimated_duration']} seconds")
print(f"  📁 Manifest: {manifest_path.name}")

print(f"\n🎬 HOW THE PROCESS WORKS:")
print(f"  1️⃣  News scraped from RSS feeds")
print(f"  2️⃣  LLM analyzes for viral potential")
print(f"  3️⃣  AI generates debate perspectives")
print(f"  4️⃣  Script created from analysis")
print(f"  5️⃣  AI generates pixel art scenes")
print(f"  6️⃣  Stock video footage downloaded")
print(f"  7️⃣  Voice synthesis creates audio")
print(f"  8️⃣  All assets saved for video assembly")

print(f"\n🚀 Next step: Use video assembler to combine all assets!")
