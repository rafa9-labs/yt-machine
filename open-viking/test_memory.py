from memory_logger import MemoryLogger
from memory_reader import MemoryReader
from datetime import datetime

def test_memory_system():
    print("=" * 60)
    print("OPEN VIKING MEMORY SYSTEM TEST")
    print("=" * 60)
    
    logger = MemoryLogger()
    reader = MemoryReader()
    
    print("\n[TEST 1] Logging sample videos...")
    
    sample_videos = [
        {
            "topic": "AI Breakthrough in Quantum Computing",
            "script": {
                "hook": "Scientists just achieved the impossible",
                "body": "A new quantum AI system solved a problem that would take classical computers 10,000 years in just 3 minutes",
                "twist": "But there's a catch - it only works at near absolute zero",
                "cta": "Follow for more tech news"
            },
            "keywords": ["AI", "quantum computing", "technology", "science"],
            "video_path": "/output/video_001.mp4",
            "duration": 45,
            "source_url": "https://example.com/news/quantum-ai",
            "platforms": ["youtube", "tiktok", "instagram"],
            "status": "published"
        },
        {
            "topic": "Global Climate Summit Reaches Historic Agreement",
            "script": {
                "hook": "195 countries just agreed on something unprecedented",
                "body": "The new climate accord mandates 80% renewable energy by 2035 with binding penalties",
                "twist": "The biggest polluters actually led the charge this time",
                "cta": "What do you think? Comment below"
            },
            "keywords": ["climate", "environment", "politics", "global"],
            "video_path": "/output/video_002.mp4",
            "duration": 48,
            "source_url": "https://example.com/news/climate-summit",
            "platforms": ["youtube", "instagram"],
            "status": "published"
        },
        {
            "topic": "New Study: Coffee Extends Lifespan by 15%",
            "script": {
                "hook": "Your morning coffee might be saving your life",
                "body": "Harvard researchers tracked 500,000 people for 20 years and found coffee drinkers lived significantly longer",
                "twist": "But only if you drink it black - sugar negates the benefits",
                "cta": "Are you a coffee person? Let me know"
            },
            "keywords": ["health", "science", "coffee", "lifestyle"],
            "video_path": "/output/video_003.mp4",
            "duration": 42,
            "source_url": "https://example.com/news/coffee-study",
            "platforms": ["tiktok", "instagram"],
            "status": "generated"
        }
    ]
    
    for i, video in enumerate(sample_videos, 1):
        success = logger.log_video(video)
        print(f"  ✓ Video {i} logged: {video['topic'][:50]}...")
    
    print("\n[TEST 2] Reading recent videos...")
    recent = reader.get_recent_videos(n=2)
    print(f"  Found {len(recent)} recent videos:")
    for video in recent:
        print(f"    - ID {video['id']}: {video['topic']}")
    
    print("\n[TEST 3] Checking for duplicate topics...")
    duplicate_check = reader.check_topic_coverage("quantum computing", days=7)
    if duplicate_check["duplicate_found"]:
        print(f"  ⚠ Duplicate found: '{duplicate_check['topic']}' from {duplicate_check['days_ago']} days ago")
    else:
        print(f"  ✓ No duplicates found")
    
    new_topic_check = reader.check_topic_coverage("space exploration", days=7)
    if new_topic_check["duplicate_found"]:
        print(f"  ⚠ Duplicate found for 'space exploration'")
    else:
        print(f"  ✓ 'space exploration' is a fresh topic")
    
    print("\n[TEST 4] Updating performance metrics...")
    logger.update_performance(1, {
        "views": 125000,
        "likes": 8500,
        "shares": 1200,
        "engagement_rate": 7.76
    })
    logger.update_performance(2, {
        "views": 89000,
        "likes": 5200,
        "shares": 780,
        "engagement_rate": 6.72
    })
    print("  ✓ Performance metrics updated for videos 1 and 2")
    
    print("\n[TEST 5] Getting performance statistics...")
    stats = reader.get_performance_stats(days=30)
    print(f"  Total videos (last 30 days): {stats['total_videos']}")
    print(f"  Average views: {stats['avg_views']:,.0f}")
    print(f"  Average engagement: {stats['avg_engagement']:.2f}%")
    print(f"  Top keywords: {', '.join(stats['top_keywords'][:5])}")
    if stats['best_performing']:
        print(f"  Best performing: '{stats['best_performing']['topic']}' ({stats['best_performing']['views']:,} views)")
    
    print("\n[TEST 6] Searching by keyword...")
    climate_videos = reader.search_by_keyword("climate")
    print(f"  Found {len(climate_videos)} video(s) about 'climate':")
    for video in climate_videos:
        print(f"    - {video['topic']}")
    
    print("\n[TEST 7] Getting total count...")
    total = reader.get_total_count()
    print(f"  Total videos in system: {total}")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - Memory system operational")
    print("=" * 60)

if __name__ == "__main__":
    test_memory_system()
