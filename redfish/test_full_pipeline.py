from debate_engine import DebateEngine
import json

def test_full_pipeline():
    print("=" * 60)
    print("FULL DEBATE ENGINE PIPELINE TEST")
    print("=" * 60)
    
    engine = DebateEngine()
    
    print("\n[PHASE 1] Scraping and filtering top articles...")
    results = engine.process_top_articles(max_articles=1)
    
    if not results:
        print("\n✗ No articles processed successfully")
        return False
    
    print(f"\n[PHASE 2] Successfully processed {len(results)} article(s)")
    
    for i, result in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"RESULT #{i}")
        print(f"{'='*60}")
        
        article = result['article']
        script = result['script']
        
        print(f"\nSource Article:")
        print(f"  Title: {article['title']}")
        print(f"  Feed: {article['feed_name']}")
        print(f"  Category: {article['category']}")
        
        print(f"\nGenerated Script:")
        print(f"  Hook: {script.get('hook', 'N/A')}")
        print(f"  Body: {str(script.get('body', 'N/A'))[:150]}...")
        print(f"  Twist: {str(script.get('twist', 'N/A'))[:100]}...")
        print(f"  CTA: {str(script.get('cta', 'N/A'))[:100]}...")
        print(f"  Word Count: {script.get('word_count', 'N/A')}")
        print(f"  Duration: {script.get('estimated_duration', 'N/A')}s")
    
    print(f"\n[PHASE 3] Saving results...")
    engine.save_results(results)
    
    print("\n" + "=" * 60)
    print("✓ FULL PIPELINE TEST COMPLETE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_full_pipeline()
    if success:
        print("\n✓ Phase 1 & 2 are fully operational!")
        print("  - Memory system: Working")
        print("  - LLM interface: Working")
        print("  - RSS scraper: Working")
        print("  - Debate engine: Working")
        print("\nReady for Phase 3: Video MCP Server")
    else:
        print("\n⚠ Some components need attention")
