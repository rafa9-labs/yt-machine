import sys
from pathlib import Path
import importlib.util

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from brain.llm_interface import LLMInterface

rss_path = Path(__file__).parent / "rss_scraper.py"
spec = importlib.util.spec_from_file_location("rss_scraper", rss_path)
rss_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rss_module)
RSScraper = rss_module.RSScraper

memory_path = project_root / "open-viking" / "memory_reader.py"
spec = importlib.util.spec_from_file_location("memory_reader", memory_path)
memory_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_module)
MemoryReader = memory_module.MemoryReader
from typing import Dict, Any, Optional, List
import json

class DebateEngine:
    def __init__(self):
        self.llm = LLMInterface()
        self.scraper = RSScraper()
        self.memory = MemoryReader()
    
    def run_full_pipeline(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        print(f"\n{'='*60}")
        print(f"Processing: {article['title'][:50]}...")
        print(f"{'='*60}")
        
        duplicate_check = self.memory.check_topic_coverage(article['title'], days=7)
        if duplicate_check.get("duplicate_found"):
            print(f"⚠ Skipping - similar topic covered {duplicate_check['days_ago']} days ago")
            return None
        
        article_text = self.scraper.get_article_text(article)
        
        print("\n[1/4] Analyzing news article...")
        news_analysis = self.llm.process_news(article_text)
        if not news_analysis:
            print("  ✗ Failed to analyze article")
            return None
        print(f"  ✓ Topic: {news_analysis.get('topic', 'N/A')}")
        print(f"  ✓ Shift Vector: {news_analysis.get('shift_vector', 'N/A')}")
        print(f"  ✓ Impact Score: {news_analysis.get('impact_score', 0)}/10")
        print(f"  ✓ Contrarian Hook: {news_analysis.get('contrarian_hook', '')[:80]}")
        
        if news_analysis.get('impact_score', 0) < 5:
            print("  ⚠ Low impact score - skipping")
            return None
        
        print("\n[2/4] Running Skeptic debate...")
        skeptic = self.llm.debate_skeptic(news_analysis)
        if not skeptic:
            print("  ✗ Skeptic debate failed")
            return None
        print(f"  ✓ Critique: {skeptic.get('critique', '')[:60]}...")
        
        print("\n[3/4] Running Explainer debate...")
        explainer = self.llm.debate_explainer(news_analysis, skeptic)
        if not explainer:
            print("  ✗ Explainer debate failed")
            return None
        print(f"  ✓ Explanation: {explainer.get('explanation', '')[:60]}...")
        
        print("\n[4/4] Synthesizing final script...")
        script = self.llm.synthesize_script(news_analysis, skeptic, explainer)
        if not script:
            print("  ✗ Script synthesis failed")
            return None
        print(f"  ✓ Script generated ({script.get('word_count', 0)} words, ~{script.get('estimated_duration', 0)}s)")
        
        result = {
            "article": article,
            "analysis": news_analysis,
            "debate": {
                "skeptic": skeptic,
                "explainer": explainer
            },
            "script": script,
            "metadata": {
                "source_url": article.get("link", ""),
                "category": article.get("category", "general"),
                "feed_name": article.get("feed_name", "Unknown")
            }
        }
        
        print(f"\n{'='*60}")
        print("✓ PIPELINE COMPLETE")
        print(f"{'='*60}")
        
        return result
    
    def process_top_articles(self, max_articles: int = 3) -> List[Dict[str, Any]]:
        print("Scraping RSS feeds...")
        articles = self.scraper.scrape_all(max_age_hours=24)
        print(f"\nTotal articles fetched: {len(articles)}")
        
        print("\nFiltering for viral potential...")
        viral_articles = self.scraper.filter_viral_potential(articles, top_n=max_articles * 2)
        print(f"Top viral candidates: {len(viral_articles)}")
        
        results = []
        processed = 0
        
        for article in viral_articles:
            if processed >= max_articles:
                break
            
            result = self.run_full_pipeline(article)
            if result:
                results.append(result)
                processed += 1
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str = None):
        if output_path is None:
            base_dir = Path(__file__).parent.parent
            output_path = base_dir / "open-viking" / "resources" / "generated_scripts.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {output_path}")
