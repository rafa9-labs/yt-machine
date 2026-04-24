import feedparser
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from scraper_config import (
    GEOPOLITICAL_FEEDS,
    GEOPOLITICAL_KEYWORDS,
    VIRALITY_BOOST_KEYWORDS,
    IMPACT_SCORE_THRESHOLD,
    MAX_ARTICLE_AGE_HOURS,
    TOP_N_CANDIDATES
)

class RSScraper:
    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "rss_feeds.json"
        
        self.config_path = Path(config_path)
        self.feeds = self._load_feeds()
    
    def _load_feeds(self) -> List[Dict[str, str]]:
        if not self.config_path.exists():
            return self._get_default_feeds()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("feeds", self._get_default_feeds())
        except:
            return self._get_default_feeds()
    
    def _get_default_feeds(self) -> List[Dict[str, str]]:
        return GEOPOLITICAL_FEEDS
    
    def fetch_feed(self, feed_url: str, max_age_hours: int = 24) -> List[Dict[str, Any]]:
        try:
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                print(f"Warning: Feed parsing issue for {feed_url}")
            
            articles = []
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for entry in feed.entries[:20]:
                published = self._parse_date(entry)
                
                if published and published < cutoff_time:
                    continue
                
                article = {
                    "title": entry.get("title", "").strip(),
                    "summary": entry.get("summary", entry.get("description", "")).strip(),
                    "link": entry.get("link", ""),
                    "published": published.isoformat() if published else None,
                    "source": feed.feed.get("title", "Unknown")
                }
                
                if article["title"] and len(article["title"]) > 10:
                    articles.append(article)
            
            return articles
        
        except Exception as e:
            print(f"Error fetching feed {feed_url}: {e}")
            return []
    
    def _parse_date(self, entry) -> Optional[datetime]:
        date_fields = ["published_parsed", "updated_parsed", "created_parsed"]
        
        for field in date_fields:
            if hasattr(entry, field):
                time_struct = getattr(entry, field)
                if time_struct:
                    try:
                        return datetime(*time_struct[:6])
                    except:
                        pass
        
        return datetime.now()
    
    def scrape_all(self, max_age_hours: int = 24, min_articles: int = 5) -> List[Dict[str, Any]]:
        all_articles = []
        
        for feed_config in self.feeds:
            feed_name = feed_config["name"]
            feed_url = feed_config["url"]
            category = feed_config.get("category", "general")
            
            print(f"Fetching {feed_name}...", end=" ")
            
            articles = self.fetch_feed(feed_url, max_age_hours)
            
            for article in articles:
                article["category"] = category
                article["feed_name"] = feed_name
                article["feed_priority"] = feed_config.get("priority", 2)
            
            all_articles.extend(articles)
            print(f"✓ {len(articles)} articles")
        
        all_articles = sorted(
            all_articles,
            key=lambda x: x.get("published", ""),
            reverse=True
        )
        
        return all_articles
    
    def filter_viral_potential(self, articles: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        scored_articles = []

        for article in articles:
            score = 0
            title_lower = article["title"].lower()
            summary_lower = article.get("summary", "").lower()
            combined = title_lower + " " + summary_lower

            for vector, keywords in GEOPOLITICAL_KEYWORDS.items():
                for kw in keywords:
                    if kw in title_lower:
                        score += 4
                    elif kw in summary_lower:
                        score += 1

            for kw in VIRALITY_BOOST_KEYWORDS:
                if kw in title_lower:
                    score += 3
                elif kw in summary_lower:
                    score += 1

            feed_priority = article.get("feed_priority", 2)
            if feed_priority == 1:
                score += 3

            title_length = len(article["title"])
            if 40 <= title_length <= 100:
                score += 1

            article["virality_score"] = score
            scored_articles.append(article)

        scored_articles.sort(key=lambda x: x["virality_score"], reverse=True)

        return scored_articles[:top_n]
    
    def get_article_text(self, article: Dict[str, Any]) -> str:
        text = f"{article['title']}\n\n{article.get('summary', '')}"
        return text.strip()
<<<<<<< HEAD

    def get_full_article_text(self, article: Dict[str, Any]) -> str:
        """
        Fetch full article body via trafilatura, fall back to RSS title+summary.
        
        Args:
            article: Article dict with 'link', 'title', 'summary' keys
            
        Returns:
            Full article text or RSS fallback
        """
        url = article.get('link', '')
        if url:
            try:
                import trafilatura
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    body = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=False,
                        no_fallback=False
                    )
                    if body and len(body) > 100:
                        return f"{article['title']}\n\n{body}".strip()
            except Exception as e:
                print(f"  Full scrape failed for {url[:60]}...: {e}")
        
        return self.get_article_text(article)
=======
>>>>>>> 54a25d2 (Initial commit: Add all agents and core modules)
