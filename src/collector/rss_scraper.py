import feedparser
import re
import requests
from datetime import datetime, timedelta
from html import unescape as html_unescape
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

def _strip_html(text: str) -> str:
    """Remove HTML tags, decode entities, and collapse whitespace from RSS text."""
    if not text:
        return ''
    # Decode HTML entities (&amp; -> &, etc.)
    text = html_unescape(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove leftover URLs that were in href attributes
    text = re.sub(r'https?://\S+', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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
                    "title": _strip_html(entry.get("title", "")),
                    "summary": _strip_html(entry.get("summary", entry.get("description", ""))),
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
        
        # Import rotation system
        from .category_rotation import CategoryRotation
        rotation = CategoryRotation()
        today_category = rotation.get_today_category()

        for article in articles:
            score = 0
            title_lower = article["title"].lower()
            summary_lower = article.get("summary", "").lower()
            combined = title_lower + " " + summary_lower

            # Base keyword scoring with expanded categories
            matched_categories = []
            for category, keywords in GEOPOLITICAL_KEYWORDS.items():
                for kw in keywords:
                    if kw in title_lower:
                        score += 4
                        if category not in matched_categories:
                            matched_categories.append(category)
                    elif kw in summary_lower:
                        score += 1

            # Category rotation boost (+10 for today's category)
            rotation_boost = rotation.boost_category_score(article, matched_categories)
            score += rotation_boost

            # Virality boost keywords
            for kw in VIRALITY_BOOST_KEYWORDS:
                if kw in title_lower:
                    score += 3
                elif kw in summary_lower:
                    score += 1

            # Feed priority bonus
            feed_priority = article.get("feed_priority", 2)
            if feed_priority == 1:
                score += 3

            # Title length bonus
            title_length = len(article["title"])
            if 40 <= title_length <= 100:
                score += 1

            # Diversity penalty for overused topics
            if rotation.is_topic_overused(article["title"]):
                score -= 5  # Penalize repetitive topics

            # Store extra metadata for debugging
            article["virality_score"] = score
            article["matched_categories"] = matched_categories
            article["rotation_boost"] = rotation_boost
            article["today_category"] = today_category
            
            scored_articles.append(article)

        scored_articles.sort(key=lambda x: x["virality_score"], reverse=True)

        return scored_articles[:top_n]
    
    def get_article_text(self, article: Dict[str, Any]) -> str:
        text = f"{article['title']}\n\n{article.get('summary', '')}"
        return text.strip()

    def get_full_article_text(self, article: Dict[str, Any], timeout: int = 15,
                              max_chars: int = 50000) -> str:
        """
        Fetch full article body via trafilatura (text-only, no JS execution),
        fall back to RSS title+summary.
        
        Args:
            article: Article dict with 'link', 'title', 'summary' keys
            timeout: HTTP fetch timeout in seconds
            max_chars: Maximum characters to return (safety cap)
            
        Returns:
            Full article text or RSS fallback
        """
        url = article.get('link', '')
        if url:
            try:
                import trafilatura
                # trafilatura.fetch_url uses requests internally — text-only, no JS
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    # Cap raw HTML size before extraction to avoid processing huge pages
                    if len(downloaded) > max_chars * 3:
                        downloaded = downloaded[:max_chars * 3]
                    
                    body = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=False,
                        include_links=False,
                        include_images=False,
                        no_fallback=False
                    )
                    if body and len(body) > 100:
                        # Final safety: strip any residual HTML and cap length
                        body = _strip_html(body)[:max_chars]
                        return f"{article['title']}\n\n{body}".strip()
            except Exception as e:
                print(f"  Full scrape failed for {url[:60]}...: {e}")
        
        return self.get_article_text(article)
