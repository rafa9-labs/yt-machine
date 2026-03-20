from rss_scraper import RSScraper
import json

def test_rss_scraper():
    print("=" * 60)
    print("REDFISH RSS SCRAPER TEST")
    print("=" * 60)
    
    scraper = RSScraper()
    
    print(f"\n[TEST 1] Loaded {len(scraper.feeds)} RSS feeds:")
    for feed in scraper.feeds:
        print(f"  - {feed['name']} ({feed['category']})")
    
    print("\n[TEST 2] Fetching articles from all feeds...")
    print("(This may take 30-60 seconds)\n")
    
    articles = scraper.scrape_all(max_age_hours=48)
    
    print(f"\n✓ Total articles fetched: {len(articles)}")
    
    if articles:
        print("\n[TEST 3] Sample articles:")
        for i, article in enumerate(articles[:5], 1):
            print(f"\n  {i}. {article['title']}")
            print(f"     Source: {article['feed_name']}")
            print(f"     Category: {article['category']}")
            print(f"     Published: {article.get('published', 'Unknown')}")
    
    print("\n[TEST 4] Filtering for viral potential...")
    viral_articles = scraper.filter_viral_potential(articles, top_n=10)
    
    print(f"\nTop 10 viral candidates:")
    for i, article in enumerate(viral_articles, 1):
        score = article.get('virality_score', 0)
        print(f"\n  {i}. [{score} points] {article['title']}")
        print(f"     {article['feed_name']} | {article['category']}")
    
    print("\n" + "=" * 60)
    print("✓ RSS SCRAPER TEST COMPLETE")
    print("=" * 60)
    
    return viral_articles

if __name__ == "__main__":
    viral_articles = test_rss_scraper()
    
    print("\n[BONUS] Sample article text:")
    if viral_articles:
        from rss_scraper import RSScraper
        scraper = RSScraper()
        sample = scraper.get_article_text(viral_articles[0])
        print("\n" + "-" * 60)
        print(sample[:500] + "...")
        print("-" * 60)
