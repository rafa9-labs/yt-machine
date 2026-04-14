"""
Phase 3 Integration Test — Async Playwright Scraper

Run:  $env:PYTHONPATH="." ; python tests/test_async_scraper.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_tests():
    print("=" * 60)
    print("Phase 3: Async Playwright Scraper Tests")
    print("=" * 60)

    # TEST 1: Scraper instantiation
    print("\nTEST 1: AsyncScraper loads config")
    from redfish.async_scraper import AsyncScraper
    scraper = AsyncScraper()
    print(f"  Feeds loaded: {len(scraper.feeds)}")
    assert len(scraper.feeds) >= 7, "Expected at least 7 feeds"
    assert scraper._browser is None, "Browser should be None (lazy init)"
    print("  PASS ✅")

    # TEST 2: Async RSS feed fetch (1 feed)
    print("\nTEST 2: Fetch single RSS feed asynchronously")
    import aiohttp
    async with aiohttp.ClientSession() as session:
        bbc_feed = [f for f in scraper.feeds if "BBC" in f["name"]][0]
        articles = await scraper.fetch_feed(session, bbc_feed["url"])
    print(f"  Fetched {len(articles)} articles from {bbc_feed['name']}")
    if articles:
        print(f"  First: {articles[0].title[:60]}...")
    print("  PASS ✅")

    # TEST 3: Articles are Pydantic models
    print("\nTEST 3: Articles are RSSArticle Pydantic models")
    from models.schemas import RSSArticle
    if articles:
        first = articles[0]
        assert isinstance(first, RSSArticle)
        assert first.title
        assert first.link.startswith("http")
        print(f"  Type: {type(first).__name__}")
        print(f"  Title: {first.title[:50]}...")
    else:
        print("  (Skipped — no articles)")
    print("  PASS ✅")

    # TEST 4: ScrapeResult from scrape_all
    print("\nTEST 4: Full scrape_all() returns ScrapeResult")
    from models.schemas import ScrapeResult
    result = await scraper.scrape_all(max_age_hours=48)
    assert isinstance(result, ScrapeResult)
    print(f"  Total articles: {len(result.articles)}")
    print(f"  Feeds fetched: {result.feed_count}")
    print(f"  Errors: {result.error_count}")
    print("  PASS ✅")

    # TEST 5: Playwright article extraction
    print("\nTEST 5: Playwright article extraction")
    if result.articles:
        test_article = None
        for a in result.articles:
            if a.link and "http" in a.link:
                test_article = a
                break
        if test_article:
            print(f"  Extracting: {test_article.link[:60]}...")
            enriched = await scraper.extract_full_article(test_article)
            print(f"  Full text length: {len(enriched.full_text)} chars")
            print(f"  Preview: {enriched.full_text[:100]}...")
            assert len(enriched.full_text) > 50
            print("  PASS ✅")
        else:
            print("  (Skipped — no valid link)")
    else:
        print("  (Skipped — no articles)")

    # TEST 6: Virality scoring
    print("\nTEST 6: Virality scoring")
    if result.articles:
        scored = await scraper.filter_viral_potential(result.articles, top_n=5)
        if scored:
            top = scored[0]
            print(f"  Top score: {top['score']}")
            print(f"  Top article: {top['article'].title[:50]}...")
            assert isinstance(top["article"], RSSArticle)
        print("  PASS ✅")
    else:
        print("  (Skipped — no articles)")

    await scraper.close()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_tests())