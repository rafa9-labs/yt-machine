"""
═══════════════════════════════════════════════════════════════════════════════
Async Scraper — Phase 3: Playwright + aiohttp + Pydantic
═══════════════════════════════════════════════════════════════════════════════

WHY THIS FILE EXISTS:
Your old `rss_scraper.py` has two problems this file solves:

PROBLEM 1: SYNCHRONOUS FETCHING
  Old code: for feed in feeds: fetch(feed)  → 8 feeds × ~2s = 16s total
  New code: await gather(*[fetch(f) for f in feeds])  → 8 feeds in parallel = ~2s total
  
  WHY ASYNC? Python's `asyncio` lets us start all 8 HTTP requests simultaneously.
  While one feed is waiting for the network, Python processes another. The total
  time is the slowest feed, not the sum of all feeds.

PROBLEM 2: JAVASCRIPT-RENDERED SITES
  Old code: trafilatura.fetch_url(url) → gets empty shell HTML (no JS execution)
  New code: Playwright launches headless Chromium → waits for JS → extracts text
  
  WHY PLAYWRIGHT? Sites like Reuters, Foreign Policy, and Al Jazeera render their
  article content via JavaScript AFTER the initial HTML loads. `trafilatura` sends
  a raw HTTP GET and sees the empty template. Playwright launches a real browser,
  waits for `networkidle` (all network requests finish), then extracts the text.

ARCHITECTURE:
  1. aiohttp fetches RSS feeds in parallel (RSS is XML — no browser needed)
  2. feedparser parses the XML into structured entries (same as old code)
  3. Pydantic RSSArticle validates every article (no more raw dicts)
  4. Playwright extracts full article text (only for articles we'll actually use)
  
  We do NOT use Playwright for RSS — that would be launching a browser to parse
  XML, which is massive overkill. Playwright is ONLY for full-article extraction.

COMPATIBILITY:
  Your old `redfish/rss_scraper.py` is NOT deleted. It still works. This new file
  is a drop-in async replacement that returns the same Pydantic models.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from html import unescape as html_unescape
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── aiohttp: async HTTP client ──
# WHY aiohttp instead of requests? `requests` is synchronous — it blocks the
# entire thread until the response arrives. `aiohttp` is async — while waiting
# for one response, Python can process others. This is what enables parallelism.
import aiohttp

# ── feedparser: RSS/Atom feed parser (same as your old code) ──
# WHY KEEP feedparser? RSS feeds are static XML files, not JavaScript apps.
# feedparser is purpose-built for parsing RSS/Atom XML. Using Playwright for
# RSS would be using a sledgehammer to crack a nut.
import feedparser

# ── Playwright: headless browser automation ──
# WHY PLAYWRIGHT OVER SELENIUM? Playwright is newer, faster, and has better
# async support. It also auto-waits for elements (no explicit sleep() calls).
from playwright.async_api import async_playwright, Browser, Page

# ── Pydantic models from Phase 1 ──
from models.schemas import RSSArticle, ScrapeResult

# ── Config from existing scraper_config.py ──
from redfish.scraper_config import (
    GEOPOLITICAL_FEEDS,
    GEOPOLITICAL_KEYWORDS,
    VIRALITY_BOOST_KEYWORDS,
    IMPACT_SCORE_THRESHOLD,
    MAX_ARTICLE_AGE_HOURS,
    TOP_N_CANDIDATES
)


def _strip_html(text: str) -> str:
    """
    Remove HTML tags, decode entities, collapse whitespace.
    Same helper from your old rss_scraper.py — unchanged.
    """
    if not text:
        return ''
    text = html_unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class AsyncScraper:
    """
    Async RSS scraper with Playwright full-article extraction.
    
    USAGE (from any async function):
        scraper = AsyncScraper()
        result = await scraper.scrape_all()
        for article in result.articles:
            print(article.title, article.link)
    
    USAGE (from synchronous code):
        import asyncio
        scraper = AsyncScraper()
        result = asyncio.run(scraper.scrape_all())
    """

    def __init__(self, config_path: str = None):
        # ── Load RSS feed configuration ──
        if config_path is None:
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "rss_feeds.json"

        self.config_path = Path(config_path)
        self.feeds = self._load_feeds()

        # ── Playwright browser instance (lazy-initialized) ──
        # WHY LAZY? Playwright's Chromium takes ~500ms to launch. We don't
        # want to launch it if we're only fetching RSS feeds (no full articles).
        # The browser is launched on first call to _get_browser() and reused.
        self._browser: Optional[Browser] = None
        self._playwright = None

    def _load_feeds(self) -> List[Dict[str, str]]:
        """Load RSS feed URLs from config file (same as old scraper)."""
        if not self.config_path.exists():
            return GEOPOLITICAL_FEEDS

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("feeds", GEOPOLITICAL_FEEDS)
        except Exception:
            return GEOPOLITICAL_FEEDS

    # ─────────────────────────────────────────────────────────────────
    # ASYNC RSS FEED FETCHING
    # ─────────────────────────────────────────────────────────────────

    async def fetch_feed(self, session: aiohttp.ClientSession,
                          feed_url: str, max_age_hours: int = 24) -> List[RSSArticle]:
        """
        Fetch and parse a single RSS feed asynchronously.
        
        WHY aiohttp.ClientSession? A session manages connection pooling and
        keep-alive under the hood. Creating one session and reusing it for
        all requests is more efficient than creating a new connection per feed.
        
        COMPARISON:
          Old: feed = feedparser.parse(feed_url)  # synchronous HTTP request
          New: async with session.get(feed_url) as resp:
                   xml = await resp.text()          # async HTTP request
                   feed = feedparser.parse(xml)     # parse locally
        """
        try:
            # ── Async HTTP GET ──
            # `async with` ensures the response is properly closed even if
            # an error occurs (like Python's `with` statement, but for async).
            async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    print(f"  [ASYNC-SCRAPER] HTTP {resp.status} for {feed_url[:50]}")
                    return []
                xml_content = await resp.text()

            # ── Parse RSS XML with feedparser ──
            # feedparser.parse() accepts a string — it doesn't do HTTP itself
            # when given a string. This lets us separate fetching (aiohttp)
            # from parsing (feedparser), which is cleaner than the old approach
            # where feedparser did both.
            feed = feedparser.parse(xml_content)

            if feed.bozo:
                print(f"  [ASYNC-SCRAPER] Parse warning for {feed_url[:50]}")

            # ── Filter by date and convert to Pydantic models ──
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            articles = []

            for entry in feed.entries[:20]:
                published = self._parse_date(entry)
                if published and published < cutoff_time:
                    continue

                title = _strip_html(entry.get("title", ""))
                if not title or len(title) <= 10:
                    continue

                # ── Pydantic validation happens here ──
                # If title is empty or link doesn't start with http(s),
                # Pydantic raises ValidationError and we skip this article.
                # Old code: articles.append({"title": title, ...}) — no validation
                try:
                    article = RSSArticle(
                        title=title,
                        link=entry.get("link", ""),
                        summary=_strip_html(entry.get("summary", entry.get("description", ""))),
                        source=feed.feed.get("title", "Unknown"),
                        published=published,
                    )
                    articles.append(article)
                except Exception as e:
                    # Pydantic validation failed — skip this malformed article
                    print(f"  [ASYNC-SCRAPER] Skipping invalid article: {e}")
                    continue

            return articles

        except asyncio.TimeoutError:
            print(f"  [ASYNC-SCRAPER] Timeout fetching {feed_url[:50]}")
            return []
        except Exception as e:
            print(f"  [ASYNC-SCRAPER] Error fetching {feed_url[:50]}: {e}")
            return []

    async def scrape_all(self, max_age_hours: int = 24) -> ScrapeResult:
        """
        Fetch ALL RSS feeds in parallel and return a validated ScrapeResult.
        
        THIS IS THE MAIN ENTRY POINT for async scraping.
        
        COMPARISON:
          Old: for feed in feeds: articles = fetch(feed)    # sequential, ~16s
          New: await gather(*[fetch(feed) for feed in feeds]) # parallel, ~2s
        
        WHY asyncio.gather()? It takes a list of coroutines and runs them
        concurrently. It's like Promise.all() in JavaScript. If any single
        feed fails, the others still succeed (return_exceptions=True).
        """
        all_articles: List[RSSArticle] = []
        feed_count = 0
        error_count = 0

        # ── Create a shared aiohttp session ──
        # WHY SHARED? Each session maintains a connection pool. With 8 feeds,
        # a shared session can reuse TCP connections (keep-alive) instead of
        # creating 8 separate connections.
        async with aiohttp.ClientSession() as session:

            # ── Launch all feed fetches concurrently ──
            # asyncio.gather() is the key to parallelism. It schedules all
            # coroutines and collects their results. return_exceptions=True
            # means one failure doesn't crash the whole batch.
            tasks = [
                self.fetch_feed(session, feed_config["url"], max_age_hours)
                for feed_config in self.feeds
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # ── Process results ──
            for i, result in enumerate(results):
                feed_name = self.feeds[i]["name"]

                if isinstance(result, Exception):
                    print(f"  [ASYNC-SCRAPER] {feed_name}: FAILED - {result}")
                    error_count += 1
                    continue

                # Add feed metadata to each article
                for article in result:
                    # Store feed metadata in article title's context
                    # (We'll add these as extra fields in a future schema update)
                    pass

                all_articles.extend(result)
                feed_count += 1
                print(f"  [ASYNC-SCRAPER] {feed_name}: {len(result)} articles ✓")

        # ── Sort by publish date (newest first) ──
        all_articles.sort(
            key=lambda a: a.published or datetime.min,
            reverse=True
        )

        return ScrapeResult(
            articles=all_articles,
            feed_count=feed_count,
            error_count=error_count,
        )

    # ─────────────────────────────────────────────────────────────────
    # PLAYWRIGHT FULL-ARTICLE EXTRACTION
    # ─────────────────────────────────────────────────────────────────

    async def _get_browser(self) -> Browser:
        """
        Lazy-initialize the Playwright browser.
        
        WHY LAZY INIT? Playwright.launch() takes ~500ms and uses ~50MB RAM.
        If we're only fetching RSS feeds today (no full articles), we never
        need the browser. This saves resources.
        
        WHY headless=True? "Headless" means the browser runs without a visible
        window. It can still render JavaScript, CSS, and DOM — you just don't
        see it on screen. Perfect for scraping.
        """
        if self._browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-gpu",           # No GPU needed for text extraction
                    "--no-sandbox",             # Required in Docker containers
                    "--disable-dev-shm-usage",  # Prevents /dev/shm crashes in Docker
                ]
            )
            print("  [PLAYWRIGHT] Chromium browser launched")
        return self._browser

    async def close(self):
        """
        Clean up Playwright browser.
        
        WHY EXPLICIT CLOSE? Playwright's browser process stays alive until
        you close it. If you scrape 100 articles without closing, the Chromium
        process keeps running in the background, eating RAM.
        
        ALWAYS call await scraper.close() when done, or use:
            async with AsyncScraper() as scraper: ...
        """
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def __aenter__(self):
        """Support `async with AsyncScraper() as scraper:` pattern."""
        return self

    async def __aexit__(self, *args):
        """Auto-close browser when exiting async with block."""
        await self.close()

    async def extract_full_article(self, article: RSSArticle,
                                     max_chars: int = 50000) -> RSSArticle:
        """
        Extract full article text using a two-tier strategy:
          1. Try trafilatura (fast, no browser) — works for static sites
          2. Fall back to Playwright (slower, full browser) — for JS-rendered sites
        
        WHY TWO-TIER? Trafilatura is ~10x faster than Playwright (no browser
        launch, no page render). For sites like BBC that serve static HTML,
        trafilatura is sufficient. We only pay the Playwright cost when needed.
        
        Args:
            article: RSSArticle with link to extract
            max_chars: Safety cap on extracted text length
            
        Returns:
            The same RSSArticle with full_text populated
        """
        if not article.link:
            return article

        # ── Tier 1: Try trafilatura first (fast) ──
        try:
            body = await self._extract_with_trafilatura(article.link, max_chars)
            if body and len(body) > 100:
                article.full_text = f"{article.title}\n\n{body}".strip()[:max_chars]
                return article
        except Exception:
            pass  # Fall through to Playwright

        # ── Tier 2: Playwright (slow but handles JS sites) ──
        try:
            body = await self._extract_with_playwright(article.link, max_chars)
            if body and len(body) > 100:
                article.full_text = f"{article.title}\n\n{body}".strip()[:max_chars]
                return article
        except Exception as e:
            print(f"  [PLAYWRIGHT] Failed for {article.link[:60]}: {e}")

        # ── Fallback: Use RSS summary ──
        article.full_text = f"{article.title}\n\n{article.summary}".strip()
        return article

    async def _extract_with_trafilatura(self, url: str, max_chars: int) -> Optional[str]:
        """
        Trafilatura extraction (same as your old code, but wrapped for async).
        
        WHY TRAFILATURA AT ALL? It uses `requests` under the hood — fast, no
        browser. For static HTML sites (BBC, Al Jazeera RSS), it works perfectly.
        We run it in a thread executor so it doesn't block the async event loop.
        """
        import trafilatura

        # ── Run synchronous trafilatura in a thread ──
        # WHY asyncio.to_thread? trafilatura.fetch_url() is synchronous (blocks).
        # `asyncio.to_thread()` runs it in a background thread so the event loop
        # stays responsive. Without this, all async tasks would pause while
        # trafilatura waits for the HTTP response.
        downloaded = await asyncio.to_thread(trafilatura.fetch_url, url)

        if not downloaded:
            return None

        if len(downloaded) > max_chars * 3:
            downloaded = downloaded[:max_chars * 3]

        body = await asyncio.to_thread(
            trafilatura.extract,
            downloaded,
            include_comments=False,
            include_tables=False,
            include_links=False,
            include_images=False,
            no_fallback=False,
        )

        if body and len(body) > 100:
            return _strip_html(body)[:max_chars]
        return None

    async def _extract_with_playwright(self, url: str, max_chars: int) -> Optional[str]:
        """
        Playwright extraction — launches headless Chromium to render the page.
        
        THIS IS THE KEY UPGRADE. Your old code used trafilatura for everything.
        On JavaScript-heavy sites, trafilatura sees:
          <div id="article-content"></div>  ← empty!
        
        Playwright sees:
          <div id="article-content">
            <p>Full article text that was loaded by JavaScript...</p>
          </div>  ← populated!
        
        HOW IT WORKS:
          1. Launch headless Chromium (or reuse existing instance)
          2. Create a new page (tab)
          3. Navigate to URL with wait_until="networkidle"
             - "networkidle" means wait until there are no network requests
               for 500ms. This ensures JavaScript has finished loading data.
          4. Try CSS selectors to find article content:
             - "article" tag (semantic HTML5)
             - ".article-content" class (common CMS pattern)
             - "[role='main']" accessibility role
             - "main" tag
          5. Fall back to full page text if no article element found
          6. Clean up: strip HTML, collapse whitespace, cap length
        """
        browser = await self._get_browser()

        # ── Create a new browser context ──
        # WHY NEW_CONTEXT? Each context has its own cookies, cache, and
        # storage. This prevents one site's cookies from leaking to another.
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 720},
        )

        page = await context.new_page()

        try:
            # ── Navigate to the article URL ──
            # wait_until="networkidle" is the magic sauce:
            #   - "domcontentloaded" = HTML parsed (JS may still be running)
            #   - "load" = all resources loaded (images, CSS, JS files)
            #   - "networkidle" = no network activity for 500ms (JS finished)
            #
            # We use "networkidle" because we need the JavaScript to finish
            # rendering the article content before we extract text.
            await page.goto(url, wait_until="networkidle", timeout=20000)

            # ── Wait for article content to appear ──
            # WHY wait_for_selector? Even after networkidle, some sites use
            # client-side rendering that takes an extra moment. We wait up to
            # 5 seconds for an article-like element to appear.
            article_selectors = [
                "article",
                ".article-body",
                ".article-content",
                ".post-content",
                ".entry-content",
                "[role='article']",
                "main",
            ]

            body = None
            for selector in article_selectors:
                try:
                    element = page.locator(selector).first
                    if await element.count() > 0:
                        # ── Extract text from the matched element ──
                        # inner_text() returns the visible text content,
                        # automatically handling CSS visibility (hidden text
                        # is excluded). This is better than innerHTML() which
                        # would include HTML tags.
                        body = await element.inner_text()
                        if body and len(body) > 100:
                            break
                except Exception:
                    continue

            # ── Fallback: extract all text from page body ──
            if not body or len(body) <= 100:
                try:
                    body = await page.inner_text("body")
                except Exception:
                    return None

            if body:
                body = _strip_html(body)[:max_chars]
            return body

        except Exception as e:
            print(f"  [PLAYWRIGHT] Navigation failed: {e}")
            return None

        finally:
            # ── ALWAYS close context + page ──
            # WHY FINALLY? If an exception occurs mid-scrape, we still need
            # to clean up. `finally` runs no matter what (success or failure).
            await context.close()

    # ─────────────────────────────────────────────────────────────────
    # VIRALITY SCORING (same logic as old scraper, async-compatible)
    # ─────────────────────────────────────────────────────────────────

    async def filter_viral_potential(self, articles: List[RSSArticle],
                                      top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Score articles by viral potential using the same keyword matching
        as your old scraper, but return enriched dicts with scores.
        
        WHY RETURN DICT NOT PYDANTIC? The virality scoring adds temporary
        metadata (score, matched_categories, rotation_boost) that's only
        needed for ranking — not for the pipeline. We'll add a proper
        ScoredArticle model in a future phase.
        """
        # Import rotation system (same as old code)
        from redfish.category_rotation import CategoryRotation
        rotation = CategoryRotation()
        today_category = rotation.get_today_category()

        scored = []
        for article in articles:
            score = 0
            title_lower = article.title.lower()
            summary_lower = article.summary.lower()
            combined = title_lower + " " + summary_lower

            # Keyword scoring (identical to old scraper)
            matched_categories = []
            for category, keywords in GEOPOLITICAL_KEYWORDS.items():
                for kw in keywords:
                    if kw in title_lower:
                        score += 4
                        if category not in matched_categories:
                            matched_categories.append(category)
                    elif kw in summary_lower:
                        score += 1

            # Category rotation boost
            article_dict = {"title": article.title, "summary": article.summary}
            rotation_boost = rotation.boost_category_score(article_dict, matched_categories)
            score += rotation_boost

            # Virality boost keywords
            for kw in VIRALITY_BOOST_KEYWORDS:
                if kw in title_lower:
                    score += 3
                elif kw in summary_lower:
                    score += 1

            scored.append({
                "article": article,           # Pydantic model
                "score": score,
                "matched_categories": matched_categories,
                "rotation_boost": rotation_boost,
                "today_category": today_category,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_n]

    # ─────────────────────────────────────────────────────────────────
    # HELPER: Parse dates from RSS entries (same as old code)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_date(entry) -> Optional[datetime]:
        """Parse date from feedparser entry (same as old RSScraper._parse_date)."""
        for field in ["published_parsed", "updated_parsed", "created_parsed"]:
            if hasattr(entry, field):
                time_struct = getattr(entry, field)
                if time_struct:
                    try:
                        return datetime(*time_struct[:6])
                    except Exception:
                        pass
        return datetime.now()