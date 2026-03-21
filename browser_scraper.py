"""
tools/browser_scraper.py — Headless Browser Engine for ARIA-LAB v2

Primary: Playwright async with stealth mode
Fallback: requests + BeautifulSoup

Features:
- SPA support (React, Next.js, Vue, Angular)
- Stealth mode: rotating user agents, disabled automation flags
- Wait strategies: selector, load state, custom conditions
- Structured data extraction: tables → JSON, cards → JSON, pricing → JSON
- Pagination: auto-click "next page" up to configurable limit
- Infinite scroll: scroll + extract until content stops
- Screenshot capability
- XHR/fetch network interception
- Cookie/session persistence
- Exponential backoff retry
- Rate limiting

Usage:
    python tools/browser_scraper.py https://example.com
    python tools/browser_scraper.py https://example.com --tables --screenshot
    python tools/browser_scraper.py https://producthunt.com/posts --paginate --max-pages 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
SCREENSHOTS = BASE_DIR / "working" / "screenshots"
CACHE_DIR   = BASE_DIR / "working" / "scrape_cache"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]


# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

@dataclass
class ScrapedPage:
    url:         str
    title:       str
    text:        str
    html:        str          = ""
    tables:      list[list[list[str]]] = field(default_factory=list)
    cards:       list[dict]            = field(default_factory=list)
    links:       list[str]             = field(default_factory=list)
    images:      list[str]             = field(default_factory=list)
    xhr_data:    list[dict]            = field(default_factory=list)
    screenshot:  Optional[str]         = None   # path to screenshot
    page_num:    int                   = 1
    scrape_time: float                 = field(default_factory=time.time)
    engine:      str                   = "playwright"

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────
# Playwright Scraper
# ──────────────────────────────────────────────

class PlaywrightScraper:
    """
    Async Playwright-based headless scraper with stealth mode,
    pagination, infinite scroll, and XHR interception.
    """

    def __init__(
        self,
        headless:    bool          = True,
        rate_limit:  float         = 1.5,
        max_retries: int           = 3,
        screenshot:  bool          = False,
        intercept_xhr: bool        = True,
    ) -> None:
        self.headless      = headless
        self.rate_limit    = rate_limit
        self.max_retries   = max_retries
        self.take_screenshot = screenshot
        self.intercept_xhr = intercept_xhr
        SCREENSHOTS.mkdir(parents=True, exist_ok=True)

    async def scrape(
        self,
        url:          str,
        wait_for:     Optional[str]  = None,
        wait_state:   str            = "domcontentloaded",
        extract_tables: bool         = True,
        extract_cards:  bool         = True,
        paginate:     bool           = False,
        max_pages:    int            = 5,
        scroll:       bool           = False,
        cookies:      Optional[list] = None,
    ) -> list[ScrapedPage]:
        """
        Scrape one or more pages, handling pagination and infinite scroll.
        Returns a list of ScrapedPage objects (one per page).
        """
        try:
            from playwright.async_api import async_playwright, TimeoutError as PWTimeout
        except ImportError:
            logger.warning("Playwright not installed — falling back to requests")
            return await FallbackScraper().scrape(url)

        pages: list[ScrapedPage] = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=self.headless,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled",
                      "--disable-infobars", "--disable-extensions",
                      "--disable-dev-shm-usage"],
            )
            ctx = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={"width": 1920, "height": 1080},
                java_script_enabled=True,
                ignore_https_errors=True,
            )
            # Apply stealth patches
            await ctx.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
            """)

            if cookies:
                await ctx.add_cookies(cookies)

            current_url = url
            for page_num in range(1, max_pages + 1):
                try:
                    result = await self._scrape_single(
                        ctx, current_url, wait_for, wait_state,
                        extract_tables, extract_cards, scroll, page_num
                    )
                    pages.append(result)

                    if not paginate:
                        break

                    # Try to find and click "Next" button
                    next_url = await self._find_next_page(ctx)
                    if not next_url or next_url == current_url:
                        break
                    current_url = next_url
                    await asyncio.sleep(self.rate_limit)

                except Exception as e:
                    logger.error("Scrape failed (page %d, %s): %s", page_num, current_url, e)
                    break

            await browser.close()

        return pages if pages else await FallbackScraper().scrape(url)

    async def _scrape_single(
        self,
        ctx:          Any,
        url:          str,
        wait_for:     Optional[str],
        wait_state:   str,
        extract_tables: bool,
        extract_cards:  bool,
        scroll:       bool,
        page_num:     int,
    ) -> ScrapedPage:
        xhr_data: list[dict] = []

        page = await ctx.new_page()
        try:
            if self.intercept_xhr:
                async def _on_response(response: Any) -> None:
                    ct = response.headers.get("content-type", "")
                    if "json" in ct and response.status == 200:
                        try:
                            body = await response.json()
                            xhr_data.append({
                                "url":    response.url,
                                "status": response.status,
                                "data":   body,
                            })
                        except Exception:
                            pass
                page.on("response", _on_response)

            # Navigate with retry
            for attempt in range(self.max_retries):
                try:
                    await page.goto(url, wait_until=wait_state, timeout=30_000)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait_s = 2 ** attempt
                    logger.warning("Navigate attempt %d failed: %s, retrying in %ds", attempt+1, e, wait_s)
                    await asyncio.sleep(wait_s)

            # Wait for specific selector if requested
            if wait_for:
                try:
                    await page.wait_for_selector(wait_for, timeout=10_000)
                except Exception:
                    logger.debug("wait_for selector '%s' not found", wait_for)

            # Handle infinite scroll
            if scroll:
                await self._infinite_scroll(page)

            title   = await page.title()
            text    = await page.inner_text("body")
            html    = await page.content()
            links   = await self._extract_links(page, url)
            images  = await self._extract_images(page)

            tables:  list[list[list[str]]] = []
            cards:   list[dict]            = []

            if extract_tables:
                tables = await self._extract_tables(page)
            if extract_cards:
                cards = await self._extract_cards(page)

            screenshot_path: Optional[str] = None
            if self.take_screenshot:
                ss_file = SCREENSHOTS / f"screen_{int(time.time())}_{page_num}.png"
                await page.screenshot(path=str(ss_file), full_page=True)
                screenshot_path = str(ss_file)

            return ScrapedPage(
                url=url, title=title, text=text[:10_000], html=html[:50_000],
                tables=tables, cards=cards, links=links, images=images,
                xhr_data=xhr_data[:20], screenshot=screenshot_path,
                page_num=page_num, engine="playwright",
            )
        finally:
            await page.close()

    async def _infinite_scroll(self, page: Any, max_scrolls: int = 20, pause: float = 1.5) -> None:
        prev_height = 0
        for _ in range(max_scrolls):
            height = await page.evaluate("document.body.scrollHeight")
            if height == prev_height:
                break
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(pause)
            prev_height = height

    async def _find_next_page(self, ctx: Any) -> Optional[str]:
        """Try to find and return the URL of the 'next page' link."""
        try:
            pages = ctx.pages
            if not pages:
                return None
            page = pages[-1]
            selectors = [
                "a[rel='next']",
                "a:text('Next')", "a:text('next')",
                "button:text('Next')", "button:text('Load more')",
                ".pagination .next", "[aria-label='Next page']",
            ]
            for sel in selectors:
                try:
                    el = await page.query_selector(sel)
                    if el:
                        href = await el.get_attribute("href")
                        if href:
                            return href
                        # Button — click it and return new URL
                        await el.click()
                        await page.wait_for_load_state("domcontentloaded")
                        return page.url
                except Exception:
                    continue
        except Exception:
            pass
        return None

    async def _extract_tables(self, page: Any) -> list[list[list[str]]]:
        """Extract all HTML tables as 2D string arrays."""
        try:
            return await page.evaluate("""() => {
                return Array.from(document.querySelectorAll('table')).map(table => {
                    return Array.from(table.querySelectorAll('tr')).map(tr => {
                        return Array.from(tr.querySelectorAll('th, td')).map(cell =>
                            cell.innerText.trim().replace(/\\s+/g, ' ')
                        );
                    }).filter(row => row.length > 0);
                }).filter(t => t.length > 0);
            }""")
        except Exception:
            return []

    async def _extract_cards(self, page: Any) -> list[dict]:
        """Extract product/content cards from common card patterns."""
        try:
            return await page.evaluate("""() => {
                const cards = [];
                const selectors = [
                    '[class*="card"]', '[class*="product"]', '[class*="item"]',
                    '[class*="listing"]', '[class*="result"]', 'article', 'li[class]'
                ];
                for (const sel of selectors) {
                    const els = document.querySelectorAll(sel);
                    if (els.length >= 3 && els.length <= 200) {
                        Array.from(els).slice(0, 50).forEach(el => {
                            const title = el.querySelector('h1,h2,h3,h4,[class*="title"],[class*="name"]');
                            const price = el.querySelector('[class*="price"],[class*="cost"]');
                            const rating = el.querySelector('[class*="rating"],[class*="star"],[aria-label*="rating"]');
                            const link = el.querySelector('a[href]');
                            if (title || price) {
                                cards.push({
                                    title:  title ? title.innerText.trim() : '',
                                    price:  price ? price.innerText.trim() : '',
                                    rating: rating ? rating.innerText.trim() : '',
                                    url:    link ? link.href : '',
                                    text:   el.innerText.trim().slice(0, 300),
                                });
                            }
                        });
                        if (cards.length >= 3) break;
                    }
                }
                return cards.slice(0, 100);
            }""")
        except Exception:
            return []

    async def _extract_links(self, page: Any, base_url: str) -> list[str]:
        try:
            links = await page.evaluate("""() =>
                Array.from(document.querySelectorAll('a[href]'))
                    .map(a => a.href)
                    .filter(h => h.startsWith('http'))
                    .slice(0, 200)
            """)
            return list(set(links))
        except Exception:
            return []

    async def _extract_images(self, page: Any) -> list[str]:
        try:
            return await page.evaluate("""() =>
                Array.from(document.querySelectorAll('img[src]'))
                    .map(i => i.src)
                    .filter(s => s.startsWith('http'))
                    .slice(0, 50)
            """)
        except Exception:
            return []


# ──────────────────────────────────────────────
# Fallback Scraper (requests + BeautifulSoup)
# ──────────────────────────────────────────────

class FallbackScraper:
    """
    Lightweight fallback using requests + BeautifulSoup.
    Works for static/SSR pages without JavaScript.
    """

    HEADERS = {"User-Agent": random.choice(USER_AGENTS)}

    async def scrape(self, url: str) -> list[ScrapedPage]:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError as e:
            logger.error("Fallback scraper dependencies missing: %s", e)
            return [ScrapedPage(url=url, title="", text="[scrape failed]", engine="fallback")]

        loop = asyncio.get_event_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: requests.get(url, headers=self.HEADERS, timeout=15, allow_redirects=True)
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error("Fallback fetch failed for %s: %s", url, e)
            return [ScrapedPage(url=url, title="", text="[fetch failed]", engine="fallback")]

        soup    = BeautifulSoup(resp.text, "html.parser")
        title   = soup.title.string.strip() if soup.title else ""
        text    = soup.get_text(separator=" ", strip=True)[:10_000]
        html    = str(soup)[:50_000]
        links   = [a["href"] for a in soup.find_all("a", href=True) if a["href"].startswith("http")][:200]
        images  = [img["src"] for img in soup.find_all("img", src=True) if img["src"].startswith("http")][:50]

        # Extract tables
        tables: list[list[list[str]]] = []
        for tbl in soup.find_all("table"):
            rows = [[cell.get_text(strip=True) for cell in row.find_all(["th","td"])]
                    for row in tbl.find_all("tr")]
            rows = [r for r in rows if r]
            if rows:
                tables.append(rows)

        return [ScrapedPage(
            url=url, title=title, text=text, html=html,
            tables=tables, links=links, images=images,
            engine="fallback",
        )]


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

async def scrape_url(
    url:          str,
    wait_for:     Optional[str] = None,
    extract_tables: bool        = True,
    extract_cards:  bool        = True,
    paginate:     bool          = False,
    max_pages:    int           = 5,
    scroll:       bool          = False,
    screenshot:   bool          = False,
    headless:     bool          = True,
    cookies:      Optional[list]= None,
    intercept_xhr:bool          = True,
) -> list[ScrapedPage]:
    """Convenience function: scrape a URL and return pages."""
    scraper = PlaywrightScraper(
        headless=headless,
        screenshot=screenshot,
        intercept_xhr=intercept_xhr,
    )
    return await scraper.scrape(
        url=url,
        wait_for=wait_for,
        extract_tables=extract_tables,
        extract_cards=extract_cards,
        paginate=paginate,
        max_pages=max_pages,
        scroll=scroll,
        cookies=cookies,
    )


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="ARIA-LAB Browser Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url",            help="URL to scrape")
    parser.add_argument("--wait",         type=str,  help="CSS selector to wait for")
    parser.add_argument("--tables",       action="store_true", help="Extract tables")
    parser.add_argument("--cards",        action="store_true", help="Extract cards")
    parser.add_argument("--paginate",     action="store_true", help="Follow pagination")
    parser.add_argument("--max-pages",    type=int, default=5, help="Max pages to scrape")
    parser.add_argument("--scroll",       action="store_true", help="Infinite scroll")
    parser.add_argument("--screenshot",   action="store_true", help="Take screenshot")
    parser.add_argument("--no-headless",  action="store_true", help="Show browser window")
    parser.add_argument("--out",          type=Path, help="Output JSON file")
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    async def run() -> None:
        pages = await scrape_url(
            url=args.url,
            wait_for=args.wait,
            extract_tables=args.tables,
            extract_cards=args.cards,
            paginate=args.paginate,
            max_pages=args.max_pages,
            scroll=args.scroll,
            screenshot=args.screenshot,
            headless=not args.no_headless,
        )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps([p.to_dict() for p in pages], indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"Saved {len(pages)} pages to {args.out}")
        else:
            for i, page in enumerate(pages, 1):
                print(f"\n{'='*60}")
                print(f"Page {i}: {page.title}")
                print(f"URL: {page.url}")
                print(f"Text preview: {page.text[:500]}…")
                if page.tables:
                    print(f"\n{len(page.tables)} tables extracted")
                if page.cards:
                    print(f"\n{len(page.cards)} cards extracted")
                if page.xhr_data:
                    print(f"\n{len(page.xhr_data)} XHR responses captured")
                if page.screenshot:
                    print(f"\nScreenshot: {page.screenshot}")

    asyncio.run(run())


if __name__ == "__main__":
    _cli()
