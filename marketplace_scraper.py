"""
tools/marketplace_scraper.py — Platform Intelligence Engine for ARIA-LAB v2

Supported platforms:
  - Gumroad, Etsy, AppSumo, Product Hunt, Payhip, Lemon Squeezy
  - Reddit (old.reddit, no auth), Indie Hackers

Returns standardized ProductData objects from all platforms.
Uses browser_scraper.py for JS-rendered sites, requests for static.
Caches with 6h TTL.

Usage:
    python tools/marketplace_scraper.py gumroad --query "notion templates"
    python tools/marketplace_scraper.py etsy --query "digital planner" --limit 20
    python tools/marketplace_scraper.py producthunt --category "productivity"
    python tools/marketplace_scraper.py reddit --subreddit "entrepreneur"
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen, Request

logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "working" / "marketplace_cache"
CACHE_TTL = 21600  # 6 hours

# ──────────────────────────────────────────────
# Standardized Product Data
# ──────────────────────────────────────────────

@dataclass
class ProductData:
    """Unified product record from any marketplace."""
    name:           str
    platform:       str
    url:            str
    price:          Optional[float]  = None
    price_raw:      str              = ""
    category:       str              = ""
    sales_estimate: Optional[int]    = None
    rating:         Optional[float]  = None
    reviews_count:  Optional[int]    = None
    seller:         str              = ""
    description:    str              = ""
    tags:           list[str]        = field(default_factory=list)
    created_date:   str              = ""
    scraped_at:     float            = field(default_factory=time.time)
    raw_data:       dict             = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────

class _Cache:
    def __init__(self, ttl: int = CACHE_TTL) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _path(self, key: str) -> Path:
        return CACHE_DIR / (hashlib.md5(key.encode()).hexdigest() + ".json")

    def get(self, key: str) -> Optional[list]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if time.time() - d.get("_at", 0) > self.ttl:
                p.unlink(missing_ok=True)
                return None
            return d.get("_v")
        except Exception:
            return None

    def set(self, key: str, value: list) -> None:
        try:
            self._path(key).write_text(
                json.dumps({"_at": time.time(), "_v": value}, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception:
            pass


_cache = _Cache()

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":     "text/html,application/xhtml+xml,application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _get(url: str, json_mode: bool = False) -> Any:
    """Simple synchronous HTTP GET with headers."""
    try:
        req = Request(url, headers=_HEADERS)
        raw = urlopen(req, timeout=20).read()
        if json_mode:
            return json.loads(raw)
        return raw.decode("utf-8", errors="replace")
    except Exception as e:
        logger.error("GET %s failed: %s", url, e)
        return {} if json_mode else ""


def _parse_price(s: str) -> Optional[float]:
    """Extract decimal price from a price string."""
    if not s:
        return None
    m = re.search(r"[\d,]+(?:\.\d{2})?", s.replace(",", ""))
    if m:
        try:
            return float(m.group().replace(",", ""))
        except ValueError:
            pass
    return None


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip() if s else ""


# ──────────────────────────────────────────────
# Platform Scrapers
# ──────────────────────────────────────────────

class GumroadScraper:
    """Scrape Gumroad discover/search pages."""

    BASE = "https://gumroad.com"

    async def search(self, query: str, limit: int = 20) -> list[ProductData]:
        cache_key = f"gumroad:{query}:{limit}"
        cached = _cache.get(cache_key)
        if cached:
            return [ProductData(**d) for d in cached]

        url = f"{self.BASE}/discover?query={urllib.parse.quote(query)}"
        try:
            from browser_scraper import scrape_url
            pages = await scrape_url(url, wait_for="[data-component-name]", extract_cards=True, scroll=True)
            if not pages or not pages[0].cards:
                raise ValueError("No cards from browser, trying XHR data")
            products = self._parse_cards(pages[0].cards, pages[0].xhr_data)
        except Exception as e:
            logger.warning("Gumroad browser scrape failed: %s. Trying API.", e)
            products = await self._api_search(query, limit)

        result = products[:limit]
        _cache.set(cache_key, [p.to_dict() for p in result])
        return result

    async def _api_search(self, query: str, limit: int) -> list[ProductData]:
        """Try Gumroad's JSON API endpoint."""
        url = f"{self.BASE}/api/v2/products?query={urllib.parse.quote(query)}&per_page={limit}"
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: _get(url, json_mode=True))
        products = []
        for item in data.get("products", [])[:limit]:
            products.append(ProductData(
                name           = _clean(item.get("name", "")),
                platform       = "gumroad",
                url            = f"{self.BASE}/l/{item.get('permalink', '')}",
                price          = _parse_price(str(item.get("price", 0))),
                price_raw      = str(item.get("formatted_price", "")),
                category       = item.get("product_type", ""),
                seller         = _clean(item.get("creator_profile", {}).get("name", "")),
                description    = _clean(item.get("description", ""))[:300],
                tags           = item.get("tags", []),
                raw_data       = item,
            ))
        return products

    def _parse_cards(self, cards: list[dict], xhr: list[dict]) -> list[ProductData]:
        # Try XHR JSON first (more accurate)
        for xhr_item in xhr:
            data = xhr_item.get("data", {})
            if isinstance(data, dict) and "products" in data:
                products = []
                for item in data["products"]:
                    products.append(ProductData(
                        name      = _clean(item.get("name", "")),
                        platform  = "gumroad",
                        url       = item.get("url", ""),
                        price     = _parse_price(str(item.get("price", ""))),
                        price_raw = str(item.get("price", "")),
                        seller    = _clean(item.get("creator_name", "")),
                        raw_data  = item,
                    ))
                if products:
                    return products
        # Fall back to card parsing
        products = []
        for card in cards:
            if not card.get("title"):
                continue
            products.append(ProductData(
                name     = _clean(card["title"]),
                platform = "gumroad",
                url      = card.get("url", ""),
                price    = _parse_price(card.get("price", "")),
                price_raw = card.get("price", ""),
                seller   = _clean(card.get("text", "")[:50]),
            ))
        return products


class EtsyScraper:
    """Scrape Etsy search results with sales estimates."""

    BASE = "https://www.etsy.com"

    async def search(self, query: str, limit: int = 20) -> list[ProductData]:
        cache_key = f"etsy:{query}:{limit}"
        cached = _cache.get(cache_key)
        if cached:
            return [ProductData(**d) for d in cached]

        url = f"{self.BASE}/search?q={urllib.parse.quote(query)}&explicit=1"
        try:
            from browser_scraper import scrape_url
            pages = await scrape_url(url, wait_for=".listing-link", extract_cards=True, scroll=True)
            products = self._parse_page(pages[0] if pages else None, query)
        except Exception as e:
            logger.warning("Etsy browser scrape failed: %s", e)
            products = self._parse_html_fallback(url, query)

        result = products[:limit]
        _cache.set(cache_key, [p.to_dict() for p in result])
        return result

    def _parse_page(self, page: Any, query: str) -> list[ProductData]:
        if not page:
            return []
        products: list[ProductData] = []
        for card in page.cards:
            if not card.get("title"):
                continue
            text   = card.get("text", "")
            sales_m = re.search(r"(\d[\d,]*)\s*sales?", text, re.I)
            reviews_m = re.search(r"\((\d[\d,]*)\)", text)
            rating_m  = re.search(r"([0-9.]+)\s+out\s+of\s+5", text, re.I)
            products.append(ProductData(
                name           = _clean(card["title"]),
                platform       = "etsy",
                url            = card.get("url", ""),
                price          = _parse_price(card.get("price", "")),
                price_raw      = card.get("price", ""),
                sales_estimate = int(sales_m.group(1).replace(",","")) if sales_m else None,
                reviews_count  = int(reviews_m.group(1).replace(",","")) if reviews_m else None,
                rating         = float(rating_m.group(1)) if rating_m else None,
                category       = query,
            ))
        return products

    def _parse_html_fallback(self, url: str, query: str) -> list[ProductData]:
        """Minimal HTML parse fallback."""
        html = _get(url)
        if not html:
            return []
        products: list[ProductData] = []
        title_pattern = re.compile(r'aria-label="([^"]{5,100})"')
        price_pattern = re.compile(r'\$[\d,.]+')
        titles = title_pattern.findall(html)
        prices = price_pattern.findall(html)
        for i, title in enumerate(titles[:30]):
            products.append(ProductData(
                name     = _clean(title),
                platform = "etsy",
                url      = url,
                price    = _parse_price(prices[i]) if i < len(prices) else None,
                price_raw = prices[i] if i < len(prices) else "",
                category = query,
            ))
        return products


class AppSumoScraper:
    """Scrape AppSumo deals."""

    BASE = "https://appsumo.com"

    async def search(self, query: str = "", category: str = "", limit: int = 20) -> list[ProductData]:
        cache_key = f"appsumo:{query}:{category}:{limit}"
        cached = _cache.get(cache_key)
        if cached:
            return [ProductData(**d) for d in cached]

        url = f"{self.BASE}/search?query={urllib.parse.quote(query)}" if query else f"{self.BASE}/browse"
        try:
            from browser_scraper import scrape_url
            pages = await scrape_url(url, extract_cards=True, scroll=True)
            products = self._parse(pages[0] if pages else None)
        except Exception as e:
            logger.warning("AppSumo scrape failed: %s", e)
            products = []

        result = products[:limit]
        _cache.set(cache_key, [p.to_dict() for p in result])
        return result

    def _parse(self, page: Any) -> list[ProductData]:
        if not page:
            return []
        products: list[ProductData] = []
        for card in page.cards:
            if not card.get("title"):
                continue
            text = card.get("text", "")
            rating_m  = re.search(r"([0-9.]+)\s*/\s*5", text)
            reviews_m = re.search(r"(\d+)\s+reviews?", text, re.I)
            products.append(ProductData(
                name          = _clean(card["title"]),
                platform      = "appsumo",
                url           = card.get("url", ""),
                price         = _parse_price(card.get("price", "")),
                price_raw     = card.get("price", ""),
                rating        = float(rating_m.group(1)) if rating_m else None,
                reviews_count = int(reviews_m.group(1)) if reviews_m else None,
            ))
        return products


class ProductHuntScraper:
    """Scrape Product Hunt trending/category pages."""

    API_BASE = "https://api.producthunt.com/v2/api/graphql"
    WEB_BASE = "https://www.producthunt.com"

    async def trending(self, limit: int = 20) -> list[ProductData]:
        return await self._fetch_web("/", limit)

    async def search(self, query: str, limit: int = 20) -> list[ProductData]:
        return await self._fetch_web(f"/search?q={urllib.parse.quote(query)}", limit)

    async def _fetch_web(self, path: str, limit: int) -> list[ProductData]:
        cache_key = f"producthunt:{path}:{limit}"
        cached = _cache.get(cache_key)
        if cached:
            return [ProductData(**d) for d in cached]

        url = self.WEB_BASE + path
        try:
            from browser_scraper import scrape_url
            pages = await scrape_url(url, extract_cards=True, scroll=True)
            products = self._parse(pages[0] if pages else None)
        except Exception as e:
            logger.warning("Product Hunt scrape failed: %s", e)
            products = self._html_fallback(url)

        result = products[:limit]
        _cache.set(cache_key, [p.to_dict() for p in result])
        return result

    def _parse(self, page: Any) -> list[ProductData]:
        if not page:
            return []
        products: list[ProductData] = []
        for card in (page.cards or []):
            title = card.get("title","").strip()
            if not title or len(title) < 3:
                continue
            text = card.get("text", "")
            votes_m = re.search(r"(\d+)\s*(?:upvote|vote|▲)", text, re.I)
            products.append(ProductData(
                name          = title,
                platform      = "producthunt",
                url           = card.get("url", ""),
                description   = _clean(text[:200]),
                sales_estimate= int(votes_m.group(1)) if votes_m else None,
            ))
        return products

    def _html_fallback(self, url: str) -> list[ProductData]:
        html = _get(url)
        items = re.findall(r'"name"\s*:\s*"([^"]{3,80})".*?"tagline"\s*:\s*"([^"]{0,200})"', html, re.S)
        return [
            ProductData(name=_clean(name), platform="producthunt", url=url, description=_clean(desc))
            for name, desc in items[:20]
        ]


class RedditScraper:
    """Scrape Reddit subreddits (old.reddit, no auth required)."""

    BASE = "https://old.reddit.com"

    async def scrape_subreddit(
        self, subreddit: str, sort: str = "hot", limit: int = 25
    ) -> list[ProductData]:
        cache_key = f"reddit:{subreddit}:{sort}:{limit}"
        cached = _cache.get(cache_key)
        if cached:
            return [ProductData(**d) for d in cached]

        url  = f"{self.BASE}/r/{subreddit}/{sort}.json?limit={limit}"
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: _get(url, json_mode=True))
        products = []
        for post in data.get("data", {}).get("children", []):
            p = post.get("data", {})
            products.append(ProductData(
                name          = _clean(p.get("title", ""))[:120],
                platform      = "reddit",
                url           = f"https://reddit.com{p.get('permalink', '')}",
                seller        = p.get("author", ""),
                description   = _clean(p.get("selftext", ""))[:300],
                rating        = float(p.get("score", 0)),
                reviews_count = p.get("num_comments", 0),
                category      = subreddit,
                raw_data      = {k: p[k] for k in ("score","num_comments","created_utc","upvote_ratio")
                                 if k in p},
            ))

        _cache.set(cache_key, [p.to_dict() for p in products])
        return products

    async def search(self, query: str, subreddit: str = "", limit: int = 25) -> list[ProductData]:
        cache_key = f"reddit_search:{query}:{subreddit}:{limit}"
        cached = _cache.get(cache_key)
        if cached:
            return [ProductData(**d) for d in cached]

        base = f"{self.BASE}/r/{subreddit}" if subreddit else self.BASE
        url  = f"{base}/search.json?q={urllib.parse.quote(query)}&limit={limit}&sort=relevance"
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: _get(url, json_mode=True))
        products = []
        for post in data.get("data", {}).get("children", []):
            p = post.get("data", {})
            products.append(ProductData(
                name          = _clean(p.get("title", ""))[:120],
                platform      = "reddit",
                url           = f"https://reddit.com{p.get('permalink', '')}",
                seller        = p.get("author", ""),
                description   = _clean(p.get("selftext", ""))[:300],
                rating        = float(p.get("score", 0)),
                reviews_count = p.get("num_comments", 0),
                category      = subreddit or query,
                raw_data      = {"score": p.get("score"), "subreddit": p.get("subreddit")},
            ))
        _cache.set(cache_key, [p.to_dict() for p in products])
        return products


class IndieHackersScraper:
    """Scrape Indie Hackers product listings and revenue data."""

    BASE = "https://www.indiehackers.com"

    async def trending(self, limit: int = 20) -> list[ProductData]:
        cache_key = f"indiehackers:trending:{limit}"
        cached = _cache.get(cache_key)
        if cached:
            return [ProductData(**d) for d in cached]

        url = f"{self.BASE}/products"
        try:
            from browser_scraper import scrape_url
            pages = await scrape_url(url, extract_cards=True, scroll=True)
            products = self._parse(pages[0] if pages else None)
        except Exception as e:
            logger.warning("Indie Hackers scrape failed: %s", e)
            products = []

        result = products[:limit]
        _cache.set(cache_key, [p.to_dict() for p in result])
        return result

    def _parse(self, page: Any) -> list[ProductData]:
        if not page:
            return []
        products: list[ProductData] = []
        for card in (page.cards or []):
            title = _clean(card.get("title", ""))
            if not title:
                continue
            text = card.get("text", "")
            revenue_m = re.search(r"\$[\d,]+\s*/\s*(?:mo|month|yr|year)", text, re.I)
            products.append(ProductData(
                name          = title,
                platform      = "indiehackers",
                url           = card.get("url", ""),
                description   = _clean(text[:300]),
                price_raw     = revenue_m.group(0) if revenue_m else "",
                price         = _parse_price(revenue_m.group(0)) if revenue_m else None,
            ))
        return products


class PayhipScraper:
    """Scrape Payhip product listings."""

    BASE = "https://payhip.com"

    async def search(self, query: str, limit: int = 20) -> list[ProductData]:
        cache_key = f"payhip:{query}:{limit}"
        cached = _cache.get(cache_key)
        if cached:
            return [ProductData(**d) for d in cached]

        url = f"{self.BASE}/discover?q={urllib.parse.quote(query)}"
        try:
            from browser_scraper import scrape_url
            pages = await scrape_url(url, extract_cards=True, scroll=True)
            products = []
            for card in (pages[0].cards if pages else []):
                if not card.get("title"):
                    continue
                products.append(ProductData(
                    name     = _clean(card["title"]),
                    platform = "payhip",
                    url      = card.get("url", ""),
                    price    = _parse_price(card.get("price", "")),
                    price_raw = card.get("price", ""),
                ))
        except Exception as e:
            logger.warning("Payhip scrape failed: %s", e)
            products = []

        result = products[:limit]
        _cache.set(cache_key, [p.to_dict() for p in result])
        return result


class LemonSqueezyScraper:
    """Scrape Lemon Squeezy marketplace data."""

    BASE = "https://www.lemonsqueezy.com"

    async def search(self, query: str, limit: int = 20) -> list[ProductData]:
        cache_key = f"lemonsqueezy:{query}:{limit}"
        cached = _cache.get(cache_key)
        if cached:
            return [ProductData(**d) for d in cached]

        url = f"{self.BASE}/marketplace?search={urllib.parse.quote(query)}"
        try:
            from browser_scraper import scrape_url
            pages = await scrape_url(url, extract_cards=True, scroll=True)
            products = []
            for card in (pages[0].cards if pages else []):
                if not card.get("title"):
                    continue
                products.append(ProductData(
                    name       = _clean(card["title"]),
                    platform   = "lemonsqueezy",
                    url        = card.get("url", ""),
                    price      = _parse_price(card.get("price", "")),
                    price_raw  = card.get("price", ""),
                    description = _clean(card.get("text", ""))[:200],
                ))
        except Exception as e:
            logger.warning("Lemon Squeezy scrape failed: %s", e)
            products = []

        result = products[:limit]
        _cache.set(cache_key, [p.to_dict() for p in result])
        return result


# ──────────────────────────────────────────────
# Unified Platform Dispatcher
# ──────────────────────────────────────────────

PLATFORM_MAP = {
    "gumroad":      GumroadScraper,
    "etsy":         EtsyScraper,
    "appsumo":      AppSumoScraper,
    "producthunt":  ProductHuntScraper,
    "reddit":       RedditScraper,
    "payhip":       PayhipScraper,
    "lemonsqueezy": LemonSqueezyScraper,
    "indiehackers": IndieHackersScraper,
}


async def scrape_platform(
    platform: str,
    query:    str     = "",
    category: str     = "",
    subreddit: str    = "",
    limit:    int     = 20,
) -> list[ProductData]:
    """Unified entry point to scrape any supported marketplace."""
    platform = platform.lower().replace(" ", "").replace("-", "")
    platform = {"ph": "producthunt", "ih": "indiehackers", "ls": "lemonsqueezy"}.get(platform, platform)

    if platform not in PLATFORM_MAP:
        raise ValueError(f"Unsupported platform: {platform}. Choose from: {list(PLATFORM_MAP.keys())}")

    scraper = PLATFORM_MAP[platform]()

    if platform == "reddit":
        if subreddit:
            return await scraper.scrape_subreddit(subreddit, limit=limit)
        elif query:
            return await scraper.search(query, limit=limit)
        else:
            return await scraper.scrape_subreddit("entrepreneur", limit=limit)
    elif platform == "producthunt":
        if query:
            return await scraper.search(query, limit=limit)
        return await scraper.trending(limit=limit)
    elif platform == "indiehackers":
        return await scraper.trending(limit=limit)
    elif platform == "appsumo":
        return await scraper.search(query=query, category=category, limit=limit)
    else:
        return await scraper.search(query=query, limit=limit)


async def scrape_all_platforms(
    query:     str,
    platforms: Optional[list[str]] = None,
    limit:     int                 = 10,
) -> dict[str, list[ProductData]]:
    """Scrape multiple platforms in parallel for the same query."""
    if platforms is None:
        platforms = ["gumroad", "etsy", "appsumo", "producthunt", "reddit", "payhip"]

    tasks = {p: scrape_platform(p, query=query, limit=limit) for p in platforms}
    results: dict[str, list[ProductData]] = {}

    for platform, coro in tasks.items():
        try:
            results[platform] = await coro
            logger.info("Scraped %s: %d products", platform, len(results[platform]))
        except Exception as e:
            logger.error("Platform %s failed: %s", platform, e)
            results[platform] = []

    return results


Optional = __builtins__.__dict__.get("Optional") or type(None)  # re-import guard


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="ARIA-LAB Marketplace Intelligence Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python tools/marketplace_scraper.py gumroad --query "notion templates"
  python tools/marketplace_scraper.py etsy --query "ai planner pdf"
  python tools/marketplace_scraper.py producthunt --trending
  python tools/marketplace_scraper.py reddit --subreddit entrepreneur
  python tools/marketplace_scraper.py all --query "AI writing tool"
""",
    )
    parser.add_argument("platform", choices=list(PLATFORM_MAP.keys()) + ["all"],
                        help="Platform to scrape")
    parser.add_argument("--query",     type=str, default="", help="Search query")
    parser.add_argument("--category",  type=str, default="", help="Category filter")
    parser.add_argument("--subreddit", type=str, default="", help="Reddit subreddit")
    parser.add_argument("--limit",     type=int, default=20, help="Max results")
    parser.add_argument("--trending",  action="store_true",  help="Fetch trending (ProductHunt/IH)")
    parser.add_argument("--out",       type=Path,            help="Save results to JSON")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    async def run() -> None:
        if args.platform == "all":
            data = await scrape_all_platforms(args.query, limit=args.limit)
            all_products = [p.to_dict() for prods in data.values() for p in prods]
            if args.out:
                args.out.write_text(json.dumps(all_products, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"Saved {len(all_products)} products to {args.out}")
            else:
                for platform, products in data.items():
                    print(f"\n── {platform.upper()} ({len(products)}) ──")
                    for p in products[:5]:
                        print(f"  {p.name} — {p.price_raw or 'no price'} [{p.url[:60]}]")
        else:
            products = await scrape_platform(
                args.platform,
                query=args.query,
                category=args.category,
                subreddit=args.subreddit,
                limit=args.limit,
            )
            if args.out:
                args.out.write_text(
                    json.dumps([p.to_dict() for p in products], indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                print(f"Saved {len(products)} products to {args.out}")
            else:
                print(f"\n{args.platform.upper()} — {len(products)} results:\n")
                for p in products:
                    print(f"  {p.name}")
                    if p.price_raw:
                        print(f"    Price: {p.price_raw}")
                    if p.rating:
                        print(f"    Rating: {p.rating}")
                    if p.sales_estimate:
                        print(f"    Sales: {p.sales_estimate:,}")
                    if p.url:
                        print(f"    URL: {p.url[:80]}")
                    print()

    asyncio.run(run())


if __name__ == "__main__":
    _cli()
