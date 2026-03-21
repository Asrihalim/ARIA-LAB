"""
tools/web_research.py — Deep Research Engine for ARIA-LAB v2

Primary: Exa AI API (EXA_API_KEY)
Fallback: SerpAPI (SERPAPI_KEY)
Final fallback: DuckDuckGo (no key required)

Generates 5–10 queries, executes in parallel, scores/ranks results,
extracts full page content, caches with 24h TTL, deduplicates.

Usage:
    python tools/web_research.py "best AI productivity tools 2024"
    python tools/web_research.py --goal "find top Gumroad products" --queries 8
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
CACHE_DIR  = BASE_DIR / "working" / "web_cache"
CACHE_TTL  = 86400  # 24 hours

# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

@dataclass
class ResearchResult:
    url:               str
    title:             str
    snippet:           str
    full_content:      str
    quality_score:     float
    extracted_entities: list[dict]        = field(default_factory=list)
    source_engine:     str               = ""
    query:             str               = ""
    fetched_at:        float             = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────

class DiskCache:
    """Simple file-based cache with TTL."""

    def __init__(self, cache_dir: Path, ttl: int = CACHE_TTL) -> None:
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._key_path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if time.time() - data.get("_cached_at", 0) > self.ttl:
                p.unlink(missing_ok=True)
                return None
            return data.get("_value")
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: Any) -> None:
        p = self._key_path(key)
        try:
            p.write_text(
                json.dumps({"_cached_at": time.time(), "_value": value}, ensure_ascii=False),
                encoding="utf-8"
            )
        except OSError as e:
            logger.warning("Cache write failed: %s", e)

    def invalidate(self, key: str) -> None:
        self._key_path(key).unlink(missing_ok=True)


# ──────────────────────────────────────────────
# Query Generator
# ──────────────────────────────────────────────

class QueryGenerator:
    """Generate diverse, targeted search queries from a research goal."""

    TEMPLATES = [
        "{goal}",
        "{goal} 2024 2025",
        "{goal} review comparison",
        "{goal} best alternatives",
        "{goal} market analysis",
        "{goal} top products list",
        "{goal} case study results",
        "{goal} pricing revenue data",
        "site:reddit.com {goal}",
        "site:news.ycombinator.com {goal}",
    ]

    def generate(self, goal: str, n: int = 7) -> list[str]:
        """Generate n diverse queries from a research goal."""
        goal = goal.strip()
        queries: list[str] = []

        for tpl in self.TEMPLATES[:n]:
            q = tpl.format(goal=goal)
            if q not in queries:
                queries.append(q)

        # Add keyword permutations
        words = goal.split()
        if len(words) >= 3:
            flipped = " ".join(words[1:] + [words[0]])
            if flipped not in queries and len(queries) < n:
                queries.append(flipped)

        return queries[:n]


# ──────────────────────────────────────────────
# Search Engines
# ──────────────────────────────────────────────

class ExaSearchEngine:
    """Exa AI neural search API."""

    BASE_URL = "https://api.exa.ai/search"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(self, query: str, num_results: int = 10) -> list[dict]:
        payload = {
            "query":          query,
            "numResults":     num_results,
            "useAutoprompt":  True,
            "type":           "neural",
            "contents": {
                "text":      {"maxCharacters": 3000},
                "highlights": {"numSentences": 3, "highlightsPerUrl": 2},
            },
        }
        headers = {
            "Content-Type":  "application/json",
            "x-api-key":      self.api_key,
            "Accept":         "application/json",
        }
        try:
            body = json.dumps(payload).encode()
            req  = Request(self.BASE_URL, data=body, headers=headers, method="POST")
            loop = asyncio.get_event_loop()
            raw  = await loop.run_in_executor(None, lambda: urlopen(req, timeout=20).read())
            data = json.loads(raw)
            results = []
            for item in data.get("results", []):
                text    = item.get("text", "") or ""
                excerpt = item.get("highlights", [])
                snippet = " ".join(excerpt) if excerpt else text[:300]
                results.append({
                    "url":      item.get("url", ""),
                    "title":    item.get("title", ""),
                    "snippet":  snippet,
                    "full_content": text,
                    "score":    item.get("score", 0.5),
                })
            return results
        except (HTTPError, URLError, json.JSONDecodeError, Exception) as e:
            logger.error("Exa search failed for '%s': %s", query, e)
            return []


class SerpAPIEngine:
    """SerpAPI Google search wrapper."""

    BASE_URL = "https://serpapi.com/search"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(self, query: str, num_results: int = 10) -> list[dict]:
        params = urllib.parse.urlencode({
            "q":        query,
            "api_key":  self.api_key,
            "num":      num_results,
            "engine":   "google",
            "hl":       "en",
            "gl":       "us",
        })
        url = f"{self.BASE_URL}?{params}"
        try:
            loop = asyncio.get_event_loop()
            req  = Request(url, headers={"Accept": "application/json"})
            raw  = await loop.run_in_executor(None, lambda: urlopen(req, timeout=20).read())
            data = json.loads(raw)
            results = []
            for item in data.get("organic_results", [])[:num_results]:
                results.append({
                    "url":          item.get("link", ""),
                    "title":        item.get("title", ""),
                    "snippet":      item.get("snippet", ""),
                    "full_content": item.get("snippet", ""),
                    "score":        0.5,
                })
            return results
        except Exception as e:
            logger.error("SerpAPI failed for '%s': %s", query, e)
            return []


class DuckDuckGoEngine:
    """DuckDuckGo search via HTML scraping (no API key required)."""

    BASE_URL = "https://html.duckduckgo.com/html/"

    async def search(self, query: str, num_results: int = 10) -> list[dict]:
        params  = urllib.parse.urlencode({"q": query, "kl": "us-en"})
        headers = {
            "User-Agent":  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept":      "text/html",
            "Referer":     "https://duckduckgo.com/",
        }
        try:
            body = params.encode()
            req  = Request(self.BASE_URL, data=body, headers=headers, method="POST")
            loop = asyncio.get_event_loop()
            raw  = await loop.run_in_executor(None, lambda: urlopen(req, timeout=15).read())
            html = raw.decode("utf-8", errors="replace")
            return self._parse_html(html, num_results)
        except Exception as e:
            logger.error("DuckDuckGo search failed for '%s': %s", query, e)
            return []

    def _parse_html(self, html: str, limit: int) -> list[dict]:
        results: list[dict] = []
        # Extract result links
        link_pattern   = re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
        snippet_pattern = re.compile(r'class="result__snippet"[^>]*>(.*?)</[a-z]+>', re.S)
        links   = link_pattern.findall(html)
        snippets_raw = snippet_pattern.findall(html)

        def clean(s: str) -> str:
            return re.sub(r"<[^>]+>", "", s).strip()

        for i, (url, title) in enumerate(links[:limit]):
            snip = snippets_raw[i] if i < len(snippets_raw) else ""
            results.append({
                "url":          url,
                "title":        clean(title),
                "snippet":      clean(snip),
                "full_content": clean(snip),
                "score":        0.3,
            })
        return results


# ──────────────────────────────────────────────
# Content Fetcher
# ──────────────────────────────────────────────

class ContentFetcher:
    """Fetches and cleans full-page text content from URLs."""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept":     "text/html,application/xhtml+xml,*/*",
    }

    async def fetch(self, url: str, max_chars: int = 5000) -> str:
        """Fetch URL and return cleaned text."""
        try:
            loop = asyncio.get_event_loop()
            req  = Request(url, headers=self.HEADERS)
            raw  = await loop.run_in_executor(None, lambda: urlopen(req, timeout=15).read())
            html = raw.decode("utf-8", errors="replace")
            return self._extract_text(html)[:max_chars]
        except Exception as e:
            logger.debug("Content fetch failed for %s: %s", url, e)
            return ""

    def _extract_text(self, html: str) -> str:
        """Extract readable text from HTML."""
        # Remove scripts, styles, nav, footer
        html = re.sub(r"<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>", "", html,
                      flags=re.S | re.I)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove HTML entities
        text = re.sub(r"&[a-z]+;", " ", text)
        return text.strip()


# ──────────────────────────────────────────────
# Quality Scorer
# ──────────────────────────────────────────────

class QualityScorer:
    """Score search results by relevance and source quality."""

    HIGH_QUALITY_DOMAINS = {
        "github.com", "stackoverflow.com", "medium.com", "techcrunch.com",
        "producthunt.com", "indiehackers.com", "a16z.com", "ycombinator.com",
        "reddit.com", "twitter.com", "linkedin.com", "forbes.com", "wired.com",
    }

    LOW_QUALITY_PATTERNS = [
        r"download.*free", r"click here", r"buy now", r"sign up",
        r"spam", r"popup", r"advertisement",
    ]

    def score(self, result: dict, query: str) -> float:
        """Return a quality score 0.0–1.0 for a search result."""
        score = result.get("score", 0.3)

        url     = result.get("url", "").lower()
        title   = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        content = result.get("full_content", "").lower()

        # Domain quality bonus
        try:
            domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
            if domain in self.HIGH_QUALITY_DOMAINS:
                score += 0.2
        except Exception:
            pass

        # Query keyword presence bonus
        query_words = set(query.lower().split())
        combined = title + " " + snippet
        matches = sum(1 for w in query_words if w in combined)
        score += (matches / max(len(query_words), 1)) * 0.3

        # Content length bonus (more content ≈ more useful)
        content_len = len(content)
        if content_len > 1000:
            score += 0.1
        if content_len > 3000:
            score += 0.1

        # Low-quality penalties
        for pattern in self.LOW_QUALITY_PATTERNS:
            if re.search(pattern, snippet):
                score -= 0.15
                break

        # HTTPS bonus
        if url.startswith("https://"):
            score += 0.05

        return min(max(score, 0.0), 1.0)


# ──────────────────────────────────────────────
# Entity Extractor (lightweight)
# ──────────────────────────────────────────────

class EntityExtractor:
    """Simple regex-based entity extraction from text."""

    PRICE_RE    = re.compile(r"\$[\d,]+(?:\.\d{2})?(?:/\w+)?|\d+\s*(?:dollars?|USD)")
    PERCENT_RE  = re.compile(r"\d+(?:\.\d+)?\s*%")
    URL_RE      = re.compile(r"https?://[^\s\"'>]+")
    NUMBER_RE   = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")
    EMAIL_RE    = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")

    def extract(self, text: str) -> list[dict]:
        entities: list[dict] = []
        for price in self.PRICE_RE.findall(text)[:5]:
            entities.append({"type": "price",   "value": price})
        for pct in self.PERCENT_RE.findall(text)[:5]:
            entities.append({"type": "percent", "value": pct})
        for url in self.URL_RE.findall(text)[:3]:
            entities.append({"type": "url",     "value": url})
        for email in self.EMAIL_RE.findall(text)[:3]:
            entities.append({"type": "email",   "value": email})
        return entities


# ──────────────────────────────────────────────
# Main Research Engine
# ──────────────────────────────────────────────

class WebResearchEngine:
    """
    Deep research engine with multi-engine fallback, parallel execution,
    caching, deduplication, scoring, and entity extraction.
    """

    def __init__(
        self,
        exa_key:    Optional[str] = None,
        serp_key:   Optional[str] = None,
        cache_ttl:  int           = CACHE_TTL,
        cache_dir:  Optional[Path] = None,
        rate_limit: float          = 0.5,
    ) -> None:
        self.exa_key   = exa_key   or os.environ.get("EXA_API_KEY", "")
        self.serp_key  = serp_key  or os.environ.get("SERPAPI_KEY", "")
        self.cache     = DiskCache(cache_dir or CACHE_DIR, ttl=cache_ttl)
        self.rate_limit = rate_limit
        self.query_gen = QueryGenerator()
        self.scorer    = QualityScorer()
        self.fetcher   = ContentFetcher()
        self.extractor = EntityExtractor()

    async def research(
        self,
        goal:        str,
        num_queries: int = 7,
        results_per_query: int = 8,
        fetch_content: bool = True,
    ) -> list[ResearchResult]:
        """
        Full research pipeline:
        1. Generate queries
        2. Execute all queries in parallel across preferred engine
        3. Score, deduplicate, and sort results
        4. Fetch full content for top results
        5. Extract entities
        """
        queries = self.query_gen.generate(goal, n=num_queries)
        logger.info("Generated %d queries for goal: %s", len(queries), goal[:60])

        # Execute all queries in parallel
        raw_results = await self._execute_queries(queries, results_per_query)

        # Score results
        scored: list[tuple[float, dict]] = []
        seen_urls: set[str] = set()
        for res in raw_results:
            url = res.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            score = self.scorer.score(res, goal)
            scored.append((score, res))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:30]  # keep top 30 unique results

        # Fetch full content for top results
        results: list[ResearchResult] = []
        fetch_tasks = []
        if fetch_content:
            for score, res in top[:15]:  # fetch top 15
                fetch_tasks.append(self._enrich_result(score, res))
            items = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for item in items:
                if isinstance(item, ResearchResult):
                    results.append(item)
        else:
            for score, res in top:
                results.append(self._dict_to_result(score, res, goal))

        logger.info("Research complete: %d results", len(results))
        return results

    async def _execute_queries(self, queries: list[str], n: int) -> list[dict]:
        """Execute all queries in parallel, with engine fallback per query."""
        tasks = [self._search_with_fallback(q, n) for q in queries]
        results_nested = await asyncio.gather(*tasks, return_exceptions=True)
        flat: list[dict] = []
        for batch in results_nested:
            if isinstance(batch, list):
                flat.extend(batch)
        return flat

    async def _search_with_fallback(self, query: str, n: int) -> list[dict]:
        """Try Exa → SerpAPI → DuckDuckGo for a single query."""
        # Check cache first
        cached = self.cache.get(f"search:{query}")
        if cached is not None:
            logger.debug("Cache hit for query: %s", query[:50])
            return cached

        results: list[dict] = []

        if self.exa_key:
            results = await ExaSearchEngine(self.exa_key).search(query, num_results=n)
            if results:
                for r in results:
                    r["_engine"] = "exa"

        if not results and self.serp_key:
            results = await SerpAPIEngine(self.serp_key).search(query, num_results=n)
            if results:
                for r in results:
                    r["_engine"] = "serpapi"

        if not results:
            results = await DuckDuckGoEngine().search(query, num_results=n)
            for r in results:
                r["_engine"] = "duckduckgo"

        if results:
            self.cache.set(f"search:{query}", results)

        await asyncio.sleep(self.rate_limit)
        return results

    async def _enrich_result(self, score: float, res: dict) -> ResearchResult:
        """Fetch full content and extract entities for a single result."""
        url     = res.get("url", "")
        content = res.get("full_content", "")

        # Only fetch if content is sparse
        if len(content) < 500 and url:
            fetched = await self.fetcher.fetch(url)
            if fetched:
                content = fetched

        entities = self.extractor.extract(content or res.get("snippet", ""))

        return ResearchResult(
            url              = url,
            title            = res.get("title", ""),
            snippet          = res.get("snippet", ""),
            full_content     = content,
            quality_score    = score,
            extracted_entities=entities,
            source_engine    = res.get("_engine", "unknown"),
        )

    def _dict_to_result(self, score: float, res: dict, query: str) -> ResearchResult:
        return ResearchResult(
            url           = res.get("url", ""),
            title         = res.get("title", ""),
            snippet       = res.get("snippet", ""),
            full_content  = res.get("full_content", ""),
            quality_score = score,
            source_engine = res.get("_engine", "unknown"),
            query         = query,
        )

    def save_results(self, results: list[ResearchResult], path: Path) -> None:
        """Save research results to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in results]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved %d results to %s", len(data), path)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="ARIA-LAB Web Research Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("goal", nargs="?", help="Research goal / query")
    parser.add_argument("--queries",  type=int, default=7,   help="Number of search queries to generate")
    parser.add_argument("--results",  type=int, default=8,   help="Results per query")
    parser.add_argument("--no-fetch", action="store_true",   help="Skip full content fetching")
    parser.add_argument("--out",      type=Path,              help="Save results to JSON file")
    parser.add_argument("--exa-key",  type=str, default="",  help="Exa API key (or set EXA_API_KEY)")
    parser.add_argument("--serp-key", type=str, default="",  help="SerpAPI key (or set SERPAPI_KEY)")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.goal:
        parser.print_help()
        sys.exit(1)

    engine = WebResearchEngine(
        exa_key=args.exa_key,
        serp_key=args.serp_key,
    )

    async def run() -> None:
        results = await engine.research(
            goal=args.goal,
            num_queries=args.queries,
            results_per_query=args.results,
            fetch_content=not args.no_fetch,
        )
        if args.out:
            engine.save_results(results, args.out)
        else:
            print(f"\nTop {min(10, len(results))} results:\n")
            for i, r in enumerate(results[:10], 1):
                print(f"{i:2}. [{r.quality_score:.2f}] {r.title}")
                print(f"    {r.url}")
                print(f"    {r.snippet[:120]}…" if len(r.snippet) > 120 else f"    {r.snippet}")
                print()

    asyncio.run(run())


if __name__ == "__main__":
    _cli()
