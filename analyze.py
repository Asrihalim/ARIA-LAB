"""
tools/analyze.py — LLM-Powered Analysis Engine for ARIA-LAB v2

Provides:
- LLM entity extraction (product, company, price, person, date, statistic)
- Relationship extraction
- Sentiment analysis on reviews/comments
- Trend detection (growing/declining)
- Gap analysis (what's missing from a market/dataset)
- Opportunity identification (market patterns)
- Insight synthesis (conclusions, not just data)
- Confidence intervals on extracted claims
- Source quality weighting
- Cross-reference validation (≥3 sources → higher confidence)

LLM backends (auto-detected in order):
  1. OpenAI (OPENAI_API_KEY)
  2. Anthropic (ANTHROPIC_API_KEY)
  3. Ollama local (no key, http://localhost:11434)
  4. Rule-based fallback (no LLM required)

Usage:
    python tools/analyze.py --text "content..." 
    python tools/analyze.py --file working/research.json --mode entities
    python tools/analyze.py --file working/research.json --mode insights
    python tools/analyze.py --file working/research.json --mode full
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent


# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

@dataclass
class ExtractedEntity:
    text:        str
    entity_type: str    # product | company | price | person | date | statistic | location | technology
    context:     str    = ""
    confidence:  float  = 0.8
    source:      str    = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Relationship:
    subject:    str
    predicate:  str        # sells | competes_with | mentions | has_price | created_by | located_in
    obj:        str
    confidence: float = 0.7
    source:     str   = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SentimentResult:
    text:       str
    sentiment:  str    # positive | negative | neutral | mixed
    score:      float  # -1.0 to 1.0
    confidence: float  = 0.7
    aspects:    list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrendSignal:
    topic:      str
    direction:  str    # growing | declining | stable | volatile
    evidence:   list[str] = field(default_factory=list)
    confidence: float     = 0.6
    timeframe:  str       = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Gap:
    description: str
    category:    str     # underserved_segment | missing_feature | price_gap | format_gap
    opportunity: str     = ""
    confidence:  float   = 0.6
    evidence:    list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Insight:
    title:      str
    body:       str
    evidence:   list[str]    = field(default_factory=list)
    confidence: float        = 0.7
    source_quality: float    = 0.5
    cross_references: int    = 0   # how many sources support this

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalysisResult:
    entities:       list[ExtractedEntity]  = field(default_factory=list)
    relationships:  list[Relationship]     = field(default_factory=list)
    sentiments:     list[SentimentResult]  = field(default_factory=list)
    trends:         list[TrendSignal]      = field(default_factory=list)
    gaps:           list[Gap]              = field(default_factory=list)
    insights:       list[Insight]          = field(default_factory=list)
    executive_summary: str                 = ""
    analysis_time:  float                  = 0.0
    llm_used:       str                    = "fallback"

    def to_dict(self) -> dict:
        return {
            "entities":          [e.to_dict() for e in self.entities],
            "relationships":     [r.to_dict() for r in self.relationships],
            "sentiments":        [s.to_dict() for s in self.sentiments],
            "trends":            [t.to_dict() for t in self.trends],
            "gaps":              [g.to_dict() for g in self.gaps],
            "insights":          [i.to_dict() for i in self.insights],
            "executive_summary": self.executive_summary,
            "analysis_time":     self.analysis_time,
            "llm_used":          self.llm_used,
        }


# ──────────────────────────────────────────────
# LLM Interface
# ──────────────────────────────────────────────

class LLMInterface:
    """Abstract LLM client with auto-detection of available backends."""

    def __init__(self) -> None:
        self.backend, self.model = self._detect_backend()
        logger.info("LLM backend: %s (%s)", self.backend, self.model)

    def _detect_backend(self) -> tuple[str, str]:
        if os.environ.get("OPENAI_API_KEY"):
            return "openai", "gpt-4o-mini"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic", "claude-3-haiku-20240307"
        if self._ollama_available():
            return "ollama", "llama3.2"
        return "fallback", "regex"

    def _ollama_available(self) -> bool:
        try:
            urlopen("http://localhost:11434/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        """Send a completion request to the active backend."""
        if self.backend == "openai":
            return self._openai(system, user, max_tokens)
        elif self.backend == "anthropic":
            return self._anthropic(system, user, max_tokens)
        elif self.backend == "ollama":
            return self._ollama(system, user, max_tokens)
        return ""   # fallback returns empty → rule-based kicks in

    def _openai(self, system: str, user: str, max_tokens: int) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens":  max_tokens,
            "temperature": 0.1,
        }).encode()
        req = Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            },
            method="POST",
        )
        try:
            raw  = urlopen(req, timeout=60).read()
            data = json.loads(raw)
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("OpenAI error: %s", e)
            return ""

    def _anthropic(self, system: str, user: str, max_tokens: int) -> str:
        payload = json.dumps({
            "model":      self.model,
            "max_tokens": max_tokens,
            "system":     system,
            "messages":   [{"role": "user", "content": user}],
        }).encode()
        req = Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            raw  = urlopen(req, timeout=60).read()
            data = json.loads(raw)
            return data["content"][0]["text"]
        except Exception as e:
            logger.error("Anthropic error: %s", e)
            return ""

    def _ollama(self, system: str, user: str, max_tokens: int) -> str:
        payload = json.dumps({
            "model":  self.model,
            "prompt": f"<system>{system}</system>\n\nUser: {user}\nAssistant:",
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.1},
        }).encode()
        req = Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            raw  = urlopen(req, timeout=120).read()
            data = json.loads(raw)
            return data.get("response", "")
        except Exception as e:
            logger.error("Ollama error: %s", e)
            return ""


# ──────────────────────────────────────────────
# Rule-Based Fallback Extractors
# ──────────────────────────────────────────────

class RuleBasedExtractor:
    """Regex + heuristic extraction for when no LLM is available."""

    PRICE_RE     = re.compile(r"\$[\d,]+(?:\.\d{2})?(?:/\w+)?|\b\d+\s*(?:dollars?|USD|EUR|GBP)\b")
    DATE_RE      = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b"
                               r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b20\d{2}\b")
    PERCENT_RE   = re.compile(r"\d+(?:\.\d+)?\s*%")
    NUMBER_STAT  = re.compile(r"\b(\d[\d,]*(?:\.\d+)?[KMB]?)\s*(users?|customers?|sales?|downloads?|"
                               r"revenue|subscribers?|monthly|annual|growth)\b", re.I)
    URL_RE       = re.compile(r"https?://[^\s\"'>]+")

    POSITIVE_WORDS = {"great","amazing","excellent","fantastic","perfect","love","best","outstanding",
                      "brilliant","superb","wonderful","recommended","top","stellar","impressive"}
    NEGATIVE_WORDS = {"terrible","awful","horrible","worst","bad","poor","broken","useless","scam",
                      "disappointed","garbage","hate","avoid","waste","fail","buggy","slow","expensive"}

    TREND_GROWTH = re.compile(r"\b(?:growing|rising|increasing|surging|booming|trending|spike|growth|"
                               r"expanding|adoption|popular|demand)\b", re.I)
    TREND_DECLINE = re.compile(r"\b(?:declining|falling|shrinking|dropping|dying|saturated|overhyped|"
                                r"decreasing|losing|abandoned)\b", re.I)

    GAP_SIGNALS  = re.compile(r"\b(?:no\s+\w+\s+for|missing|lack\s+of|nobody\s+(?:makes?|sells?|offers?)|"
                               r"wish\s+(?:there\s+was|someone)|can't\s+find|underserved|niche)\b", re.I)

    def extract_entities(self, text: str, source: str = "") -> list[ExtractedEntity]:
        entities: list[ExtractedEntity] = []
        for m in self.PRICE_RE.finditer(text):
            entities.append(ExtractedEntity(text=m.group(), entity_type="price",
                                             context=text[max(0,m.start()-30):m.end()+30], source=source))
        for m in self.DATE_RE.finditer(text):
            entities.append(ExtractedEntity(text=m.group(), entity_type="date",
                                             context=text[max(0,m.start()-30):m.end()+30], source=source))
        for m in self.NUMBER_STAT.finditer(text):
            entities.append(ExtractedEntity(text=m.group(), entity_type="statistic",
                                             context=text[max(0,m.start()-40):m.end()+40], source=source,
                                             confidence=0.75))
        return entities[:50]

    def analyze_sentiment(self, text: str) -> SentimentResult:
        words = set(re.findall(r"\b[a-z]+\b", text.lower()))
        pos = len(words & self.POSITIVE_WORDS)
        neg = len(words & self.NEGATIVE_WORDS)
        total = pos + neg or 1
        score = (pos - neg) / total
        if score > 0.2:
            sentiment = "positive"
        elif score < -0.2:
            sentiment = "negative"
        elif abs(score) <= 0.05 and (pos + neg) > 2:
            sentiment = "mixed"
        else:
            sentiment = "neutral"
        return SentimentResult(
            text=text[:200], sentiment=sentiment, score=round(score, 3),
            confidence=min(0.9, 0.5 + 0.1 * (pos + neg))
        )

    def detect_trends(self, texts: list[str]) -> list[TrendSignal]:
        signals: dict[str, list[str]] = {"growing": [], "declining": [], "stable": []}
        topic_words: dict[str, list[str]] = {}
        for text in texts:
            if self.TREND_GROWTH.search(text):
                kw = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", text)
                signals["growing"].extend(kw[:2])
            if self.TREND_DECLINE.search(text):
                kw = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", text)
                signals["declining"].extend(kw[:2])

        trends: list[TrendSignal] = []
        for direction, topics in signals.items():
            if not topics:
                continue
            # Count most common topics
            counts: dict[str, int] = {}
            for t in topics:
                counts[t] = counts.get(t, 0) + 1
            top_topics = sorted(counts, key=counts.get, reverse=True)[:3]  # type: ignore
            for topic in top_topics:
                trends.append(TrendSignal(
                    topic=topic, direction=direction,
                    evidence=[f"Mentioned {counts[topic]} times in {direction} context"],
                    confidence=min(0.9, 0.5 + 0.1 * counts[topic]),
                ))
        return trends[:10]

    def detect_gaps(self, texts: list[str]) -> list[Gap]:
        gaps: list[Gap] = []
        for text in texts:
            for m in self.GAP_SIGNALS.finditer(text):
                context = text[max(0,m.start()-20):m.end()+80]
                gaps.append(Gap(
                    description=context.strip()[:200],
                    category="underserved_segment",
                    evidence=[context],
                    confidence=0.55,
                ))
        return gaps[:10]


# ──────────────────────────────────────────────
# LLM-Powered Extractors
# ──────────────────────────────────────────────

class LLMExtractor:
    """Uses LLM for higher-quality extraction when available."""

    ENTITY_SYSTEM = """You are an expert entity extraction engine. 
Extract named entities from the text as JSON.
Return ONLY valid JSON in this format:
{"entities": [{"text": "...", "type": "product|company|price|person|date|statistic|technology|location", "context": "...", "confidence": 0.0-1.0}]}
Extract up to 20 most important entities. Be precise."""

    RELATIONSHIP_SYSTEM = """You are a relationship extraction engine.
Extract factual relationships from the text as JSON.
Return ONLY valid JSON:
{"relationships": [{"subject": "...", "predicate": "sells|competes_with|has_price|created_by|located_in|mentions|acquired_by", "object": "...", "confidence": 0.0-1.0}]}
Extract up to 10 key relationships."""

    INSIGHT_SYSTEM = """You are a market intelligence analyst.
Analyze the provided text and extract:
1. Key market insights (non-obvious conclusions, not just facts)
2. Market gaps (unmet needs, underserved segments)
3. Opportunity signals (patterns that indicate business opportunities)
4. Trend direction (growing/declining/stable topics)

Return ONLY valid JSON:
{
  "insights": [{"title": "...", "body": "...", "evidence": ["..."], "confidence": 0.0-1.0}],
  "gaps": [{"description": "...", "category": "...", "opportunity": "...", "confidence": 0.0-1.0}],
  "trends": [{"topic": "...", "direction": "growing|declining|stable", "evidence": ["..."], "confidence": 0.0-1.0}],
  "executive_summary": "150-200 word summary of key findings"
}"""

    def __init__(self, llm: LLMInterface) -> None:
        self.llm = llm

    def _call_llm(self, system: str, content: str, max_tokens: int = 2000) -> Optional[dict]:
        """Call LLM and parse JSON response."""
        # Truncate content to avoid token limits
        content = content[:8000]
        response = self.llm.complete(system, content, max_tokens=max_tokens)
        if not response:
            return None
        # Extract JSON block
        json_match = re.search(r"\{.*\}", response, re.S)
        if not json_match:
            return None
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            # Try to repair common issues
            fixed = json_match.group().replace("'", '"').replace("\n", " ")
            try:
                return json.loads(fixed)
            except Exception:
                logger.debug("JSON parse failed for LLM response: %s…", response[:100])
                return None

    def extract_entities(self, text: str, source: str = "") -> list[ExtractedEntity]:
        data = self._call_llm(self.ENTITY_SYSTEM, text)
        if not data:
            return []
        entities = []
        for item in data.get("entities", []):
            entities.append(ExtractedEntity(
                text        = item.get("text", ""),
                entity_type = item.get("type", "unknown"),
                context     = item.get("context", ""),
                confidence  = float(item.get("confidence", 0.7)),
                source      = source,
            ))
        return entities

    def extract_relationships(self, text: str, source: str = "") -> list[Relationship]:
        data = self._call_llm(self.RELATIONSHIP_SYSTEM, text)
        if not data:
            return []
        rels = []
        for item in data.get("relationships", []):
            rels.append(Relationship(
                subject    = item.get("subject", ""),
                predicate  = item.get("predicate", ""),
                obj        = item.get("object", ""),
                confidence = float(item.get("confidence", 0.7)),
                source     = source,
            ))
        return rels

    def extract_insights(self, text: str) -> tuple[list[Insight], list[Gap], list[TrendSignal], str]:
        data = self._call_llm(self.INSIGHT_SYSTEM, text, max_tokens=3000)
        if not data:
            return [], [], [], ""
        insights = [
            Insight(
                title      = item.get("title", ""),
                body       = item.get("body", ""),
                evidence   = item.get("evidence", []),
                confidence = float(item.get("confidence", 0.7)),
            )
            for item in data.get("insights", [])
        ]
        gaps = [
            Gap(
                description = item.get("description", ""),
                category    = item.get("category", "market_gap"),
                opportunity = item.get("opportunity", ""),
                confidence  = float(item.get("confidence", 0.6)),
            )
            for item in data.get("gaps", [])
        ]
        trends = [
            TrendSignal(
                topic      = item.get("topic", ""),
                direction  = item.get("direction", "stable"),
                evidence   = item.get("evidence", []),
                confidence = float(item.get("confidence", 0.6)),
            )
            for item in data.get("trends", [])
        ]
        summary = data.get("executive_summary", "")
        return insights, gaps, trends, summary


# ──────────────────────────────────────────────
# Main Analyzer
# ──────────────────────────────────────────────

class Analyzer:
    """
    Full analysis pipeline. Uses LLM when available, falls back to rules.
    Performs cross-reference validation to boost confidence of repeated claims.
    """

    def __init__(self, source_quality: Optional[dict[str, float]] = None) -> None:
        self.source_quality = source_quality or {}
        self.llm    = LLMInterface()
        self.rules  = RuleBasedExtractor()
        self.use_llm = self.llm.backend != "fallback"
        self.llm_extractor = LLMExtractor(self.llm) if self.use_llm else None

    def analyze(
        self,
        items:  list[dict],    # list of {text, url, title, quality_score} dicts
        mode:   str = "full",  # entities | insights | full
        goal:   str = "",
    ) -> AnalysisResult:
        """Run full analysis pipeline on a list of text items."""
        start = time.time()

        # Combine texts for batch LLM analysis
        combined = self._build_combined_text(items, max_chars=12000)
        texts     = [item.get("text","") or item.get("full_content","") or item.get("snippet","") for item in items]
        texts     = [t for t in texts if t]

        result = AnalysisResult(llm_used=self.llm.backend)

        # Entity extraction
        if mode in ("entities", "full"):
            if self.llm_extractor:
                result.entities     = self.llm_extractor.extract_entities(combined)
                result.relationships = self.llm_extractor.extract_relationships(combined)
            else:
                for item in items[:20]:
                    text = item.get("text","") or item.get("snippet","")
                    result.entities.extend(
                        self.rules.extract_entities(text, source=item.get("url",""))
                    )

        # Sentiment analysis per item
        if mode in ("entities", "full"):
            for item in items[:30]:
                text = item.get("text","") or item.get("snippet","")
                if text:
                    sent = self.rules.analyze_sentiment(text)
                    sent.source = item.get("url","")
                    result.sentiments.append(sent)

        # Insights, gaps, trends
        if mode in ("insights", "full"):
            if self.llm_extractor:
                insights, gaps, trends, summary = self.llm_extractor.extract_insights(combined)
                result.insights          = insights
                result.gaps              = gaps
                result.trends            = trends
                result.executive_summary = summary
            else:
                result.trends = self.rules.detect_trends(texts)
                result.gaps   = self.rules.detect_gaps(texts)
                result.insights = self._synthesize_insights_from_rules(result)
                result.executive_summary = self._generate_summary(result, goal, items)

        # Cross-reference validation
        self._cross_reference_validate(result, items)

        # Source quality weighting
        self._apply_source_quality(result, items)

        result.analysis_time = round(time.time() - start, 2)
        logger.info("Analysis complete in %.1fs (%d entities, %d insights)",
                    result.analysis_time, len(result.entities), len(result.insights))
        return result

    def _build_combined_text(self, items: list[dict], max_chars: int = 12000) -> str:
        parts: list[str] = []
        remaining = max_chars
        for item in items:
            text = item.get("text","") or item.get("full_content","") or item.get("snippet","")
            title = item.get("title","")
            if title:
                entry = f"[{title}] {text}"
            else:
                entry = text
            chunk = entry[:remaining]
            parts.append(chunk)
            remaining -= len(chunk)
            if remaining <= 0:
                break
        return "\n\n---\n\n".join(parts)

    def _synthesize_insights_from_rules(self, result: AnalysisResult) -> list[Insight]:
        insights: list[Insight] = []
        # Insight from trends
        for trend in result.trends[:3]:
            insights.append(Insight(
                title=f"Trend: {trend.topic} is {trend.direction}",
                body=f"The topic '{trend.topic}' appears to be {trend.direction} based on content analysis.",
                evidence=trend.evidence,
                confidence=trend.confidence,
            ))
        # Insight from gaps
        for gap in result.gaps[:3]:
            insights.append(Insight(
                title="Market Gap Detected",
                body=gap.description,
                evidence=gap.evidence,
                confidence=gap.confidence,
            ))
        # Insight from price entities
        prices = [e for e in result.entities if e.entity_type == "price"]
        if prices:
            insights.append(Insight(
                title=f"Pricing Data Found: {len(prices)} price points",
                body=f"Found {len(prices)} price references: {', '.join(p.text for p in prices[:5])}",
                confidence=0.8,
            ))
        return insights

    def _generate_summary(
        self, result: AnalysisResult, goal: str, items: list[dict]
    ) -> str:
        n_sources = len(items)
        n_entities = len(result.entities)
        n_insights = len(result.insights)
        n_trends   = len(result.trends)
        growing    = [t.topic for t in result.trends if t.direction == "growing"]
        declining  = [t.topic for t in result.trends if t.direction == "declining"]

        summary = f"Analysis of {n_sources} sources yielded {n_entities} entities and {n_insights} insights. "
        if goal:
            summary += f"Research goal: {goal}. "
        if growing:
            summary += f"Growing topics: {', '.join(growing[:3])}. "
        if declining:
            summary += f"Declining topics: {', '.join(declining[:3])}. "
        if result.gaps:
            summary += f"Detected {len(result.gaps)} potential market gaps. "
        avg_conf = sum(s.score for s in result.sentiments) / len(result.sentiments) if result.sentiments else 0
        sentiment_label = "positive" if avg_conf > 0.1 else "negative" if avg_conf < -0.1 else "neutral"
        summary += f"Overall content sentiment: {sentiment_label}."
        return summary

    def _cross_reference_validate(self, result: AnalysisResult, items: list[dict]) -> None:
        """Boost confidence of claims that appear in multiple sources."""
        source_count = len(items)
        # Boost entity confidence if text appears in multiple sources
        for entity in result.entities:
            count = sum(
                1 for item in items
                if entity.text.lower() in (item.get("text","") or "").lower()
            )
            if count >= 3:
                entity.confidence = min(1.0, entity.confidence + 0.15)
            elif count >= 2:
                entity.confidence = min(1.0, entity.confidence + 0.07)
        # Boost insight confidence
        for insight in result.insights:
            insight.cross_references = sum(
                1 for item in items
                if any(kw in (item.get("text","") or "").lower()
                       for kw in insight.title.lower().split()[:3])
            )
            if insight.cross_references >= 3:
                insight.confidence = min(1.0, insight.confidence + 0.15)

    def _apply_source_quality(self, result: AnalysisResult, items: list[dict]) -> None:
        """Weight confidence by source quality scores."""
        if not self.source_quality:
            return
        for entity in result.entities:
            quality = self.source_quality.get(entity.source, 0.5)
            entity.confidence = round(entity.confidence * (0.7 + 0.6 * quality), 3)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="ARIA-LAB Analysis Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file",    type=Path, help="JSON file with text items to analyze")
    parser.add_argument("--text",    type=str,  help="Inline text to analyze")
    parser.add_argument("--mode",    choices=["entities", "insights", "full"], default="full")
    parser.add_argument("--goal",    type=str,  default="", help="Research goal for context")
    parser.add_argument("--out",     type=Path, help="Save analysis to JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Build items list
    if args.file:
        raw = json.loads(args.file.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            items = [raw]
        else:
            items = []
    elif args.text:
        items = [{"text": args.text, "title": "Input text", "url": ""}]
    else:
        parser.print_help()
        sys.exit(1)

    analyzer = Analyzer()
    result   = analyzer.analyze(items, mode=args.mode, goal=args.goal)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Analysis saved to {args.out}")
    else:
        print(f"\n{'='*60}")
        print(f"Analysis Results (backend: {result.llm_used})")
        print(f"Time: {result.analysis_time:.1f}s")
        print(f"{'='*60}")
        if result.executive_summary:
            print(f"\nExecutive Summary:\n{result.executive_summary}")
        print(f"\nEntities     : {len(result.entities)}")
        print(f"Relationships: {len(result.relationships)}")
        print(f"Insights     : {len(result.insights)}")
        print(f"Trends       : {len(result.trends)}")
        print(f"Gaps         : {len(result.gaps)}")
        if result.insights:
            print("\nKey Insights:")
            for ins in result.insights[:5]:
                print(f"  • [{ins.confidence:.0%}] {ins.title}")
                print(f"    {ins.body[:120]}…" if len(ins.body) > 120 else f"    {ins.body}")


if __name__ == "__main__":
    _cli()
