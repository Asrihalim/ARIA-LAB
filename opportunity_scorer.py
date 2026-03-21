"""
tools/opportunity_scorer.py — Decision Engine for ARIA-LAB v2

Multi-criteria scoring with configurable weights, threshold filtering,
score explanations, comparison matrix, sensitivity analysis, and full ranking.

Default criteria: demand(0.3), competition(0.25), ease_of_creation(0.2), revenue_potential(0.25)
Override weights in task.md: "weight revenue_potential at 0.5"

Usage:
    python tools/opportunity_scorer.py --input findings.json
    python tools/opportunity_scorer.py --input findings.json --weights '{"demand":0.5,"revenue_potential":0.5}'
    python tools/opportunity_scorer.py --input findings.json --top 10 --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent


# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

@dataclass
class CriterionScore:
    """Score for one criterion with explanation."""
    name:        str
    raw_score:   float   # 0–10
    weight:      float
    weighted:    float   # raw_score * weight
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoredOpportunity:
    """A ranked and scored opportunity."""
    name:            str
    total_score:     float          # 0–10
    rank:            int            = 0
    criteria:        list[CriterionScore] = field(default_factory=list)
    meets_threshold: bool           = True
    source:          str            = ""
    url:             str            = ""
    tags:            list[str]      = field(default_factory=list)
    raw_data:        dict           = field(default_factory=dict)
    sensitivity_note: str          = ""

    def to_dict(self) -> dict:
        return {
            "rank":            self.rank,
            "name":            self.name,
            "total_score":     round(self.total_score, 2),
            "meets_threshold": self.meets_threshold,
            "source":          self.source,
            "url":             self.url,
            "tags":            self.tags,
            "criteria":        [c.to_dict() for c in self.criteria],
            "sensitivity_note": self.sensitivity_note,
        }


# ──────────────────────────────────────────────
# Scoring Heuristics
# ──────────────────────────────────────────────

class ScoringHeuristics:
    """
    Heuristic scoring functions for each criterion.
    Each function returns (score: float 0–10, explanation: str).
    """

    @staticmethod
    def demand(record: dict) -> tuple[float, str]:
        """Score demand based on sales estimates, reviews, upvotes, search volume."""
        reasons = []
        score   = 5.0  # neutral start

        # Sales estimates
        sales = record.get("sales_estimate") or record.get("sales") or 0
        if isinstance(sales, (int, float)) and sales > 0:
            if sales >= 5000:
                score = min(10, score + 4)
                reasons.append(f"{sales:,} sales ≥ 5,000")
            elif sales >= 1000:
                score = min(10, score + 2.5)
                reasons.append(f"{sales:,} sales ≥ 1,000")
            elif sales >= 100:
                score = min(10, score + 1)
                reasons.append(f"{sales:,} sales")
            else:
                score = max(0, score - 1)
                reasons.append(f"Only {sales} sales")

        # Reviews/comments as proxy for demand
        reviews = record.get("reviews_count") or record.get("rating_count") or 0
        if isinstance(reviews, (int, float)) and reviews > 0:
            if reviews >= 1000:
                score = min(10, score + 1.5)
                reasons.append(f"{reviews:,} reviews")
            elif reviews >= 100:
                score = min(10, score + 0.7)
                reasons.append(f"{reviews} reviews")

        # Upvotes/votes (Product Hunt, Reddit)
        upvotes = record.get("rating") if record.get("platform") in ("producthunt","reddit") else 0
        if isinstance(upvotes, (int, float)) and upvotes > 0:
            if upvotes >= 500:
                score = min(10, score + 1.5)
                reasons.append(f"{upvotes:,} upvotes")
            elif upvotes >= 100:
                score = min(10, score + 0.5)

        # Price as proxy for willingness to pay
        price = record.get("price") or 0
        if isinstance(price, (int, float)) and price > 0:
            if price >= 20:
                score = min(10, score + 0.5)
                reasons.append(f"${price} price point (premium)")

        score = round(min(10, max(0, score)), 2)
        explanation = "; ".join(reasons) if reasons else f"Demand score: {score}/10 (estimated)"
        return score, explanation

    @staticmethod
    def competition(record: dict) -> tuple[float, str]:
        """
        Score competition INVERSELY — lower competition = higher score.
        High score (8-10) = LOW competition = good for entering.
        """
        reasons  = []
        score    = 6.0  # assume moderate competition

        platform = (record.get("platform") or "").lower()

        # Niche platforms → lower competition
        if platform in ("payhip", "lemonsqueezy", "gumroad"):
            score = min(10, score + 1.5)
            reasons.append(f"Niche platform ({platform}) → lower competition")

        # High sales with few reviews → less-crowded market
        sales   = record.get("sales_estimate", 0) or 0
        reviews = record.get("reviews_count", 0) or 0
        if sales > 500 and reviews < 50:
            score = min(10, score + 2)
            reasons.append("High sales, few reviews → early market")

        # Very high sales/reviews → crowded market
        if reviews and isinstance(reviews, (int, float)) and reviews > 10000:
            score = max(0, score - 3)
            reasons.append(f"{reviews:,} reviews → saturated market")
        elif reviews and isinstance(reviews, (int, float)) and reviews > 1000:
            score = max(0, score - 1.5)
            reasons.append(f"{reviews:,} reviews → competitive market")

        # Tags suggesting generic/oversaturated niches
        tags = record.get("tags", [])
        if isinstance(tags, list):
            over_saturated = {"resume", "cv", "cover letter", "recipe", "generic"}
            if any(t.lower() in over_saturated for t in tags):
                score = max(0, score - 1)
                reasons.append("Generic/saturated niche keywords in tags")

        score = round(min(10, max(0, score)), 2)
        explanation = "; ".join(reasons) if reasons else f"Competition score: {score}/10 (estimated)"
        return score, explanation

    @staticmethod
    def ease_of_creation(record: dict) -> tuple[float, str]:
        """Score ease based on product type, price, and category."""
        reasons = []
        score   = 5.0

        category = str(record.get("category", "")).lower()
        name     = str(record.get("name", "")).lower()
        combined = category + " " + name

        # Easy digital products
        easy_keywords = {
            "template","spreadsheet","checklist","worksheet","planner",
            "guide","ebook","cheatsheet","swipe file","pdf","notion",
            "canva","menu","calendar","tracker","script",
        }
        hard_keywords = {
            "software","saas","app","plugin","code","api","integration",
            "database","platform","custom","automated","ai","ml",
        }

        easy_matches = [k for k in easy_keywords if k in combined]
        hard_matches = [k for k in hard_keywords if k in combined]

        if easy_matches:
            score = min(10, score + len(easy_matches) * 0.8)
            reasons.append(f"Easy product type: {', '.join(easy_matches[:3])}")
        if hard_matches:
            score = max(0, score - len(hard_matches) * 1.2)
            reasons.append(f"Complex product type: {', '.join(hard_matches[:3])}")

        # Price proxy for complexity
        price = record.get("price") or 0
        if isinstance(price, (int, float)):
            if price < 10:
                score = min(10, score + 0.5)
                reasons.append(f"Low price (${price}) → likely simple product")
            elif price > 200:
                score = max(0, score - 1)
                reasons.append(f"High price (${price}) → likely complex product")

        score = round(min(10, max(0, score)), 2)
        explanation = "; ".join(reasons) if reasons else f"Ease score: {score}/10 (estimated)"
        return score, explanation

    @staticmethod
    def revenue_potential(record: dict) -> tuple[float, str]:
        """Score revenue potential from price × sales estimates."""
        reasons = []
        score   = 5.0

        price  = record.get("price") or 0
        sales  = record.get("sales_estimate") or 0

        if isinstance(price, (int, float)) and isinstance(sales, (int, float)):
            if price > 0 and sales > 0:
                mrr = price * sales  # not monthly but total potential
                if mrr >= 100_000:
                    score = min(10, 9.0)
                    reasons.append(f"${mrr:,.0f} total revenue potential (${price} × {sales:,} sales)")
                elif mrr >= 10_000:
                    score = min(10, 7.0)
                    reasons.append(f"${mrr:,.0f} total revenue potential")
                elif mrr >= 1_000:
                    score = 5.5
                    reasons.append(f"${mrr:,.0f} total revenue potential")
                else:
                    score = 3.0
                    reasons.append(f"${mrr:,.0f} total revenue potential (low)")
            elif price > 50:
                score = min(10, score + 1.5)
                reasons.append(f"Premium price (${price})")
            elif price > 0:
                score = min(10, score + 0.5)
                reasons.append(f"Price: ${price}")

        # Platform commission consideration
        platform = (record.get("platform") or "").lower()
        if platform == "gumroad":
            reasons.append("Gumroad: ~9% fees")
        elif platform == "etsy":
            reasons.append("Etsy: ~6.5% fees + listing fees")
            score = max(0, score - 0.3)
        elif platform in ("payhip", "lemonsqueezy"):
            score = min(10, score + 0.2)
            reasons.append(f"{platform}: competitive fees")

        score = round(min(10, max(0, score)), 2)
        explanation = "; ".join(reasons) if reasons else f"Revenue potential: {score}/10 (estimated)"
        return score, explanation


# ──────────────────────────────────────────────
# OpportunityScorer
# ──────────────────────────────────────────────

class OpportunityScorer:
    """
    Scores, ranks, and explains opportunities using multi-criteria analysis.
    """

    DEFAULT_WEIGHTS = {
        "demand":            0.30,
        "competition":       0.25,
        "ease_of_creation":  0.20,
        "revenue_potential": 0.25,
    }

    DEFAULT_THRESHOLDS = {
        "demand":            6.0,
        "competition":       3.0,   # Remember: competition score is inverted!
        "ease_of_creation":  4.0,
    }

    CRITERIA_FN = {
        "demand":            ScoringHeuristics.demand,
        "competition":       ScoringHeuristics.competition,
        "ease_of_creation":  ScoringHeuristics.ease_of_creation,
        "revenue_potential": ScoringHeuristics.revenue_potential,
    }

    def __init__(
        self,
        weights:    Optional[dict[str, float]] = None,
        thresholds: Optional[dict[str, float]] = None,
    ) -> None:
        self.weights    = self._normalize_weights(weights or self.DEFAULT_WEIGHTS.copy())
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()

    def _normalize_weights(self, w: dict[str, float]) -> dict[str, float]:
        total = sum(w.values())
        if total == 0:
            return self.DEFAULT_WEIGHTS.copy()
        return {k: v / total for k, v in w.items()}

    def score_all(
        self,
        records: list[dict],
        top_n:   Optional[int] = None,
    ) -> list[ScoredOpportunity]:
        """Score and rank all records."""
        opportunities: list[ScoredOpportunity] = []
        for record in records:
            opp = self._score_record(record)
            opportunities.append(opp)

        # Sort by total score descending, then by demand as tiebreaker
        opportunities.sort(
            key=lambda o: (o.total_score, next((c.raw_score for c in o.criteria if c.name=="demand"), 0)),
            reverse=True,
        )

        # Assign ranks (only among those meeting thresholds first)
        rank = 1
        for opp in opportunities:
            if opp.meets_threshold:
                opp.rank = rank
                rank += 1
        for opp in opportunities:
            if not opp.meets_threshold:
                opp.rank = rank
                rank += 1

        if top_n:
            opportunities = opportunities[:top_n]

        logger.info("Scored %d opportunities, top score: %.2f",
                    len(opportunities), opportunities[0].total_score if opportunities else 0)
        return opportunities

    def _score_record(self, record: dict) -> ScoredOpportunity:
        """Score a single record across all criteria."""
        criteria_scores: list[CriterionScore] = []
        total = 0.0
        meets_all = True

        for criterion, fn in self.CRITERIA_FN.items():
            weight = self.weights.get(criterion, 0.0)
            if weight == 0:
                continue
            raw, explanation = fn(record)
            weighted = raw * weight
            total   += weighted

            # Check threshold
            threshold = self.thresholds.get(criterion)
            if threshold is not None:
                # competition threshold works differently (score = inverse, so lower raw_score = bad)
                if criterion == "competition" and raw < (10 - threshold):
                    meets_all = False
                elif criterion != "competition" and raw < threshold:
                    meets_all = False

            criteria_scores.append(CriterionScore(
                name=criterion,
                raw_score=raw,
                weight=weight,
                weighted=round(weighted, 3),
                explanation=explanation,
            ))

        # Scale total to 0–10
        max_possible = sum(self.weights.get(c, 0) for c in self.CRITERIA_FN) * 10
        normalized   = (total / max_possible * 10) if max_possible > 0 else total

        name = str(record.get("name") or record.get("title") or "Unknown")
        return ScoredOpportunity(
            name            = name,
            total_score     = round(normalized, 2),
            criteria        = criteria_scores,
            meets_threshold = meets_all,
            source          = str(record.get("platform") or record.get("source") or ""),
            url             = str(record.get("url") or ""),
            tags            = record.get("tags", []) or [],
            raw_data        = {k: v for k, v in record.items() if not k.startswith("_")},
        )

    def compare(self, opportunities: list[ScoredOpportunity], top_n: int = 10) -> str:
        """Generate a Markdown side-by-side comparison matrix of top N opportunities."""
        candidates = [o for o in opportunities if o.meets_threshold][:top_n]
        if not candidates:
            candidates = opportunities[:top_n]
        if not candidates:
            return "No opportunities to compare."

        criteria_names = [c.name for c in candidates[0].criteria]

        # Header
        lines = [f"## Opportunity Comparison Matrix (Top {len(candidates)})\n"]
        col_w = 20
        header = f"{'Opportunity':<{col_w}}" + "".join(f"{c:<12}" for c in ["Score", *criteria_names])
        lines.append(header)
        lines.append("-" * len(header))

        for opp in candidates:
            scores = {c.name: c.raw_score for c in opp.criteria}
            row = f"{opp.name[:col_w-1]:<{col_w}}"
            row += f"{opp.total_score:<12.2f}"
            for c in criteria_names:
                row += f"{scores.get(c, 0):<12.1f}"
            lines.append(row)

        return "\n".join(lines)

    def sensitivity_analysis(
        self,
        opportunities: list[ScoredOpportunity],
        vary_criterion: str = "ease_of_creation",
    ) -> str:
        """
        Show how rankings change if the given criterion weight is doubled.
        """
        # Build alternate weights
        alt_weights = dict(self.weights)
        alt_weights[vary_criterion] = min(1.0, alt_weights.get(vary_criterion, 0) * 2)
        alt_scorer = OpportunityScorer(weights=alt_weights, thresholds=self.thresholds)

        alt_opp = alt_scorer.score_all([o.raw_data for o in opportunities])

        original_order = {o.name: i+1 for i, o in enumerate(opportunities)}
        lines = [f"## Sensitivity Analysis: if weight({vary_criterion}) × 2\n"]
        lines.append(f"{'Name':<30} {'Current Rank':<15} {'Alt Rank':<12} {'Δ Rank'}")
        lines.append("-" * 65)
        for i, o in enumerate(alt_opp[:10]):
            orig = original_order.get(o.name, 99)
            alt  = i + 1
            delta = orig - alt
            sign  = "+" if delta > 0 else ""
            lines.append(f"{o.name[:29]:<30} {orig:<15} {alt:<12} {sign}{delta}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="ARIA-LAB Opportunity Scorer — Decision Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python tools/opportunity_scorer.py --input findings.json
  python tools/opportunity_scorer.py --input findings.json --top 10 --compare
  python tools/opportunity_scorer.py --input findings.json --weights '{"demand":0.5,"revenue_potential":0.5}'
  python tools/opportunity_scorer.py --input findings.json --min-demand 7 --min-ease 6
""",
    )
    parser.add_argument("--input",       type=Path, required=True,     help="JSON file with records")
    parser.add_argument("--out",         type=Path,                    help="Save scores to JSON")
    parser.add_argument("--top",         type=int,  default=20,        help="Return top N results")
    parser.add_argument("--compare",     action="store_true",          help="Generate comparison matrix")
    parser.add_argument("--sensitivity", type=str,                     help="Run sensitivity on this criterion")
    parser.add_argument("--weights",     type=json.loads,              help="JSON weight overrides")
    parser.add_argument("--min-demand",  type=float, default=None,     help="Min demand threshold (0-10)")
    parser.add_argument("--min-ease",    type=float, default=None,     help="Min ease threshold (0-10)")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = [records]

    # Build thresholds
    thresholds: dict[str, float] = {}
    if args.min_demand is not None:
        thresholds["demand"] = args.min_demand
    if args.min_ease is not None:
        thresholds["ease_of_creation"] = args.min_ease

    scorer = OpportunityScorer(
        weights=args.weights,
        thresholds=thresholds if thresholds else None,
    )
    opps = scorer.score_all(records, top_n=args.top)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([o.to_dict() for o in opps], indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"Saved {len(opps)} scored opportunities to {args.out}")
    else:
        qualifies = sum(1 for o in opps if o.meets_threshold)
        print(f"\nTop {len(opps)} Opportunities ({qualifies} meet all thresholds):\n")
        for opp in opps[:args.top]:
            status = "✓" if opp.meets_threshold else "✗"
            print(f"  {status} #{opp.rank:>2}  [{opp.total_score:>5.2f}/10]  {opp.name}")
            for c in opp.criteria:
                print(f"         {c.name:<22}: {c.raw_score:>4.1f}/10  — {c.explanation[:60]}")
            print()

    if args.compare:
        print(scorer.compare(opps, top_n=args.top))

    if args.sensitivity:
        print(scorer.sensitivity_analysis(opps, vary_criterion=args.sensitivity))


if __name__ == "__main__":
    _cli()
