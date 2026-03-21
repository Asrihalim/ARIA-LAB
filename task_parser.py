"""
tools/task_parser.py — Advanced Task Specification Parser for ARIA-LAB v2

Parses inputs/task.md (rigid or free-form) into a fully structured TaskSpec object.
Detects task type, required tools, deliverables with schemas, constraints, and weights.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Enums & Data Classes
# ──────────────────────────────────────────────

class TaskType(str, Enum):
    RESEARCH  = "research"
    ANALYZE   = "analyze"
    BUILD     = "build"
    COMPARE   = "compare"
    SCORE     = "score"
    MONITOR   = "monitor"
    UNKNOWN   = "unknown"


@dataclass
class DeliverableSpec:
    """Parsed output file requirement with optional JSON schema."""
    filename: str
    format: str                        # md | json | csv | html | txt
    schema: dict[str, Any]            = field(default_factory=dict)
    description: str                   = ""
    required: bool                     = True


@dataclass
class ScoringWeights:
    """Configurable scoring weights for opportunity_scorer."""
    demand:             float = 0.30
    competition:        float = 0.25
    ease_of_creation:   float = 0.20
    revenue_potential:  float = 0.25

    def validate(self) -> None:
        total = sum([self.demand, self.competition, self.ease_of_creation, self.revenue_potential])
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total:.3f}")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ConditionalRule:
    """Parsed conditional logic from task.md."""
    condition: str
    action: str
    raw: str


@dataclass
class TaskSpec:
    """Fully structured task specification produced by the parser."""
    # Core fields
    target:       str                       = ""
    goal:         str                       = ""
    context:      str                       = ""
    raw_markdown: str                       = ""

    # Derived fields
    task_type:    TaskType                  = TaskType.UNKNOWN
    deliverables: list[DeliverableSpec]    = field(default_factory=list)
    constraints:  list[str]               = field(default_factory=list)
    scoring_weights: ScoringWeights        = field(default_factory=ScoringWeights)
    scoring_thresholds: dict[str, float]   = field(default_factory=dict)
    conditions:   list[ConditionalRule]    = field(default_factory=list)
    required_tools: list[str]             = field(default_factory=list)
    keywords:     list[str]               = field(default_factory=list)
    ambiguities:  list[str]               = field(default_factory=list)
    is_complete:  bool                     = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task_type"] = self.task_type.value
        d["scoring_weights"] = self.scoring_weights.as_dict()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ──────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────

class TaskParser:
    """
    Parses task.md into a TaskSpec.

    Supports both the canonical heading format (# Target / # Goal / # Context)
    and free-form prose where fields are inferred heuristically.
    """

    # Keyword → tool mapping
    TOOL_KEYWORDS: dict[str, list[str]] = {
        "web_research":        ["research", "search", "find", "discover", "investigate", "explore", "look up"],
        "browser_scraper":     ["scrape", "crawl", "extract", "webpage", "website", "page", "html", "javascript", "spa"],
        "marketplace_scraper": ["gumroad", "etsy", "appsumo", "product hunt", "payhip", "lemon squeezy",
                                 "marketplace", "listing", "product", "seller", "bestseller"],
        "knowledge_base":      ["remember", "recall", "knowledge", "memory", "store", "history"],
        "data_synthesizer":    ["merge", "combine", "fuse", "deduplicate", "enrich", "unify", "aggregate"],
        "analyze":             ["analyze", "analysis", "insight", "trend", "sentiment", "entity", "extract"],
        "opportunity_scorer":  ["score", "rank", "opportunity", "evaluate", "rate", "compare", "best"],
        "report":              ["report", "output", "deliverable", "summary", "document"],
    }

    # Type keywords → TaskType
    TYPE_KEYWORDS: dict[TaskType, list[str]] = {
        TaskType.RESEARCH: ["research", "find", "discover", "investigate", "search", "learn about"],
        TaskType.ANALYZE:  ["analyze", "analysis", "breakdown", "breakdown", "examine", "evaluate data"],
        TaskType.BUILD:    ["build", "create", "generate", "produce", "make", "develop"],
        TaskType.COMPARE:  ["compare", "comparison", "vs", "versus", "contrast", "side-by-side", "matrix"],
        TaskType.SCORE:    ["score", "rank", "rating", "evaluate", "rate", "top n", "best"],
        TaskType.MONITOR:  ["monitor", "track", "watch", "alert", "trend over time", "recurring"],
    }

    # Weight override patterns
    WEIGHT_PATTERN = re.compile(
        r"weight\s+(demand|competition|ease_of_creation|revenue_potential|ease)\s+(?:at\s+)?([0-9.]+)",
        re.IGNORECASE,
    )

    # Deliverable extraction
    DELIVERABLE_PATTERN = re.compile(
        r"(?:produce|output|generate|create|write|save)\s+(outputs/[\w./\-]+\.\w+)"
        r"(?:\s+with\s+schema\s*:\s*(\{[^}]+\}))?",
        re.IGNORECASE,
    )

    # Conditional rule patterns
    CONDITION_PATTERN = re.compile(
        r"if\s+(.+?),\s+(?:then\s+)?(.+?)(?:\.|$)",
        re.IGNORECASE,
    )

    # Threshold patterns
    THRESHOLD_PATTERN = re.compile(
        r"(demand|competition|ease_of_creation|revenue_potential|ease)\s*[≥>=]+\s*([0-9.]+)",
        re.IGNORECASE,
    )

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or Path(__file__).parent.parent

    # ── Public API ─────────────────────────────

    def parse_file(self, path: Optional[Path] = None) -> TaskSpec:
        """Parse inputs/task.md (or given path) into a TaskSpec."""
        if path is None:
            path = self.base_dir / "inputs" / "task.md"
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {path}")

        raw = path.read_text(encoding="utf-8")
        logger.debug("Read task file: %s (%d chars)", path, len(raw))
        return self.parse_text(raw)

    def parse_text(self, markdown: str) -> TaskSpec:
        """Parse raw markdown string into a TaskSpec."""
        spec = TaskSpec(raw_markdown=markdown)

        sections = self._extract_sections(markdown)

        spec.target  = self._clean(sections.get("target",  ""))
        spec.goal    = self._clean(sections.get("goal",    ""))
        spec.context = self._clean(sections.get("context", ""))

        # Fallback: infer from free-form prose if sections are empty
        if not spec.target and not spec.goal:
            spec = self._infer_from_prose(spec, markdown)

        full_text = " ".join([spec.target, spec.goal, spec.context, markdown]).lower()

        spec.task_type       = self._detect_task_type(full_text)
        spec.keywords        = self._extract_keywords(full_text)
        spec.required_tools  = self._detect_required_tools(full_text)
        spec.deliverables    = self._extract_deliverables(markdown)
        spec.constraints     = self._extract_constraints(sections.get("constraints", ""), markdown)
        spec.scoring_weights = self._extract_weights(markdown)
        spec.scoring_thresholds = self._extract_thresholds(markdown)
        spec.conditions      = self._extract_conditions(markdown)
        spec.ambiguities     = self._detect_ambiguities(spec)
        spec.is_complete     = len(spec.ambiguities) == 0

        if not spec.deliverables:
            spec.deliverables = self._default_deliverables(spec)

        logger.info(
            "Parsed task: type=%s, tools=%s, deliverables=%d",
            spec.task_type.value,
            spec.required_tools,
            len(spec.deliverables),
        )
        return spec

    # ── Section Extraction ─────────────────────

    def _extract_sections(self, markdown: str) -> dict[str, str]:
        """Extract canonical heading sections from markdown."""
        sections: dict[str, str] = {}
        pattern = re.compile(r"^#+\s+(.+)$", re.MULTILINE)
        matches = list(pattern.finditer(markdown))

        for i, match in enumerate(matches):
            heading = match.group(1).strip().lower()
            start   = match.end()
            end     = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
            body    = markdown[start:end].strip()
            sections[heading] = body

        return sections

    def _infer_from_prose(self, spec: TaskSpec, text: str) -> TaskSpec:
        """Heuristic extractor for free-form tasks."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # First non-empty line → target
        if lines:
            spec.target = lines[0]

        # Look for goal-like sentence
        goal_patterns = [
            re.compile(r"(?:goal|objective|task|want|need).*?:\s*(.+)", re.IGNORECASE),
            re.compile(r"(?:please|i want|find|research|analyze)\s+(.+)", re.IGNORECASE),
        ]
        for line in lines[1:]:
            for pat in goal_patterns:
                m = pat.search(line)
                if m:
                    spec.goal = m.group(1).strip()
                    break
            if spec.goal:
                break

        if not spec.goal and len(lines) > 1:
            spec.goal = lines[1]

        # Remaining lines → context
        context_lines = lines[2:]
        spec.context = " ".join(context_lines)

        spec.ambiguities.append("Task format is non-standard; fields were inferred heuristically")
        return spec

    # ── Type Detection ─────────────────────────

    def _detect_task_type(self, text: str) -> TaskType:
        scores: dict[TaskType, int] = {t: 0 for t in TaskType}
        for task_type, keywords in self.TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    scores[task_type] += 1

        best = max(scores, key=lambda t: scores[t])
        if scores[best] == 0:
            return TaskType.RESEARCH  # Default to research
        return best

    # ── Keyword Extraction ─────────────────────

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from combined text."""
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "be",
            "this", "that", "it", "i", "you", "we", "they", "have", "has",
        }
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        seen: set[str] = set()
        keywords = []
        for w in words:
            if w not in stopwords and w not in seen:
                seen.add(w)
                keywords.append(w)
        return keywords[:50]  # cap at 50

    # ── Tool Detection ─────────────────────────

    def _detect_required_tools(self, text: str) -> list[str]:
        tools: list[str] = []
        for tool, keywords in self.TOOL_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                if tool not in tools:
                    tools.append(tool)
        # research always requires web_research
        if "web_research" not in tools:
            tools.insert(0, "web_research")
        return tools

    # ── Deliverable Extraction ─────────────────

    def _extract_deliverables(self, markdown: str) -> list[DeliverableSpec]:
        deliverables: list[DeliverableSpec] = []
        for match in self.DELIVERABLE_PATTERN.finditer(markdown):
            filename = match.group(1).strip()
            schema_raw = match.group(2)
            schema: dict[str, Any] = {}
            if schema_raw:
                try:
                    # Allow bare JSON5-ish: {name, price, score}
                    normalised = re.sub(r"(\w+)", r'"\1"', schema_raw)
                    normalised = re.sub(r'"([^"]+)"(\s*[:,])', lambda m:
                        f'"{m.group(1)}"{m.group(2)}', normalised)
                    schema = json.loads(normalised)
                except (json.JSONDecodeError, TypeError):
                    schema = {"_raw": schema_raw}

            ext = Path(filename).suffix.lstrip(".")
            deliverables.append(DeliverableSpec(
                filename=filename,
                format=ext or "txt",
                schema=schema,
                required=True,
            ))
        return deliverables

    def _default_deliverables(self, spec: TaskSpec) -> list[DeliverableSpec]:
        """Generate sensible default deliverables based on task type."""
        defaults: list[DeliverableSpec] = []
        base = "outputs/"
        if spec.task_type in (TaskType.RESEARCH, TaskType.ANALYZE):
            defaults.append(DeliverableSpec(filename=f"{base}report.md",   format="md"))
            defaults.append(DeliverableSpec(filename=f"{base}findings.json", format="json"))
        elif spec.task_type == TaskType.COMPARE:
            defaults.append(DeliverableSpec(filename=f"{base}comparison.md",  format="md"))
            defaults.append(DeliverableSpec(filename=f"{base}comparison.json", format="json"))
            defaults.append(DeliverableSpec(filename=f"{base}comparison.csv",  format="csv"))
        elif spec.task_type == TaskType.SCORE:
            defaults.append(DeliverableSpec(filename=f"{base}scored_opportunities.json", format="json"))
            defaults.append(DeliverableSpec(filename=f"{base}opportunities_report.md",   format="md"))
        elif spec.task_type == TaskType.BUILD:
            defaults.append(DeliverableSpec(filename=f"{base}build_report.md", format="md"))
        elif spec.task_type == TaskType.MONITOR:
            defaults.append(DeliverableSpec(filename=f"{base}monitor_report.md", format="md"))
            defaults.append(DeliverableSpec(filename=f"{base}monitor_data.json", format="json"))
        else:
            defaults.append(DeliverableSpec(filename=f"{base}report.md", format="md"))
        return defaults

    # ── Constraint Extraction ──────────────────

    def _extract_constraints(self, constraint_section: str, markdown: str) -> list[str]:
        constraints: list[str] = []
        # From explicit Constraints section
        if constraint_section:
            for line in constraint_section.splitlines():
                line = line.lstrip("- •*").strip()
                if line:
                    constraints.append(line)
        # Inline constraint patterns
        inline_patterns = [
            re.compile(r"(?:must|should|only|exclude|limit|maximum|minimum|at most|at least)\s+.+", re.IGNORECASE),
            re.compile(r"(?:don't|do not|avoid|never)\s+.+", re.IGNORECASE),
        ]
        for pat in inline_patterns:
            for m in pat.finditer(markdown):
                c = m.group(0).strip().rstrip(".,;")
                if c not in constraints:
                    constraints.append(c)

        return constraints[:20]  # cap

    # ── Weight Extraction ──────────────────────

    def _extract_weights(self, markdown: str) -> ScoringWeights:
        weights = ScoringWeights()
        for match in self.WEIGHT_PATTERN.finditer(markdown):
            key   = match.group(1).lower().replace("ease", "ease_of_creation")
            value = float(match.group(2))
            if hasattr(weights, key):
                setattr(weights, key, value)
                logger.debug("Override weight: %s = %.2f", key, value)
        try:
            weights.validate()
        except ValueError as e:
            logger.warning("Weight override invalid (%s). Using defaults.", e)
            weights = ScoringWeights()
        return weights

    # ── Threshold Extraction ───────────────────

    def _extract_thresholds(self, markdown: str) -> dict[str, float]:
        thresholds: dict[str, float] = {}
        for match in self.THRESHOLD_PATTERN.finditer(markdown):
            key   = match.group(1).lower()
            value = float(match.group(2))
            thresholds[key] = value
        return thresholds

    # ── Condition Extraction ───────────────────

    def _extract_conditions(self, markdown: str) -> list[ConditionalRule]:
        rules: list[ConditionalRule] = []
        for match in self.CONDITION_PATTERN.finditer(markdown):
            rules.append(ConditionalRule(
                condition=match.group(1).strip(),
                action=match.group(2).strip(),
                raw=match.group(0).strip(),
            ))
        return rules

    # ── Ambiguity Detection ────────────────────

    def _detect_ambiguities(self, spec: TaskSpec) -> list[str]:
        ambiguities: list[str] = list(spec.ambiguities)  # carry over inference warnings
        if not spec.target:
            ambiguities.append("No target specified (URL, company, topic).")
        if not spec.goal:
            ambiguities.append("No goal specified — what should be accomplished?")
        if spec.task_type == TaskType.UNKNOWN:
            ambiguities.append("Could not determine task type (research/analyze/build/compare/score).")
        if not spec.deliverables:
            ambiguities.append("No deliverables specified; defaults will be used.")
        return ambiguities

    # ── Helpers ────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """Strip markdown list markers and extra whitespace."""
        text = re.sub(r"^[-*•]\s+", "", text, flags=re.MULTILINE)
        return text.strip()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Parse inputs/task.md into a structured TaskSpec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python tools/task_parser.py
  python tools/task_parser.py --file inputs/task.md
  python tools/task_parser.py --file inputs/task.md --json
  python tools/task_parser.py --text "Research top AI tools on Product Hunt"
""",
    )
    parser.add_argument("--file", type=Path, help="Path to task.md (default: inputs/task.md)")
    parser.add_argument("--text", type=str,  help="Parse inline text instead of a file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    base_dir = Path(__file__).parent.parent
    tp = TaskParser(base_dir=base_dir)

    try:
        if args.text:
            spec = tp.parse_text(args.text)
        else:
            spec = tp.parse_file(args.file)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during parsing")
        sys.exit(1)

    if args.json:
        print(spec.to_json())
    else:
        print(f"{'═' * 60}")
        print(f"  TASK SPEC")
        print(f"{'═' * 60}")
        print(f"  Target       : {spec.target or '(none)'}")
        print(f"  Goal         : {spec.goal or '(none)'}")
        print(f"  Task Type    : {spec.task_type.value}")
        print(f"  Required Tools: {', '.join(spec.required_tools) or '(none)'}")
        print(f"  Deliverables : {len(spec.deliverables)}")
        for d in spec.deliverables:
            schema_str = f" [schema: {list(d.schema.keys())}]" if d.schema else ""
            print(f"    • {d.filename} ({d.format}){schema_str}")
        print(f"  Constraints  : {len(spec.constraints)}")
        for c in spec.constraints[:5]:
            print(f"    • {c}")
        if len(spec.constraints) > 5:
            print(f"    … and {len(spec.constraints) - 5} more")
        print(f"  Weights      : {spec.scoring_weights.as_dict()}")
        print(f"  Conditions   : {len(spec.conditions)}")
        for cond in spec.conditions:
            print(f"    • IF {cond.condition} → {cond.action}")
        print(f"  Ambiguities  : {len(spec.ambiguities)}")
        for amb in spec.ambiguities:
            print(f"    ⚠ {amb}")
        print(f"  Complete     : {'Yes' if spec.is_complete else 'No'}")
        print(f"{'═' * 60}")


if __name__ == "__main__":
    _cli()
