"""
tools/report.py — Multi-Format Report Generator for ARIA-LAB v2

Generates professional reports in: Markdown, JSON, CSV, HTML
All formats simultaneously from one call.
Uses Jinja2 templating from templates/ directory.

Pre-built templates:
  * market_analysis.md.j2
  * opportunity_report.md.j2
  * competitive_landscape.md.j2
  * product_comparison.md.j2

Features:
- Executive summary (auto-generated 150-200 words)
- Auto-generated table of contents
- ASCII charts for data visualization
- Citation/source tracking
- Output versioning (filename_v2.md, etc.)
- Quality gates (min length, required sections, source citations)

Usage:
    python tools/report.py --from-state
    python tools/report.py --template market_analysis --data findings.json
    python tools/report.py --template opportunity_report --data scored.json --formats md json csv html
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUTS_DIR   = BASE_DIR / "outputs"


# ──────────────────────────────────────────────
# Report Models
# ──────────────────────────────────────────────

class ReportSection:
    """A named section of the report with content."""
    def __init__(self, title: str, content: str, level: int = 2) -> None:
        self.title   = title
        self.content = content
        self.level   = level   # markdown heading level (1-4)

    def to_markdown(self) -> str:
        hashes = "#" * self.level
        return f"{hashes} {self.title}\n\n{self.content}\n"


class Report:
    """Container for a fully realized report."""
    def __init__(
        self,
        title:     str,
        sections:  list[ReportSection],
        metadata:  dict,
        sources:   list[dict],
    ) -> None:
        self.title    = title
        self.sections = sections
        self.metadata = metadata
        self.sources  = sources
        self.generated_at = datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────
# ASCII Chart Renderer
# ──────────────────────────────────────────────

class ASCIIChart:
    """Generate simple ASCII bar charts."""

    @staticmethod
    def bar(data: list[tuple[str, float]], width: int = 40, title: str = "") -> str:
        if not data:
            return ""
        max_val   = max(v for _, v in data) or 1
        max_label = max(len(label) for label, _ in data)
        lines = []
        if title:
            lines.append(f"\n{title}")
            lines.append("─" * (max_label + width + 10))
        for label, value in data:
            bar_len = int((value / max_val) * width)
            bar     = "█" * bar_len
            lines.append(f"  {label:<{max_label}}  {bar} {value:.1f}")
        return "\n".join(lines)

    @staticmethod
    def sparkline(values: list[float]) -> str:
        if not values:
            return ""
        chars = "▁▂▃▄▅▆▇█"
        mn, mx = min(values), max(values)
        if mx == mn:
            return chars[4] * len(values)
        return "".join(chars[int((v - mn) / (mx - mn) * 7)] for v in values)


# ──────────────────────────────────────────────
# Jinja2 Template Renderer
# ──────────────────────────────────────────────

class TemplateRenderer:
    """Renders Jinja2 templates from the templates/ directory."""

    def __init__(self, templates_dir: Path = TEMPLATES_DIR) -> None:
        self.templates_dir = templates_dir

    def render(self, template_name: str, context: dict) -> str:
        """Render a template with given context. Falls back to inline rendering."""
        template_path = self.templates_dir / template_name
        if template_path.exists():
            template_src = template_path.read_text(encoding="utf-8")
        else:
            # Use built-in templates
            template_src = self._builtin(template_name)

        try:
            from jinja2 import Environment, BaseLoader, StrictUndefined
            env = Environment(
                loader=BaseLoader(),
                undefined=StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            env.filters["bar_chart"] = lambda data: ASCIIChart.bar(
                [(k, float(v)) for k, v in data.items()]
            )
            return env.from_string(template_src).render(**context)
        except ImportError:
            # Jinja2 not installed — use simple string substitution
            return self._simple_render(template_src, context)
        except Exception as e:
            logger.warning("Template render failed (%s), using fallback: %s", template_name, e)
            return self._simple_render(template_src, context)

    def _simple_render(self, template: str, context: dict) -> str:
        """Simple {{variable}} substitution fallback."""
        result = template
        for key, value in context.items():
            if isinstance(value, (str, int, float)):
                result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def _builtin(self, name: str) -> str:
        """Return built-in template strings when file doesn't exist."""
        templates = {
            "market_analysis.md.j2": BUILTIN_MARKET_ANALYSIS,
            "opportunity_report.md.j2": BUILTIN_OPPORTUNITY_REPORT,
            "competitive_landscape.md.j2": BUILTIN_COMPETITIVE_LANDSCAPE,
            "product_comparison.md.j2": BUILTIN_PRODUCT_COMPARISON,
        }
        return templates.get(name, BUILTIN_GENERIC)


# ──────────────────────────────────────────────
# Built-in Jinja2 Templates
# ──────────────────────────────────────────────

BUILTIN_MARKET_ANALYSIS = """# {{ title }}

**Generated:** {{ generated_at }}  
**Sources analysed:** {{ sources | length }}  
**Task:** {{ task.goal if task else 'N/A' }}

---

## Executive Summary

{{ executive_summary }}

---

## Table of Contents

1. [Key Findings](#key-findings)
2. [Market Trends](#market-trends)
3. [Top Products/Entities](#top-productsentitiess)
4. [Market Gaps & Opportunities](#market-gaps--opportunities)
5. [Sources](#sources)

---

## Key Findings

{% for insight in insights %}
### {{ insight.title }}

{{ insight.body }}

{% if insight.evidence %}
**Evidence:**
{% for e in insight.evidence %}
- {{ e }}
{% endfor %}
{% endif %}

> Confidence: {{ (insight.confidence * 100)|round|int }}%  
{% endfor %}

---

## Market Trends

{% for trend in trends %}
**{{ trend.topic }}** — {{ trend.direction | upper }}  
{{ trend.evidence | join('; ') }}  
_(Confidence: {{ (trend.confidence * 100)|round|int }}%)_

{% endfor %}

---

## Top Products/Entities

{% for entity in entities[:20] %}
- **{{ entity.text }}** _({{ entity.entity_type }})_ — {{ entity.context[:80] }}
{% endfor %}

---

## Market Gaps & Opportunities

{% for gap in gaps %}
### Gap: {{ gap.category | replace('_', ' ') | title }}

{{ gap.description }}

{% if gap.opportunity %}**Opportunity:** {{ gap.opportunity }}{% endif %}

> Confidence: {{ (gap.confidence * 100)|round|int }}%

{% endfor %}

---

## Sources

{% for source in sources %}
{{ loop.index }}. [{{ source.title or source.url }}]({{ source.url }})
{% endfor %}
"""

BUILTIN_OPPORTUNITY_REPORT = """# {{ title }}

**Generated:** {{ generated_at }}  
**Total Opportunities Scored:** {{ opportunities | length }}

---

## Executive Summary

{{ executive_summary }}

---

## Table of Contents

1. [Top Opportunities](#top-opportunities)
2. [Score Breakdown](#score-breakdown)
3. [Comparison Matrix](#comparison-matrix)

---

## Top Opportunities

{% for opp in opportunities %}
### #{{ opp.rank }} — {{ opp.name }}

**Overall Score:** {{ opp.total_score }}/10 {% if opp.meets_threshold %}✅{% else %}⚠️ (below threshold){% endif %}

| Criterion | Score | Explanation |
|-----------|-------|-------------|
{% for c in opp.criteria %}| {{ c.name | replace('_',' ') | title }} | {{ c.raw_score }}/10 | {{ c.explanation[:80] }} |
{% endfor %}

{% if opp.url %}**URL:** {{ opp.url }}{% endif %}

{% endfor %}

---

## Comparison Matrix

{{ comparison_matrix }}

---

## Scoring Weights

| Criterion | Weight |
|-----------|--------|
{% for k, v in weights.items() %}| {{ k | replace('_',' ') | title }} | {{ (v * 100)|round|int }}% |
{% endfor %}
"""

BUILTIN_COMPETITIVE_LANDSCAPE = """# {{ title }}

**Generated:** {{ generated_at }}

---

## Executive Summary

{{ executive_summary }}

---

## Competitive Overview

| Name | Platform | Price | Sales Est. | Rating | Reviews |
|------|----------|-------|-----------|--------|---------|
{% for p in products %}| {{ p.name[:30] }} | {{ p.platform }} | {{ p.price_raw or 'N/A' }} | {{ p.sales_estimate or 'N/A' }} | {{ p.rating or 'N/A' }} | {{ p.reviews_count or 'N/A' }} |
{% endfor %}

---

## Market Insights

{{ insights_text }}

---

## Sources

{% for source in sources %}
- {{ source.url }}
{% endfor %}
"""

BUILTIN_PRODUCT_COMPARISON = """# {{ title }}

**Generated:** {{ generated_at }}

---

## Product Comparison

{% for product in products %}
### {{ product.name }}

| Field | Value |
|-------|-------|
| Platform | {{ product.platform }} |
| Price | {{ product.price_raw or 'N/A' }} |
| Rating | {{ product.rating or 'N/A' }} |
| Reviews | {{ product.reviews_count or 'N/A' }} |
| Sales Est. | {{ product.sales_estimate or 'N/A' }} |
| URL | {{ product.url }} |

{{ product.description }}

{% endfor %}
"""

BUILTIN_GENERIC = """# {{ title }}

**Generated:** {{ generated_at }}

---

## Executive Summary

{{ executive_summary }}

---

## Key Findings

{{ content }}

---

## Sources

{% for source in sources %}
- {{ source.url }}
{% endfor %}
"""


# ──────────────────────────────────────────────
# Quality Gates
# ──────────────────────────────────────────────

class QualityGate:
    """Validate report meets minimum quality standards before writing."""

    MIN_LENGTH      = 300    # characters
    REQUIRED_WORDS  = {"summary", "source", "finding", "insight", "result"}

    def check(self, content: str) -> list[str]:
        """Returns list of quality issues (empty = pass)."""
        issues: list[str] = []
        if len(content) < self.MIN_LENGTH:
            issues.append(f"Report too short ({len(content)} chars, min {self.MIN_LENGTH})")
        if not any(w in content.lower() for w in self.REQUIRED_WORDS):
            issues.append("Report lacks key sections (summary/findings/insights/sources)")
        if "source" not in content.lower() and "http" not in content.lower():
            issues.append("No sources cited in report")
        return issues


# ──────────────────────────────────────────────
# Versioned Output
# ──────────────────────────────────────────────

def _versioned_path(path: Path) -> Path:
    """Return path_v2.ext, path_v3.ext, etc. if path exists."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for v in range(2, 100):
        candidate = path.parent / f"{stem}_v{v}{suffix}"
        if not candidate.exists():
            return candidate
    return path.parent / f"{stem}_final{suffix}"


# ──────────────────────────────────────────────
# Report Generator
# ──────────────────────────────────────────────

class ReportGenerator:
    """
    Generates professional reports in multiple formats simultaneously.
    """

    def __init__(self, outputs_dir: Path = OUTPUTS_DIR) -> None:
        self.outputs_dir = outputs_dir
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.renderer = TemplateRenderer()
        self.gate     = QualityGate()

    def generate(
        self,
        context:   dict,
        template:  str              = "market_analysis.md.j2",
        formats:   list[str]        = ("md", "json"),
        base_name: str              = "report",
        version:   bool             = True,
    ) -> dict[str, Path]:
        """
        Generate report in all requested formats.
        Returns dict of {format: output_path}.
        """
        ctx = self._enrich_context(context)
        outputs: dict[str, Path] = {}

        for fmt in formats:
            try:
                path = self._write_format(ctx, template, fmt, base_name, version)
                if path:
                    outputs[fmt] = path
                    logger.info("Report written: %s", path)
            except Exception as e:
                logger.error("Failed to generate %s format: %s", fmt, e)

        return outputs

    def _enrich_context(self, ctx: dict) -> dict:
        """Add auto-generated fields to context."""
        ctx = dict(ctx)
        ctx.setdefault("generated_at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        ctx.setdefault("sources",      [])
        ctx.setdefault("insights",     [])
        ctx.setdefault("trends",       [])
        ctx.setdefault("gaps",         [])
        ctx.setdefault("entities",     [])
        ctx.setdefault("opportunities", [])
        ctx.setdefault("products",     [])
        ctx.setdefault("weights",      {})

        # Auto-generate executive summary if not provided
        if not ctx.get("executive_summary"):
            ctx["executive_summary"] = self._auto_summary(ctx)

        # Add comparison matrix for opportunity reports
        if not ctx.get("comparison_matrix") and ctx.get("opportunities"):
            from opportunity_scorer import OpportunityScorer
            scorer = OpportunityScorer()
            ctx["comparison_matrix"] = scorer.compare(
                [type("O", (), o)() for o in ctx["opportunities"]]  # type: ignore
            )

        return ctx

    def _auto_summary(self, ctx: dict) -> str:
        """Generate a 150-200 word executive summary from context data."""
        parts: list[str] = []

        title   = ctx.get("title", "This report")
        task    = ctx.get("task", {})
        goal    = task.get("goal", "") if isinstance(task, dict) else ""
        n_src   = len(ctx.get("sources", []))
        n_ins   = len(ctx.get("insights", []))
        n_opps  = len(ctx.get("opportunities", []))

        parts.append(f"{title} presents findings from analysis of {n_src} sources.")
        if goal:
            parts.append(f"The research goal was: {goal}")

        trends = ctx.get("trends", [])
        growing = [t.get("topic") or (t.topic if hasattr(t,"topic") else "") 
                   for t in trends if (t.get("direction") or getattr(t,"direction","")) == "growing"]
        if growing:
            parts.append(f"Key growing trends identified: {', '.join(growing[:3])}.")

        gaps = ctx.get("gaps", [])
        if gaps:
            parts.append(f"Analysis revealed {len(gaps)} market gap(s) and underserved segment(s).")

        if n_ins > 0:
            parts.append(f"A total of {n_ins} actionable insights were extracted.")

        if n_opps > 0:
            qual = sum(1 for o in ctx["opportunities"] if isinstance(o, dict) and o.get("meets_threshold", True))
            parts.append(f"Out of {n_opps} scored opportunities, {qual} meet all quality thresholds.")

        insights = ctx.get("insights", [])
        if insights:
            top = insights[0]
            top_title = top.get("title") if isinstance(top, dict) else getattr(top, "title", "")
            if top_title:
                parts.append(f"The top insight is: {top_title}.")

        summary = " ".join(parts)
        # Target 150-200 words
        words = summary.split()
        if len(words) > 200:
            summary = " ".join(words[:200]) + "…"
        return summary

    def _write_format(
        self,
        ctx:       dict,
        template:  str,
        fmt:       str,
        base_name: str,
        version:   bool,
    ) -> Optional[Path]:
        if fmt == "md":
            return self._write_markdown(ctx, template, base_name, version)
        elif fmt == "json":
            return self._write_json(ctx, base_name, version)
        elif fmt == "csv":
            return self._write_csv(ctx, base_name, version)
        elif fmt == "html":
            return self._write_html(ctx, base_name, version)
        else:
            logger.warning("Unknown format: %s", fmt)
            return None

    def _write_markdown(self, ctx: dict, template: str, base_name: str, version: bool) -> Path:
        content = self.renderer.render(template, ctx)

        # Add table of contents if not present
        if "Table of Contents" not in content and "## " in content:
            toc   = self._build_toc(content)
            split = content.split("\n---\n", 1)
            content = split[0] + "\n---\n\n" + toc + "\n\n---\n" + (split[1] if len(split)>1 else "")

        # Quality gate
        issues = self.gate.check(content)
        if issues:
            logger.warning("Quality gate issues: %s", "; ".join(issues))

        path = _versioned_path(self.outputs_dir / f"{base_name}.md") if version else (
            self.outputs_dir / f"{base_name}.md"
        )
        path.write_text(content, encoding="utf-8")
        return path

    def _write_json(self, ctx: dict, base_name: str, version: bool) -> Path:
        # Sanitize context for JSON serialization
        data = _make_serializable(ctx)
        path = _versioned_path(self.outputs_dir / f"{base_name}.json") if version else (
            self.outputs_dir / f"{base_name}.json"
        )
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _write_csv(self, ctx: dict, base_name: str, version: bool) -> Path:
        path = _versioned_path(self.outputs_dir / f"{base_name}.csv") if version else (
            self.outputs_dir / f"{base_name}.csv"
        )

        # Determine what tabular data is available
        rows: list[dict] = []
        if ctx.get("opportunities"):
            for o in ctx["opportunities"]:
                if isinstance(o, dict):
                    rows.append({
                        "name":        o.get("name",""),
                        "score":       o.get("total_score",""),
                        "rank":        o.get("rank",""),
                        "source":      o.get("source",""),
                        "url":         o.get("url",""),
                        "meets_threshold": o.get("meets_threshold",""),
                    })
        elif ctx.get("products"):
            for p in ctx["products"]:
                if isinstance(p, dict):
                    rows.append({
                        "name":     p.get("name",""),
                        "platform": p.get("platform",""),
                        "price":    p.get("price",""),
                        "rating":   p.get("rating",""),
                        "reviews":  p.get("reviews_count",""),
                        "sales":    p.get("sales_estimate",""),
                        "url":      p.get("url",""),
                    })
        elif ctx.get("entities"):
            for e in ctx["entities"]:
                if isinstance(e, dict):
                    rows.append(e)

        if not rows:
            rows = [{"key": k, "value": str(v)[:200]} for k, v in ctx.items()
                    if isinstance(v, (str, int, float, bool))]

        buf = io.StringIO()
        if rows:
            writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        path.write_text(buf.getvalue(), encoding="utf-8")
        return path

    def _write_html(self, ctx: dict, base_name: str, version: bool) -> Path:
        # Generate Markdown first, then convert to HTML
        md_content = self.renderer.render("market_analysis.md.j2", ctx)
        html = self._md_to_html(md_content, ctx.get("title", "ARIA-LAB Report"))

        path = _versioned_path(self.outputs_dir / f"{base_name}.html") if version else (
            self.outputs_dir / f"{base_name}.html"
        )
        path.write_text(html, encoding="utf-8")
        return path

    def _md_to_html(self, md: str, title: str) -> str:
        """Convert markdown to HTML (basic, no external library required)."""
        html = md
        # Headings
        for level in range(4, 0, -1):
            html = re.sub(
                r"^" + "#" * level + r" (.+)$",
                lambda m, l=level: f"<h{l}>{m.group(1)}</h{l}>",
                html, flags=re.MULTILINE
            )
        # Bold, italic
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         html)
        # Links
        html = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html)
        # Inline code
        html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)
        # HR
        html = html.replace("---", "<hr>")
        # Tables
        html = self._convert_tables(html)
        # Blockquotes
        html = re.sub(r"^> (.+)$", r"<blockquote>\1</blockquote>", html, flags=re.MULTILINE)
        # Lists
        html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
        html = re.sub(r"(<li>.+</li>\n?)+", r"<ul>\g<0></ul>", html)
        # Paragraphs
        html = re.sub(r"\n\n", r"</p><p>", html)
        html = f"<p>{html}</p>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 960px; margin: 40px auto; padding: 0 20px; color: #333; line-height: 1.6; }}
  h1,h2,h3,h4 {{ color: #1a1a2e; border-bottom: 1px solid #eee; padding-bottom: 8px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  blockquote {{ background: #f9f9f9; border-left: 4px solid #007bff; padding: 8px 16px; }}
  code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
  a {{ color: #007bff; }}
  hr {{ border: none; border-top: 1px solid #eee; margin: 24px 0; }}
</style>
</head>
<body>
{html}
</body>
</html>"""

    def _convert_tables(self, html: str) -> str:
        """Convert Markdown tables to HTML tables."""
        table_re = re.compile(
            r"(\|[^\n]+\|\n\|[-| :]+\|\n(?:\|[^\n]+\|\n)*)",
            re.MULTILINE
        )
        def _replace_table(m: re.Match) -> str:
            lines = [l for l in m.group(1).strip().splitlines() if l.strip()]
            if len(lines) < 2:
                return m.group(0)
            header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
            rows = []
            for row_line in lines[2:]:
                cells = [c.strip() for c in row_line.strip("|").split("|")]
                rows.append(cells)
            html_table = "<table>\n<thead><tr>"
            html_table += "".join(f"<th>{c}</th>" for c in header_cells)
            html_table += "</tr></thead>\n<tbody>"
            for row in rows:
                html_table += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            html_table += "</tbody></table>"
            return html_table
        return table_re.sub(_replace_table, html)

    def _build_toc(self, content: str) -> str:
        """Generate a Markdown table of contents from heading lines."""
        toc_lines = ["## Table of Contents\n"]
        for line in content.splitlines():
            m = re.match(r"^(#{2,4})\s+(.+)$", line)
            if m:
                level  = len(m.group(1)) - 2
                title  = m.group(2).strip()
                anchor = re.sub(r"[^a-z0-9\- ]", "", title.lower()).replace(" ", "-")
                indent = "  " * level
                toc_lines.append(f"{indent}- [{title}](#{anchor})")
        return "\n".join(toc_lines)


# ──────────────────────────────────────────────
# Serialization Helper
# ──────────────────────────────────────────────

def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    elif hasattr(obj, "to_dict"):
        return _make_serializable(obj.to_dict())
    elif hasattr(obj, "__dataclass_fields__"):
        import dataclasses
        return _make_serializable(dataclasses.asdict(obj))
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="ARIA-LAB Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python tools/report.py --from-state
  python tools/report.py --data findings.json --template market_analysis
  python tools/report.py --data scored.json --template opportunity_report --formats md json csv html
""",
    )
    parser.add_argument("--from-state", action="store_true", help="Build report from state/session.json")
    parser.add_argument("--data",       type=Path,           help="JSON data file")
    parser.add_argument("--template",   type=str, default="market_analysis", help="Template name (without .md.j2)")
    parser.add_argument("--formats",    nargs="+", default=["md", "json"],
                        choices=["md", "json", "csv", "html"], help="Output formats")
    parser.add_argument("--out",        type=str, default="report", help="Base output filename")
    parser.add_argument("--title",      type=str, default="ARIA-LAB Research Report", help="Report title")
    parser.add_argument("--no-version", action="store_true", help="Overwrite existing files")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    gen = ReportGenerator()

    # Build context
    context: dict = {"title": args.title}

    if args.from_state:
        state_file = BASE_DIR / "state" / "session.json"
        if state_file.exists():
            context.update(json.loads(state_file.read_text(encoding="utf-8")))
            context["title"] = context.get("task", {}).get("goal", args.title) or args.title
    elif args.data and args.data.exists():
        data = json.loads(args.data.read_text(encoding="utf-8"))
        if isinstance(data, list):
            # Detect type
            first = data[0] if data else {}
            if "total_score" in first:
                context["opportunities"] = data
                context.setdefault("title", "Opportunity Report")
            elif "platform" in first:
                context["products"] = data
                context.setdefault("title", "Market Intelligence Report")
            else:
                context["entities"] = data
        elif isinstance(data, dict):
            context.update(data)

    template_name = args.template
    if not template_name.endswith(".j2"):
        template_name += ".md.j2"

    outputs = gen.generate(
        context=context,
        template=template_name,
        formats=args.formats,
        base_name=args.out,
        version=not args.no_version,
    )

    print(f"\nGenerated {len(outputs)} report file(s):")
    for fmt, path in outputs.items():
        print(f"  [{fmt.upper()}] {path}")


if __name__ == "__main__":
    _cli()
