"""
tools/run.py — Master Orchestrator for ARIA-LAB v2

The brain of the system. Reads inputs/task.md and coordinates the full pipeline:
1. Parse task.md → TaskSpec
2. Detect task type and required tools
3. Build dynamic execution plan
4. Execute steps (parallel where possible via asyncio)
5. Handle failures with retry and fallback
6. Stream real-time progress to terminal
7. Generate reports in multiple formats
8. Extract learnings and update knowledge base

Usage:
    python tools/run.py
    python tools/run.py --task inputs/task.md
    python tools/run.py --dry-run   (show plan without executing)
    python tools/run.py --format md json csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

BASE_DIR  = Path(__file__).parent.parent
TOOLS_DIR = BASE_DIR / "tools"
sys.path.insert(0, str(TOOLS_DIR))

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Step System
# ──────────────────────────────────────────────

@dataclass
class Step:
    """A discrete pipeline step."""
    name:        str
    description: str
    fn:          Any                   # async callable
    args:        tuple                 = field(default_factory=tuple)
    kwargs:      dict                  = field(default_factory=dict)
    required:    bool                  = True
    retries:     int                   = 2
    timeout:     float                 = 300.0
    parallel_group: Optional[str]      = None  # steps in same group run in parallel
    result:      Any                   = None
    error:       Optional[str]         = None
    duration:    float                 = 0.0
    status:      str                   = "pending"   # pending | running | done | failed | skipped


@dataclass
class PipelineResult:
    """Final result of a pipeline run."""
    session_id:   str
    steps:        list[Step]
    outputs:      dict[str, str]       # format → file path
    findings:     dict
    duration:     float
    success:      bool


# ──────────────────────────────────────────────
# Progress Printer
# ──────────────────────────────────────────────

class ProgressPrinter:
    """Stream real-time progress to terminal."""

    WIDTH = 60

    def header(self, text: str) -> None:
        print(f"\n{'═'*self.WIDTH}")
        print(f"  {text}")
        print(f"{'═'*self.WIDTH}")

    def section(self, text: str) -> None:
        print(f"\n{'─'*self.WIDTH}")
        print(f"  {text}")

    def step_start(self, name: str, desc: str) -> None:
        print(f"\n  ⏳  {name}: {desc}", end="", flush=True)

    def step_done(self, name: str, duration: float, note: str = "") -> None:
        print(f"\r  ✅  {name} ({duration:.1f}s){' — ' + note if note else ''}")

    def step_fail(self, name: str, error: str) -> None:
        print(f"\r  ❌  {name} — {error[:80]}")

    def step_skip(self, name: str, reason: str) -> None:
        print(f"  ⏭️   {name} — skipped ({reason})")

    def info(self, msg: str) -> None:
        print(f"  ℹ️   {msg}")

    def warn(self, msg: str) -> None:
        print(f"  ⚠️   {msg}")

    def success(self, msg: str) -> None:
        print(f"  ✅  {msg}")

    def final(self, result: PipelineResult) -> None:
        self.header("ARIA-LAB Run Complete")
        status = "✅ SUCCESS" if result.success else "⚠️ PARTIAL SUCCESS"
        passed = sum(1 for s in result.steps if s.status == "done")
        total  = len(result.steps)
        print(f"  Status   : {status}")
        print(f"  Steps    : {passed}/{total} completed")
        print(f"  Duration : {result.duration:.1f}s")
        print(f"\n  Outputs:")
        for fmt, path in result.outputs.items():
            print(f"    [{fmt.upper()}] {path}")
        print()


# ──────────────────────────────────────────────
# Pipeline Executor
# ──────────────────────────────────────────────

class PipelineExecutor:
    """Executes steps with retry, timeout, and parallel group support."""

    def __init__(self, printer: ProgressPrinter) -> None:
        self.printer = printer

    async def run(self, steps: list[Step]) -> None:
        """Execute all steps, grouping parallel ones."""
        # Group steps by parallel_group
        pending = list(steps)
        while pending:
            step = pending.pop(0)
            if step.parallel_group:
                # Collect all steps in the same group
                siblings = [s for s in pending if s.parallel_group == step.parallel_group]
                for s in siblings:
                    pending.remove(s)
                group_steps = [step] + siblings
                await self._run_parallel(group_steps)
            else:
                await self._run_single(step)

    async def _run_single(self, step: Step) -> None:
        self.printer.step_start(step.name, step.description)
        step.status = "running"
        start = time.time()

        for attempt in range(step.retries + 1):
            try:
                result = await asyncio.wait_for(
                    step.fn(*step.args, **step.kwargs),
                    timeout=step.timeout,
                )
                step.result   = result
                step.status   = "done"
                step.duration = time.time() - start
                self.printer.step_done(step.name, step.duration)
                return
            except asyncio.TimeoutError:
                err = f"timeout after {step.timeout:.0f}s"
            except Exception as e:
                err = str(e)
                if attempt < step.retries:
                    wait = 2 ** attempt
                    logger.warning("Step '%s' attempt %d failed: %s — retrying in %ds",
                                   step.name, attempt + 1, err, wait)
                    await asyncio.sleep(wait)

        step.status   = "failed"
        step.error    = err
        step.duration = time.time() - start
        self.printer.step_fail(step.name, error=err or "unknown error")
        if step.required:
            raise RuntimeError(f"Required step '{step.name}' failed: {err}")

    async def _run_parallel(self, steps: list[Step]) -> None:
        """Run a group of steps concurrently."""
        self.printer.info(f"Running {len(steps)} steps in parallel: {[s.name for s in steps]}")
        tasks = [asyncio.create_task(self._run_single(s)) for s in steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for step, result in zip(steps, results):
            if isinstance(result, Exception) and step.required:
                raise result


# ──────────────────────────────────────────────
# Pipeline Builder
# ──────────────────────────────────────────────

class PipelineBuilder:
    """Builds a dynamic execution plan from a TaskSpec."""

    def __init__(self, spec: Any, state: Any, printer: ProgressPrinter) -> None:
        self.spec    = spec
        self.state   = state
        self.printer = printer
        self._research_results: list = []
        self._products:         list = []
        self._analysis_result:  Any  = None
        self._scored_opps:      list = []
        self._outputs:          dict = {}

    def build(self) -> list[Step]:
        """Build steps based on task type and detected tools."""
        steps: list[Step] = []

        # Step 1: Web research (always)
        if "web_research" in self.spec.required_tools:
            steps.append(Step(
                name="web_research",
                description=f"Deep web research for: {self.spec.goal[:50]}",
                fn=self._do_web_research,
                required=True, retries=2, timeout=180,
            ))

        # Step 2: Marketplace scraping (if detected)
        if "marketplace_scraper" in self.spec.required_tools:
            steps.append(Step(
                name="marketplace_scrape",
                description="Scraping marketplace platforms",
                fn=self._do_marketplace_scrape,
                required=False, retries=1, timeout=120,
                parallel_group="scraping",
            ))

        # Step 3: Browser scraping (if URLs in target or task)
        if "browser_scraper" in self.spec.required_tools:
            target_urls = self._extract_urls(self.spec.target)
            if target_urls:
                steps.append(Step(
                    name="browser_scrape",
                    description=f"Browser scraping {len(target_urls)} URL(s)",
                    fn=self._do_browser_scrape,
                    kwargs={"urls": target_urls},
                    required=False, retries=1, timeout=120,
                    parallel_group="scraping",
                ))

        # Step 4: Analysis
        steps.append(Step(
            name="analyze",
            description="LLM-powered analysis of gathered data",
            fn=self._do_analyze,
            required=True, retries=1, timeout=120,
        ))

        # Step 5: Data synthesis (if multiple sources)
        if "data_synthesizer" in self.spec.required_tools:
            steps.append(Step(
                name="synthesize",
                description="Fusing and deduplicating multi-source data",
                fn=self._do_synthesize,
                required=False, retries=1, timeout=60,
            ))

        # Step 6: Opportunity scoring (if task type is score/compare/research)
        from task_parser import TaskType
        if self.spec.task_type in (TaskType.SCORE, TaskType.COMPARE, TaskType.RESEARCH):
            steps.append(Step(
                name="score_opportunities",
                description="Scoring and ranking opportunities",
                fn=self._do_score,
                required=False, retries=1, timeout=30,
            ))

        # Step 7: Generate reports
        steps.append(Step(
            name="generate_reports",
            description=f"Generating reports: {self.spec.deliverables[:3]}",
            fn=self._do_report,
            required=True, retries=1, timeout=60,
        ))

        # Step 8: Save learnings to knowledge base
        steps.append(Step(
            name="save_learnings",
            description="Updating knowledge base with session learnings",
            fn=self._do_save_learnings,
            required=False, retries=1, timeout=30,
        ))

        return steps

    # ── Step Functions ──────────────────────────

    async def _do_web_research(self) -> list:
        from web_research import WebResearchEngine
        engine = WebResearchEngine()
        results = await engine.research(
            goal=f"{self.spec.target} {self.spec.goal}",
            num_queries=7,
            results_per_query=8,
            fetch_content=True,
        )
        self._research_results = [r.to_dict() for r in results]
        self.state.add_source(r.url, r.title) for r in results[:20] if hasattr(r, "url") and r.url
        # Save to working/
        out = BASE_DIR / "working" / "research_results.json"
        out.write_text(json.dumps(self._research_results, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Research: %d results saved to %s", len(results), out)
        return self._research_results

    async def _do_marketplace_scrape(self) -> list:
        from marketplace_scraper import scrape_all_platforms
        query = self.spec.target or self.spec.keywords[0] if self.spec.keywords else "digital products"
        data  = await scrape_all_platforms(query, limit=15)
        products = [p.to_dict() for plat_prods in data.values() for p in plat_prods]
        self._products = products
        out = BASE_DIR / "working" / "marketplace_data.json"
        out.write_text(json.dumps(products, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Marketplace: %d products from %d platforms", len(products), len(data))
        return products

    async def _do_browser_scrape(self, urls: list[str]) -> list:
        from browser_scraper import scrape_url
        all_pages = []
        for url in urls[:5]:
            try:
                pages = await scrape_url(url, extract_tables=True, extract_cards=True)
                all_pages.extend([p.to_dict() for p in pages])
            except Exception as e:
                logger.warning("Browser scrape failed for %s: %s", url, e)
        return all_pages

    async def _do_analyze(self) -> Any:
        from analyze import Analyzer

        # Build items list from all gathered data
        items: list[dict] = []
        for r in self._research_results[:30]:
            items.append({
                "text":  r.get("full_content", "") or r.get("snippet", ""),
                "title": r.get("title", ""),
                "url":   r.get("url", ""),
            })
        for p in self._products[:20]:
            items.append({
                "text":  p.get("description", "") or p.get("name", ""),
                "title": p.get("name", ""),
                "url":   p.get("url", ""),
            })

        if not items:
            items = [{"text": self.spec.goal, "title": "Task goal", "url": ""}]

        analyzer = Analyzer()
        result   = analyzer.analyze(items, mode="full", goal=self.spec.goal)
        self._analysis_result = result

        # Add insights to state
        for ins in result.insights[:5]:
            self.state.add_insight(ins.title)

        out = BASE_DIR / "working" / "analysis.json"
        out.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    async def _do_synthesize(self) -> list:
        from data_synthesizer import DataSynthesizer

        datasets = []
        if self._research_results:
            datasets.append(("web_research", self._research_results))
        if self._products:
            datasets.append(("marketplace", self._products))

        if not datasets:
            return []

        synth = DataSynthesizer()
        records, stats = synth.synthesize(datasets)
        out = BASE_DIR / "working" / "synthesized.json"
        out.write_text(
            json.dumps([r.to_dict() for r in records], indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info("Synthesized %d → %d records", stats.input_count, stats.output_count)
        return records

    async def _do_score(self) -> list:
        from opportunity_scorer import OpportunityScorer

        all_records: list[dict] = []
        synth_file = BASE_DIR / "working" / "synthesized.json"
        if synth_file.exists():
            raw = json.loads(synth_file.read_text(encoding="utf-8"))
            all_records = [_flatten_record(r) for r in raw]
        elif self._products:
            all_records = self._products
        elif self._research_results:
            all_records = self._research_results[:50]

        if not all_records:
            return []

        scorer = OpportunityScorer(
            weights=self.spec.scoring_weights.as_dict(),
            thresholds=self.spec.scoring_thresholds or None,
        )
        opps = scorer.score_all(all_records, top_n=25)
        self._scored_opps = [o.to_dict() for o in opps]

        out = BASE_DIR / "working" / "scored_opportunities.json"
        out.write_text(json.dumps(self._scored_opps, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Scored %d opportunities, top: %.2f", len(opps), opps[0].total_score if opps else 0)
        return opps

    async def _do_report(self) -> dict:
        from report import ReportGenerator, _make_serializable

        # Determine best template
        from task_parser import TaskType
        template_map = {
            TaskType.SCORE:   "opportunity_report.md.j2",
            TaskType.COMPARE: "competitive_landscape.md.j2",
            TaskType.ANALYZE: "market_analysis.md.j2",
        }
        template = template_map.get(self.spec.task_type, "market_analysis.md.j2")

        # Build context
        analysis = self._analysis_result
        context: dict = {
            "title":    self.spec.goal or "ARIA-LAB Research Report",
            "task":     self.spec.to_dict() if hasattr(self.spec, "to_dict") else {},
            "sources":  [{"url": r.get("url",""), "title": r.get("title","")}
                         for r in self._research_results[:30]],
            "insights":  _make_serializable(getattr(analysis, "insights", [])[:10]) if analysis else [],
            "trends":    _make_serializable(getattr(analysis, "trends",  [])[:8])   if analysis else [],
            "gaps":      _make_serializable(getattr(analysis, "gaps",    [])[:8])   if analysis else [],
            "entities":  _make_serializable(getattr(analysis, "entities",[])[:20])  if analysis else [],
            "executive_summary": getattr(analysis, "executive_summary", "") if analysis else "",
            "opportunities": self._scored_opps[:20],
            "products":   self._products[:30],
            "weights":    self.spec.scoring_weights.as_dict(),
        }

        # Determine formats from deliverables
        formats = list({
            d.format for d in self.spec.deliverables
            if d.format in ("md", "json", "csv", "html")
        }) or ["md", "json"]

        gen = ReportGenerator()
        outputs = gen.generate(
            context=context,
            template=template,
            formats=formats,
            base_name="report",
            version=True,
        )
        self._outputs = {fmt: str(path) for fmt, path in outputs.items()}

        for path in outputs.values():
            self.state.record_output(str(path))

        return self._outputs

    async def _do_save_learnings(self) -> None:
        from knowledge_base import KnowledgeBase, Entity
        import uuid

        kb    = KnowledgeBase()
        state = self.state.load()

        # Store top insights as entities
        analysis = self._analysis_result
        if analysis and hasattr(analysis, "insights"):
            for ins in analysis.insights[:5]:
                entity = Entity(
                    entity_id   = str(uuid.uuid4()),
                    entity_type = "insight",
                    name        = ins.title,
                    attributes  = {"body": ins.body, "confidence": ins.confidence},
                    tags        = [self.spec.task_type.value if hasattr(self.spec.task_type, "value") else ""],
                    source      = "aria-lab-run",
                    confidence  = ins.confidence,
                )
                kb.store(entity, ttl_hours=168)  # 1 week TTL

        # Store session summary
        summary = self.state.summarize()
        session_id = state.get("session_id", str(uuid.uuid4()))
        kb.save_session_summary(session_id, summary)

        # Add learning to session state
        self.state.add_learning(
            f"Researched {self.spec.target}: found {len(self._research_results)} sources, "
            f"{len(self._products)} products",
            source="aria-lab-run",
        )
        logger.info("Learnings saved to knowledge base")

    # ── Helpers ────────────────────────────────

    def _extract_urls(self, text: str) -> list[str]:
        import re
        return re.findall(r"https?://[^\s\"'>]+", text or "")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _flatten_record(record: dict) -> dict:
    """Flatten a SynthesizedRecord to a plain dict for scoring."""
    out = {"name": record.get("name", ""), "entity_type": record.get("entity_type", "")}
    for k, v in record.get("fields", {}).items():
        if isinstance(v, dict):
            out[k] = v.get("value")
        else:
            out[k] = v
    return out


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

async def main(
    task_file:   Optional[Path] = None,
    dry_run:     bool           = False,
    formats:     list[str]      = ("md", "json"),
    verbose:     bool           = False,
) -> PipelineResult:
    printer = ProgressPrinter()
    printer.header("ARIA-LAB v2 — Autonomous Research Agent")

    # ── Parse task ──
    sys.path.insert(0, str(TOOLS_DIR))
    from task_parser import TaskParser
    from remember   import StateManager

    tp   = TaskParser(base_dir=BASE_DIR)
    spec = tp.parse_file(task_file)
    printer.info(f"Task type  : {spec.task_type.value}")
    printer.info(f"Target     : {spec.target[:60]}")
    printer.info(f"Goal       : {spec.goal[:80]}")
    printer.info(f"Tools      : {', '.join(spec.required_tools)}")
    printer.info(f"Deliverables: {len(spec.deliverables)}")

    if spec.ambiguities:
        printer.warn(f"Ambiguities: {', '.join(spec.ambiguities[:2])}")

    # ── Init session ──
    state = StateManager()
    state.set_task(spec.target, spec.goal, spec.context,
                   [d.filename for d in spec.deliverables])
    state.log(f"Run started — task type: {spec.task_type.value}")

    # ── Inject past context ──
    past_context = state.inject_context(f"{spec.target} {spec.goal}")
    if "No prior" not in past_context:
        printer.info("Relevant past knowledge found — injecting context")

    # ── Build pipeline ──
    builder = PipelineBuilder(spec, state, printer)
    steps   = builder.build()

    printer.section(f"Execution Plan ({len(steps)} steps)")
    for i, step in enumerate(steps, 1):
        group = f" [parallel: {step.parallel_group}]" if step.parallel_group else ""
        print(f"  {i}. {step.name}: {step.description}{group}")

    if dry_run:
        printer.info("Dry run — stopping before execution")
        return PipelineResult(
            session_id=str(uuid.uuid4()), steps=steps,
            outputs={}, findings={}, duration=0, success=True
        )

    # ── Execute ──
    printer.section("Executing Pipeline")
    executor = PipelineExecutor(printer)
    start = time.time()
    try:
        await executor.run(steps)
        success = all(s.status in ("done", "skipped") for s in steps if s.required)
    except RuntimeError as e:
        printer.warn(f"Pipeline halted: {e}")
        success = False

    duration = time.time() - start

    # ── Finalize ──
    state.complete_step("pipeline")

    result = PipelineResult(
        session_id = state.load().get("session_id", ""),
        steps      = steps,
        outputs    = builder._outputs,
        findings   = state.load().get("findings", {}),
        duration   = round(duration, 1),
        success    = success,
    )
    printer.final(result)
    return result


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="ARIA-LAB v2 — Autonomous Research & Intelligence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python tools/run.py
  python tools/run.py --task inputs/task.md
  python tools/run.py --dry-run
  python tools/run.py --format md json csv html
""",
    )
    parser.add_argument("--task",    type=Path, help="Path to task.md (default: inputs/task.md)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--format",  nargs="+", default=["md", "json"],
                        choices=["md","json","csv","html"], help="Output formats")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    try:
        result = asyncio.run(main(
            task_file=args.task,
            dry_run=args.dry_run,
            formats=args.format,
            verbose=args.verbose,
        ))
        sys.exit(0 if result.success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Create inputs/task.md with your research task and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    _cli()
