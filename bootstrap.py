"""
tools/bootstrap.py — Environment Setup & Validator for ARIA-LAB v2

Responsibilities:
- Install all required packages from requirements.txt
- Validate API keys (test each one with a live call)
- Create directory structure
- Initialize knowledge base
- Run self-test suite
- Report system readiness

Usage:
    python tools/bootstrap.py
    python tools/bootstrap.py --skip-install
    python tools/bootstrap.py --check-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent

REQUIRED_DIRS = [
    "inputs", "outputs", "working", "working/web_cache",
    "working/scrape_cache", "working/screenshots",
    "working/marketplace_cache", "state", "state/knowledge",
    "state/knowledge", "templates",
]

REQUIRED_PACKAGES = [
    "requests",
    "aiohttp",
    "beautifulsoup4",
    "jinja2",
    "tqdm",
    "python-dotenv",
]

OPTIONAL_PACKAGES = {
    "playwright":           "Headless browser scraping",
    "sentence-transformers": "Semantic similarity search",
    "openpyxl":             "Excel file profiling",
    "pyyaml":               "YAML file profiling",
    "pdfplumber":           "PDF text extraction",
    "pypdf":                "PDF fallback (lighter than pdfplumber)",
}


# ──────────────────────────────────────────────
# Console Helpers
# ──────────────────────────────────────────────

def _ok(msg: str)  -> None: print(f"  ✅  {msg}")
def _warn(msg: str) -> None: print(f"  ⚠️  {msg}")
def _fail(msg: str) -> None: print(f"  ❌  {msg}")
def _info(msg: str) -> None: print(f"  ℹ️  {msg}")
def _head(msg: str) -> None:
    print(f"\n{'─'*55}\n  {msg}\n{'─'*55}")


# ──────────────────────────────────────────────
# Install Packages
# ──────────────────────────────────────────────

def install_packages(skip: bool = False) -> bool:
    _head("Step 1 — Installing required packages")
    if skip:
        _info("Skipping package installation (--skip-install)")
        return True

    req_file = BASE_DIR / "requirements.txt"
    if req_file.exists():
        _info(f"Installing from {req_file}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            _ok("Requirements installed successfully")
        else:
            _warn(f"Some requirements failed:\n{result.stderr[:300]}")
    else:
        _warn("requirements.txt not found — installing core packages directly")
        for pkg in REQUIRED_PACKAGES:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                _ok(f"{pkg}")
            else:
                _fail(f"{pkg}: {result.stderr[:100]}")

    # Check optional packages
    _info("Checking optional packages:")
    for pkg, desc in OPTIONAL_PACKAGES.items():
        pkg_name = pkg.replace("-", "_").split("[")[0]
        try:
            __import__(pkg_name.replace("-","_"))
            _ok(f"{pkg} — {desc}")
        except ImportError:
            _warn(f"{pkg} not installed (optional) — {desc}")

    return True


# ──────────────────────────────────────────────
# Create Directories
# ──────────────────────────────────────────────

def create_directories() -> bool:
    _head("Step 2 — Creating directory structure")
    all_ok = True
    for dir_name in REQUIRED_DIRS:
        p = BASE_DIR / dir_name
        try:
            p.mkdir(parents=True, exist_ok=True)
            gitkeep = p / ".gitkeep"
            if not any(p.iterdir()):
                gitkeep.touch()
            _ok(f"{dir_name}/")
        except Exception as e:
            _fail(f"{dir_name}/: {e}")
            all_ok = False
    return all_ok


# ──────────────────────────────────────────────
# Validate API Keys
# ──────────────────────────────────────────────

def validate_api_keys() -> dict[str, bool]:
    _head("Step 3 — Validating API keys")

    # Load .env if present
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        _info(f"Loading .env from {env_file}")
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    results: dict[str, bool] = {}

    # EXA_API_KEY
    exa_key = os.environ.get("EXA_API_KEY", "")
    if exa_key:
        try:
            from urllib.request import urlopen, Request
            from urllib.error import HTTPError
            payload = json.dumps({"query": "test", "numResults": 1}).encode()
            req = Request(
                "https://api.exa.ai/search",
                data=payload,
                headers={"Content-Type": "application/json", "x-api-key": exa_key},
                method="POST"
            )
            urlopen(req, timeout=10)
            _ok("EXA_API_KEY — valid")
            results["exa"] = True
        except HTTPError as e:
            if e.code == 401:
                _fail("EXA_API_KEY — invalid (401 Unauthorized)")
                results["exa"] = False
            else:
                _ok(f"EXA_API_KEY — set (HTTP {e.code}, may be valid)")
                results["exa"] = True
        except Exception as e:
            _warn(f"EXA_API_KEY — could not verify ({e})")
            results["exa"] = None
    else:
        _warn("EXA_API_KEY — not set (will use DuckDuckGo fallback)")
        results["exa"] = False

    # SERPAPI_KEY
    serp_key = os.environ.get("SERPAPI_KEY", "")
    if serp_key:
        try:
            from urllib.request import urlopen, Request
            url = f"https://serpapi.com/account?api_key={serp_key}"
            raw = urlopen(Request(url), timeout=10).read()
            data = json.loads(raw)
            if data.get("account_email"):
                _ok(f"SERPAPI_KEY — valid ({data['account_email']})")
                results["serp"] = True
            else:
                _warn("SERPAPI_KEY — unknown response")
                results["serp"] = None
        except Exception as e:
            _warn(f"SERPAPI_KEY — set but unverified ({e})")
            results["serp"] = None
    else:
        _warn("SERPAPI_KEY — not set (optional)")
        results["serp"] = False

    # OPENAI_API_KEY
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        try:
            from urllib.request import urlopen, Request
            from urllib.error import HTTPError
            req = Request(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {openai_key}"}
            )
            urlopen(req, timeout=10)
            _ok("OPENAI_API_KEY — valid")
            results["openai"] = True
        except HTTPError as e:
            if e.code == 401:
                _fail("OPENAI_API_KEY — invalid (401)")
                results["openai"] = False
            else:
                _ok(f"OPENAI_API_KEY — set (HTTP {e.code})")
                results["openai"] = True
        except Exception as e:
            _warn(f"OPENAI_API_KEY — unverified ({e})")
            results["openai"] = None
    else:
        _warn("OPENAI_API_KEY — not set (will use rule-based fallback)")
        results["openai"] = False

    # ANTHROPIC_API_KEY
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        _ok("ANTHROPIC_API_KEY — set (not verified, requires credit)")
        results["anthropic"] = True
    else:
        _warn("ANTHROPIC_API_KEY — not set (optional)")
        results["anthropic"] = False

    # Ollama local
    try:
        from urllib.request import urlopen
        urlopen("http://localhost:11434/api/tags", timeout=2)
        _ok("Ollama — running locally at port 11434")
        results["ollama"] = True
    except Exception:
        _info("Ollama — not running (optional)")
        results["ollama"] = False

    return results


# ──────────────────────────────────────────────
# Initialize Knowledge Base
# ──────────────────────────────────────────────

def init_knowledge_base() -> bool:
    _head("Step 4 — Initializing knowledge base")
    try:
        sys.path.insert(0, str(BASE_DIR / "tools"))
        from knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        stats = kb.stats()
        _ok(f"Knowledge base ready ({stats['total_entities']} entities, {stats['db_size_kb']} KB)")
        return True
    except Exception as e:
        _fail(f"Knowledge base init failed: {e}")
        return False


# ──────────────────────────────────────────────
# Initialize Session State
# ──────────────────────────────────────────────

def init_session() -> bool:
    _head("Step 5 — Initializing session state")
    try:
        from remember import StateManager
        sm = StateManager()
        state = sm.load()
        _ok(f"Session state ready (id: {state['session_id'][:8]}…)")
        return True
    except Exception as e:
        _fail(f"Session init failed: {e}")
        return False


# ──────────────────────────────────────────────
# Self-test Suite
# ──────────────────────────────────────────────

def run_self_tests() -> dict[str, bool]:
    _head("Step 6 — Running self-test suite")
    results: dict[str, bool] = {}

    # Test task_parser
    try:
        from task_parser import TaskParser
        spec = TaskParser().parse_text("# Target\nGumroad\n\n# Goal\nFind top AI tools\n")
        assert spec.target == "Gumroad"
        assert spec.goal
        _ok("task_parser — OK")
        results["task_parser"] = True
    except Exception as e:
        _fail(f"task_parser — FAIL: {e}")
        results["task_parser"] = False

    # Test web_research (query generator only — no network)
    try:
        from web_research import QueryGenerator
        q = QueryGenerator().generate("test AI tools", n=5)
        assert len(q) >= 3
        _ok(f"web_research — OK ({len(q)} queries generated)")
        results["web_research"] = True
    except Exception as e:
        _fail(f"web_research — FAIL: {e}")
        results["web_research"] = False

    # Test data_synthesizer
    try:
        from data_synthesizer import DataSynthesizer
        records, stats = DataSynthesizer().synthesize(
            [("src_a", [{"name": "Product X", "price": 29.0}]),
             ("src_b", [{"name": "Product X", "price": 31.0}])],
        )
        assert records
        _ok(f"data_synthesizer — OK ({stats.conflicts} conflicts resolved)")
        results["data_synthesizer"] = True
    except Exception as e:
        _fail(f"data_synthesizer — FAIL: {e}")
        results["data_synthesizer"] = False

    # Test opportunity_scorer
    try:
        from opportunity_scorer import OpportunityScorer
        opps = OpportunityScorer().score_all([
            {"name": "AI Templates", "price": 29, "sales_estimate": 500, "platform": "gumroad"},
        ])
        assert opps and opps[0].total_score > 0
        _ok(f"opportunity_scorer — OK (score: {opps[0].total_score:.2f}/10)")
        results["opportunity_scorer"] = True
    except Exception as e:
        _fail(f"opportunity_scorer — FAIL: {e}")
        results["opportunity_scorer"] = False

    # Test profile
    try:
        from profile import FileProfiler
        test_file = BASE_DIR / "tools" / "task_parser.py"
        p = FileProfiler().profile(test_file)
        assert not p.error
        _ok(f"profile — OK ({p.summary[:60]})")
        results["profile"] = True
    except Exception as e:
        _fail(f"profile — FAIL: {e}")
        results["profile"] = False

    # Test remember
    try:
        from remember import StateManager
        sm = StateManager(state_file=BASE_DIR / "working" / "_test_state.json")
        sm.write("task.goal", "test goal")
        assert sm.read("task.goal") == "test goal"
        sm.state_file.unlink(missing_ok=True)
        _ok("remember — OK")
        results["remember"] = True
    except Exception as e:
        _fail(f"remember — FAIL: {e}")
        results["remember"] = False

    # Test knowledge_base
    try:
        from knowledge_base import KnowledgeBase, Entity
        import uuid
        kb = KnowledgeBase()
        eid = kb.store(Entity(
            entity_id=str(uuid.uuid4()), entity_type="test",
            name="Bootstrap Test", source="bootstrap.py"
        ))
        e = kb.get(eid)
        assert e and e.name == "Bootstrap Test"
        kb.delete(eid)
        _ok("knowledge_base — OK")
        results["knowledge_base"] = True
    except Exception as e:
        _fail(f"knowledge_base — FAIL: {e}")
        results["knowledge_base"] = False

    # Test report generator
    try:
        from report import ReportGenerator
        gen = ReportGenerator(outputs_dir=BASE_DIR / "working")
        outputs = gen.generate(
            context={"title": "Test", "sources": [], "insights": []},
            formats=["md"],
            base_name="_test_report",
            version=False,
        )
        assert "md" in outputs
        outputs["md"].unlink(missing_ok=True)
        _ok("report — OK")
        results["report"] = True
    except Exception as e:
        _fail(f"report — FAIL: {e}")
        results["report"] = False

    return results


# ──────────────────────────────────────────────
# Playwright Install
# ──────────────────────────────────────────────

def install_playwright_browsers(skip: bool = False) -> None:
    _head("Step 7 — Playwright browser setup (optional)")
    if skip:
        _info("Skipping Playwright installation")
        return
    try:
        import playwright
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium", "--with-deps"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            _ok("Playwright Chromium installed")
        else:
            _warn(f"Playwright install: {result.stderr[:200]}")
    except ImportError:
        _warn("Playwright not installed — browser scraping will use fallback")
    except subprocess.TimeoutExpired:
        _warn("Playwright install timed out — run manually: playwright install chromium")
    except Exception as e:
        _warn(f"Playwright setup: {e}")


# ──────────────────────────────────────────────
# Summary Report
# ──────────────────────────────────────────────

def print_summary(api_results: dict[str, bool], test_results: dict[str, bool]) -> None:
    _head("System Readiness Report")

    # API status
    print("  API Keys:")
    for api, ok in api_results.items():
        if ok is True:
            _ok(f"  {api}")
        elif ok is False:
            _warn(f"  {api} — not configured")
        else:
            _warn(f"  {api} — unverified")

    # Test status
    print("\n  Module Tests:")
    passed = sum(1 for v in test_results.values() if v)
    total  = len(test_results)
    for name, ok in test_results.items():
        if ok:
            _ok(f"  {name}")
        else:
            _fail(f"  {name}")

    # Capability matrix
    print("\n  Capability Matrix:")
    has_exa    = api_results.get("exa")
    has_llm    = api_results.get("openai") or api_results.get("anthropic") or api_results.get("ollama")
    has_playwright = False
    try:
        import playwright
        has_playwright = True
    except ImportError:
        pass

    _ok("  Web research (DuckDuckGo fallback) — ALWAYS available")
    if has_exa:
        _ok("  Web research (Exa neural search) — AVAILABLE")
    else:
        _warn("  Web research (Exa neural search) — needs EXA_API_KEY")
    if has_llm:
        _ok("  LLM analysis — AVAILABLE")
    else:
        _warn("  LLM analysis — fallback to rule-based (set OPENAI_API_KEY to enable)")
    if has_playwright:
        _ok("  Browser scraping (Playwright) — AVAILABLE")
    else:
        _warn("  Browser scraping — fallback to requests (run: playwright install chromium)")

    overall = "🟢 READY" if passed == total else f"🟡 PARTIAL ({passed}/{total} tests passed)"
    print(f"\n  Overall Status: {overall}")
    print(f"\n  Run: python tools/run.py\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="ARIA-LAB v2 — Environment Bootstrap & Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-install",    action="store_true", help="Skip pip install step")
    parser.add_argument("--check-only",      action="store_true", help="Only run checks, no installs")
    parser.add_argument("--skip-playwright", action="store_true", help="Skip Playwright browser install")
    parser.add_argument("--verbose",         action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    print("\n" + "═" * 55)
    print("  ARIA-LAB v2 — Bootstrap & System Validator")
    print("═" * 55)

    sys.path.insert(0, str(BASE_DIR / "tools"))

    skip_install = args.skip_install or args.check_only
    install_packages(skip=skip_install)
    create_directories()
    api_results  = validate_api_keys()
    init_knowledge_base()
    init_session()
    test_results = run_self_tests()
    if not args.check_only:
        install_playwright_browsers(skip=args.skip_playwright)

    print_summary(api_results, test_results)


if __name__ == "__main__":
    _cli()
