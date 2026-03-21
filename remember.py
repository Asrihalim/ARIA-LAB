"""
tools/remember.py — Session State Manager for ARIA-LAB v2

Manages state/session.json and interfaces with knowledge_base.py for long-term storage.
Supports read/write/update/reset/summarize/inject/log operations.

Usage:
    python tools/remember.py write key value
    python tools/remember.py read [key]
    python tools/remember.py log "note"
    python tools/remember.py context
    python tools/remember.py inject --task "research AI tools"
    python tools/remember.py reset
    python tools/remember.py summarize
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).parent.parent
STATE_FILE   = BASE_DIR / "state" / "session.json"

# ──────────────────────────────────────────────
# Default State Schema
# ──────────────────────────────────────────────

def _empty_state() -> dict:
    return {
        "session_id": str(uuid.uuid4()),
        "created":    _now(),
        "updated":    _now(),
        "task": {
            "target":      "",
            "goal":        "",
            "context":     "",
            "deliverables": [],
        },
        "progress": {
            "completed_steps": [],
            "current_step":    "",
            "failed_steps":    [],
        },
        "findings": {
            "entities": [],
            "sources":  [],
            "insights": [],
        },
        "outputs": {
            "files_written":    [],
            "schemas_validated": [],
        },
        "learnings": [],
        "log": [],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────
# StateManager
# ──────────────────────────────────────────────

class StateManager:
    """
    Manages the current session state (state/session.json).
    Provides atomic read-modify-write with auto-backup.
    """

    def __init__(self, state_file: Optional[Path] = None) -> None:
        self.state_file = state_file or STATE_FILE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state: Optional[dict] = None

    # ── I/O ────────────────────────────────────

    def load(self) -> dict:
        """Load state from disk, creating if absent."""
        if self.state_file.exists():
            try:
                raw = self.state_file.read_text(encoding="utf-8")
                self._state = json.loads(raw)
                # Ensure required keys exist (schema evolution)
                default = _empty_state()
                for key, val in default.items():
                    if key not in self._state:
                        self._state[key] = val
                return self._state
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load state (%s), resetting.", e)
        self._state = _empty_state()
        self.save()
        return self._state

    def save(self) -> None:
        """Write state to disk atomically."""
        if self._state is None:
            return
        self._state["updated"] = _now()
        tmp = self.state_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._state, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.state_file)
        except OSError as e:
            logger.error("Failed to save state: %s", e)
            raise

    def reset(self) -> dict:
        """Reset session to empty state."""
        self._state = _empty_state()
        self.save()
        logger.info("Session state reset.")
        return self._state

    # ── Read / Write ───────────────────────────

    def read(self, key: Optional[str] = None) -> Any:
        """Read entire state or a specific top-level key."""
        state = self.load()
        if key is None:
            return state
        return _nested_get(state, key)

    def write(self, key: str, value: Any) -> None:
        """Write a value at the given dot-path key."""
        state = self.load()
        _nested_set(state, key, value)
        self.save()

    def update(self, data: dict) -> None:
        """Shallow-merge data into the state root."""
        state = self.load()
        _deep_merge(state, data)
        self.save()

    # ── Task ───────────────────────────────────

    def set_task(self, target: str, goal: str, context: str = "", deliverables: Optional[list] = None) -> None:
        state = self.load()
        state["task"]["target"]       = target
        state["task"]["goal"]         = goal
        state["task"]["context"]      = context
        state["task"]["deliverables"] = deliverables or []
        self.save()

    # ── Progress ───────────────────────────────

    def start_step(self, step: str) -> None:
        state = self.load()
        state["progress"]["current_step"] = step
        self.save()

    def complete_step(self, step: str) -> None:
        state = self.load()
        if step not in state["progress"]["completed_steps"]:
            state["progress"]["completed_steps"].append(step)
        state["progress"]["current_step"] = ""
        self.save()

    def fail_step(self, step: str, reason: str = "") -> None:
        state = self.load()
        entry = {"step": step, "reason": reason, "time": _now()}
        state["progress"]["failed_steps"].append(entry)
        state["progress"]["current_step"] = ""
        self.save()

    # ── Findings ───────────────────────────────

    def add_entity(self, entity: dict) -> None:
        state = self.load()
        state["findings"]["entities"].append(entity)
        self.save()

    def add_source(self, url: str, title: str = "", quality: float = 0.5) -> None:
        state = self.load()
        source = {"url": url, "title": title, "quality": quality, "added": _now()}
        if not any(s["url"] == url for s in state["findings"]["sources"]):
            state["findings"]["sources"].append(source)
            self.save()

    def add_insight(self, insight: str) -> None:
        state = self.load()
        state["findings"]["insights"].append({"text": insight, "added": _now()})
        self.save()

    # ── Outputs ────────────────────────────────

    def record_output(self, filepath: str, schema_validated: bool = False) -> None:
        state = self.load()
        if filepath not in state["outputs"]["files_written"]:
            state["outputs"]["files_written"].append(filepath)
        if schema_validated and filepath not in state["outputs"]["schemas_validated"]:
            state["outputs"]["schemas_validated"].append(filepath)
        self.save()

    # ── Log ────────────────────────────────────

    def log(self, note: str) -> None:
        state = self.load()
        state["log"].append({"note": note, "time": _now()})
        # Keep last 200 log entries
        state["log"] = state["log"][-200:]
        self.save()

    # ── Learnings ──────────────────────────────

    def add_learning(self, learning: str, source: str = "") -> None:
        state = self.load()
        state["learnings"].append({"text": learning, "source": source, "time": _now()})
        self.save()

    # ── Summarize ──────────────────────────────

    def summarize(self) -> str:
        """Generate a human-readable session summary."""
        state = self.load()
        task = state.get("task", {})
        progress = state.get("progress", {})
        findings = state.get("findings", {})
        outputs  = state.get("outputs", {})
        learnings = state.get("learnings", [])

        lines = [
            f"Session: {state.get('session_id', 'N/A')[:8]}",
            f"Created: {state.get('created', 'N/A')[:19]}",
            f"Updated: {state.get('updated', 'N/A')[:19]}",
            "",
            f"Task:",
            f"  Target : {task.get('target', '(none)')}",
            f"  Goal   : {task.get('goal', '(none)')}",
            "",
            f"Progress:",
            f"  Completed steps : {len(progress.get('completed_steps', []))}",
            f"  Failed steps    : {len(progress.get('failed_steps', []))}",
            f"  Current         : {progress.get('current_step', '(idle)')}",
            "",
            f"Findings:",
            f"  Entities  : {len(findings.get('entities', []))}",
            f"  Sources   : {len(findings.get('sources', []))}",
            f"  Insights  : {len(findings.get('insights', []))}",
            "",
            f"Outputs: {len(outputs.get('files_written', []))} files",
        ]
        for f_ in outputs.get("files_written", []):
            lines.append(f"  • {f_}")
        if learnings:
            lines.append("")
            lines.append(f"Learnings ({len(learnings)}):")
            for l in learnings[-5:]:
                lines.append(f"  • {l['text'][:80]}")

        return "\n".join(lines)

    # ── Context Injection ──────────────────────

    def inject_context(self, task_text: str = "") -> str:
        """
        Retrieve relevant past knowledge from knowledge_base and current session state.
        Returns a formatted context string for LLM prompts.
        """
        # Try knowledge base first
        context_parts: list[str] = []
        try:
            from knowledge_base import KnowledgeBase
            kb = KnowledgeBase()
            kb_context = kb.get_context_for_task(task_text or self.read("task.goal") or "")
            if kb_context and "No relevant" not in kb_context:
                context_parts.append(kb_context)
        except Exception as e:
            logger.debug("Knowledge base unavailable for context: %s", e)

        # Add current session learnings
        state = self.load()
        learnings = state.get("learnings", [])
        if learnings:
            lines = ["## Session Learnings\n"]
            for l in learnings[-10:]:
                lines.append(f"- {l['text']}")
            context_parts.append("\n".join(lines))

        # Add failed approaches
        failed = state["progress"].get("failed_steps", [])
        if failed:
            lines = ["## Approaches That Failed\n"]
            for f in failed[-5:]:
                lines.append(f"- {f['step']}: {f.get('reason', 'unknown reason')}")
            context_parts.append("\n".join(lines))

        return "\n\n".join(context_parts) if context_parts else "No prior context available."


# ──────────────────────────────────────────────
# Nested Key Helpers
# ──────────────────────────────────────────────

def _nested_get(d: dict, path: str) -> Any:
    """Get value at dot-path like 'task.goal'."""
    keys = path.split(".")
    cur: Any = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, list):
            try:
                cur = cur[int(k)]
            except (IndexError, ValueError):
                return None
        else:
            return None
    return cur


def _nested_set(d: dict, path: str, value: Any) -> None:
    """Set value at dot-path like 'task.goal'."""
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="ARIA-LAB Session State Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Commands:
  write  <key> <value>   Write a value (dot-path supported: task.goal)
  read   [key]           Read state (or a specific key)
  log    <note>          Append a note to the session log
  context                Show current session summary
  inject [--task TEXT]   Get relevant past knowledge for a task
  reset                  Reset session state
  summarize              Generate session summary
""",
    )
    parser.add_argument("command", choices=["write","read","log","context","inject","reset","summarize","update"])
    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument("--task",    type=str, help="Task text for context injection")
    parser.add_argument("--state",   type=Path, help="Custom state file path")
    parser.add_argument("--verbose", action="store_true")
    cli_args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if cli_args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    sm = StateManager(state_file=cli_args.state)

    if cli_args.command == "write":
        if len(cli_args.args) < 2:
            print("Usage: remember.py write <key> <value>", file=sys.stderr)
            sys.exit(1)
        key   = cli_args.args[0]
        value_raw = " ".join(cli_args.args[1:])
        # Try to parse as JSON, fall back to string
        try:
            value = json.loads(value_raw)
        except (json.JSONDecodeError, ValueError):
            value = value_raw
        sm.write(key, value)
        print(f"Written: {key} = {value!r}")

    elif cli_args.command == "read":
        key = cli_args.args[0] if cli_args.args else None
        val = sm.read(key)
        print(json.dumps(val, indent=2, ensure_ascii=False))

    elif cli_args.command == "log":
        note = " ".join(cli_args.args) if cli_args.args else ""
        if not note:
            print("Provide a note to log", file=sys.stderr)
            sys.exit(1)
        sm.log(note)
        print(f"Logged: {note}")

    elif cli_args.command in ("context", "summarize"):
        print(sm.summarize())

    elif cli_args.command == "inject":
        task_text = cli_args.task or (" ".join(cli_args.args) if cli_args.args else "")
        print(sm.inject_context(task_text))

    elif cli_args.command == "reset":
        sm.reset()
        print("Session state reset.")

    elif cli_args.command == "update":
        if not cli_args.args:
            print("Usage: remember.py update '{\"key\": \"value\"}'", file=sys.stderr)
            sys.exit(1)
        data = json.loads(" ".join(cli_args.args))
        sm.update(data)
        print("State updated.")


if __name__ == "__main__":
    _cli()
