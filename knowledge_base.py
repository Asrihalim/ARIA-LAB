"""
tools/knowledge_base.py — Persistent Intelligence Layer for ARIA-LAB v2

Stores entities with full attribute graphs, supports full-text search (SQLite FTS5),
semantic similarity search, knowledge graphs, auto-expiration, session summaries,
context injection, and source quality tracking.

Usage:
    python tools/knowledge_base.py search "digital products"
    python tools/knowledge_base.py store --entity '{"type":"product","name":"X"}'
    python tools/knowledge_base.py context --task "find AI tools"
    python tools/knowledge_base.py export --out knowledge.json
    python tools/knowledge_base.py stats
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
KB_PATH  = BASE_DIR / "state" / "knowledge" / "knowledge.db"

# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

@dataclass
class Entity:
    entity_id:     str
    entity_type:   str                    # product | company | person | market | insight | session
    name:          str
    attributes:    dict[str, Any]        = field(default_factory=dict)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    tags:          list[str]            = field(default_factory=list)
    confidence:    float                 = 1.0
    source:        str                   = ""
    source_quality: float                = 0.5
    created:       str                   = ""
    updated:       str                   = ""
    expires_at:    Optional[str]         = None

    def __post_init__(self) -> None:
        now = _now()
        if not self.entity_id:
            self.entity_id = str(uuid.uuid4())
        if not self.created:
            self.created = now
        if not self.updated:
            self.updated = now

    def to_dict(self) -> dict:
        return asdict(self)

    def searchable_text(self) -> str:
        """Build a searchable text blob from key fields."""
        parts = [
            self.name,
            self.entity_type,
            self.source,
            " ".join(self.tags),
            " ".join(str(v) for v in self.attributes.values() if isinstance(v, (str, int, float))),
        ]
        return " ".join(p for p in parts if p)


@dataclass
class KnowledgeRelation:
    from_id:   str
    relation:  str              # "sells" | "competes_with" | "mentions" | "has_price"
    to_id:     str
    weight:    float = 1.0
    source:    str   = ""
    created:   str   = ""

    def __post_init__(self) -> None:
        if not self.created:
            self.created = _now()


@dataclass
class SourceQuality:
    url:            str
    quality_score:  float
    visits:         int
    useful_count:   int
    updated:        str

    @property
    def effective_score(self) -> float:
        if self.visits == 0:
            return self.quality_score
        ratio = self.useful_count / self.visits
        return (self.quality_score + ratio) / 2


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity (no numpy required)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _tfidf_vector(text: str, vocab: dict[str, int], idf: dict[str, float]) -> list[float]:
    """Compute a TF-IDF vector for text given a vocabulary and IDF table."""
    words = _tokenize(text)
    tf: dict[str, float] = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    total = len(words) or 1
    vec = [0.0] * len(vocab)
    for w, idx in vocab.items():
        if w in tf:
            vec[idx] = (tf[w] / total) * idf.get(w, 1.0)
    return vec


def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


# ──────────────────────────────────────────────
# KnowledgeBase
# ──────────────────────────────────────────────

class KnowledgeBase:
    """
    Persistent knowledge store backed by SQLite with FTS5 full-text search,
    lightweight TF-IDF semantic similarity, knowledge graph, and TTL expiration.
    """

    SCHEMA_VERSION = 2

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or KB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.debug("KnowledgeBase ready at %s", self.db_path)

    # ── DB Setup ───────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        con = sqlite3.connect(str(self.db_path), timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_db(self) -> None:
        with self._conn() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id     TEXT PRIMARY KEY,
                    entity_type   TEXT NOT NULL,
                    name          TEXT NOT NULL,
                    attributes    TEXT NOT NULL DEFAULT '{}',
                    relationships TEXT NOT NULL DEFAULT '[]',
                    tags          TEXT NOT NULL DEFAULT '[]',
                    confidence    REAL NOT NULL DEFAULT 1.0,
                    source        TEXT NOT NULL DEFAULT '',
                    source_quality REAL NOT NULL DEFAULT 0.5,
                    created       TEXT NOT NULL,
                    updated       TEXT NOT NULL,
                    expires_at    TEXT
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
                    entity_id UNINDEXED,
                    searchable_text,
                    content='entities',
                    content_rowid='rowid'
                );

                CREATE TABLE IF NOT EXISTS relations (
                    from_id   TEXT NOT NULL,
                    relation  TEXT NOT NULL,
                    to_id     TEXT NOT NULL,
                    weight    REAL NOT NULL DEFAULT 1.0,
                    source    TEXT NOT NULL DEFAULT '',
                    created   TEXT NOT NULL,
                    PRIMARY KEY (from_id, relation, to_id)
                );

                CREATE TABLE IF NOT EXISTS source_quality (
                    url           TEXT PRIMARY KEY,
                    quality_score REAL NOT NULL DEFAULT 0.5,
                    visits        INTEGER NOT NULL DEFAULT 0,
                    useful_count  INTEGER NOT NULL DEFAULT 0,
                    updated       TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS failed_approaches (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    task     TEXT NOT NULL,
                    approach TEXT NOT NULL,
                    reason   TEXT NOT NULL,
                    created  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary    TEXT NOT NULL,
                    created    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kb_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                INSERT OR IGNORE INTO kb_meta VALUES ('schema_version', '2');
            """)
            # FTS triggers
            con.executescript("""
                CREATE TRIGGER IF NOT EXISTS entities_fts_insert
                AFTER INSERT ON entities BEGIN
                    INSERT INTO entities_fts(entity_id, searchable_text)
                    VALUES (new.entity_id,
                        new.name || ' ' || new.entity_type || ' ' ||
                        new.source || ' ' || new.tags || ' ' || new.attributes);
                END;

                CREATE TRIGGER IF NOT EXISTS entities_fts_delete
                AFTER DELETE ON entities BEGIN
                    INSERT INTO entities_fts(entities_fts, entity_id, searchable_text)
                    VALUES('delete', old.entity_id,
                        old.name || ' ' || old.entity_type || ' ' ||
                        old.source || ' ' || old.tags || ' ' || old.attributes);
                END;

                CREATE TRIGGER IF NOT EXISTS entities_fts_update
                AFTER UPDATE ON entities BEGIN
                    INSERT INTO entities_fts(entities_fts, entity_id, searchable_text)
                    VALUES('delete', old.entity_id,
                        old.name || ' ' || old.entity_type || ' ' ||
                        old.source || ' ' || old.tags || ' ' || old.attributes);
                    INSERT INTO entities_fts(entity_id, searchable_text)
                    VALUES (new.entity_id,
                        new.name || ' ' || new.entity_type || ' ' ||
                        new.source || ' ' || new.tags || ' ' || new.attributes);
                END;
            """)

    # ── CRUD ───────────────────────────────────

    def store(self, entity: Entity, ttl_hours: Optional[float] = None) -> str:
        """Store or update an entity. Returns entity_id."""
        entity.updated = _now()
        if ttl_hours is not None:
            expires_s = time.time() + ttl_hours * 3600
            entity.expires_at = datetime.fromtimestamp(expires_s, tz=timezone.utc).isoformat()

        with self._conn() as con:
            con.execute("""
                INSERT INTO entities
                    (entity_id, entity_type, name, attributes, relationships, tags,
                     confidence, source, source_quality, created, updated, expires_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    name=excluded.name,
                    attributes=excluded.attributes,
                    relationships=excluded.relationships,
                    tags=excluded.tags,
                    confidence=excluded.confidence,
                    source=excluded.source,
                    source_quality=excluded.source_quality,
                    updated=excluded.updated,
                    expires_at=excluded.expires_at
            """, (
                entity.entity_id,
                entity.entity_type,
                entity.name,
                json.dumps(entity.attributes),
                json.dumps(entity.relationships),
                json.dumps(entity.tags),
                entity.confidence,
                entity.source,
                entity.source_quality,
                entity.created,
                entity.updated,
                entity.expires_at,
            ))
        logger.debug("Stored entity: %s (%s)", entity.name, entity.entity_id[:8])
        return entity.entity_id

    def get(self, entity_id: str) -> Optional[Entity]:
        with self._conn() as con:
            row = con.execute("SELECT * FROM entities WHERE entity_id = ?", (entity_id,)).fetchone()
        if not row:
            return None
        return self._row_to_entity(row)

    def delete(self, entity_id: str) -> bool:
        with self._conn() as con:
            cur = con.execute("DELETE FROM entities WHERE entity_id = ?", (entity_id,))
        return cur.rowcount > 0

    def store_batch(self, entities: list[Entity], ttl_hours: Optional[float] = None) -> list[str]:
        return [self.store(e, ttl_hours=ttl_hours) for e in entities]

    # ── Full-Text Search ───────────────────────

    def search(
        self,
        query: str,
        limit:       int             = 20,
        entity_type: Optional[str]   = None,
        since_hours: Optional[float] = None,
    ) -> list[Entity]:
        """Full-text search using FTS5. Supports BM25 ranking."""
        now = _now()
        params: list[Any] = [query, limit]
        type_filter = "AND e.entity_type = ?" if entity_type else ""
        time_filter = ""
        if entity_type:
            params.insert(1, entity_type)
        if since_hours is not None:
            cutoff = datetime.fromtimestamp(
                time.time() - since_hours * 3600, tz=timezone.utc
            ).isoformat()
            time_filter = "AND e.created >= ?"
            params.insert(-1, cutoff)

        sql = f"""
            SELECT e.*
            FROM entities e
            JOIN entities_fts fts ON e.entity_id = fts.entity_id
            WHERE entities_fts MATCH ?
              {type_filter}
              {time_filter}
              AND (e.expires_at IS NULL OR e.expires_at > '{now}')
            ORDER BY rank
            LIMIT ?
        """
        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [self._row_to_entity(r) for r in rows]

    # ── Semantic Search (TF-IDF) ───────────────

    def semantic_search(self, query: str, top_n: int = 10) -> list[tuple[float, Entity]]:
        """
        TF-IDF cosine similarity search across all entities.
        Falls back gracefully if sentence-transformers is unavailable.
        """
        try:
            return self._semantic_search_transformers(query, top_n)
        except Exception:
            return self._semantic_search_tfidf(query, top_n)

    def _semantic_search_transformers(self, query: str, top_n: int) -> list[tuple[float, Entity]]:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np  # type: ignore
        model = SentenceTransformer("all-MiniLM-L6-v2")
        entities = self.list_all(limit=2000)
        if not entities:
            return []
        texts = [e.searchable_text() for e in entities]
        q_emb = model.encode([query])[0]
        e_embs = model.encode(texts)
        scores = np.dot(e_embs, q_emb) / (np.linalg.norm(e_embs, axis=1) * np.linalg.norm(q_emb) + 1e-9)
        top = sorted(zip(scores.tolist(), entities), key=lambda x: x[0], reverse=True)
        return top[:top_n]

    def _semantic_search_tfidf(self, query: str, top_n: int) -> list[tuple[float, Entity]]:
        entities = self.list_all(limit=2000)
        if not entities:
            return []
        texts = [e.searchable_text() for e in entities]
        all_texts = texts + [query]
        # Build vocabulary
        vocab: dict[str, int] = {}
        doc_freq: dict[str, int] = {}
        tokenized = [_tokenize(t) for t in all_texts]
        for tokens in tokenized:
            unique = set(tokens)
            for w in unique:
                if w not in vocab:
                    vocab[w] = len(vocab)
                doc_freq[w] = doc_freq.get(w, 0) + 1
        N = len(all_texts)
        idf = {w: math.log(N / (df + 1)) for w, df in doc_freq.items()}
        q_vec = _tfidf_vector(query, vocab, idf)
        results: list[tuple[float, Entity]] = []
        for i, e in enumerate(entities):
            e_vec = _tfidf_vector(texts[i], vocab, idf)
            sim = _cosine_similarity(q_vec, e_vec)
            results.append((sim, e))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_n]

    # ── Listing & Filtering ────────────────────

    def list_all(
        self,
        entity_type: Optional[str] = None,
        limit:       int            = 1000,
        include_expired: bool      = False,
    ) -> list[Entity]:
        now = _now()
        type_filter = "AND entity_type = ?" if entity_type else ""
        exp_filter  = "" if include_expired else f"AND (expires_at IS NULL OR expires_at > '{now}')"
        params = [limit]
        if entity_type:
            params.insert(0, entity_type)
        sql = f"SELECT * FROM entities WHERE 1=1 {type_filter} {exp_filter} ORDER BY updated DESC LIMIT ?"
        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [self._row_to_entity(r) for r in rows]

    # ── Knowledge Graph ────────────────────────

    def add_relation(self, rel: KnowledgeRelation) -> None:
        with self._conn() as con:
            con.execute("""
                INSERT INTO relations (from_id, relation, to_id, weight, source, created)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(from_id, relation, to_id) DO UPDATE SET
                    weight=excluded.weight, source=excluded.source
            """, (rel.from_id, rel.relation, rel.to_id, rel.weight, rel.source, rel.created))

    def get_relations(self, entity_id: str) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM relations WHERE from_id = ? OR to_id = ?",
                (entity_id, entity_id)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_graph_neighbors(self, entity_id: str, depth: int = 2) -> list[Entity]:
        """BFS traversal of the knowledge graph up to `depth` hops."""
        visited: set[str] = {entity_id}
        frontier = [entity_id]
        results: list[Entity] = []
        for _ in range(depth):
            next_frontier: list[str] = []
            with self._conn() as con:
                for fid in frontier:
                    rows = con.execute(
                        "SELECT from_id, to_id FROM relations WHERE from_id=? OR to_id=?",
                        (fid, fid)
                    ).fetchall()
                    for r in rows:
                        for eid in (r["from_id"], r["to_id"]):
                            if eid not in visited:
                                visited.add(eid)
                                next_frontier.append(eid)
            for eid in next_frontier:
                e = self.get(eid)
                if e:
                    results.append(e)
            frontier = next_frontier
        return results

    # ── Source Quality ─────────────────────────

    def record_source_visit(self, url: str, was_useful: bool = True) -> None:
        with self._conn() as con:
            con.execute("""
                INSERT INTO source_quality (url, quality_score, visits, useful_count, updated)
                VALUES (?, 0.5, 1, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    visits=visits+1,
                    useful_count=useful_count + ?,
                    updated=excluded.updated
            """, (url, int(was_useful), _now(), int(was_useful)))

    def get_source_quality(self, url: str) -> float:
        with self._conn() as con:
            row = con.execute("SELECT * FROM source_quality WHERE url = ?", (url,)).fetchone()
        if not row:
            return 0.5
        sq = SourceQuality(**dict(row))
        return sq.effective_score

    def top_sources(self, n: int = 10) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM source_quality ORDER BY quality_score DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Failed Approaches ──────────────────────

    def log_failed_approach(self, task: str, approach: str, reason: str) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT INTO failed_approaches (task, approach, reason, created) VALUES (?,?,?,?)",
                (task, approach, reason, _now()),
            )
        logger.debug("Logged failed approach: %s → %s", approach, reason)

    def get_failed_approaches(self, task_pattern: str = "") -> list[dict]:
        if task_pattern:
            sql    = "SELECT * FROM failed_approaches WHERE task LIKE ? ORDER BY created DESC LIMIT 50"
            params = (f"%{task_pattern}%",)
        else:
            sql    = "SELECT * FROM failed_approaches ORDER BY created DESC LIMIT 50"
            params = ()
        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── Session Summaries ──────────────────────

    def save_session_summary(self, session_id: str, summary: str) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO session_summaries (session_id, summary, created) VALUES (?,?,?)",
                (session_id, summary, _now()),
            )

    def get_session_summary(self, session_id: str) -> Optional[str]:
        with self._conn() as con:
            row = con.execute(
                "SELECT summary FROM session_summaries WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row["summary"] if row else None

    # ── Context Injection ──────────────────────

    def get_context_for_task(self, task_text: str, max_entities: int = 10) -> str:
        """
        Retrieve the top-N most relevant past learnings for a new task.
        Returns a formatted context string ready for injection into an LLM prompt.
        """
        results = self.semantic_search(task_text, top_n=max_entities)
        if not results:
            return "No relevant prior knowledge found."

        lines = ["## Relevant Prior Knowledge\n"]
        for score, entity in results:
            if score < 0.05:
                continue
            lines.append(f"### {entity.name} ({entity.entity_type}) — confidence {entity.confidence:.0%}")
            lines.append(f"Source: {entity.source or 'unknown'}")
            for k, v in list(entity.attributes.items())[:5]:
                lines.append(f"- {k}: {v}")
            lines.append("")
        return "\n".join(lines)

    # ── TTL Expiration ─────────────────────────

    def purge_expired(self) -> int:
        now = _now()
        with self._conn() as con:
            cur = con.execute("DELETE FROM entities WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
        deleted = cur.rowcount
        if deleted:
            logger.info("Purged %d expired entities", deleted)
        return deleted

    # ── Export / Import ───────────────────────

    def export_json(self, path: Path) -> None:
        entities = self.list_all(limit=100_000, include_expired=True)
        data = [e.to_dict() for e in entities]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Exported %d entities to %s", len(data), path)

    def import_json(self, path: Path) -> int:
        raw = json.loads(path.read_text(encoding="utf-8"))
        count = 0
        for item in raw:
            entity = Entity(
                entity_id    = item.get("entity_id", ""),
                entity_type  = item.get("entity_type", "unknown"),
                name         = item.get("name", ""),
                attributes   = item.get("attributes", {}),
                relationships= item.get("relationships", []),
                tags         = item.get("tags", []),
                confidence   = item.get("confidence", 1.0),
                source       = item.get("source", ""),
                source_quality=item.get("source_quality", 0.5),
                created      = item.get("created", _now()),
                updated      = item.get("updated", _now()),
                expires_at   = item.get("expires_at"),
            )
            self.store(entity)
            count += 1
        logger.info("Imported %d entities from %s", count, path)
        return count

    # ── Stats ──────────────────────────────────

    def stats(self) -> dict:
        with self._conn() as con:
            total   = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            expired = con.execute(
                "SELECT COUNT(*) FROM entities WHERE expires_at IS NOT NULL AND expires_at <= ?", (_now(),)
            ).fetchone()[0]
            by_type = con.execute(
                "SELECT entity_type, COUNT(*) as n FROM entities GROUP BY entity_type"
            ).fetchall()
            relations = con.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
            sources = con.execute("SELECT COUNT(*) FROM source_quality").fetchone()[0]
            failed = con.execute("SELECT COUNT(*) FROM failed_approaches").fetchone()[0]
        return {
            "total_entities":  total,
            "expired_entities": expired,
            "by_type":          {r["entity_type"]: r["n"] for r in by_type},
            "total_relations":  relations,
            "tracked_sources":  sources,
            "failed_approaches": failed,
            "db_size_kb":       self.db_path.stat().st_size // 1024 if self.db_path.exists() else 0,
        }

    # ── Helpers ────────────────────────────────

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        def _safe_json(val: Any, default: Any) -> Any:
            if isinstance(val, (dict, list)):
                return val
            try:
                return json.loads(val) if val else default
            except (json.JSONDecodeError, TypeError):
                return default

        return Entity(
            entity_id    = row["entity_id"],
            entity_type  = row["entity_type"],
            name         = row["name"],
            attributes   = _safe_json(row["attributes"], {}),
            relationships= _safe_json(row["relationships"], []),
            tags         = _safe_json(row["tags"], []),
            confidence   = float(row["confidence"]),
            source       = row["source"] or "",
            source_quality= float(row["source_quality"]),
            created      = row["created"],
            updated      = row["updated"],
            expires_at   = row["expires_at"],
        )


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="ARIA-LAB Knowledge Base — persistent intelligence layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Commands:
  search   <query>        Full-text search
  semantic <query>        Semantic similarity search
  store    --entity JSON  Store an entity
  get      <id>           Retrieve entity by ID
  context  --task <text>  Get relevant past knowledge for a task
  stats                   Show knowledge base statistics
  export   --out <file>   Export all entities to JSON
  import   --in  <file>   Import entities from JSON
  purge                   Delete expired entities
  sources                 Show top sources by quality
  failed   [pattern]      Show failed approaches
""",
    )
    parser.add_argument("command", choices=["search","semantic","store","get","context","stats",
                                             "export","import","purge","sources","failed"])
    parser.add_argument("query",   nargs="?", help="Search query or entity ID")
    parser.add_argument("--entity",  type=str,  help="JSON entity to store")
    parser.add_argument("--task",    type=str,  help="Task text for context injection")
    parser.add_argument("--out",     type=Path, help="Output file for export")
    parser.add_argument("--in",      dest="infile", type=Path, help="Input file for import")
    parser.add_argument("--limit",   type=int,  default=20, help="Max results")
    parser.add_argument("--type",    type=str,  help="Filter by entity type")
    parser.add_argument("--hours",   type=float, help="Filter to last N hours")
    parser.add_argument("--db",      type=Path, help="Custom DB path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    kb = KnowledgeBase(db_path=args.db)

    if args.command == "search":
        if not args.query:
            print("Query required for search", file=sys.stderr)
            sys.exit(1)
        results = kb.search(args.query, limit=args.limit, entity_type=args.type, since_hours=args.hours)
        print(f"Found {len(results)} results for '{args.query}':\n")
        for e in results:
            print(f"  [{e.entity_type}] {e.name}  (confidence={e.confidence:.0%})")
            print(f"    Source: {e.source}")
            for k, v in list(e.attributes.items())[:3]:
                print(f"    {k}: {v}")
            print()

    elif args.command == "semantic":
        if not args.query:
            print("Query required", file=sys.stderr)
            sys.exit(1)
        results = kb.semantic_search(args.query, top_n=args.limit)
        print(f"Semantic search for '{args.query}':\n")
        for score, e in results:
            if score < 0.01:
                continue
            print(f"  [{score:.3f}] [{e.entity_type}] {e.name}")

    elif args.command == "store":
        if not args.entity:
            print("--entity JSON required", file=sys.stderr)
            sys.exit(1)
        data = json.loads(args.entity)
        e = Entity(
            entity_id   = data.get("entity_id", str(uuid.uuid4())),
            entity_type = data.get("type", data.get("entity_type", "unknown")),
            name        = data.get("name", ""),
            attributes  = data.get("attributes", {}),
            tags        = data.get("tags", []),
            confidence  = data.get("confidence", 1.0),
            source      = data.get("source", ""),
        )
        eid = kb.store(e)
        print(f"Stored entity: {eid}")

    elif args.command == "get":
        if not args.query:
            print("Entity ID required", file=sys.stderr)
            sys.exit(1)
        e = kb.get(args.query)
        if e:
            print(json.dumps(e.to_dict(), indent=2))
        else:
            print(f"Entity not found: {args.query}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "context":
        task_text = args.task or args.query or ""
        if not task_text:
            print("--task <text> required", file=sys.stderr)
            sys.exit(1)
        print(kb.get_context_for_task(task_text))

    elif args.command == "stats":
        s = kb.stats()
        print(json.dumps(s, indent=2))

    elif args.command == "export":
        out = args.out or Path("knowledge_export.json")
        kb.export_json(out)
        print(f"Exported to {out}")

    elif args.command == "import":
        if not args.infile:
            print("--in <file> required", file=sys.stderr)
            sys.exit(1)
        n = kb.import_json(args.infile)
        print(f"Imported {n} entities")

    elif args.command == "purge":
        n = kb.purge_expired()
        print(f"Purged {n} expired entities")

    elif args.command == "sources":
        sources = kb.top_sources(n=args.limit)
        for s in sources:
            print(f"  [{s['quality_score']:.2f}] {s['url']} (visits={s['visits']})")

    elif args.command == "failed":
        pattern = args.query or ""
        failures = kb.get_failed_approaches(task_pattern=pattern)
        for f in failures:
            print(f"  [{f['created'][:10]}] {f['approach']} → {f['reason']}")


if __name__ == "__main__":
    _cli()
