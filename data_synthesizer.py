"""
tools/data_synthesizer.py — Multi-Source Fusion Engine for ARIA-LAB v2

Accepts arrays of raw data from multiple sources and produces a clean,
unified, enriched dataset with:
- Entity resolution (fuzzy name + URL + attribute matching)
- Conflict resolution (weighted by source quality scores)
- Data enrichment (cross-reference fill missing fields)
- Deduplication (exact and near-duplicate removal)
- Confidence scoring per field (0.0–1.0)
- Source attribution per field
- Statistical aggregation (averages, medians, ranges)

Usage:
    python tools/data_synthesizer.py --input a.json b.json --out fused.json
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


# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

@dataclass
class FieldValue:
    """A field value with confidence and source attribution."""
    value:      Any
    confidence: float
    source:     str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SynthesizedRecord:
    """An enriched, unified record after multi-source fusion."""
    record_id:   str
    name:        str
    entity_type: str
    fields:      dict[str, FieldValue]  = field(default_factory=dict)
    sources:     list[str]             = field(default_factory=list)
    tags:        list[str]             = field(default_factory=list)
    confidence:  float                 = 1.0
    duplicates_merged: int             = 0

    def get(self, key: str, default: Any = None) -> Any:
        fv = self.fields.get(key)
        return fv.value if fv else default

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "name":      self.name,
            "entity_type": self.entity_type,
            "fields":    {k: v.to_dict() for k, v in self.fields.items()},
            "sources":   self.sources,
            "tags":      self.tags,
            "confidence": self.confidence,
            "duplicates_merged": self.duplicates_merged,
        }


@dataclass
class SynthesisStats:
    """Statistics about the synthesis run."""
    input_count:     int
    output_count:    int
    duplicates:      int
    conflicts:       int
    enrichments:     int
    avg_confidence:  float
    sources_used:    list[str]

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────
# Similarity Helpers
# ──────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Normalize a string for comparison: lowercase, strip punctuation."""
    return re.sub(r"[^a-z0-9\s]", "", s.lower()).strip()


def _token_similarity(a: str, b: str) -> float:
    """Jaccard similarity on token sets."""
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _url_similarity(a: str, b: str) -> float:
    """Compare URL paths for similarity."""
    if not a or not b:
        return 0.0
    try:
        import urllib.parse
        pa = urllib.parse.urlparse(a)
        pb = urllib.parse.urlparse(b)
        if pa.netloc == pb.netloc and pa.path == pb.path:
            return 1.0
        if pa.netloc == pb.netloc:
            return 0.5
    except Exception:
        pass
    return 0.0


def _attribute_similarity(a: dict, b: dict) -> float:
    """Compare overlapping numeric/string attributes."""
    common_keys = set(a.keys()) & set(b.keys())
    if not common_keys:
        return 0.0
    matches = 0
    for k in common_keys:
        av, bv = a.get(k), b.get(k)
        if av is None or bv is None:
            continue
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            if max(av, bv) > 0 and abs(av - bv) / max(av, bv) < 0.1:
                matches += 1
        elif isinstance(av, str) and isinstance(bv, str):
            if _token_similarity(av, bv) > 0.7:
                matches += 1
    return matches / max(len(common_keys), 1)


def _entity_similarity(a: dict, b: dict) -> float:
    """Composite similarity score for entity resolution."""
    name_sim = _token_similarity(str(a.get("name","") or a.get("title","")),
                                  str(b.get("name","") or b.get("title","")))
    url_sim  = _url_similarity(str(a.get("url","")), str(b.get("url","")))
    attr_sim = _attribute_similarity(a, b)
    return 0.5 * name_sim + 0.3 * url_sim + 0.2 * attr_sim


# ──────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────

def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]


def _stats_for(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "count":  len(values),
        "mean":   sum(values) / len(values),
        "median": _median(values),
        "min":    min(values),
        "max":    max(values),
        "range":  max(values) - min(values),
        "stdev":  math.sqrt(sum((x - sum(values)/len(values))**2 for x in values) / len(values)),
    }


# ──────────────────────────────────────────────
# DataSynthesizer
# ──────────────────────────────────────────────

class DataSynthesizer:
    """
    Fuses multiple raw datasets into a clean, unified, enriched dataset.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        source_quality:       Optional[dict[str, float]] = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.source_quality = source_quality or {}  # source_name → quality 0.0–1.0

    def synthesize(
        self,
        datasets: list[tuple[str, list[dict]]],  # [(source_name, records), ...]
        entity_type: str = "product",
    ) -> tuple[list[SynthesizedRecord], SynthesisStats]:
        """
        Full synthesis pipeline:
        1. Flatten + tag records with source
        2. Entity resolution (group duplicates)
        3. Merge each group into one SynthesizedRecord
        4. Enrich (fill missing fields from siblings)
        5. Compute statistics
        """
        # 1. Flatten
        all_records: list[dict] = []
        for source, records in datasets:
            quality = self.source_quality.get(source, 0.5)
            for rec in records:
                tagged = dict(rec)
                tagged["_source"]  = source
                tagged["_quality"] = quality
                all_records.append(tagged)

        logger.info("Synthesizing %d records from %d sources", len(all_records), len(datasets))

        # 2. Entity resolution
        groups = self._resolve_entities(all_records)
        conflicts = 0
        enrichments = 0

        # 3. Merge each group
        synthesized: list[SynthesizedRecord] = []
        for group in groups:
            record, c, e = self._merge_group(group, entity_type)
            conflicts  += c
            enrichments += e
            synthesized.append(record)

        # 4. Final deduplication (exact name match)
        synthesized = self._final_dedup(synthesized)

        # 5. Stats
        confidences = [r.confidence for r in synthesized]
        all_sources = list({s for _, recs in datasets for s in [s]})
        stats = SynthesisStats(
            input_count    = len(all_records),
            output_count   = len(synthesized),
            duplicates     = len(all_records) - len(synthesized),
            conflicts      = conflicts,
            enrichments    = enrichments,
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0,
            sources_used   = [s for s, _ in datasets],
        )
        logger.info("Synthesis complete: %d → %d records (%d conflicts, %d enrichments)",
                    stats.input_count, stats.output_count, conflicts, enrichments)
        return synthesized, stats

    def _resolve_entities(self, records: list[dict]) -> list[list[dict]]:
        """
        Group records that describe the same entity using
        fuzzy name + URL + attribute similarity.
        Uses a greedy O(n²) approach (acceptable for ≤ 10k records).
        """
        used  = [False] * len(records)
        groups: list[list[dict]] = []

        for i, a in enumerate(records):
            if used[i]:
                continue
            group = [a]
            used[i] = True
            for j, b in enumerate(records):
                if used[j]:
                    continue
                sim = _entity_similarity(a, b)
                if sim >= self.similarity_threshold:
                    group.append(b)
                    used[j] = True
            groups.append(group)

        return groups

    def _merge_group(
        self,
        group: list[dict],
        entity_type: str,
    ) -> tuple[SynthesizedRecord, int, int]:
        """Merge a group of duplicate records into one SynthesizedRecord."""
        import uuid

        # Sort by source quality (descending)
        group.sort(key=lambda r: r.get("_quality", 0.5), reverse=True)
        primary  = group[0]
        sources  = list({r.get("_source","unknown") for r in group})
        conflicts = 0
        enrichments = 0

        # Choose name from highest-quality source
        name = str(primary.get("name") or primary.get("title") or "")

        # Build field map
        fields: dict[str, FieldValue] = {}
        numeric_fields: dict[str, list[float]] = {}

        # Collect all field values across sources
        field_candidates: dict[str, list[tuple[Any, float, str]]] = {}  # key → [(val, quality, source)]
        for rec in group:
            source  = rec.get("_source", "unknown")
            quality = float(rec.get("_quality", 0.5))
            for k, v in rec.items():
                if k.startswith("_") or v is None or v == "" or v == [] or v == {}:
                    continue
                field_candidates.setdefault(k, []).append((v, quality, source))

        # Resolve each field
        for k, candidates in field_candidates.items():
            if len(candidates) == 1:
                v, q, s = candidates[0]
                fields[k] = FieldValue(value=v, confidence=q, source=s)
                continue

            # Multiple sources — resolve conflict
            numeric_vals = [(v, q, s) for v, q, s in candidates
                            if isinstance(v, (int, float))]
            if numeric_vals:
                # Weighted average by source quality
                total_weight = sum(q for _, q, _ in numeric_vals)
                if total_weight > 0:
                    avg = sum(v * q for v, q, _ in numeric_vals) / total_weight
                else:
                    avg = sum(v for v, _, _ in numeric_vals) / len(numeric_vals)
                confidence = min(1.0, sum(q for _, q, _ in numeric_vals) / len(numeric_vals) + 0.1)
                fields[k] = FieldValue(
                    value=round(avg, 4),
                    confidence=confidence,
                    source=",".join(s for _, _, s in numeric_vals),
                )
                numeric_fields.setdefault(k, []).extend(v for v, _, _ in numeric_vals)
                conflicts += 1
            else:
                # String fields — use highest-quality source value
                best = max(candidates, key=lambda x: x[1])
                v, q, s = best
                # Confidence: if all sources agree, higher confidence
                all_vals = [c[0] for c in candidates]
                agreement = sum(1 for av in all_vals if _token_similarity(str(av), str(v)) > 0.8)
                confidence = min(1.0, q + 0.05 * agreement)
                fields[k] = FieldValue(value=v, confidence=confidence, source=s)
                if len(set(str(c[0]) for c in candidates)) > 1:
                    conflicts += 1

        # Enrichment: fill missing fields from secondary sources
        primary_keys = set(primary.keys()) - {"_source", "_quality"}
        for rec in group[1:]:
            for k, v in rec.items():
                if k.startswith("_"):
                    continue
                if k not in primary_keys and k not in fields and v:
                    quality = rec.get("_quality", 0.5)
                    fields[k] = FieldValue(value=v, confidence=quality * 0.7, source=rec.get("_source",""))
                    enrichments += 1

        # Overall record confidence
        if fields:
            record_confidence = sum(fv.confidence for fv in fields.values()) / len(fields)
        else:
            record_confidence = primary.get("_quality", 0.5)

        tags = []
        if "tags" in fields:
            raw_tags = fields["tags"].value
            if isinstance(raw_tags, list):
                tags = raw_tags
            elif isinstance(raw_tags, str):
                tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

        record = SynthesizedRecord(
            record_id   = str(uuid.uuid4()),
            name        = name,
            entity_type = entity_type,
            fields      = fields,
            sources     = sources,
            tags        = tags,
            confidence  = round(record_confidence, 4),
            duplicates_merged = len(group) - 1,
        )
        return record, conflicts, enrichments

    def _final_dedup(self, records: list[SynthesizedRecord]) -> list[SynthesizedRecord]:
        """Remove exactly duplicate names (case-insensitive)."""
        seen: dict[str, SynthesizedRecord] = {}
        for rec in records:
            key = _normalize(rec.name)
            if key and key not in seen:
                seen[key] = rec
            elif key in seen:
                # Merge: keep higher confidence
                if rec.confidence > seen[key].confidence:
                    seen[key] = rec
        return list(seen.values()) if seen else records

    def compute_statistics(
        self,
        records: list[SynthesizedRecord],
        numeric_fields: Optional[list[str]] = None,
    ) -> dict:
        """Compute aggregate statistics across the synthesized dataset."""
        if not numeric_fields:
            # Auto-detect numeric fields
            numeric_fields = []
            for rec in records:
                for k, fv in rec.fields.items():
                    if isinstance(fv.value, (int, float)) and k not in numeric_fields:
                        numeric_fields.append(k)

        stats: dict[str, Any] = {"total_records": len(records)}
        for field_name in numeric_fields:
            values = [
                rec.fields[field_name].value
                for rec in records
                if field_name in rec.fields and isinstance(rec.fields[field_name].value, (int, float))
            ]
            if values:
                stats[field_name] = _stats_for([float(v) for v in values])

        # Source distribution
        source_counts: dict[str, int] = {}
        for rec in records:
            for s in rec.sources:
                source_counts[s] = source_counts.get(s, 0) + 1
        stats["source_distribution"] = source_counts

        # Confidence distribution
        confs = [rec.confidence for rec in records]
        if confs:
            stats["confidence"] = _stats_for(confs)

        return stats


# ──────────────────────────────────────────────
# Convenience Functions
# ──────────────────────────────────────────────

def synthesize_from_files(
    paths: list[Path],
    source_names: Optional[list[str]] = None,
    entity_type: str = "product",
    similarity_threshold: float = 0.65,
) -> tuple[list[SynthesizedRecord], SynthesisStats]:
    """Load JSON files and synthesize them."""
    datasets: list[tuple[str, list[dict]]] = []
    for i, path in enumerate(paths):
        source = source_names[i] if source_names and i < len(source_names) else path.stem
        records = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = [records]
        datasets.append((source, records))
    synth = DataSynthesizer(similarity_threshold=similarity_threshold)
    return synth.synthesize(datasets, entity_type=entity_type)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="ARIA-LAB Data Synthesizer — Multi-source fusion engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python tools/data_synthesizer.py --input a.json b.json --out fused.json
  python tools/data_synthesizer.py --input *.json --entity product --threshold 0.7
""",
    )
    parser.add_argument("--input",     nargs="+", type=Path, required=True, help="Input JSON files")
    parser.add_argument("--out",       type=Path, default=Path("working/synthesized.json"), help="Output file")
    parser.add_argument("--entity",    type=str,  default="product",  help="Entity type label")
    parser.add_argument("--threshold", type=float, default=0.65,      help="Similarity threshold for dedup")
    parser.add_argument("--stats",     action="store_true",           help="Print aggregate statistics")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    records, stats = synthesize_from_files(
        paths=args.input,
        entity_type=args.entity,
        similarity_threshold=args.threshold,
    )

    out_data = [r.to_dict() for r in records]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSynthesis complete:")
    print(f"  Input records  : {stats.input_count}")
    print(f"  Output records : {stats.output_count}")
    print(f"  Duplicates     : {stats.duplicates}")
    print(f"  Conflicts      : {stats.conflicts}")
    print(f"  Enrichments    : {stats.enrichments}")
    print(f"  Avg confidence : {stats.avg_confidence:.1%}")
    print(f"  Saved to       : {args.out}")

    if args.stats:
        synth  = DataSynthesizer()
        aggr   = synth.compute_statistics(records)
        print("\nAggregate statistics:")
        print(json.dumps(aggr, indent=2))


if __name__ == "__main__":
    _cli()
