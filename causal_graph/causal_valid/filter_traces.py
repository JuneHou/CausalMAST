#!/usr/bin/env python3
"""
Step 0: Filter MAST-AG2 traces eligible for causal intervention.

Adapted from TRAIL causal/patch/filter_traces.py.
Key changes vs TRAIL:
  - Reads from a single annotation_ag2_filtered.jsonl (not per-file annotations_dir).
  - Category codes are normalised: "2.2 Fail to Ask for Clarification" → "2.2".
  - Only errors where mast_annotation[cat] == 1 (ground-truth positive) are used.

A trace is eligible if it has:
  1. >= min_errors GT-positive annotated errors with locations, AND
  2. At least one error whose type is an A-type (source node) in the causal graph.

Optionally also requires at least one B-type error appearing AFTER the A-type
(strict mode: ensures b_present_baseline=True for at least one (A,B) pair).

Output format:
  eligible_traces.json — {
    "n_total": int,
    "n_eligible": int,
    "a_types": [...],
    "b_types": [...],
    "eligible": [
      { "trace_id": str/int, "n_errors": int,
        "a_errors": [{"type": str, "step_id": str, "index": int}],
        "b_errors_after_a": [{"type": str, "step_id": str, "index": int}],
        "covered_edges": [{"a": str, "b": str}] }
    ]
  }

Usage:
    python filter_traces.py --input ../data/annotation/annotation_ag2_filtered.jsonl \\
                            --causal_graph ../outputs/capri_graph.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Category normalisation
# ---------------------------------------------------------------------------

def _norm_cat(cat: str) -> str:
    """'2.2 Fail to Ask for Clarification' → '2.2'"""
    if not cat:
        return ""
    cat = cat.strip()
    parts = cat.split()
    if parts and re.match(r"^\d+\.\d+$", parts[0]):
        return parts[0]
    return cat


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def load_graph(graph_path: str) -> Tuple[Set[str], Set[str], Set[Tuple[str, str]]]:
    """Returns a_types, b_types, edge_set (all as category code strings)."""
    with open(graph_path, "r", encoding="utf-8") as f:
        g = json.load(f)
    edges = [(_norm_cat(e["a"]), _norm_cat(e["b"])) for e in g.get("edges", [])]
    a_types = {a for a, _ in edges}
    b_types = {b for _, b in edges}
    return a_types, b_types, set(edges)


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------

def filter_traces(
    input_jsonl: str,
    graph_path: str,
    min_errors: int = 2,
    strict: bool = False,
) -> dict:
    """
    Scan annotation JSONL and return eligible trace metadata.

    strict=False : require only >=1 A-type error.
    strict=True  : require at least one (A,B) pair where B appears after A.
    """
    a_types, b_types, edge_set = load_graph(graph_path)
    a_to_bs: Dict[str, List[str]] = defaultdict(list)
    for a, b in edge_set:
        a_to_bs[a].append(b)

    eligible = []
    n_total = 0

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n_total += 1
            trace_id = record.get("trace_id")
            gt_flags = record.get("mast_annotation", {})

            # Collect GT-positive errors with locations (from errors array)
            errors = []
            for i, err in enumerate(record.get("errors", [])):
                cat_full = err.get("category", "")
                cat_code = _norm_cat(cat_full)
                # Only use errors where GT flag is 1
                if not gt_flags.get(cat_code, 0):
                    continue
                step_id = err.get("location", "")
                if not step_id:
                    continue
                errors.append({
                    "type": cat_code,
                    "step_id": step_id,
                    "index": i,
                })

            if len(errors) < min_errors:
                continue

            # Find A-type errors
            a_errors = [e for e in errors if e["type"] in a_types]
            if not a_errors:
                continue

            # Find B-type errors after each A (by step index order)
            def _step_num(step_id: str) -> int:
                try:
                    return int(step_id.split("_")[1])
                except (IndexError, ValueError):
                    return -1

            covered_edges = []
            b_after_a = []
            for ae in a_errors:
                a_sn = _step_num(ae["step_id"])
                for be in errors:
                    if _step_num(be["step_id"]) <= a_sn:
                        continue
                    if be["type"] in b_types and (ae["type"], be["type"]) in edge_set:
                        b_after_a.append({
                            "type": be["type"],
                            "step_id": be["step_id"],
                            "index": be["index"],
                        })
                        edge = {"a": ae["type"], "b": be["type"]}
                        if edge not in covered_edges:
                            covered_edges.append(edge)

            if strict and not covered_edges:
                continue

            eligible.append({
                "trace_id": trace_id,
                "n_errors": len(errors),
                "a_errors": [{"type": e["type"], "step_id": e["step_id"], "index": e["index"]}
                              for e in a_errors],
                "b_errors_after_a": b_after_a,
                "covered_edges": covered_edges,
            })

    return {
        "n_total": n_total,
        "n_eligible": len(eligible),
        "a_types": sorted(a_types),
        "b_types": sorted(b_types),
        "strict_mode": strict,
        "min_errors": min_errors,
        "eligible": eligible,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter MAST-AG2 traces eligible for do(A=0) causal intervention."
    )
    parser.add_argument("--input", default="../data/annotation/annotation_ag2_filtered.jsonl",
                        help="Path to annotation_ag2_filtered.jsonl")
    parser.add_argument("--causal_graph", default="../outputs/capri_graph.json",
                        help="Path to capri_graph.json")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: same dir as causal_graph)")
    parser.add_argument("--min_errors", type=int, default=2)
    parser.add_argument("--strict", action="store_true",
                        help="Also require at least one B-type error after an A-type error")
    args = parser.parse_args()

    result = filter_traces(
        args.input, args.causal_graph,
        min_errors=args.min_errors, strict=args.strict,
    )

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.causal_graph))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eligible_traces.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nA-types in graph : {result['a_types']}")
    print(f"Total traces     : {result['n_total']}")
    print(f"Eligible traces  : {result['n_eligible']} "
          f"(min_errors={args.min_errors}, strict={args.strict})")

    edge_counts: Dict[str, int] = defaultdict(int)
    for t in result["eligible"]:
        for e in t["covered_edges"]:
            edge_counts[f"{e['a']} -> {e['b']}"] += 1

    if edge_counts:
        print("\nTraces covering each graph edge:")
        for edge, count in sorted(edge_counts.items()):
            print(f"  {count:3d}  {edge}")
    else:
        print("\nNo (A→B) pairs found (try without --strict).")

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
