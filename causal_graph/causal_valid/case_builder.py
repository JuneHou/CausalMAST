#!/usr/bin/env python3
"""
Step 1: Build AInstanceRecords and EdgePairs from MAST-AG2 traces + annotations.

Adapted from TRAIL causal/patch/case_builder.py.
Key changes vs TRAIL:
  - No trail_io / OpenInference spans. Reads directly from annotation_ag2_filtered.jsonl.
  - Location = step_XX (sequential step ID) instead of span_id.
  - patch_side is always "replace_step_content" (no LLM/TOOL span hierarchy).
  - local_snippet = the step content text (after stripping [agent_name] prefix).
  - prefix_context = generic MAST multi-agent context.
  - user_requirements = content of step_00 (initial task description).
  - error_id generated as "{trace_id}_s{step_num:02d}_{cat_code}".
  - Category normalisation: "2.2 Fail to Ask for Clarification" → "2.2".

AInstanceRecord — one per unique (trace_id, error_id) A-instance.
EdgePair        — one per (AInstanceRecord × B-type) graph edge.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple


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


def _step_num(step_id: str) -> int:
    """'step_04' → 4"""
    try:
        return int(step_id.split("_")[1])
    except (IndexError, ValueError):
        return -1


# ---------------------------------------------------------------------------
# Step content helpers
# ---------------------------------------------------------------------------

def _parse_step_content(content: str) -> Tuple[str, str]:
    """
    Parse '[agent_name]\\ncontent...' → (agent_name, content_text).
    Returns ("assistant", full_content) if no role prefix found.
    """
    lines = content.strip().split("\n", 1)
    first = lines[0].strip()
    if first.startswith("[") and first.endswith("]"):
        agent = first[1:-1].strip()
        text = lines[1].strip() if len(lines) > 1 else ""
        return agent, text
    return "assistant", content.strip()


def _get_local_snippet(step_content: str, max_chars: int = 6000) -> str:
    """Return the step text (without role prefix) for the patch generator."""
    _, text = _parse_step_content(step_content)
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Data structures (identical interface to TRAIL)
# ---------------------------------------------------------------------------

@dataclass
class AInstanceRecord:
    """One per unique (trace_id, error_id) A-instance. Used for patch → rerun → Judge A."""
    trace_id: str
    error_id: str
    a_instance: dict          # category, location, description, evidence, impact, annotation_index
    local_snippet: str        # step text to replace (without [agent_name] prefix)
    patch_side: str           # always "replace_step_content" for MAST
    annotated_location: str   # step_XX from the annotation
    intervention_location: str  # same as annotated_location (no span remap needed)
    annotated_span_kind: str  # always "STEP" for MAST
    intervention_span_kind: str  # always "STEP" for MAST
    prefix_context: str       # generic multi-agent system context
    user_requirements: str    # task description (step_00 content)
    tools_available: list     # empty for MAST (no explicit tool list in traces)
    suffix_window_spec: dict  # {"mode": "until_end"}
    b_types: list             # B-type categories this A-instance serves (informational)


@dataclass
class EdgePair:
    """One per (AInstanceRecord × B-type) graph edge. Used for Judge B and aggregation."""
    trace_id: str
    error_id: str             # FK → AInstanceRecord
    edge: dict                # {"a": ..., "b": ...}
    b_def: dict               # {"category": b_cat}
    b_present_baseline: bool  # whether B appears after step_A in original annotations
    b_onset_baseline: int     # step number of first B after A (-1 if absent)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def load_graph_edges(graph_path: str) -> Tuple[Set[Tuple[str, str]], Dict[str, List[str]]]:
    with open(graph_path, "r", encoding="utf-8") as f:
        g = json.load(f)
    edges = [(_norm_cat(e["a"]), _norm_cat(e["b"])) for e in g.get("edges", [])]
    allowed_edges: Set[Tuple[str, str]] = set(edges)
    a_to_bs: Dict[str, List[str]] = {}
    for a, b in edges:
        a_to_bs.setdefault(a, []).append(b)
    return allowed_edges, a_to_bs


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

PREFIX_CONTEXT = (
    "You are an AI assistant agent in a multi-agent system built with AG2 (AutoGen). "
    "Multiple agents collaborate to complete tasks. Follow instructions carefully and "
    "produce accurate, well-reasoned responses."
)


def build_cases(
    input_jsonl: str,
    graph_path: str,
    eligible_trace_ids: Optional[List] = None,
    max_traces: Optional[int] = None,
) -> Tuple[List[AInstanceRecord], List[EdgePair]]:
    """
    Build AInstanceRecords and EdgePairs for all (A-instance, B-type) pairs in the
    causal graph.

    Returns:
      a_instances : one per unique (trace_id, error_id) A-instance
      edge_pairs  : one per (A-instance × B-type) graph edge
    """
    allowed_edges, a_to_bs = load_graph_edges(graph_path)
    a_types: Set[str] = set(a for a, _ in allowed_edges)

    eligible_set = set(str(t) for t in eligible_trace_ids) if eligible_trace_ids else None

    a_instances: List[AInstanceRecord] = []
    edge_pairs: List[EdgePair] = []
    seen_error_ids: Set[str] = set()
    n_traces = 0

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            trace_id = str(record.get("trace_id", ""))

            if eligible_set and trace_id not in eligible_set:
                continue
            if max_traces and n_traces >= max_traces:
                break
            n_traces += 1

            gt_flags = record.get("mast_annotation", {})
            steps = record.get("steps", [])
            errors_raw = record.get("errors", [])

            # Collect GT-positive errors with locations
            errors = []
            for i, err in enumerate(errors_raw):
                cat_full = err.get("category", "")
                cat_code = _norm_cat(cat_full)
                if not gt_flags.get(cat_code, 0):
                    continue
                step_id = err.get("location", "")
                if not step_id:
                    continue
                errors.append({
                    "type": cat_code,
                    "step_id": step_id,
                    "step_num": _step_num(step_id),
                    "raw_err": err,
                    "annotation_index": i,
                })

            # Task description from step_00
            user_requirements = steps[0]["content"][:3000] if steps else ""
            # Step content index
            step_content_by_id = {s["id"]: s["content"] for s in steps}

            # B-type categories present in this trace
            trace_b_types: Set[str] = {e["type"] for e in errors}

            for err in errors:
                a_cat = err["type"]
                if a_cat not in a_types:
                    continue

                step_id = err["step_id"]
                step_content = step_content_by_id.get(step_id, "")
                if not step_content:
                    continue

                b_targets = [b for b in a_to_bs.get(a_cat, []) if b in trace_b_types]
                if not b_targets:
                    continue

                # Generate unique error ID
                raw_err = err["raw_err"]
                error_id = f"{trace_id}_s{err['step_num']:02d}_{a_cat}"

                local_snippet = _get_local_snippet(step_content)

                a_instance_dict = {
                    "category": a_cat,
                    "location": step_id,
                    "description": raw_err.get("description", ""),
                    "evidence": raw_err.get("evidence", ""),
                    "impact": raw_err.get("impact", ""),
                    "error_id": error_id,
                    "annotation_index": err["annotation_index"],
                }

                # One AInstanceRecord per unique error_id
                if error_id not in seen_error_ids:
                    seen_error_ids.add(error_id)
                    a_instances.append(AInstanceRecord(
                        trace_id=trace_id,
                        error_id=error_id,
                        a_instance=a_instance_dict,
                        local_snippet=local_snippet,
                        patch_side="replace_step_content",
                        annotated_location=step_id,
                        intervention_location=step_id,
                        annotated_span_kind="STEP",
                        intervention_span_kind="STEP",
                        prefix_context=PREFIX_CONTEXT,
                        user_requirements=user_requirements,
                        tools_available=[],
                        suffix_window_spec={"mode": "until_end"},
                        b_types=b_targets,
                    ))

                # One EdgePair per (error_id, b_type)
                a_step_num = err["step_num"]
                for b_cat in b_targets:
                    b_present = False
                    b_onset = -1
                    for other_err in errors:
                        if other_err["step_num"] <= a_step_num:
                            continue
                        if other_err["type"] == b_cat:
                            b_present = True
                            b_onset = other_err["step_num"]
                            break

                    edge_pairs.append(EdgePair(
                        trace_id=trace_id,
                        error_id=error_id,
                        edge={"a": a_cat, "b": b_cat},
                        b_def={"category": b_cat},
                        b_present_baseline=b_present,
                        b_onset_baseline=b_onset,
                    ))

    return a_instances, edge_pairs


# ---------------------------------------------------------------------------
# Dedup (same interface as TRAIL)
# ---------------------------------------------------------------------------

def dedup_by_intervention_location(
    a_instances: List[AInstanceRecord],
    conflicts_path: Optional[str] = None,
) -> List[AInstanceRecord]:
    """
    When multiple A-instances share the same (trace_id, step_id), keep only the first.
    For MAST, a single step can contain at most one patch.
    """
    seen: Dict[str, str] = {}   # (trace_id, step_id) → kept error_id
    active: List[AInstanceRecord] = []
    conflicts: List[dict] = []

    for rec in a_instances:
        key = f"{rec.trace_id}:{rec.intervention_location}"
        if key not in seen:
            seen[key] = rec.error_id
            active.append(rec)
        else:
            conflicts.append({
                **asdict(rec),
                "conflict_reason": "shared_intervention_location",
                "kept_error_id": seen[key],
            })

    if conflicts and conflicts_path:
        os.makedirs(os.path.dirname(os.path.abspath(conflicts_path)), exist_ok=True)
        with open(conflicts_path, "w", encoding="utf-8") as f:
            for r in conflicts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(conflicts)} location-conflict records → {conflicts_path}")

    if conflicts:
        print(f"[dedup] Kept {len(active)}/{len(active) + len(conflicts)} A-instances "
              f"({len(conflicts)} skipped: shared_intervention_location)")

    return active


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build AInstanceRecords and EdgePairs for MAST-AG2 causal intervention."
    )
    parser.add_argument("--input", default="../data/annotation/annotation_ag2_filtered.jsonl")
    parser.add_argument("--causal_graph", default="../outputs/capri_graph.json")
    parser.add_argument("--eligible_traces", default=None,
                        help="Path to eligible_traces.json from filter_traces.py")
    parser.add_argument("--out_dir", default="outputs/interventions")
    parser.add_argument("--max_traces", type=int, default=None)
    args = parser.parse_args()

    eligible_ids = None
    if args.eligible_traces and os.path.isfile(args.eligible_traces):
        with open(args.eligible_traces, "r", encoding="utf-8") as f:
            et = json.load(f)
        eligible_ids = [str(t["trace_id"]) for t in et.get("eligible", [])]
        print(f"Using {len(eligible_ids)} eligible traces from {args.eligible_traces}")

    os.makedirs(args.out_dir, exist_ok=True)
    a_instances, edge_pairs = build_cases(
        args.input, args.causal_graph,
        eligible_trace_ids=eligible_ids,
        max_traces=args.max_traces,
    )

    conflicts_path = os.path.join(args.out_dir, "intervention_location_conflicts.jsonl")
    a_instances = dedup_by_intervention_location(a_instances, conflicts_path=conflicts_path)

    a_path = os.path.join(args.out_dir, "a_instances.jsonl")
    e_path = os.path.join(args.out_dir, "edge_pairs.jsonl")

    with open(a_path, "w", encoding="utf-8") as f:
        for rec in a_instances:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    with open(e_path, "w", encoding="utf-8") as f:
        for rec in edge_pairs:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    from collections import Counter
    edge_counts = Counter(f"{ep.edge['a']} -> {ep.edge['b']}" for ep in edge_pairs)
    b_rate = sum(1 for ep in edge_pairs if ep.b_present_baseline) / max(len(edge_pairs), 1)
    print(f"Built {len(a_instances)} A-instances → {a_path}")
    print(f"Built {len(edge_pairs)} edge pairs  → {e_path}")
    print(f"b_present_baseline rate: {b_rate:.2%}")
    print("Edge pairs per edge:")
    for edge, count in sorted(edge_counts.items()):
        print(f"  {count:3d}  {edge}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
