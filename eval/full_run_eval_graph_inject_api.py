"""
eval/full_run_eval_graph_inject_api.py — E4: MAST yes/no with dynamic 2-pass graph injection (API, code+name edges).

Pass 1: standard yes/no detection prompt (no graph).
Pass 2: targeted re-analysis injecting only the causal edges whose source was detected in Pass 1.
        Pass 2 is skipped for traces where no edges are triggered.
        Pass 2 can only upgrade no→yes, never downgrade.

Usage (run from MAST/):
    python eval/full_run_eval_graph_inject_api.py --model openai/gpt-4o --causal_only
    python eval/full_run_eval_graph_inject_api.py --model openai/gpt-4o --causal_only --max_workers 5
"""

import os
import re
import json
import time
import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math

import litellm
from litellm import completion, ContextWindowExceededError, RateLimitError
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL                 = "openai/gpt-4o"
DEFAULT_EDGE_THRESHOLD        = 0.5
DEFAULT_PROPAGATION_THRESHOLD = 0.1

_EVAL_DIR  = Path(__file__).resolve().parent
_MAST_DIR  = _EVAL_DIR.parent
_GRAPH_DIR = _MAST_DIR / "causal_graph" / "outputs"

DEFAULT_STABILITY_GRAPH = _GRAPH_DIR / "edge_stability.json"
DEFAULT_EFFECT_EDGES    = _GRAPH_DIR / "interventions" / "effect_edges.json"
DEFAULT_SUPPES_GRAPH    = _GRAPH_DIR / "suppes_graph.json"

_TAXONOMY_DIR = _MAST_DIR / "taxonomy_definitions_examples"
DEFINITIONS   = (_TAXONOMY_DIR / "definitions.txt").read_text()
EXAMPLES      = (_TAXONOMY_DIR / "examples.txt").read_text()

MAST_MODES = ["1.1", "1.2", "1.3", "1.4", "1.5",
              "2.1", "2.2", "2.3", "2.4", "2.6",
              "3.1", "3.2", "3.3"]

MAST_NAMES = {
    "1.1": "Disobey Task Specification",
    "1.2": "Disobey Role Specification",
    "1.3": "Step Repetition",
    "1.4": "Loss of Conversation History",
    "1.5": "Unaware of Termination Conditions",
    "2.1": "Conversation Reset",
    "2.2": "Fail to Ask for Clarification",
    "2.3": "Task Derailment",
    "2.4": "Information Withholding",
    "2.6": "Action-Reasoning Mismatch",
    "3.1": "Premature Termination",
    "3.2": "Weak Verification",
    "3.3": "No or Incorrect Verification",
}


# ---------------------------------------------------------------------------
# Graph loading + propagation
# ---------------------------------------------------------------------------

def load_graph_edges(
    threshold: float = DEFAULT_EDGE_THRESHOLD,
    causal_only: bool = False,
    stability_graph: Path = DEFAULT_STABILITY_GRAPH,
    effect_edges: Path = DEFAULT_EFFECT_EDGES,
    suppes_graph: Path = DEFAULT_SUPPES_GRAPH,
) -> List[Tuple[str, str, float]]:
    if causal_only:
        with open(effect_edges) as f:
            data = json.load(f)
        edges = [
            (v["a"], v["b"], abs(v["delta"]))
            for v in data["edges"].values()
            if v.get("validated", False)
        ]
    else:
        with open(stability_graph) as f:
            data = json.load(f)
        stable_pairs = {
            (e["a"], e["b"])
            for e in data["edges"]
            if e["frequency"] >= threshold
        }
        with open(suppes_graph) as f:
            suppes_data = json.load(f)
        suppes_idx = {(e["a"], e["b"]): e for e in suppes_data["edges"]}
        edges = []
        for a, b in stable_pairs:
            s = suppes_idx.get((a, b))
            if s:
                score = math.sqrt(s["p_b_given_a"] * s["pr_delta"])
                edges.append((a, b, score))
    edges.sort(key=lambda x: -x[2])
    return edges


def propagate_confidence(
    detected_cats: List[str],
    edges: List[Tuple[str, str, float]],
    threshold: float,
) -> List[Tuple[str, str, float]]:
    detected_set = set(detected_cats)
    boosted: Dict[str, float] = {}
    for src, dst, w in edges:
        if src in detected_set:
            boosted[dst] = boosted.get(dst, 0.0) + w

    return [
        (src, dst, w)
        for src, dst, w in edges
        if src in detected_set
        and dst not in detected_set
        and boosted.get(dst, 0.0) > threshold
    ]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def format_trace(steps: list) -> str:
    lines = []
    for s in steps:
        lines.append(f"[{s['id']}]")
        lines.append(s["content"].strip())
        lines.append("")
    return "\n".join(lines)


def get_pass1_prompt(trace_text: str) -> str:
    return (
        "You are analyzing a multiagent system trace for failure modes and inefficiencies.\n"
        "Read the definitions and examples carefully before examining the trace.\n\n"
        "FAILURE MODE DEFINITIONS:\n"
        f"{DEFINITIONS}\n\n"
        "EXAMPLES OF FAILURE MODES:\n"
        f"{EXAMPLES}\n\n"
        "Now analyze the trace below. For each failure mode, answer yes or no.\n"
        "Multiple failure modes can and do co-occur in the same trace — be thorough and mark all that apply.\n"
        "Use a LIBERAL annotation standard: mark yes if there is any plausible indication of the failure mode,\n"
        "even if minor or partial. When in doubt, lean toward yes rather than no.\n"
        "Human annotators marked failure modes broadly — absence of good practice counts as a failure.\n\n"
        "Answer between the @@ symbols exactly as shown:\n"
        "*** begin of things you should answer *** @@\n"
        "A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>\n"
        "B. Whether the task is successfully completed or not: <yes or no>\n"
        "C. Whether you encounter any of the failure modes or inefficiencies:\n"
        "1.1 Disobey Task Specification: <yes or no>\n"
        "1.2 Disobey Role Specification: <yes or no>\n"
        "1.3 Step Repetition: <yes or no>\n"
        "1.4 Loss of Conversation History: <yes or no>\n"
        "1.5 Unaware of Termination Conditions: <yes or no>\n"
        "2.1 Conversation Reset: <yes or no>\n"
        "2.2 Fail to Ask for Clarification: <yes or no>\n"
        "2.3 Task Derailment: <yes or no>\n"
        "2.4 Information Withholding: <yes or no>\n"
        "2.6 Action-Reasoning Mismatch: <yes or no>\n"
        "3.1 Premature Termination: <yes or no>\n"
        "3.2 Weak Verification: <yes or no>\n"
        "3.3 No or Incorrect Verification: <yes or no>\n"
        "@@*** end of your answer ***\n\n"
        "An example answer is:\n"
        "A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.\n"
        "B. no\n"
        "C.\n"
        "1.1 no\n"
        "1.2 no\n"
        "1.3 no\n"
        "1.4 no\n"
        "1.5 no\n"
        "2.1 no\n"
        "2.2 no\n"
        "2.3 yes\n"
        "2.4 no\n"
        "2.6 yes\n"
        "3.1 no\n"
        "3.2 yes\n"
        "3.3 no\n\n"
        "Here is the trace:\n"
        f"{trace_text}"
    )


def get_pass2_prompt(
    trace_text: str,
    pass1_detected: List[str],
    filtered_edges: List[Tuple[str, str, float]],
) -> str:
    detected_summary = (
        ", ".join(f"{c} ({MAST_NAMES[c]})" for c in pass1_detected)
        if pass1_detected else "(none)"
    )
    edge_lines = "\n".join(
        f"  {src}({MAST_NAMES[src]}) -> {dst}({MAST_NAMES[dst]})  [strength: {w:.2f}]"
        for src, dst, w in filtered_edges
    )
    target_cats = list(dict.fromkeys(dst for _, dst, _ in filtered_edges))
    target_lines = "\n".join(
        f"{cat} {MAST_NAMES.get(cat, cat)}: <yes or no>"
        for cat in target_cats
    )

    return (
        "You are performing a TARGETED SECOND-PASS analysis of a multiagent system trace.\n\n"
        f"PASS 1 RESULTS — The following error types were already detected:\n{detected_summary}\n\n"
        "CAUSAL GRAPH CONTEXT — Statistical analysis of hundreds of agent traces has shown\n"
        "that when the PASS 1 errors above occur, the following TARGET errors very frequently\n"
        "co-occur or follow causally (code(name) -> code(name) [strength]):\n"
        f"{edge_lines}\n\n"
        "TASK: Re-read the trace below and decide YES or NO for each TARGET error type.\n"
        "Because these targets are statistically likely given the Pass 1 detections, look\n"
        "carefully and actively for supporting evidence. Err on the side of YES when there\n"
        "is any plausible indication — even indirect — in the trace.\n\n"
        "Answer yes or no for each target category between @@ symbols:\n"
        "@@ \n"
        f"{target_lines}\n"
        "@@\n\n"
        "IMPORTANT:\n"
        "- Only answer for the target categories listed above.\n"
        "- Do not repeat errors already detected in Pass 1.\n"
        "- Multiple error types can and do co-occur in the same trace.\n\n"
        "Here is the trace:\n"
        f"{trace_text}"
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_response(response: str) -> dict:
    cleaned = response.strip()
    if cleaned.startswith("@@"):
        cleaned = cleaned[2:]
    if cleaned.endswith("@@"):
        cleaned = cleaned[:-2]
    cleaned = re.sub(r'\*\*(yes|no)\*\*', r'\1', cleaned, flags=re.IGNORECASE)
    result = {}
    for mode in MAST_MODES:
        patterns = [
            rf"{mode}\s*[^:\n]*:\s*(yes|no)",
            rf"{mode}\s+(yes|no)",
            rf"{mode}\s*\n\s*(yes|no)",
        ]
        found = False
        for pattern in patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                result[mode] = 1 if match.group(1).lower() == "yes" else 0
                found = True
                break
        if not found:
            result[mode] = 0
    return result


def parse_pass2_response(response: str, target_cats: List[str]) -> Dict[str, int]:
    cleaned = response.strip()
    if cleaned.startswith("@@"):
        cleaned = cleaned[2:]
    if cleaned.endswith("@@"):
        cleaned = cleaned[:-2]
    result = {}
    for mode in target_cats:
        patterns = [
            rf"{mode}\s*[^:\n]*:\s*(yes|no)",
            rf"{mode}\s+(yes|no)",
            rf"{mode}\s*\n\s*(yes|no)",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                result[mode] = 1 if match.group(1).lower() == "yes" else 0
                break
    return result


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _is_reasoning_model(model: str) -> bool:
    return any(x in model for x in ("o1", "o3", "o4", "anthropic", "gemini-2.5"))


def call_llm(prompt: str, model: str, max_tokens: int = 4000) -> str:
    messages = [{"role": "user", "content": prompt}]
    if _is_reasoning_model(model):
        params = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": "high",
            "drop_params": True,
        }
    else:
        params = {
            "messages": messages,
            "model": model,
            "temperature": 0.0,
            "top_p": 1,
            "max_completion_tokens": max_tokens,
            "drop_params": True,
        }
    for attempt in range(5):
        try:
            response = completion(**params)
            return response.choices[0].message.content
        except RateLimitError:
            wait = 60 * (2 ** attempt)
            print(f"Rate limit (attempt {attempt+1}/5): sleeping {wait}s...")
            time.sleep(wait)
    raise RateLimitError("Exceeded 5 retries due to rate limiting")


# ---------------------------------------------------------------------------
# Per-record processing (Pass 1 + optional Pass 2)
# ---------------------------------------------------------------------------

def process_record(
    r: dict,
    output_dir: str,
    model: str,
    edges: List[Tuple[str, str, float]],
    propagation_threshold: float,
) -> None:
    rec_id = r["_rec_id"]
    output_file = os.path.join(output_dir, f"{rec_id}.json")
    if os.path.exists(output_file):
        return

    trace_text = format_trace(r["steps"])

    # Pass 1
    p1_raw = ""
    try:
        p1_raw = call_llm(get_pass1_prompt(trace_text), model)
        p1_pred = parse_response(p1_raw)
    except ContextWindowExceededError:
        p1_pred = {m: 0 for m in MAST_MODES}
        p1_raw = "Context window exceeded."
    except Exception as e:
        print(f"Error on record {rec_id} (Pass 1): {e}")
        p1_pred = {m: 0 for m in MAST_MODES}

    # Propagation check
    detected = [m for m, v in p1_pred.items() if v == 1]
    filtered_edges = propagate_confidence(detected, edges, propagation_threshold) if detected else []

    # Pass 2
    merged = dict(p1_pred)
    p2_upgrades: Dict[str, int] = {}
    p2_triggered = bool(filtered_edges)

    if p2_triggered:
        target_cats = list(dict.fromkeys(dst for _, dst, _ in filtered_edges))
        p2_raw = ""
        try:
            p2_raw = call_llm(get_pass2_prompt(trace_text, detected, filtered_edges), model)
            p2_pred = parse_pass2_response(p2_raw, target_cats)
        except ContextWindowExceededError:
            p2_pred = {}
        except Exception as e:
            print(f"Error on record {rec_id} (Pass 2): {e}")
            p2_pred = {}

        for cat, val in p2_pred.items():
            if val == 1 and merged.get(cat, 0) == 0:
                merged[cat] = 1
                p2_upgrades[cat] = 1

    out = {
        "rec_id": rec_id,
        "trace_id": r.get("trace_id"),
        "predictions": merged,
        "raw_response": p1_raw,
        "pass2_triggered": p2_triggered,
        "pass2_upgrades": p2_upgrades,
    }
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="E4: MAST yes/no with dynamic 2-pass graph injection (API, code+name edges)"
    )
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="LiteLLM model string (default: openai/gpt-4o)")
    ap.add_argument("--input", default="data/annotation/annotation_ag2_filtered.jsonl")
    ap.add_argument("--output_dir", default="outputs_full_api")
    ap.add_argument("--max_workers", type=int, default=1,
                    help="Parallel API workers (default: 1; use 1 for o1/reasoning models)")
    ap.add_argument("--causal_only", action="store_true",
                    help="Use only intervention-validated edges from effect_edges.json")
    ap.add_argument("--edge_threshold", type=float, default=DEFAULT_EDGE_THRESHOLD)
    ap.add_argument("--propagation_threshold", type=float, default=DEFAULT_PROPAGATION_THRESHOLD,
                    help="Min boosted score to trigger Pass 2 for a target category (default: 0.1)")
    ap.add_argument("--stability_graph", type=str, default=None)
    ap.add_argument("--effect_edges", type=str, default=None)
    ap.add_argument("--suppes_graph", type=str, default=None)
    ap.add_argument("--model_tag", type=str, default=None,
                    help="Override the model tag used in the output directory name")
    ap.add_argument("--sample_indices", type=str, default=None,
                    help="Path to JSON file with list of record indices to run.")
    args = ap.parse_args()

    # Load graph
    stability_path = Path(args.stability_graph) if args.stability_graph else DEFAULT_STABILITY_GRAPH
    effect_path    = Path(args.effect_edges)    if args.effect_edges    else DEFAULT_EFFECT_EDGES
    suppes_path    = Path(args.suppes_graph)    if args.suppes_graph    else DEFAULT_SUPPES_GRAPH
    edges = load_graph_edges(args.edge_threshold, args.causal_only, stability_path, effect_path, suppes_path)
    mode_str = "causal_only" if args.causal_only else f"stability>={args.edge_threshold}"
    print(f"Graph: {len(edges)} edges ({mode_str})")
    for src, dst, w in edges:
        print(f"  {src}({MAST_NAMES[src]}) -> {dst}({MAST_NAMES[dst]})  ({w:.3f})")

    # Load records
    records = []
    with open(args.input) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            r["_rec_id"] = f"{idx:04d}"
            records.append(r)
    print(f"\nLoaded {len(records)} traces")

    if args.sample_indices:
        with open(args.sample_indices) as f:
            sample_data = json.load(f)
        sample_idx_set = set(sample_data["indices"])
        records = [r for i, r in enumerate(records) if i in sample_idx_set]
        print(f"Filtered to {len(records)} traces from sample file: {args.sample_indices}")

    # Build output dir
    graph_tag = "causal_only" if args.causal_only else f"t{args.edge_threshold}"
    model_tag = args.model_tag if args.model_tag else args.model.replace("/", "-")
    out_dir   = os.path.join(args.output_dir, f"{model_tag}-yesno-graph-inject-codename-{graph_tag}")
    os.makedirs(out_dir, exist_ok=True)

    pending = [r for r in records
               if not os.path.exists(os.path.join(out_dir, f"{r['_rec_id']}.json"))]
    print(f"Pending: {len(pending)} (skipping {len(records) - len(pending)} already done)")
    if not pending:
        print("Nothing to do.")
        return

    print(f"\nModel: {args.model}  workers={args.max_workers}  "
          f"reasoning={_is_reasoning_model(args.model)}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_record, r, out_dir, args.model, edges, args.propagation_threshold
            )
            for r in pending
        ]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    print(f"\nOutputs saved to {out_dir}/")
    print(f"  Next: python eval/calculate_scores_yesno.py --pred_dir {out_dir}")


if __name__ == "__main__":
    litellm.drop_params = True
    main()
