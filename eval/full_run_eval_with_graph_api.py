"""
eval/full_run_eval_with_graph_api.py — E3: MAST yes/no with static causal graph guidance (API, code+name edges).

Graph edges use code+name inline format:
  1.1(Disobey Task Specification) -> 3.3(No or Incorrect Verification)  (strength: X.XX)

Use this for API-served models (GPT-4o, o1, etc.).
For o1, pass --sample_indices to run on a stratified subset instead of all 393 traces.

Usage (run from MAST/):
    python eval/full_run_eval_with_graph_api.py --model openai/gpt-4o --causal_only
    python eval/full_run_eval_with_graph_api.py --model openai/gpt-4o --causal_only --max_workers 5
    python eval/full_run_eval_with_graph_api.py --model openai/o1 --causal_only --sample_indices data/o1_sample_indices.json
"""

import os
import re
import json
import time
import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import math

import litellm
from litellm import completion, ContextWindowExceededError, RateLimitError
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL          = "openai/gpt-4o"
DEFAULT_EDGE_THRESHOLD = 0.2   # min geomean score for observational edges

_EVAL_DIR    = Path(__file__).resolve().parent
_MAST_DIR    = _EVAL_DIR.parent
_GRAPH_DIR   = _MAST_DIR / "causal_graph" / "outputs"

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
# Graph loading
# ---------------------------------------------------------------------------

def load_graph_edges(
    threshold: float = DEFAULT_EDGE_THRESHOLD,
    causal_only: bool = False,
    corr_threshold: float = 1.0,
    stability_graph: Path = DEFAULT_STABILITY_GRAPH,
    effect_edges: Path = DEFAULT_EFFECT_EDGES,
    suppes_graph: Path = DEFAULT_SUPPES_GRAPH,
) -> List[Tuple[str, str, float]]:
    """
    Three mutually-exclusive branches (mirrors TRAIL's
    benchmarking/eval/run_eval_graph_inject_vllm.py:load_graph_edges):

      causal_only=True       -> validated causal edges only, weight=|Δ|.
      corr_threshold < 1.0   -> UNION: (Suppes geomean ≥ corr_threshold)
                                 ∪ (validated causal edges, regardless of geomean).
                                 Weight = geomean for all edges.
      else (default)         -> pure Suppes geomean ≥ threshold (no validated union).
                                 Weight = geomean.
    """
    if causal_only:
        with open(effect_edges) as f:
            data = json.load(f)
        edges = [
            (v["a"], v["b"], abs(v["delta"]))
            for v in data["edges"].values()
            if v.get("validated", False)
        ]
    elif corr_threshold < 1.0:
        with open(suppes_graph) as f:
            suppes_data = json.load(f)
        causal_keys: set = set()
        with open(effect_edges) as f:
            ef = json.load(f)
        for v in ef["edges"].values():
            if v.get("validated", False):
                causal_keys.add((v["a"], v["b"]))
        edges = []
        for e in suppes_data["edges"]:
            a, b = e["a"], e["b"]
            score = math.sqrt(e["precedence"] * e["pr_delta"])
            if (a, b) in causal_keys or score >= corr_threshold:
                edges.append((a, b, score))
    else:
        with open(suppes_graph) as f:
            suppes_data = json.load(f)
        edges = []
        for e in suppes_data["edges"]:
            score = math.sqrt(e["precedence"] * e["pr_delta"])
            if score >= threshold:
                edges.append((e["a"], e["b"], score))
    edges.sort(key=lambda x: -x[2])
    return edges


def format_graph_guidance(edges: List[Tuple[str, str, float]], causal_only: bool = True) -> str:
    """Format edges using code+name inline format — no separate lookup table needed."""
    if not edges:
        return ""
    if causal_only:
        lines = [
            "CAUSAL ERROR PATTERNS (intervention-validated):",
            "The following edges were validated via counterfactual patching experiments.",
            "When you identify error type A, actively look for error type B,",
            "as removing A causally reduces B's occurrence rate.",
            "Higher values indicate stronger causal effect (reduction in B's rate when A is patched).",
            "",
            "Format: [code(name)] -> [code(name)]  (causal effect: X.XX)",
            "",
        ]
        for src, dst, w in edges:
            lines.append(
                f"  {src}({MAST_NAMES[src]}) -> {dst}({MAST_NAMES[dst]})  (causal effect: {w:.2f})"
            )
    else:
        lines = [
            "CORRELATED ERROR PATTERNS (observational, precedence-filtered):",
            "The following error pairs consistently co-occur with A preceding B across agent traces.",
            "Score = geometric mean of precedence P(A precedes B | both occur) and probability-raising delta P(B|A)−P(B|¬A).",
            "When you identify error type A, consider also checking for error type B.",
            "Higher values indicate stronger observational association.",
            "",
            "Format: [code(name)] -> [code(name)]  (observational score: X.XX)",
            "",
        ]
        for src, dst, w in edges:
            lines.append(
                f"  {src}({MAST_NAMES[src]}) -> {dst}({MAST_NAMES[dst]})  (observational score: {w:.2f})"
            )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt + parsing
# ---------------------------------------------------------------------------

def format_trace(steps: list) -> str:
    lines = []
    for s in steps:
        lines.append(f"[{s['id']}]")
        lines.append(s["content"].strip())
        lines.append("")
    return "\n".join(lines)


def get_prompt(trace_text: str, graph_guidance: str) -> str:
    return (
        "You are analyzing a multiagent system trace for failure modes and inefficiencies.\n"
        "Read the definitions and examples carefully before examining the trace.\n\n"
        "FAILURE MODE DEFINITIONS:\n"
        f"{DEFINITIONS}\n\n"
        "EXAMPLES OF FAILURE MODES:\n"
        f"{EXAMPLES}\n\n"
        f"{graph_guidance}"
        "Now analyze the trace below. For each failure mode, answer yes or no.\n"
        "Multiple failure modes can and do co-occur in the same trace — be thorough and mark all that apply.\n"
        "Use a LIBERAL annotation standard: mark yes if there is any plausible indication of the failure mode,\n"
        "even if minor or partial. When in doubt, lean toward yes rather than no.\n"
        "Human annotators marked failure modes broadly — absence of good practice counts as a failure.\n"
        "Every trace in this dataset contains AT LEAST 2 failure modes (median 3, range 2-12 out of 13). "
        "An all-no response is wrong by construction; if you find yourself marking all no, re-examine the trace.\n\n"
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
        "Disambiguation rules:\n"
        "- 3.2 vs 3.3: Mark 3.2 yes ONLY if a verification step exists but is insufficient/cursory. "
        "If no verification step exists at all, mark 3.3 yes (and 3.2 no). They are mutually exclusive — "
        "do not mark both.\n"
        "- 1.4 Loss of Conversation History: mark yes when an agent's later turn ignores, contradicts, or "
        "asks for information that was already established earlier in the conversation.\n"
        "- 2.1 Conversation Reset: mark yes when context appears wiped or an agent restarts the task as if "
        "from scratch, dropping prior progress.\n\n"
        "An example answer is:\n"
        "A. The task is not completed correctly. The user agent restated the problem mid-trace and the "
        "solver re-derived earlier intermediate values from scratch, ignoring computations already produced. "
        "The solver also stopped before the termination criterion was met. A verifier was invoked but only "
        "checked surface formatting, not numerical correctness.\n"
        "B. no\n"
        "C.\n"
        "1.1 no\n"
        "1.2 no\n"
        "1.3 yes\n"
        "1.4 yes\n"
        "1.5 yes\n"
        "2.1 no\n"
        "2.2 no\n"
        "2.3 no\n"
        "2.4 no\n"
        "2.6 no\n"
        "3.1 yes\n"
        "3.2 yes\n"
        "3.3 no\n\n"
        "Here is the trace:\n"
        f"{trace_text}"
    )


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


# ---------------------------------------------------------------------------
# Per-record API call
# ---------------------------------------------------------------------------

def _is_reasoning_model(model: str) -> bool:
    return any(x in model for x in ("o1", "o3", "o4", "anthropic", "gemini-2.5"))


def call_llm(prompt: str, model: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if _is_reasoning_model(model):
        params = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": 8000,
            "reasoning_effort": "high",
            "drop_params": True,
        }
    else:
        params = {
            "messages": messages,
            "model": model,
            "temperature": 0.0,
            "top_p": 1,
            "max_completion_tokens": 4000,
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


def process_record(r: dict, output_dir: str, model: str, graph_guidance: str) -> None:
    rec_id = r["_rec_id"]
    output_file = os.path.join(output_dir, f"{rec_id}.json")
    if os.path.exists(output_file):
        return

    trace_text = format_trace(r["steps"])
    prompt = get_prompt(trace_text, graph_guidance)
    raw_response = ""
    try:
        raw_response = call_llm(prompt, model)
        predictions = parse_response(raw_response)
    except ContextWindowExceededError:
        predictions = {m: 0 for m in MAST_MODES}
        raw_response = "Context window exceeded."
    except Exception as e:
        print(f"Error on record {rec_id}: {e}")
        predictions = {m: 0 for m in MAST_MODES}

    out = {
        "rec_id": rec_id,
        "trace_id": r.get("trace_id"),
        "predictions": predictions,
        "raw_response": raw_response,
    }
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="E3: MAST yes/no with static causal graph guidance (API, code+name edges)"
    )
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="LiteLLM model string (default: openai/gpt-4o)")
    ap.add_argument("--input", default="data/annotation/annotation_ag2_filtered.jsonl")
    ap.add_argument("--output_dir", default="outputs_full_api")
    ap.add_argument("--max_workers", type=int, default=1,
                    help="Parallel API workers (default: 1; use 1 for o1/reasoning models)")
    ap.add_argument("--causal_only", action="store_true",
                    help="Use only intervention-validated edges from effect_edges.json")
    ap.add_argument("--corr_threshold", type=float, default=1.0,
                    help="If < 1.0, use UNION ablation: (Suppes geomean ≥ τ) ∪ "
                         "(validated causal edges). Mirrors TRAIL's --corr_threshold. "
                         "Mutually exclusive with --causal_only.")
    ap.add_argument("--edge_threshold", type=float, default=DEFAULT_EDGE_THRESHOLD,
                    help="Min geomean score sqrt(precedence*PR_delta) for observational edges (default: 0.2)")
    ap.add_argument("--stability_graph", type=str, default=None)
    ap.add_argument("--effect_edges", type=str, default=None)
    ap.add_argument("--suppes_graph", type=str, default=None)
    ap.add_argument("--model_tag", type=str, default=None,
                    help="Override the model tag used in the output directory name")
    ap.add_argument("--sample_indices", type=str, default=None,
                    help="Path to JSON file with list of record indices to run (e.g. data/o1_sample_indices.json). "
                         "Outputs use original rec_ids so scoring against full GT works normally.")
    args = ap.parse_args()

    # Load graph
    stability_path = Path(args.stability_graph) if args.stability_graph else DEFAULT_STABILITY_GRAPH
    effect_path    = Path(args.effect_edges)    if args.effect_edges    else DEFAULT_EFFECT_EDGES
    suppes_path    = Path(args.suppes_graph)    if args.suppes_graph    else DEFAULT_SUPPES_GRAPH
    if args.causal_only and args.corr_threshold < 1.0:
        ap.error("--causal_only and --corr_threshold are mutually exclusive")
    edges = load_graph_edges(
        threshold=args.edge_threshold,
        causal_only=args.causal_only,
        corr_threshold=args.corr_threshold,
        stability_graph=stability_path,
        effect_edges=effect_path,
        suppes_graph=suppes_path,
    )
    graph_guidance = format_graph_guidance(edges, causal_only=args.causal_only)
    if args.causal_only:
        mode_str = "causal_only"
    elif args.corr_threshold < 1.0:
        mode_str = f"corr>={args.corr_threshold} (Suppes ∪ validated)"
    else:
        mode_str = f"geomean>={args.edge_threshold}"
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

    # Apply sample filter if provided
    if args.sample_indices:
        with open(args.sample_indices) as f:
            sample_data = json.load(f)
        sample_idx_set = set(sample_data["indices"])
        records = [r for i, r in enumerate(records) if i in sample_idx_set]
        print(f"Filtered to {len(records)} traces from sample file: {args.sample_indices}")

    # Build output dir
    if args.causal_only:
        graph_tag = "causal_only"
    elif args.corr_threshold < 1.0:
        graph_tag = f"corr{args.corr_threshold}"
    else:
        graph_tag = f"t{args.edge_threshold}"
    model_tag = args.model_tag if args.model_tag else args.model.replace("/", "-")
    out_dir   = os.path.join(args.output_dir, f"{model_tag}-yesno-with-graph-codename-{graph_tag}")
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
            executor.submit(process_record, r, out_dir, args.model, graph_guidance)
            for r in pending
        ]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    print(f"\n✓ Outputs saved to {out_dir}/")
    print(f"  Next: python eval/calculate_scores_yesno.py --pred_dir {out_dir}")


if __name__ == "__main__":
    litellm.drop_params = True
    main()
