"""
eval/code_name/run_eval_with_graph.py — E3 variant: graph edges use code(name) format.

Identical to eval/run_eval_with_graph.py except graph edges are formatted as:
  1.1(Disobey Task Specification) -> 3.3(No or Incorrect Verification)  (strength: X.XX)
instead of codes-only. The separate category lookup table is omitted since names
are now inline in each edge line.

Usage (run from MAST/):
    CUDA_VISIBLE_DEVICES=4,5 python eval/full_run_eval_with_graph.py --causal_only
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple

from vllm import LLM, SamplingParams
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL        = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
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
    random_edges: bool = False,
    random_seed: int = 42,
    random_n: int = 11,
) -> List[Tuple[str, str, float]]:
    """
    Four mutually-exclusive branches (mirrors TRAIL's
    benchmarking/eval/run_eval_graph_inject_vllm.py:load_graph_edges):

      causal_only=True       -> validated causal edges only, weight=|Δ|.
      corr_threshold < 1.0   -> UNION: (Suppes geomean ≥ corr_threshold)
                                 ∪ (validated causal edges, regardless of geomean).
                                 Weight = geomean for all edges.
      random_edges=True      -> sample random_n directed pairs from MAST_MODES
                                excluding Suppes-screened edges. Null-graph control.
      else (default)         -> pure Suppes geomean ≥ threshold (no validated union).
                                 Weight = geomean.
    """
    if random_edges:
        import random as _rnd
        with open(suppes_graph) as f:
            suppes_data = json.load(f)
        suppes_keys = {(e["a"], e["b"]) for e in suppes_data["edges"]}
        nodes = sorted(MAST_MODES)
        candidate = [(a, b) for a in nodes for b in nodes if a != b and (a, b) not in suppes_keys]
        rnd = _rnd.Random(random_seed)
        sampled = rnd.sample(candidate, min(random_n, len(candidate)))
        return [(a, b, 1.0) for a, b in sampled]

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
        # Validated causal edges (forced into the union regardless of geomean)
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


def format_graph_guidance(
    edges: List[Tuple[str, str, float]],
    causal_only: bool = True,
    random_edges: bool = False,
) -> str:
    """Format edges as a static guidance block using code(name) format."""
    if not edges:
        return ""
    if random_edges:
        lines = [
            "RANDOM ERROR PATTERN BASELINE (uncalibrated):",
            "The following edges are sampled uniformly at random from directed category pairs",
            "outside the Suppes-screened graph. They carry no probabilistic interpretation and",
            "serve as a control for graph-structure ablations.",
            "When you identify error type A, consider also checking for error type B.",
            "",
            "Format: [code(name)] -> [code(name)]",
            "",
        ]
        for src, dst, _ in edges:
            lines.append(f"  {src}({MAST_NAMES[src]}) -> {dst}({MAST_NAMES[dst]})")
    elif causal_only:
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
        "Only mark a failure mode if you can identify a specific example of it in the trace.\n\n"
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


def strip_thinking(text: str) -> tuple:
    """Remove thinking block; return (thinking_text, remaining_text).

    Handles two formats:
      1. Complete <think>...</think> pair (Qwen3, DeepSeek with explicit open tag).
      2. Orphan </think> only — vLLM injects the opening <think> via chat template
         so generated tokens start mid-thought with no <think> tag (QwQ-32B pattern).
    """
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), (text[:match.start()] + text[match.end():]).strip()
    idx = text.find('</think>')
    if idx != -1:
        return text[:idx].strip(), text[idx + 8:].strip()
    return "", text.strip()


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
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="E3: MAST yes/no with static causal graph guidance")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tp", type=int, default=None,
                    help="Tensor parallel size (auto-detected from CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--input", default="data/annotation/annotation_ag2_filtered.jsonl")
    ap.add_argument("--output_dir", default="outputs_full")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--max_model_len", type=int, default=128000)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                    help="Fraction of GPU memory vLLM may use per device (default: 0.9)")
    ap.add_argument("--causal_only", action="store_true",
                    help="Use only intervention-validated edges from effect_edges.json")
    ap.add_argument("--corr_threshold", type=float, default=1.0,
                    help="If < 1.0, use UNION ablation: (Suppes geomean ≥ τ) ∪ "
                         "(validated causal edges). Mirrors TRAIL's --corr_threshold. "
                         "Mutually exclusive with --causal_only.")
    ap.add_argument("--edge_threshold", type=float, default=DEFAULT_EDGE_THRESHOLD,
                    help="Min geomean score sqrt(precedence*PR_delta) for observational edges (default: 0.2)")
    ap.add_argument("--random_edges", action="store_true",
                    help="Random-N baseline: sample edges from MAST taxonomy minus Suppes graph.")
    ap.add_argument("--random_seed", type=int, default=42,
                    help="Seed for --random_edges sampling.")
    ap.add_argument("--random_n", type=int, default=11,
                    help="Number of random edges to sample (default 11, matches causal-only).")
    ap.add_argument("--stability_graph", type=str, default=None)
    ap.add_argument("--effect_edges", type=str, default=None)
    ap.add_argument("--suppes_graph", type=str, default=None)
    ap.add_argument("--model_tag", type=str, default=None,
                    help="Override the model tag used in the output directory name")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Pass enable_thinking=True via chat_template_kwargs (for QwQ/Qwen3/DeepSeek-R1)")
    ap.add_argument("--span_index", action="store_true",
                    help="Accepted for CLI parity with TRAIL sweep; no-op (MAST has no location prediction).")
    args = ap.parse_args()

    if args.tp is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        args.tp = len([x for x in cvd.split(",") if x.strip()]) if cvd.strip() else 1

    # Load graph
    stability_path = Path(args.stability_graph) if args.stability_graph else DEFAULT_STABILITY_GRAPH
    effect_path    = Path(args.effect_edges)    if args.effect_edges    else DEFAULT_EFFECT_EDGES
    suppes_path    = Path(args.suppes_graph)    if args.suppes_graph    else DEFAULT_SUPPES_GRAPH
    mode_count = sum([args.causal_only, args.corr_threshold < 1.0, args.random_edges])
    if mode_count > 1:
        ap.error("--causal_only, --corr_threshold, and --random_edges are mutually exclusive")
    edges = load_graph_edges(
        threshold=args.edge_threshold,
        causal_only=args.causal_only,
        corr_threshold=args.corr_threshold,
        stability_graph=stability_path,
        effect_edges=effect_path,
        suppes_graph=suppes_path,
        random_edges=args.random_edges,
        random_seed=args.random_seed,
        random_n=args.random_n,
    )
    # Causal-only -> intervention-validated. corr/edge_threshold paths -> observational.
    is_causal_prose = args.causal_only
    graph_guidance = format_graph_guidance(
        edges, causal_only=is_causal_prose, random_edges=args.random_edges,
    )
    if args.random_edges:
        mode_str = f"random{args.random_n}_seed{args.random_seed} (null-graph control)"
    elif args.causal_only:
        mode_str = "causal_only"
    elif args.corr_threshold < 1.0:
        mode_str = f"corr>={args.corr_threshold} (Suppes ∪ validated)"
    else:
        mode_str = f"geomean>={args.edge_threshold}"
    print(f"Graph: {len(edges)} edges ({mode_str})")
    for src, dst, w in edges:
        print(f"  {src} → {dst}  ({w:.3f})")

    # Load records
    records = []
    with open(args.input) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            r["_rec_id"] = f"{idx:04d}"
            records.append(r)
    print(f"\nLoaded {len(records)} traces")

    if args.random_edges:
        graph_tag = f"random{args.random_n}_seed{args.random_seed}"
    elif args.causal_only:
        graph_tag = "causal_only"
    elif args.corr_threshold < 1.0:
        graph_tag = f"corr{args.corr_threshold}"
    else:
        graph_tag = f"t{args.edge_threshold}"
    model_tag = args.model_tag if args.model_tag else args.model.replace("/", "-")
    thinking_suffix = "-thinking" if args.enable_thinking else ""
    out_dir   = os.path.join(args.output_dir, f"{model_tag}-yesno-with-graph-codename-{graph_tag}{thinking_suffix}")
    os.makedirs(out_dir, exist_ok=True)

    pending = [r for r in records
               if not os.path.exists(os.path.join(out_dir, f"{r['_rec_id']}.json"))]
    print(f"Pending: {len(pending)} (skipping {len(records) - len(pending)} already done)")
    if not pending:
        print("Nothing to do.")
        return

    # Load model
    print(f"\nLoading model: {args.model}  (tp={args.tp})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Build conversations
    conversations = []
    for r in pending:
        trace_text = format_trace(r["steps"])
        conversations.append([{"role": "user", "content": get_prompt(trace_text, graph_guidance)}])

    chat_template_kwargs = {"enable_thinking": True} if args.enable_thinking else {}

    # Batch inference
    print(f"Running inference (batch_size={args.batch_size}, enable_thinking={args.enable_thinking})...")
    all_outputs = []
    for i in tqdm(range(0, len(conversations), args.batch_size)):
        batch = conversations[i: i + args.batch_size]
        outputs = llm.chat(batch, sampling_params=sampling, use_tqdm=False,
                           chat_template_kwargs=chat_template_kwargs)
        all_outputs.extend(outputs)

    # Save
    for r, output in zip(pending, all_outputs):
        full_text = output.outputs[0].text if output.outputs else ""
        thinking, raw_response = strip_thinking(full_text)
        predictions  = parse_response(raw_response)
        out = {
            "rec_id": r["_rec_id"],
            "trace_id": r.get("trace_id"),
            "predictions": predictions,
            "raw_response": raw_response,
        }
        if thinking:
            out["thinking"] = thinking
        with open(os.path.join(out_dir, f"{r['_rec_id']}.json"), "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Outputs saved to {out_dir}/")
    print(f"  Next: python eval/calculate_scores_yesno.py --pred_dir {out_dir}")


if __name__ == "__main__":
    main()
