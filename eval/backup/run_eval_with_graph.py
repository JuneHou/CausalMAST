"""
eval/run_eval_with_graph.py — E3: MAST yes/no evaluation with static causal graph guidance.

Single-pass. The validated causal graph is prepended to the prompt as static background
context before the yes/no questions. The model reads all edges upfront and uses them
during holistic error detection — graph is passive context, not dynamic injection.

Two graph sources (both plain JSON from causal_graph/outputs/):
  --causal_only  — 7 intervention-validated edges from effect_edges.json (weight = abs(delta))
  default        — bootstrap-stable edges from edge_stability.json with frequency >= threshold

Usage (run from MAST/):
    CUDA_VISIBLE_DEVICES=4,5 python eval/run_eval_with_graph.py
    CUDA_VISIBLE_DEVICES=4,5 python eval/run_eval_with_graph.py --causal_only
    CUDA_VISIBLE_DEVICES=4,5 python eval/run_eval_with_graph.py --edge_threshold 0.7
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Tuple

from vllm import LLM, SamplingParams
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL        = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
DEFAULT_EDGE_THRESHOLD = 0.5   # bootstrap stability frequency threshold

_EVAL_DIR    = Path(__file__).resolve().parent
_MAST_DIR    = _EVAL_DIR.parent
_GRAPH_DIR   = _MAST_DIR / "causal_graph" / "outputs"

DEFAULT_STABILITY_GRAPH = _GRAPH_DIR / "edge_stability.json"
DEFAULT_EFFECT_EDGES    = _GRAPH_DIR / "interventions" / "effect_edges.json"

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
    stability_graph: Path = DEFAULT_STABILITY_GRAPH,
    effect_edges: Path = DEFAULT_EFFECT_EDGES,
) -> List[Tuple[str, str, float]]:
    """
    Load causal edges from MAST graph outputs.

    causal_only=True  — intervention-validated edges from effect_edges.json (weight=abs(delta))
    default           — bootstrap-stable edges from edge_stability.json (weight=frequency)
                        with frequency >= threshold
    """
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
        edges = [
            (e["a"], e["b"], e["frequency"])
            for e in data["edges"]
            if e["frequency"] >= threshold
        ]

    edges.sort(key=lambda x: -x[2])
    return edges


def format_graph_guidance(edges: List[Tuple[str, str, float]]) -> str:
    """Format edges as a static guidance block for injection into the prompt."""
    if not edges:
        return ""
    lines = [
        "# Causal Error Patterns (data-driven, from prior trace analysis)",
        "The following causal relationships between MAST error types have been statistically",
        "validated. When you identify an error of type A in the trace, actively look for",
        "errors of type B, as B has been found to causally follow A.",
        "Higher strength values indicate stronger causal association.",
        "",
        "Format: [Source Error] → [Consequent Error]  (strength: X.XX)",
        "",
    ]
    for src, dst, w in edges:
        src_name = MAST_NAMES.get(src, src)
        dst_name = MAST_NAMES.get(dst, dst)
        lines.append(f"  {src} {src_name} → {dst} {dst_name}  (strength: {w:.2f})")
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


def parse_response(response: str) -> dict:
    cleaned = response.strip()
    if cleaned.startswith("@@"):
        cleaned = cleaned[2:]
    if cleaned.endswith("@@"):
        cleaned = cleaned[:-2]
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
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=2000)
    ap.add_argument("--max_model_len", type=int, default=108000)
    ap.add_argument("--causal_only", action="store_true",
                    help="Use only intervention-validated edges from effect_edges.json")
    ap.add_argument("--edge_threshold", type=float, default=DEFAULT_EDGE_THRESHOLD,
                    help="Min bootstrap stability frequency for default mode (default: 0.5)")
    ap.add_argument("--stability_graph", type=str, default=None)
    ap.add_argument("--effect_edges", type=str, default=None)
    ap.add_argument("--model_tag", type=str, default=None,
                    help="Override the model tag used in the output directory name")
    args = ap.parse_args()

    if args.tp is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        args.tp = len([x for x in cvd.split(",") if x.strip()]) if cvd.strip() else 1

    # Load graph
    stability_path = Path(args.stability_graph) if args.stability_graph else DEFAULT_STABILITY_GRAPH
    effect_path    = Path(args.effect_edges)    if args.effect_edges    else DEFAULT_EFFECT_EDGES
    edges = load_graph_edges(args.edge_threshold, args.causal_only, stability_path, effect_path)
    graph_guidance = format_graph_guidance(edges)
    mode_str = "causal_only" if args.causal_only else f"stability>={args.edge_threshold}"
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

    graph_tag = "causal_only" if args.causal_only else f"t{args.edge_threshold}"
    model_tag = args.model_tag if args.model_tag else args.model.replace("/", "-")
    out_dir   = os.path.join(args.output_dir, f"{model_tag}-yesno-with-graph-{graph_tag}")
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
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Build conversations
    conversations = []
    for r in pending:
        trace_text = format_trace(r["steps"])
        conversations.append([{"role": "user", "content": get_prompt(trace_text, graph_guidance)}])

    # Batch inference
    print(f"Running inference (batch_size={args.batch_size})...")
    all_outputs = []
    for i in tqdm(range(0, len(conversations), args.batch_size)):
        batch = conversations[i: i + args.batch_size]
        outputs = llm.chat(batch, sampling_params=sampling, use_tqdm=False)
        all_outputs.extend(outputs)

    # Save
    for r, output in zip(pending, all_outputs):
        raw_response = output.outputs[0].text if output.outputs else ""
        predictions  = parse_response(raw_response)
        out = {
            "rec_id": r["_rec_id"],
            "trace_id": r.get("trace_id"),
            "predictions": predictions,
            "raw_response": raw_response,
        }
        with open(os.path.join(out_dir, f"{r['_rec_id']}.json"), "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Outputs saved to {out_dir}/")
    print(f"  Next: python eval/calculate_scores_yesno.py --pred_dir {out_dir}")


if __name__ == "__main__":
    main()
