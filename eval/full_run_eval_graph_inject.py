"""
eval/code_name/run_eval_graph_inject.py — E4 variant: graph edges use code(name) format.

Identical to eval/run_eval_graph_inject.py (2-pass dynamic injection) except edges are
formatted as:
  1.1(Disobey Task Specification) -> 3.3(No or Incorrect Verification)  [strength: X.XX]
The separate category lookup table in Pass 2 is also dropped since names are inline.

Usage (run from MAST/):
    CUDA_VISIBLE_DEVICES=4,5 python eval/code_name/run_eval_graph_inject.py --causal_only
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from vllm import LLM, SamplingParams
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL               = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
DEFAULT_EDGE_THRESHOLD      = 0.5    # bootstrap stability frequency for default mode
DEFAULT_PROPAGATION_THRESHOLD = 0.1  # min boosted score to trigger Pass 2 for a target

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
# Graph loading
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
    """
    Hard-binary confidence propagation.
    Returns filtered edges: src ∈ detected, dst ∉ detected,
    and boosted_score(dst) = Σ edge_weight > threshold.
    """
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
    """Standard baseline yes/no prompt — same structure as run_eval_yesno_vllm.py."""
    return (
        "You are analyzing a multiagent system trace for failure modes and inefficiencies.\n"
        "Read the definitions and examples carefully before examining the trace.\n\n"
        "FAILURE MODE DEFINITIONS:\n"
        f"{DEFINITIONS}\n\n"
        "EXAMPLES OF FAILURE MODES:\n"
        f"{EXAMPLES}\n\n"
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


def get_pass2_prompt(
    trace_text: str,
    pass1_detected: List[str],
    filtered_edges: List[Tuple[str, str, float]],
) -> str:
    """
    Targeted second-pass prompt. Only asks about the categories that are
    statistically likely given Pass 1 detections (dst nodes in filtered_edges).
    """
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
    """Parse full yes/no response for all 13 MAST categories."""
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
    """Parse Pass 2 response — only extract yes/no for target categories."""
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
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="E4: MAST yes/no with dynamic 2-pass graph injection")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--input", default="data/annotation/annotation_ag2_filtered.jsonl")
    ap.add_argument("--output_dir", default="outputs_full")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--max_model_len", type=int, default=108000)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                    help="Fraction of GPU memory vLLM may use per device (default: 0.9)")
    ap.add_argument("--causal_only", action="store_true",
                    help="Use only intervention-validated edges from effect_edges.json")
    ap.add_argument("--edge_threshold", type=float, default=DEFAULT_EDGE_THRESHOLD,
                    help="Min bootstrap stability frequency for default mode (default: 0.5)")
    ap.add_argument("--propagation_threshold", type=float, default=DEFAULT_PROPAGATION_THRESHOLD,
                    help="Min boosted score to trigger Pass 2 for a target category (default: 0.1)")
    ap.add_argument("--stability_graph", type=str, default=None)
    ap.add_argument("--effect_edges", type=str, default=None)
    ap.add_argument("--suppes_graph", type=str, default=None)
    ap.add_argument("--model_tag", type=str, default=None,
                    help="Override the model tag used in the output directory name")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Pass enable_thinking=True via chat_template_kwargs (for QwQ/Qwen3/DeepSeek-R1)")
    args = ap.parse_args()

    if args.tp is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        args.tp = len([x for x in cvd.split(",") if x.strip()]) if cvd.strip() else 1

    # Load graph
    stability_path = Path(args.stability_graph) if args.stability_graph else DEFAULT_STABILITY_GRAPH
    effect_path    = Path(args.effect_edges)    if args.effect_edges    else DEFAULT_EFFECT_EDGES
    suppes_path    = Path(args.suppes_graph)    if args.suppes_graph    else DEFAULT_SUPPES_GRAPH
    edges = load_graph_edges(args.edge_threshold, args.causal_only, stability_path, effect_path, suppes_path)
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
    thinking_suffix = "-thinking" if args.enable_thinking else ""
    out_dir   = os.path.join(args.output_dir, f"{model_tag}-yesno-graph-inject-codename-{graph_tag}{thinking_suffix}")
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

    chat_template_kwargs = {"enable_thinking": True} if args.enable_thinking else {}

    # ------------------------------------------------------------------
    # Pass 1 — batch all pending traces
    # ------------------------------------------------------------------
    print(f"\n--- Pass 1: holistic yes/no detection ({len(pending)} traces, enable_thinking={args.enable_thinking}) ---")
    p1_conversations = [
        [{"role": "user", "content": get_pass1_prompt(format_trace(r["steps"]))}]
        for r in pending
    ]
    p1_outputs = []
    for i in tqdm(range(0, len(p1_conversations), args.batch_size)):
        batch = p1_conversations[i: i + args.batch_size]
        outputs = llm.chat(batch, sampling_params=sampling, use_tqdm=False,
                           chat_template_kwargs=chat_template_kwargs)
        p1_outputs.extend(outputs)

    p1_predictions = []
    for output in p1_outputs:
        full_text = output.outputs[0].text if output.outputs else ""
        _, visible = strip_thinking(full_text)
        p1_predictions.append(parse_response(visible))

    # ------------------------------------------------------------------
    # Propagation — determine which traces need Pass 2
    # ------------------------------------------------------------------
    pass2_needed = []   # list of (record, p1_pred, filtered_edges)
    for r, p1_pred in zip(pending, p1_predictions):
        detected = [m for m, v in p1_pred.items() if v == 1]
        if detected and edges:
            filtered = propagate_confidence(detected, edges, args.propagation_threshold)
            if filtered:
                pass2_needed.append((r, p1_pred, filtered))

    print(f"\n--- Pass 2: graph injection ({len(pass2_needed)}/{len(pending)} traces triggered) ---")

    # ------------------------------------------------------------------
    # Pass 2 — batch only triggered traces
    # ------------------------------------------------------------------
    p2_results: Dict[str, Dict[str, int]] = {}  # rec_id → upgraded predictions

    if pass2_needed:
        p2_conversations = [
            [{"role": "user", "content": get_pass2_prompt(
                format_trace(r["steps"]),
                [m for m, v in p1_pred.items() if v == 1],
                filtered,
            )}]
            for r, p1_pred, filtered in pass2_needed
        ]
        p2_outputs = []
        for i in tqdm(range(0, len(p2_conversations), args.batch_size)):
            batch = p2_conversations[i: i + args.batch_size]
            outputs = llm.chat(batch, sampling_params=sampling, use_tqdm=False,
                               chat_template_kwargs=chat_template_kwargs)
            p2_outputs.extend(outputs)

        for (r, p1_pred, filtered), output in zip(pass2_needed, p2_outputs):
            full_text = output.outputs[0].text if output.outputs else ""
            _, raw = strip_thinking(full_text)
            target_cats = list(dict.fromkeys(dst for _, dst, _ in filtered))
            p2_pred = parse_pass2_response(raw, target_cats)
            p2_results[r["_rec_id"]] = p2_pred

    # ------------------------------------------------------------------
    # Merge Pass 1 + Pass 2, save results
    # ------------------------------------------------------------------
    for r, p1_pred, p1_output in zip(pending, p1_predictions, p1_outputs):
        rec_id = r["_rec_id"]
        merged = dict(p1_pred)

        # Pass 2 can only upgrade no→yes (never downgrade a detected error)
        p2_upgrades = p2_results.get(rec_id, {})
        for cat, val in p2_upgrades.items():
            if val == 1 and merged.get(cat, 0) == 0:
                merged[cat] = 1

        p2_triggered = rec_id in p2_results
        full_p1_text = p1_output.outputs[0].text if p1_output.outputs else ""
        p1_thinking, p1_raw = strip_thinking(full_p1_text)
        out = {
            "rec_id": rec_id,
            "trace_id": r.get("trace_id"),
            "predictions": merged,
            "raw_response": p1_raw,
            "pass2_triggered": p2_triggered,
            "pass2_upgrades": {k: v for k, v in p2_upgrades.items() if v == 1 and p1_pred.get(k, 0) == 0},
        }
        if p1_thinking:
            out["thinking"] = p1_thinking
        with open(os.path.join(out_dir, f"{rec_id}.json"), "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    n_upgraded = sum(
        1 for r in pending
        if any(v == 1 for v in p2_results.get(r["_rec_id"], {}).values())
    )
    print(f"\n✓ Outputs saved to {out_dir}/")
    print(f"  Pass 2 triggered: {len(pass2_needed)} traces, {n_upgraded} with upgrades")
    print(f"  Next: python eval/calculate_scores_yesno.py --pred_dir {out_dir}")


if __name__ == "__main__":
    main()
