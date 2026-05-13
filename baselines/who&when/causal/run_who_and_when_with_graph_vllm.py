"""
baselines/who&when/causal/run_who_and_when_with_graph_vllm.py

+CG (one-pass causal graph guidance) applied to MAST's Who&When W1/W2 prompts.

Mirrors:
  - TRAIL's baselines/who_and_when/causal/run_who_and_when_with_graph_vllm.py
    (W&W +CG structural design)
  - MAST's eval/full_run_eval_with_graph.py (code(name) edge format,
    graph loading, two graph modes)

Differences vs the TRAIL W&W +CG file:
  - --span_index accepted but is a no-op on prompts (MAST has no location prediction);
    it only adds a _span_index suffix to the output directory name.
  - No "FINAL LEAF" rule (MAST has a flat 13-code taxonomy, not hierarchical).
  - No "Resource Abuse last instance" rule (no location).
  - Yes/no output format (matches MAST W&W baseline).
  - Graph modes restricted to {causal_only, corr_threshold}.

W1 injects the graph block once in the single full-trace prompt.
W2 injects the graph block in every per-step prompt.

Output naming:
    {model_tag}-yesno-who_and_when_{w1|w2}_graph_{causal_only|corr<value>}/

Usage (run from MAST/):
    CUDA_VISIBLE_DEVICES=0,1 python "baselines/who&when/causal/run_who_and_when_with_graph_vllm.py" \\
        --variant w1 --causal_only \\
        --output_dir "baselines/who&when/outputs"
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
# Paths / defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL          = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
DEFAULT_EDGE_THRESHOLD = 0.2

_HERE      = Path(__file__).resolve().parent
_WW_DIR    = _HERE.parent
_MAST_ROOT = _WW_DIR.parent.parent
_GRAPH_DIR = _MAST_ROOT / "causal_graph" / "outputs"

DEFAULT_EFFECT_EDGES = _GRAPH_DIR / "interventions" / "effect_edges.json"
DEFAULT_SUPPES_GRAPH = _GRAPH_DIR / "suppes_graph.json"

_TAXONOMY_DIR = _MAST_ROOT / "taxonomy_definitions_examples"
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
# Graph loading — causal_only / corr_threshold (mirrors MAST eval helpers)
# ---------------------------------------------------------------------------

def load_graph_edges(
    causal_only: bool = False,
    corr_threshold: float = 1.0,
    effect_edges: Path = DEFAULT_EFFECT_EDGES,
    suppes_graph: Path = DEFAULT_SUPPES_GRAPH,
) -> List[Tuple[str, str, float]]:
    """
    Two mutually-exclusive branches:
      causal_only=True       -> validated causal edges only, weight = |Delta|.
      corr_threshold < 1.0   -> UNION: (Suppes geomean >= corr_threshold)
                                 union (validated causal edges, regardless of geomean).
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
        raise ValueError(
            "Must specify --causal_only or --corr_threshold < 1.0 "
            "(MAST W&W +CG exposes only these two graph modes)."
        )
    edges.sort(key=lambda x: -x[2])
    return edges


def format_graph_guidance(
    edges: List[Tuple[str, str, float]],
    causal_only: bool = True,
) -> str:
    """Format edges as a guidance block using code(name) format."""
    if not edges:
        return ""
    if causal_only:
        lines = [
            "CAUSAL ERROR PATTERNS (intervention-validated):",
            "The following edges were validated via counterfactual patching experiments.",
            "When you identify error type A, actively look for error type B,",
            "as removing A causally reduces B's occurrence rate.",
            "Higher values indicate stronger causal effect.",
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
            "Score = geometric mean of precedence P(A precedes B | both occur) and probability-raising delta P(B|A)-P(B|not A).",
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
# Trace formatting + task extraction (verbatim from MAST W&W baseline)
# ---------------------------------------------------------------------------

def format_trace(steps: list) -> str:
    lines = []
    for s in steps:
        lines.append(f"[{s['id']}]")
        lines.append(s["content"].strip())
        lines.append("")
    return "\n".join(lines)


def extract_task_description(steps: list, max_len: int = 600) -> str:
    if not steps:
        return "(task unknown)"
    text = (steps[0].get("content") or "").strip()
    text = re.sub(r"^\s*\[[^\]]+\]\s*\n?", "", text)
    text = re.sub(r"^\s*(?:Problem|Question)\s*:\s*\n?", "", text, flags=re.IGNORECASE)
    text = text.strip()
    if not text:
        return "(task unknown)"
    return text[:max_len]


def build_cumulative_text(steps: list, upto_idx: int, max_chars: int) -> str:
    if upto_idx <= 0:
        return ""
    items = []
    for s in steps[:upto_idx]:
        items.append(f"[{s['id']}]\n{s['content'].strip()}\n")
    text = "\n".join(items)
    if len(text) <= max_chars:
        return text
    keep_from = 0
    while keep_from < len(items) - 1:
        keep_from += 1
        truncated = "\n".join(items[keep_from:])
        if len(truncated) <= max_chars:
            return f"[... {keep_from} earlier step(s) omitted ...]\n\n" + truncated
    return f"[... {len(items) - 1} earlier step(s) omitted ...]\n\n" + items[-1]


# ---------------------------------------------------------------------------
# Prompts with graph_guidance slot
# ---------------------------------------------------------------------------

def get_w1_prompt(trace_text: str, task_description: str, graph_guidance: str) -> str:
    graph_block = (graph_guidance + "\n") if graph_guidance else ""
    return (
        "You are an AI assistant tasked with analyzing a multiagent system trace when solving a real-world problem.\n"
        f"The problem is: {task_description}\n\n"
        "Read the failure mode definitions and examples below carefully before examining the trace.\n\n"
        "FAILURE MODE DEFINITIONS:\n"
        f"{DEFINITIONS}\n\n"
        "EXAMPLES OF FAILURE MODES:\n"
        f"{EXAMPLES}\n\n"
        f"{graph_block}"
        "Identify which failure modes from the taxonomy above are present in the trace, and explain the reason for each.\n\n"
        "Here is the trace:\n"
        f"{trace_text}\n\n"
        "Based on this trace, please predict the following:\n"
        "1. Which of the 13 failure modes are present in the trace. Multiple failure modes can and do co-occur in the same trace -- be thorough and mark all that apply. Only mark a failure mode if you can identify a specific example of it in the trace.\n"
        "2. A brief reason for each predicted failure mode.\n"
        "3. Whether the task is successfully completed or not.\n\n"
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
        "3.3 no\n"
    )


def get_w2_step_prompt(cumulative_text: str, step_id: str,
                       step_num: int, total_steps: int,
                       task_description: str, graph_guidance: str) -> str:
    graph_block = (graph_guidance + "\n") if graph_guidance else ""
    return (
        "You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multiagent system trace aimed at solving a real-world problem.\n"
        f"The problem being addressed is: {task_description}\n\n"
        "Read the failure mode definitions and examples below carefully before examining the step.\n\n"
        "FAILURE MODE DEFINITIONS:\n"
        f"{DEFINITIONS}\n\n"
        "EXAMPLES OF FAILURE MODES:\n"
        f"{EXAMPLES}\n\n"
        f"{graph_block}"
        f"Conversation history up to and including the CURRENT step (step {step_num} of {total_steps}):\n"
        f"{cumulative_text}\n\n"
        f"The most recent step (step {step_num}) is [{step_id}].\n\n"
        "Your task is to determine whether this most recent step contains any failure modes from the 13 categories above. A single step may exhibit zero, one, or multiple failure modes.\n\n"
        "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process.\n\n"
        "Answer between the @@ symbols exactly as shown:\n"
        "*** begin of things you should answer *** @@\n"
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
        "@@*** end of your answer ***\n"
    )


# ---------------------------------------------------------------------------
# Response parsing (verbatim from MAST W&W baseline)
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> tuple:
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), (text[:match.start()] + text[match.end():]).strip()
    idx = text.find('</think>')
    if idx != -1:
        return text[:idx].strip(), text[idx + 8:].strip()
    open_idx = text.find('<think>')
    if open_idx != -1:
        return text[open_idx + 7:].strip(), text[:open_idx].strip()
    return "", text.strip()


def strip_harmony(text: str) -> str:
    if "assistantfinal" in text.lower():
        m = re.search(r"assistantfinal\b\s*", text, re.IGNORECASE)
        if m:
            text = text[m.end():]
    text = re.sub(r"assistant(?:analysis|commentary|final)\b\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|channel\|>(?:analysis|commentary|final)<\|message\|>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|(?:start|end|return)\|>", "", text, flags=re.IGNORECASE)
    return text


def parse_response(response: str) -> dict:
    cleaned = strip_harmony(response).strip()
    if cleaned.startswith("@@"):
        cleaned = cleaned[2:]
    if cleaned.endswith("@@"):
        cleaned = cleaned[:-2]
    cleaned = re.sub(r'\*\*(yes|no)\*\*', r'\1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<(yes|no)>', r'\1', cleaned, flags=re.IGNORECASE)

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
# Variant runners
# ---------------------------------------------------------------------------

def run_w1(records, llm, sampling, chat_template_kwargs, batch_size, out_dir, graph_guidance):
    conversations = []
    for r in records:
        trace_text = format_trace(r["steps"])
        task_desc  = extract_task_description(r["steps"])
        conversations.append([{"role": "user",
                               "content": get_w1_prompt(trace_text, task_desc, graph_guidance)}])

    print(f"[W1] Running inference on {len(conversations)} traces (batch_size={batch_size})...")
    all_outputs = []
    for i in tqdm(range(0, len(conversations), batch_size)):
        batch = conversations[i: i + batch_size]
        outputs = llm.chat(batch, sampling_params=sampling, use_tqdm=False,
                           chat_template_kwargs=chat_template_kwargs)
        all_outputs.extend(outputs)

    for r, output in zip(records, all_outputs):
        full_text = output.outputs[0].text if output.outputs else ""
        thinking, raw_response = strip_thinking(full_text)
        predictions = parse_response(raw_response)
        out = {
            "rec_id":       r["_rec_id"],
            "trace_id":     r.get("trace_id"),
            "predictions":  predictions,
            "raw_response": raw_response,
            "variant":      "w1",
            "meta":         {"n_calls": 1, "n_steps": len(r.get("steps", []))},
        }
        if thinking:
            out["thinking"] = thinking
        with open(os.path.join(out_dir, f"{r['_rec_id']}.json"), "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


def run_w2(records, llm, sampling, chat_template_kwargs, batch_size, out_dir,
           max_history_chars, graph_guidance):
    flat = []
    for rec_idx, r in enumerate(records):
        steps = r.get("steps", [])
        n_steps = len(steps)
        task_desc = extract_task_description(steps)
        for step_idx, step in enumerate(steps):
            cumulative = build_cumulative_text(steps, step_idx + 1, max_history_chars)
            user = get_w2_step_prompt(
                cumulative_text=cumulative,
                step_id=step["id"],
                step_num=step_idx + 1,
                total_steps=n_steps,
                task_description=task_desc,
                graph_guidance=graph_guidance,
            )
            flat.append((rec_idx, step_idx, [{"role": "user", "content": user}]))

    print(f"[W2] Running inference on {len(flat)} step-prompts "
          f"across {len(records)} traces (batch_size={batch_size})...")
    all_outputs = [None] * len(flat)
    for i in tqdm(range(0, len(flat), batch_size)):
        batch_items = flat[i: i + batch_size]
        batch_convs = [c for (_, _, c) in batch_items]
        outputs = llm.chat(batch_convs, sampling_params=sampling, use_tqdm=False,
                           chat_template_kwargs=chat_template_kwargs)
        for j, output in enumerate(outputs):
            all_outputs[i + j] = output

    per_rec_step_preds = [[None] * len(r.get("steps", [])) for r in records]
    per_rec_step_raw   = [[None] * len(r.get("steps", [])) for r in records]
    per_rec_step_think = [[None] * len(r.get("steps", [])) for r in records]
    for (rec_idx, step_idx, _), output in zip(flat, all_outputs):
        full_text = output.outputs[0].text if output and output.outputs else ""
        thinking, raw_response = strip_thinking(full_text)
        per_rec_step_preds[rec_idx][step_idx] = parse_response(raw_response)
        per_rec_step_raw[rec_idx][step_idx]   = raw_response
        per_rec_step_think[rec_idx][step_idx] = thinking if thinking else None

    for rec_idx, r in enumerate(records):
        step_preds = per_rec_step_preds[rec_idx]
        trace_pred = {m: 0 for m in MAST_MODES}
        for sp in step_preds:
            if sp is None:
                continue
            for m in MAST_MODES:
                if sp.get(m, 0) == 1:
                    trace_pred[m] = 1

        per_step_meta = []
        for step_idx, step in enumerate(r.get("steps", [])):
            entry = {
                "step_idx":    step_idx,
                "step_id":     step["id"],
                "predictions": step_preds[step_idx] or {m: 0 for m in MAST_MODES},
            }
            if per_rec_step_think[rec_idx][step_idx]:
                entry["thinking"] = per_rec_step_think[rec_idx][step_idx]
            per_step_meta.append(entry)

        out = {
            "rec_id":       r["_rec_id"],
            "trace_id":     r.get("trace_id"),
            "predictions":  trace_pred,
            "raw_response": "\n\n---\n\n".join(
                f"[step {i}] {raw}" for i, raw in enumerate(per_rec_step_raw[rec_idx])
                if raw is not None
            ),
            "variant":      "w2",
            "meta": {
                "n_calls":  len(r.get("steps", [])),
                "n_steps":  len(r.get("steps", [])),
                "per_step": per_step_meta,
            },
        }
        with open(os.path.join(out_dir, f"{r['_rec_id']}.json"), "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="MAST W&W + causal graph guidance (+CG) via vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--variant", choices=["w1", "w2"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tp", type=int, default=None,
                    help="Tensor parallel size (default: auto-detect from CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--input", default="data/annotation/annotation_ag2_filtered.jsonl")
    ap.add_argument("--output_dir", default="baselines/who&when/outputs")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=2000)
    ap.add_argument("--max_model_len", type=int, default=108000)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--model_tag", type=str, default=None)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--span_index", action="store_true",
                    help="Accepted for CLI parity with +GI; no-op on prompts "
                         "(MAST has no location prediction). Adds _span_index suffix to output dir.")
    ap.add_argument("--w2_max_history_chars", type=int, default=80000)
    ap.add_argument("--causal_only", action="store_true",
                    help="Use only intervention-validated causal edges.")
    ap.add_argument("--corr_threshold", type=float, default=1.0,
                    help="If < 1.0, use UNION ablation: (Suppes geomean >= tau) union "
                         "(validated causal edges). Mutually exclusive with --causal_only.")
    ap.add_argument("--effect_edges", type=str, default=None)
    ap.add_argument("--suppes_graph", type=str, default=None)
    args = ap.parse_args()

    if args.tp is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        args.tp = len([x for x in cvd.split(",") if x.strip()]) if cvd.strip() else 1

    is_reasoning_model = bool(re.search(r"(qwenlong|-l1-|gpt-oss|deepseek-r1|qwq)",
                                        args.model, re.IGNORECASE))
    if is_reasoning_model and args.max_tokens <= 2000:
        print(f"[INFO] Reasoning model detected ({args.model}); bumping max_tokens 2000 -> 16000")
        args.max_tokens = 16000

    if sum([args.causal_only, args.corr_threshold < 1.0]) != 1:
        ap.error("Must specify exactly one of --causal_only or --corr_threshold < 1.0")

    effect_path = Path(args.effect_edges) if args.effect_edges else DEFAULT_EFFECT_EDGES
    suppes_path = Path(args.suppes_graph) if args.suppes_graph else DEFAULT_SUPPES_GRAPH
    edges = load_graph_edges(
        causal_only    = args.causal_only,
        corr_threshold = args.corr_threshold,
        effect_edges   = effect_path,
        suppes_graph   = suppes_path,
    )
    graph_guidance = format_graph_guidance(edges, causal_only=args.causal_only)
    graph_tag = "causal_only" if args.causal_only else f"corr{args.corr_threshold}"
    print(f"Graph: {len(edges)} edges ({graph_tag})")
    for src, dst, w in edges[:15]:
        print(f"  {src}({MAST_NAMES[src]}) -> {dst}({MAST_NAMES[dst]})  ({w:.3f})")
    if len(edges) > 15:
        print(f"  ... and {len(edges) - 15} more")

    records = []
    with open(args.input) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            r["_rec_id"] = f"{idx:04d}"
            records.append(r)
    print(f"\nLoaded {len(records)} traces from {args.input}")

    model_tag = args.model_tag if args.model_tag else args.model.replace("/", "-")
    thinking_suffix   = "-thinking" if args.enable_thinking else ""
    span_index_suffix = "_span_index" if args.span_index else ""
    out_dir = os.path.join(
        args.output_dir,
        f"{model_tag}-yesno-who_and_when_{args.variant}_graph_{graph_tag}{thinking_suffix}{span_index_suffix}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    pending = [r for r in records
               if not os.path.exists(os.path.join(out_dir, f"{r['_rec_id']}.json"))]
    print(f"Pending: {len(pending)} (skipping {len(records) - len(pending)} already done)")
    if not pending:
        print("Nothing to do.")
        return

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

    if args.variant == "w1":
        run_w1(records=pending, llm=llm, sampling=sampling,
               chat_template_kwargs=chat_template_kwargs,
               batch_size=args.batch_size, out_dir=out_dir,
               graph_guidance=graph_guidance)
    else:
        run_w2(records=pending, llm=llm, sampling=sampling,
               chat_template_kwargs=chat_template_kwargs,
               batch_size=args.batch_size, out_dir=out_dir,
               max_history_chars=args.w2_max_history_chars,
               graph_guidance=graph_guidance)

    print(f"\nOutputs saved to {out_dir}/")
    print(f"  Next: python eval/calculate_scores_yesno.py --pred_dir \"{out_dir}\"")


if __name__ == "__main__":
    main()
