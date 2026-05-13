"""
baselines/who&when/run_who_and_when_vllm.py

Who&When adaptation for MAST benchmark (vLLM / open-source models).

Adapts the prompting strategies from:
  "Which Agent Causes Task Failures and When?"  arXiv:2505.00212
  https://github.com/mingyin1/Agents_Failure_Attribution
to MAST's trace-level multi-label task over a 13-category taxonomy
(1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.6, 3.1, 3.2, 3.3).

Mapping:
  "Who" (which agent)  →  "What" (which MAST failure mode)
  "When" (which step)  →  no step output — MAST is trace-level only

Variants implemented here:
  w1   — All-at-Once   : full trace in one call, multi-label 13-bit output
  w2   — Step-by-Step  : per-step prompt with cumulative context, OR-aggregate

Changes from original Who&When (utils.py), kept minimal so the baseline is
a faithful single-error -> multi-error adaptation rather than a new method:
  W1: single-error free-text answer -> multi-label 13-bit yes/no output keyed
      by MAST's 13 failure modes; task description retained verbatim from the
      original prompt structure (extracted from steps[0]); ground_truth dropped
      (MAST evaluation is reference-free).
  W2: per-step Yes/No free text -> multi-label 13-bit yes/no list; early-exit
      on first "Yes" removed (multi-label requires scanning all steps); task
      description and the original's "avoid being overly critical" calibration
      sentence retained verbatim. ground_truth dropped. Per-step results are
      OR-aggregated to a trace-level prediction since MAST ground truth is
      trace-level.
  W3: not adapted (binary search efficiency claim breaks under multi-label;
      see paper/baseline_who_and_when.tex for the argument).

Output schema matches eval/run_eval_yesno_vllm.py so
eval/calculate_scores_yesno.py works unchanged:
  {
    "rec_id":         "0000",
    "trace_id":       3,
    "predictions":    {"1.1": 0/1, ..., "3.3": 0/1},
    "raw_response":   "...",
    "variant":        "w1" | "w2",
    "meta":           { ...per-variant counters... }
  }

Usage (run from MAST/):

  # W1 — full trace, single call per trace
  CUDA_VISIBLE_DEVICES=0,1 python "baselines/who&when/run_who_and_when_vllm.py" \\
      --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \\
      --variant w1 \\
      --output_dir "baselines/who&when/outputs"

  # W2 — one call per step, no early exit, OR-aggregate
  CUDA_VISIBLE_DEVICES=0,1 python "baselines/who&when/run_who_and_when_vllm.py" \\
      --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \\
      --variant w2 \\
      --output_dir "baselines/who&when/outputs"

  # Score (run from MAST/)
  python eval/calculate_scores_yesno.py \\
      --annotation data/annotation/annotation_ag2_filtered.jsonl \\
      --pred_dir "baselines/who&when/outputs/<model_tag>-yesno-who_and_when_<variant>"
"""

import os
import re
import json
import argparse

from vllm import LLM, SamplingParams
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Defaults / taxonomy
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAST_ROOT = os.path.dirname(os.path.dirname(_HERE))
_TAXONOMY_DIR = os.path.join(_MAST_ROOT, "taxonomy_definitions_examples")

DEFINITIONS = open(os.path.join(_TAXONOMY_DIR, "definitions.txt")).read()
EXAMPLES    = open(os.path.join(_TAXONOMY_DIR, "examples.txt")).read()

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
# Trace formatting
# ---------------------------------------------------------------------------

def format_trace(steps: list) -> str:
    lines = []
    for s in steps:
        lines.append(f"[{s['id']}]")
        lines.append(s["content"].strip())
        lines.append("")
    return "\n".join(lines)


def extract_task_description(steps: list, max_len: int = 600) -> str:
    """Extract the task statement from the first step's content.

    MAST records have no top-level task field; the task lives in
    steps[0]["content"], typically as one of:
      "[mathproxyagent]\\nProblem: <task>"   (AG2 math traces)
      "[assistant]\\n<task>"                  (AG2 MCQ traces)
    We strip the leading [agent_role] tag line and an optional
    "Problem:"/"Question:" header, then truncate. The original Who&When
    prompts include this as a separate {problem} field; surfacing it here
    restores that grounding.
    """
    if not steps:
        return "(task unknown)"
    text = (steps[0].get("content") or "").strip()
    text = re.sub(r"^\s*\[[^\]]+\]\s*\n?", "", text)
    text = re.sub(r"^\s*(?:Problem|Question)\s*:\s*\n?", "", text, flags=re.IGNORECASE)
    text = text.strip()
    if not text:
        return "(task unknown)"
    return text[:max_len]


# ---------------------------------------------------------------------------
# W1 prompt — full trace, all 13 modes at once
# ---------------------------------------------------------------------------

def get_w1_prompt(trace_text: str, task_description: str) -> str:
    return (
        "You are an AI assistant tasked with analyzing a multiagent system trace when solving a real-world problem.\n"
        f"The problem is: {task_description}\n\n"
        "Read the failure mode definitions and examples below carefully before examining the trace.\n\n"
        "FAILURE MODE DEFINITIONS:\n"
        f"{DEFINITIONS}\n\n"
        "EXAMPLES OF FAILURE MODES:\n"
        f"{EXAMPLES}\n\n"
        "Identify which failure modes from the taxonomy above are present in the trace, and explain the reason for each.\n\n"
        "Here is the trace:\n"
        f"{trace_text}\n\n"
        "Based on this trace, please predict the following:\n"
        "1. Which of the 13 failure modes are present in the trace. Multiple failure modes can and do co-occur in the same trace — be thorough and mark all that apply. Only mark a failure mode if you can identify a specific example of it in the trace.\n"
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


# ---------------------------------------------------------------------------
# W2 prompt — one step at a time, multi-label per step
# ---------------------------------------------------------------------------

def get_w2_step_prompt(cumulative_text: str, step_id: str,
                       step_num: int, total_steps: int,
                       task_description: str) -> str:
    return (
        "You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multiagent system trace aimed at solving a real-world problem.\n"
        f"The problem being addressed is: {task_description}\n\n"
        "Read the failure mode definitions and examples below carefully before examining the step.\n\n"
        "FAILURE MODE DEFINITIONS:\n"
        f"{DEFINITIONS}\n\n"
        "EXAMPLES OF FAILURE MODES:\n"
        f"{EXAMPLES}\n\n"
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
# Parsing
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> tuple:
    """Remove thinking block; return (thinking_text, remaining_text).

    Handles three cases:
      1. balanced <think>...</think>
      2. dangling </think> (open tag dropped by tokenizer)
      3. open <think> with no </think> (reasoning truncated mid-stream — common
         when max_tokens is hit while the model is still in its think block)
    """
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
    """Remove gpt-oss Harmony channel markers and keep only the 'final' channel.

    gpt-oss models emit channels like 'assistantanalysis<text>assistantfinal<answer>'.
    For MAST's regex-based parsing this is mostly survivable, but if the same
    '1.1: yes' line appears in both analysis and final channels with different
    values we'd grab the first one (wrong). Strip non-final channels here.
    """
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
# Cumulative-context truncation for W2
# ---------------------------------------------------------------------------

def build_cumulative_text(steps: list, upto_idx: int, max_chars: int) -> str:
    """Join steps[0..upto_idx-1] (i.e. history BEFORE the current step) into a
    text block. If too long, drop the oldest steps until it fits.

    Each entry shown as [step_id]\\n<content>\\n. Returns empty string if upto_idx==0.
    """
    if upto_idx <= 0:
        return ""
    items = []
    for s in steps[:upto_idx]:
        items.append(f"[{s['id']}]\n{s['content'].strip()}\n")
    text = "\n".join(items)
    if len(text) <= max_chars:
        return text
    # drop oldest until under budget
    keep_from = 0
    while keep_from < len(items) - 1:
        keep_from += 1
        truncated = "\n".join(items[keep_from:])
        if len(truncated) <= max_chars:
            return f"[... {keep_from} earlier step(s) omitted ...]\n\n" + truncated
    return f"[... {len(items) - 1} earlier step(s) omitted ...]\n\n" + items[-1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="MAST Who&When adaptation — vLLM (variants W1, W2)")
    ap.add_argument("--variant", choices=["w1", "w2"], required=True,
                    help="Who&When variant to run")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Model path or HuggingFace ID (default: {DEFAULT_MODEL})")
    ap.add_argument("--tp", type=int, default=None,
                    help="Tensor parallel size (default: auto-detect from CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--input", default="data/annotation/annotation_ag2_filtered.jsonl",
                    help="Path to MAST annotation jsonl")
    ap.add_argument("--output_dir", default="baselines/who&when/outputs",
                    help="Root output directory")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Inference batch size")
    ap.add_argument("--max_tokens", type=int, default=2000,
                    help="Max new tokens per response. Auto-bumped to 16000 for "
                         "reasoning models (QwenLong / *-L1-* / gpt-oss / DeepSeek-R1 / QwQ) "
                         "unless explicitly set above 2000.")
    ap.add_argument("--max_model_len", type=int, default=108000,
                    help="Max context length for the model")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                    help="Fraction of GPU memory vLLM may use per device")
    ap.add_argument("--model_tag", type=str, default=None,
                    help="Override the model tag used in the output directory name")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Pass enable_thinking=True via chat_template_kwargs")
    ap.add_argument("--w2_max_history_chars", type=int, default=80000,
                    help="W2 only: char budget for cumulative history before current step")
    args = ap.parse_args()

    # Auto-detect tensor parallel size from CUDA_VISIBLE_DEVICES
    if args.tp is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        args.tp = len([x for x in cvd.split(",") if x.strip()]) if cvd.strip() else 1

    # Reasoning models burn most of max_tokens on hidden <think>/analysis text
    # before the answer block. 2000 truncates the answer mid-stream, which makes
    # parse_response find no '<mode>: yes/no' lines and silently default to all-zero.
    is_reasoning_model = bool(re.search(
        r"(qwenlong|-l1-|gpt-oss|deepseek-r1|qwq)", args.model, re.IGNORECASE))
    if is_reasoning_model and args.max_tokens <= 2000:
        print(f"[INFO] Reasoning model detected ({args.model}); "
              f"bumping max_tokens 2000 → 16000")
        args.max_tokens = 16000

    # Load records
    records = []
    with open(args.input) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            r["_rec_id"] = f"{idx:04d}"
            records.append(r)
    print(f"Loaded {len(records)} traces from {args.input}")

    model_tag = args.model_tag if args.model_tag else args.model.replace("/", "-")
    thinking_suffix = "-thinking" if args.enable_thinking else ""
    out_dir = os.path.join(
        args.output_dir,
        f"{model_tag}-yesno-who_and_when_{args.variant}{thinking_suffix}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Filter out already-completed records
    pending = [r for r in records
               if not os.path.exists(os.path.join(out_dir, f"{r['_rec_id']}.json"))]
    print(f"Pending: {len(pending)} (skipping {len(records) - len(pending)} already done)")
    if not pending:
        print("Nothing to do.")
        return

    # Load model
    print(f"Loading model: {args.model}  (tp={args.tp})")
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
               batch_size=args.batch_size, out_dir=out_dir)
    else:
        run_w2(records=pending, llm=llm, sampling=sampling,
               chat_template_kwargs=chat_template_kwargs,
               batch_size=args.batch_size, out_dir=out_dir,
               max_history_chars=args.w2_max_history_chars)

    print(f"\n✓ Outputs saved to {out_dir}/")
    print(f"  Next: python eval/calculate_scores_yesno.py --pred_dir \"{out_dir}\"")


# ---------------------------------------------------------------------------
# W1 runner
# ---------------------------------------------------------------------------

def run_w1(records, llm, sampling, chat_template_kwargs, batch_size, out_dir):
    conversations = []
    for r in records:
        trace_text = format_trace(r["steps"])
        task_desc  = extract_task_description(r["steps"])
        conversations.append([{"role": "user",
                               "content": get_w1_prompt(trace_text, task_desc)}])

    print(f"[W1] Running inference on {len(conversations)} traces (batch_size={batch_size})...")
    all_outputs = []
    for i in tqdm(range(0, len(conversations), batch_size)):
        batch = conversations[i : i + batch_size]
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


# ---------------------------------------------------------------------------
# W2 runner — one prompt per step, no early exit, OR-aggregate to trace level
# ---------------------------------------------------------------------------

def run_w2(records, llm, sampling, chat_template_kwargs, batch_size, out_dir,
           max_history_chars):
    # Build a flat list of (rec_idx, step_idx, conversation) so we can batch
    # across records and across steps in one pass.
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
            )
            flat.append((rec_idx, step_idx, [{"role": "user", "content": user}]))

    print(f"[W2] Running inference on {len(flat)} step-prompts "
          f"across {len(records)} traces (batch_size={batch_size})...")
    all_outputs = [None] * len(flat)
    for i in tqdm(range(0, len(flat), batch_size)):
        batch_items = flat[i : i + batch_size]
        batch_convs = [c for (_, _, c) in batch_items]
        outputs = llm.chat(batch_convs, sampling_params=sampling, use_tqdm=False,
                           chat_template_kwargs=chat_template_kwargs)
        for j, output in enumerate(outputs):
            all_outputs[i + j] = output

    # Group results back per record
    per_rec_step_preds = [[None] * len(r.get("steps", [])) for r in records]
    per_rec_step_raw   = [[None] * len(r.get("steps", [])) for r in records]
    per_rec_step_think = [[None] * len(r.get("steps", [])) for r in records]
    for (rec_idx, step_idx, _), output in zip(flat, all_outputs):
        full_text = output.outputs[0].text if output and output.outputs else ""
        thinking, raw_response = strip_thinking(full_text)
        per_rec_step_preds[rec_idx][step_idx] = parse_response(raw_response)
        per_rec_step_raw[rec_idx][step_idx]   = raw_response
        per_rec_step_think[rec_idx][step_idx] = thinking if thinking else None

    # OR-aggregate to trace level and write one file per record
    for rec_idx, r in enumerate(records):
        step_preds = per_rec_step_preds[rec_idx]
        trace_pred = {m: 0 for m in MAST_MODES}
        for sp in step_preds:
            if sp is None:
                continue
            for m in MAST_MODES:
                if sp.get(m, 0) == 1:
                    trace_pred[m] = 1

        # Per-step debug info
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
                "n_calls":        len(r.get("steps", [])),
                "n_steps":        len(r.get("steps", [])),
                "per_step":       per_step_meta,
            },
        }
        with open(os.path.join(out_dir, f"{r['_rec_id']}.json"), "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
