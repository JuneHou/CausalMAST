"""
eval_detect/run_eval_yesno_vllm.py — MAST yes/no evaluation, detection-oriented prompt.

Identical to eval/run_eval_yesno_vllm.py except:
  - Removes section A (freeform summary) — avoids conservative reasoning anchor
  - Removes task-completion question (old section B)
  - Goes straight to yes/no labels
  - Adds liberal threshold calibration sentence ("~4 failures per trace, err toward yes")

Usage (run from MAST/):
    python eval_detect/run_eval_yesno_vllm.py
    python eval_detect/run_eval_yesno_vllm.py --model /path/to/model --tp 4
"""

import os
import re
import json
import argparse

from vllm import LLM, SamplingParams
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

# ---------------------------------------------------------------------------
# Taxonomy definitions + examples
# ---------------------------------------------------------------------------

_TAXONOMY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "taxonomy_definitions_examples")
DEFINITIONS = open(os.path.join(_TAXONOMY_DIR, "definitions.txt")).read()
EXAMPLES    = open(os.path.join(_TAXONOMY_DIR, "examples.txt")).read()

MAST_MODES = ["1.1", "1.2", "1.3", "1.4", "1.5",
              "2.1", "2.2", "2.3", "2.4", "2.6",
              "3.1", "3.2", "3.3"]


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


def get_prompt(trace_text: str) -> str:
    return (
        "You are analyzing a multiagent system trace for failure modes and inefficiencies.\n"
        "Read the definitions and examples carefully before examining the trace.\n\n"
        "FAILURE MODE DEFINITIONS:\n"
        f"{DEFINITIONS}\n\n"
        "EXAMPLES OF FAILURE MODES:\n"
        f"{EXAMPLES}\n\n"
        "Now analyze the trace below. For each failure mode, answer yes or no.\n"
        "Multiple failure modes can and do co-occur in the same trace — be thorough and mark all that apply.\n"
        "Only mark a failure mode if you can identify a specific example of it in the trace.\n"
        "On average, a trace contains about 4 of these 13 failure modes. Err toward yes when you see any indication — do not require certainty.\n\n"
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
        "@@*** end of your answer ***\n\n"
        "An example answer is:\n"
        "1.1 no\n"
        "1.2 no\n"
        "1.3 yes\n"
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
    """Remove thinking block; return (thinking_text, remaining_text)."""
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
    ap = argparse.ArgumentParser(description="MAST yes/no evaluation — detection prompt, in-process vLLM")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Model path or HuggingFace ID (default: {DEFAULT_MODEL})")
    ap.add_argument("--tp", type=int, default=None,
                    help="Tensor parallel size (default: auto-detect from CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--input", default="data/annotation/annotation_ag2_filtered.jsonl",
                    help="Path to annotation_ag2_filtered.jsonl")
    ap.add_argument("--output_dir", default="outputs_detect",
                    help="Root output directory (default: outputs_detect/)")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Inference batch size (default: 8)")
    ap.add_argument("--max_tokens", type=int, default=8000,
                    help="Max new tokens per response (default: 8000)")
    ap.add_argument("--max_model_len", type=int, default=108000,
                    help="Max context length for the model (default: 108000)")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                    help="Fraction of GPU memory vLLM may use per device (default: 0.9)")
    ap.add_argument("--model_tag", type=str, default=None,
                    help="Override the model tag used in the output directory name")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Pass enable_thinking=True via chat_template_kwargs (for QwQ/Qwen3/DeepSeek-R1)")
    args = ap.parse_args()

    if args.tp is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd.strip():
            args.tp = len([x for x in cvd.split(",") if x.strip()])
        else:
            args.tp = 1

    records = []
    with open(args.input) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            r["_rec_id"] = f"{idx:04d}"
            records.append(r)
    print(f"Loaded {len(records)} traces from {args.input}")

    model_tag = args.model_tag if args.model_tag else args.model.replace("/", "-")
    thinking_suffix = "-thinking" if args.enable_thinking else ""
    out_dir = os.path.join(args.output_dir, f"{model_tag}-yesno-baseline{thinking_suffix}")
    os.makedirs(out_dir, exist_ok=True)

    pending = [r for r in records
               if not os.path.exists(os.path.join(out_dir, f"{r['_rec_id']}.json"))]
    print(f"Pending: {len(pending)} (skipping {len(records) - len(pending)} already done)")

    if not pending:
        print("Nothing to do.")
        return

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

    conversations = []
    for r in pending:
        trace_text = format_trace(r["steps"])
        conversations.append([{"role": "user", "content": get_prompt(trace_text)}])

    chat_template_kwargs = {"enable_thinking": True} if args.enable_thinking else {}

    print(f"Running inference (batch_size={args.batch_size}, enable_thinking={args.enable_thinking})...")
    all_outputs = []
    for i in tqdm(range(0, len(conversations), args.batch_size)):
        batch = conversations[i : i + args.batch_size]
        outputs = llm.chat(batch, sampling_params=sampling, use_tqdm=False,
                           chat_template_kwargs=chat_template_kwargs)
        all_outputs.extend(outputs)

    for r, output in zip(pending, all_outputs):
        full_text = output.outputs[0].text if output.outputs else ""
        thinking, raw_response = strip_thinking(full_text)
        predictions = parse_response(raw_response)
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
