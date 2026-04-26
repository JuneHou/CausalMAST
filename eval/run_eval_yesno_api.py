"""
eval/run_eval_yesno_api.py — Baseline yes/no evaluation using litellm (API models).

Prompt format: definitions before trace, conservative annotation standard (matching
run_eval_with_graph_api.py baseline — "only mark if you can identify a specific example").
Supports --sample_indices to run on a stratified subset (e.g. 100 traces for o1).

Usage (run from MAST/):
    python eval/run_eval_yesno_api.py --model openai/gpt-4o --max_workers 5
    python eval/run_eval_yesno_api.py --model openai/o1 \
        --sample_indices data/o1_sample_indices.json \
        --output_dir outputs_o1
"""

import os
import re
import json
import time
import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm
from litellm import completion, ContextWindowExceededError, RateLimitError
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openai/o1"

_EVAL_DIR     = Path(__file__).resolve().parent
_MAST_DIR     = _EVAL_DIR.parent
_TAXONOMY_DIR = _MAST_DIR / "taxonomy_definitions_examples"
DEFINITIONS   = (_TAXONOMY_DIR / "definitions.txt").read_text()
EXAMPLES      = (_TAXONOMY_DIR / "examples.txt").read_text()

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

def call_llm(prompt: str, model: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    is_reasoning = any(x in model for x in ("o1", "o3", "o4", "anthropic", "gemini-2.5"))
    if is_reasoning:
        params = {
            "messages": messages, "model": model,
            "reasoning_effort": "high", "drop_params": True,
        }
    else:
        params = {
            "messages": messages, "model": model,
            "temperature": 0.0, "top_p": 1,
            "max_completion_tokens": 2000,
            "reasoning_effort": None, "drop_params": True,
        }
    for attempt in range(5):
        try:
            response = completion(**params)
            content = response.choices[0].message.content
            return content if content is not None else ""
        except RateLimitError:
            wait = 60 * (2 ** attempt)
            print(f"Rate limit (attempt {attempt+1}/5): sleeping {wait}s...")
            time.sleep(wait)
    raise RateLimitError("Exceeded 5 retries due to rate limiting")


def process_record(r: dict, output_dir: str, model: str) -> None:
    rec_id = r["_rec_id"]
    output_file = os.path.join(output_dir, f"{rec_id}.json")
    if os.path.exists(output_file):
        return

    trace_text = format_trace(r["steps"])
    prompt = get_prompt(trace_text)
    raw_response = ""
    try:
        raw_response = call_llm(prompt, model)
        predictions = parse_response(raw_response)
    except ContextWindowExceededError:
        predictions = {m: 0 for m in MAST_MODES}
        raw_response = "ERROR: context window exceeded"
    except Exception as e:
        predictions = {m: 0 for m in MAST_MODES}
        raw_response = f"ERROR: {type(e).__name__}: {e}"
        print(f"[{rec_id}] {raw_response}")

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
    ap = argparse.ArgumentParser(description="MAST yes/no baseline evaluation (API models)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="LiteLLM model string (default: openai/o1)")
    ap.add_argument("--input", default="data/annotation/annotation_ag2_filtered.jsonl")
    ap.add_argument("--output_dir", default="outputs_o1")
    ap.add_argument("--max_workers", type=int, default=1,
                    help="Parallel workers (default: 1; use 1 for o1)")
    ap.add_argument("--sample_indices", type=str, default=None,
                    help="Path to o1_sample_indices.json to run on a subset of traces")
    ap.add_argument("--model_tag", type=str, default=None,
                    help="Override model tag in output directory name")
    args = ap.parse_args()

    records = []
    with open(args.input) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            r["_rec_id"] = f"{idx:04d}"
            records.append(r)
    print(f"Loaded {len(records)} traces from {args.input}")

    if args.sample_indices:
        with open(args.sample_indices) as f:
            sample = json.load(f)
        keep = set(sample["indices"])
        records = [r for i, r in enumerate(records) if i in keep]
        print(f"Filtered to {len(records)} traces from {args.sample_indices}")

    model_tag = args.model_tag if args.model_tag else args.model.replace("/", "-")
    out_dir = os.path.join(args.output_dir, f"{model_tag}-yesno-baseline")
    os.makedirs(out_dir, exist_ok=True)

    pending = [r for r in records
               if not os.path.exists(os.path.join(out_dir, f"{r['_rec_id']}.json"))]
    print(f"Pending: {len(pending)} (skipping {len(records) - len(pending)} already done)")
    if not pending:
        print("Nothing to do.")
        return

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_record, r, out_dir, args.model)
            for r in pending
        ]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    print(f"\n✓ Outputs saved to {out_dir}/")
    print(f"  Next: python eval/calculate_scores_yesno.py --pred_dir {out_dir}")


if __name__ == "__main__":
    litellm.drop_params = True
    main()
