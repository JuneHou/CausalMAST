"""
eval/run_eval_yesno.py — Baseline yes/no error-type prediction for MAST-AG2.

Replicates the original MAST llm_judge_pipeline.ipynb evaluation using the same
prompt format (yes/no per category between @@ delimiters), so that results are
directly comparable to the MAST paper baseline (which used o1).

Ground truth: mast_annotation field in annotation_ag2_filtered.jsonl (human labels).
Output: one JSON file per trace in outputs/{model_tag}-yesno-baseline/

Usage (run from MAST/):
    python eval/run_eval_yesno.py --model openai/o1
    python eval/run_eval_yesno.py --model openai/gpt-4o --max_workers 5
    python eval/run_eval_yesno.py --model openai/gpt-4o \\
        --input data/annotation/annotation_ag2_filtered.jsonl \\
        --output_dir outputs
"""

import os
import re
import json
import time
import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import litellm
from litellm import completion, ContextWindowExceededError, RateLimitError
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

load_dotenv(find_dotenv())

# --- Taxonomy definitions and examples (loaded from files, matching notebook) ---

_TAXONOMY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "taxonomy_definitions_examples")
DEFINITIONS = open(os.path.join(_TAXONOMY_DIR, "definitions.txt")).read()
EXAMPLES    = open(os.path.join(_TAXONOMY_DIR, "examples.txt")).read()


def format_trace(steps: list) -> str:
    lines = []
    for s in steps:
        lines.append(f"[{s['id']}]")
        lines.append(s["content"].strip())
        lines.append("")
    return "\n".join(lines)


def get_prompt(trace_text: str) -> str:
    """
    Prompt format matching the original MAST llm_judge_pipeline.ipynb exactly
    (modulo: 3.2/3.3 kept in correct GT order; 2.5 omitted since always-0 for AG2).
    Definitions and examples loaded from taxonomy_definitions_examples/ files.
    """
    return (
        "Below I will provide a multiagent system trace. provide me an analysis of the failure modes and inefficiencies as I will say below. \n"
        "In the traces, analyze the system behaviour."
        "There are several failure modes in multiagent systems I identified. I will provide them below. Tell me if you encounter any of them, as a binary yes or no. \n"
        "Also, give me a one sentence (be brief) summary of the problems with the inefficiencies or failure modes in the trace. Only mark a failure mode if you can provide an example of it in the trace, and specify that in your summary at the end"
        "Also tell me whether the task is successfully completed or not, as a binary yes or no."
        "At the very end, I provide you with the definitions of the failure modes and inefficiencies. After the definitions, I will provide you with examples of the failure modes and inefficiencies for you to understand them better."
        "Tell me if you encounter any of them between the @@ symbols as I will say below, as a binary yes or no."
        "Here are the things you should answer. Start after the @@ sign and end before the next @@ sign (do not include the @@ symbols in your answer):"
        "*** begin of things you should answer *** @@"
        "A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>"
        "B. Whether the task is successfully completed or not: <yes or no>"
        "C. Whether you encounter any of the failure modes or inefficiencies:"
        "1.1 Disobey Task Specification: <yes or no>"
        "1.2 Disobey Role Specification: <yes or no>"
        "1.3 Step Repetition: <yes or no>"
        "1.4 Loss of Conversation History: <yes or no>"
        "1.5 Unaware of Termination Conditions: <yes or no>"
        "2.1 Conversation Reset: <yes or no>"
        "2.2 Fail to Ask for Clarification: <yes or no>"
        "2.3 Task Derailment: <yes or no>"
        "2.4 Information Withholding: <yes or no>"
        "2.6 Action-Reasoning Mismatch: <yes or no>"
        "3.1 Premature Termination: <yes or no>"
        "3.2 Weak Verification: <yes or no>"
        "3.3 No or Incorrect Verification: <yes or no>"
        "@@*** end of your answer ***"
        "An example answer is: \n"
        "A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.\n"
        "B. no \n"
        "C. \n"
        "1.1 no \n"
        "1.2 no \n"
        "1.3 no \n"
        "1.4 no \n"
        "1.5 no \n"
        "1.6 yes \n"
        "2.1 no \n"
        "2.2 no \n"
        "2.3 yes \n"
        "2.4 no \n"
        "2.5 no \n"
        "2.6 yes \n"
        "2.7 no \n"
        "3.1 no \n"
        "3.2 yes \n"
        "3.3 no \n"
        "Here is the trace: \n"
        f"{trace_text}"
        "Also, here are the explanations (definitions) of the failure modes and inefficiencies: \n"
        f"{DEFINITIONS} \n"
        "Here are some examples of the failure modes and inefficiencies: \n"
        f"{EXAMPLES}"
    )


def call_llm(trace_text: str, model: str) -> str:
    prompt = get_prompt(trace_text)
    messages = [{"role": "user", "content": prompt}]

    if any(x in model for x in ("o1", "o3", "o4", "anthropic", "gemini-2.5")):
        params = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": 2000,
            "reasoning_effort": "high",
            "drop_params": True,
        }
    else:
        params = {
            "messages": messages,
            "model": model,
            "temperature": 0.0,
            "top_p": 1,
            "max_completion_tokens": 2000,
            "reasoning_effort": None,
            "drop_params": True,
        }

    for attempt in range(5):
        try:
            response = completion(**params)
            return response.choices[0].message["content"]
        except RateLimitError:
            wait = 60 * (2 ** attempt)
            print(f"Rate limit (attempt {attempt+1}/5): sleeping {wait}s...")
            time.sleep(wait)
    raise RateLimitError("Exceeded 5 retries due to rate limiting")


MAST_MODES = ["1.1", "1.2", "1.3", "1.4", "1.5",
              "2.1", "2.2", "2.3", "2.4", "2.6",
              "3.1", "3.2", "3.3"]


def parse_response(response: str) -> dict:
    """
    Parse the LLM response to extract yes/no for each failure mode.
    Returns dict like {"1.1": 1, "1.2": 0, ...} (1=yes, 0=no).
    """
    # Strip @@ markers if present
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
            result[mode] = 0  # default to no if not parseable
    return result


def process_record(r: dict, output_dir: str, model: str) -> None:
    rec_id = r["_rec_id"]
    output_file = os.path.join(output_dir, f"{rec_id}.json")
    if os.path.exists(output_file):
        return

    trace_text = format_trace(r["steps"])
    raw_response = ""
    try:
        raw_response = call_llm(trace_text, model)
        predictions = parse_response(raw_response)
    except ContextWindowExceededError:
        predictions = {m: 0 for m in MAST_MODES}
        raw_response = "Context window exceeded."
    except Exception as e:
        print(f"Error on record {rec_id} (trace_id={r.get('trace_id')}): {e}")
        predictions = {m: 0 for m in MAST_MODES}

    out = {
        "rec_id": rec_id,
        "trace_id": r.get("trace_id"),
        "predictions": predictions,
        "raw_response": raw_response,
    }
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="MAST-AG2 yes/no error-type prediction (baseline)")
    ap.add_argument("--model", default="openai/o1",
                    help="LiteLLM model string (default: openai/o1, matching original MAST paper)")
    ap.add_argument("--input", default="data/annotation/annotation_ag2_filtered.jsonl",
                    help="Path to annotation_ag2_filtered.jsonl")
    ap.add_argument("--output_dir", default="outputs",
                    help="Root output directory (default: outputs/)")
    ap.add_argument("--max_workers", type=int, default=1,
                    help="Parallel workers (default: 1; use 1 for o1 to avoid rate limits)")
    args = ap.parse_args()

    records = []
    with open(args.input) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            r["_rec_id"] = f"{idx:04d}"
            records.append(r)
    print(f"Loaded {len(records)} traces from {args.input}")

    model_tag = args.model.replace("/", "-")
    out_dir = os.path.join(args.output_dir, f"{model_tag}-yesno-baseline")
    os.makedirs(out_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_record, r, out_dir, args.model)
            for r in records
        ]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    print(f"\n✓ Outputs saved to {out_dir}/")
    print(f"  Next: python eval/calculate_scores_yesno.py --pred_dir {out_dir}")


if __name__ == "__main__":
    litellm.drop_params = True
    main()
