"""
eval/run_eval.py — LLM-as-judge for MAST-AG2 traces (baseline, no graph guidance).

Given a trace of steps, the LLM identifies errors using the MAST taxonomy and
predicts the location (step_XX) of each error.

Input: annotation_ag2_filtered.jsonl (or a test split of it)
Output: outputs/{model}-{split}/  — one JSON file per trace_id

Usage:
    python eval/run_eval.py --model openai/gpt-4o
    python eval/run_eval.py --model openai/gpt-4o --input ../data/annotation/annotation_ag2_filtered.jsonl
"""

import os
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

MAST_TAXONOMY = """
├── 1. Task Compliance Errors
│   ├── 1.1 Disobey Task Specification — Violates constraints or requirements explicitly stated in the task
│   ├── 1.2 Disobey Role Specification — Ignores or violates the assigned agent role
│   ├── 1.3 Step Repetition — Repeats a task or phase already completed with a result
│   ├── 1.4 Loss of Conversation History — Fails to retain or use prior conversation context
│   └── 1.5 Unaware of Termination Conditions — Continues past a valid stopping point or stops too early
├── 2. Multi-Agent Coordination Errors
│   ├── 2.1 Conversation Reset — Resets the conversation, losing prior context and progress
│   ├── 2.2 Fail to Ask for Clarification — Proceeds without resolving ambiguity that required clarification
│   ├── 2.3 Task Derailment — Shifts focus away from the intended objective
│   ├── 2.4 Information Withholding — Fails to share information needed by other agents
│   └── 2.6 Action-Reasoning Mismatch — Executes an action inconsistent with the stated reasoning
└── 3. Output Verification Errors
    ├── 3.1 Premature Termination — Stops before the task is complete or a required output is produced
    ├── 3.2 Weak Verification — Performs only superficial or incomplete verification
    └── 3.3 No or Incorrect Verification — Skips verification entirely or verifies against wrong criteria
"""


def strip_markdown(text: str) -> str:
    """Remove markdown code fences that GPT-4o sometimes wraps around JSON output."""
    import re
    text = text.strip()
    text = re.sub(r"^```\w*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def format_trace(steps: list) -> str:
    """Format steps list into readable text for the LLM."""
    lines = []
    for s in steps:
        lines.append(f"[{s['id']}]")
        lines.append(s["content"].strip())
        lines.append("")
    return "\n".join(lines)


def get_prompt(trace_text: str) -> str:
    return f"""Follow the MAST taxonomy below carefully and provide the output in the exact JSON format shown.

# MAST Error Taxonomy
{MAST_TAXONOMY}

- Analyze the multi-agent trace below and identify all errors present.
- Use ONLY the leaf-level categories (e.g. "1.1 Disobey Task Specification", not "1. Task Compliance Errors").
- For each error, identify the FIRST step where it occurs, using the exact step ID shown in the trace (e.g. "step_00", "step_01", ...).
- Output strictly valid JSON — no markdown, no extra text.

Output template:
{{
    "errors": [
        {{
            "category": "<exact leaf category name from taxonomy>",
            "location": "<step_XX where the error first occurs>",
            "evidence": "<quoted text from the trace that shows the error>",
            "description": "<explanation of why this is an error>",
            "impact": "<HIGH | MEDIUM | LOW>"
        }}
    ]
}}

If no errors are found:
{{
    "errors": []
}}

Trace to analyze:

{trace_text}

Output only the JSON, nothing else.
"""


def call_llm(trace_text: str, model: str) -> str:
    prompt = get_prompt(trace_text)
    messages = [{"role": "user", "content": prompt}]

    if any(x in model for x in ("o1", "o3", "o4", "anthropic", "gemini-2.5")):
        params = {"messages": messages, "model": model,
                  "max_completion_tokens": 4000, "reasoning_effort": "high", "drop_params": True}
    else:
        params = {"messages": messages, "model": model,
                  "temperature": 0.0, "top_p": 1, "max_completion_tokens": 4000,
                  "reasoning_effort": None, "drop_params": True}

    for attempt in range(3):
        try:
            response = completion(**params)
            return response.choices[0].message["content"]
        except RateLimitError:
            print(f"Rate limit (attempt {attempt+1}/3): sleeping 60s...")
            time.sleep(60)
    raise RateLimitError("Exceeded 3 retries due to rate limiting")


def process_record(r: dict, output_dir: str, model: str) -> None:
    rec_id = r["_rec_id"]
    output_file = os.path.join(output_dir, f"{rec_id}.json")
    if os.path.exists(output_file):
        return

    trace_text = format_trace(r["steps"])
    try:
        response = call_llm(trace_text, model)
    except ContextWindowExceededError:
        response = '{"errors": []}'
    except Exception as e:
        print(f"Error on record {rec_id} (trace_id={r['trace_id']}): {e}")
        response = '{"errors": []}'

    with open(output_file, "w") as f:
        f.write(strip_markdown(response) or '{"errors": []}')


def main():
    ap = argparse.ArgumentParser(description="MAST-AG2 LLM eval (no graph guidance)")
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--input", default="../data/annotation/annotation_ag2_filtered.jsonl",
                    help="Path to annotation_ag2_filtered.jsonl (or test split)")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--max_workers", type=int, default=5)
    args = ap.parse_args()

    records = []
    with open(args.input) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            r["_rec_id"] = f"{idx:04d}"
            records.append(r)
    print(f"Loaded {len(records)} traces from {args.input}")

    model_tag = args.model.replace("/", "-")
    out_dir = os.path.join(args.output_dir, f"{model_tag}-baseline")
    os.makedirs(out_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_record, r, out_dir, args.model) for r in records]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    print(f"✓ Outputs saved to {out_dir}/")


if __name__ == "__main__":
    litellm.drop_params = True
    main()
