"""
eval/run_eval_with_graph.py — LLM-as-judge augmented with MAST causal graph guidance.

Identical to run_eval.py except the prompt is augmented with a "Causal Error Patterns"
block derived from intervention-validated edges in outputs/interventions/effect_edges.json.

Only edges with validated=True are included. Strength is |delta| (the causal effect size
from the do(A=0) intervention experiment).

Usage:
    python eval/run_eval_with_graph.py --model openai/gpt-4o
    python eval/run_eval_with_graph.py --model openai/gpt-4o \
        --effect_edges outputs/interventions/effect_edges.json

Outputs saved to:
    outputs/{model}-causal/
"""

import os
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

DEFAULT_EFFECT_EDGES_PATH = "outputs/interventions/effect_edges.json"


def load_validated_edges(effect_edges_path: str) -> list:
    """Load intervention-validated edges from effect_edges.json.

    Returns list of (a, b, strength) tuples where strength = |delta|,
    sorted by descending effect size. Only validated=True edges are included.
    """
    with open(effect_edges_path) as f:
        data = json.load(f)
    edges = []
    for key, info in data["edges"].items():
        if not info.get("validated"):
            continue
        a = info["a"]
        b = info["b"]
        delta = info.get("delta")
        strength = abs(delta) if delta is not None else 0.0
        edges.append((a, b, strength))
    edges.sort(key=lambda x: -x[2])
    return edges


def format_graph_guidance(edges: list) -> str:
    if not edges:
        return ""
    lines = [
        "# Causal Error Patterns (experimentally validated via do-calculus interventions)",
        "When you identify an error of type A in the trace, actively look for errors of type B",
        "in subsequent steps. These relationships were validated by patching A and measuring",
        "whether B disappeared in the counterfactual trace.",
        "Higher strength values indicate larger causal effect size (|Δ| from intervention).",
        "",
        "Format: [Source Error] → [Consequent Error]  (causal strength: X.XX)",
        "",
    ]
    for a, b, w in edges:
        lines.append(f"  {a} → {b}  (causal strength: {w:.2f})")
    lines.append("")
    return "\n".join(lines)


def strip_markdown(text: str) -> str:
    """Remove markdown code fences that models sometimes wrap around JSON output."""
    import re
    text = text.strip()
    text = re.sub(r"^```\w*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def format_trace(steps: list) -> str:
    lines = []
    for s in steps:
        lines.append(f"[{s['id']}]")
        lines.append(s["content"].strip())
        lines.append("")
    return "\n".join(lines)


def get_prompt(trace_text: str, graph_guidance: str) -> str:
    return f"""Follow the MAST taxonomy below carefully and provide the output in the exact JSON format shown.

# MAST Error Taxonomy
{MAST_TAXONOMY}

{graph_guidance}- Analyze the multi-agent trace below and identify all errors present.
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


def call_llm(trace_text: str, graph_guidance: str, model: str) -> str:
    prompt = get_prompt(trace_text, graph_guidance)
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


def process_record(r: dict, output_dir: str, model: str, graph_guidance: str) -> None:
    rec_id = r["_rec_id"]
    output_file = os.path.join(output_dir, f"{rec_id}.json")
    if os.path.exists(output_file):
        return

    trace_text = format_trace(r["steps"])
    try:
        response = call_llm(trace_text, graph_guidance, model)
    except ContextWindowExceededError:
        response = '{"errors": []}'
    except Exception as e:
        print(f"Error on record {rec_id} (trace_id={r['trace_id']}): {e}")
        response = '{"errors": []}'

    with open(output_file, "w") as f:
        f.write(strip_markdown(response) or '{"errors": []}')


def main():
    ap = argparse.ArgumentParser(description="MAST-AG2 LLM eval with causal graph guidance")
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--input", default="../data/annotation/annotation_ag2_filtered.jsonl")
    ap.add_argument("--effect_edges", default=DEFAULT_EFFECT_EDGES_PATH,
                    help="Path to effect_edges.json from causal intervention validation")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--max_workers", type=int, default=5)
    args = ap.parse_args()

    # Load graph guidance
    print(f"Loading validated causal edges from {args.effect_edges}...")
    edges = load_validated_edges(args.effect_edges)
    graph_guidance = format_graph_guidance(edges)
    print(f"  {len(edges)} validated edges included:")
    for a, b, w in edges:
        print(f"    {a} → {b}  (|Δ|={w:.3f})")

    records = []
    with open(args.input) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            r["_rec_id"] = f"{idx:04d}"
            records.append(r)
    print(f"\nLoaded {len(records)} traces")

    model_tag = args.model.replace("/", "-")
    out_dir = os.path.join(args.output_dir, f"{model_tag}-causal")
    os.makedirs(out_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_record, r, out_dir, args.model, graph_guidance)
                   for r in records]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    print(f"✓ Outputs saved to {out_dir}/")


if __name__ == "__main__":
    litellm.drop_params = True
    main()
