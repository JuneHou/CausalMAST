"""
eval/run_eval_with_graph_vllm.py — vLLM evaluation augmented with intervention-validated
causal graph guidance. Mirrors run_eval_with_graph.py but uses vLLM for local models.

Identical to run_eval_vllm.py except the prompt includes the 7 validated causal edges
from outputs/interventions/effect_edges.json (validated=True, strength=|delta|).

Usage (from causal_graph/):
    CUDA_VISIBLE_DEVICES=0,1 python eval/run_eval_with_graph_vllm.py \
        --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
        --tensor_parallel_size 2

    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval/run_eval_with_graph_vllm.py \
        --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
        --tensor_parallel_size 4 \
        --max_model_len 65536

Outputs saved to:
    outputs/{model_tag}-causal/
"""

import os
import json
import argparse
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


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


def strip_markdown(text: str) -> str:
    """Remove markdown code fences that models sometimes wrap around JSON output."""
    import re
    text = text.strip()
    text = re.sub(r"^```\w*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def apply_chat_template(tokenizer, user_text: str) -> str:
    if tokenizer.chat_template is None:
        bos = tokenizer.bos_token or "<s>"
        return f"{bos}[INST] {user_text} [/INST]"
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="MAST-AG2 vLLM eval with causal graph guidance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str,
                        default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    parser.add_argument("--input", type=str, default="../data/annotation/annotation_ag2_filtered.jsonl")
    parser.add_argument("--effect_edges", type=str, default=DEFAULT_EFFECT_EDGES_PATH,
                        help="Path to effect_edges.json from causal intervention validation")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--max_model_len", type=int, default=131072,
                        help="Context window size (tokens). Mistral-Small-3.1 supports 128K.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_new_tokens", type=int, default=4000)
    parser.add_argument("--min_output_tokens", type=int, default=1024,
                        help="Reserve this many tokens for generation; skip trace if prompt exceeds "
                             "max_model_len - min_output_tokens.")
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    args = parser.parse_args()

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
    print(f"\nLoaded {len(records)} traces from {args.input}")

    model_tag = args.model.replace("/", "-")
    out_dir = os.path.join(args.output_dir, f"{model_tag}-causal")
    os.makedirs(out_dir, exist_ok=True)

    # Filter already-done traces
    todo = [r for r in records
            if not os.path.exists(os.path.join(out_dir, f"{r['_rec_id']}.json"))]
    print(f"  {len(records) - len(todo)} already done, {len(todo)} to process")
    if not todo:
        print("Nothing to do.")
        return

    print(f"Loading tokenizer for {args.model} ...")
    is_mistral = "mistral" in args.model.lower()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
        **({"use_fast": False} if is_mistral else {}),
    )

    print(f"Loading model {args.model} ...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    token_budget = args.max_model_len - args.min_output_tokens

    # Build prompts, tracking which traces are skipped
    prompt_texts = []
    valid_records = []
    skipped = 0
    print("Building prompts and checking token lengths...")
    for r in tqdm(todo):
        trace_text = format_trace(r["steps"])
        user_text = get_prompt(trace_text, graph_guidance)
        prompt_text = apply_chat_template(tokenizer, user_text)
        tok_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        if tok_len > token_budget:
            print(f"  Skipping record {r['_rec_id']} (trace_id={r['trace_id']}): prompt too long "
                  f"({tok_len:,} tokens, budget {token_budget:,})")
            out_file = os.path.join(out_dir, f"{r['_rec_id']}.json")
            with open(out_file, "w") as f:
                f.write('{"errors": []}')
            skipped += 1
        else:
            prompt_texts.append(prompt_text)
            valid_records.append(r)

    print(f"Generating {len(prompt_texts)} responses (skipped {skipped} over-length)...")
    outputs = llm.generate(prompt_texts, sp)

    print("Writing outputs...")
    for r, out in tqdm(zip(valid_records, outputs), total=len(valid_records)):
        response = out.outputs[0].text
        out_file = os.path.join(out_dir, f"{r['_rec_id']}.json")
        with open(out_file, "w") as f:
            f.write(strip_markdown(response) or '{"errors": []}')

    print(f"\nDone. {len(valid_records)} generated, {skipped} skipped (over-length → empty).")
    print(f"Outputs saved to {out_dir}/")
    print(f"\nScore with (from causal_graph/):")
    print(f"  python eval/calculate_scores.py --gt_dir data/gt --pred_dir {out_dir}")


if __name__ == "__main__":
    main()
