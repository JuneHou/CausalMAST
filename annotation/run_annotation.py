"""
Main CLI for MAST step-level annotation — two-pass pipeline.

Pass 1 (onset localization): for each error label in the trace, ask the LLM
  which step is the earliest candidate onset. Returns category + location +
  short_reason.

Pass 2 (verification + full annotation): given the Pass 1 candidate locations,
  verify each and render the full annotation: evidence, description,
  why_not_earlier, impact.

Workflow:
  1. Load records for the given task from MAD
  2. For each record: extract steps → Pass 1 → Pass 2 → write JSONL
  3. Output one line per trace:
     {trace_id, mas_name, mast_annotation, steps,
      pass1_candidates, errors, prompt_variant, model}

Currently supported tasks: openmanus
Future: appworld, chatdev, ag2, metagpt, hyperagent

Usage examples:
    # Phase 0: inspect steps before annotating
    python -m annotation.inspect_steps --task openmanus --n 3 --output inspect.txt

    # Debug run: 3 traces, prints full prompts + raw responses to stderr
    python -m annotation.run_annotation \\
        --task openmanus --model gpt-4o --prompt_variant few_shot \\
        --debug --output debug_openmanus.jsonl

    # Full run with GPT-4o
    python -m annotation.run_annotation \\
        --task openmanus --model gpt-4o --prompt_variant few_shot \\
        --output annotation_openmanus.jsonl

    # Full run with vLLM offline
    python -m annotation.run_annotation \\
        --task openmanus --offline \\
        --model Qwen/Qwen3-32B --prompt_variant few_shot \\
        --output annotation_openmanus.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from . import config
from . import openmanus as _openmanus_mod
from . import metagpt as _metagpt_mod
from . import ag2 as _ag2_mod
from .definitions import parse_definitions, category_label
from .prompt import (build_pass1_prompt, build_pass2_prompt,
                     build_pass1_retry_prompt, build_pass2_retry_prompt)
from .llm_client import (
    LLMClient,
    extract_json,
    validate_pass1_response,
    validate_pass2_response,
)

# ---------------------------------------------------------------------------
# Task registry: task_name -> (load_records_fn, extract_steps_fn)
# ---------------------------------------------------------------------------

_TASK_REGISTRY = {
    "openmanus": (_openmanus_mod.load_records, _openmanus_mod.extract_steps),
    "metagpt": (_metagpt_mod.load_records, _metagpt_mod.extract_steps),
    "ag2": (_ag2_mod.load_records, _ag2_mod.extract_steps),
}


def _resolve_task(name: str):
    key = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    match = _TASK_REGISTRY.get(key)
    if not match:
        available = list(_TASK_REGISTRY.keys())
        print(f"Error: unknown task {name!r}. Available: {available}", file=sys.stderr)
        sys.exit(1)
    return match


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------

def run(args):
    mad_path = Path(args.mad)
    if not mad_path.is_file():
        print(f"MAD dataset not found at {mad_path}.\nRun: python old/scripts/0_download_mad.py", file=sys.stderr)
        sys.exit(1)

    load_fn, extract_fn = _resolve_task(args.task)

    # Load definitions
    defs = parse_definitions()

    # Load records — min_errors=2 required to build causal relationships between errors
    records = load_fn(mad_path, min_errors=2)
    print(f"Loaded {len(records)} {args.task} records with >= 2 error labels.", file=sys.stderr)
    if not records:
        print("No records to annotate. Exiting.", file=sys.stderr)
        sys.exit(0)

    # --debug caps at 3 traces and enables verbose prompt/response printing
    debug = getattr(args, "debug", False)
    if debug:
        records = records[:3]
        print(f"[DEBUG] Limited to {len(records)} traces.", file=sys.stderr)
    elif args.limit:
        records = records[: args.limit]
        print(f"Limited to {len(records)} records (--limit {args.limit}).", file=sys.stderr)

    # Initialise LLM client
    client = LLMClient(
        model=args.model,
        offline=args.offline,
        base_url=args.base_url,
        api_key=args.api_key,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        temperature=0.0,
        max_tokens=2048,
    )

    # Prepare output file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    annotated = 0
    pass1_errors = 0
    pass2_errors = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for idx, rec in enumerate(records):
            trace_id = rec.get("trace_id", f"trace_{idx}")
            error_ids = rec.get("error_ids", [])

            # Step extraction
            steps = extract_fn(rec["trajectory"])
            if not steps:
                print(f"[{idx+1}/{len(records)}] {trace_id}: SKIPPED (0 steps extracted)", file=sys.stderr)
                skipped += 1
                continue

            valid_step_ids = {s["id"] for s in steps}
            valid_categories = {category_label(eid, defs) for eid in error_ids}

            print(
                f"[{idx+1}/{len(records)}] {trace_id}: {len(steps)} steps, errors={error_ids}",
                file=sys.stderr,
                flush=True,
            )

            # ------------------------------------------------------------------
            # Pass 1: onset localization (with one retry for missing categories)
            # ------------------------------------------------------------------
            p1_prompt = build_pass1_prompt(steps, error_ids, defs, variant=args.prompt_variant, task=args.task)

            if debug:
                _print_block("[DEBUG] PASS 1 PROMPT", p1_prompt)

            p1_raw = client.generate([p1_prompt])[0]

            if debug:
                _print_block("[DEBUG] PASS 1 RAW RESPONSE", p1_raw)

            p1_parsed = extract_json(p1_raw)
            if p1_parsed is None:
                print(f"  -> Pass 1 parse error: could not extract JSON", file=sys.stderr)
                _write_record(out_f, rec, steps, [], [], args,
                              pass1_error=p1_raw[:200])
                pass1_errors += 1
                annotated += 1
                continue

            p1_validated = validate_pass1_response(p1_parsed, valid_step_ids, valid_categories)
            candidates = p1_validated["candidates"]

            # Retry Pass 1 for any missing categories (one attempt)
            missing_p1 = _missing_ids(error_ids, candidates)
            if missing_p1:
                print(f"  -> Pass 1: missing {missing_p1}, retrying...", file=sys.stderr)
                p1_retry_prompt = build_pass1_retry_prompt(steps, missing_p1, defs, task=args.task)
                if debug:
                    _print_block("[DEBUG] PASS 1 RETRY PROMPT", p1_retry_prompt)
                p1_retry_raw = client.generate([p1_retry_prompt])[0]
                if debug:
                    _print_block("[DEBUG] PASS 1 RETRY RESPONSE", p1_retry_raw)
                p1_retry_parsed = extract_json(p1_retry_raw)
                if p1_retry_parsed:
                    p1_retry_validated = validate_pass1_response(p1_retry_parsed, valid_step_ids, valid_categories)
                    existing_cats = {c["category"] for c in candidates}
                    for c in p1_retry_validated["candidates"]:
                        if c["category"] not in existing_cats:
                            candidates.append(c)
                            existing_cats.add(c["category"])

            if debug:
                print(f"[DEBUG] PASS 1 CANDIDATES ({len(candidates)}/{len(error_ids)}):", file=sys.stderr)
                print(json.dumps(candidates, indent=2, ensure_ascii=False), file=sys.stderr)

            if len(candidates) < len(error_ids):
                print(
                    f"  -> Pass 1: {len(candidates)}/{len(error_ids)} candidates after retry",
                    file=sys.stderr,
                )

            if not candidates:
                print(f"  -> Pass 1 produced no valid candidates; skipping Pass 2", file=sys.stderr)
                _write_record(out_f, rec, steps, candidates, [], args)
                annotated += 1
                continue

            # ------------------------------------------------------------------
            # Pass 2: verification + full annotation (with one retry for missing)
            # ------------------------------------------------------------------
            p2_prompt = build_pass2_prompt(steps, candidates, defs, variant=args.prompt_variant, task=args.task)

            if debug:
                _print_block("[DEBUG] PASS 2 PROMPT", p2_prompt)

            p2_raw = client.generate([p2_prompt])[0]

            if debug:
                _print_block("[DEBUG] PASS 2 RAW RESPONSE", p2_raw)

            p2_parsed = extract_json(p2_raw)
            if p2_parsed is None:
                print(f"  -> Pass 2 parse error: could not extract JSON", file=sys.stderr)
                _write_record(out_f, rec, steps, candidates, [],
                              args, pass2_error=p2_raw[:200])
                pass2_errors += 1
                annotated += 1
                continue

            p2_validated = validate_pass2_response(p2_parsed, valid_step_ids, valid_categories)
            final_errors = p2_validated["errors"]

            # Retry Pass 2 for any missing categories (one attempt)
            candidate_ids = [c.get("category", "").split()[0] for c in candidates
                             if c.get("category", "").split()]
            missing_p2 = _missing_ids(candidate_ids, final_errors)
            if missing_p2:
                print(f"  -> Pass 2: missing {missing_p2}, retrying...", file=sys.stderr)
                missing_candidates = [c for c in candidates
                                      if c.get("category", "").split()[0] in missing_p2]
                p2_retry_prompt = build_pass2_retry_prompt(steps, missing_candidates, defs, task=args.task)
                if debug:
                    _print_block("[DEBUG] PASS 2 RETRY PROMPT", p2_retry_prompt)
                p2_retry_raw = client.generate([p2_retry_prompt])[0]
                if debug:
                    _print_block("[DEBUG] PASS 2 RETRY RESPONSE", p2_retry_raw)
                p2_retry_parsed = extract_json(p2_retry_raw)
                if p2_retry_parsed:
                    p2_retry_validated = validate_pass2_response(p2_retry_parsed, valid_step_ids, valid_categories)
                    existing_cats = {e["category"] for e in final_errors}
                    for e in p2_retry_validated["errors"]:
                        if e["category"] not in existing_cats:
                            final_errors.append(e)
                            existing_cats.add(e["category"])

            if debug:
                print(f"[DEBUG] PASS 2 FINAL ERRORS ({len(final_errors)}/{len(candidates)}):", file=sys.stderr)
                print(json.dumps(final_errors, indent=2, ensure_ascii=False), file=sys.stderr)

            if len(final_errors) < len(candidates):
                print(
                    f"  -> Pass 2: {len(final_errors)}/{len(candidates)} errors after retry",
                    file=sys.stderr,
                )

            _write_record(out_f, rec, steps, candidates, final_errors, args)
            annotated += 1

    print(
        f"\nDone. annotated={annotated}, skipped={skipped}, "
        f"pass1_errors={pass1_errors}, pass2_errors={pass2_errors}",
        file=sys.stderr,
    )
    print(f"Output: {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _missing_ids(expected_ids: List[str], annotated: List[Dict]) -> List[str]:
    """Return error IDs from expected_ids that are absent from annotated entries."""
    annotated_ids = {e.get("category", "").split()[0] for e in annotated}
    return [eid for eid in expected_ids if eid not in annotated_ids]


def _print_block(label: str, content: str):
    sep = "=" * 72
    print(f"\n{sep}", file=sys.stderr)
    print(f"{label} ({len(content)} chars):", file=sys.stderr)
    print(sep, file=sys.stderr)
    print(content, file=sys.stderr)
    print(sep, file=sys.stderr)


def _write_record(out_f, rec, steps, candidates, errors, args,
                  pass1_error=None, pass2_error=None):
    out_record = {
        "trace_id": rec.get("trace_id", ""),
        "mas_name": rec["mas_name"],
        "mast_annotation": rec.get("mast_annotation", {}),
        "steps": steps,
        "pass1_candidates": candidates,
        "errors": errors,
        "prompt_variant": args.prompt_variant,
        "model": args.model,
    }
    if pass1_error is not None:
        out_record["pass1_error"] = pass1_error
    if pass2_error is not None:
        out_record["pass2_error"] = pass2_error
    out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
    out_f.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MAST step-level annotation pipeline (two-pass).")

    # Task + data
    ap.add_argument("--task", required=True,
                    help="Task to annotate (e.g. openmanus)")
    ap.add_argument("--mad", default=str(config.DEFAULT_MAD_PATH),
                    help="Path to MAD_full_dataset.json")

    # LLM mode
    ap.add_argument("--offline", action="store_true",
                    help="Load vLLM in-process (no separate server)")
    ap.add_argument("--model", default=config.DEFAULT_VLLM_MODEL,
                    help=f"Model name (default: {config.DEFAULT_VLLM_MODEL})")
    ap.add_argument("--base_url", default="https://api.openai.com/v1",
                    help="API base URL (default: OpenAI)")
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""),
                    help="API key (default: $OPENAI_API_KEY env var)")
    ap.add_argument("--tensor_parallel_size", type=int, default=None,
                    help="vLLM tensor parallel size (--offline only)")
    ap.add_argument("--pipeline_parallel_size", type=int, default=1,
                    help="vLLM pipeline parallel size (--offline only; default: 1)")

    # Prompt
    ap.add_argument("--prompt_variant", default=config.DEFAULT_PROMPT_VARIANT,
                    choices=config.PROMPT_VARIANTS,
                    help=f"Prompt variant (default: {config.DEFAULT_PROMPT_VARIANT})")

    # Output
    ap.add_argument("--output", default="annotation_output.jsonl",
                    help="Output JSONL file path")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max traces to process (useful for debugging)")
    ap.add_argument("--debug", action="store_true",
                    help="Debug mode: run on 3 traces, print full prompts and raw responses to stderr")

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
