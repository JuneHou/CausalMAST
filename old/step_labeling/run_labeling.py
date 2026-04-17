#!/usr/bin/env python3
"""
MAST step-level failure labeling: preprocessing + Pass 1 + Pass 2.
Data: MAD dataset (from original MAST; trace + mast_annotation in one file) or traces/ + optional annotations.
Excludes Magentic. Filters to >=2 failure types. LLM: vLLM (default Qwen/Qwen3-32B; --model for gpt-oss).
"""
import argparse
import json
from pathlib import Path
from typing import Optional

from . import config
from .preprocessing import discover_traces, filter_traces_with_n_failures, get_failure_types_set, load_mad_records
from .step_extraction import get_steps_from_trace
from .load_definitions import load_definitions
from .pass1_pass2 import run_labeling_for_trace
from .vllm_client import get_offline_llm


def main():
    ap = argparse.ArgumentParser(description="MAST step-level failure labeling (vLLM)")
    ap.add_argument(
        "--mad",
        type=str,
        default=None,
        help=f"Path to MAD_full_dataset.json. Default: {config.DEFAULT_MAD_PATH} if present. Download first: python scripts/0_download_mad.py",
    )
    ap.add_argument(
        "--steps_dataset",
        type=str,
        default=None,
        help="Path to pre-extracted steps_dataset.jsonl (from extract_steps_dataset). When set, use these steps for labeling instead of re-extracting from MAD.",
    )
    ap.add_argument("--traces_dir", type=str, default=str(config.TRACES_DIR), help="Official traces directory (used only when not --mad and not --steps_dataset)")
    ap.add_argument("--annotations", type=str, default=None, help="Optional JSONL/JSON with mast_annotation when using --traces_dir (custom annotations)")
    ap.add_argument("--min_failures", type=int, default=2, help="Keep only traces with >= this many failure types")
    ap.add_argument("--manifest_out", type=str, default=None, help="Write preprocessed manifest to this path")
    ap.add_argument("--definitions", type=str, default=str(config.DEFINITIONS_PATH), help="MAST definitions.txt path")
    ap.add_argument("--model", type=str, default=config.DEFAULT_VLLM_MODEL, help=f"Model name (default: {config.DEFAULT_VLLM_MODEL}; use {config.GPT_OSS_MODEL} for gpt-oss)")
    ap.add_argument("--offline", action="store_true", help="Load vLLM in-process and send queries directly (no server or URL)")
    ap.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="vLLM tensor parallel size (only for --offline). Default: number of visible GPUs from CUDA_VISIBLE_DEVICES.",
    )
    ap.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=None,
        help="vLLM pipeline parallel size (only for --offline). Default: 1.",
    )
    ap.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="vLLM server base URL (only when not using --offline)")
    ap.add_argument("--api_key", type=str, default="token-abc123", help="API key for vLLM server (only when not using --offline)")
    ap.add_argument("--output", type=str, default="step_labeling_output.jsonl", help="Output JSONL path (one JSON object per trace); ignored when --task is set (then output goes under --output_dir/<task>/).")
    ap.add_argument("--output_dir", type=str, default="step_labeling_results", help="When --task is set, write output to <output_dir>/<task>/step_labeling_output.jsonl (and debug log under same folder).")
    ap.add_argument("--task", type=str, default=None, help="Run only this task (e.g. ChatDev, AG2). Case-insensitive. Output written to <output_dir>/<task>/.")
    ap.add_argument("--debug_log", type=str, default=None, help="If set, write one JSONL line per Pass 1 step (for debugging). When --task is set, defaults to <output_dir>/<task>/step_labeling_debug.jsonl.")
    ap.add_argument("--limit", type=int, default=None, help="Max number of traces to process (for debugging)")
    ap.add_argument("--no_adjudicate", action="store_true", help="Skip Pass 2 adjudication when multiple candidates")
    args = ap.parse_args()

    mad_path = Path(args.mad) if args.mad else config.DEFAULT_MAD_PATH
    steps_dataset_path = Path(args.steps_dataset) if args.steps_dataset else None
    use_steps_dataset = steps_dataset_path is not None and steps_dataset_path.is_file()
    use_mad = not use_steps_dataset and mad_path.is_file()

    if use_steps_dataset:
        records = []
        with open(steps_dataset_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                failure_types = obj.get("failure_types") or get_failure_types_set(obj.get("mast_annotation") or {})
                if len(failure_types) < args.min_failures:
                    continue
                records.append({
                    "trace_id": obj.get("trace_id", ""),
                    "mas_name": obj.get("mas_name", ""),
                    "steps": obj.get("steps", []),
                    "failure_types": failure_types,
                    "mast_annotation": obj.get("mast_annotation", {}),
                })
        print(f"Loaded steps dataset from {steps_dataset_path}: {len(records)} traces (>= {args.min_failures} failure types).")
    elif use_mad:
        records = load_mad_records(mad_path, min_failures=args.min_failures, exclude_magentic=True)
        print(f"Loaded MAD from {mad_path}: {len(records)} traces (>= {args.min_failures} failure types, Magentic excluded).")
    else:
        traces_dir = Path(args.traces_dir)
        records = discover_traces(traces_dir)
        print(f"Discovered {len(records)} traces under traces/ (AG2, AppWorld, ChatDev, HyperAgent, MetaGPT, OpenManus).")
        if args.annotations and Path(args.annotations).is_file():
            records = filter_traces_with_n_failures(records, Path(args.annotations), args.min_failures)
            print(f"After filtering (>= {args.min_failures} failure types): {len(records)} traces.")
        else:
            if args.annotations:
                print("Warning: --annotations path missing or not file; not filtering by failure count.")
            if not records:
                print("No traces found. Exiting.")
                return
            if not args.annotations and records:
                print("Warning: No annotations file. Skipping labeling; use --manifest_out to write discovered traces, or use --mad to load MAD.")
                if args.manifest_out:
                    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
                    with open(args.manifest_out, "w", encoding="utf-8") as f:
                        json.dump(records, f, indent=2, ensure_ascii=False)
                return

    definitions_path = Path(args.definitions)
    definitions_text = ""
    if definitions_path.is_file():
        with open(definitions_path, "r", encoding="utf-8", errors="replace") as f:
            definitions_text = f.read()
    else:
        print(f"Warning: definitions not found at {definitions_path}")

    if args.manifest_out and records:
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        # Write without large trajectory when from MAD
        manifest_records = [{k: v for k, v in r.items() if k != "trajectory"} for r in records]
        with open(args.manifest_out, "w", encoding="utf-8") as f:
            json.dump(manifest_records, f, indent=2, ensure_ascii=False)
        print(f"Wrote manifest to {args.manifest_out}.")

    if not records:
        print("No traces to label. Exiting.")
        return

    # Filter by task if --task set
    task_filter = (args.task or "").strip().lower()
    if task_filter:
        records = [r for r in records if (r.get("mas_name") or "").strip().lower() == task_filter]
        if not records:
            print(f"No traces for task '{args.task}'. Exiting.")
            return
        # Canonical task name for folder (from first record or config)
        canonical_task = records[0].get("mas_name") or args.task
        if canonical_task not in config.SUPPORTED_TASK_TYPES:
            for t in config.SUPPORTED_TASK_TYPES:
                if t.lower() == task_filter:
                    canonical_task = t
                    break
        print(f"Running for task only: {canonical_task} ({len(records)} traces).")

    # Optional: load vLLM in-process at start (no separate server/URL)
    llm = None
    use_offline = getattr(args, "offline", False)
    if use_offline:
        print(f"Loading model in-process: {args.model} (no server or URL)...")
        llm = get_offline_llm(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
        )
        print("Model loaded. Sending queries directly to vLLM.")
    else:
        # In server mode, model parallelism is configured when you start the vLLM server,
        # not through this client. We keep these args for a consistent CLI surface.
        if args.tensor_parallel_size is not None or args.pipeline_parallel_size is not None:
            print("Note: --tensor_parallel_size/--pipeline_parallel_size are only used with --offline (in-process vLLM).")

    if task_filter:
        out_dir = Path(args.output_dir) / canonical_task
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "step_labeling_output.jsonl"
        # When running by task, default debug log to task folder (or use --debug_log path if given)
        debug_log_path = args.debug_log or str(out_dir / "step_labeling_debug.jsonl")
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        debug_log_path = args.debug_log

    if debug_log_path:
        Path(debug_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(debug_log_path).open("w").close()  # truncate so one run = one debug file
        print(f"Debug log: {debug_log_path} (one JSONL line per Pass 1 step)", flush=True)
    limit = args.limit or len(records)
    processed = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for r in records:
            if processed >= limit:
                break
            trace_id = r.get("trace_id", "")
            mas_name = r.get("mas_name", "")
            failure_types = r.get("failure_types") or get_failure_types_set(r.get("mast_annotation") or {})
            if not failure_types:
                continue
            if use_steps_dataset:
                steps = r.get("steps", [])
            elif use_mad:
                trajectory = r.get("trajectory")
                steps, _ = get_steps_from_trace(data={"trajectory": trajectory}, task_type=mas_name)
            else:
                path = r.get("path")
                if not path:
                    continue
                steps, _ = get_steps_from_trace(path=Path(path))
            if not steps:
                continue
            print(f"Trace {processed + 1}/{min(limit, len(records))}: {trace_id or '(no id)'} ({mas_name}, {len(steps)} steps, {len(failure_types)} failure types)", flush=True)
            result = run_labeling_for_trace(
                steps,
                failure_type_ids=failure_types,
                definitions_text=definitions_text or None,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                adjudicate=not args.no_adjudicate,
                use_offline=use_offline,
                llm=llm,
                debug_log_path=Path(debug_log_path) if debug_log_path else None,
                trace_id=trace_id,
            )
            out_f.write(json.dumps({
                "trace_id": trace_id,
                "mas_name": mas_name,
                "failure_type_to_first_step": result,
            }, ensure_ascii=False) + "\n")
            out_f.flush()
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed} traces.")

    print(f"Done. Wrote {processed} results to {out_path}.")


if __name__ == "__main__":
    main()
