"""
eval/full_run_eval_with_graph_api_deepinfra.py — E3 (MAST +CG) via DeepInfra.

DeepInfra variant of the one-pass in-prompt graph guidance pipeline for MAST
(the +CG / "with_graph" arm). Mirrors eval/full_run_eval_with_graph.py (vLLM)
but swaps the inference path for DeepInfra's OpenAI-compatible endpoint
(https://api.deepinfra.com/v1/openai) with sequential calls + a sliding-window
RPM limiter. DeepInfra is pay-per-token (not fairshare), so the limiter is
loose by default; tighten --rpm if 429s appear.

Prompts, edge loading, and graph guidance formatting are imported verbatim
from eval/full_run_eval_with_graph.py so the +CG arm stays in sync across
backends. Only inference, auth, rate-limiting, and the per-record
write-as-you-go loop differ from the vLLM script.

Auth: expects DEEPINFRA_API_KEY (or API_KEY) in the environment.
    set -a; source /data/wang/junh/.cache/keys/deepinfra.sh; set +a
    export DEEPINFRA_API_KEY="$API_KEY"

Usage (run from MAST/):
    python eval/full_run_eval_with_graph_api_deepinfra.py --causal_only
    python eval/full_run_eval_with_graph_api_deepinfra.py \\
        --model openai/gpt-oss-20b --corr_threshold 0.5
    python eval/full_run_eval_with_graph_api_deepinfra.py \\
        --model google/gemma-3-27b-it --random_edges --random_n 11
"""

import os
import re
import sys
import json
import time
import argparse
from collections import deque
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI
from tqdm import tqdm

# Reuse helpers from the sibling vLLM script. vllm imports cleanly without a
# live GPU; only LLM() construction needs CUDA. Same trick as
# eval/full_run_eval_graph_inject_api_deepinfra.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_run_eval_with_graph import (  # noqa: E402
    MAST_MODES, MAST_NAMES,
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_STABILITY_GRAPH, DEFAULT_EFFECT_EDGES, DEFAULT_SUPPES_GRAPH,
    load_graph_edges, format_graph_guidance,
    format_trace, get_prompt,
    strip_thinking, parse_response,
)


DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_MODEL      = "google/gemma-3-27b-it"


# ---------------------------------------------------------------------------
# Rate limiter (sliding-window; single RPM rule by default)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window limiter for (max_count, window_seconds) rules.

    Defaults here are loose (DeepInfra is paid, not fairshare). The limiter is
    kept so we can tighten via --rpm if 429s start appearing. Sequential use
    only — no thread safety.
    """
    def __init__(self, limits):
        self.limits = limits
        self.longest_window = max(w for _, w in limits) if limits else 0
        self.times: deque = deque()
        self.n_total = 0

    def acquire(self):
        if not self.limits:
            self.n_total += 1
            return
        while True:
            now = time.time()
            while self.times and now - self.times[0] > self.longest_window:
                self.times.popleft()
            sleep_for = 0.0
            offender = None
            for max_n, window in self.limits:
                in_win = [t for t in self.times if now - t < window]
                if len(in_win) >= max_n:
                    target = in_win[len(in_win) - max_n]
                    delta = target + window - now + 0.5
                    if delta > sleep_for:
                        sleep_for = delta
                        offender = (max_n, window, len(in_win))
            if sleep_for <= 0:
                self.times.append(time.time())
                self.n_total += 1
                return
            mx, win, cur = offender
            print(f"[rate-limit] {cur}/{mx} in last {win}s — sleeping "
                  f"{sleep_for:.1f}s  (calls so far: {self.n_total})")
            time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# DeepInfra chat call with retry
# ---------------------------------------------------------------------------

def call_chat(client, model, prompt, max_tokens, limiter, max_retries=5):
    messages = [{"role": "user", "content": prompt}]
    last_err = None
    for attempt in range(max_retries):
        limiter.acquire()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            # DeepInfra wraps upstream 400s (notably context-length overflow)
            # inside a 500 InternalServerError, so the status check above misses
            # them and we'd burn 31s of backoff re-sending an oversized prompt.
            err_str = str(e)
            if "maximum context length" in err_str or "BadRequestError" in err_str:
                raise
            last_err = e
            backoff = min(60, 2 ** attempt)
            print(f"[retry {attempt + 1}/{max_retries}] {type(e).__name__}: "
                  f"{str(e)[:200]} — sleeping {backoff}s")
            time.sleep(backoff)
    raise RuntimeError(f"Failed after {max_retries} retries; last error: {last_err}")


# ---------------------------------------------------------------------------
# Per-record processing (one-pass +CG)
# ---------------------------------------------------------------------------

def process_record(
    r: dict,
    out_dir: str,
    client,
    model: str,
    max_tokens: int,
    limiter: RateLimiter,
    graph_guidance: str,
) -> int:
    """Run a single +CG call for one record and write the JSON.

    Returns 1 if a new record was written, 0 if already cached.
    """
    rec_id = r["_rec_id"]
    out_path = os.path.join(out_dir, f"{rec_id}.json")
    if os.path.exists(out_path):
        return 0

    trace_text = format_trace(r["steps"])
    prompt = get_prompt(trace_text, graph_guidance)

    raw_response = ""
    thinking = ""
    error_str = None
    try:
        full_text = call_chat(client, model, prompt, max_tokens, limiter)
        thinking, raw_response = strip_thinking(full_text)
        predictions = parse_response(raw_response)
    except Exception as e:
        print(f"[ERROR] rec {rec_id}: call failed: {e}")
        predictions = {m: 0 for m in MAST_MODES}
        error_str = str(e)[:300]

    out = {
        "rec_id":       rec_id,
        "trace_id":     r.get("trace_id"),
        "predictions":  predictions,
        "raw_response": raw_response,
        "meta": {
            "api":   "deepinfra",
            "model": model,
        },
    }
    if thinking:
        out["thinking"] = thinking
    if error_str:
        out["error"] = error_str

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="E3: MAST yes/no with static causal graph guidance (DeepInfra API)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="DeepInfra model id (e.g. google/gemma-3-27b-it, "
                         "openai/gpt-oss-20b, openai/gpt-oss-120b, "
                         "mistralai/Mistral-Small-3.1-24B-Instruct-2503).")
    ap.add_argument("--input", default="data/annotation/annotation_ag2_filtered.jsonl")
    ap.add_argument("--output_dir", default="outputs_full")
    ap.add_argument("--max_tokens", type=int, default=2000,
                    help="Auto-bumped to 16000 for reasoning models "
                         "(gpt-oss, qwq, deepseek-r1, qwenlong, *-thinking-*).")
    ap.add_argument("--model_tag", type=str, default=None,
                    help="Override the model tag used in the output directory name")

    # Graph variant selectors (mirrors the vLLM script's signature)
    ap.add_argument("--causal_only", action="store_true",
                    help="Use only intervention-validated edges from effect_edges.json")
    ap.add_argument("--corr_threshold", type=float, default=1.0,
                    help="If < 1.0, use UNION ablation: (Suppes geomean ≥ τ) ∪ "
                         "(validated causal edges). Mutually exclusive with --causal_only.")
    ap.add_argument("--edge_threshold", type=float, default=DEFAULT_EDGE_THRESHOLD,
                    help="Min geomean score for observational edges (default: 0.2)")
    ap.add_argument("--random_edges", action="store_true",
                    help="Random-N baseline: sample edges from MAST taxonomy minus Suppes graph.")
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--random_n", type=int, default=11,
                    help="Number of random edges to sample (default 11, matches causal-only).")
    ap.add_argument("--stability_graph", type=str, default=None)
    ap.add_argument("--effect_edges", type=str, default=None)
    ap.add_argument("--suppes_graph", type=str, default=None)

    # DeepInfra rate-limit knob (soft cap; loose default since DeepInfra is paid)
    ap.add_argument("--rpm", type=int, default=600,
                    help="Soft RPM cap (DeepInfra is paid; default is loose). "
                         "Set to 0 to disable the limiter entirely.")
    ap.add_argument("--max_retries", type=int, default=5)

    # Sweep / smoke-test
    ap.add_argument("--limit_traces", type=int, default=None,
                    help="Only process the first N pending traces (smoke test).")

    # CLI-parity no-ops with the vLLM script + sweep driver
    ap.add_argument("--span_index", action="store_true",
                    help="Accepted for CLI parity; no-op (MAST has no location prediction).")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Accepted for CLI parity; no-op on DeepInfra "
                         "(reasoning behavior inferred from model name).")

    args = ap.parse_args()

    # Auth
    api_key = os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        print("ERROR: DEEPINFRA_API_KEY (or API_KEY) env var not set.", file=sys.stderr)
        print("  set -a; source /data/wang/junh/.cache/keys/deepinfra.sh; set +a", file=sys.stderr)
        print('  export DEEPINFRA_API_KEY="$API_KEY"', file=sys.stderr)
        sys.exit(1)

    # Mutual-exclusion check (matches vLLM script)
    mode_count = sum([args.causal_only, args.corr_threshold < 1.0, args.random_edges])
    if mode_count > 1:
        ap.error("--causal_only, --corr_threshold, and --random_edges are mutually exclusive")

    # Reasoning-model detection: bumps max_tokens budget. Kept separate from
    # is_thinking_model below (which controls the output-dir "-thinking" suffix
    # and must match the sweep script's THINKING_SUFFIX rule).
    is_reasoning_model = bool(re.search(
        r"(gpt-oss|qwenlong|-l1-|deepseek-r1|qwq|thinking)",
        args.model, re.IGNORECASE,
    ))
    if is_reasoning_model and args.max_tokens <= 2000:
        print(f"[INFO] reasoning model ({args.model}); bumping max_tokens 2000 -> 16000")
        args.max_tokens = 16000

    is_thinking_model = "thinking" in args.model.lower()

    # Graph
    stability_path = Path(args.stability_graph) if args.stability_graph else DEFAULT_STABILITY_GRAPH
    effect_path    = Path(args.effect_edges)    if args.effect_edges    else DEFAULT_EFFECT_EDGES
    suppes_path    = Path(args.suppes_graph)    if args.suppes_graph    else DEFAULT_SUPPES_GRAPH
    edges = load_graph_edges(
        threshold       = args.edge_threshold,
        causal_only     = args.causal_only,
        corr_threshold  = args.corr_threshold,
        stability_graph = stability_path,
        effect_edges    = effect_path,
        suppes_graph    = suppes_path,
        random_edges    = args.random_edges,
        random_seed     = args.random_seed,
        random_n        = args.random_n,
    )
    if args.random_edges:
        mode_str  = f"random{args.random_n}_seed{args.random_seed} (null-graph control)"
        graph_tag = f"random{args.random_n}_seed{args.random_seed}"
    elif args.causal_only:
        mode_str  = "causal_only"
        graph_tag = "causal_only"
    elif args.corr_threshold < 1.0:
        mode_str  = f"corr>={args.corr_threshold} (Suppes ∪ validated)"
        graph_tag = f"corr{args.corr_threshold}"
    else:
        mode_str  = f"geomean>={args.edge_threshold}"
        graph_tag = f"t{args.edge_threshold}"
    print(f"Graph: {len(edges)} edges ({mode_str})")
    for src, dst, w in edges[:15]:
        print(f"  {src}({MAST_NAMES[src]}) -> {dst}({MAST_NAMES[dst]})  ({w:.3f})")
    if len(edges) > 15:
        print(f"  ... and {len(edges) - 15} more")

    graph_guidance = format_graph_guidance(
        edges, causal_only=args.causal_only, random_edges=args.random_edges,
    )

    # Records
    records = []
    with open(args.input) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            r["_rec_id"] = f"{idx:04d}"
            records.append(r)
    print(f"\nLoaded {len(records)} traces from {args.input}")

    # Output dir (matches vLLM convention)
    model_tag = args.model_tag if args.model_tag else args.model.replace("/", "-")
    thinking_suffix = "-thinking" if is_thinking_model else ""
    out_dir = os.path.join(
        args.output_dir,
        f"{model_tag}-yesno-with-graph-codename-{graph_tag}{thinking_suffix}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    pending = [r for r in records
               if not os.path.exists(os.path.join(out_dir, f"{r['_rec_id']}.json"))]
    print(f"Pending: {len(pending)} (skipping {len(records) - len(pending)} already done)")
    if args.limit_traces is not None:
        pending = pending[:args.limit_traces]
        print(f"limit_traces={args.limit_traces}; will process {len(pending)} traces")
    if not pending:
        print("Nothing to do.")
        return

    print(f"Estimated calls: {len(pending)} (one pass per trace).")

    # Inference
    client  = OpenAI(base_url=DEEPINFRA_BASE_URL, api_key=api_key)
    limiter = RateLimiter([(args.rpm, 60)] if args.rpm > 0 else [])

    n_written = 0
    for r in tqdm(pending, desc="traces"):
        n_written += process_record(
            r, out_dir, client, args.model, args.max_tokens, limiter, graph_guidance,
        )

    print(f"\n✓ Outputs saved to {out_dir}/")
    print(f"  Records written : {n_written}")
    print(f"  Total API calls : {limiter.n_total}")
    print(f"  Next: python eval/calculate_scores_yesno.py --pred_dir \"{out_dir}\"")


if __name__ == "__main__":
    main()
