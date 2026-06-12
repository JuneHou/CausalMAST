"""
eval/full_run_eval_graph_inject_api_arc.py — E4 (MAST +GI) via VT ARC LLM API.

ARC variant of the two-pass dynamic graph-injection pipeline for MAST. Mirrors
eval/full_run_eval_graph_inject.py (vLLM) but swaps the inference path for the
ARC OpenAI-compatible endpoint (https://llm-api.arc.vt.edu/api/v1/) with
sequential calls + a sliding-window rate limiter honoring ARC's fairshare:
    30 req/min, 1000 req/hr, 3000 req per rolling 3 hr.

Prompts / parsing / graph logic are imported verbatim from
eval/full_run_eval_graph_inject.py so the two stay in sync. Only inference,
auth, rate-limiting, and the per-record write-as-you-go loop differ.

Auth: expects ARC_LLM_API_KEY (or API_KEY) in the environment.
    set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a
    export ARC_LLM_API_KEY="$API_KEY"

Usage (run from MAST/):
    python eval/full_run_eval_graph_inject_api_arc.py --causal_only
    python eval/full_run_eval_graph_inject_api_arc.py --corr_threshold 0.5
    python eval/full_run_eval_graph_inject_api_arc.py --random_edges --random_n 11
"""

import os
import sys
import json
import time
import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm

# Reuse helpers from the sibling vLLM script. vllm imports cleanly without a
# live GPU; only LLM() construction needs CUDA. Same trick as
# baselines/who&when/causal/run_who_and_when_graph_inject_api_arc.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_run_eval_graph_inject import (  # noqa: E402
    MAST_MODES, MAST_NAMES,
    DEFAULT_EDGE_THRESHOLD, DEFAULT_PROPAGATION_THRESHOLD,
    DEFAULT_STABILITY_GRAPH, DEFAULT_EFFECT_EDGES, DEFAULT_SUPPES_GRAPH,
    load_graph_edges, propagate_confidence,
    format_trace, get_pass1_prompt, get_pass2_prompt,
    strip_thinking, parse_response, parse_pass2_response,
)


ARC_BASE_URL  = "https://llm-api.arc.vt.edu/api/v1/"
DEFAULT_MODEL = "gpt-oss-120b"


# ---------------------------------------------------------------------------
# Rate limiter (sliding-window; multiple rules)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window limiter for multiple (max_count, window_seconds) rules.

    Sequential use only — no thread safety. On acquire() we sleep just long
    enough that every rule permits one more call, then record the call.
    """
    def __init__(self, limits):
        self.limits = limits
        self.longest_window = max(w for _, w in limits)
        self.times: deque = deque()
        self.n_total = 0

    def acquire(self):
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
# ARC chat call with retry
# ---------------------------------------------------------------------------

def call_chat(client, model, prompt, max_tokens, limiter, max_retries=5, temperature=0.0, seed=0):
    messages = [{"role": "user", "content": prompt}]
    last_err = None
    for attempt in range(max_retries):
        limiter.acquire()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            backoff = min(60, 2 ** attempt)
            print(f"[retry {attempt + 1}/{max_retries}] {type(e).__name__}: "
                  f"{str(e)[:200]} — sleeping {backoff}s")
            time.sleep(backoff)
    raise RuntimeError(f"Failed after {max_retries} retries; last error: {last_err}")


# ---------------------------------------------------------------------------
# Two-pass per-record processing
# ---------------------------------------------------------------------------

def process_record(
    r: dict,
    out_dir: str,
    client,
    model: str,
    max_tokens: int,
    limiter: RateLimiter,
    edges: List[Tuple[str, str, float]],
    propagation_threshold: float,
    temperature: float = 0.0,
    seed: int = 0,
) -> Dict[str, int]:
    """Run pass-1 then optionally pass-2 for one record and write the JSON.

    Returns counts dict {triggered: 0/1, upgraded: 0/1} for run-level stats.
    """
    rec_id = r["_rec_id"]
    out_path = os.path.join(out_dir, f"{rec_id}.json")
    if os.path.exists(out_path):
        return {"triggered": 0, "upgraded": 0}

    trace_text = format_trace(r["steps"])

    # Pass 1
    p1_raw = ""
    p1_thinking = ""
    p1_error = None
    try:
        full_p1 = call_chat(client, model, get_pass1_prompt(trace_text),
                            max_tokens, limiter, temperature=temperature, seed=seed)
        p1_thinking, p1_raw = strip_thinking(full_p1)
        p1_pred = parse_response(p1_raw)
    except Exception as e:
        print(f"[ERROR] rec {rec_id}: pass1 failed: {e}")
        p1_pred = {m: 0 for m in MAST_MODES}
        p1_error = str(e)[:300]

    # Propagation
    detected = [m for m, v in p1_pred.items() if v == 1]
    filtered_edges = (
        propagate_confidence(detected, edges, propagation_threshold)
        if (detected and edges) else []
    )
    p2_triggered = bool(filtered_edges)

    # Pass 2
    merged = dict(p1_pred)
    p2_upgrades: Dict[str, int] = {}
    p2_raw = None
    p2_error = None
    if p2_triggered:
        target_cats = list(dict.fromkeys(dst for _, dst, _ in filtered_edges))
        try:
            full_p2 = call_chat(client, model,
                                get_pass2_prompt(trace_text, detected, filtered_edges),
                                max_tokens, limiter, temperature=temperature, seed=seed)
            _, p2_raw = strip_thinking(full_p2)
            p2_pred = parse_pass2_response(p2_raw, target_cats)
        except Exception as e:
            print(f"[ERROR] rec {rec_id}: pass2 failed: {e} (keeping pass-1)")
            p2_pred = {}
            p2_error = str(e)[:300]

        for cat, val in p2_pred.items():
            if val == 1 and merged.get(cat, 0) == 0:
                merged[cat] = 1
                p2_upgrades[cat] = 1

    out = {
        "rec_id":           rec_id,
        "trace_id":         r.get("trace_id"),
        "predictions":      merged,
        "raw_response":     p1_raw,
        "pass2_triggered":  p2_triggered,
        "pass2_upgrades":   p2_upgrades,
        "pass2_raw":        p2_raw,
        "meta": {
            "api":   "arc",
            "model": model,
        },
    }
    if p1_thinking:
        out["thinking"] = p1_thinking
    if p1_error:
        out["pass1_error"] = p1_error
    if p2_error:
        out["pass2_error"] = p2_error

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return {"triggered": int(p2_triggered), "upgraded": int(bool(p2_upgrades))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="E4: MAST yes/no with dynamic 2-pass graph injection (ARC API)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="ARC model name (gpt-oss-120b, gpt-oss-120b-thinking-high, kimi-k2.6, ...)")
    ap.add_argument("--input", default="data/annotation/annotation_ag2_filtered.jsonl")
    ap.add_argument("--output_dir", default="outputs_full")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Decoding temperature. >0 enables stochastic sampling; "
                         "re-invoke the script multiple times to collect i.i.d. samples.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Per-request sampling seed; pass distinct values across "
                         "invocations to obtain i.i.d. samples at temperature>0.")
    ap.add_argument("--max_tokens", type=int, default=2000,
                    help="Auto-bumped to 16000 if the model name contains 'thinking'.")
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
    ap.add_argument("--propagation_threshold", type=float, default=DEFAULT_PROPAGATION_THRESHOLD,
                    help="Min boosted score to trigger Pass 2 for a target category (default: 0.1)")
    ap.add_argument("--random_edges", action="store_true",
                    help="Random-N baseline: sample edges from MAST taxonomy minus Suppes graph.")
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--random_n", type=int, default=11,
                    help="Number of random edges to sample (default 11, matches causal-only).")
    ap.add_argument("--stability_graph", type=str, default=None)
    ap.add_argument("--effect_edges", type=str, default=None)
    ap.add_argument("--suppes_graph", type=str, default=None)

    # ARC rate-limit knobs
    ap.add_argument("--rpm",  type=int, default=30,
                    help="Requests per minute  (ARC fairshare: 30).")
    ap.add_argument("--rph",  type=int, default=1000,
                    help="Requests per hour    (ARC fairshare: 1000).")
    ap.add_argument("--rp3h", type=int, default=3000,
                    help="Requests per 3 hours (ARC fairshare: 3000).")
    ap.add_argument("--max_retries", type=int, default=5)

    # Sweep / smoke-test
    ap.add_argument("--limit_traces", type=int, default=None,
                    help="Only process the first N pending traces (smoke test).")

    # CLI-parity no-op with the vLLM script + sweep driver
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Accepted for CLI parity; no-op on ARC (thinking inferred from model name).")

    args = ap.parse_args()

    # Auth
    api_key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        print("ERROR: ARC_LLM_API_KEY (or API_KEY) env var not set.", file=sys.stderr)
        print("  set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a", file=sys.stderr)
        print('  export ARC_LLM_API_KEY="$API_KEY"', file=sys.stderr)
        sys.exit(1)

    # Mutual-exclusion check (matches vLLM script)
    mode_count = sum([args.causal_only, args.corr_threshold < 1.0, args.random_edges])
    if mode_count > 1:
        ap.error("--causal_only, --corr_threshold, and --random_edges are mutually exclusive")

    # Thinking models need bigger budget for reasoning chain
    is_thinking_model = "thinking" in args.model.lower()
    if is_thinking_model and args.max_tokens <= 2000:
        print(f"[INFO] thinking model ({args.model}); bumping max_tokens 2000 -> 16000")
        args.max_tokens = 16000

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
        f"{model_tag}-yesno-graph-inject-codename-{graph_tag}{thinking_suffix}",
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

    # Rate-limit floor estimate (pass-1 always; pass-2 up to once per trace)
    floor_min = (2 * len(pending)) / args.rpm
    print(f"Estimated calls: up to {2 * len(pending)} "
          f"(pass-1: {len(pending)}, pass-2: ≤{len(pending)}). "
          f"Rate-limit floor at {args.rpm}/min: ~{floor_min:.0f} minutes mandatory wait.")

    # Inference
    client  = OpenAI(base_url=ARC_BASE_URL, api_key=api_key)
    limiter = RateLimiter([(args.rpm, 60), (args.rph, 3600), (args.rp3h, 10800)])

    n_triggered = 0
    n_upgraded  = 0
    for r in tqdm(pending, desc="traces"):
        stats = process_record(
            r, out_dir, client, args.model, args.max_tokens, limiter,
            edges, args.propagation_threshold, temperature=args.temperature,
            seed=args.seed,
        )
        n_triggered += stats["triggered"]
        n_upgraded  += stats["upgraded"]

    print(f"\n✓ Outputs saved to {out_dir}/")
    print(f"  Pass 2 triggered: {n_triggered} traces, {n_upgraded} with upgrades")
    print(f"  Total API calls : {limiter.n_total}")
    print(f"  Next: python eval/calculate_scores_yesno.py --pred_dir \"{out_dir}\"")


if __name__ == "__main__":
    main()
