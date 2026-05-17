"""
baselines/who&when/causal/run_who_and_when_with_graph_api_arc.py

ARC-API variant of +CG (one-pass causal graph guidance) for Who&When.
Uses VT ARC's free OpenAI-compatible endpoint
(https://llm-api.arc.vt.edu/api/v1/) with sequential calls + a sliding-window
rate limiter honoring ARC's fairshare caps:
    30 req/min, 1000 req/hr, 3000 req per rolling 3 hr.

Prompts/parsing/graph logic are imported verbatim from the sibling vLLM
script (run_who_and_when_with_graph_vllm.py) so the two stay in sync.
Only the inference path changes: OpenAI client + per-record sequential loop
that writes one file per trace immediately, so the run is resumable.

Auth: expects ARC_LLM_API_KEY (or API_KEY) in the environment. Setup:
    set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a
    export ARC_LLM_API_KEY="$API_KEY"

Usage (run from MAST/):
    python "baselines/who&when/causal/run_who_and_when_with_graph_api_arc.py" \\
        --variant w1 --causal_only \\
        --output_dir "baselines/who&when/outputs"
"""

import os
import sys
import json
import time
import argparse
from collections import deque
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# Reuse helpers from the sibling vLLM script. The vllm package is imported
# at module load there; importing vllm without a live GPU is supported
# (only LLM() construction needs CUDA).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_who_and_when_with_graph_vllm import (  # noqa: E402
    MAST_MODES, MAST_NAMES,
    DEFAULT_EFFECT_EDGES, DEFAULT_SUPPES_GRAPH,
    load_graph_edges, format_graph_guidance,
    format_trace, extract_task_description, build_cumulative_text,
    get_w1_prompt, get_w2_step_prompt,
    strip_thinking, parse_response,
)


ARC_BASE_URL  = "https://llm-api.arc.vt.edu/api/v1/"
DEFAULT_MODEL = "gpt-oss-120b"


class RateLimiter:
    """Sliding-window limiter for multiple (max_count, window_seconds) rules.

    On acquire() we sleep just long enough that every rule permits one more
    call, then record the call. Sequential use only (no thread safety).
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
            last_err = e
            backoff = min(60, 2 ** attempt)
            print(f"[retry {attempt + 1}/{max_retries}] {type(e).__name__}: "
                  f"{str(e)[:200]} — sleeping {backoff}s")
            time.sleep(backoff)
    raise RuntimeError(f"Failed after {max_retries} retries; last error: {last_err}")


def run_w1_one(r, client, model, max_tokens, limiter, graph_guidance):
    trace_text = format_trace(r["steps"])
    task_desc  = extract_task_description(r["steps"])
    prompt = get_w1_prompt(trace_text, task_desc, graph_guidance)
    full_text = call_chat(client, model, prompt, max_tokens, limiter)
    thinking, raw = strip_thinking(full_text)
    return {
        "rec_id":       r["_rec_id"],
        "trace_id":     r.get("trace_id"),
        "predictions":  parse_response(raw),
        "raw_response": raw,
        "variant":      "w1",
        "meta":         {"n_calls": 1, "n_steps": len(r.get("steps", []))},
        "_thinking":    thinking,
    }


def run_w2_one(r, client, model, max_tokens, limiter,
               max_history_chars, graph_guidance):
    steps = r.get("steps", [])
    n_steps = len(steps)
    task_desc = extract_task_description(steps)
    step_preds, step_raw, step_think = [], [], []
    for step_idx, step in enumerate(steps):
        cumulative = build_cumulative_text(steps, step_idx + 1, max_history_chars)
        prompt = get_w2_step_prompt(
            cumulative_text=cumulative,
            step_id=step["id"],
            step_num=step_idx + 1,
            total_steps=n_steps,
            task_description=task_desc,
            graph_guidance=graph_guidance,
        )
        full_text = call_chat(client, model, prompt, max_tokens, limiter)
        thinking, raw = strip_thinking(full_text)
        step_preds.append(parse_response(raw))
        step_raw.append(raw)
        step_think.append(thinking if thinking else None)

    trace_pred = {m: 0 for m in MAST_MODES}
    for sp in step_preds:
        for m in MAST_MODES:
            if sp.get(m, 0) == 1:
                trace_pred[m] = 1

    per_step_meta = []
    for step_idx, step in enumerate(steps):
        entry = {
            "step_idx":    step_idx,
            "step_id":     step["id"],
            "predictions": step_preds[step_idx],
        }
        if step_think[step_idx]:
            entry["thinking"] = step_think[step_idx]
        per_step_meta.append(entry)

    return {
        "rec_id":       r["_rec_id"],
        "trace_id":     r.get("trace_id"),
        "predictions":  trace_pred,
        "raw_response": "\n\n---\n\n".join(
            f"[step {i}] {raw}" for i, raw in enumerate(step_raw) if raw is not None
        ),
        "variant":      "w2",
        "meta": {
            "n_calls":  n_steps,
            "n_steps":  n_steps,
            "per_step": per_step_meta,
        },
        "_thinking":    None,
    }


def main():
    ap = argparse.ArgumentParser(
        description="MAST W&W + causal graph guidance (+CG) via ARC LLM API "
                    "(sequential, rate-limited).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--variant", choices=["w1", "w2"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="ARC model name (gpt-oss-120b, "
                         "gpt-oss-120b-thinking-high, kimi-k2.6, ...)")
    ap.add_argument("--input", default="data/annotation/annotation_ag2_filtered.jsonl")
    ap.add_argument("--output_dir", default="baselines/who&when/outputs")
    ap.add_argument("--max_tokens", type=int, default=2000,
                    help="Auto-bumped to 16000 if the model name contains 'thinking'.")
    ap.add_argument("--model_tag", type=str, default=None)
    ap.add_argument("--w2_max_history_chars", type=int, default=80000)
    ap.add_argument("--causal_only", action="store_true")
    ap.add_argument("--corr_threshold", type=float, default=1.0)
    ap.add_argument("--effect_edges", type=str, default=None)
    ap.add_argument("--suppes_graph", type=str, default=None)
    ap.add_argument("--limit_traces", type=int, default=None,
                    help="Only process the first N pending traces (smoke test).")
    ap.add_argument("--rpm",  type=int, default=30,
                    help="Requests per minute  (ARC fairshare: 30).")
    ap.add_argument("--rph",  type=int, default=1000,
                    help="Requests per hour    (ARC fairshare: 1000).")
    ap.add_argument("--rp3h", type=int, default=3000,
                    help="Requests per 3 hours (ARC fairshare: 3000).")
    ap.add_argument("--max_retries", type=int, default=5)
    args = ap.parse_args()

    api_key = os.environ.get("ARC_LLM_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        print("ERROR: ARC_LLM_API_KEY (or API_KEY) env var not set.", file=sys.stderr)
        print("  set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a", file=sys.stderr)
        print('  export ARC_LLM_API_KEY="$API_KEY"', file=sys.stderr)
        sys.exit(1)

    is_thinking_model = "thinking" in args.model.lower()
    if is_thinking_model and args.max_tokens <= 2000:
        print(f"[INFO] thinking model ({args.model}); "
              f"bumping max_tokens 2000 -> 16000")
        args.max_tokens = 16000

    if sum([args.causal_only, args.corr_threshold < 1.0]) != 1:
        ap.error("Must specify exactly one of --causal_only or --corr_threshold < 1.0")

    effect_path = Path(args.effect_edges) if args.effect_edges else DEFAULT_EFFECT_EDGES
    suppes_path = Path(args.suppes_graph) if args.suppes_graph else DEFAULT_SUPPES_GRAPH
    edges = load_graph_edges(
        causal_only    = args.causal_only,
        corr_threshold = args.corr_threshold,
        effect_edges   = effect_path,
        suppes_graph   = suppes_path,
    )
    graph_guidance = format_graph_guidance(edges, causal_only=args.causal_only)
    graph_tag = "causal_only" if args.causal_only else f"corr{args.corr_threshold}"
    print(f"Graph: {len(edges)} edges ({graph_tag})")
    for src, dst, w in edges[:15]:
        print(f"  {src}({MAST_NAMES[src]}) -> {dst}({MAST_NAMES[dst]})  ({w:.3f})")
    if len(edges) > 15:
        print(f"  ... and {len(edges) - 15} more")

    records = []
    with open(args.input) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            r["_rec_id"] = f"{idx:04d}"
            records.append(r)
    print(f"\nLoaded {len(records)} traces from {args.input}")

    model_tag = args.model_tag if args.model_tag else args.model.replace("/", "-")
    thinking_suffix   = "-thinking" if is_thinking_model else ""
    out_dir = os.path.join(
        args.output_dir,
        f"{model_tag}-yesno-who_and_when_{args.variant}_graph_{graph_tag}{thinking_suffix}",
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

    if args.variant == "w1":
        est_calls = len(pending)
    else:
        est_calls = sum(len(r.get("steps", [])) for r in pending)
    print(f"Estimated calls: {est_calls}")
    print(f"Rate-limit floor at 30/min: ~{est_calls / 30:.0f} minutes mandatory wait;"
          f" hourly cap (1000/hr) may extend this.")

    client = OpenAI(base_url=ARC_BASE_URL, api_key=api_key)
    limiter = RateLimiter([(args.rpm, 60), (args.rph, 3600), (args.rp3h, 10800)])

    for r in tqdm(pending, desc="traces"):
        rec_id = r["_rec_id"]
        try:
            if args.variant == "w1":
                out = run_w1_one(r, client, args.model, args.max_tokens,
                                 limiter, graph_guidance)
            else:
                out = run_w2_one(r, client, args.model, args.max_tokens,
                                 limiter, args.w2_max_history_chars,
                                 graph_guidance)
        except Exception as e:
            print(f"[ERROR] rec {rec_id}: failed: {e} — skipping")
            continue

        thinking = out.pop("_thinking", None)
        out["meta"]["graph_tag"] = graph_tag
        out["meta"]["api"] = "arc"
        out["meta"]["model"] = args.model
        if thinking:
            out["thinking"] = thinking
        with open(os.path.join(out_dir, f"{rec_id}.json"), "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nOutputs saved to {out_dir}/")
    print(f"  Total API calls: {limiter.n_total}")
    print(f"  Next: python eval/calculate_scores_yesno.py --pred_dir \"{out_dir}\"")


if __name__ == "__main__":
    main()
