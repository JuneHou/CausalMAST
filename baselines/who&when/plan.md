# Who&When Adaptation Plan for MAST Benchmark

**Source paper:** "Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems" (Yin et al., ICML 2025 Spotlight)
**arXiv:** 2505.00212
**GitHub:** https://github.com/mingyin1/Agents_Failure_Attribution
**Companion adaptation (TRAIL, span-level):** `/data/wang/junh/githubs/trail-benchmark/baselines/who_and_when/`
**Last updated:** 2026-05-01

---

## 1. What Who&When does originally

Who&When defines three localization **strategies** for multi-agent failure attribution. Each strategy answers the same question — *which agent caused the failure, at which step?* — but uses a different search policy over the trace:

| Variant | Strategy | LLM calls per trace |
|---|---|---|
| **W1 — All-at-Once** | Single prompt sees the full trace and emits the responsible agent + decisive step in one shot. | 1 |
| **W2 — Step-by-Step** | Walks the trace one step at a time, asking yes/no at each step. **Stops at the first "yes"** and reports that step. | up to N |
| **W3 — Binary Search** | Recursively halves the trace, asking the model which half contains the decisive error. Forces a choice between upper/lower at every level. | ≈ ⌈log₂ N⌉ |

All three produce **a single (agent, step) prediction** because the original task assumes exactly one responsible agent and one decisive step per trace.

---

## 2. What the TRAIL adaptation already changed

`trail-benchmark/baselines/who_and_when/run_who_and_when_vllm.py` ports the three strategies to TRAIL, where each trace can have **0..K (category, span_id) errors** drawn from a 19-class taxonomy. The mapping is:

- "Who" (which agent) → **What** (which leaf category)
- "When" (which step) → **Where** (which `span_id`)

To break the single-error assumption it makes three changes:

| | TRAIL change |
|---|---|
| W1 | Removed `errors = [errors[0]]` truncation; prompt now asks "for each of the 19 categories, decide independently whether it is present, and report the span where it first occurs." |
| W2 | Removed the early-exit `break` on first "yes"; per-step prompt returns a JSON list of zero-or-more categories; results aggregated by deduping `(category, span_id)` across all spans. |
| W3 | Original bisects to find the single most-critical step. Adapted version runs **per-label** recursive bisection (one bisection tree per category), allows **both halves to be positive**, and recurses into all positive halves. Final leaf call asks `present: true/false` for that one label at that one span. |

So the TRAIL version still produces **span-level** annotations, just multi-label.

---

## 3. How MAST is different from TRAIL

MAST is also multi-label, but:

| Aspect | TRAIL | MAST |
|---|---|---|
| Annotation granularity | Per-span: each error has a `span_id` location | **Trace-level only**: `mast_annotation` is a flat dict `{category: 0/1}` |
| Taxonomy | 19 leaf categories (TRAIL paper) | 13 leaf categories: 1.1-1.5, 2.1-2.4, 2.6, 3.1-3.3 (skipping 2.5) |
| Output schema | `{"errors": [{category, location, evidence, ...}], "scores": []}` | `{"predictions": {"1.1": 0/1, "1.2": 0/1, ..., "3.3": 0/1}}` |
| Scoring | W-F1 + Location Acc + Joint Acc (`calculate_scores.py`) | Per-category P/R/F1, macro/weighted F1, kappa (`eval/calculate_scores_yesno.py`) |
| Trace data | `data/processed/<split>.json` with rich span tree | `data/annotation/annotation_ag2_filtered.jsonl` — each record has a flat `steps: [{id, content}, ...]` list (3-N steps; pre-formatted text) |
| Existing baseline | TRAIL Baseline / +GI+SI prompts | `eval/run_eval_yesno_vllm.py` — single call, multi-label, definitions+examples in prompt |

The user-stated scope is **multi-label error detection, no step needed**. So in MAST the search granularity is "is category X present anywhere in this trace?" — there is no equivalent to TRAIL's `span_id` field to predict.

That changes the role of the three Who&When variants:

- **W1** is essentially what `run_eval_yesno_vllm.py` already does (full trace, all categories at once). We will reproduce it under the Who&When name as a clean baseline so the three variants share an interface.
- **W2** still gives N step-level prompts but their outputs collapse to the trace level: `trace_pred[c] = OR over step_pred[step][c]`.
- **W3** loses some of its motivation (no location to localize) but remains useful as a **focused-attention** strategy: for very long traces, a per-label bisection still surfaces stronger per-segment evidence than asking about the whole trace at once. The bisection short-circuits the moment any segment confirms the label.

---

## 4. Mapping of the three variants to MAST

For each variant, output is the **same JSON shape** as the existing yes/no baseline so `eval/calculate_scores_yesno.py` works without modification:

```json
{
  "rec_id": "0000",
  "trace_id": 3,
  "predictions": {"1.1": 0, "1.2": 0, ..., "3.3": 1},
  "raw_response": "...",
  "variant": "w1" | "w2" | "w3",
  "meta": { ... per-variant call counts, errors, ... }
}
```

### 4.1 Variant W1 — All-at-Once
**LLM calls per trace:** 1

A single prompt with the full trace and all 13 category definitions, asking for an independent yes/no per category. This mirrors the existing `run_eval_yesno_vllm.py` baseline; we keep it under the W1 name to make the three-variant ablation self-contained.

**Prompt (skeleton — reuse `DEFINITIONS` and `EXAMPLES` from `taxonomy_definitions_examples/`):**
```
You are analyzing a multiagent system trace for failure modes.
Read the definitions and examples below carefully.

FAILURE MODE DEFINITIONS:
{DEFINITIONS}

EXAMPLES OF FAILURE MODES:
{EXAMPLES}

The agent execution trace:
{trace_text}

This is a multi-label task. Zero, one, or multiple failure modes may be present
in the trace. For EACH of the 13 failure modes, make an INDEPENDENT decision:
is this failure mode present anywhere in the trace? Do NOT force a label if
there is no clear evidence.

Answer between the @@ symbols, exactly as shown:
@@
A. Summary: <one paragraph>
B. Task completed: <yes|no>
C.
1.1 Disobey Task Specification: <yes|no>
1.2 Disobey Role Specification: <yes|no>
... (all 13)
3.3 No or Incorrect Verification: <yes|no>
@@
```

**Difference from W1 in TRAIL:** no `location` field, no `span_id` index — outputs collapse to a 13-bit vector. Otherwise identical framing ("decide independently for each category").

### 4.2 Variant W2 — Step-by-Step (sequential scan, no early exit)
**LLM calls per trace:** N (one per step in `steps`)

For each step `step_i`, prompt the model with the **cumulative** conversation up to and including `step_i`, asking which of the 13 categories show evidence **at the current step** (multi-label). After all N calls, aggregate to trace level via OR.

**Per-step prompt:**
```
You are evaluating one step of a multiagent system trace.

FAILURE MODE DEFINITIONS:
{DEFINITIONS}

Conversation history up to and including the CURRENT step:
{cumulative_steps_up_to_i}

The CURRENT step is:
[{step_i.id}] {step_i.content}

Decide, for the CURRENT STEP only, which failure modes show direct evidence here.
A single step may exhibit zero, one, or multiple failure modes. Mark a mode only
if the current step itself provides evidence (not a downstream consequence of an
earlier step).

Answer between @@ symbols:
@@
1.1: <yes|no>
1.2: <yes|no>
... (all 13)
3.3: <yes|no>
@@
```

**Aggregation:**
```python
trace_pred = {c: 0 for c in MAST_MODES}
for i, step in enumerate(steps):
    step_pred = ask(prompt_for_step_i)        # 13-bit vector
    for c in MAST_MODES:
        trace_pred[c] |= step_pred[c]
```

**Differences from TRAIL W2:**
- No `span_id` field — only category labels per step, then OR across steps.
- No deduplication of `(category, span_id)` pairs needed.
- Cumulative-prefix budget: if `cumulative_steps_up_to_i` exceeds the model's context budget, keep only the most recent K steps as context (drop the oldest steps). MAST traces are often long (one HyperAgent record can have hundreds of steps), so this guard matters.
- Early-exit is **not** added (we want the full multi-label vector, not just the first hit).

**Cost note:** N calls per trace. AG2 traces in `annotation_ag2_filtered.jsonl` have a median of ~5-15 steps per trace (some far longer), so figure ~10-20× the W1 cost on average.

### 4.3 Variant W3 — Binary Search per Label (presence-only)
**LLM calls per trace:** ≈ 13 × ⌈log₂ N⌉ in the worst case, much less with short-circuit

Per label, recursively bisect the step list and ask the model whether the label is present in each half. Because MAST asks only for trace-level presence, **as soon as any segment confirms the label we set `trace_pred[c] = 1` and stop bisecting that label**. This is a strict simplification of TRAIL W3, which had to enumerate every span the label occurs at.

**Bisect prompt (per label, per interval):**
```
You are evaluating an excerpt of a multiagent system trace.

Target failure mode: {mode_id} {mode_name}
Definition: {mode_definition}

Steps {low}-{high} of the trace:
{steps_window}

Does "{mode_name}" occur somewhere in this excerpt?
Answer yes only if this excerpt contains DIRECT evidence of the target failure
mode. If the excerpt only shows downstream consequences of an error from before
this excerpt, answer no unless the failure mode independently recurs here.

Answer:
@@ {mode_id}: <yes|no> @@
```

**Algorithm (presence-only, with short-circuit):**
```python
def label_present(label, steps_window):
    if not steps_window:
        return False
    if len(steps_window) <= LEAF_SIZE:        # e.g., LEAF_SIZE = 1 or 2
        return ask_present(label, steps_window)
    mid = len(steps_window) // 2
    if ask_present(label, steps_window[:mid]):
        return True                            # short-circuit: trace-level only
    return ask_present(label, steps_window[mid:])

trace_pred = {}
for c in MAST_MODES:
    trace_pred[c] = int(label_present(c, steps))
```

**Differences from TRAIL W3:**
- TRAIL W3 returns **all** locations a label occurs at; we only need a single boolean per label, so we can short-circuit on the first positive segment.
- TRAIL asks `lower_half_present` AND `upper_half_present` in one call (both can be true) because both might contain occurrences. Here we ask **lower first**, then upper only if lower was negative. This roughly halves the call count.
- No final per-span localization call — the leaf decision is itself the answer.

**Cost note:** Worst case 13 × log₂ N calls, but the short-circuit pulls the typical cost down sharply when a label is actually present (often resolved at the first or second level).

---

## 5. Files to write

```
baselines/who&when/
├── plan.md                          ← this file
├── run_who_and_when_vllm.py         ← runner with --variant w1|w2|w3
└── outputs/                         ← per-variant prediction dirs (mirrors MAST/outputs/ schema)
```

`run_who_and_when_vllm.py` should:

1. Reuse `DEFINITIONS` and `EXAMPLES` loaded from `taxonomy_definitions_examples/`.
2. Reuse the `MAST_MODES = ["1.1",...,"3.3"]` list and the `parse_response` style of `eval/run_eval_yesno_vllm.py` for W1.
3. Add small per-variant parsers for W2 (per-step 13-bit vector) and W3 (single yes/no per call).
4. Write outputs to `outputs/<model_tag>-yesno-who_and_when_<variant>/<rec_id>.json` so the existing scorer pattern works:
   ```bash
   python eval/calculate_scores_yesno.py \
       --annotation data/annotation/annotation_ag2_filtered.jsonl \
       --pred_dir outputs/<model_tag>-yesno-who_and_when_<variant>
   ```
5. Skip records whose output file already exists (resume-safe, like the existing baseline).
6. Cap context: for W2, bound the cumulative prefix to a recent-K-steps window when total tokens exceed `max_model_len - max_tokens`.

---

## 6. Implementation steps

1. **Scaffold the runner.** Copy the structure of `eval/run_eval_yesno_vllm.py`. Add `--variant {w1,w2,w3}` argparse flag. Output dir name includes the variant.
2. **W1 (all-at-once).** Wire the existing yes/no prompt under `variant == "w1"`. Verify identical output schema to the baseline by spot-checking one record.
3. **W2 (step-by-step).** Build per-step prompts with cumulative context + recent-K-steps fallback. Parse a 13-bit vector per call. OR-aggregate. Save aggregated `predictions` plus a `meta.per_step_predictions` list for debugging.
4. **W3 (per-label bisection).** Implement `label_present(label, steps)` recursion with short-circuit. Per-label call count goes into `meta.calls_per_label`. Trace-level prediction is the OR.
5. **Validate.** Run all three variants on a 5-trace subset of `annotation_ag2_filtered.jsonl` and score with `eval/calculate_scores_yesno.py`. Confirm metric file is written for each variant.
6. **Full run.** Mistral-Small-3.1-24B-Instruct on full 393 traces, GPUs as configured. (User runs the actual model calls — this plan does not include LLM-calling commands per the env rules.)

---

## 7. Comparison table the runs are meant to support

All numbers per-category and macro F1, on the same 393-trace AG2 split, same model.

| Condition | Localization strategy | Calls/trace | Output | Comment |
|---|---|---|---|---|
| Existing baseline (`run_eval_yesno_vllm.py`) | Full trace, single pass | 1 | 13-bit vector | Reference point |
| Who&When **W1** | Full trace, single pass | 1 | 13-bit vector | Should be ≈ baseline |
| Who&When **W2** | Step-by-step, no early exit, OR-aggregate | N | 13-bit vector (+ per-step debug) | Tests whether per-step focus helps recall |
| Who&When **W3** | Per-label bisection, short-circuit | ≤ 13 × log₂ N | 13-bit vector (+ per-label call counts) | Tests whether per-label focused attention helps in long traces |

Reading the comparison: the gap between W1 and W2/W3 (if any) isolates the contribution of **how the trace is searched** rather than the prompt content, because the definitions/examples block is identical across variants. If W1 ≈ W2 ≈ W3, the conclusion is that for MAST's trace-level multi-label task, search strategy is not the bottleneck — and the gains seen elsewhere (e.g., the causal-graph injection variants in `eval/run_eval_with_graph.py`) come from added structure, not from how the model scans the trace.
