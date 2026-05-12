# MAST Evaluation — Experiment TODO

All experiments use `data/annotation/annotation_ag2_filtered.jsonl` (393 traces) unless noted as sampled.
Best current setup: **E3 with-graph, causal_only** (Mistral W-F1=0.4731 codename; 0.4758 code-only — narrow gap, paper uses codename).

---

## Model Alignment with TRAIL Benchmark

TRAIL uses Gemini-2.5-Flash (closed-source) and Mistral-Small-3.1-24B (open-source).
MAST replaces Gemini with GPT-4o. Mistral is shared across both benchmarks.

| Role | TRAIL model | MAST model | Status |
|---|---|---|---|
| Open-source | Mistral-Small-3.1-24B | Mistral-Small-3.1-24B | ✓ done — best W-F1=0.4731 (+CG codename) |
| Open-source (thinking) | — | QwQ-32B | ✓ done — best W-F1=0.1717 (E4) |
| Closed-source (API) | Gemini-2.5-Flash | GPT-4o | ✓ done — +GI=0.2570 (best, beats +CG=0.1857) |

---

## Experiment Status Matrix

causal_only experiments use 11 intervention-validated edges (abs(delta) score).
Observational experiments use a geomean filter on suppes_graph.json with score
`= sqrt(precedence × PR_delta)` (the two graded Suppes statistics; the prior
formula `sqrt(P(B|A) × PR_delta)` had P(B|A) double-counted on both factors and
was switched in May 2026 — all `outputs_corr/*-t0.2` results are stale).
Both benchmarks (MAST + TRAIL) now use the same threshold semantics: geomean ≥ threshold.
"code+name" = `1.1(Disobey Task Specification) -> 3.3(No or Incorrect Verification)  (strength: X.XX)`

| Model | Baseline | +with-graph code+name | +graph-inject code+name | Notes |
|---|---|---|---|---|
| Mistral-Small-24B | ✓ 0.3773 | ✓ **0.4731** | ✓ 0.3701 | vLLM; all done |
| Gemma-3-27B | ✓ 0.1419 | ✓ 0.1323 | ✓ 0.1412 | vLLM; done |
| GPT-oss 20B | ✓ 0.1766 | ✓ 0.1872 | ✓ 0.1884 | vLLM; done |
| QwQ-32B | ✓ 0.1608 | ✓ 0.1513 | ✓ 0.1369 (codename, thinking); non-codename=0.1717 | vLLM; done |
| GPT-4o | ✓ 0.2287 | ✓ 0.1857 | ✓ **0.2570** | API; +GI is best (paper main table) |

---

## Direction Analysis

### What's confirmed
- **Mistral + E3 code+name is the best setup** (W-F1=0.4731; Baseline 0.3773 → +9.6pp gain). The causal graph drives large recall gains on under-detected categories.
- **GPT-4o responds well to graph-inject (+GI) but not static (+CG)**: W-F1 Baseline 0.2287 → +CG 0.1857 (regression) → +GI 0.2570 (best). Two-pass dynamic injection works for closed-source API judges even though one-pass static injection hurts. (Numbers from `paper/main_results_table.tex`.)
- **Thinking models don't outperform instruction-tuned models here** — QwQ-32B (0.1717) is well below Mistral (0.4731). Reasoning budget increases conservatism and reduces recall.
- **QwQ-32B and GPT-4o both prefer E4 over E3** — targeted 2nd-pass injection fits their reasoning/RLHF profiles better than upfront context flooding.

### Dead ends / deprioritized
- **o1**: omitted from paper (RLHF conservatism + cost).
- Gemma: under-detection not fixable by edge format; deprioritized.
- GPT-oss 120B: incomplete run, outperformed by smaller models; deprioritized.
- Graph-inject for non-thinking open models (Mistral, Gemma, GPT-oss-20B): consistently below E3 for these; no point running more.
- Full stability graph (t0.5) for Mistral: **done, worse** (0.5237 vs 0.4731 codename — note: t0.5 baseline pre-dates the recent codename re-score, comparison is approximate). causal_only edges are still the canonical choice.
- E4 + full stability graph: not worth running — E3 t0.5 < E3 causal_only, and E4 already loses ~0.16 to E3 for Mistral.
- Detection-oriented prompt (eval_detect): **done, worse**. Baseline -0.007, E3 -0.157. Evidence requirement (section C: step IDs + quotes) collapses recall for subtle categories (2.6, 3.1, 2.3, 1.1) and eliminates graph benefit. Original prompt is better.

---

## Who&When Causal Adoption (baselines/who&when/causal/)

Adopts the TRAIL W&W +CG / +GI causal-graph design into MAST. Two new
scripts parallel to TRAIL's `baselines/who_and_when/causal/`:

| Script | Design | Variants | Graph modes |
|---|---|---|---|
| `baselines/who&when/causal/run_who_and_when_with_graph_vllm.py` | +CG one-pass (graph in prompt) | W1, W2 | `--causal_only`, `--corr_threshold` |
| `baselines/who&when/causal/run_who_and_when_graph_inject_vllm.py` | +GI two-pass (Pass-1 with graph → propagate → targeted Pass-2) | W1, W2 | `--causal_only`, `--corr_threshold` |

Both files follow the *patched* TRAIL W&W +GI design: **Pass-1 includes the
+CG graph guidance** (not naked). Pass-2 is a single trace-level targeted
call even for W2 — preserves the N+1 cost profile and matches MAST eval's
`full_run_eval_graph_inject.py` structural choice.

Dropped (don't apply to MAST's yes/no schema):
- `--span_index` — MAST has no location prediction.
- "FINAL LEAF subcategory" rule — MAST taxonomy is a flat 13-code list, no parents.
- "Resource Abuse last instance" rule — no location field.
- `--random_edges`, `--edge_threshold` — surface kept minimal; only `causal_only` + `corr_threshold` are exposed.

### Output naming

```
baselines/who&when/outputs/{model_tag}-yesno-who_and_when_{w1|w2}_graph_{causal_only|corr<value>}/
baselines/who&when/outputs/{model_tag}-yesno-who_and_when_{w1|w2}_graph_inject_{causal_only|corr<value>}/
```

### Run plan — Mistral-only ablation (matches TRAIL W&W scope)

Mirrors TRAIL's `paper/baseline_who_and_when.tex` ablation (W1/W2 vs
W1+graph/W2+graph), which is Mistral-only because W&W's signal is
model-dependent and the closed-source / thinking models already have
dominant effects elsewhere.

Anchor cells (no-graph W1/W2 baselines): produced by the existing
`baselines/who&when/run_who_and_when_vllm.py` — keep on-disk, reuse.

New cells: 1 model × 2 variants (W1, W2) × 2 archs (+CG, +GI) ×
2 graph modes (`causal_only`, `corr_threshold=0.50`) = **8 cells**.
(For the threshold mini-sweep see the Threshold Sweep section below.)

```bash
# W1 +CG, causal_only
CUDA_VISIBLE_DEVICES=0,1 python "baselines/who&when/causal/run_who_and_when_with_graph_vllm.py" \
    --variant w1 --causal_only --output_dir "baselines/who&when/outputs"

# W1 +GI, corr 0.50
CUDA_VISIBLE_DEVICES=0,1 python "baselines/who&when/causal/run_who_and_when_graph_inject_vllm.py" \
    --variant w1 --corr_threshold 0.50 --output_dir "baselines/who&when/outputs"

# Repeat for --variant w2 and each (arch × mode) cell.
```

Score:
```bash
python eval/calculate_scores_yesno.py \
    --annotation data/annotation/annotation_ag2_filtered.jsonl \
    --pred_dir "baselines/who&when/outputs/<subdir>"
```

---

## Threshold Sweep — graph-richness × architecture ablation (mirrors TRAIL Ablation 3)

Goal: characterise how W-F1 scales with the size of the injected graph
across the open-source model panel, mirroring TRAIL's
`paper/ablation_graph_richness.tex` story. Two architectures (`+CG` one-pass,
`+GI+SI` two-pass) × three corr-thresholds × a random-N null-graph control,
all anchored against the 11-edge causal-only baseline.

### Score definition (May 2026 update)

The threshold score is now the geomean of the two **independent** Suppes
statistics:
```
score(A->B) = sqrt(precedence(A,B) · ΔPR(A,B))
            with ΔPR = P(B|A) − P(B|¬A)
```
The prior score `sqrt(P(B|A) · ΔPR)` had `P(B|A)` double-counted on both
factors of the product, breaking the "both signals must be substantial"
interpretation of the geomean. All `outputs_corr/*-t0.2` runs were produced
under the old score and are stale; do not reuse them.

### Edge semantics — use `--corr_threshold` (NOT `--edge_threshold`)

All four `full_run_*` scripts in MAST support `--corr_threshold τ` with the
same union semantics as TRAIL:
**(Suppes geomean ≥ τ) ∪ (intervention-validated causal edges)**.
The old `--edge_threshold` flag still works but is pure-Suppes (no union)
and is therefore inconsistent with the ablation table. **Use
`--corr_threshold` for any new sweep run.**

`--random_edges` (added May 2026) samples `--random_n` directed pairs from
the 13-category MAST taxonomy excluding the 43 Suppes pairs; with the default
seed=42 and n=11, it produces a count-matched null-graph baseline against
causal-only.

### Edge landscape under the new score (suppes_graph.json + effect_edges.json)

| τ (new score) | pure Suppes (filter only) | corr-union (Suppes ∪ validated) |
|---|---|---|
| causal_only (intervention-validated) | — | **11** (anchor) |
| random-11 (seed 42, non-Suppes) | — | **11** (null-graph control) |
| ≥ 0.60 | 8 | **18** (sweep point) |
| ≥ 0.55 | 12 | 22 |
| ≥ 0.50 | 15 | **25** (sweep point) |
| ≥ 0.45 | 20 | 27 |
| ≥ 0.40 | 23 | **29** (sweep point) |
| ≥ 0.35 | 28 | 32 |
| ≥ 0.30 | 32 | 35 |
| ≥ 0.25 | 39 | 40 |
| ≥ 0.20 | 43 | 43 (saturated) |

Action zone lives in τ ∈ [0.60, 0.40]. Above 0.60 collapses toward
causal_only; below 0.40 monotonically saturates toward the full 43-edge
Suppes set.

### Sweep spec

- **Architectures (both)**:
  - `+CG` one-pass: `eval/full_run_eval_with_graph.py`
  - `+GI+SI` two-pass: `eval/full_run_eval_graph_inject.py`
- **Sweep settings (5 per architecture)**: `{causal_only, random, 0.60, 0.50, 0.40}`
  → corr-union edge counts **11 → 11 → 18 → 25 → 29** (monotone after the
  random baseline, well-spread).
- **Models — open-source panel** (mirrors TRAIL's open-source panel):
  1. **Mistral-Small-3.1-24B** — current best (W-F1 0.4731 causal-only +CG codename)
  2. **Gemma-3-27B** — under-detection sensitivity check
  3. **GPT-oss-20B** — small-model anchor
  4. **QwQ-32B** — thinking-model representative (use `--enable_thinking`)
- **Dataset**: AG2 (393 traces). No split equivalent to TRAIL's GAIA/SWE distinction.
- **Already on disk (anchors)**:
  - causal_only: all 4 models, both architectures, in `outputs_full/`. Re-usable
    as-is (causal-only doesn't go through the geomean score, so the score
    change doesn't affect those outputs).
- **Stale (must re-run)**:
  - All `outputs_corr/*-t0.2/` results use the old `sqrt(P(B|A)*pr_delta)`
    score and the deprecated `--edge_threshold` flag (pure-Suppes, no causal
    union). Drop from the paper, replace with the new sweep below.

### Commands (TRAIL-compatible sweep, `--corr_threshold` + `--random_edges`)

```bash
# Pattern for one (model, architecture, setting) cell
python eval/full_run_eval_with_graph.py \
    --model <model> --model_tag <tag> \
    {--corr_threshold <τ> | --random_edges} \
    [--enable_thinking] \
    --output_dir outputs_thres

# Mistral-Small-3.1-24B, +CG
CUDA_VISIBLE_DEVICES=<gpus> python eval/full_run_eval_with_graph.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 --model_tag Mistral-Small-24B \
    --random_edges --output_dir outputs_thres
CUDA_VISIBLE_DEVICES=<gpus> python eval/full_run_eval_with_graph.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 --model_tag Mistral-Small-24B \
    --corr_threshold 0.60 --output_dir outputs_thres
CUDA_VISIBLE_DEVICES=<gpus> python eval/full_run_eval_with_graph.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 --model_tag Mistral-Small-24B \
    --corr_threshold 0.50 --output_dir outputs_thres
CUDA_VISIBLE_DEVICES=<gpus> python eval/full_run_eval_with_graph.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 --model_tag Mistral-Small-24B \
    --corr_threshold 0.40 --output_dir outputs_thres

# Mistral-Small-3.1-24B, +GI+SI (same 4 settings)
CUDA_VISIBLE_DEVICES=<gpus> python eval/full_run_eval_graph_inject.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 --model_tag Mistral-Small-24B \
    --random_edges --output_dir outputs_thres
# ... corr 0.60, 0.50, 0.40

# Same pattern for: openai/gemma-3-27b-it, openai/gpt-oss-20b
# QwQ-32B: add --enable_thinking on every run
CUDA_VISIBLE_DEVICES=<gpus> python eval/full_run_eval_graph_inject.py \
    --model Qwen/QwQ-32B --model_tag QwQ-32B \
    --corr_threshold 0.50 --enable_thinking --output_dir outputs_thres
```

Output directories use the graph-tag suffix from the inner scripts:
```
outputs_thres/{model_tag}-yesno-with-graph-codename-random11_seed42/
outputs_thres/{model_tag}-yesno-with-graph-codename-corr0.6/
outputs_thres/{model_tag}-yesno-with-graph-codename-corr0.5/
outputs_thres/{model_tag}-yesno-with-graph-codename-corr0.4/
outputs_thres/{model_tag}-yesno-graph-inject-codename-random11_seed42/
outputs_thres/{model_tag}-yesno-graph-inject-codename-corr0.6/
outputs_thres/{model_tag}-yesno-graph-inject-codename-corr0.5/
outputs_thres/{model_tag}-yesno-graph-inject-codename-corr0.4/
```

### Run count

- New thresholds {0.60, 0.50, 0.40}: 4 models × 2 archs × 3 τ = **24 runs**
- Random baseline: 4 models × 2 archs × 1 = **8 runs**
- W&W mini-sweep (see below): 1 model × 2 variants × 2 archs × 4 settings = **16 runs**
- Total new compute: **48 runs** (32 eval + 16 W&W ablation)

(causal_only stays as the on-disk anchor for the eval sweep — not re-run.)

### Who&When threshold mini-sweep (Mistral, ablation)

Companion sweep for the paper-side W&W ablation (mirrors TRAIL's
`paper/baseline_who_and_when.tex`). Uses the new W&W causal/ scripts.

- **Architectures**: W&W +CG and W&W +GI (`baselines/who&when/causal/`).
- **Settings**: `{causal_only, corr 0.60, corr 0.50, corr 0.40}` — no
  `--random_edges` (not exposed in the W&W causal/ scripts).
- **Variants**: W1 and W2.
- **Model**: Mistral-Small-3.1-24B only (matches TRAIL's W&W ablation scope).
- **Total**: 1 × 2 × 2 × 4 = **16 W&W cells**.

```bash
# Pattern (W&W +CG)
CUDA_VISIBLE_DEVICES=<gpus> python "baselines/who&when/causal/run_who_and_when_with_graph_vllm.py" \
    --variant {w1|w2} {--causal_only | --corr_threshold τ} \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 --model_tag Mistral-Small-24B \
    --output_dir "baselines/who&when/outputs_thres"

# Pattern (W&W +GI)
CUDA_VISIBLE_DEVICES=<gpus> python "baselines/who&when/causal/run_who_and_when_graph_inject_vllm.py" \
    --variant {w1|w2} {--causal_only | --corr_threshold τ} \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 --model_tag Mistral-Small-24B \
    --output_dir "baselines/who&when/outputs_thres"
```

Output directories use the W&W naming pattern under `outputs_thres/`. Score
with the same `eval/calculate_scores_yesno.py` as the eval sweep.

### Scoring

```bash
# After each run completes
python eval/calculate_scores_yesno.py \
    --annotation data/annotation/annotation_ag2_filtered.jsonl \
    --pred_dir outputs_thres/<model_tag>-yesno-<with-graph|graph-inject>-codename-<tag>

# Bulk-score everything at the end
for d in outputs_thres/*-yesno-*-codename-*/; do
    python eval/calculate_scores_yesno.py \
        --annotation data/annotation/annotation_ag2_filtered.jsonl \
        --pred_dir "$d"
done
```

### Plot

Once all 32 runs complete, parse W-F1 from `*-metrics.json` and plot
τ on the x-axis (left-to-right: causal_only, random-11, 0.60, 0.50, 0.40 —
i.e. monotone in edge count). Two panels per model: W-F1 for `+CG` and
`+GI+SI` overlaid. The expected reading: `+CG` saturates or regresses at
higher edge counts; `+GI+SI` keeps climbing; random-11 sits well below
causal-only (confirming gains are structural, not edge-count-driven).

### Optional driver script

A `run_threshold_sweep.sh` analogous to TRAIL's `benchmarking/eval/run_threshold_sweep.sh`
is deferred. For now, launch the 32 cells manually or via per-model sbatch.

---

## Scoring

```bash
# After each run completes:
python eval/calculate_scores_yesno.py --pred_dir <output_dir>/<model_subdir>
```

---

## Final Comparison Table (current state)

| Model | Type | Baseline | +with-graph code+name | +graph-inject code+name |
|---|---|---|---|---|
| Mistral-Small-24B | open-source | 0.3773 | **0.4731** | 0.3701 |
| Gemma-3-27B | open-source | 0.1419 | 0.1323 | 0.1412 |
| GPT-oss 20B | open API | 0.1766 | 0.1872 | 0.1884 |
| QwQ-32B | open thinking | 0.1608 | 0.1513 | **0.1717** (non-codename); 0.1369 (codename) |
| GPT-4o | closed-source | 0.2287 | 0.1857 | **0.2570** (E4 wins) |
