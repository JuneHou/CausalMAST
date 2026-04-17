# MAST Step-Level Failure Labeling

This folder implements **step-level** (turn-level) failure labeling for MAST traces following the [MAST step-level labeling plan](../../Agents_Failure_Attribution/docs/MAST_step_level_labeling_plan.md).

## Data: MAD (primary) or traces/ + annotations

- **MAD** is the MAST-derived dataset: one JSON file with trace content + `mast_annotation` per record (same as in `notebooks/failure_distribution_by_task.ipynb`). **No separate annotations path.** Download once into the MAST repo:

  ```bash
  # From MAST repo root
  python scripts/0_download_mad.py
  ```
  This writes `data/raw/MAD_full_dataset.json` (HuggingFace `mcemri/MAD`). Then run labeling with `--mad` or with no `--mad` (default path `data/raw/MAD_full_dataset.json` is used if present).

- **Before labeling:** Extract and save the step-split dataset, then **use it for labeling** so the same step boundaries you inspected are used:
  ```bash
  python -m step_labeling.extract_steps_dataset
  ```
  This writes `data/processed/steps_dataset.jsonl` (one JSON per trace: trace_id, mas_name, n_steps, steps, failure_types, mast_annotation) and `data/processed/steps_summary.json` (n_steps stats per mas_name). Inspect the summary to verify extraction (e.g. ChatDev should have ~14–18 turns, not 200+). **Recommended:** pass this file to `run_labeling` via `--steps_dataset` so labeling uses the exact extracted steps (no re-extraction from MAD).

- **Optional:** Use `traces/` plus a custom annotations file (`--traces_dir`, `--annotations`) instead of MAD.

## Preprocessing

1. **Splittable tasks only:** Magentic is **excluded**.
2. **Task types:** AG2, AppWorld, ChatDev, HyperAgent, MetaGPT, OpenManus.
3. **Filter:** Only traces with **≥ 2 failure types** are kept (configurable via `--min_failures`).

When using **annotations** (traces/ mode): JSONL or JSON array with `trace_id`/path and `mast_annotation` per record.

## Pipeline

- **Pass 1:** One LLM call per step. For each step, ask: which of the failure types in **F** (present in this trace) appear in this step? Require **verbatim evidence** from the step content. Context is a **local window** (step t or t−1..t).
- **Pass 2:** For each failure type in F, set the first step to the earliest step where Pass 1 flagged it with valid evidence. If multiple candidates, run one **adjudication** call.
- **Evidence rule:** Evidence must be an exact substring of the step content; otherwise the (step, type) pair is rejected.

## LLM: vLLM (in-process or server)

- **Default model:** `Qwen/Qwen3-32B`.
- **GPT-OSS option:** Use `--model openai-community/gpt2` (or another open-source model name).

**Recommended: run with `--offline`** so vLLM is launched at the beginning in the same process. The model is loaded once, and all queries are sent directly to vLLM—no separate host or URL to configure.

```bash
# 1. Download MAD (one-time)
python scripts/0_download_mad.py

# 2. Extract steps and save locally (inspect steps_summary.json before labeling)
python -m step_labeling.extract_steps_dataset

# 3. Label using the extracted steps dataset (recommended: same step boundaries as inspection)
python -m step_labeling.run_labeling --offline --min_failures 2 \
  --steps_dataset data/processed/steps_dataset.jsonl \
  --output step_labeling_output.jsonl

# Or use MAD directly (steps re-extracted on the fly)
python -m step_labeling.run_labeling --offline --min_failures 2 --output step_labeling_output.jsonl
# Or pass MAD path explicitly
python -m step_labeling.run_labeling --offline --mad data/raw/MAD_full_dataset.json --output step_labeling_output.jsonl

# gpt-oss / smaller model
python -m step_labeling.run_labeling --offline --model openai-community/gpt2 --output step_labeling_output.jsonl
```

### Multi-GPU behavior (offline)

- We assume you set `CUDA_VISIBLE_DEVICES` before running.
- By default, `--offline` sets **vLLM tensor parallel size** to **the number of visible GPUs**.
- You can override explicitly with `--tensor_parallel_size N` (and optionally `--pipeline_parallel_size M`).

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m step_labeling.run_labeling --offline --tensor_parallel_size 4 --annotations /path/to/annotations.jsonl --output step_labeling_output.jsonl
```

Optional **server mode** (separate vLLM server): omit `--offline` and start the server yourself, then pass `--base_url` if needed. Use this only if you prefer a long-lived server process.

## Usage

```bash
# From MAST repo root
cd /path/to/MAST

# MAD (recommended): download once, then run
python scripts/0_download_mad.py
python -m step_labeling.run_labeling --offline --min_failures 2 --output step_labeling_output.jsonl

# Traces/ + custom annotations (optional)
python -m step_labeling.run_labeling --offline \
  --traces_dir ./traces --annotations /path/to/annotations.jsonl \
  --min_failures 2 --manifest_out step_labeling/manifest.json
python -m step_labeling.run_labeling --offline \
  --traces_dir ./traces --annotations /path/to/annotations.jsonl \
  --output step_labeling_output.jsonl
```

## Output

JSONL: one JSON object per trace, e.g.:

```json
{
  "trace_id": "...",
  "mas_name": "ChatDev",
  "failure_type_to_first_step": {
    "1.2": { "first_step": 3, "evidence": "verbatim from step 3", "confidence": "high" },
    "2.1": { "first_step": 7, "evidence": "...", "confidence": "medium" }
  }
}
```

## Files

| File | Purpose |
|------|--------|
| `config.py` | Paths (traces/, definitions.txt), task types, vLLM default model and gpt-oss option |
| `step_extraction.py` | Step (turn) extraction per task type from official traces |
| `extract_steps_dataset.py` | Extract steps from MAD, save to data/processed/ (run before labeling) |
| `preprocessing.py` | Load MAD or discover traces (excl. Magentic), filter by annotations ≥2 |
| `load_definitions.py` | Load MAST taxonomy from definitions.txt |
| `prompts.py` | Pass 1 and Pass 2 (adjudication) prompt builders |
| `vllm_client.py` | In-process vLLM (--offline) or OpenAI-compatible server client; default Qwen3-32B, --model for gpt-oss |
| `pass1_pass2.py` | Pass 1 + Pass 2 + evidence validation + adjudication |
| `run_labeling.py` | CLI: preprocess → label → write JSONL |

## Requirements

- Python 3.8+
- **In-process (--offline):** `vllm` (loads model at start, no server).
- **Server mode:** `openai` and a running vLLM API server.

Install for in-process (recommended):

```bash
pip install vllm
# Then run with --offline (no separate host or URL)
```
