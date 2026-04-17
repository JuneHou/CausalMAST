# MAST Step-Level Annotation Pipeline: Workflow

## Goal

Produce TRAIL-like step-level annotations for MAST traces. Given a trace already labeled at the
trace level (which error types are present), ask an LLM to locate the **earliest step** where each
error first becomes observable, with evidence and description grounded in the trace.

This converts full error detection into **constrained onset localization**: the model is told which
errors exist and only needs to find where each one starts.

---

## Important Caveat: Steps Are Not Native to MAST

The original MAST benchmark has no step-level structure. `mast_annotation` is a flat trace-level
dict `{"1.1": 0, "1.3": 1, ...}` with no location information.

Steps are produced by our own splitting logic in `openmanus.py` (and future per-task modules),
which segments the raw trajectory using per-MAS-type heuristics. Step boundaries are not ground
truth — always validate with `inspect_steps.py` before running annotation at scale.

---

## Data Source

**MAD dataset** (`old/data/raw/MAD_full_dataset.json`, HuggingFace `mcemri/MAD`).

Each MAD record:
- `trace_id`, `mas_name`, `llm_name`, `benchmark_name`
- `trace.trajectory` — raw agent trace (string)
- `mast_annotation` — `{error_id: 0|1}` for all 14 MAST error types

**OpenManus in MAD:** 30 total traces, 16 error-free, **14 with at least one error**.
`run_annotation.py` only processes traces with ≥ 1 error (`min_errors=1`).

---

## Output Format

One JSONL file, one JSON object per trace. Steps are included so every `location` reference is
self-contained without re-running extraction.

```json
{
  "trace_id": "5",
  "mas_name": "OpenManus",
  "mast_annotation": {"1.1": 1, "1.2": 0, "2.6": 1, "3.3": 1, ...},
  "steps": [
    {"id": "step_00", "content": "Executing step 1/20\n✨ Manus's thoughts: ..."},
    {"id": "step_01", "content": "..."}
  ],
  "errors": [
    {
      "category": "1.1 Disobey Task Specification",
      "location": "step_02",
      "evidence": "...",
      "description": "...",
      "impact": "LOW | MEDIUM | HIGH"
    }
  ],
  "prompt_variant": "few_shot",
  "model": "gpt-4o"
}
```

Differences from TRAIL:
- `location` is a sequential step ID (`step_00`, `step_01`, ...) not a content hash
- `steps` field included for self-contained lookup
- `category` uses MAST taxonomy IDs + names (e.g. `"1.3 Step Repetition"`)
- No `scores` block

---

## Phase 0: Validate Step Extraction

Before running annotation, inspect extracted steps on a sample:

```bash
python -m annotation.inspect_steps --task openmanus --n 3 --output inspect_openmanus.txt
```

Output shows — for each sampled trace — the known error labels, step count, and up to 600 chars of
each step. Confirms step boundaries are sensible and content is substantive before committing to a
full LLM run.

`inspect_steps.py` loads all 30 traces (including error-free ones) so you can check extraction
quality across the full range. `run_annotation.py` only processes the 14 with errors.

---

## OpenManus Step Extraction (`openmanus.py`)

### Split boundary

The OpenManus log contains `Executing step N/M` markers from the agent runtime:

```
2025-04-01 01:27:47.939 | INFO | app.agent.base:run:140 - Executing step 1/20
```

Each `Executing step N/M` line starts one agent action cycle. We split on these boundaries —
everything from one marker to the next (exclusive) is one step.

### Log cleaning (`_clean_step_content`)

Raw log blocks contain verbose infrastructure that wastes tokens and dilutes the signal:
- Timestamps: `2025-04-01 02:17:03.592`
- Module paths: `app.agent.toolcall:think:81`
- Token accounting: `Token usage: Input=14704, Completion=20, ...`
- Redundant activation lines: `🔧 Activating tool: 'browser_use'...`

The cleaner strips the full log prefix (`TIMESTAMP | LEVEL | module:fn:line - `) from every log
line, drops token usage and tool activation lines entirely, and collapses blank lines. Non-log
lines (thought continuation text, tool results) are kept as-is.

**Result per step after cleaning:**

```
Executing step 13/20
✨ Manus's thoughts: To continue, you need to identify the word deleted in the last amendment...

### Next Steps:
1. Check the Last Amendment: ...
2. Extract Amendment Details: ...

🛠️ Manus selected 1 tools to use
🧰 Tools being prepared: ['browser_use']
🔧 Tool arguments: {"action":"extract_content","goal":"Identify deleted word..."}
🎯 Tool 'browser_use' completed its mission! Result: Observed output of cmd `browser_use` executed:
Extracted from page:
{'text': 'The last amendment to Rule 612 involved the deletion of the word "adverse"...'}
```

### Token savings from cleaning

Removing log noise cuts 12K–41K chars per trace:

| trace | steps | before cleaning | after cleaning |
|-------|-------|-----------------|----------------|
| 3     | 20    | 35,282          | 21,614         |
| 5     | 23    | 39,825          | 22,288         |
| 8     | 66    | 138,333         | 92,265         |
| 10    | 50    | 85,244          | 57,377         |
| 19    | 38    | 104,266         | 84,140         |

Trace 8 (66 steps, 9 error types) dropped from 138K to 92K chars, bringing it within GPT-4o's
128K context limit.

---

## Annotation Pipeline (`run_annotation.py`)

```
MAD dataset (old/data/raw/MAD_full_dataset.json)
    │
    ▼
1. Load & filter
   - Task: OpenManus only (--task openmanus)
   - Keep traces with >= 1 error label (min_errors=1)
   - Result: 14/30 OpenManus traces
    │
    ▼
2. Step extraction  [openmanus.py: extract_steps()]
   - Split on "Executing step N/M" boundaries
   - Clean each block: strip log prefixes, drop token/activation lines
   - Assign IDs: step_00, step_01, ... (zero-padded)
   - Skip traces with 0 steps extracted
    │
    ▼
3. Prompt construction  [prompt.py: build_prompt()]
   - Resolve error IDs to full labels: "1.3 Step Repetition"
   - Fetch definitions for present errors only (not all 14)
   - Truncate each definition to its first paragraph (strip long examples)
   - Format steps as [step_NN]\n<content>
   - Fill zero_shot or few_shot template (--prompt_variant)
    │
    ▼
4. LLM call  [llm_client.py: LLMClient.generate()]
   - One call per trace (all error types in a single prompt)
   - OpenAI API (default) or vLLM offline (--offline)
    │
    ▼
5. Parse & validate  [llm_client.py: extract_json(), validate_response()]
   - Extract JSON from response (handles markdown fences, embedded JSON)
   - Reject entries whose category is not in the given error list
   - Reject entries whose location is not a valid step ID for this trace
    │
    ▼
6. Write JSONL
   - One line: {trace_id, mas_name, mast_annotation, steps, errors, prompt_variant, model}
   - Traces with parse failures: errors=[], parse_error="<snippet>"
```

---

## Prompt Design (`prompt.py`)

### Key instructions

The prompt tells the model:
- The error labels are **already known** — do not detect, only locate
- Work through error categories **one at a time**
- For each category, find the **single earliest** step, write one entry, then move on
- **Output exactly one entry per category** — never two entries for the same category
- Evidence must be a direct quote or close paraphrase from the trace
- Description must explain both (a) why the evidence matches the definition and (b) why this is the earliest valid location

### Prompt variants

**`zero_shot`** — Instructions only, no examples. Fewer tokens, suitable for strong models.

**`few_shot`** (default) — Same instructions + one in-context example using `1.3 Step Repetition`.
The example demonstrates the expected output format and the "earliest step" reasoning.

### Definitions

Only definitions for error types **present in this trace** are included (never all 14).
Each definition is truncated to its first paragraph — long illustrative examples are stripped.
This saves ~1K–2K chars per included definition.

### Output format enforced by prompt

```json
{
  "errors": [
    {
      "category": "1.3 Step Repetition",
      "location": "step_02",
      "evidence": "...",
      "description": "...",
      "impact": "LOW | MEDIUM | HIGH"
    }
  ]
}
```

---

## Results: First Full Run (gpt-4o, few_shot)

14 traces annotated, 0 skipped, 0 parse errors.

| trace | steps | known errors | annotated | note |
|-------|-------|-------------|-----------|------|
| 3     | 20    | 2           | 2         | ✓    |
| 5     | 23    | 3           | 3         | ✓    |
| 6     | 18    | 1           | 1         | ✓    |
| 7     | 26    | 2           | 2         | ✓    |
| 8     | 66    | 9           | 3         | model skipped 6 (context pressure, 9 errors) |
| 10    | 50    | 5           | 4         | model skipped 1 |
| 11    | 23    | 1           | 1         | ✓    |
| 12    | 15    | 1           | 1         | ✓    |
| 15    | 24    | 2           | 2         | ✓    |
| 17    | 17    | 2           | 2         | ✓    |
| 18    | 19    | 3           | 1         | model skipped 2 |
| 19    | 38    | 2           | 2         | ✓    |
| 21    | 30    | 1           | 2         | duplicate category (fixed in prompt) |
| 27    | 52    | 3           | 3         | ✓    |

**Fixes applied after first run:**
1. Prompt: added "one entry per category, then move on" rule → fixes duplicate category (trace 21)
2. Log cleaning: strip timestamps/token lines from step content → 12K–41K chars saved per trace
3. Definitions: truncate to first paragraph only → additional ~1K–2K chars saved per error type

---

## File Structure

```
annotation/
  config.py          — paths, task types, failure keys, model/prompt defaults
  definitions.py     — parse definitions.txt into {id: {name, text}}; truncate to core paragraph
  prompt.py          — zero_shot and few_shot prompt builders
  llm_client.py      — vLLM offline + OpenAI-compatible API client; JSON extraction + validation
  openmanus.py       — OpenManus record loader + step extractor + log cleaner
  inspect_steps.py   — Phase 0: dump sampled steps to text for manual review
  run_annotation.py  — CLI: load → filter → extract → prompt → LLM → JSONL
  workflow.md        — this file
```

---

## CLI Reference

```bash
# Phase 0: inspect step extraction before annotating
python -m annotation.inspect_steps --task openmanus --n 3 --output inspect_openmanus.txt

# Debug run: 3 traces, prints full prompt + raw response to stderr
python -m annotation.run_annotation \
    --task openmanus \
    --model gpt-4o --prompt_variant few_shot \
    --debug --output debug_openmanus.jsonl

# Full run with GPT-4o
python -m annotation.run_annotation \
    --task openmanus \
    --model gpt-4o --prompt_variant few_shot \
    --output annotation_openmanus.jsonl

# Full run with vLLM offline
python -m annotation.run_annotation \
    --task openmanus --offline \
    --model Qwen/Qwen3-32B --prompt_variant few_shot \
    --output annotation_openmanus.jsonl
```

---

## Known Limitations

1. **Traces where model skips errors:** For traces with many errors (e.g. trace 8 with 9 types)
   or ambiguous evidence, the model may choose not to annotate some categories. This is partially
   intentional (the model should only annotate what it can support), but warrants manual review.

2. **Context pressure on very long traces:** Trace 8 (66 steps) is at 92K chars after cleaning,
   close to GPT-4o's limit. A future improvement: chunk long traces or run one call per error type.

3. **OpenManus only:** Other MAS types (AppWorld, ChatDev, AG2, MetaGPT, HyperAgent) are not yet
   implemented. Each needs its own extractor module.

  ┌───────────┬──────────────────┬──────────────┬──────────────┬───────────────────────────────┐
  │    MAS    │ Traces w/ errors │ Median steps │ Median chars │       Format difficulty       │
  ├───────────┼──────────────────┼──────────────┼──────────────┼───────────────────────────────┤
  │ AG2       │ 497              │ ~6           │ ~5K          │ Medium (YAML-like)            │
  ├───────────┼──────────────────┼──────────────┼──────────────┼───────────────────────────────┤
  │ MetaGPT   │ 172              │ ~6           │ ~7K          │ Easy (timestamped log)        │
  ├───────────┼──────────────────┼──────────────┼──────────────┼───────────────────────────────┤
  │ Magentic  │ 152              │ unknown      │ ~62K         │ Hard (Docker build noise)     │
  ├───────────┼──────────────────┼──────────────┼──────────────┼───────────────────────────────┤
  │ ChatDev   │ 93               │ unknown      │ ~200K        │ Hard (very long, multi-phase) │
  ├───────────┼──────────────────┼──────────────┼──────────────┼───────────────────────────────┤
  │ OpenManus │ 14               │ 25–66        │ ~22–92K      │ Already done                  │
  └───────────┴──────────────────┴──────────────┴──────────────┴───────────────────────────────┘