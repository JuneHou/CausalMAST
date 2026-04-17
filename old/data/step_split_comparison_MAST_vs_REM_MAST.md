# Step/Turn Split Strategy: MAST vs REM_MAST

Comparison of how **MAST** (this repo, `step_labeling/`) and **REM_MAST** (`/data/wang/junh/githubs/REM_MAST`) split traces into segments/steps for labeling.

---

## Summary

| Aspect | **MAST (current)** | **REM_MAST** |
|--------|--------------------|--------------|
| **Goal** | Turn-level steps for failure-type onset labeling (which step first shows failure) | Chunks for segment-level annotation / causal pipelines |
| **Task awareness** | **Per-MAS rules** (AG2, AppWorld, ChatDev, HyperAgent, MetaGPT, OpenManus; Magentic skipped) | **Task-agnostic** (same strategy for all MAS) or char chunking only |
| **Output unit** | **Turn**: `speaker`, `content`, `meta` → steps as content (optional `[speaker]` prefix) | **Segment**: `text`, `role` or `chunk_*`, `method` |
| **Trajectory type** | String **or** list (AG2/HyperAgent lists supported) | String only (list trajectories skipped in REM_MAST scripts) |
| **Fallback** | None per task; some tasks have internal fallback (e.g. ChatDev `**X**` fallback) | Character-based chunking with overlap (e.g. 3000 chars, 300 overlap) |
| **Noise / filtering** | `drop_noise_blocks(min_chars=20)`, `skip_if_not_conversation(turns)` | No filtering by length or “conversation-ness” |

---

## REM_MAST Split Strategy

### 1. `scripts/2_segment_traces.py` (hierarchical)

Three strategies in order:

1. **JSON parsing**  
   - If `trajectory` parses as a **JSON list** of dicts with `"content"` → one segment per message.  
   - `role` from `msg.get("role", f"message_{i}")`.  
   - **Caveat:** Trajectory is taken as a **string**; if MAD stores AG2/HyperAgent as a list, REM_MAST scripts first need it as a string (e.g. `json.dumps(trajectory)`), else they skip the record (`if not traj or not isinstance(traj, str)`).

2. **Role/turn markers** (regex, task-agnostic)  
   - Split on common markers, e.g.  
     - `[\w+Agent]`  
     - `(\w+\s+Agent):`  
     - `(User|Assistant|System):`  
     - `(Action|Observation|Thought):`  
     - `(Agent\s+\d+):`  
     - `>>>\s*(\w+)`  
   - Splits are **not** inside code fences (```).  
   - Produces `(role, content)` tuples; if only one segment, strategy is not used.

3. **Character chunking (fallback)**  
   - `chunk_text(text, chunk_chars=3000, overlap=300)`.  
   - Segments labeled e.g. `role="chunk_0"`, `method="char_chunking"`.

**Output:** List of segment dicts with `text`, `role`, `method` (`"json"` | `"role_markers"` | `"char_chunking"`).

### 2. `scripts_REM/2_segment_traces.py` (chunking only)

- **No** JSON or role-marker logic.  
- Trajectory must be a string.  
- Segments = `chunk_text(traj, chunk_chars, overlap)` only.  
- Output: `segments: [{"t": i, "text": chunk}, ...]` (no `role` or `method`).

### 3. `src/segmenting.py`

- `chunk_text(text, chunk_chars=3000, overlap=300)` and `create_segments()` (same, with index `t`).  
- No JSON, no role-based splitting.

---

## MAST (current) Split Strategy

- **Task-specific** rules in `step_labeling/step_extraction.py` and documented in `data/step_split_analysis.md`.

- **AG2:** List of `{content, role, name}` → one turn per element; string → triplets `content:` / `role:` / `name:`.
- **AppWorld:** Turn = `Response from <X> Agent` (or `Reply from`) + following block; optional “Message to &lt;X&gt; Agent” prepended.
- **ChatDev:** Turn = `[timestamp] Speaker: **...<->... turn k**`; fallback: split on `**X**` lines.
- **HyperAgent:** List → one turn per item; string → split on `- INFO - <label>:` and derive speaker from label.
- **MetaGPT:** Timestamp blocks; within block, `FROM: X TO: Y` or `NEW MESSAGES` + speaker lines.
- **OpenManus:** Split on `app.agent.toolcall:think:81 - ✨ Manus's thoughts:`.
- **Magentic:** Always skip (no turn structure).

- **Shared helpers:** `clean()`, `drop_noise_blocks(min_chars=20)`, `skip_if_not_conversation(turns)`, `split_by_headers()`.
- **Output:** `extract_turns(task, trajectory)` → `List[Turn]`; `get_steps_from_trace()` maps to step strings (content, optional `[speaker]\n` prefix).

---

## When to use which

- **MAST (task-specific):**  
  - When you need **turn-level** steps for “first step where failure X appears” and want boundaries that match each MAS (e.g. ChatDev phases, AppWorld agent responses, AG2 role turns).  
  - When you have **mixed trajectory types** (string and list) and want to support all MAD task types.

- **REM_MAST (task-agnostic / chunking):**  
  - When you want **uniform segment length** (e.g. for causal/REM pipelines) or a single strategy across MAS.  
  - When you only have **string** trajectories and are fine with role-marker heuristics or pure character chunking as fallback.

---

## Differences at a glance

| Feature | MAST | REM_MAST (scripts/2) | REM_MAST (scripts_REM/2) |
|--------|------|------------------------|----------------------------|
| MAS-specific rules | Yes (6 MAS) | No | No |
| JSON list trajectory | Yes (AG2, HyperAgent) | No (string only) | No (string only) |
| Role/turn markers | Per-MAS patterns | Generic regex set | No |
| Char chunking fallback | No | Yes (if role split gives ≤1 segment) | Yes (only method) |
| Min turn length / conversation check | Yes | No | No |
| Speaker/role in output | Yes (Turn.speaker) | Yes (segment["role"]) | No |
| Magentic | Explicitly skipped | Not special-cased | Not special-cased |

---

## Statistical comparison (ChatDev)

Both methods were run on the **same raw ChatDev data**. Failure-onset labels come from: **MAST** `step_labeling_output.jsonl` (first step where each failure type appears, turn-level) and **REM_MAST** `onsets_chatdev.jsonl` (first segment index per failure type). Alignment is by `trace_id`; only traces present in both MAST steps_dataset and REM_MAST segments/onsets are used (**61 common traces**).

### 1. Unit counts (steps vs segments per trace)

| Metric | MAST (steps) | REM_MAST (segments) |
|--------|----------------|----------------------|
| Min per trace | 13 | 25 |
| Max per trace | 17 | 31 |
| Mean | 13.87 | 25.69 |
| Median | 14 | 25 |

MAST yields fewer, larger units per trace (turn-level); REM_MAST yields more, smaller segments (e.g. role-marker or chunk-based).

### 2. Presence (trace × failure_type)

The total number of (trace, failure_type) pairs **to be labeled** is the cumulative count over traces: for each trace, only the failure types that are in scope for that trace (from the labeling setup, e.g. per-trace failure-type set). That gives **226** pairs (not 61 × 14), since each trace has a variable number of failure types to label (2–11 per trace).

- **Total (trace, failure_type) pairs to label:** 226 (cumulative over 61 traces).
- **MAST has a first-step label:** 123 (54.4%).
- **REM_MAST has an onset:** 70 (31.0%).
- **Both have an onset:** 50.
- **MAST only:** 73 | **REM only:** 20.
- **Agreement** (both present or both absent): **133 / 226 (58.8%)**.

MAST labels more failure onsets; agreement on presence is moderate when restricted to in-scope pairs.

### 3. Relative onset position (where both have onset)

For the **50 (trace, failure_type)** pairs where both methods have an onset, relative position is defined as (first_step − 1) / n_steps (MAST) and onset_segment_index / n_segments (REM_MAST), both in [0, 1].

| Metric | MAST | REM_MAST |
|--------|------|----------|
| Mean relative position | 0.287 | 0.327 |
| **Pearson r** (across 50 pairs) | — | **0.027** |
| Mean absolute difference | — | 0.171 |

Correlation of relative position is very low: the two methods often disagree on *where* in the trace the failure first appears, even when they agree that it appears (different granularity and boundaries).

### 4. Traces with ≥1 failure

| Method | Traces with ≥1 failure |
|--------|------------------------|
| MAST | 57 / 61 (93.4%) |
| REM_MAST | 47 / 61 (77.0%) |

**Summary:** MAST (task-specific turn splits) produces fewer, semantically coherent steps and labels more failure onsets with higher trace-level coverage. REM_MAST (task-agnostic segments) produces more segments and fewer onset labels, with weak correlation in relative position when both label the same (trace, failure_type). Use MAST for turn-level “first step of failure” analysis; use REM_MAST for segment-level or causal pipelines that need fixed granularity.
