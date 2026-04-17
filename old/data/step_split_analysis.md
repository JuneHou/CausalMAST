# Step Split Strategy Analysis

Analysis of **turn-level** step extraction for each MAST task type in the MAD dataset. Used by `step_labeling/extract_steps_dataset.py` and `step_labeling/run_labeling.py`.

**Common output schema:** Each extracted unit is a **Turn** with `speaker`, `content`, and optional `meta` (timestamp, phase, header, etc.). Steps passed to the labeling prompt are derived from turns (content, optionally prefixed with `[speaker]`).

**Helpers:** `clean()`, `drop_noise_blocks(turns, min_chars=20)`, `skip_if_not_conversation(turns)` (skip trace if &lt;2 turns or &gt;80% very short), `split_by_headers(text, header_re, speaker_fn)`.

**Magentic:** Always skip (no turn structure).

---

## 1. AG2

**Rule:**
- If `trajectory` is **list[dict]** with keys `{content, role, name}`: **each element = one turn**, `speaker = name or role`.
- If `trajectory` is **MAD-style string**: parse repeated **triplets** `content:` … `role:` … `name:`; each triplet = one turn.

**Implementation:** `split_ag2(trajectory)`. List: one Turn per message, speaker from name/role. String: regex for `content:\s*(?P<content>...)\s*role:\s*(?P<role>...)\s*name:\s*(?P<name>...)` (until next `content:` or end). Then `drop_noise_blocks` and `skip_if_not_conversation`.

**Example (MAD string):**
```
content:
    Let's use Python to solve...
role: assistant
name: mathproxyagent

content:
    First, let's calculate...
role: user
name: assistant
```

---

## 2. AppWorld

**Rule:**
- Turn boundary = `Response from <X> Agent` OR `Reply from <X> Agent`.
- Turn content = response block + any immediately following `Code Execution Output` / error lines until next response/reply.
- Optional: attach nearest preceding `Message to <X> Agent` as input inside the same turn.

**Implementation:** `split_appworld(text)`. Find all `(Response from|Reply from) <speaker> Agent` headers; for each, take block until next such header. Optionally prepend the last `Message to <speaker> Agent` block before this position. `drop_noise_blocks`, `skip_if_not_conversation`.

**Example:**
```
Response from Supervisor Agent
    # First, I need to interact with the Spotify agent...
    send_message(app_name='spotify', ...)

Code Execution Output
    # CallExtension
    reply = self.send_message(...)
```

---

## 3. ChatDev

**Rule:**
- Turn boundary = lines like `[timestamp] Speaker: **Speaker<->Other on : Phase, turn k**`.
- Do **not** split on generic `**Timestamp**:` / config fields.

**Implementation:** `split_chatdev(text)`. Header regex: `^\[\d{4}-\d{2}-\d{2} .*?\]\s+(?P<speaker>[^:]+):\s+\*\*.*?<->.*?turn\s+\d+\*\*.*$`. Use `split_by_headers`; optionally strip leading timestamp header line from each turn content. If no matches (e.g. older trace format), fallback: split on every `**X**` line and use speaker UNKNOWN. Then `drop_noise_blocks`, `skip_if_not_conversation`.

**Example (new format):**
```
[2025-31-03 19:09:41 INFO] CEO: **CEO<->CTO on : Phase, turn 1**
Hello, we need to build a game.
```

---

## 4. HyperAgent

**Rule:**
- If `trajectory` is **list[str]**: each element = one turn.
- If `trajectory` is **string**: turn boundary = `- INFO - <LABEL>:` where LABEL matches e.g. `Planner's Response`, `Inner-...-Assistant's Response`, `<A>-><B>`.

**Implementation:** `split_hyperagent(trajectory)`. List: one Turn per string. String: regex `^\S+\s*-\s*INFO\s*-\s*(?P<label>[^:]+?):\s*`; for each match, derive speaker from label (e.g. before `'` in "X's Response", or before `->`). `drop_noise_blocks`, `skip_if_not_conversation`.

---

## 5. MetaGPT

**Rule:**
- Outer boundary = each timestamp header `^[YYYY-MM-DD HH:MM:SS]`.
- If a block contains `FROM: X TO: Y` and `CONTENT:`, that block = one turn (speaker = X).
- If a block contains `NEW MESSAGES:`, split inside it by speaker lines like `SimpleCoder:` (one per agent).

**Implementation:** `split_metagpt(text)`. Split by timestamp regex `^\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s`. For each chunk: (A) if `FROM: ... TO: ...` present, one Turn with speaker from FROM; (B) if `NEW MESSAGES` present, split body by `^(?P<sp>[A-Za-z0-9_]+):\s*$` and one Turn per speaker block. Else skip chunk. `drop_noise_blocks`, `skip_if_not_conversation`.

---

## 6. OpenManus

**Rule:**
- Turn boundary = each `app.agent.toolcall:think:81 - ✨ Manus's thoughts:` (one “assistant turn”).
- Turn content = subsequent tool selection + execute_tool + observations until next `think:81`.

**Implementation:** `split_openmanus(text)`. Header regex: `^.*app\.agent\.toolcall:think:81\s+-\s+✨\s+Manus's thoughts:.*$`. `split_by_headers` with speaker "Manus"; strip header line from content. `drop_noise_blocks`, `skip_if_not_conversation`.

---

## 7. Magentic (excluded)

No turn structure (runtime/install output). `extract_turns("magentic", ...)` always returns `[]`.

---

## Dispatcher

```python
def extract_turns(task: str, trajectory) -> list[Turn]:
    task = task.strip().lower()
    if task == "magentic":
        return []
    if task == "ag2": return split_ag2(trajectory)
    if task == "appworld": return split_appworld(...)
    if task == "chatdev": return split_chatdev(...)
    if task == "hyperagent": return split_hyperagent(trajectory)
    if task == "metagpt": return split_metagpt(...)
    if task == "openmanus": return split_openmanus(...)
    return []
```

`get_steps_from_trace(path=..., data=..., text=..., task_type=...)` loads trajectory, calls `extract_turns(task_type, trajectory)`, then maps turns to step strings via `_turns_to_steps` (content with optional `[speaker]` prefix).
