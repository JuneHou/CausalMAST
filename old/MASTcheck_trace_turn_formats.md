# MAST Task Types: Trace / Turn Format Check

For step-level failure labeling we need to split each trace into **steps** (turns). This document records, for each MAST task type present in the **official dataset** (`traces/`), whether traces use **consistent keywords** or a clear **turn structure** so we can break into steps reliably.

**Source:** Official MAST traces only, under `traces/` (no external or REM_MAST paths).

---

## Summary Table

| Task type     | Location in `traces/`           | Consistent keyword / delimiter? | Turn structure              |
|---------------|----------------------------------|---------------------------------|-----------------------------|
| **AG2**       | `AG2/`, `math_interventions/`    | Yes                             | JSON list of messages, or role-prefixed lines |
| **AppWorld**  | `AppWorld/`                     | Yes                             | Section headers (Task, Response from, Message to, etc.) |
| **ChatDev**   | `programdev/chatdev/`           | Yes                             | `[DATE INFO]` + `**Role**: **... on : Phase, turn N**` |
| **HyperAgent**| `HyperAgent/`                  | Yes                             | List of log-line strings; optional block by keyword |
| **Magentic**  | `MagenticOne_GAIA/`            | No (runtime log only)           | No conversation turns in trace files |
| **MetaGPT**   | `programdev/metagpt/`, `mmlu/metagpt_mmlu/` | Yes                    | `[YYYY-MM-DD HH:MM:SS]`, `NEW MESSAGES:`, `----` |
| **OpenManus** | `OpenManus_GAIA/`              | Yes                             | `Executing step X/N`; `✨ Manus's thoughts:` |

---

## 1. AG2

- **Paths:**  
  - `traces/AG2/*.json` and `traces/AG2/experiments/*/*.json` — JSON traces.  
  - `traces/math_interventions/topology_traces/trace_*.txt`, `traces/math_interventions/org_traces/`, `traces/math_interventions/prompt_traces/` — plain-text traces.
- **Format (JSON):** Each JSON has **`trajectory`** = list of message objects. Each object has `content` (list of strings), `role`, `name` (e.g. `"user"`, `"assistant"`, `"mathproxyagent"`). One list element = one turn.
- **Format (plain text, e.g. topology_traces):** Lines starting with **`user:`**, **`assistant:`**, or **`Agent_<Name>:`** (e.g. `Agent_Code_Executor:`, `Agent_Problem_Solver:`, `Agent_Verifier:`). Content runs until the next such prefix.
- **Consistent:** Yes.
- **Step extraction:**  
  - **JSON:** Use `trajectory[i]` as step `i`; step content = `"\n".join(msg["content"])`.  
  - **Plain text:** Split on lines matching `^(user|assistant|Agent_\w+):`; each contiguous block (prefix + following lines) = one step.

---

## 2. AppWorld

- **Path:** `traces/AppWorld/*.txt` (e.g. `229360a_1.txt`).
- **Format:** Plain text with section headers. Task line: **`******************** Task X/Y (id)  ********************`**. Then alternating blocks: **`Response from Supervisor Agent`**, **`Code Execution Output`**, **`Entering <Name> Agent message loop`**, **`Message to <Name> Agent`**, **`Response from <Name> Agent`**.
- **Consistent:** Yes. Same header patterns across AppWorld traces.
- **Step extraction:** Split on lines that are section headers (e.g. regex for `^\*+ Task`, `^Response from`, `^Code Execution Output`, `^Entering .* message loop`, `^Message to`, `^Response from`); treat each such block (header + following lines until next header) as one step.

---

## 3. ChatDev

- **Path:** `traces/programdev/chatdev/<TaskName>/*.log` (e.g. `Wordle/Wordle_DefaultOrganization_20250329233722.log`). Also `traces/mmlu/chatdev_mmlu/*.log`.
- **Format:** Log-style text. **Timestamp:** `[YYYY-DD-MM HH:MM:SS INFO]`. **Turn marker:** line like `**Role Name**: **Role Name<->Other Role on : PhaseName, turn N**` (e.g. `Chief Product Officer: **Chief Product Officer<->Chief Executive Officer on : DemandAnalysis, turn 0**`). Next turn starts at the next timestamp line.
- **Consistent:** Yes.
- **Step extraction:** Split on regex for `\[[^\]]+ INFO\]`; each block from one such line (inclusive) to the next (exclusive) = one step. Optionally parse role and phase from the following `**...**` line.

---

## 4. HyperAgent

- **Path:** `traces/HyperAgent/*.json` (e.g. `sympy__sympy-24213.json`).
- **Format:** JSON with **`trajectory`** = **list of strings** (each string is one log line). Line prefix: `HyperAgent_<instance_id> - INFO -`. For turn-like blocks: **`Planner's Response:`**, **`Intern Name:`**, **`Subgoal:`** (from MAST definitions/examples).
- **Consistent:** Yes.
- **Step extraction:**  
  - **Option A:** Each `trajectory[i]` = step `i`.  
  - **Option B:** Group consecutive lines into blocks by lines containing `Planner's Response`, `Intern Name:`, or `Subgoal:`; each block = one step.

---

## 5. Magentic (MagenticOne_GAIA)

- **Path:** `traces/MagenticOne_GAIA/gaia_validation_level_*__MagenticOne/<run_id>/0/` — contains `console_log.txt`, `prompt.txt`, `expected_answer.txt`, `scenario.py`, etc.
- **Format:** **`console_log.txt`** is runtime/install output (pip, tracebacks, Docker). It does not contain structured agent conversation turns. No separate conversation-trace file is present in the official dataset.
- **Consistent:** N/A for turns. No turn structure to parse from these trace files.
- **Step extraction:** Not defined from current trace contents. If step-level labeling is required for Magentic, either a different trace export (e.g. message list) would need to be added to the dataset, or turn boundaries would need to be inferred heuristically from `console_log.txt` (e.g. search for agent-like lines if any appear in longer runs).

---

## 6. MetaGPT

- **Paths:**  
  - `traces/programdev/metagpt/programdev_*.txt`  
  - `traces/mmlu/metagpt_mmlu/*.txt`
- **Format:** Text with timestamp lines and horizontal rules. **Timestamp:** `[YYYY-MM-DD HH:MM:SS]` followed by either `FROM: ... TO: ...` / `ACTION: ...` / `CONTENT:` or **`NEW MESSAGES:`** then agent name (e.g. `SimpleCoder:`, `SimpleTester:`). **Block separator:** full line `--------------------------------------------------------------------------------`.
- **Consistent:** Yes.
- **Step extraction:** Split on regex for `\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]`; optionally treat `NEW MESSAGES` as start of a new turn and group following lines until the next timestamp or `----`.

---

## 7. OpenManus

- **Path:** `traces/OpenManus_GAIA/*.log`.
- **Format:** Log lines. **Timestamp:** `YYYY-MM-DD HH:MM:SS.mmm | INFO     |`. **Step start:** **`Executing step X/20`** (or similar). **Content markers:** **`✨ Manus's thoughts:`**, **`🛠️ Manus selected ... tools`**.
- **Consistent:** Yes.
- **Step extraction:** Split on lines containing `Executing step `; each block from one `Executing step k/N` through the line before the next `Executing step (k+1)/N` (or end of file) = one step.

---

## Implementation Notes

- **Unified interface:** Implement a dispatcher that, given a trace (file path or loaded record), selects a splitter by task type (inferred from path or explicit argument). Each splitter returns `steps = [step_0, step_1, ...]` and optionally agent/label per step. Use this in the MAST step-level labeling pipeline (Pass 1/Pass 2) for local-window context and verbatim-evidence checks.
- **Path → task type:** Map `traces/AG2/` and `traces/math_interventions/` → AG2; `traces/AppWorld/` → AppWorld; `traces/programdev/chatdev/` and `traces/mmlu/chatdev_mmlu/` → ChatDev; `traces/HyperAgent/` → HyperAgent; `traces/MagenticOne_GAIA/` → Magentic; `traces/programdev/metagpt/` and `traces/mmlu/metagpt_mmlu/` → MetaGPT; `traces/OpenManus_GAIA/` → OpenManus.
