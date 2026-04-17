"""
Step (turn) extraction from MAST traces.
Turn-level split rules with Turn dataclass; dispatcher by path or task_type.
Magentic: always skip.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Any, Callable, Union

from . import config


@dataclass
class Turn:
    speaker: str
    content: str
    meta: dict  # optional: timestamp, phase, turn_id, header, label, etc.


def clean(s: str) -> str:
    s = s.replace("\r\n", "\n")
    return s.strip()


def drop_noise_blocks(turns: List[Turn], min_chars: int = 20) -> List[Turn]:
    out = []
    for t in turns:
        c = clean(t.content)
        if len(c) < min_chars:
            continue
        out.append(Turn(t.speaker.strip(), c, dict(t.meta)))
    return out


def skip_if_not_conversation(turns: List[Turn]) -> bool:
    if len(turns) < 2:
        return True
    short = sum(1 for t in turns if len(t.content.strip()) < 30)
    return (short / max(1, len(turns))) > 0.8


def split_by_headers(
    text: str,
    header_re: str,
    speaker_fn: Optional[Callable[[re.Match, str], str]] = None,
) -> List[Turn]:
    """
    header_re must match the *start* of each new turn header.
    speaker_fn: callable(match_obj, header_line) -> speaker
    """
    text = text.replace("\r\n", "\n")
    pat = re.compile(header_re, flags=re.M)

    hits = list(pat.finditer(text))
    if not hits:
        return []

    turns = []
    for i, m in enumerate(hits):
        start = m.start()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        block = text[start:end]
        header_line = block.split("\n", 1)[0]
        speaker = speaker_fn(m, header_line) if speaker_fn else "UNKNOWN"
        turns.append(Turn(speaker=speaker, content=block, meta={"header": header_line}))
    return turns


# ----- AG2 -----


def split_ag2(trajectory: Union[List, str]) -> List[Turn]:
    if isinstance(trajectory, list):
        turns = []
        for msg in trajectory:
            if not isinstance(msg, dict):
                continue
            speaker = str(msg.get("name") or msg.get("role") or "UNKNOWN")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(str(c) for c in content)
            turns.append(
                Turn(speaker, str(content), {"role": msg.get("role"), "name": msg.get("name")})
            )
        turns = drop_noise_blocks(turns)
        return [] if skip_if_not_conversation(turns) else turns

    text = (trajectory or "").replace("\r\n", "\n")
    triplet_re = re.compile(
        r"(?ms)^content:\s*(?P<content>.*?)^\s*role:\s*(?P<role>.*?)^\s*name:\s*(?P<name>.*?)(?=^\s*content:|\Z)"
    )
    turns = []
    for m in triplet_re.finditer(text):
        speaker = clean(m.group("name")) or clean(m.group("role")) or "UNKNOWN"
        turns.append(
            Turn(speaker, clean(m.group("content")), {"role": clean(m.group("role"))})
        )
    turns = drop_noise_blocks(turns)
    return [] if skip_if_not_conversation(turns) else turns


# ----- AppWorld -----


def split_appworld(text: str) -> List[Turn]:
    text = text.replace("\r\n", "\n")

    hdr = re.compile(
        r"(?m)^(Response from|Reply from)\s+(?P<speaker>.+?)\s+Agent\s*$"
    )
    hits = list(hdr.finditer(text))
    if not hits:
        return []

    msg_hdr = re.compile(r"(?m)^Message to\s+(?P<to>.+?)\s+Agent\s*$")
    msg_hits = list(msg_hdr.finditer(text))

    def nearest_message_to(speaker: str, pos: int) -> Optional[str]:
        best = None
        for m in msg_hits:
            if m.start() < pos and clean(m.group("to")) == speaker:
                best = m
        if not best:
            return None
        end = len(text)
        for nxt in list(hits) + msg_hits:
            if nxt.start() > best.start():
                end = min(end, nxt.start())
        return text[best.start() : end].strip()

    turns = []
    for i, m in enumerate(hits):
        start = m.start()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        block = text[start:end].strip()
        speaker = clean(m.group("speaker"))
        inp = nearest_message_to(speaker, start)
        if inp:
            block = f"{inp}\n\n{block}"
        turns.append(Turn(speaker=speaker, content=block, meta={}))
    turns = drop_noise_blocks(turns)
    return [] if skip_if_not_conversation(turns) else turns


# ----- ChatDev -----


def split_chatdev(text: str) -> List[Turn]:
    text = text.replace("\r\n", "\n")

    header_re = r"(?m)^\[\d{4}-\d{2}-\d{2} .*?\]\s+(?P<speaker>[^:]+):\s+\*\*.*?<->.*?turn\s+\d+\*\*.*$"

    def speaker_fn(m: re.Match, _line: str) -> str:
        return clean(m.group("speaker"))

    turns = split_by_headers(text, header_re, speaker_fn=speaker_fn)

    if turns:
        for t in turns:
            lines = t.content.split("\n")
            t.meta["header"] = lines[0]
            t.content = "\n".join(lines[1:]).strip()
    else:
        # Fallback: no "Speaker<->... turn k" lines (e.g. older format); split on **X**
        pattern = re.compile(r"^\s*\*\*[^*]+\*\*", re.MULTILINE)
        splits = list(pattern.finditer(text))
        if not splits:
            if text.strip():
                turns = [Turn("UNKNOWN", text.strip(), {})]
            else:
                turns = []
        else:
            if splits[0].start() > 0:
                pre = text[: splits[0].start()].strip()
                if pre:
                    turns.append(Turn("UNKNOWN", pre, {}))
            for i, m in enumerate(splits):
                start = m.start()
                end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
                block = text[start:end].strip()
                if block:
                    turns.append(Turn("UNKNOWN", block, {}))

    turns = drop_noise_blocks(turns)
    return [] if skip_if_not_conversation(turns) else turns


# ----- HyperAgent -----


def split_hyperagent(trajectory: Union[List, str]) -> List[Turn]:
    if isinstance(trajectory, list):
        turns = [
            Turn("UNKNOWN", s, {})
            for s in trajectory
            if s and isinstance(s, str) and s.strip()
        ]
        turns = drop_noise_blocks(turns)
        return [] if skip_if_not_conversation(turns) else turns

    text = (trajectory or "").replace("\r\n", "\n")
    header_re = r"(?m)^\S+\s*-\s*INFO\s*-\s*(?P<label>[^:]+?):\s*"
    pat = re.compile(header_re, flags=re.M)

    hits = list(pat.finditer(text))
    if not hits:
        return []

    def label_to_speaker(label: str) -> Optional[str]:
        label = label.strip()
        if "Response" in label and "'" in label:
            return label.split("'")[0].strip()
        if "->" in label:
            return label.split("->")[0].strip()
        return None

    turns = []
    for i, m in enumerate(hits):
        start = m.start()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        label = m.group("label").strip()
        speaker = label_to_speaker(label)
        if not speaker:
            continue
        block = text[start:end].strip()
        turns.append(Turn(speaker, block, meta={"label": label}))
    turns = drop_noise_blocks(turns)
    return [] if skip_if_not_conversation(turns) else turns


# ----- MetaGPT -----


def split_metagpt(text: str) -> List[Turn]:
    text = text.replace("\r\n", "\n")

    ts_re = r"(?m)^\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s"
    chunks = split_by_headers(text, ts_re, speaker_fn=lambda m, l: "TIMESTAMP")

    turns = []
    for ch in chunks:
        block = ch.content
        header = block.split("\n", 1)[0]
        body = block[len(header) :].strip() if len(block) > len(header) else ""

        m_from = re.search(
            r"FROM:\s*(?P<sp>.+?)\s+TO:\s*(?P<to>.+?)(?:\s*$|\s*\n)", block
        )
        if m_from:
            sp = clean(m_from.group("sp"))
            turns.append(Turn(sp, block, meta={"timestamp": header}))
            continue

        if "NEW MESSAGES" in body:
            sp_re = re.compile(r"(?m)^(?P<sp>[A-Za-z0-9_]+):\s*$")
            hits = list(sp_re.finditer(body))
            if hits:
                for i, m in enumerate(hits):
                    sp = m.group("sp").strip()
                    start = m.end()
                    end = hits[i + 1].start() if i + 1 < len(hits) else len(body)
                    msg = body[start:end].strip()
                    turns.append(Turn(sp, msg, meta={"timestamp": header}))
                continue

        continue

    turns = drop_noise_blocks(turns)
    return [] if skip_if_not_conversation(turns) else turns


# ----- OpenManus -----


def split_openmanus(text: str) -> List[Turn]:
    text = text.replace("\r\n", "\n")

    header_re = r"(?m)^.*app\.agent\.toolcall:think:81\s+-\s+✨\s+Manus's thoughts:.*$"
    turns = split_by_headers(text, header_re, speaker_fn=lambda _m, _l: "Manus")

    if turns:
        for t in turns:
            lines = t.content.split("\n")
            t.meta["header"] = lines[0]
            t.content = "\n".join(lines[1:]).strip()

    turns = drop_noise_blocks(turns)
    return [] if skip_if_not_conversation(turns) else turns


# ----- Dispatcher -----


def extract_turns(task: str, trajectory: Any) -> List[Turn]:
    """Extract turns from trajectory. Returns [] for Magentic or unknown task."""
    task_lower = (task or "").strip().lower()
    if task_lower == "magentic":
        return []

    if task_lower == "ag2":
        return split_ag2(trajectory)
    if task_lower == "appworld":
        text = trajectory if isinstance(trajectory, str) else ""
        if isinstance(trajectory, list):
            text = "\n".join(str(x) for x in trajectory)
        return split_appworld(text or "")
    if task_lower == "chatdev":
        text = trajectory if isinstance(trajectory, str) else ""
        if isinstance(trajectory, list):
            text = "\n".join(str(x) for x in trajectory)
        return split_chatdev(text or "")
    if task_lower == "hyperagent":
        return split_hyperagent(trajectory)
    if task_lower == "metagpt":
        text = trajectory if isinstance(trajectory, str) else ""
        if isinstance(trajectory, list):
            text = "\n".join(str(x) for x in trajectory)
        return split_metagpt(text or "")
    if task_lower == "openmanus":
        text = trajectory if isinstance(trajectory, str) else ""
        if isinstance(trajectory, list):
            text = "\n".join(str(x) for x in trajectory)
        return split_openmanus(text or "")

    return []


def _turns_to_steps(turns: List[Turn]) -> List[str]:
    """Convert turns to step content strings for prompts (optionally include speaker)."""
    steps = []
    for t in turns:
        if t.speaker and t.speaker != "UNKNOWN":
            steps.append(f"[{t.speaker}]\n{t.content}")
        else:
            steps.append(t.content)
    return steps


# ----- Path / legacy API -----


def _infer_task_type(path: Path) -> Optional[str]:
    p = path.as_posix()
    if "/AG2/" in p or "/math_interventions/" in p:
        return "AG2"
    if "/AppWorld/" in p:
        return "AppWorld"
    if "/chatdev/" in p or "/chatdev_mmlu/" in p:
        return "ChatDev"
    if "/HyperAgent/" in p:
        return "HyperAgent"
    if "/metagpt/" in p or "/metagpt_mmlu/" in p:
        return "MetaGPT"
    if "/OpenManus_GAIA/" in p:
        return "OpenManus"
    return None


def get_steps_from_trace(
    path: Optional[Path] = None,
    data: Optional[dict] = None,
    text: Optional[str] = None,
    task_type: Optional[str] = None,
) -> Tuple[List[str], Optional[str]]:
    """
    Return (steps, task_type). Steps are list of step content strings (with optional [speaker] prefix).
    Uses turn-level extraction; Magentic returns [].
    Provide path, or (data or text) with task_type.
    """
    trajectory: Any = None
    task = task_type

    if path is not None:
        path = Path(path)
        task = task or _infer_task_type(path)
        if not task or task not in config.SUPPORTED_TASK_TYPES:
            return [], task
        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            trajectory = data.get("trajectory") if data else None
            if trajectory is None and data:
                text = json.dumps(data, ensure_ascii=False)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            data = None

    if task is None:
        return [], None

    if trajectory is None:
        if data is not None and "trajectory" in data:
            trajectory = data["trajectory"]
        elif text is not None:
            trajectory = text

    if trajectory is None:
        return [], task

    turns = extract_turns(task, trajectory)
    steps = _turns_to_steps(turns)
    return steps, task
