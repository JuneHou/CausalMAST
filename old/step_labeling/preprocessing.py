"""
Preprocessing for MAST step-level labeling.
1. **MAD (primary):** Load MAD dataset (from original MAST; same as failure_distribution_by_task.ipynb).
   MAD has trace content + mast_annotation in one place. Filter: exclude Magentic, keep >= 2 failure types.
2. **Optional:** Discover traces under traces/ and join with annotations file (custom JSONL) for path-based runs.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from . import config
from .step_extraction import _infer_task_type, get_steps_from_trace


def count_failures(mast_annotation: Dict[str, Any]) -> int:
    """Count number of failure types present (value == 1)."""
    if not mast_annotation:
        return 0
    return sum(1 for v in mast_annotation.values() if (isinstance(v, int) and v == 1) or (isinstance(v, str) and str(v).strip() == "1"))


def get_failure_types_set(mast_annotation: Dict[str, Any]) -> List[str]:
    """Return list of failure type keys with value 1 (present)."""
    if not mast_annotation:
        return []
    return [k for k in config.MAST_FAILURE_KEYS if k in mast_annotation and (mast_annotation[k] == 1 or str(mast_annotation.get(k, "")).strip() == "1")]


def discover_traces(traces_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Discover all trace files under traces_dir for supported task types (Magentic excluded).
    Returns list of { "path": str, "trace_id": str, "mas_name": str }.
    """
    traces_dir = Path(traces_dir or config.TRACES_DIR)
    if not traces_dir.is_dir():
        return []

    records = []
    seen_paths = set()

    for mas_name, patterns in config.TRACE_PATTERNS.items():
        for pat in patterns:
            full_pat = traces_dir / pat
            # glob: ** for experiments/*/*.json
            for path in traces_dir.glob(pat):
                if not path.is_file():
                    continue
                key = path.resolve().as_posix()
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                trace_id = path.stem or path.name
                records.append({
                    "path": str(path.resolve()),
                    "trace_id": trace_id,
                    "mas_name": mas_name,
                })
    return records


def load_annotations(annotations_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load annotations file: JSONL (one JSON object per line) or JSON array.
    Each record should have trace_id or path, and mast_annotation.
    Returns dict mapping trace_id or path -> { mast_annotation, ... }.
    """
    path = Path(annotations_path)
    if not path.is_file():
        return {}

    by_id = {}
    by_path = {}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
        f.seek(0)
        try:
            if first.strip().startswith("["):
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
            else:
                items = [json.loads(line) for line in f if line.strip()]
        except Exception:
            return {}

    for r in items:
        ann = r.get("mast_annotation") or {}
        tid = r.get("trace_id") or r.get("id")
        p = r.get("path") or r.get("trace_path")
        if tid is not None:
            by_id[str(tid)] = {**r, "mast_annotation": ann}
        if p is not None:
            by_path[str(Path(p).resolve())] = {**r, "mast_annotation": ann}
    return {"by_id": by_id, "by_path": by_path}


def filter_traces_with_n_failures(
    records: List[Dict[str, Any]],
    annotations_path: Optional[Path] = None,
    min_failures: int = 2,
) -> List[Dict[str, Any]]:
    """
    If annotations_path is given, attach mast_annotation to each record and keep only
    records with count(mast_annotation) >= min_failures.
    If annotations_path is not given, return records as-is (no mast_annotation; user must supply later).
    """
    if not annotations_path or not Path(annotations_path).is_file():
        return records

    ann_map = load_annotations(Path(annotations_path))
    by_id = ann_map.get("by_id", {})
    by_path = ann_map.get("by_path", {})

    out = []
    for r in records:
        path = r.get("path", "")
        tid = r.get("trace_id", "")
        ann_record = by_path.get(path) or by_path.get(str(Path(path).resolve())) or by_id.get(str(tid))
        if not ann_record:
            continue
        mast_ann = ann_record.get("mast_annotation") or {}
        n = count_failures(mast_ann)
        if n >= min_failures:
            r = {**r, "mast_annotation": mast_ann, "n_failures": n}
            r["failure_types"] = get_failure_types_set(mast_ann)
            out.append(r)
    return out


def load_mad_records(
    mad_path: Path,
    min_failures: int = 2,
    exclude_magentic: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load MAD dataset (MAD_full_dataset.json from HuggingFace mcemri/MAD or local).
    MAD is the MAST-derived dataset: each record has trace content + mast_annotation (same as in
    notebooks/failure_distribution_by_task.ipynb). Filter: exclude Magentic, keep >= min_failures.
    Returns list of { trace_id, mas_name, mast_annotation, failure_types, trajectory }.
    trajectory = trace["trajectory"] string for step extraction.
    """
    path = Path(mad_path)
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    records = []
    for r in data:
        mas_name = r.get("mas_name") or ""
        if exclude_magentic and mas_name == "Magentic":
            continue
        if mas_name not in config.SUPPORTED_TASK_TYPES:
            continue
        mast_ann = r.get("mast_annotation") or {}
        n = count_failures(mast_ann)
        if n < min_failures:
            continue
        trace_obj = r.get("trace") or r
        if isinstance(trace_obj, dict):
            trajectory = trace_obj.get("trajectory")
        else:
            trajectory = trace_obj
        if trajectory is None or (isinstance(trajectory, str) and not trajectory.strip()):
            continue
        # Allow both string and list trajectories (AG2/HyperAgent can be list)
        if not isinstance(trajectory, (str, list)):
            continue
        records.append({
            "trace_id": r.get("trace_id", ""),
            "mas_name": mas_name,
            "mast_annotation": mast_ann,
            "n_failures": n,
            "failure_types": get_failure_types_set(mast_ann),
            "trajectory": trajectory,
        })
    return records


def run_preprocessing(
    traces_dir: Optional[Path] = None,
    annotations_path: Optional[Path] = None,
    min_failures: int = 2,
    output_manifest_path: Optional[Path] = None,
    mad_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    If mad_path is set: load MAD (trace + annotations from original MAST), filter, return records.
    Else: discover traces under traces_dir, optionally filter by annotations file >= min_failures.
    """
    if mad_path and Path(mad_path).is_file():
        records = load_mad_records(Path(mad_path), min_failures=min_failures, exclude_magentic=True)
        if output_manifest_path:
            Path(output_manifest_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_manifest_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        return records
    records = discover_traces(traces_dir)
    if annotations_path:
        records = filter_traces_with_n_failures(records, annotations_path, min_failures)
    if output_manifest_path:
        output_manifest_path = Path(output_manifest_path)
        output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_manifest_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    return records
