#!/usr/bin/env python3
"""
Download MAD (Multi-Agent systems failure Dataset) from HuggingFace.

MAD is the MAST-derived dataset: trace content + mast_annotation per record.
Same dataset as used in notebooks/failure_distribution_by_task.ipynb.
Saves into MAST/data/raw/ so the MAST repo is self-contained (no REM_MAST dependency).

Usage:
    python scripts/0_download_mad.py
    python scripts/0_download_mad.py --out_dir /path/to/mast/data/raw
"""

import argparse
import shutil
import sys
from pathlib import Path

# MAST repo root (script is in MAST/scripts/)
MAST_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = MAST_ROOT / "data" / "raw"
REPO_ID = "mcemri/MAD"
FILENAME = "MAD_full_dataset.json"


def main():
    ap = argparse.ArgumentParser(description="Download MAD dataset from HuggingFace into MAST repo")
    ap.add_argument(
        "--repo_id",
        default=REPO_ID,
        help=f"HuggingFace dataset repo (default: {REPO_ID})",
    )
    ap.add_argument(
        "--filename",
        default=FILENAME,
        help=f"Dataset filename (default: {FILENAME})",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: MAST/data/raw)",
    )
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.filename

    print(f"Downloading {args.filename} from {args.repo_id}...")
    cached = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        repo_type="dataset",
    )
    cached_path = Path(cached)

    if out_path.resolve() == cached_path.resolve():
        print(f"✓ Already at {out_path}")
        return

    if out_path.exists():
        out_path.unlink()
    shutil.copy2(cached_path, out_path)
    print(f"✓ Copied to {out_path}")


if __name__ == "__main__":
    main()
