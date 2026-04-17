"""
Configuration for MAST step-level failure labeling.
Uses only the official dataset under traces/.
"""
from pathlib import Path

# Official dataset root (MAST repo)
MAST_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = MAST_ROOT / "traces"
DEFINITIONS_PATH = MAST_ROOT / "taxonomy_definitions_examples" / "definitions.txt"

# MAD dataset (from original MAST; download via scripts/0_download_mad.py)
DATA_DIR = MAST_ROOT / "data" / "raw"
DEFAULT_MAD_PATH = DATA_DIR / "MAD_full_dataset.json"

# Task types we support (splittable; Magentic excluded)
SUPPORTED_TASK_TYPES = ["AG2", "AppWorld", "ChatDev", "HyperAgent", "MetaGPT", "OpenManus"]
MAST_FAILURE_KEYS = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
    "3.1", "3.2", "3.3",
]

# vLLM: default and optional models
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-32B"
GPT_OSS_MODEL = "openai-community/gpt2"  # example open-source; user can override via --model

# Path patterns per task type (relative to TRACES_DIR)
TRACE_PATTERNS = {
    "AG2": ["AG2/*.json", "AG2/experiments/*/*.json", "math_interventions/topology_traces/*.txt", "math_interventions/org_traces/*.txt", "math_interventions/prompt_traces/*.txt"],
    "AppWorld": ["AppWorld/*.txt"],
    "ChatDev": ["programdev/chatdev/*/*.log", "mmlu/chatdev_mmlu/*.log"],
    "HyperAgent": ["HyperAgent/*.json"],
    "MetaGPT": ["programdev/metagpt/*.txt", "mmlu/metagpt_mmlu/*.txt"],
    "OpenManus": ["OpenManus_GAIA/*.log"],
}
