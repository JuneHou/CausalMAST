"""Configuration for MAST step-level annotation pipeline."""
from pathlib import Path

MAST_ROOT = Path(__file__).resolve().parent.parent
DEFINITIONS_PATH = MAST_ROOT / "taxonomy_definitions_examples" / "definitions.txt"
DEFAULT_MAD_PATH = MAST_ROOT / "old" / "data" / "raw" / "MAD_full_dataset.json"

SUPPORTED_TASK_TYPES = ["AG2", "AppWorld", "ChatDev", "HyperAgent", "MetaGPT", "OpenManus"]

MAST_FAILURE_KEYS = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
    "3.1", "3.2", "3.3",
]

DEFAULT_VLLM_MODEL = "Qwen/Qwen3-32B"

PROMPT_VARIANTS = ["zero_shot", "few_shot"]
DEFAULT_PROMPT_VARIANT = "few_shot"
