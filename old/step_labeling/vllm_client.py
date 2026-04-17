"""
vLLM-backed LLM client for MAST step-level labeling.
Supports (1) in-process vLLM: load model at start, send queries directly (no server/URL).
(2) Optional: OpenAI-compatible API when a vLLM server is already running.
Default model: Qwen/Qwen3-32B; option for gpt-oss open-source model.
"""
import os
from typing import Optional, Any

from . import config

# When using server mode
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "token-abc123"

# In-process vLLM: one LLM instance per process (lazy init)
_offline_llm: Any = None
_offline_llm_model: Optional[str] = None
_offline_tp: Optional[int] = None
_offline_pp: Optional[int] = None


def _visible_gpu_count() -> int:
    """
    Number of visible GPUs, derived from CUDA_VISIBLE_DEVICES.
    If CUDA_VISIBLE_DEVICES is unset, fall back to torch.cuda.device_count() when available.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        cvd = cvd.strip()
        if cvd == "":
            return 0
        # e.g. "0,1,2" or "0"
        return len([x for x in cvd.split(",") if x.strip() != ""])
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def get_offline_llm(
    model: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
    pipeline_parallel_size: Optional[int] = None,
    dtype: str = "auto",
) -> Any:
    """
    Create or return cached vLLM LLM instance for in-process inference.
    Call once at startup when using --offline; then pass the returned instance to complete().
    """
    global _offline_llm, _offline_llm_model, _offline_tp, _offline_pp
    model = model or config.DEFAULT_VLLM_MODEL
    if tensor_parallel_size is None:
        # User said they will set CUDA_VISIBLE_DEVICES; default TP = visible GPU count.
        # If 0 GPUs visible, default to 1 (CPU / fallback behavior depends on vLLM install).
        tensor_parallel_size = max(1, _visible_gpu_count())
    if pipeline_parallel_size is None:
        pipeline_parallel_size = 1

    if (
        _offline_llm is not None
        and _offline_llm_model == model
        and _offline_tp == tensor_parallel_size
        and _offline_pp == pipeline_parallel_size
    ):
        return _offline_llm
    try:
        from vllm import LLM
    except ImportError:
        raise ImportError("Install vllm for offline mode: pip install vllm")
    _offline_llm = LLM(
        model=model,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
    )
    _offline_llm_model = model
    _offline_tp = tensor_parallel_size
    _offline_pp = pipeline_parallel_size
    return _offline_llm


def complete_offline(
    prompt: str,
    llm: Any,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    """
    Single completion using an in-process vLLM LLM instance. No server or URL.
    Formats prompt as a simple user turn so the model generates the assistant reply.
    """
    try:
        from vllm import SamplingParams
    except ImportError:
        raise ImportError("Install vllm: pip install vllm")
    # Chat-style: many models accept "User: ...\n\nAssistant:" and continue
    formatted = f"User: {prompt}\n\nAssistant:"
    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate([formatted], sampling)
    if not outputs or not outputs[0].outputs:
        return ""
    return (outputs[0].outputs[0].text or "").strip()


def get_client(base_url: Optional[str] = None, api_key: Optional[str] = None):
    """Return OpenAI-compatible client (for server mode)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai for server mode: pip install openai")
    return OpenAI(
        base_url=base_url or DEFAULT_VLLM_BASE_URL,
        api_key=api_key or DEFAULT_API_KEY,
    )


def complete(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    use_offline: bool = False,
    llm: Any = None,
) -> str:
    """
    Single completion.
    - If use_offline=True and llm is set: use in-process vLLM (no server/URL).
    - Otherwise: use OpenAI-compatible API (vLLM server at base_url).
    """
    if use_offline and llm is not None:
        return complete_offline(prompt, llm=llm, max_tokens=max_tokens, temperature=temperature)
    # Server mode
    client = get_client(base_url=base_url, api_key=api_key)
    model = model or config.DEFAULT_VLLM_MODEL
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not resp.choices:
        return ""
    return (resp.choices[0].message.content or "").strip()
