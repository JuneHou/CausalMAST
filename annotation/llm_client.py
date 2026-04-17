"""
LLM client for MAST annotation.

Two modes:
  offline  — vLLM loaded in-process (--offline flag). Best for batch runs on a GPU server.
  api      — OpenAI-compatible HTTP API (vLLM server, OpenAI, etc.).

Both modes use chat format: a single user message containing the full prompt.
"""
import json
import os
import re
from typing import List, Optional


# ---------------------------------------------------------------------------
# JSON extraction from raw LLM output
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[dict]:
    """
    Robustly extract a JSON object from LLM output.
    Handles:
      - Clean JSON
      - Markdown code blocks (```json ... ``` or ``` ... ```)
      - JSON preceded/followed by explanation text
    Returns parsed dict, or None on failure.
    """
    # Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fenced:
        candidate = fenced.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Find the outermost {...} in the text
    start = text.find("{")
    if start == -1:
        return None
    # Walk forward matching braces
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break
    return None


def _category_matches(cat: str, valid_categories: set) -> bool:
    """Return True if cat matches any entry in valid_categories (full label or ID prefix)."""
    cat = cat.strip()
    return cat in valid_categories or any(cat.startswith(vc) for vc in valid_categories)


def validate_pass1_response(parsed: dict, valid_step_ids: set, valid_categories: set) -> dict:
    """
    Filter Pass 1 LLM response (candidates list).
    Removes entries whose category is not in valid_categories or whose
    location is not a valid step ID.
    Returns {"candidates": [...]}.
    """
    candidates = parsed.get("candidates") or []
    clean = []
    for c in candidates:
        cat = (c.get("category") or "").strip()
        loc = (c.get("location") or "").strip()
        if _category_matches(cat, valid_categories) and loc in valid_step_ids:
            clean.append(c)
    return {"candidates": clean}


def validate_pass2_response(parsed: dict, valid_step_ids: set, valid_categories: set) -> dict:
    """
    Filter Pass 2 LLM response (errors list).
    Removes entries whose category is not in valid_categories or whose
    location is not a valid step ID.
    Returns {"errors": [...]}.
    """
    errors = parsed.get("errors") or []
    clean = []
    for e in errors:
        cat = (e.get("category") or "").strip()
        loc = (e.get("location") or "").strip()
        if _category_matches(cat, valid_categories) and loc in valid_step_ids:
            clean.append(e)
    return {"errors": clean}


# ---------------------------------------------------------------------------
# vLLM offline client
# ---------------------------------------------------------------------------

class _VLLMClient:
    def __init__(self, model: str, tensor_parallel_size: Optional[int], pipeline_parallel_size: int,
                 temperature: float, max_tokens: int):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vllm not installed. Run: pip install vllm")

        n_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
        tp = tensor_parallel_size or n_gpus
        print(f"Loading {model} in-process (tensor_parallel={tp}, pipeline_parallel={pipeline_parallel_size})...")
        self._llm = LLM(model=model, tensor_parallel_size=tp, pipeline_parallel_size=pipeline_parallel_size)
        self._params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        print("Model loaded.")

    def generate(self, prompts: List[str]) -> List[str]:
        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        outputs = self._llm.chat(messages_batch, self._params)
        return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# OpenAI-compatible API client
# ---------------------------------------------------------------------------

class _APIClient:
    def __init__(self, model: str, base_url: str, api_key: str,
                 temperature: float, max_tokens: int):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, prompts: List[str]) -> List[str]:
        results = []
        for prompt in prompts:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
            results.append(resp.choices[0].message.content or "")
        return results


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Unified LLM client. Create once, call generate() for each batch.

    Args:
        model: model name (e.g. "Qwen/Qwen3-32B")
        offline: if True, load vLLM in-process; otherwise use API
        base_url: API base URL (only for offline=False)
        api_key: API key (only for offline=False)
        tensor_parallel_size: GPU tensor parallelism (only for offline=True)
        pipeline_parallel_size: GPU pipeline parallelism (only for offline=True)
        temperature: generation temperature (0.0 = deterministic)
        max_tokens: max tokens in LLM response
    """

    def __init__(
        self,
        model: str,
        offline: bool = False,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-abc123",
        tensor_parallel_size: Optional[int] = None,
        pipeline_parallel_size: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.model = model
        if offline:
            self._backend = _VLLMClient(model, tensor_parallel_size, pipeline_parallel_size, temperature, max_tokens)
        else:
            self._backend = _APIClient(model, base_url, api_key, temperature, max_tokens)

    def generate(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for a list of prompts.
        Returns raw text outputs (one per prompt); caller should parse with extract_json().
        """
        return self._backend.generate(prompts)
