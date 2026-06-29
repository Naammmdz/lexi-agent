"""vLLM batched inference for BTC-compliant Vietnamese legal QA."""

from __future__ import annotations

from typing import Any

_VLLM_ENGINE: Any = None
_VLLM_TOKENIZER: Any = None
_VLLM_MODEL: str | None = None


def get_vllm_engine(model_name: str, max_model_len: int = 3072) -> tuple[Any, Any]:
    global _VLLM_ENGINE, _VLLM_TOKENIZER, _VLLM_MODEL
    if _VLLM_ENGINE is not None and _VLLM_MODEL == model_name:
        return _VLLM_ENGINE, _VLLM_TOKENIZER

    from transformers import AutoTokenizer
    from vllm import LLM

    print(f"Loading vLLM engine: {model_name} (max_len={max_model_len})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    engine = LLM(
        model=model_name,
        dtype="bfloat16",
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    _VLLM_ENGINE = engine
    _VLLM_TOKENIZER = tokenizer
    _VLLM_MODEL = model_name
    print("vLLM engine ready.", flush=True)
    return engine, tokenizer


def messages_to_prompt(tokenizer: Any, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"system: {system}\n\nuser: {user}\nassistant:"


def generate_batch(
    prompts: list[str],
    model_name: str,
    max_new_tokens: int = 200,
    max_model_len: int = 3072,
) -> list[str]:
    if not prompts:
        return []

    from vllm import SamplingParams

    engine, _ = get_vllm_engine(model_name, max_model_len=max_model_len)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        repetition_penalty=1.05,
    )
    outputs = engine.generate(prompts, sampling_params=sampling)
    results: list[str] = []
    for out in outputs:
        text = out.outputs[0].text if out.outputs else ""
        results.append(str(text).strip())
    return results
