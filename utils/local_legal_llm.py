"""Lazy loader for competition-compliant local legal LLM (Qwen3-4B)."""

from __future__ import annotations

import os
import re
from typing import Any

import torch
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config

_LOCAL_LLM: "LocalLegalLLM | None" = None


class LocalLegalLLM:
  """Open-source Vietnamese legal chat model (<14B, BTC-compliant)."""

  def __init__(self, model_name: str | None = None, max_new_tokens: int = 384):
    self.model_name = model_name or Config.MODEL_GEN
    self.max_new_tokens = max_new_tokens
    forced = os.getenv("LOCAL_LLM_DEVICE", "").strip().lower()
    if forced in {"cpu", "mps", "cuda"}:
      self.device = forced
    else:
      self.device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
      )
    torch_dtype = (
      torch.bfloat16
      if self.device == "cuda"
      else torch.float16
      if self.device == "mps"
      else torch.float32
    )

    print(f"Loading local LLM {self.model_name} on {self.device}...", flush=True)
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": torch_dtype}
    if self.device == "cuda":
      load_kwargs["device_map"] = "auto"
      self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
    else:
      self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
      self.model.to(self.device)
    self.model.eval()
    print(f"Local LLM ready: {self.model_name}", flush=True)

  def _messages_to_chat(self, messages: list[BaseMessage]) -> list[dict[str, str]]:
    chat: list[dict[str, str]] = []
    for msg in messages:
      if isinstance(msg, SystemMessage):
        chat.append({"role": "system", "content": str(msg.content)})
      elif isinstance(msg, HumanMessage):
        chat.append({"role": "user", "content": str(msg.content)})
      elif isinstance(msg, AIMessage):
        chat.append({"role": "assistant", "content": str(msg.content)})
    return chat

  def _generate_text(self, messages: list[BaseMessage]) -> str:
    chat = self._messages_to_chat(messages)
    if hasattr(self.tokenizer, "apply_chat_template"):
      prompt = self.tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
      )
    else:
      prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in chat) + "\nassistant:"

    inputs = self.tokenizer([prompt], return_tensors="pt")
    target_device = next(self.model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.no_grad():
      output_ids = self.model.generate(
        **inputs,
        max_new_tokens=self.max_new_tokens,
        do_sample=False,
        repetition_penalty=1.08,
        temperature=None,
        top_p=None,
      )

    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

  def invoke_plain(self, messages: list[BaseMessage]) -> str:
    """Generate plain-text answer (QA submissions)."""
    return self._generate_text(messages)

  def invoke(self, messages: list[BaseMessage]) -> AIMessage:
    text = self._generate_text(messages)
    # Keep only the first JSON object if model adds extra chatter.
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
      text = match.group(0)
    return AIMessage(content=text)


def get_local_legal_llm(model_name: str | None = None, max_new_tokens: int = 384) -> LocalLegalLLM:
  global _LOCAL_LLM
  if _LOCAL_LLM is None:
    _LOCAL_LLM = LocalLegalLLM(model_name=model_name, max_new_tokens=max_new_tokens)
  else:
    _LOCAL_LLM.max_new_tokens = max_new_tokens
  return _LOCAL_LLM


def competition_model_profile(model_name: str | None = None) -> dict[str, Any]:
  """Static profile for BTC rule check (<14B, open-source, release cutoff)."""
  name = model_name or Config.MODEL_GEN
  params_b = 4 if "4B" in name or "4b" in name else None
  return {
    "model_id": name,
    "open_source": True,
    "params_b_estimate": params_b,
    "btc_max_params_b": 14,
    "btc_release_cutoff": "2026-03-01",
    "known_release": "2025-08-15",
    "btc_compliant": params_b is not None and params_b < 14,
    "forbidden_examples": ["gpt-4o", "gemini", "claude"],
  }
