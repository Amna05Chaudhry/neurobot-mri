"""Tiny OpenAI-compatible LLM client.

Configured from .env (see .env.example). Works with Groq (default, free key,
gpt-oss / Qwen models), OpenAI, or a local Ollama server. If no key is set or the call
fails, `chat()` returns None and callers fall back to templated answers.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

_client = None


def available() -> bool:
    """True when an API key is configured (Ollama accepts any non-empty key)."""
    return bool(API_KEY.strip())


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=30, max_retries=1)
    return _client


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 1200) -> str | None:
    """Return the model's reply, or None when the LLM is offline or errors."""
    if not available():
        return None
    try:
        resp = _get_client().chat.completions.create(
            model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as e:  # noqa: BLE001 - any API/network error -> fallback
        print(f"[llm] call failed, using fallback: {type(e).__name__}: {e}")
        return None
