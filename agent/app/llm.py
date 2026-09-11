"""OpenAI-compatible LLM client pointed at the Ollama daemon.

Two helpers only: ``complete(prompt) -> str`` and ``stream(prompt) -> token
iterator``. Errors (e.g. unavailable model tag) propagate to the caller and
surface as SSE error events.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

import openai

_log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://host.docker.internal:11434/v1"
_DEFAULT_MODEL = "glm-5.3-flash:cloud"
_TEMPERATURE = 0.2  # low for routing/retrieval fidelity

_CLIENT: openai.OpenAI | None = None


def _client() -> openai.OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = openai.OpenAI(
            base_url=os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL),
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            timeout=120.0,
        )
    return _CLIENT


def _model() -> str:
    return os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)


def complete(prompt: str, temperature: float = _TEMPERATURE) -> str:
    """One-shot completion."""
    response = _client().chat.completions.create(
        model=_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def stream(prompt: str, temperature: float = _TEMPERATURE) -> Iterator[str]:
    """Yield answer tokens in order."""
    response = _client().chat.completions.create(
        model=_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True,
    )
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content