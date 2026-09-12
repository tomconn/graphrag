"""OpenAI-compatible LLM client (Chat Completions API).

Two helpers only: ``complete(prompt) -> str`` and ``stream(prompt) -> token
iterator``. Any OpenAI-compatible endpoint works — set ``LLM_BASE_URL`` /
``LLM_API_KEY`` / ``LLM_MODEL``. Errors (e.g. unavailable model tag)
propagate to the caller and surface as SSE error events.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterator

import openai

_log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://host.docker.internal:11434/v1"
_DEFAULT_API_KEY = "noop"  # placeholder; real keys only needed by authenticated endpoints
_DEFAULT_MODEL = "glm-5.3-flash:cloud"
_TEMPERATURE = 0.2  # low for routing/retrieval fidelity
# Reasoning models spend completion tokens on their reasoning channel before
# content, so the cap must be generous enough for an answer plus its thinking.
_DEFAULT_MAX_TOKENS = 4096

_CLIENT: openai.OpenAI | None = None


def _client() -> openai.OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = openai.OpenAI(
            base_url=os.environ.get("LLM_BASE_URL", _DEFAULT_BASE_URL),
            api_key=os.environ.get("LLM_API_KEY", _DEFAULT_API_KEY),
            timeout=120.0,
        )
    return _CLIENT


def _model() -> str:
    return os.environ.get("LLM_MODEL", _DEFAULT_MODEL)


def _max_tokens() -> int:
    """Completion budget per call (unbounded consumption guard)."""
    return int(os.environ.get("AGENT_MAX_TOKENS", _DEFAULT_MAX_TOKENS))


def _purpose_from_prompt(prompt: str) -> str:
    """Fallback label when the caller does not name the stage: the flattened
    first words of the prompt (prompt text is untrusted data — this is for
    operator logs only)."""
    return " ".join(prompt.strip().split())[:48]


def complete(
    prompt: str,
    temperature: float = _TEMPERATURE,
    purpose: str = "",
    max_tokens: int | None = None,
) -> str:
    """One-shot completion. ``purpose`` names the pipeline stage making the
    call so the log shows which stage used the LLM; it falls back to a
    prompt excerpt. ``max_tokens`` overrides the per-call completion budget
    (stages with long outputs may raise it)."""
    purpose = purpose or _purpose_from_prompt(prompt)
    budget = max_tokens if max_tokens is not None else _max_tokens()
    _log.info("llm call purpose=%s model=%s prompt_chars=%d max_tokens=%d",
              purpose, _model(), len(prompt), budget)
    started = time.monotonic()
    response = _client().chat.completions.create(
        model=_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=budget,
    )
    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    if getattr(choice, "finish_reason", None) == "length":
        # A truncated completion explains downstream syntax errors (e.g. a
        # Cypher statement cut mid-pattern) — say so instead of leaving a
        # mysterious parse failure.
        _log.warning("llm reply purpose=%s truncated by the %d-token cap; "
                     "output is incomplete", purpose, budget)
    _log.info("llm reply purpose=%s completion_chars=%d elapsed=%.1fs",
              purpose, len(content), time.monotonic() - started)
    return content


def stream(
    prompt: str,
    temperature: float = _TEMPERATURE,
    purpose: str = "",
    max_tokens: int | None = None,
) -> Iterator[str]:
    """Yield answer tokens in order; the reply line logs when the stream is
    fully consumed."""
    purpose = purpose or _purpose_from_prompt(prompt)
    budget = max_tokens if max_tokens is not None else _max_tokens()
    _log.info("llm call purpose=%s model=%s prompt_chars=%d max_tokens=%d",
              purpose, _model(), len(prompt), budget)
    started = time.monotonic()
    response = _client().chat.completions.create(
        model=_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=budget,
        stream=True,
    )

    def _generate() -> Iterator[str]:
        chars = 0
        finish_reason = None
        for chunk in response:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta
                if delta and delta.content:
                    chars += len(delta.content)
                    yield delta.content
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
        if finish_reason == "length":
            _log.warning("llm reply purpose=%s truncated by the %d-token "
                         "cap; output is incomplete", purpose, budget)
        _log.info("llm reply purpose=%s completion_chars=%d elapsed=%.1fs",
                  purpose, chars, time.monotonic() - started)

    return _generate()