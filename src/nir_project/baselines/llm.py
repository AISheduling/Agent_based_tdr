"""
Chat client for an OpenAI-compatible API
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    finish_reason: str = ""
    attempts: int = 1
    response_id: str = ""  # provider's id of the last response
    system_fingerprint: str = ""  # provider's backend configuration, if it reports one


class ChatClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]]) -> Completion: ...


@dataclass(frozen=True)
class LLMConfig:
    """Everything about the call that changes the answer; logged with every run."""

    name: str = "llm-structured"
    model: str = "deepseek-flash"
    temperature: float = 0.0
    max_tokens: int = 8000
    json_mode: bool = True
    retries_on_blank: int = 2  # JSON mode may return blank content; retry then give up
    extra_body: Mapping[str, Any] = field(default_factory=dict)  # provider-specific options

    @classmethod
    def from_yaml(cls, path: Path) -> LLMConfig:
        with path.open(encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls(**raw)


class OpenAICompatibleClient:
    """Server with the OpenAI chat completions API."""

    def __init__(self, cfg: LLMConfig, api_key: str, base_url: str) -> None:
        from openai import OpenAI

        self.cfg = cfg
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, messages: Sequence[Mapping[str, str]]) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [dict(m) for m in messages],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.cfg.extra_body:
            kwargs["extra_body"] = dict(self.cfg.extra_body)

        prompt_tokens = completion_tokens = 0
        started = time.perf_counter()
        attempts = 0
        text, model, finish, response_id, fingerprint = "", self.cfg.model, "", "", ""
        while attempts <= self.cfg.retries_on_blank:
            attempts += 1
            resp: Any = self._client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            text = choice.message.content or ""
            model, finish = str(resp.model), str(choice.finish_reason)
            response_id = str(getattr(resp, "id", "") or "")
            fingerprint = str(getattr(resp, "system_fingerprint", "") or "")
            if resp.usage is not None:
                prompt_tokens += int(resp.usage.prompt_tokens)
                completion_tokens += int(resp.usage.completion_tokens)
            # A cut-off answer (`length`) will be cut off again: retrying only burns tokens.
            if text.strip() or finish == "length":
                break
        return Completion(
            text=text,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            seconds=time.perf_counter() - started,
            finish_reason=finish,
            attempts=attempts,
            response_id=response_id,
            system_fingerprint=fingerprint,
        )
