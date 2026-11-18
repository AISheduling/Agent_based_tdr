"""L1: structured LLM baseline. One package -> one call -> one graph; no documents, no loop."""

from __future__ import annotations

from dataclasses import dataclass

from nir_project.baselines.answer import ParsedAnswer, parse_answer
from nir_project.baselines.llm import ChatClient, Completion
from nir_project.baselines.package import Package
from nir_project.baselines.prompt import build_messages


@dataclass(frozen=True)
class BaselineResult:
    package: Package
    messages: list[dict[str, str]]
    completion: Completion
    answer: ParsedAnswer


def predict(package: Package, client: ChatClient) -> BaselineResult:
    messages = build_messages(package)
    completion = client.complete(messages)
    return BaselineResult(package, messages, completion, parse_answer(completion.text, package))
