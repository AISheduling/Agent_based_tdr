"""L1: structured LLM baseline. One package -> one call -> one graph; no documents, no loop."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import yaml

from nir_project.baselines.answer import ParsedAnswer, parse_answer
from nir_project.baselines.llm import ChatClient, Completion, LLMConfig
from nir_project.baselines.methods import MethodOutput, Task
from nir_project.baselines.package import Package
from nir_project.baselines.prompt import PROMPT_VERSION, build_messages


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


class LLMStructured:
    """L1 as a runner method. Reads only `task.package`."""

    model_family = "llm-structured"

    def __init__(self, cfg: LLMConfig, client: ChatClient) -> None:
        self.cfg = cfg
        self.client = client
        self.name = cfg.name

    def params(self) -> Mapping[str, Any]:
        return {**asdict(self.cfg), "prompt_version": PROMPT_VERSION}

    def predict(self, task: Task) -> MethodOutput:
        result = predict(task.package, self.client)
        c = result.completion
        return MethodOutput(
            graph=result.answer.graph,
            cost={
                "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens,
                "seconds": c.seconds,
                "calls": c.attempts,
                "cut_off": float(c.finish_reason == "length"),
            },
            artifacts={
                "prompt.json": json.dumps(result.messages, ensure_ascii=False, indent=1),
                "raw.txt": c.text,
                "answer.yaml": yaml.safe_dump(
                    result.answer.as_graph_dict(), allow_unicode=True, sort_keys=False
                ),
            },
            problems=result.answer.problems,
        )
