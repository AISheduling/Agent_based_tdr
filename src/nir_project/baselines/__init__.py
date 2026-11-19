"""Baselines: methods that map an input package to a relation graph for the verifier."""

from nir_project.baselines.answer import ParsedAnswer, parse_answer
from nir_project.baselines.llm import ChatClient, Completion, LLMConfig
from nir_project.baselines.methods import Method, MethodOutput, Task
from nir_project.baselines.package import Package, build_package, load_family_meta
from nir_project.baselines.structured import BaselineResult, LLMStructured, predict

__all__ = [
    "BaselineResult",
    "ChatClient",
    "Completion",
    "LLMConfig",
    "LLMStructured",
    "Method",
    "MethodOutput",
    "Package",
    "ParsedAnswer",
    "Task",
    "build_package",
    "load_family_meta",
    "parse_answer",
    "predict",
]
