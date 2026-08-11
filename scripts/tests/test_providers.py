"""Tests for provider-neutral generation contracts."""

import importlib
import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from providers import (
    GenerationProvider,
    ProviderResult,
    TaskKind,
    TextRequest,
    WorkspaceRequest,
)


def test_task_kinds_are_exact():
    assert set(TaskKind) == {
        TaskKind.EXTRACT,
        TaskKind.COMPILE,
        TaskKind.QUERY,
        TaskKind.CONNECTIONS,
        TaskKind.FILE_ANSWER,
        TaskKind.SEMANTIC_LINT,
    }
    assert [task.value for task in TaskKind] == [
        "extract",
        "compile",
        "query",
        "connections",
        "file_answer",
        "semantic_lint",
    ]


def test_text_request_is_immutable(tmp_path):
    request = TextRequest(
        task=TaskKind.QUERY,
        prompt="What changed?",
        cwd=tmp_path,
        timeout_seconds=30,
        output_schema=tmp_path / "answer.schema.json",
    )

    with pytest.raises(FrozenInstanceError):
        request.prompt = "mutated"


def test_workspace_request_adds_relative_output_allowlist(tmp_path):
    request = WorkspaceRequest(
        task=TaskKind.COMPILE,
        prompt="Compile staged notes",
        cwd=tmp_path,
        timeout_seconds=60,
        allowed_paths=("knowledge/concepts/topic.md",),
    )

    assert isinstance(request, TextRequest)
    assert request.output_schema is None
    assert request.allowed_paths == ("knowledge/concepts/topic.md",)


def test_provider_result_is_immutable():
    result = ProviderResult(
        provider="codex",
        model="gpt-5.6-terra",
        task=TaskKind.QUERY,
        outcome="success",
        text="answer",
        input_tokens=12,
        output_tokens=4,
        elapsed_ms=25,
    )

    assert result.reason is None
    with pytest.raises(FrozenInstanceError):
        result.text = "mutated"


def test_generation_provider_declares_async_contract():
    assert inspect.iscoroutinefunction(GenerationProvider.generate_text)
    assert inspect.iscoroutinefunction(GenerationProvider.edit_workspace)


def test_request_paths_are_path_objects(tmp_path):
    schema = tmp_path / "schema.json"
    request = TextRequest(TaskKind.EXTRACT, "prompt", tmp_path, 5, schema)

    assert isinstance(request.cwd, Path)
    assert isinstance(request.output_schema, Path)


def test_package_import_uses_package_task_kind(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    package_config = importlib.import_module("scripts.config")
    package_providers = importlib.import_module("scripts.providers")

    assert package_config.TaskKind is package_providers.TaskKind
