"""Tests for provider-neutral generation contracts."""

import importlib
import inspect
import subprocess
import sys
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


@pytest.mark.parametrize(
    ("first_module", "second_module"),
    [
        ("providers", "scripts.providers"),
        ("scripts.providers", "providers"),
    ],
)
def test_provider_contract_identity_is_stable_across_import_order(
    first_module, second_module
):
    code = f"""
import importlib
import sys
from pathlib import Path

root = Path.cwd()
sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != root]
sys.path.insert(0, str(root / "scripts"))
if {first_module!r}.startswith("scripts."):
    sys.path.insert(0, str(root))
importlib.import_module({first_module!r})
sys.path.insert(0, str(root))
importlib.import_module({second_module!r})
direct = importlib.import_module("providers")
package = importlib.import_module("scripts.providers")
scripts = importlib.import_module("scripts")
assert direct is package
assert scripts.providers is package
assert direct.TaskKind is package.TaskKind
assert direct.TextRequest is package.TextRequest
assert direct.ProviderResult is package.ProviderResult
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
