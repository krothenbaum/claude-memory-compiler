"""Tests for provider-neutral generation contracts."""

import asyncio
import importlib
import inspect
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from providers import (
    GenerationProvider,
    ProviderResult,
    TaskKind,
    TextRequest,
    WorkspaceRequest,
)


@dataclass
class FakeCommandResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class FakeRunner:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            response = response(command, kwargs)
        return response


@pytest.fixture
def text_request(tmp_path):
    return TextRequest(TaskKind.QUERY, "answer this", tmp_path, 5)


@pytest.fixture
def fake_runner():
    def make(*responses):
        return FakeRunner(*responses)

    return make


def _write_codex_output(text):
    def write(command, _kwargs):
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(text, encoding="utf-8")
        return FakeCommandResult()

    return write


def _codex_provider(runner, **kwargs):
    import providers

    return providers.CodexProvider(
        runner=runner,
        task_models={task: f"model-{task.value}" for task in TaskKind},
        **kwargs,
    )


def _chatgpt_login():
    return FakeCommandResult(stdout="Logged in using ChatGPT\n")


def _run(awaitable):
    return asyncio.run(awaitable)


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
import importlib.util
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
scripts_spec = importlib.util.find_spec("scripts")
assert direct is package
assert scripts.providers is package
assert scripts.__spec__ is scripts_spec
assert scripts_spec.submodule_search_locations is not None
assert str(root / "scripts") in scripts_spec.submodule_search_locations
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


def test_codex_accepts_chatgpt_login(fake_runner, text_request):
    runner = fake_runner(_chatgpt_login(), _write_codex_output("codex answer"))

    result = _run(_codex_provider(runner).generate_text(text_request))

    assert result.outcome == "success"
    assert result.text == "codex answer"
    assert runner.calls[0][0] == ["codex", "login", "status"]


def test_codex_rejects_api_key_login(fake_runner, text_request):
    runner = fake_runner(FakeCommandResult(stdout="Logged in using an API key"))

    result = _run(_codex_provider(runner).generate_text(text_request))

    assert result.outcome == "auth_failed"
    assert len(runner.calls) == 1


def test_codex_rejects_unknown_login_output(fake_runner, text_request):
    runner = fake_runner(FakeCommandResult(stdout="Login status: active"))

    result = _run(_codex_provider(runner).generate_text(text_request))

    assert result.outcome == "auth_failed"
    assert "unsupported" in result.reason
    assert len(runner.calls) == 1


def test_codex_child_env_strips_api_keys(monkeypatch, fake_runner, text_request):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("OPENAI_ORG_ID", "org-secret")
    monkeypatch.setenv("OPENAI_CUSTOM_SECRET", "custom-secret")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "endpoint-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("CLAUDE_API_KEY", "claude-secret")
    runner = fake_runner(_chatgpt_login(), _write_codex_output("answer"))

    result = _run(_codex_provider(runner).generate_text(text_request))

    assert result.outcome == "success"
    for _command, kwargs in runner.calls:
        child_env = kwargs["env"]
        assert "OPENAI_API_KEY" not in child_env
        assert not any(name.startswith("OPENAI_") for name in child_env)
        assert not any(name.startswith("AZURE_OPENAI_") for name in child_env)
        assert "ANTHROPIC_API_KEY" not in child_env
        assert "CLAUDE_API_KEY" not in child_env
        assert child_env["AI_MEMORY_INTERNAL_JOB"] == "1"


def test_codex_text_command_is_ephemeral_read_only_and_noninteractive(
    fake_runner, text_request
):
    runner = fake_runner(_chatgpt_login(), _write_codex_output("answer"))

    _run(_codex_provider(runner).generate_text(text_request))

    command, kwargs = runner.calls[1]
    assert command[:2] == ["codex", "exec"]
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert command[command.index("--model") + 1] == "model-query"
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--ask-for-approval") + 1] == "never"
    assert command[command.index("--cd") + 1] == str(text_request.cwd)
    assert command[-1] == "-"
    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert kwargs["stdin"] == text_request.prompt
    assert kwargs["start_new_session"] is True
    assert kwargs["terminate_process_group_on_timeout"] is True


def test_codex_workspace_command_writes_only_in_stage(fake_runner, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    request = WorkspaceRequest(
        TaskKind.COMPILE,
        "edit staged files",
        stage,
        5,
        allowed_paths=("knowledge/index.md",),
    )
    runner = fake_runner(_chatgpt_login(), _write_codex_output("saved"))

    result = _run(_codex_provider(runner).edit_workspace(request))

    assert result.outcome == "success"
    command, kwargs = runner.calls[1]
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert command[command.index("--cd") + 1] == str(stage)
    assert "--skip-git-repo-check" in command
    output_path = Path(command[command.index("--output-last-message") + 1])
    assert output_path.parent == stage
    assert kwargs["cwd"] == stage
    assert not output_path.exists()


def test_codex_timeout_terminates_process_group(fake_runner, text_request):
    runner = fake_runner(_chatgpt_login(), TimeoutError("timed out"))

    result = _run(_codex_provider(runner).generate_text(text_request))

    assert result.outcome == "timeout"
    assert runner.calls[1][1]["start_new_session"] is True
    assert runner.calls[1][1]["terminate_process_group_on_timeout"] is True


def test_codex_error_reason_is_bounded_and_redacts_credentials(
    fake_runner, text_request
):
    secret = "abc"
    runner = fake_runner(
        _chatgpt_login(),
        FakeCommandResult(returncode=7, stderr=f"failure {secret} " + "x" * 1000),
    )

    result = _run(
        _codex_provider(runner, env={**os.environ, "ANTHROPIC_API_KEY": secret})
        .generate_text(text_request)
    )

    assert result.outcome == "error"
    assert secret not in result.reason
    assert len(result.reason) <= 500


@pytest.mark.parametrize(
    ("stderr", "outcome"),
    [
        ("usage limit exceeded", "capacity"),
        ("authentication failed", "auth_failed"),
        ("unexpected command failure", "error"),
    ],
)
def test_codex_classifies_nonzero_command_failures(
    fake_runner, text_request, stderr, outcome
):
    runner = fake_runner(_chatgpt_login(), FakeCommandResult(1, stderr=stderr))

    result = _run(_codex_provider(runner).generate_text(text_request))

    assert result.outcome == outcome


def test_codex_empty_output_is_invalid(fake_runner, text_request):
    runner = fake_runner(_chatgpt_login(), _write_codex_output(""))

    result = _run(_codex_provider(runner).generate_text(text_request))

    assert result.outcome == "invalid_output"


class FakeClaudeQuery:
    def __init__(self, messages=(), error=None):
        self.messages = messages
        self.error = error
        self.calls = []

    def __call__(self, *, prompt, options):
        self.calls.append((prompt, options))

        async def messages():
            if self.error is not None:
                raise self.error
            for message in self.messages:
                yield message

        return messages()


def _assistant_text(text):
    return SimpleNamespace(content=[SimpleNamespace(text=text)], error=None)


def _claude_provider(query, **kwargs):
    import providers

    return providers.ClaudeProvider(query_fn=query, **kwargs)


def test_claude_text_success_uses_subscription_safe_options(monkeypatch, text_request):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    query = FakeClaudeQuery([_assistant_text("claude answer")])

    result = _run(_claude_provider(query).generate_text(text_request))

    assert result.outcome == "success"
    assert result.text == "claude answer"
    prompt, options = query.calls[0]
    assert prompt == text_request.prompt
    assert options.allowed_tools == []
    assert options.model == "claude-sonnet-5"
    assert options.setting_sources == []
    assert options.cwd == str(text_request.cwd)
    assert options.env["AI_MEMORY_INTERNAL_JOB"] == "1"
    assert "OPENAI_API_KEY" not in options.env
    assert "ANTHROPIC_API_KEY" not in options.env


def test_claude_workspace_uses_existing_staged_tool_set(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    request = WorkspaceRequest(TaskKind.COMPILE, "edit", stage, 5)
    query = FakeClaudeQuery([_assistant_text("saved")])

    result = _run(_claude_provider(query).edit_workspace(request))

    assert result.outcome == "success"
    options = query.calls[0][1]
    assert options.allowed_tools == ["Read", "Write", "Edit", "Glob", "Grep"]
    assert options.permission_mode == "acceptEdits"
    assert options.cwd == str(stage)


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (RuntimeError("authentication failed: log in"), "auth_failed"),
        (RuntimeError("usage limit exceeded"), "capacity"),
        (TimeoutError("too slow"), "timeout"),
        (RuntimeError("unexpected SDK crash"), "error"),
    ],
)
def test_claude_classifies_exception_paths(text_request, error, outcome):
    result = _run(
        _claude_provider(FakeClaudeQuery(error=error)).generate_text(text_request)
    )

    assert result.outcome == outcome


def test_claude_empty_output_is_invalid(text_request):
    result = _run(_claude_provider(FakeClaudeQuery()).generate_text(text_request))

    assert result.outcome == "invalid_output"


@pytest.mark.parametrize(
    ("sdk_error", "outcome"),
    [("authentication_failed", "auth_failed"), ("rate_limit", "capacity")],
)
def test_claude_classifies_sdk_error_codes(text_request, sdk_error, outcome):
    message = SimpleNamespace(content=[], error=sdk_error)

    result = _run(
        _claude_provider(FakeClaudeQuery([message])).generate_text(text_request)
    )

    assert result.outcome == outcome


class FakeProvider:
    def __init__(self, result):
        self.result = result
        self.text_requests = []
        self.workspace_requests = []

    async def generate_text(self, request):
        self.text_requests.append(request)
        return self.result

    async def edit_workspace(self, request):
        self.workspace_requests.append(request)
        return self.result


def _provider_result(provider, outcome, reason=None, text=""):
    return ProviderResult(provider, "model", TaskKind.QUERY, outcome, text=text, reason=reason)


def success_result(provider, text):
    return _provider_result(provider, "success", text=text)


def capacity_result(reason):
    return _provider_result("codex", "capacity", reason=reason)


def test_router_records_reason_and_falls_back_to_claude(text_request):
    import providers

    codex = FakeProvider(result=capacity_result("usage limit exceeded"))
    claude = FakeProvider(result=success_result("claude", "saved"))

    result = _run(providers.ProviderRouter(codex, claude).generate_text(text_request))

    assert result.provider == "claude"
    assert result.fallback_reason == "codex:capacity:usage limit exceeded"
    assert [attempt.provider for attempt in result.attempts] == ["codex", "claude"]


def test_router_successful_codex_never_calls_claude(text_request):
    import providers

    codex = FakeProvider(result=success_result("codex", "done"))
    claude = FakeProvider(result=success_result("claude", "unused"))

    result = _run(providers.ProviderRouter(codex, claude).generate_text(text_request))

    assert result.provider == "codex"
    assert result.fallback_reason is None
    assert len(claude.text_requests) == 0


def test_router_calls_attempt_callback_for_both_attempts(text_request):
    import providers

    seen = []
    codex = FakeProvider(_provider_result("codex", "error", "command failed"))
    claude = FakeProvider(success_result("claude", "done"))

    result = _run(
        providers.ProviderRouter(codex, claude, attempt_callback=seen.append).generate_text(
            text_request
        )
    )

    assert seen == list(result.attempts)


def test_router_returns_failed_result_when_both_providers_fail(text_request):
    import providers

    codex = FakeProvider(_provider_result("codex", "timeout", "timed out"))
    claude = FakeProvider(_provider_result("claude", "error", "SDK failed"))

    result = _run(providers.ProviderRouter(codex, claude).generate_text(text_request))

    assert result.provider == "claude"
    assert result.outcome == "error"
    assert len(result.attempts) == 2


def test_router_accepts_failed_validation_attempt_with_fresh_stage(tmp_path):
    import providers

    stale_stage = tmp_path / "stale"
    fresh_stage = tmp_path / "fresh"
    stale_stage.mkdir()
    fresh_stage.mkdir()
    failed = _provider_result("codex", "invalid_output", "staged validation failed")
    failed = ProviderResult(
        failed.provider,
        failed.model,
        TaskKind.COMPILE,
        failed.outcome,
        reason=failed.reason,
    )
    codex = FakeProvider(success_result("codex", "must not run"))
    claude = FakeProvider(
        ProviderResult("claude", "model", TaskKind.COMPILE, "success", text="saved")
    )
    stale_request = WorkspaceRequest(TaskKind.COMPILE, "edit", stale_stage, 5)
    fresh_request = WorkspaceRequest(TaskKind.COMPILE, "edit", fresh_stage, 5)

    result = _run(
        providers.ProviderRouter(
            codex, claude, fallback_workspace_factory=lambda _request: fresh_request
        ).edit_workspace(stale_request, codex_attempt=failed)
    )

    assert [attempt.provider for attempt in result.attempts] == ["codex", "claude"]
    assert codex.workspace_requests == []
    assert claude.workspace_requests[0].cwd == fresh_stage
