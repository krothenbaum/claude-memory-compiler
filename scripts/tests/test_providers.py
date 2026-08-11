"""Tests for provider-neutral generation contracts."""

import asyncio
import importlib
import inspect
import json
import os
import signal
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


def _answer_schema(tmp_path):
    schema_path = tmp_path / "answer.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        ),
        encoding="utf-8",
    )
    return schema_path


def _unresolved_schema(tmp_path):
    schema_path = tmp_path / "unresolved.schema.json"
    schema_path.write_text(
        json.dumps({"$ref": "private-schema-reference-must-not-leak"}),
        encoding="utf-8",
    )
    return schema_path


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


def test_codex_preflight_classifies_capacity(fake_runner, text_request):
    runner = fake_runner(FakeCommandResult(1, stderr="usage limit exceeded"))

    result = _run(_codex_provider(runner).generate_text(text_request))

    assert result.outcome == "capacity"
    assert len(runner.calls) == 1


def test_codex_preflight_classifies_unknown_nonzero_as_error(
    fake_runner, text_request
):
    runner = fake_runner(FakeCommandResult(23, stderr="unexpected status failure"))

    result = _run(_codex_provider(runner).generate_text(text_request))

    assert result.outcome == "error"
    assert len(runner.calls) == 1


def test_codex_unsupported_login_reason_is_bounded_and_redacted(
    fake_runner, text_request
):
    secret = "tiny"
    runner = fake_runner(
        FakeCommandResult(stdout=f"unknown login {secret} " + "x" * 1000)
    )

    result = _run(
        _codex_provider(runner, env={**os.environ, "OPENAI_API_KEY": secret})
        .generate_text(text_request)
    )

    assert result.outcome == "auth_failed"
    assert result.reason.startswith("unsupported Codex login:")
    assert secret not in result.reason
    assert len(result.reason) <= 500


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
    assert command[:4] == ["codex", "--ask-for-approval", "never", "exec"]
    assert "--ask-for-approval" not in command[4:]
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert command[command.index("--model") + 1] == "model-query"
    assert command[command.index("--sandbox") + 1] == "read-only"
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


class FakeBoundaryProcess:
    def __init__(self, communicate_error):
        self.pid = 4321
        self.returncode = None
        self.communicate_error = communicate_error
        self.wait_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    async def communicate(self, _stdin):
        raise self.communicate_error

    async def wait(self):
        self.wait_calls += 1
        self.returncode = 0
        return 0

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1


def _run_boundary_runner(runner, tmp_path):
    return _run(
        runner(
            ["fake-command"],
            cwd=tmp_path,
            env={},
            stdin="prompt",
            timeout_seconds=1,
            start_new_session=True,
            terminate_process_group_on_timeout=True,
        )
    )


def test_command_runner_force_kills_group_after_timeout_even_if_leader_exits(
    monkeypatch, tmp_path
):
    import providers

    process = FakeBoundaryProcess(TimeoutError("timed out"))
    signals = []

    async def fake_process_factory(*_command, **_kwargs):
        return process

    monkeypatch.setattr(providers.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    runner = providers.AsyncCommandRunner(
        process_factory=fake_process_factory, platform_name="Linux", grace_seconds=0
    )

    with pytest.raises(TimeoutError):
        _run_boundary_runner(runner, tmp_path)

    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]


def test_command_runner_cancellation_cleans_group_and_reraises(monkeypatch, tmp_path):
    import providers

    process = FakeBoundaryProcess(asyncio.CancelledError())
    signals = []

    async def fake_process_factory(*_command, **_kwargs):
        return process

    monkeypatch.setattr(providers.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    runner = providers.AsyncCommandRunner(
        process_factory=fake_process_factory, platform_name="Linux", grace_seconds=0
    )

    with pytest.raises(asyncio.CancelledError):
        _run_boundary_runner(runner, tmp_path)

    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]


def test_command_runner_communicate_error_cleans_group_and_reraises(
    monkeypatch, tmp_path
):
    import providers

    process = FakeBoundaryProcess(RuntimeError("pipe failed"))
    signals = []

    async def fake_process_factory(*_command, **_kwargs):
        return process

    monkeypatch.setattr(providers.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    runner = providers.AsyncCommandRunner(
        process_factory=fake_process_factory, platform_name="Linux", grace_seconds=0
    )

    with pytest.raises(RuntimeError, match="pipe failed"):
        _run_boundary_runner(runner, tmp_path)

    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]


def test_command_runner_uses_windows_tree_termination(tmp_path):
    import providers

    process = FakeBoundaryProcess(TimeoutError("timed out"))
    opened = {}
    tree_calls = []

    async def fake_process_factory(*_command, **kwargs):
        opened.update(kwargs)
        return process

    async def fake_tree_terminator(pid, *, force):
        tree_calls.append((pid, force))

    runner = providers.AsyncCommandRunner(
        process_factory=fake_process_factory,
        platform_name="Windows",
        windows_tree_terminator=fake_tree_terminator,
        grace_seconds=0,
    )

    with pytest.raises(TimeoutError):
        _run_boundary_runner(runner, tmp_path)

    assert opened["start_new_session"] is False
    assert "creationflags" in opened
    assert tree_calls == [(process.pid, False), (process.pid, True)]


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


def test_codex_command_propagates_output_schema(fake_runner, tmp_path):
    schema_path = _answer_schema(tmp_path)
    request = TextRequest(
        TaskKind.QUERY, "structured answer", tmp_path, 5, schema_path
    )
    runner = fake_runner(
        _chatgpt_login(), _write_codex_output('{"answer": "yes"}')
    )

    result = _run(_codex_provider(runner).generate_text(request))

    command = runner.calls[1][0]
    assert result.outcome == "success"
    assert command[command.index("--output-schema") + 1] == str(schema_path)


def test_codex_rejects_parseable_schema_invalid_output(fake_runner, tmp_path):
    schema_path = _answer_schema(tmp_path)
    prompt = "prompt-content-must-stay-private"
    request = TextRequest(TaskKind.QUERY, prompt, tmp_path, 5, schema_path)
    runner = fake_runner(
        _chatgpt_login(), _write_codex_output('{"wrong": "field"}')
    )

    result = _run(_codex_provider(runner).generate_text(request))

    assert result.outcome == "invalid_output"
    assert "answer" not in result.reason
    assert prompt not in result.reason
    assert len(result.reason) <= 500


def test_codex_malformed_schema_still_preflights_without_leaking_content(
    fake_runner, tmp_path
):
    secret_content = "schema-content-must-stay-private"
    schema_path = tmp_path / "malformed.schema.json"
    schema_path.write_text(f"not-json {secret_content}", encoding="utf-8")
    request = TextRequest(TaskKind.QUERY, "private prompt", tmp_path, 5, schema_path)
    runner = fake_runner(_chatgpt_login())

    result = _run(_codex_provider(runner).generate_text(request))

    assert runner.calls[0][0] == ["codex", "login", "status"]
    assert len(runner.calls) == 1
    assert result.outcome == "error"
    assert secret_content not in result.reason
    assert request.prompt not in result.reason
    assert len(result.reason) <= 500


def test_codex_bounds_unresolved_schema_failure(fake_runner, tmp_path):
    schema_path = _unresolved_schema(tmp_path)
    request = TextRequest(TaskKind.QUERY, "private prompt", tmp_path, 5, schema_path)
    runner = fake_runner(_chatgpt_login(), _write_codex_output("{}"))

    result = _run(_codex_provider(runner).generate_text(request))

    assert result.outcome == "invalid_output"
    assert "private-schema-reference" not in result.reason
    assert request.prompt not in result.reason
    assert len(result.reason) <= 500


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


def _serialized_sdk_command(options):
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    transport = SubprocessCLITransport(prompt="", options=options)
    transport._cli_path = "claude"
    return transport._build_command()


def test_claude_text_success_uses_subscription_safe_options(monkeypatch, text_request):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    query = FakeClaudeQuery([_assistant_text("claude answer")])

    result = _run(_claude_provider(query).generate_text(text_request))

    assert result.outcome == "success"
    assert result.text == "claude answer"
    prompt, options = query.calls[0]
    assert prompt == text_request.prompt
    assert options.tools == []
    assert options.allowed_tools == []
    assert options.model == "claude-sonnet-5"
    assert options.setting_sources == []
    assert options.cwd == str(text_request.cwd)
    assert options.env["AI_MEMORY_INTERNAL_JOB"] == "1"
    assert "OPENAI_API_KEY" not in options.env
    assert "ANTHROPIC_API_KEY" not in options.env
    command = _serialized_sdk_command(options)
    assert command[command.index("--tools") + 1] == ""
    assert "--allowedTools" not in command


def test_claude_workspace_uses_existing_staged_tool_set(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    request = WorkspaceRequest(TaskKind.COMPILE, "edit", stage, 5)
    query = FakeClaudeQuery([_assistant_text("saved")])

    result = _run(_claude_provider(query).edit_workspace(request))

    assert result.outcome == "success"
    options = query.calls[0][1]
    expected_tools = ["Read", "Write", "Edit", "Glob", "Grep"]
    assert options.tools == expected_tools
    assert options.allowed_tools == ["Read", "Write", "Edit", "Glob", "Grep"]
    assert options.permission_mode == "acceptEdits"
    assert options.cwd == str(stage)
    command = _serialized_sdk_command(options)
    assert command[command.index("--tools") + 1] == ",".join(expected_tools)
    assert command[command.index("--allowedTools") + 1] == ",".join(expected_tools)


class FakeOpenedProcess:
    def __init__(self):
        self.pid = 4242
        self.returncode = 0
        self.stdin = None
        self.stdout = None
        self.stderr = None


def test_claude_transport_passes_exact_scrubbed_environment(
    monkeypatch, tmp_path
):
    import providers
    from claude_agent_sdk import ClaudeAgentOptions

    secrets = {
        "OPENAI_API_KEY": "openai-secret",
        "OPENAI_ORGANIZATION": "openai-org-secret",
        "AZURE_OPENAI_API_KEY": "azure-secret",
        "AZURE_OPENAI_ENDPOINT": "azure-endpoint-secret",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "CLAUDE_API_KEY": "claude-secret",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    clean_env = providers.subscription_child_env()
    opened = {}

    async def fake_open_process(command, **kwargs):
        opened["command"] = list(command)
        opened.update(kwargs)
        return FakeOpenedProcess()

    options = ClaudeAgentOptions(
        cli_path="/fake/claude",
        cwd=str(tmp_path),
        tools=[],
        allowed_tools=[],
        env=clean_env,
        setting_sources=[],
    )
    transport = providers.ExactEnvironmentClaudeTransport(
        prompt="private prompt",
        options=options,
        env=clean_env,
        process_opener=fake_open_process,
    )

    _run(transport.connect())

    final_env = opened["env"]
    assert final_env["AI_MEMORY_INTERNAL_JOB"] == "1"
    assert not any(name.startswith("OPENAI_") for name in final_env)
    assert not any(name.startswith("AZURE_OPENAI_") for name in final_env)
    assert "ANTHROPIC_API_KEY" not in final_env
    assert "CLAUDE_API_KEY" not in final_env
    assert not set(secrets.values()) & set(final_env.values())


def test_claude_transport_force_cleans_descendants_after_leader_exit(tmp_path):
    import providers
    from claude_agent_sdk import ClaudeAgentOptions

    process = FakeOpenedProcess()
    tree_calls = []

    async def fake_open_process(_command, **_kwargs):
        return process

    async def fake_tree_terminator(pid, *, force):
        tree_calls.append((pid, force))

    options = ClaudeAgentOptions(
        cli_path="/fake/claude",
        cwd=str(tmp_path),
        tools=[],
        env=providers.subscription_child_env(),
    )
    transport = providers.ExactEnvironmentClaudeTransport(
        prompt="private prompt",
        options=options,
        env=providers.subscription_child_env(),
        process_opener=fake_open_process,
        tree_terminator=fake_tree_terminator,
    )

    async def connect_and_close():
        await transport.connect()
        await transport.close()

    _run(connect_and_close())

    assert tree_calls == [(process.pid, True)]


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


def test_claude_normalizes_options_factory_failure(text_request):
    prompt = "prompt-content-must-stay-private"
    request = TextRequest(
        text_request.task, prompt, text_request.cwd, text_request.timeout_seconds
    )

    def broken_options_factory(**_kwargs):
        raise RuntimeError(f"options setup failed {prompt}")

    result = _run(
        _claude_provider(
            FakeClaudeQuery(), options_factory=broken_options_factory
        ).generate_text(request)
    )

    assert result.provider == "claude"
    assert result.outcome == "error"
    assert prompt not in result.reason
    assert len(result.reason) <= 500


def test_claude_normalizes_sdk_import_failure(text_request):
    import providers

    def broken_sdk_loader():
        raise ImportError("claude SDK unavailable")

    result = _run(
        providers.ClaudeProvider(sdk_loader=broken_sdk_loader).generate_text(
            text_request
        )
    )

    assert result.provider == "claude"
    assert result.outcome == "error"
    assert "unavailable" in result.reason


def test_claude_normalizes_transport_factory_failure(text_request):
    def broken_transport_factory(**_kwargs):
        raise RuntimeError("transport setup failed")

    result = _run(
        _claude_provider(
            FakeClaudeQuery(), transport_factory=broken_transport_factory
        ).generate_text(text_request)
    )

    assert result.provider == "claude"
    assert result.outcome == "error"
    assert "transport setup failed" in result.reason


@pytest.mark.parametrize("error_source", ["error", "errors", "result"])
def test_claude_redacts_message_level_errors(
    tmp_path, error_source
):
    prompt = "prompt-content-must-stay-private"
    secret = "anthropic-secret-must-stay-private"
    leaked = f"provider failed: {prompt} {secret} {'detail ' * 200}"
    attributes = {"content": [], "error": None, "is_error": False}
    if error_source == "error":
        attributes["error"] = leaked
    else:
        attributes["is_error"] = True
        attributes["errors"] = [leaked] if error_source == "errors" else []
        attributes["result"] = leaked if error_source == "result" else None
    request = TextRequest(TaskKind.QUERY, prompt, tmp_path, 5)
    query = FakeClaudeQuery([SimpleNamespace(**attributes)])

    result = _run(
        _claude_provider(
            query, env={"HOME": str(tmp_path), "ANTHROPIC_API_KEY": secret}
        ).generate_text(request)
    )

    assert result.provider == "claude"
    assert result.outcome == "error"
    assert prompt not in result.reason
    assert secret not in result.reason
    assert len(result.reason) <= 500


def test_claude_empty_output_is_invalid(text_request):
    result = _run(_claude_provider(FakeClaudeQuery()).generate_text(text_request))

    assert result.outcome == "invalid_output"


def test_claude_options_propagate_output_schema(tmp_path):
    schema_path = _answer_schema(tmp_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    request = TextRequest(
        TaskKind.QUERY, "structured answer", tmp_path, 5, schema_path
    )
    query = FakeClaudeQuery([_assistant_text('{"answer": "yes"}')])

    result = _run(_claude_provider(query).generate_text(request))

    assert result.outcome == "success"
    assert query.calls[0][1].output_format == {
        "type": "json_schema",
        "schema": schema,
    }


def test_claude_rejects_parseable_schema_invalid_output(tmp_path):
    schema_path = _answer_schema(tmp_path)
    prompt = "prompt-content-must-stay-private"
    request = TextRequest(TaskKind.QUERY, prompt, tmp_path, 5, schema_path)
    query = FakeClaudeQuery([_assistant_text('{"wrong": "field"}')])

    result = _run(_claude_provider(query).generate_text(request))

    assert result.outcome == "invalid_output"
    assert "answer" not in result.reason
    assert prompt not in result.reason
    assert len(result.reason) <= 500


def test_claude_uses_final_structured_output(tmp_path):
    schema_path = _answer_schema(tmp_path)
    request = TextRequest(TaskKind.QUERY, "structured answer", tmp_path, 5, schema_path)
    message = SimpleNamespace(
        content=[], error=None, structured_output={"answer": "final"}
    )

    result = _run(
        _claude_provider(FakeClaudeQuery([message])).generate_text(request)
    )

    assert result.outcome == "success"
    assert json.loads(result.text) == {"answer": "final"}


def test_claude_bounds_unresolved_schema_failure(tmp_path):
    schema_path = _unresolved_schema(tmp_path)
    request = TextRequest(TaskKind.QUERY, "private prompt", tmp_path, 5, schema_path)
    query = FakeClaudeQuery([_assistant_text("{}")])

    result = _run(_claude_provider(query).generate_text(request))

    assert result.outcome == "invalid_output"
    assert "private-schema-reference" not in result.reason
    assert request.prompt not in result.reason
    assert len(result.reason) <= 500


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
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    async def edit_workspace(self, request):
        self.workspace_requests.append(request)
        if isinstance(self.result, BaseException):
            raise self.result
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


def test_router_normalizes_unexpected_codex_exception_and_falls_back(text_request):
    import providers

    seen = []
    codex = FakeProvider(RuntimeError("codex adapter crashed"))
    claude = FakeProvider(success_result("claude", "done"))

    result = _run(
        providers.ProviderRouter(
            codex, claude, attempt_callback=seen.append
        ).generate_text(text_request)
    )

    assert result.outcome == "success"
    assert [attempt.outcome for attempt in seen] == ["error", "success"]
    assert result.fallback_reason == "codex:error:codex adapter crashed"


def test_router_normalizes_unexpected_claude_exception(text_request):
    import providers

    seen = []
    codex = FakeProvider(_provider_result("codex", "capacity", "limit"))
    claude = FakeProvider(RuntimeError("claude adapter crashed"))

    result = _run(
        providers.ProviderRouter(
            codex, claude, attempt_callback=seen.append
        ).generate_text(text_request)
    )

    assert result.provider == "claude"
    assert result.outcome == "error"
    assert [attempt.provider for attempt in seen] == ["codex", "claude"]


@pytest.mark.parametrize("provider_name", ["codex", "claude"])
@pytest.mark.parametrize(
    "secret",
    ["provider-only-secret", "provider-only-secret-" + "x" * 600],
    ids=["short-secret", "long-secret"],
)
def test_router_redacts_provider_owned_secrets_from_unexpected_exceptions(
    text_request, provider_name, secret
):
    import providers

    prompt = "router prompt must stay private"
    request = TextRequest(
        text_request.task,
        prompt,
        text_request.cwd,
        text_request.timeout_seconds,
    )
    crashing = FakeProvider(RuntimeError(f"adapter crashed: {secret} {prompt}"))
    crashing._source_env = {"ANTHROPIC_API_KEY": secret}
    if provider_name == "codex":
        codex = crashing
        claude = FakeProvider(success_result("claude", "done"))
        attempt_index = 0
    else:
        codex = FakeProvider(_provider_result("codex", "capacity", "limit"))
        claude = crashing
        attempt_index = 1

    result = _run(providers.ProviderRouter(codex, claude).generate_text(request))

    reason = result.attempts[attempt_index].reason
    assert prompt not in reason
    assert secret[:100] not in reason
    assert len(reason) <= 500


@pytest.mark.parametrize("callback_kind", ["sync", "async"])
def test_router_fails_closed_when_attempt_callback_fails(
    text_request, callback_kind
):
    import providers

    codex = FakeProvider(_provider_result("codex", "capacity", "limit"))
    claude = FakeProvider(success_result("claude", "must not run"))
    seen = []

    def sync_callback(attempt):
        seen.append(attempt)
        raise RuntimeError("attempt persistence failed")

    async def async_callback(attempt):
        seen.append(attempt)
        raise RuntimeError("attempt persistence failed")

    callback = sync_callback if callback_kind == "sync" else async_callback

    with pytest.raises(RuntimeError, match="attempt persistence failed"):
        _run(
            providers.ProviderRouter(
                codex, claude, attempt_callback=callback
            ).generate_text(text_request)
        )

    assert len(seen) == 1
    assert seen[0].provider == "codex"
    assert claude.text_requests == []


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


@pytest.mark.parametrize(
    "invalid_kind",
    ["wrong_type", "same_stage", "changed_task", "changed_schema", "changed_allowlist"],
)
def test_router_rejects_invalid_fallback_workspace(
    tmp_path, invalid_kind
):
    import providers

    stale_stage = tmp_path / "stale"
    fresh_stage = tmp_path / "fresh"
    stale_stage.mkdir()
    fresh_stage.mkdir()
    schema_path = _answer_schema(tmp_path)
    other_schema = tmp_path / "other.schema.json"
    other_schema.write_text(schema_path.read_text(encoding="utf-8"), encoding="utf-8")
    original = WorkspaceRequest(
        TaskKind.COMPILE,
        "private prompt",
        stale_stage,
        5,
        schema_path,
        ("knowledge/index.md",),
    )
    candidates = {
        "wrong_type": TextRequest(TaskKind.COMPILE, "prompt", fresh_stage, 5),
        "same_stage": WorkspaceRequest(
            TaskKind.COMPILE,
            "prompt",
            stale_stage,
            5,
            schema_path,
            original.allowed_paths,
        ),
        "changed_task": WorkspaceRequest(
            TaskKind.QUERY,
            "prompt",
            fresh_stage,
            5,
            schema_path,
            original.allowed_paths,
        ),
        "changed_schema": WorkspaceRequest(
            TaskKind.COMPILE,
            "prompt",
            fresh_stage,
            5,
            other_schema,
            original.allowed_paths,
        ),
        "changed_allowlist": WorkspaceRequest(
            TaskKind.COMPILE,
            "prompt",
            fresh_stage,
            5,
            schema_path,
            ("knowledge/log.md",),
        ),
    }
    failed = ProviderResult(
        "codex",
        "model",
        TaskKind.COMPILE,
        "invalid_output",
        reason="staged validation failed",
    )
    codex = FakeProvider(success_result("codex", "unused"))
    claude = FakeProvider(
        ProviderResult("claude", "model", TaskKind.COMPILE, "success", text="saved")
    )
    seen = []

    result = _run(
        providers.ProviderRouter(
            codex,
            claude,
            attempt_callback=seen.append,
            fallback_workspace_factory=lambda _request: candidates[invalid_kind],
        ).edit_workspace(original, codex_attempt=failed)
    )

    assert result.provider == "claude"
    assert result.outcome == "error"
    assert [attempt.provider for attempt in result.attempts] == ["codex", "claude"]
    assert seen == list(result.attempts)
    assert claude.workspace_requests == []
    assert original.prompt not in result.reason


def _failed_workspace_fallback(tmp_path):
    stage = tmp_path / "stale"
    stage.mkdir()
    request = WorkspaceRequest(
        TaskKind.COMPILE, "private prompt must stay private", stage, 5
    )
    codex_attempt = ProviderResult(
        "codex",
        "model",
        request.task,
        "invalid_output",
        reason="staged validation failed",
    )
    return request, codex_attempt


def test_router_normalizes_sync_fallback_factory_exception(tmp_path):
    import providers

    request, failed = _failed_workspace_fallback(tmp_path)
    secret = "fallback-secret-must-stay-private"
    query = FakeClaudeQuery([_assistant_text("saved")])
    claude = providers.ClaudeProvider(
        query_fn=query,
        env={"HOME": str(tmp_path), "ANTHROPIC_API_KEY": secret},
    )
    seen = []

    def broken_factory(_request):
        raise RuntimeError(
            f"factory failed: {request.prompt} {secret} {'detail ' * 200}"
        )

    result = _run(
        providers.ProviderRouter(
            FakeProvider(success_result("codex", "unused")),
            claude,
            attempt_callback=seen.append,
            fallback_workspace_factory=broken_factory,
        ).edit_workspace(request, codex_attempt=failed)
    )

    assert result.provider == "claude"
    assert result.outcome == "error"
    assert result.fallback_reason == "codex:invalid_output:staged validation failed"
    assert seen == list(result.attempts)
    assert [attempt.provider for attempt in result.attempts] == ["codex", "claude"]
    assert request.prompt not in result.reason
    assert secret not in result.reason
    assert len(result.reason) <= 500
    assert query.calls == []


def test_router_normalizes_async_fallback_factory_exception(tmp_path):
    import providers

    request, failed = _failed_workspace_fallback(tmp_path)
    claude = FakeProvider(
        ProviderResult("claude", "model", request.task, "success", text="saved")
    )
    seen = []

    async def broken_factory(_request):
        raise RuntimeError("async fallback factory failed")

    result = _run(
        providers.ProviderRouter(
            FakeProvider(success_result("codex", "unused")),
            claude,
            attempt_callback=seen.append,
            fallback_workspace_factory=broken_factory,
        ).edit_workspace(request, codex_attempt=failed)
    )

    assert result.provider == "claude"
    assert result.outcome == "error"
    assert seen == list(result.attempts)
    assert [attempt.provider for attempt in result.attempts] == ["codex", "claude"]
    assert "async fallback factory failed" in result.reason
    assert claude.workspace_requests == []


def test_router_reraises_fallback_factory_cancellation(tmp_path):
    import providers

    request, failed = _failed_workspace_fallback(tmp_path)
    claude = FakeProvider(
        ProviderResult("claude", "model", request.task, "success", text="saved")
    )
    seen = []

    async def cancelled_factory(_request):
        raise asyncio.CancelledError

    router = providers.ProviderRouter(
        FakeProvider(success_result("codex", "unused")),
        claude,
        attempt_callback=seen.append,
        fallback_workspace_factory=cancelled_factory,
    )

    with pytest.raises(asyncio.CancelledError):
        _run(router.edit_workspace(request, codex_attempt=failed))

    assert seen == [failed]
    assert claude.workspace_requests == []
