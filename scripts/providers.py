"""Provider-neutral request and result contracts for memory generation."""

import asyncio
import inspect
import json
import os
import platform
import re
import subprocess
import signal
import sys
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol

import jsonschema
from referencing import Registry


if __name__ == "providers":
    provider_module = sys.modules[__name__]
    scripts_package = sys.modules.get("scripts")
    if scripts_package is None:
        scripts_path = str(Path(__file__).resolve().parent)
        scripts_spec = ModuleSpec("scripts", loader=None, is_package=True)
        scripts_spec.submodule_search_locations = [scripts_path]
        scripts_package = ModuleType("scripts")
        scripts_package.__spec__ = scripts_spec
        scripts_package.__path__ = scripts_spec.submodule_search_locations
        scripts_package.__package__ = "scripts"
        sys.modules["scripts"] = scripts_package
    scripts_package.providers = provider_module
    sys.modules.setdefault("scripts.providers", provider_module)
elif __name__ == "scripts.providers":
    sys.modules.setdefault("providers", sys.modules[__name__])


class TaskKind(StrEnum):
    """Supported memory-generation task categories."""

    EXTRACT = "extract"
    COMPILE = "compile"
    QUERY = "query"
    CONNECTIONS = "connections"
    FILE_ANSWER = "file_answer"
    SEMANTIC_LINT = "semantic_lint"


@dataclass(frozen=True)
class TextRequest:
    """A provider-neutral text generation request."""

    task: TaskKind
    prompt: str
    cwd: Path
    timeout_seconds: int
    output_schema: Path | None = None


@dataclass(frozen=True)
class WorkspaceRequest(TextRequest):
    """A generation request that may edit an allowlist of relative paths."""

    allowed_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderResult:
    """Normalized result returned by any generation provider."""

    provider: Literal["codex", "claude"]
    model: str
    task: TaskKind
    outcome: Literal[
        "success", "auth_failed", "capacity", "timeout", "invalid_output", "error"
    ]
    text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_ms: int = 0
    reason: str | None = None


class GenerationProvider(Protocol):
    """Interface implemented by provider-specific generation adapters."""

    async def generate_text(self, request: TextRequest) -> ProviderResult: ...

    async def edit_workspace(self, request: WorkspaceRequest) -> ProviderResult: ...


@dataclass(frozen=True)
class CommandResult:
    """Captured result from a provider CLI command."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    output_truncated: bool = False


class AsyncCommandRunner:
    """Run a subprocess in its own session and kill that session on timeout."""

    def __init__(
        self,
        *,
        process_factory: Callable[..., Awaitable[Any]] | None = None,
        platform_name: str | None = None,
        windows_tree_terminator: Callable[..., Awaitable[None]] | None = None,
        grace_seconds: float = 2,
    ) -> None:
        self._process_factory = process_factory or asyncio.create_subprocess_exec
        self._platform_name = platform_name or platform.system()
        self._windows_tree_terminator = (
            windows_tree_terminator or self._taskkill_windows_tree
        )
        self._grace_seconds = grace_seconds

    async def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: str,
        timeout_seconds: int,
        start_new_session: bool,
        terminate_process_group_on_timeout: bool,
        max_output_bytes: int | None = None,
    ) -> CommandResult:
        windows = self._platform_name == "Windows"
        process = await self._process_factory(
            *command,
            cwd=cwd,
            env=dict(env),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=start_new_session and not windows,
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if windows and start_new_session
                else 0
            ),
        )
        try:
            if max_output_bytes is None:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(stdin.encode()), timeout=timeout_seconds
                )
                output_truncated = False
            else:
                stdout, stderr, output_truncated = await asyncio.wait_for(
                    self._communicate_capped(
                        process, stdin.encode(), max_output_bytes
                    ),
                    timeout=timeout_seconds,
                )
        except BaseException:
            cleanup = asyncio.create_task(
                self._cleanup_process_tree(
                    process,
                    terminate_process_group=(
                        terminate_process_group_on_timeout and start_new_session
                    ),
                )
            )
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
            raise
        return CommandResult(
            process.returncode,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
            output_truncated,
        )

    @staticmethod
    async def _communicate_capped(
        process: Any, stdin: bytes, max_output_bytes: int
    ) -> tuple[bytes, bytes, bool]:
        """Drain both pipes while retaining at most one shared byte budget."""
        if max_output_bytes < 0:
            raise ValueError("max_output_bytes must be non-negative")
        remaining = [max_output_bytes]
        output_truncated = [False]

        async def read_stream(stream: Any) -> bytes:
            retained: list[bytes] = []
            while chunk := await stream.read(64 * 1024):
                keep = min(len(chunk), remaining[0])
                if keep:
                    retained.append(chunk[:keep])
                    remaining[0] -= keep
                if keep < len(chunk):
                    output_truncated[0] = True
            return b"".join(retained)

        async def write_stdin() -> None:
            if process.stdin is None:
                return
            try:
                process.stdin.write(stdin)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                process.stdin.close()
                wait_closed = getattr(process.stdin, "wait_closed", None)
                if wait_closed is not None:
                    try:
                        await wait_closed()
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        _, stdout, stderr, _ = await asyncio.gather(
            write_stdin(),
            read_stream(process.stdout),
            read_stream(process.stderr),
            process.wait(),
        )
        return stdout, stderr, output_truncated[0]

    async def _cleanup_process_tree(
        self, process: Any, *, terminate_process_group: bool
    ) -> None:
        await self._signal_process_tree(
            process, force=False, process_group=terminate_process_group
        )
        try:
            await asyncio.wait_for(
                asyncio.shield(process.wait()), timeout=self._grace_seconds
            )
        except (TimeoutError, OSError):
            pass
        finally:
            await self._signal_process_tree(
                process, force=True, process_group=terminate_process_group
            )
        try:
            await asyncio.shield(process.wait())
        except (OSError, ProcessLookupError):
            pass

    async def _signal_process_tree(
        self, process: Any, *, force: bool, process_group: bool
    ) -> None:
        try:
            if process_group and self._platform_name == "Windows":
                await self._windows_tree_terminator(process.pid, force=force)
            elif process_group:
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif force:
                process.kill()
            else:
                process.terminate()
        except (OSError, ProcessLookupError, PermissionError):
            if not process_group:
                return
            try:
                process.kill() if force else process.terminate()
            except (OSError, ProcessLookupError, PermissionError):
                pass

    @staticmethod
    async def _taskkill_windows_tree(pid: int, *, force: bool) -> None:
        command = ["taskkill", "/PID", str(pid), "/T"]
        if force:
            command.append("/F")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.communicate()


_ENV_NAMES = {
    "CODEX_HOME",
    "HOME",
    "PATH",
    "LANG",
    "LANGUAGE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "TERM",
    "COLORTERM",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "USERPROFILE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
}
_SECRET_NAMES = {"ANTHROPIC_API_KEY", "CLAUDE_API_KEY"}
_CAPACITY_MARKERS = (
    "capacity",
    "quota",
    "rate limit",
    "rate_limit",
    "too many requests",
    "usage limit",
    "limit exceeded",
)
_AUTH_MARKERS = (
    "auth failed",
    "authentication failed",
    "authentication_failed",
    "not authenticated",
    "not logged in",
    "unauthorized",
    "api key",
    "invalid api key",
    "login required",
    "log in",
)
_MAX_REASON_LENGTH = 500
_CODEX_LOGIN_STATUS_MARKER = "Logged in using"
_CODEX_CHATGPT_LOGIN_STATUS = "Logged in using ChatGPT"
_CODEX_LOGIN_STATUS_MAX_OUTPUT_BYTES = 64 * 1024
_CODEX_VERSION_MAX_OUTPUT_BYTES = 64 * 1024
_CODEX_PREFLIGHT_TIMEOUT_SECONDS = 5
_MINIMUM_CODEX_VERSION = (0, 146, 1)
_CODEX_VERSION_PATTERN = re.compile(
    r"(?:codex-cli|codex) ([0-9]+)\.([0-9]+)\.([0-9]+)\Z"
)


def subscription_child_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the minimal child environment used by subscription-backed CLIs."""
    source = os.environ if source is None else source
    child = {
        name: value
        for name, value in source.items()
        if (name in _ENV_NAMES or name.startswith("LC_"))
        and not name.startswith(("OPENAI_", "AZURE_OPENAI_"))
        and name not in _SECRET_NAMES
    }
    child["AI_MEMORY_INTERNAL_JOB"] = "1"
    return child


def _safe_reason(
    message: str, *source_envs: Mapping[str, str]
) -> str:
    safe = " ".join(message.strip().split())
    secret_values = {
        value
        for source_env in source_envs
        for name, value in source_env.items()
        if value
        and (
            name in _SECRET_NAMES
            or name.startswith("OPENAI_")
            or name.startswith("AZURE_OPENAI_")
        )
    }
    redactable_values = secret_values | {
        value
        for source_env in source_envs
        for value in source_env.values()
        if len(value) >= 4
    }
    for value in sorted(redactable_values, key=len, reverse=True):
        safe = safe.replace(value, "[REDACTED]")
    return safe[:_MAX_REASON_LENGTH] or "provider failed"


def _failure_outcome(message: str) -> Literal["auth_failed", "capacity", "error"]:
    lowered = message.lower()
    if any(marker in lowered for marker in _CAPACITY_MARKERS):
        return "capacity"
    if any(marker in lowered for marker in _AUTH_MARKERS):
        return "auth_failed"
    return "error"


def _default_task_models(env: Mapping[str, str]) -> dict[TaskKind, str]:
    luna = env.get("AI_MEMORY_CODEX_LUNA_MODEL", "gpt-5.6-luna")
    terra = env.get("AI_MEMORY_CODEX_TERRA_MODEL", "gpt-5.6-terra")
    return {
        TaskKind.EXTRACT: luna,
        TaskKind.SEMANTIC_LINT: luna,
        TaskKind.COMPILE: terra,
        TaskKind.QUERY: terra,
        TaskKind.CONNECTIONS: terra,
        TaskKind.FILE_ANSWER: terra,
    }


class CodexProvider:
    """Subscription-only Codex CLI adapter."""

    def __init__(
        self,
        *,
        runner: Any | None = None,
        task_models: Mapping[TaskKind, str] | None = None,
        env: Mapping[str, str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._runner = runner or AsyncCommandRunner()
        self._source_env = dict(os.environ if env is None else env)
        self._task_models = dict(task_models or _default_task_models(self._source_env))
        self._monotonic = monotonic

    async def generate_text(self, request: TextRequest) -> ProviderResult:
        return await self._generate(request, workspace=False)

    async def edit_workspace(self, request: WorkspaceRequest) -> ProviderResult:
        return await self._generate(request, workspace=True)

    async def _run_command(
        self,
        command: Sequence[str],
        request: TextRequest,
        stdin: str = "",
        *,
        max_output_bytes: int | None = None,
        timeout_seconds: int | None = None,
    ) -> CommandResult:
        runner = self._runner.run if hasattr(self._runner, "run") else self._runner
        kwargs = {
            "cwd": request.cwd,
            "env": subscription_child_env(self._source_env),
            "stdin": stdin,
            "timeout_seconds": (
                request.timeout_seconds if timeout_seconds is None else timeout_seconds
            ),
            "start_new_session": True,
            "terminate_process_group_on_timeout": True,
        }
        if max_output_bytes is not None:
            kwargs["max_output_bytes"] = max_output_bytes
        result = await runner(command, **kwargs)
        return CommandResult(
            result.returncode,
            result.stdout,
            result.stderr,
            getattr(result, "output_truncated", False),
        )

    def _result(
        self,
        request: TextRequest,
        outcome: Literal[
            "success", "auth_failed", "capacity", "timeout", "invalid_output", "error"
        ],
        started: float,
        *,
        text: str = "",
        reason: str | None = None,
    ) -> ProviderResult:
        return ProviderResult(
            provider="codex",
            model=self._task_models[request.task],
            task=request.task,
            outcome=outcome,
            text=text,
            elapsed_ms=max(0, round((self._monotonic() - started) * 1000)),
            reason=reason,
        )

    async def _preflight(
        self, request: TextRequest, started: float, deadline: float
    ) -> ProviderResult | None:
        try:
            try:
                version = await self._run_command(
                    ["codex", "--version"],
                    request,
                    max_output_bytes=_CODEX_VERSION_MAX_OUTPUT_BYTES,
                    timeout_seconds=self._remaining_timeout(
                        deadline, _CODEX_PREFLIGHT_TIMEOUT_SECONDS
                    ),
                )
                if self._monotonic() >= deadline:
                    raise TimeoutError("Codex CLI version check timed out")
            except TimeoutError:
                return self._result(
                    request,
                    "timeout",
                    started,
                    reason="Codex CLI version check timed out",
                )
            if version.output_truncated:
                return self._result(
                    request,
                    "error",
                    started,
                    reason="Codex CLI version output too large",
                )
            if version.returncode != 0:
                return self._result(
                    request,
                    "error",
                    started,
                    reason="Codex CLI version check failed",
                )
            version_lines = [
                line.strip()
                for stream in (version.stdout, version.stderr)
                for line in stream.splitlines()
                if line.strip()
            ]
            match = (
                _CODEX_VERSION_PATTERN.fullmatch(version_lines[0])
                if len(version_lines) == 1
                else None
            )
            if match is None:
                return self._result(
                    request,
                    "error",
                    started,
                    reason="invalid Codex CLI version output",
                )
            parsed_version = tuple(int(part) for part in match.groups())
            if parsed_version < _MINIMUM_CODEX_VERSION:
                return self._result(
                    request,
                    "error",
                    started,
                    reason="unsupported Codex CLI version",
                )
            try:
                status = await self._run_command(
                    ["codex", "login", "status"],
                    request,
                    max_output_bytes=_CODEX_LOGIN_STATUS_MAX_OUTPUT_BYTES,
                    timeout_seconds=self._remaining_timeout(
                        deadline, _CODEX_PREFLIGHT_TIMEOUT_SECONDS
                    ),
                )
                if self._monotonic() >= deadline:
                    raise TimeoutError("codex login status timed out")
            except TimeoutError:
                return self._result(
                    request,
                    "timeout",
                    started,
                    reason="codex login status timed out",
                )
        except FileNotFoundError:
            return self._result(
                request, "auth_failed", started, reason="codex CLI unavailable"
            )
        except Exception as exc:
            return self._result(
                request,
                "error",
                started,
                reason=_safe_reason(str(exc), self._source_env),
            )
        if status.output_truncated:
            return self._result(
                request,
                "error",
                started,
                reason="login status output too large",
            )
        combined = "\n".join((status.stdout, status.stderr)).strip()
        status_lines = {
            line.strip()
            for stream in (status.stdout, status.stderr)
            for line in stream.splitlines()
            if line.strip()
        }
        login_status_lines = {
            line
            for line in status_lines
            if _CODEX_LOGIN_STATUS_MARKER in line
        }
        if (
            status.returncode == 0
            and _CODEX_CHATGPT_LOGIN_STATUS in status_lines
            and login_status_lines == {_CODEX_CHATGPT_LOGIN_STATUS}
        ):
            return None
        if status.returncode == 0:
            return self._result(
                request, "auth_failed", started, reason="unsupported login type"
            )
        return self._result(
            request,
            _failure_outcome(combined),
            started,
            reason="login status failed",
        )

    async def _generate(
        self, request: TextRequest, *, workspace: bool
    ) -> ProviderResult:
        started = self._monotonic()
        deadline = started + request.timeout_seconds
        preflight_failure = await self._preflight(request, started, deadline)
        if preflight_failure is not None:
            return preflight_failure
        output_schema, schema_error = _load_output_schema(request.output_schema)
        if schema_error is not None:
            return self._result(request, "error", started, reason=schema_error)

        stage = request.cwd.resolve()
        temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        if workspace:
            output_path = stage / f".ai-memory-last-message-{uuid.uuid4().hex}.txt"
        else:
            temporary_directory = tempfile.TemporaryDirectory(prefix="ai-memory-codex-")
            output_path = Path(temporary_directory.name) / "last-message.txt"

        command = [
            "codex",
            "--ask-for-approval",
            "never",
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--model",
            self._task_models[request.task],
            "--sandbox",
            "workspace-write" if workspace else "read-only",
            "--cd",
            str(stage),
            "--output-last-message",
            str(output_path),
        ]
        if request.output_schema is not None:
            command.extend(["--output-schema", str(request.output_schema.resolve())])
        command.append("-")

        try:
            completed = await self._run_command(
                command,
                request,
                request.prompt,
                timeout_seconds=self._remaining_timeout(deadline),
            )
            if self._monotonic() >= deadline:
                raise TimeoutError("codex execution timed out")
            combined = "\n".join((completed.stderr, completed.stdout)).strip()
            if completed.returncode != 0:
                outcome = _failure_outcome(combined)
                return self._result(
                    request,
                    outcome,
                    started,
                    reason=_safe_reason(combined, self._source_env),
                )
            try:
                text = output_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as exc:
                return self._result(
                    request,
                    "invalid_output",
                    started,
                    reason=_safe_reason(str(exc), self._source_env),
                )
            invalid_reason = _invalid_text_reason(text, output_schema)
            if invalid_reason is not None:
                return self._result(
                    request, "invalid_output", started, reason=invalid_reason
                )
            return self._result(request, "success", started, text=text)
        except TimeoutError as exc:
            return self._result(
                request,
                "timeout",
                started,
                reason=_safe_reason(str(exc) or "codex execution timed out", self._source_env),
            )
        except Exception as exc:
            return self._result(
                request,
                "error",
                started,
                reason=_safe_reason(str(exc), self._source_env),
            )
        finally:
            output_path.unlink(missing_ok=True)
            if temporary_directory is not None:
                temporary_directory.cleanup()

    def _remaining_timeout(
        self, deadline: float, maximum: float | None = None
    ) -> float:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise TimeoutError("Codex attempt timed out")
        return remaining if maximum is None else min(maximum, remaining)

def _load_output_schema(
    output_schema: Path | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if output_schema is None:
        return None, None
    try:
        schema = json.loads(output_schema.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "output schema could not be read"
    if not isinstance(schema, dict):
        return None, "output schema is invalid"
    try:
        jsonschema.validators.validator_for(schema).check_schema(schema)
    except jsonschema.exceptions.SchemaError:
        return None, "output schema is invalid"
    return schema, None


def _invalid_text_reason(
    text: str, output_schema: dict[str, Any] | None
) -> str | None:
    if not text:
        return "provider returned empty output"
    if output_schema is not None:
        try:
            instance = json.loads(text)
        except json.JSONDecodeError:
            return "provider returned invalid structured output"
        try:
            validator_class = jsonschema.validators.validator_for(output_schema)
            validator_class(output_schema, registry=Registry()).validate(instance)
        except jsonschema.exceptions.ValidationError:
            return "provider output did not match requested schema"
        except Exception:
            return "provider output could not be validated"
    return None


def ExactEnvironmentClaudeTransport(
    *,
    prompt: str,
    options: Any,
    env: Mapping[str, str],
    process_opener: Callable[..., Awaitable[Any]] | None = None,
    tree_terminator: Callable[..., Awaitable[None]] | None = None,
) -> Any:
    """Create an SDK transport whose process receives only the supplied environment."""
    import anyio
    from anyio.streams.text import TextReceiveStream, TextSendStream
    from claude_agent_sdk._errors import (
        CLIConnectionError,
        CLINotFoundError,
    )
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )
    from claude_agent_sdk._version import __version__ as sdk_version

    open_process = process_opener or anyio.open_process
    exact_env = dict(env)

    async def default_tree_terminator(pid: int, *, force: bool) -> None:
        if os.name == "nt":
            await AsyncCommandRunner._taskkill_windows_tree(pid, force=force)
            return
        try:
            os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)
        except (OSError, ProcessLookupError, PermissionError):
            pass

    terminate_tree = tree_terminator or default_tree_terminator

    class _ExactEnvironmentTransport(SubprocessCLITransport):
        async def connect(self) -> None:
            if self._process:
                return
            if self._cli_path is None:
                self._cli_path = await anyio.to_thread.run_sync(self._find_cli)

            command = self._build_command()
            process_env = {
                **exact_env,
                "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
                "CLAUDE_AGENT_SDK_VERSION": sdk_version,
            }
            if self._options.enable_file_checkpointing:
                process_env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] = "true"
            if self._cwd:
                process_env["PWD"] = self._cwd

            should_pipe_stderr = (
                self._options.stderr is not None
                or "debug-to-stderr" in self._options.extra_args
            )
            try:
                self._process = await open_process(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE if should_pipe_stderr else None,
                    cwd=self._cwd,
                    env=process_env,
                    user=self._options.user,
                    start_new_session=os.name != "nt",
                    creationflags=(
                        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        if os.name == "nt"
                        else 0
                    ),
                )
                if self._process.stdout:
                    self._stdout_stream = TextReceiveStream(self._process.stdout)
                if should_pipe_stderr and self._process.stderr:
                    self._stderr_stream = TextReceiveStream(self._process.stderr)
                    self._stderr_task_group = anyio.create_task_group()
                    await self._stderr_task_group.__aenter__()
                    self._stderr_task_group.start_soon(self._handle_stderr)
                if self._process.stdin:
                    self._stdin_stream = TextSendStream(self._process.stdin)
                self._ready = True
            except FileNotFoundError as exc:
                if self._cwd and not Path(self._cwd).exists():
                    error = CLIConnectionError(
                        f"Working directory does not exist: {self._cwd}"
                    )
                else:
                    error = CLINotFoundError(
                        f"Claude Code not found at: {self._cli_path}"
                    )
                self._exit_error = error
                raise error from exc
            except Exception as exc:
                error = CLIConnectionError("Failed to start Claude Code")
                self._exit_error = error
                raise error from exc

        async def close(self) -> None:
            pid = self._process.pid if self._process is not None else None
            try:
                await super().close()
            finally:
                if pid is not None:
                    try:
                        await terminate_tree(pid, force=True)
                    except (OSError, ProcessLookupError, PermissionError):
                        pass

    return _ExactEnvironmentTransport(prompt=prompt, options=options)


def _load_claude_sdk() -> tuple[Any, Callable[..., Any]]:
    from claude_agent_sdk import ClaudeAgentOptions, query

    return ClaudeAgentOptions, query


def _safe_provider_reason(
    message: str,
    source_env: Mapping[str, str],
    prompt: str,
    *additional_source_envs: Mapping[str, str],
) -> str:
    without_prompt = message.replace(prompt, "[REDACTED]") if prompt else message
    return _safe_reason(without_prompt, source_env, *additional_source_envs)


def _safe_router_reason(
    message: str, provider: GenerationProvider, prompt: str
) -> str:
    provider_env = getattr(provider, "_source_env", None)
    if isinstance(provider_env, Mapping):
        return _safe_provider_reason(
            message, os.environ, prompt, provider_env
        )
    return _safe_provider_reason(message, os.environ, prompt)


class ClaudeProvider:
    """Subscription-only Claude Agent SDK adapter."""

    def __init__(
        self,
        *,
        query_fn: Callable[..., Any] | None = None,
        options_factory: Callable[..., Any] | None = None,
        transport_factory: Callable[..., Any] | None = None,
        sdk_loader: Callable[[], tuple[Any, Callable[..., Any]]] | None = None,
        model: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._query_fn = query_fn
        self._options_factory = options_factory
        self._transport_factory = transport_factory
        self._sdk_loader = sdk_loader or _load_claude_sdk
        self._source_env = dict(os.environ if env is None else env)
        self._model = model or self._source_env.get(
            "AI_MEMORY_CLAUDE_MODEL", "claude-sonnet-5"
        )

    async def generate_text(self, request: TextRequest) -> ProviderResult:
        return await self._generate(request, workspace=False)

    async def edit_workspace(self, request: WorkspaceRequest) -> ProviderResult:
        return await self._generate(request, workspace=True)

    def _result(
        self,
        request: TextRequest,
        outcome: Literal[
            "success", "auth_failed", "capacity", "timeout", "invalid_output", "error"
        ],
        started: float,
        *,
        text: str = "",
        reason: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> ProviderResult:
        return ProviderResult(
            provider="claude",
            model=self._model,
            task=request.task,
            outcome=outcome,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_ms=max(0, round((time.monotonic() - started) * 1000)),
            reason=reason,
        )

    async def _generate(
        self, request: TextRequest, *, workspace: bool
    ) -> ProviderResult:
        started = time.monotonic()
        parts: list[str] = []
        errors: list[str] = []
        fallback_result_text = ""
        structured_result_text = ""
        input_tokens = None
        output_tokens = None
        try:
            ClaudeAgentOptions, sdk_query = self._sdk_loader()
            output_schema, schema_error = _load_output_schema(request.output_schema)
            if schema_error is not None:
                return self._result(request, "error", started, reason=schema_error)
            query_fn = self._query_fn or sdk_query
            options_factory = self._options_factory or ClaudeAgentOptions
            tools = ["Read", "Write", "Edit", "Glob", "Grep"] if workspace else []
            child_env = subscription_child_env(self._source_env)
            options = options_factory(
                cwd=str(request.cwd.resolve()),
                tools=tools,
                allowed_tools=tools,
                permission_mode="acceptEdits" if workspace else "dontAsk",
                max_turns=80 if workspace else 20,
                model=self._model,
                env=child_env,
                setting_sources=[],
                output_format=(
                    {"type": "json_schema", "schema": output_schema}
                    if output_schema is not None
                    else None
                ),
            )
            transport_factory = self._transport_factory
            if transport_factory is None and self._query_fn is None:
                transport_factory = ExactEnvironmentClaudeTransport
            transport = (
                transport_factory(
                    prompt=request.prompt,
                    options=options,
                    env=child_env,
                )
                if transport_factory is not None
                else None
            )
            query_arguments = {"prompt": request.prompt, "options": options}
            if transport is not None:
                query_arguments["transport"] = transport
            async with asyncio.timeout(request.timeout_seconds):
                async for message in query_fn(**query_arguments):
                    error = getattr(message, "error", None)
                    if error:
                        errors.append(str(error))
                    for block in getattr(message, "content", ()):
                        text = getattr(block, "text", None)
                        if isinstance(text, str):
                            parts.append(text)
                    if getattr(message, "is_error", False):
                        message_errors = getattr(message, "errors", None) or ()
                        errors.extend(str(item) for item in message_errors)
                        result_text = getattr(message, "result", None)
                        if result_text:
                            errors.append(str(result_text))
                    elif not parts and getattr(message, "result", None):
                        fallback_result_text = str(message.result)
                    structured_output = getattr(message, "structured_output", None)
                    if structured_output is not None:
                        structured_result_text = json.dumps(structured_output)
                    usage = getattr(message, "usage", None) or {}
                    input_tokens = usage.get("input_tokens", input_tokens)
                    output_tokens = usage.get("output_tokens", output_tokens)
        except TimeoutError as exc:
            return self._result(
                request,
                "timeout",
                started,
                reason=_safe_reason(str(exc) or "Claude execution timed out", self._source_env),
            )
        except Exception as exc:
            reason = _safe_provider_reason(
                str(exc), self._source_env, request.prompt
            )
            return self._result(
                request, _failure_outcome(reason), started, reason=reason
            )

        if errors:
            reason = _safe_provider_reason(
                "; ".join(errors), self._source_env, request.prompt
            )
            return self._result(
                request, _failure_outcome(reason), started, reason=reason
            )
        text = (
            structured_result_text.strip()
            or "".join(parts).strip()
            or fallback_result_text.strip()
        )
        invalid_reason = _invalid_text_reason(text, output_schema)
        if invalid_reason is not None:
            return self._result(
                request, "invalid_output", started, reason=invalid_reason
            )
        return self._result(
            request,
            "success",
            started,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


@dataclass(frozen=True)
class RoutedResult:
    """Final routed result plus every provider attempt for one logical job."""

    provider: Literal["codex", "claude"]
    model: str
    task: TaskKind
    outcome: Literal[
        "success", "auth_failed", "capacity", "timeout", "invalid_output", "error"
    ]
    text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_ms: int = 0
    reason: str | None = None
    attempts: tuple[ProviderResult, ...] = ()
    fallback_reason: str | None = None

    @classmethod
    def from_result(
        cls,
        result: ProviderResult,
        attempts: Sequence[ProviderResult],
        fallback_reason: str | None,
    ) -> "RoutedResult":
        return cls(
            provider=result.provider,
            model=result.model,
            task=result.task,
            outcome=result.outcome,
            text=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            elapsed_ms=result.elapsed_ms,
            reason=result.reason,
            attempts=tuple(attempts),
            fallback_reason=fallback_reason,
        )


AttemptCallback = Callable[[ProviderResult], Awaitable[None] | None]


class ProviderRouter:
    """Try Codex once, then Claude, preserving both attempt results."""

    def __init__(
        self,
        codex: GenerationProvider,
        claude: GenerationProvider,
        *,
        attempt_callback: AttemptCallback | None = None,
        fallback_workspace_factory: (
            Callable[[WorkspaceRequest], WorkspaceRequest | Awaitable[WorkspaceRequest]]
            | None
        ) = None,
    ) -> None:
        self._codex = codex
        self._claude = claude
        self._attempt_callback = attempt_callback
        self._fallback_workspace_factory = fallback_workspace_factory

    async def _record(self, attempt: ProviderResult) -> None:
        """Persist an attempt, failing closed if mandatory persistence fails."""
        if self._attempt_callback is None:
            return
        callback_result = self._attempt_callback(attempt)
        if inspect.isawaitable(callback_result):
            await callback_result

    async def _attempt(
        self,
        provider_name: Literal["codex", "claude"],
        provider: GenerationProvider,
        operation: Literal["generate_text", "edit_workspace"],
        request: TextRequest,
    ) -> ProviderResult:
        started = time.monotonic()
        try:
            method = getattr(provider, operation)
            return await method(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            model = getattr(provider, "_model", None)
            if provider_name == "codex":
                task_models = getattr(provider, "_task_models", {})
                model = task_models.get(request.task, model)
            return ProviderResult(
                provider=provider_name,
                model=model or "unknown",
                task=request.task,
                outcome="error",
                elapsed_ms=max(0, round((time.monotonic() - started) * 1000)),
                reason=_safe_router_reason(str(exc), provider, request.prompt),
            )

    async def generate_text(self, request: TextRequest) -> RoutedResult:
        codex_attempt = await self._attempt(
            "codex", self._codex, "generate_text", request
        )
        await self._record(codex_attempt)
        if codex_attempt.outcome == "success":
            return RoutedResult.from_result(codex_attempt, [codex_attempt], None)
        fallback_reason = _fallback_reason(codex_attempt)
        claude_attempt = await self._attempt(
            "claude", self._claude, "generate_text", request
        )
        await self._record(claude_attempt)
        return RoutedResult.from_result(
            claude_attempt, [codex_attempt, claude_attempt], fallback_reason
        )

    async def edit_workspace(
        self,
        request: WorkspaceRequest,
        *,
        codex_attempt: ProviderResult | None = None,
    ) -> RoutedResult:
        attempts: list[ProviderResult] = []
        if codex_attempt is None:
            codex_attempt = await self._attempt(
                "codex", self._codex, "edit_workspace", request
            )
        else:
            if codex_attempt.provider != "codex" or codex_attempt.outcome == "success":
                raise ValueError("codex_attempt must be an explicit failed Codex attempt")
            if codex_attempt.task != request.task:
                raise ValueError("codex_attempt and request must have the same task")
        attempts.append(codex_attempt)
        await self._record(codex_attempt)
        if codex_attempt.outcome == "success":
            return RoutedResult.from_result(codex_attempt, attempts, None)

        fallback_reason = _fallback_reason(codex_attempt)
        fallback_request = request
        if self._fallback_workspace_factory is not None:
            try:
                made_request = self._fallback_workspace_factory(request)
                fallback_request = (
                    await made_request
                    if inspect.isawaitable(made_request)
                    else made_request
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                claude_attempt = ProviderResult(
                    provider="claude",
                    model=getattr(self._claude, "_model", None) or "unknown",
                    task=request.task,
                    outcome="error",
                    reason=_safe_router_reason(
                        str(exc), self._claude, request.prompt
                    ),
                )
                attempts.append(claude_attempt)
                await self._record(claude_attempt)
                return RoutedResult.from_result(
                    claude_attempt, attempts, fallback_reason
                )
            invalid_reason = _invalid_fallback_workspace_reason(
                request, fallback_request
            )
            if invalid_reason is not None:
                claude_attempt = ProviderResult(
                    provider="claude",
                    model=getattr(self._claude, "_model", None) or "unknown",
                    task=request.task,
                    outcome="error",
                    reason=invalid_reason,
                )
                attempts.append(claude_attempt)
                await self._record(claude_attempt)
                return RoutedResult.from_result(
                    claude_attempt, attempts, fallback_reason
                )
        claude_attempt = await self._attempt(
            "claude", self._claude, "edit_workspace", fallback_request
        )
        attempts.append(claude_attempt)
        await self._record(claude_attempt)
        return RoutedResult.from_result(claude_attempt, attempts, fallback_reason)


def _fallback_reason(attempt: ProviderResult) -> str:
    reason = attempt.reason or attempt.outcome
    return f"codex:{attempt.outcome}:{reason}"


def _invalid_fallback_workspace_reason(
    original: WorkspaceRequest, candidate: object
) -> str | None:
    if not isinstance(candidate, WorkspaceRequest):
        return "fallback workspace factory returned an invalid request type"
    if candidate.cwd.resolve() == original.cwd.resolve():
        return "fallback workspace factory did not create a fresh stage"
    if candidate.task != original.task:
        return "fallback workspace request changed the task"
    if _resolved_path(candidate.output_schema) != _resolved_path(
        original.output_schema
    ):
        return "fallback workspace request changed the output schema"
    if candidate.allowed_paths != original.allowed_paths:
        return "fallback workspace request changed the allowed paths"
    return None


def _resolved_path(path: Path | None) -> Path | None:
    return path.resolve() if path is not None else None
