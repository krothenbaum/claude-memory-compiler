"""Provider-neutral request and result contracts for memory generation."""

import asyncio
import inspect
import json
import os
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


class AsyncCommandRunner:
    """Run a subprocess in its own session and kill that session on timeout."""

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
    ) -> CommandResult:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=dict(env),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=start_new_session,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin.encode()), timeout=timeout_seconds
            )
        except TimeoutError:
            if terminate_process_group_on_timeout and start_new_session:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            else:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                if terminate_process_group_on_timeout and start_new_session:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                else:
                    process.kill()
                await process.wait()
            raise
        return CommandResult(
            process.returncode,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )


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


def _safe_reason(message: str, source_env: Mapping[str, str]) -> str:
    safe = " ".join(message.strip().split())
    secret_values = {
        value
        for name, value in source_env.items()
        if value
        and (
            name in _SECRET_NAMES
            or name.startswith("OPENAI_")
            or name.startswith("AZURE_OPENAI_")
        )
    }
    redactable_values = secret_values | {
        value for value in source_env.values() if len(value) >= 4
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
    ) -> None:
        self._runner = runner or AsyncCommandRunner()
        self._source_env = dict(os.environ if env is None else env)
        self._task_models = dict(task_models or _default_task_models(self._source_env))

    async def generate_text(self, request: TextRequest) -> ProviderResult:
        return await self._generate(request, workspace=False)

    async def edit_workspace(self, request: WorkspaceRequest) -> ProviderResult:
        return await self._generate(request, workspace=True)

    async def _run_command(
        self, command: Sequence[str], request: TextRequest, stdin: str = ""
    ) -> CommandResult:
        runner = self._runner.run if hasattr(self._runner, "run") else self._runner
        result = await runner(
            command,
            cwd=request.cwd,
            env=subscription_child_env(self._source_env),
            stdin=stdin,
            timeout_seconds=request.timeout_seconds,
            start_new_session=True,
            terminate_process_group_on_timeout=True,
        )
        return CommandResult(result.returncode, result.stdout, result.stderr)

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
            elapsed_ms=max(0, round((time.monotonic() - started) * 1000)),
            reason=reason,
        )

    async def _preflight(
        self, request: TextRequest, started: float
    ) -> ProviderResult | None:
        try:
            status = await self._run_command(["codex", "login", "status"], request)
        except TimeoutError as exc:
            return self._result(
                request,
                "timeout",
                started,
                reason=_safe_reason(str(exc) or "codex login status timed out", self._source_env),
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
        combined = "\n".join((status.stdout, status.stderr)).strip()
        if status.returncode == 0 and "Logged in using ChatGPT" in status.stdout:
            return None
        if status.returncode == 0:
            reason = _safe_reason(
                f"unsupported Codex login: {combined or 'unknown status'}",
                self._source_env,
            )
            return self._result(request, "auth_failed", started, reason=reason)
        reason = _safe_reason(combined or "codex login status failed", self._source_env)
        return self._result(
            request, _failure_outcome(combined), started, reason=reason
        )

    async def _generate(
        self, request: TextRequest, *, workspace: bool
    ) -> ProviderResult:
        started = time.monotonic()
        preflight_failure = await self._preflight(request, started)
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
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--model",
            self._task_models[request.task],
            "--sandbox",
            "workspace-write" if workspace else "read-only",
            "--ask-for-approval",
            "never",
            "--cd",
            str(stage),
            "--output-last-message",
            str(output_path),
        ]
        if request.output_schema is not None:
            command.extend(["--output-schema", str(request.output_schema.resolve())])
        if workspace and not _is_git_worktree(stage):
            command.append("--skip-git-repo-check")
        command.append("-")

        try:
            completed = await self._run_command(command, request, request.prompt)
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


def _is_git_worktree(path: Path) -> bool:
    return any((candidate / ".git").exists() for candidate in (path, *path.parents))


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


class ClaudeProvider:
    """Subscription-only Claude Agent SDK adapter."""

    def __init__(
        self,
        *,
        query_fn: Callable[..., Any] | None = None,
        options_factory: Callable[..., Any] | None = None,
        model: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._query_fn = query_fn
        self._options_factory = options_factory
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
        from claude_agent_sdk import ClaudeAgentOptions, query

        started = time.monotonic()
        output_schema, schema_error = _load_output_schema(request.output_schema)
        if schema_error is not None:
            return self._result(request, "error", started, reason=schema_error)
        query_fn = self._query_fn or query
        options_factory = self._options_factory or ClaudeAgentOptions
        options = options_factory(
            cwd=str(request.cwd.resolve()),
            allowed_tools=(
                ["Read", "Write", "Edit", "Glob", "Grep"] if workspace else []
            ),
            permission_mode="acceptEdits" if workspace else "dontAsk",
            max_turns=80 if workspace else 20,
            model=self._model,
            env=subscription_child_env(self._source_env),
            setting_sources=[],
            output_format=(
                {"type": "json_schema", "schema": output_schema}
                if output_schema is not None
                else None
            ),
        )
        parts: list[str] = []
        errors: list[str] = []
        fallback_result_text = ""
        structured_result_text = ""
        input_tokens = None
        output_tokens = None
        try:
            async with asyncio.timeout(request.timeout_seconds):
                async for message in query_fn(prompt=request.prompt, options=options):
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
            reason = _safe_reason(str(exc), self._source_env)
            return self._result(
                request, _failure_outcome(reason), started, reason=reason
            )

        if errors:
            reason = _safe_reason("; ".join(errors), self._source_env)
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
        if self._attempt_callback is None:
            return
        callback_result = self._attempt_callback(attempt)
        if inspect.isawaitable(callback_result):
            await callback_result

    async def generate_text(self, request: TextRequest) -> RoutedResult:
        codex_attempt = await self._codex.generate_text(request)
        await self._record(codex_attempt)
        if codex_attempt.outcome == "success":
            return RoutedResult.from_result(codex_attempt, [codex_attempt], None)
        fallback_reason = _fallback_reason(codex_attempt)
        claude_attempt = await self._claude.generate_text(request)
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
            codex_attempt = await self._codex.edit_workspace(request)
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
            made_request = self._fallback_workspace_factory(request)
            fallback_request = (
                await made_request if inspect.isawaitable(made_request) else made_request
            )
        claude_attempt = await self._claude.edit_workspace(fallback_request)
        attempts.append(claude_attempt)
        await self._record(claude_attempt)
        return RoutedResult.from_result(claude_attempt, attempts, fallback_reason)


def _fallback_reason(attempt: ProviderResult) -> str:
    reason = attempt.reason or attempt.outcome
    return f"codex:{attempt.outcome}:{reason}"
