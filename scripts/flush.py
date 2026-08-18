"""
Memory flush agent - extracts important knowledge from conversation context.

Spawned by session-end.py or pre-compact.py as a background process. Reads
pre-extracted conversation context from a .md file, uses the Claude Agent SDK
to decide what's worth saving, and appends the result to today's daily log.

Usage:
    uv run python flush.py <context_file.md> <session_id>
"""

from __future__ import annotations

import os

import asyncio
import ctypes
from dataclasses import dataclass
import hashlib
import json
import logging
import re
import secrets
import shlex
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Literal

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "daily"
SCRIPTS_DIR = ROOT / "scripts"
STATE_FILE = SCRIPTS_DIR / "last-flush.json"
LOG_FILE = SCRIPTS_DIR / "flush.log"

sys.path.insert(0, str(SCRIPTS_DIR))
from utils import (  # noqa: E402
    ExclusiveFileLock,
    append_daily_entry,
    open_secure_log_stream,
    read_text_with_baseline,
)
from config import load_config  # noqa: E402
from providers import (  # noqa: E402
    ClaudeProvider,
    CodexProvider,
    ProviderRouter,
    TaskKind,
    TextRequest,
)


def _resolve_tty_path() -> None:
    """Compatibility seam retained for older tests; TTY routing is disabled."""


def notify_terminal(_message: str) -> None:
    """Compatibility no-op; background work never writes to an interactive TTY."""


class _SecureLogHandler(logging.StreamHandler):
    def flush(self) -> None:
        if self.stream is None or self.stream.closed:
            return
        super().flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if self.stream is not None and not self.stream.closed:
                self.stream.close()
            super().close()


def _remove_tagged_handlers(logger: logging.Logger, attribute: str) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, attribute, False):
            logger.removeHandler(handler)
            handler.close()


def configure_logging() -> None:
    """Configure flush observability only when the CLI actually runs."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(
        getattr(handler, "_memory_flush_console", False)
        for handler in root_logger.handlers
    ):
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        console._memory_flush_console = True  # type: ignore[attr-defined]
        root_logger.addHandler(console)
    target = Path(os.path.abspath(LOG_FILE))
    tagged = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, "_memory_flush_file", False)
    ]
    if len(tagged) == 1 and Path(tagged[0]._memory_log_path) == target:
        return
    _remove_tagged_handlers(root_logger, "_memory_flush_file")
    file_handler = _SecureLogHandler(open_secure_log_stream(target))
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    file_handler._memory_flush_file = True  # type: ignore[attr-defined]
    file_handler._memory_log_path = str(target)  # type: ignore[attr-defined]
    root_logger.addHandler(file_handler)


def load_flush_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_flush_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state), encoding="utf-8")


def append_to_daily_log(
    content: str,
    section: str = "Session",
    project_key: str = "unknown",
    cwd: str = "",
    agent: str = "claude",
    memory_home: Path | str | None = None,
) -> Path:
    """Append content to today's daily log, tagged with project metadata."""
    root = Path(memory_home).expanduser().resolve() if memory_home is not None else ROOT
    return append_daily_entry(
        root,
        content,
        section=section,
        project_key=project_key,
        cwd=cwd,
        agent=agent,
    )


def build_flush_prompt(context: str, project_key: str = "unknown", cwd: str = "") -> str:
    """Build the provider-neutral live extraction prompt."""
    project_block = f"**Project:** {project_key}"
    if cwd:
        project_block += f"\n**CWD:** {cwd}"

    return f"""Review the conversation context below and extract everything worth preserving
in the daily log. Do NOT use any tools, return plain text only.

This conversation took place in the following project:

{project_block}

Scope everything to "{project_key}" so it can be filtered later by project.

The context is pre-filtered to signal. Lines marked [Decision requested],
[Decision made], and [Subagent result] are high-value: they carry decisions the
user made and findings from research. Always preserve these together with the
reasoning behind them. Never discard them as small talk.

Format your response as a daily-log entry, using only the sections that have
real content:

**Context:** One line on what the user was working on.

**Decisions Made:**
- Each decision with its reasoning. Capture every [Decision made]: the question,
  the option chosen, and why.

**Findings & Lessons:**
- Research results ([Subagent result]), gotchas, root causes, and patterns.

**Action Items:**
- Follow-ups, TODOs, and unresolved questions.

Skip only genuinely disposable content:
- Routine tool calls, file reads, and command output
- Trivial confirmations ("looks good", "yes", "go ahead")
- Restating something already captured earlier

Do NOT skip a decision, a design choice, a bug root cause, or a research finding
just because it arrived through a question-and-answer exchange.

Respond with exactly FLUSH_OK and nothing else ONLY if the session contains no
decisions, no findings, no blockers, and no durable facts. A session that made
design or architecture decisions, or resolved a bug, is never FLUSH_OK.

## Conversation Context

{context}"""


def _default_router(memory_home: Path):
    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(memory_home)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    config = load_config(environment)
    return ProviderRouter(
        CodexProvider(task_models=config.task_models),
        ClaudeProvider(model=config.claude_model),
    ), config


async def run_flush(
    context: str,
    project_key: str = "unknown",
    cwd: str = "",
    *,
    router: object | None = None,
    memory_home: Path | str | None = None,
) -> str:
    """Extract important knowledge through the subscription provider router."""
    home = Path(memory_home).expanduser().resolve() if memory_home is not None else ROOT
    if router is None:
        router, config = _default_router(home)
        timeout = config.job_timeout_seconds
    else:
        environment = dict(os.environ)
        environment["AI_MEMORY_HOME"] = str(home)
        environment.pop("CLAUDE_MEMORY_HOME", None)
        timeout = load_config(environment).job_timeout_seconds
    request = TextRequest(
        task=TaskKind.EXTRACT,
        prompt=build_flush_prompt(context, project_key, cwd),
        cwd=home,
        timeout_seconds=timeout,
    )
    try:
        result = await router.generate_text(request)
    except Exception as exc:
        logging.exception("Provider router error")
        return f"FLUSH_ERROR: {type(exc).__name__}: {exc}"
    if result.outcome != "success":
        return f"FLUSH_ERROR: {result.reason or result.outcome}"
    return result.text


COMPILE_AFTER_HOUR = 16  # 4 PM local time
AUTO_COMPILE_LEASE_SECONDS = 120
AUTO_COMPILE_HEARTBEAT_SECONDS = 30
AUTO_COMPILE_WATCHER_POLL_SECONDS = 5
AUTO_COMPILE_SPAWN_ATTEMPTS = 3
AUTO_COMPILE_SPAWN_RETRY_SECONDS = 0.05
AUTO_COMPILE_STARTUP_WAIT_SECONDS = 5
AUTO_COMPILE_BOOTSTRAP_WAIT_SECONDS = AUTO_COMPILE_LEASE_SECONDS * 2
AUTO_COMPILE_MAX_ATTEMPTS = 3
AUTO_COMPILE_RETRY_BASE_SECONDS = 5
_COMPILED_MARKER_RE = re.compile(r"<!--\s*@compiled-through:([^\s>]+)\s*-->")


class _AutoCompileContentChanged(RuntimeError):
    pass


class _AutoCompileReadUnavailable(RuntimeError):
    pass


class _AutoCompileMarkerInvalid(RuntimeError):
    pass


class _AutoCompileWatchdogSpawnError(RuntimeError):
    pass


@dataclass(frozen=True)
class _DailyCompileRead:
    status: Literal["unreadable", "covered", "uncompiled"]
    markers: tuple[str, ...]
    fingerprint: str | None


def _is_memory_codex_provider_command(arguments: list[str]) -> bool:
    """Recognize only the complete command emitted by ``CodexProvider``."""
    if len(arguments) < 17:
        return False
    if arguments[1:8] != [
        "--ask-for-approval",
        "never",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
    ]:
        return False
    cursor = 8
    values: dict[str, str] = {}
    for option in ("--model", "--sandbox", "--cd", "--output-last-message"):
        if arguments[cursor : cursor + 1] != [option] or cursor + 1 >= len(arguments):
            return False
        values[option] = arguments[cursor + 1]
        cursor += 2
    schema_value: str | None = None
    if arguments[cursor : cursor + 1] == ["--output-schema"]:
        if cursor + 1 >= len(arguments):
            return False
        schema_value = arguments[cursor + 1]
        cursor += 2
    if arguments[cursor:] != ["-"]:
        return False

    path_values = [values["--cd"], values["--output-last-message"]]
    if schema_value is not None:
        path_values.append(schema_value)
    windows_paths = any("\\" in value for value in path_values)
    path_type = PureWindowsPath if windows_paths else Path
    workspace = path_type(values["--cd"])
    output = path_type(values["--output-last-message"])
    schema = path_type(schema_value) if schema_value is not None else None
    if (
        not workspace.is_absolute()
        or not output.is_absolute()
        or (schema is not None and not schema.is_absolute())
        or not values["--model"]
    ):
        return False
    if values["--sandbox"] == "read-only":
        return (
            output.name == "last-message.txt"
            and output.parent.name.startswith("ai-memory-codex-")
            and len(output.parent.name) > len("ai-memory-codex-")
        )
    if values["--sandbox"] == "workspace-write":
        return (
            output.parent == workspace
            and re.fullmatch(
                r"\.ai-memory-last-message-[0-9a-f]{32}\.txt", output.name
            )
            is not None
        )
    return False


def _split_windows_command_line(command_line: str) -> list[str]:
    """Parse one Windows command line with the operating system's argv rules."""
    if not command_line:
        return []
    shell32 = ctypes.windll.shell32
    kernel32 = ctypes.windll.kernel32
    count = ctypes.c_int()
    shell32.CommandLineToArgvW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    pointer = shell32.CommandLineToArgvW(command_line, ctypes.byref(count))
    if not pointer:
        raise OSError("CommandLineToArgvW failed")
    try:
        return [pointer[index] for index in range(count.value)]
    finally:
        kernel32.LocalFree(ctypes.cast(pointer, ctypes.c_void_p))


def _list_windows_processes() -> list[tuple[str, list[str]]]:
    """List Windows executable paths and argv through a bounded CIM query."""
    command = [
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-CimInstance Win32_Process "
        "-Filter \"Name = 'claude.exe' OR Name = 'codex.exe'\" | "
        "Select-Object Name,ExecutablePath,CommandLine | ConvertTo-Json -Compress",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=5,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise OSError("Windows process enumeration failed")
    payload = result.stdout.strip()
    if not payload:
        return []
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise OSError("Windows process enumeration returned invalid JSON") from exc
    if decoded is None:
        return []
    rows = decoded if isinstance(decoded, list) else [decoded]
    processes: list[tuple[str, list[str]]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise OSError("Windows process enumeration returned an invalid row")
        executable = row.get("ExecutablePath") or row.get("Name")
        command_line = row.get("CommandLine")
        if not isinstance(executable, str) or not isinstance(command_line, str):
            raise OSError("Windows process enumeration found an inaccessible process")
        processes.append((executable, _split_windows_command_line(command_line)))
    return processes


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _is_bundled_claude(executable: str, arguments: list[str]) -> bool:
    candidates = [executable, arguments[0] if arguments else ""]
    return any(
        re.search(
            r"/_bundled/claude(?:\.exe)?$",
            candidate.replace("\\", "/").lower(),
        )
        is not None
        for candidate in candidates
    )


def _count_agent_processes(processes: list[tuple[str, list[str]]]) -> int:
    count = 0
    for executable, arguments in processes:
        command = _basename(executable)
        if command in {"claude", "claude.exe"}:
            if _is_bundled_claude(executable, arguments):
                continue
            count += 1
        elif command in {"codex", "codex.exe"}:
            if _is_memory_codex_provider_command(arguments):
                continue
            count += 1
    return count


def count_interactive_agent_sessions() -> int:
    """Count interactive Claude Code and Codex CLI processes.

    Provider children are not interactive sessions. Claude's Agent SDK uses a
    bundled ``claude`` binary, while Codex provider calls use the ``exec``
    subcommand. Return ``-1`` when process inspection is unavailable or cannot
    be parsed safely so callers can fail closed.
    """
    if sys.platform == "win32":
        try:
            return _count_agent_processes(_list_windows_processes())
        except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
            return -1

    try:
        result = subprocess.run(
            ["ps", "-A", "-o", "pid=,comm=,args="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return -1
    if result.returncode != 0:
        return -1

    processes: list[tuple[str, list[str]]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 2:
            continue
        comm = parts[1]
        flattened = parts[2] if len(parts) > 2 else ""
        try:
            arguments = shlex.split(flattened)
        except ValueError:
            return -1
        processes.append((comm, arguments))
    return _count_agent_processes(processes)


def _read_daily_compile_state(path: Path) -> _DailyCompileRead:
    """Read fingerprint and marker state once without collapsing failures."""
    try:
        content, _baseline = read_text_with_baseline(path)
    except (OSError, UnicodeError, ValueError):
        return _DailyCompileRead("unreadable", (), None)
    markers = tuple(match.group(1) for match in _COMPILED_MARKER_RE.finditer(content))
    last_end = 0
    for marker in _COMPILED_MARKER_RE.finditer(content):
        last_end = marker.end()
    uncompiled = content[last_end:].strip()
    if not uncompiled:
        return _DailyCompileRead("covered", markers, None)
    material = f"{path.name}\0{uncompiled}".encode()
    return _DailyCompileRead(
        "uncompiled", markers, hashlib.sha256(material).hexdigest()
    )


def _uncompiled_fingerprint(path: Path) -> str | None:
    """Return only a readable uncompiled fingerprint for compatibility callers."""
    read = _read_daily_compile_state(path)
    return read.fingerprint if read.status == "uncompiled" else None


def _queue_content_read(path: Path) -> tuple[str, str | None, tuple[str, ...]]:
    read = _read_daily_compile_state(path)
    return read.status, read.fingerprint, read.markers


def _release_auto_compile(root: Path, token: str, fingerprint: str) -> None:
    from scripts.queue import QueueRepository

    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(root)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    config = load_config(environment)
    with QueueRepository(
        config.queue_path, memory_home=root, sync_usage=False
    ) as repository:
        repository.release_auto_compile(token, fingerprint)


def _clear_auto_compile_watcher(root: Path, token: str) -> None:
    from scripts.queue import QueueRepository

    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(root)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    config = load_config(environment)
    with QueueRepository(
        config.queue_path, memory_home=root, sync_usage=False
    ) as repository:
        repository.clear_auto_compile_watcher(token)


def _clear_auto_compile_contender(root: Path, token: str) -> None:
    from scripts.queue import QueueRepository

    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(root)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    config = load_config(environment)
    with QueueRepository(
        config.queue_path, memory_home=root, sync_usage=False
    ) as repository:
        repository.clear_auto_compile_contender(token)


def _fail_auto_compile_owner_spawn(
    root: Path, token: str, fingerprint: str
) -> None:
    from scripts.queue import QueueRepository

    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(root)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    config = load_config(environment)
    with QueueRepository(
        config.queue_path, memory_home=root, sync_usage=False
    ) as repository:
        repository.fail_auto_compile_owner_spawn(
            token, fingerprint, now=datetime.now(timezone.utc)
        )


def _rollback_auto_compile_schedule(
    root: Path, owner_token: str, watchdog_token: str
) -> None:
    from scripts.queue import QueueRepository

    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(root)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    config = load_config(environment)
    with QueueRepository(
        config.queue_path, memory_home=root, sync_usage=False
    ) as repository:
        repository.rollback_auto_compile_schedule(
            owner_token,
            watchdog_token,
            now=datetime.now(timezone.utc),
        )


def _auto_compile_lock_is_held(root: Path) -> bool:
    lock = ExclusiveFileLock(
        root / "scripts" / "memory-auto-compile.lock", blocking=False
    )
    acquired = lock.acquire()
    if acquired:
        lock.release()
    return not acquired


def _compile_process_options(
    root: Path,
    log_handle: object,
    status_run_id: int | None = None,
) -> dict[str, object]:
    options: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "cwd": str(root),
        "env": _auto_compile_environment(root, status_run_id),
        "close_fds": True,
    }
    if sys.platform == "win32":
        options["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        options["start_new_session"] = True
    return options


def _auto_compile_environment(
    root: Path, status_run_id: int | None = None
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["AI_MEMORY_HOME"] = str(root)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    environment["AI_MEMORY_INTERNAL_JOB"] = "1"
    environment["AI_MEMORY_AUTO_COMPILE"] = "1"
    environment.pop("CLAUDE_MEMORY_TTY", None)
    environment.pop("AI_MEMORY_STATUS_RUN_ID", None)
    if status_run_id is not None:
        environment["AI_MEMORY_STATUS_RUN_ID"] = str(status_run_id)
    return environment


def _spawn_auto_compile_watchdog(
    root: Path,
    coordinator_script: Path,
    token: str,
    predecessor_token: str | None = None,
    *,
    role: str = "watchdog",
    bootstrap: tuple[str, str, str, tuple[str, ...]] | None = None,
) -> None:
    command = [
        sys.executable,
        str(coordinator_script),
        role,
        str(root),
    ]
    if bootstrap is not None:
        owner_token, fingerprint, log_name, marker_prefix = bootstrap
        command.extend(
            [owner_token, fingerprint, log_name, json.dumps(list(marker_prefix))]
        )
    command.append(token)
    if predecessor_token is not None:
        command.append(predecessor_token)
    last_error: Exception | None = None
    for attempt in range(AUTO_COMPILE_SPAWN_ATTEMPTS):
        try:
            with open_secure_log_stream(root / "scripts" / "compile.log") as log_handle:
                subprocess.Popen(
                    command,
                    **_compile_process_options(root, log_handle),
                )
            return
        except Exception as exc:
            last_error = exc
            if attempt + 1 < AUTO_COMPILE_SPAWN_ATTEMPTS:
                time.sleep(AUTO_COMPILE_SPAWN_RETRY_SECONDS)
    raise _AutoCompileWatchdogSpawnError(
        f"watchdog spawn failed after {AUTO_COMPILE_SPAWN_ATTEMPTS} attempts"
    ) from last_error


def maybe_trigger_compilation(
    *,
    memory_home: Path | str | None = None,
    now: datetime | None = None,
) -> bool:
    """Start one detached compile when the local end-of-day gates pass."""
    root = (
        Path(memory_home).expanduser().resolve()
        if memory_home is not None
        else ROOT
    )
    scripts_dir = root / "scripts"
    daily_dir = root / "daily"
    local_now = now or datetime.now(timezone.utc).astimezone()
    if local_now.hour < COMPILE_AFTER_HOUR:
        return False

    instances = count_interactive_agent_sessions()
    if instances != 0:
        if instances < 0:
            logging.info(
                "Skipping compile: cannot verify that Claude and Codex sessions are closed"
            )
        else:
            logging.info(
                "Skipping compile: %d interactive Claude/Codex session(s) still open",
                instances,
            )
        return False

    # Skip if today's log has no unprocessed content past its last
    # `@compiled-through` marker. This is the source of truth — state.json
    # hash is incidental. Lets us auto-fire any number of times per day,
    # processing only the new sessions each time.
    today_log = f"{local_now.strftime('%Y-%m-%d')}.md"
    log_path = daily_dir / today_log
    daily_read = _read_daily_compile_state(log_path)
    if daily_read.status != "uncompiled" or daily_read.fingerprint is None:
        return False
    fingerprint = daily_read.fingerprint

    compile_script = scripts_dir / "compile.py"
    if not compile_script.is_file() or compile_script.is_symlink():
        return False
    coordinator_script = scripts_dir / "auto-compile.py"
    if not coordinator_script.is_file() or coordinator_script.is_symlink():
        return False

    from scripts.queue import QueueRepository

    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(root)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    config = load_config(environment)
    owner_token = secrets.token_hex(32)
    watchdog_token = secrets.token_hex(32)
    lease_now = datetime.now(timezone.utc)
    lease_expires = lease_now + timedelta(seconds=AUTO_COMPILE_LEASE_SECONDS)

    def launch_roles(roles: tuple[str, ...], predecessor: str | None) -> None:
        if "watchdog" in roles:
            _spawn_auto_compile_watchdog(
                root,
                coordinator_script,
                watchdog_token,
                bootstrap=(
                    owner_token,
                    fingerprint,
                    today_log,
                    daily_read.markers,
                ),
            )
        if "contender" in roles:
            if predecessor is None:
                raise _AutoCompileWatchdogSpawnError(
                    "contender registration has no predecessor"
                )
            _spawn_auto_compile_watchdog(
                root,
                coordinator_script,
                watchdog_token,
                predecessor,
                role="contender",
            )
        if "owner" in roles:
            owner_command = [
                sys.executable,
                str(coordinator_script),
                "owner",
                str(root),
                owner_token,
                fingerprint,
            ]
            with open_secure_log_stream(scripts_dir / "compile.log") as log_handle:
                subprocess.Popen(
                    owner_command, **_compile_process_options(root, log_handle)
                )

    try:
        with QueueRepository(
            config.queue_path, memory_home=root, sync_usage=False
        ) as repository:
            roles = repository.schedule_auto_compile(
                owner_token,
                watchdog_token,
                fingerprint,
                log_name=today_log,
                required_marker_prefix=daily_read.markers,
                now=lease_now,
                expires_at=lease_expires,
                launch_roles=launch_roles,
            )
    except Exception as exc:
        logging.error("Failed to provision automatic compile processes: %s", exc)
        return False
    if not roles:
        return False

    logging.info("End-of-day compilation reserved (after %d:00)", COMPILE_AFTER_HOUR)
    return True


def _wait_for_auto_compile_registration(
    root: Path,
    token: str,
    role: str,
    *,
    predecessor_token: str | None,
    sleeper: Callable[[float], None],
    clock: Callable[[], datetime],
) -> bool:
    """Wait briefly for a pre-commit child registration to become visible."""
    from scripts.queue import QueueRepository

    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(root)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    config = load_config(environment)
    deadline = clock() + timedelta(seconds=AUTO_COMPILE_STARTUP_WAIT_SECONDS)
    while True:
        observed_at = clock()
        try:
            with QueueRepository(
                config.queue_path, memory_home=root, sync_usage=False
            ) as repository:
                if repository.auto_compile_role_registered(
                    token, role, predecessor_token=predecessor_token
                ):
                    return True
        except sqlite3.Error:
            pass
        if observed_at >= deadline:
            return False
        sleeper(AUTO_COMPILE_WATCHER_POLL_SECONDS)


def run_auto_compile_owner_startup(
    memory_home: Path | str,
    token: str,
    fingerprint: str,
    *,
    coordinator: Callable[[Path, str, str], bool] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> bool:
    """Wait for the scheduling commit, then enter the ordinary coordinator."""
    root = Path(memory_home).expanduser().resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", token) or not re.fullmatch(
        r"[0-9a-f]{64}", fingerprint
    ):
        return False
    if not _wait_for_auto_compile_registration(
        root,
        token,
        "owner",
        predecessor_token=None,
        sleeper=sleeper,
        clock=clock,
    ):
        return False
    run_coordinator = coordinator or run_auto_compile_coordinator
    return run_coordinator(root, token, fingerprint)


def run_auto_compile_watchdog_bootstrap(
    memory_home: Path | str,
    watchdog_token: str,
    owner_token: str,
    fingerprint: str,
    log_name: str,
    required_marker_prefix: tuple[str, ...],
    *,
    coordinator: Callable[[Path, str, str], bool] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    lock_probe: Callable[[Path], bool] = _auto_compile_lock_is_held,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> bool:
    """Recover or join the genesis reservation after the parent transaction."""
    from scripts.queue import QueueRepository

    root = Path(memory_home).expanduser().resolve()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", watchdog_token)
        or not re.fullmatch(r"[0-9a-f]{64}", owner_token)
        or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        or Path(log_name).name != log_name
        or not all(isinstance(marker, str) for marker in required_marker_prefix)
    ):
        return False
    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(root)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    config = load_config(environment)
    log_path = root / "daily" / log_name
    deadline = clock() + timedelta(seconds=AUTO_COMPILE_BOOTSTRAP_WAIT_SECONDS)
    while True:
        observed_at = clock()
        try:
            with QueueRepository(
                config.queue_path, memory_home=root, sync_usage=False
            ) as repository:
                outcome = repository.bootstrap_auto_compile_watchdog(
                    watchdog_token,
                    owner_token,
                    fingerprint,
                    log_name=log_name,
                    required_marker_prefix=required_marker_prefix,
                    current_content=lambda: _queue_content_read(log_path),
                    now=observed_at,
                    watcher_expires_at=observed_at
                    + timedelta(seconds=AUTO_COMPILE_LEASE_SECONDS),
                )
        except sqlite3.Error:
            if observed_at >= deadline:
                return False
            sleeper(AUTO_COMPILE_WATCHER_POLL_SECONDS)
            continue
        if outcome == "rejected":
            return False
        return run_auto_compile_watcher(
            root,
            watchdog_token,
            coordinator=coordinator,
            sleeper=sleeper,
            lock_probe=lock_probe,
            clock=clock,
        )


def run_auto_compile_coordinator(
    memory_home: Path | str,
    token: str,
    fingerprint: str,
    *,
    compile_launcher: Callable[..., Any] = subprocess.Popen,
    sleeper: Callable[[float], None] = time.sleep,
    lock_probe: Callable[[Path], bool] = _auto_compile_lock_is_held,
) -> bool:
    """Run reserved generations serially until all observed content is covered."""
    from scripts.queue import QueueRepository

    root = Path(memory_home).expanduser().resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        return False
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        return False
    compile_script = root / "scripts" / "compile.py"
    if not compile_script.is_file() or compile_script.is_symlink():
        _release_auto_compile(root, token, fingerprint)
        return False

    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(root)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    config = load_config(environment)
    command = ["uv", "run", "--directory", str(root), "python", str(compile_script)]
    active_fingerprint = fingerprint

    def matching_log(candidate: str) -> Path | None:
        matches = [
            path
            for path in sorted((root / "daily").glob("*.md"))
            if _read_daily_compile_state(path).fingerprint == candidate
        ]
        return matches[0] if len(matches) == 1 else None

    def record_failure(
        log_path: Path,
        error_class: str,
        *,
        reset_on_fingerprint_change: bool = True,
        required_marker_prefix: tuple[str, ...] | None = None,
    ) -> None:
        failed_at = datetime.now(timezone.utc)
        with QueueRepository(
            config.queue_path, memory_home=root, sync_usage=False
        ) as repository:
            repository.record_auto_compile_failure(
                token,
                active_fingerprint,
                error_class,
                lambda: _queue_content_read(log_path),
                now=failed_at,
                expires_at=failed_at
                + timedelta(seconds=AUTO_COMPILE_LEASE_SECONDS),
                max_attempts=AUTO_COMPILE_MAX_ATTEMPTS,
                retry_base_seconds=AUTO_COMPILE_RETRY_BASE_SECONDS,
                reset_on_fingerprint_change=reset_on_fingerprint_change,
                required_marker_prefix=required_marker_prefix,
            )

    try:
        while True:
            log_path = matching_log(active_fingerprint)
            while log_path is None:
                reservation_now = datetime.now(timezone.utc)
                with QueueRepository(
                    config.queue_path, memory_home=root, sync_usage=False
                ) as repository:
                    reservation = repository.auto_compile_reservation(
                        token, now=reservation_now
                    )
                    if reservation is None or reservation[0] != active_fingerprint:
                        return False
                    pending_fingerprint = reservation[1]
                    active_log_name = reservation[2]
                    active_log: Path | None = None
                    if active_log_name is not None:
                        active_log = root / "daily" / active_log_name
                        active_read = _read_daily_compile_state(active_log)
                        if active_read.status == "unreadable":
                            record_failure(active_log, "compile_read_unreadable")
                            return False
                        if active_read.fingerprint == active_fingerprint:
                            log_path = active_log
                            break
                        if (
                            active_read.status == "covered"
                            and pending_fingerprint is None
                        ):
                            required_markers = reservation[4]
                            if required_markers is not None and not (
                                len(active_read.markers) > len(required_markers)
                                and active_read.markers[: len(required_markers)]
                                == required_markers
                            ):
                                record_failure(
                                    active_log,
                                    "compile_invalid_marker_progress",
                                    reset_on_fingerprint_change=False,
                                    required_marker_prefix=required_markers,
                                )
                                return False
                            next_fingerprint = repository.finish_auto_compile_generation(
                                token,
                                active_fingerprint,
                                lambda: _queue_content_read(active_log),
                                now=reservation_now,
                                expires_at=reservation_now
                                + timedelta(seconds=AUTO_COMPILE_LEASE_SECONDS),
                            )
                            if next_fingerprint is None:
                                return True
                            active_fingerprint = next_fingerprint
                            log_path = active_log
                            break
                    if pending_fingerprint is None:
                        return False
                    if active_log is None:
                        return False
                    pending_log_name = reservation[3]
                    pending_log = (
                        root / "daily" / pending_log_name
                        if pending_log_name is not None
                        else matching_log(pending_fingerprint)
                    )
                    if pending_log is None:
                        return False
                    pending_read = _read_daily_compile_state(pending_log)
                    if pending_read.status == "unreadable":
                        record_failure(pending_log, "compile_read_unreadable")
                        return False
                    if pending_read.fingerprint != pending_fingerprint:
                        return False
                    promoted = repository.promote_pending_auto_compile(
                        token,
                        active_fingerprint,
                        pending_fingerprint,
                        lambda: _queue_content_read(active_log),
                        now=reservation_now,
                        expires_at=reservation_now
                        + timedelta(seconds=AUTO_COMPILE_LEASE_SECONDS),
                    )
                if promoted:
                    active_fingerprint = pending_fingerprint
                    log_path = pending_log

            if count_interactive_agent_sessions() != 0:
                return False

            read_before: _DailyCompileRead | None = None
            with QueueRepository(
                config.queue_path, memory_home=root, sync_usage=False
            ) as repository:
                owned = repository.auto_compile_reservation(
                    token, now=datetime.now(timezone.utc)
                )
                if owned is None or owned[0] != active_fingerprint:
                    return False
                required_markers = owned[4]
                status_run_id = owned[5]
                if required_markers is None:
                    return False

                def launch() -> object:
                    nonlocal read_before
                    read_before = _read_daily_compile_state(log_path)
                    if read_before.status == "unreadable":
                        raise _AutoCompileReadUnavailable(
                            "daily log became unreadable before compile launch"
                        )
                    if (
                        len(read_before.markers) < len(required_markers)
                        or read_before.markers[: len(required_markers)]
                        != required_markers
                    ):
                        raise _AutoCompileMarkerInvalid(
                            "daily marker history changed before compile launch"
                        )
                    if read_before.fingerprint != active_fingerprint:
                        raise _AutoCompileContentChanged(
                            "daily content changed before compile launch"
                        )
                    with open_secure_log_stream(
                        root / "scripts" / "compile.log"
                    ) as log_handle:
                        return compile_launcher(
                            command,
                            **_compile_process_options(
                                root, log_handle, status_run_id
                            ),
                        )

                try:
                    launched = repository.launch_reserved_auto_compile(
                        token,
                        active_fingerprint,
                        launch,
                        now=datetime.now(timezone.utc),
                    )
                except _AutoCompileContentChanged:
                    continue
                except _AutoCompileReadUnavailable:
                    record_failure(log_path, "compile_read_unreadable")
                    return False
                except _AutoCompileMarkerInvalid:
                    record_failure(
                        log_path,
                        "compile_invalid_marker_progress",
                        reset_on_fingerprint_change=False,
                        required_marker_prefix=required_markers,
                    )
                    return False
                except Exception as exc:
                    record_failure(log_path, f"compile_launch_{type(exc).__name__}")
                    return False
            if launched is None:
                return False

            while True:
                try:
                    return_code = launched.wait(
                        timeout=AUTO_COMPILE_HEARTBEAT_SECONDS
                    )
                    break
                except subprocess.TimeoutExpired:
                    renewed_at = datetime.now(timezone.utc)
                    with QueueRepository(
                        config.queue_path, memory_home=root, sync_usage=False
                    ) as repository:
                        renewed = repository.renew_auto_compile(
                            token,
                            active_fingerprint,
                            now=renewed_at,
                            expires_at=renewed_at
                            + timedelta(seconds=AUTO_COMPILE_LEASE_SECONDS),
                        )
                    if renewed:
                        continue
                    launched.terminate()
                    try:
                        launched.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        launched.kill()
                        launched.wait()
                    return False
                except Exception as exc:
                    record_failure(log_path, f"compile_wait_{type(exc).__name__}")
                    return False

            if return_code == 75:
                while True:
                    deferred_at = datetime.now(timezone.utc)
                    previous_fingerprint = active_fingerprint
                    compiler_lock_held = lock_probe(root)
                    with QueueRepository(
                        config.queue_path, memory_home=root, sync_usage=False
                    ) as repository:
                        deferred_fingerprint = (
                            repository.defer_auto_compile_generation(
                                token,
                                active_fingerprint,
                                lambda: _queue_content_read(log_path),
                                compiler_lock_held=compiler_lock_held,
                                now=deferred_at,
                                expires_at=deferred_at
                                + timedelta(seconds=AUTO_COMPILE_LEASE_SECONDS),
                            )
                        )
                    if deferred_fingerprint is None:
                        return True
                    active_fingerprint = deferred_fingerprint
                    if deferred_fingerprint != previous_fingerprint:
                        break
                    if not compiler_lock_held:
                        break
                    sleeper(AUTO_COMPILE_WATCHER_POLL_SECONDS)
                continue

            if return_code != 0:
                record_failure(log_path, f"compile_exit_{return_code}")
                return False

            read_after = _read_daily_compile_state(log_path)
            assert read_before is not None
            if read_after.status == "unreadable":
                record_failure(log_path, "compile_read_unreadable")
                return False
            marker_history_valid = (
                len(read_after.markers) >= len(read_before.markers)
                and read_after.markers[: len(read_before.markers)]
                == read_before.markers
            )
            marker_advanced = marker_history_valid and (
                len(read_after.markers) > len(read_before.markers)
            )
            if not marker_advanced:
                markers_unchanged = read_after.markers == read_before.markers
                record_failure(
                    log_path,
                    (
                        "compile_no_progress"
                        if markers_unchanged
                        else "compile_invalid_marker_progress"
                    ),
                    reset_on_fingerprint_change=markers_unchanged,
                    required_marker_prefix=(
                        read_before.markers
                        if read_after.status == "covered"
                        else None
                    ),
                )
                return False

            handoff_at = datetime.now(timezone.utc)
            with QueueRepository(
                config.queue_path, memory_home=root, sync_usage=False
            ) as repository:
                next_fingerprint = repository.finish_auto_compile_generation(
                    token,
                    active_fingerprint,
                    lambda: _queue_content_read(log_path),
                    now=handoff_at,
                    expires_at=handoff_at
                    + timedelta(seconds=AUTO_COMPILE_LEASE_SECONDS),
                )
            if next_fingerprint is None:
                return return_code == 0
            active_fingerprint = next_fingerprint
    except Exception as exc:
        logging.error("Automatic compile failed: %s", exc)
        return False


def _observe_auto_compile_content(
    root: Path, reservation: dict[str, object]
) -> tuple[str, dict[str, object] | None]:
    def current(name: object) -> tuple[_DailyCompileRead, str] | None:
        if name is None:
            return None
        if not isinstance(name, str) or Path(name).name != name:
            return _DailyCompileRead("unreadable", (), None), ""
        path = root / "daily" / name
        return _read_daily_compile_state(path), name

    active = current(reservation.get("log_name"))
    pending = current(reservation.get("pending_log_name"))
    if (active is not None and active[0].status == "unreadable") or (
        pending is not None and pending[0].status == "unreadable"
    ):
        return "unreadable", None
    required_marker_prefix = reservation.get("required_marker_prefix")
    if not isinstance(required_marker_prefix, list) or not all(
        isinstance(marker, str) for marker in required_marker_prefix
    ):
        return "unreadable", None
    required = tuple(required_marker_prefix)
    if active is not None:
        marker_history_valid = (
            len(active[0].markers) >= len(required)
            and active[0].markers[: len(required)] == required
        )
        marker_advanced = marker_history_valid and len(active[0].markers) > len(
            required
        )
        if not marker_history_valid or (
            active[0].status == "covered" and not marker_advanced
        ):
            fingerprint = reservation.get("fingerprint")
            if not isinstance(fingerprint, str):
                return "unreadable", None
            retained = {
                "fingerprint": fingerprint,
                "log_name": active[1],
                "required_marker_prefix": list(required),
            }
            pending_fingerprint = reservation.get("pending_fingerprint")
            pending_log_name = reservation.get("pending_log_name")
            pending_prefix = reservation.get("pending_required_marker_prefix")
            if (
                isinstance(pending_fingerprint, str)
                and isinstance(pending_log_name, str)
                and isinstance(pending_prefix, list)
                and all(isinstance(marker, str) for marker in pending_prefix)
            ):
                retained.update(
                    {
                        "pending_fingerprint": pending_fingerprint,
                        "pending_log_name": pending_log_name,
                        "pending_required_marker_prefix": pending_prefix,
                    }
                )
            return "uncompiled", retained
    if active is None or active[0].status == "covered":
        if pending is None or pending[0].status == "covered":
            return "covered", None
        assert pending[0].fingerprint is not None
        pending_prefix = reservation.get("pending_required_marker_prefix")
        if not isinstance(pending_prefix, list) or not all(
            isinstance(marker, str) for marker in pending_prefix
        ):
            return "unreadable", None
        return "uncompiled", {
            "fingerprint": pending[0].fingerprint,
            "log_name": pending[1],
            "required_marker_prefix": pending_prefix,
        }
    assert active[0].fingerprint is not None
    observed = {"fingerprint": active[0].fingerprint, "log_name": active[1]}
    if len(active[0].markers) > len(required):
        observed["required_marker_prefix"] = list(active[0].markers)
    if (
        pending is not None
        and pending[0].status == "uncompiled"
        and pending[1] != active[1]
    ):
        assert pending[0].fingerprint is not None
        pending_prefix = reservation.get("pending_required_marker_prefix")
        if not isinstance(pending_prefix, list) or not all(
            isinstance(marker, str) for marker in pending_prefix
        ):
            return "unreadable", None
        observed["pending_fingerprint"] = pending[0].fingerprint
        observed["pending_log_name"] = pending[1]
        observed["pending_required_marker_prefix"] = pending_prefix
    return "uncompiled", observed


def run_auto_compile_watcher(
    memory_home: Path | str,
    token: str,
    *,
    predecessor_token: str | None = None,
    registration_required: bool = False,
    coordinator: Callable[[Path, str, str], bool] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    lock_probe: Callable[[Path], bool] = _auto_compile_lock_is_held,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> bool:
    """Wait for one owner lease and durably take over after it expires."""
    from scripts.queue import QueueRepository

    root = Path(memory_home).expanduser().resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        return False
    if predecessor_token is not None and not re.fullmatch(
        r"[0-9a-f]{64}", predecessor_token
    ):
        return False
    if registration_required and not _wait_for_auto_compile_registration(
        root,
        token,
        "contender" if predecessor_token is not None else "watchdog",
        predecessor_token=predecessor_token,
        sleeper=sleeper,
        clock=clock,
    ):
        return False
    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(root)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    config = load_config(environment)
    coordinator_script = root / "scripts" / "auto-compile.py"
    if not coordinator_script.is_file() or coordinator_script.is_symlink():
        return False
    run_coordinator = coordinator or (
        lambda target, owner_token, fingerprint: run_auto_compile_coordinator(
            target, owner_token, fingerprint
        )
    )
    while True:
        observed_at = clock()
        successor_token = secrets.token_hex(32)
        compiler_lock_held = lock_probe(root)
        if compiler_lock_held:
            logging.debug("Automatic compile watcher observed a live child lock")
        try:
            with QueueRepository(
                config.queue_path, memory_home=root, sync_usage=False
            ) as repository:
                status, fingerprint = repository.poll_auto_compile_watcher(
                    token,
                    successor_token,
                    lambda reservation: _observe_auto_compile_content(
                        root, dict(reservation)
                    ),
                    lambda candidate: _spawn_auto_compile_watchdog(
                        root, coordinator_script, candidate, token
                    ),
                    predecessor_token=predecessor_token,
                    registration_required=registration_required,
                    compiler_lock_held=compiler_lock_held,
                    now=observed_at,
                    watcher_expires_at=observed_at
                    + timedelta(seconds=AUTO_COMPILE_LEASE_SECONDS),
                    owner_expires_at=observed_at
                    + timedelta(seconds=AUTO_COMPILE_LEASE_SECONDS),
                )
        except _AutoCompileWatchdogSpawnError as exc:
            logging.error("Failed to spawn successor auto-compile watchdog: %s", exc)
            sleeper(AUTO_COMPILE_WATCHER_POLL_SECONDS)
            continue
        except sqlite3.Error as exc:
            logging.error("Automatic compile watcher SQLite retry: %s", exc)
            sleeper(AUTO_COMPILE_WATCHER_POLL_SECONDS)
            continue
        if status == "done":
            return True
        if status == "claimed":
            assert fingerprint is not None
            if run_coordinator(root, token, fingerprint):
                return True
            continue
        sleeper(AUTO_COMPILE_WATCHER_POLL_SECONDS)


def main():
    configure_logging()
    os.environ.setdefault("CLAUDE_INVOKED_BY", "memory_flush")
    os.environ.setdefault("AI_MEMORY_INTERNAL_JOB", "1")
    if len(sys.argv) < 3:
        logging.error(
            "Usage: %s <context_file.md> <session_id> [project_key] [cwd]",
            sys.argv[0],
        )
        sys.exit(1)

    context_file = Path(sys.argv[1])
    session_id = sys.argv[2]
    project_key = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    cwd = sys.argv[4] if len(sys.argv) > 4 else ""

    logging.info(
        "flush.py started for session %s project=%s context=%s",
        session_id,
        project_key,
        context_file,
    )

    if not context_file.exists():
        logging.error("Context file not found: %s", context_file)
        return

    # Deduplication: skip if same session was flushed within 60 seconds
    state = load_flush_state()
    if (
        state.get("session_id") == session_id
        and time.time() - state.get("timestamp", 0) < 60
    ):
        logging.info("Skipping duplicate flush for session %s", session_id)
        context_file.unlink(missing_ok=True)
        return

    # Read pre-extracted context
    context = context_file.read_text(encoding="utf-8").strip()
    if not context:
        logging.info("Context file is empty, skipping")
        context_file.unlink(missing_ok=True)
        return

    logging.info("Flushing session %s: %d chars", session_id, len(context))

    # Run the LLM extraction
    response = asyncio.run(run_flush(context, project_key, cwd))

    # Append to daily log
    if response.strip() == "FLUSH_OK":
        logging.info("Result: FLUSH_OK")
        append_to_daily_log(
            "FLUSH_OK - Nothing worth saving from this session",
            "Memory Flush",
            project_key,
            cwd,
        )
    elif "FLUSH_ERROR" in response:
        logging.error("Result: %s", response)
        append_to_daily_log(response, "Memory Flush", project_key, cwd)
        # Keep the context file so the flush can be retried by hand:
        #   uv run python scripts/flush.py <context_file> <session_id> [project] [cwd]
        # No dedup-state update here: the dedup path deletes the context
        # file, which would defeat an immediate retry.
        logging.error("Context file preserved for retry: %s", context_file)
        return
    else:
        logging.info("Result: saved to daily log (%d chars)", len(response))
        append_to_daily_log(response, "Session", project_key, cwd)

    # Update dedup state
    save_flush_state({"session_id": session_id, "timestamp": time.time()})

    # Clean up context file
    context_file.unlink(missing_ok=True)


    # End-of-day auto-compilation: if it's past the compile hour and today's
    # log hasn't been compiled yet, trigger compile.py in the background.
    maybe_trigger_compilation()

    logging.info("Flush complete for session %s", session_id)


if __name__ == "__main__":
    main()
