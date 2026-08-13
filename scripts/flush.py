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
import hashlib
import json
import logging
import re
import secrets
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "daily"
SCRIPTS_DIR = ROOT / "scripts"
STATE_FILE = SCRIPTS_DIR / "last-flush.json"
LOG_FILE = SCRIPTS_DIR / "flush.log"

sys.path.insert(0, str(SCRIPTS_DIR))
from utils import (  # noqa: E402
    _resolve_tty_path,
    append_daily_entry,
    notify_terminal,
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


class _AutoCompileContentChanged(RuntimeError):
    pass


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
    if arguments[cursor : cursor + 1] == ["--output-schema"]:
        if cursor + 1 >= len(arguments):
            return False
        schema = Path(arguments[cursor + 1])
        if not schema.is_absolute():
            return False
        cursor += 2
    if arguments[cursor:] != ["-"]:
        return False

    workspace = Path(values["--cd"])
    output = Path(values["--output-last-message"])
    if not workspace.is_absolute() or not output.is_absolute() or not values["--model"]:
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


def count_interactive_agent_sessions() -> int:
    """Count interactive Claude Code and Codex CLI processes.

    Provider children are not interactive sessions. Claude's Agent SDK uses a
    bundled ``claude`` binary, while Codex provider calls use the ``exec``
    subcommand. Return ``-1`` when process inspection is unavailable or cannot
    be parsed safely so callers can fail closed.
    """
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

    count = 0
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 2:
            continue
        comm = parts[1]
        args = parts[2] if len(parts) > 2 else ""
        command = comm.rsplit("/", 1)[-1].lower()
        if command == "claude":
            if "/_bundled/claude" in args.replace("\\", "/"):
                continue
            count += 1
            continue
        if command != "codex":
            continue
        try:
            arguments = shlex.split(args)
        except ValueError:
            return -1
        if _is_memory_codex_provider_command(arguments):
            continue
        count += 1
    return count


def _uncompiled_fingerprint(path: Path) -> str | None:
    try:
        content, _baseline = read_text_with_baseline(path)
    except (OSError, UnicodeError, ValueError):
        return None
    marker_re = re.compile(r"<!--\s*@compiled-through:([^\s>]+)\s*-->")
    last_end = 0
    for marker in marker_re.finditer(content):
        last_end = marker.end()
    uncompiled = content[last_end:].strip()
    if not uncompiled:
        return None
    material = f"{path.name}\0{uncompiled}".encode()
    return hashlib.sha256(material).hexdigest()


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


def _compile_process_options(root: Path, log_handle: object) -> dict[str, object]:
    options: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "cwd": str(root),
        "env": _auto_compile_environment(root),
        "close_fds": True,
    }
    if sys.platform == "win32":
        options["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        options["start_new_session"] = True
    return options


def _auto_compile_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["AI_MEMORY_HOME"] = str(root)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    environment["AI_MEMORY_INTERNAL_JOB"] = "1"
    environment["AI_MEMORY_AUTO_COMPILE"] = "1"
    tty_path = _resolve_tty_path()
    if tty_path:
        environment["CLAUDE_MEMORY_TTY"] = tty_path
    return environment


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
    fingerprint = _uncompiled_fingerprint(log_path)
    if fingerprint is None:
        return False

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
    token = secrets.token_hex(32)
    lease_now = datetime.now(timezone.utc)
    lease_expires = lease_now + timedelta(seconds=AUTO_COMPILE_LEASE_SECONDS)
    with QueueRepository(
        config.queue_path, memory_home=root, sync_usage=False
    ) as repository:
        if not repository.reserve_auto_compile(
            token,
            fingerprint,
            now=lease_now,
            expires_at=lease_expires,
        ):
            return False

    logging.info("End-of-day compilation reserved (after %d:00)", COMPILE_AFTER_HOUR)
    command = [
        sys.executable,
        str(coordinator_script),
        str(root),
        token,
        fingerprint,
    ]

    try:
        with open_secure_log_stream(scripts_dir / "compile.log") as log_handle:
            subprocess.Popen(command, **_compile_process_options(root, log_handle))
        notify_terminal("end-of-day compile scheduled")
        return True
    except Exception as exc:
        _release_auto_compile(root, token, fingerprint)
        logging.error("Failed to spawn auto-compile coordinator: %s", exc)
        return False


def run_auto_compile_coordinator(
    memory_home: Path | str,
    token: str,
    fingerprint: str,
    *,
    compile_launcher: Callable[..., Any] = subprocess.Popen,
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
            if _uncompiled_fingerprint(path) == candidate
        ]
        return matches[0] if len(matches) == 1 else None

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
                    if pending_fingerprint is None:
                        return False
                    pending_log = matching_log(pending_fingerprint)
                    if pending_log is None:
                        return False
                    promoted = repository.promote_pending_auto_compile(
                        token,
                        active_fingerprint,
                        pending_fingerprint,
                        now=reservation_now,
                        expires_at=reservation_now
                        + timedelta(seconds=AUTO_COMPILE_LEASE_SECONDS),
                    )
                if promoted:
                    active_fingerprint = pending_fingerprint
                    log_path = pending_log

            if count_interactive_agent_sessions() != 0:
                return False

            with QueueRepository(
                config.queue_path, memory_home=root, sync_usage=False
            ) as repository:
                def launch() -> object:
                    if _uncompiled_fingerprint(log_path) != active_fingerprint:
                        raise _AutoCompileContentChanged(
                            "daily content changed before compile launch"
                        )
                    with open_secure_log_stream(
                        root / "scripts" / "compile.log"
                    ) as log_handle:
                        return compile_launcher(
                            command, **_compile_process_options(root, log_handle)
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

            handoff_at = datetime.now(timezone.utc)
            with QueueRepository(
                config.queue_path, memory_home=root, sync_usage=False
            ) as repository:
                next_fingerprint = repository.finish_auto_compile_generation(
                    token,
                    active_fingerprint,
                    lambda: _uncompiled_fingerprint(log_path),
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
    finally:
        _release_auto_compile(root, token, active_fingerprint)


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
    notify_terminal(f"flush started — project={project_key} ({len(context)} chars)")

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
        result_summary = "FLUSH_OK (nothing worth saving)"
    elif "FLUSH_ERROR" in response:
        logging.error("Result: %s", response)
        append_to_daily_log(response, "Memory Flush", project_key, cwd)
        result_summary = "FLUSH_ERROR (see flush.log)"
        # Keep the context file so the flush can be retried by hand:
        #   uv run python scripts/flush.py <context_file> <session_id> [project] [cwd]
        # No dedup-state update here: the dedup path deletes the context
        # file, which would defeat an immediate retry.
        logging.error("Context file preserved for retry: %s", context_file)
        notify_terminal(f"flush FAILED — context preserved at {context_file.name}")
        return
    else:
        logging.info("Result: saved to daily log (%d chars)", len(response))
        append_to_daily_log(response, "Session", project_key, cwd)
        result_summary = f"saved {len(response)} chars to daily log"

    # Update dedup state
    save_flush_state({"session_id": session_id, "timestamp": time.time()})

    # Clean up context file
    context_file.unlink(missing_ok=True)

    notify_terminal(f"flush complete — {result_summary}")

    # End-of-day auto-compilation: if it's past the compile hour and today's
    # log hasn't been compiled yet, trigger compile.py in the background.
    maybe_trigger_compilation()

    logging.info("Flush complete for session %s", session_id)


if __name__ == "__main__":
    main()
