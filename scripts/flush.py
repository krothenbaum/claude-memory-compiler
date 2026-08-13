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
import json
import logging
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
        if "exec" in arguments[1:]:
            continue
        count += 1
    return count


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
    try:
        content = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    marker_re = re.compile(r"<!--\s*@compiled-through:([^\s>]+)\s*-->")
    last_end = 0
    for marker in marker_re.finditer(content):
        last_end = marker.end()
    if not content[last_end:].strip():
        return False

    compile_script = scripts_dir / "compile.py"
    if not compile_script.is_file() or compile_script.is_symlink():
        return False

    logging.info("End-of-day compilation triggered (after %d:00)", COMPILE_AFTER_HOUR)

    cmd = ["uv", "run", "--directory", str(root), "python", str(compile_script)]

    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True

    # Hand the controlling-TTY path to the detached compile.py subprocess so
    # it can write progress messages even though `start_new_session=True`
    # severs /dev/tty access.
    env = os.environ.copy()
    env["AI_MEMORY_HOME"] = str(root)
    env.pop("CLAUDE_MEMORY_HOME", None)
    env["AI_MEMORY_INTERNAL_JOB"] = "1"
    tty_path = _resolve_tty_path()
    if tty_path:
        env["CLAUDE_MEMORY_TTY"] = tty_path

    try:
        with open_secure_log_stream(scripts_dir / "compile.log") as log_handle:
            subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=str(root),
                env=env,
                close_fds=True,
                **kwargs,
            )
        notify_terminal("end-of-day compile triggered")
        return True
    except Exception as exc:
        logging.error("Failed to spawn compile.py: %s", exc)
        return False


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
