"""
Memory flush agent - extracts important knowledge from conversation context.

Spawned by session-end.py or pre-compact.py as a background process. Reads
pre-extracted conversation context from a .md file, uses the Claude Agent SDK
to decide what's worth saving, and appends the result to today's daily log.

Usage:
    uv run python flush.py <context_file.md> <session_id>
"""

from __future__ import annotations

# Recursion prevention: set this BEFORE any imports that might trigger Claude
import os
os.environ["CLAUDE_INVOKED_BY"] = "memory_flush"

import asyncio
import json
import logging
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
from utils import notify_terminal, _resolve_tty_path  # noqa: E402

# Set up file-based logging so we can verify the background process ran.
# The parent process sends stdout/stderr to DEVNULL (to avoid the inherited
# file handle bug on Windows), so this is our only observability channel.
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
) -> None:
    """Append content to today's daily log, tagged with project metadata."""
    today = datetime.now(timezone.utc).astimezone()
    log_path = DAILY_DIR / f"{today.strftime('%Y-%m-%d')}.md"

    if not log_path.exists():
        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"# Daily Log: {today.strftime('%Y-%m-%d')}\n\n## Sessions\n\n## Memory Maintenance\n\n",
            encoding="utf-8",
        )

    time_str = today.strftime("%H:%M")
    header = f"### {section} [{project_key}] ({time_str})"
    metadata_lines = [f"**Project:** {project_key}"]
    if cwd:
        metadata_lines.append(f"**CWD:** {cwd}")
    metadata_block = "\n".join(metadata_lines)
    entry = f"{header}\n\n{metadata_block}\n\n{content}\n\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


async def run_flush(context: str, project_key: str = "unknown", cwd: str = "") -> str:
    """Use Claude Agent SDK to extract important knowledge from conversation context."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    project_block = f"**Project:** {project_key}"
    if cwd:
        project_block += f"\n**CWD:** {cwd}"

    prompt = f"""Review the conversation context below and respond with a concise summary
of important items that should be preserved in the daily log.
Do NOT use any tools — just return plain text.

This conversation took place in the following project:

{project_block}

Treat the project key as the canonical scope for everything you extract. Anything
project-specific (e.g. a coding pattern, a bug, a decision) should be described
as belonging to "{project_key}" so it can be filtered later by project.

Format your response as a structured daily log entry with these sections:

**Context:** [One line about what the user was working on]

**Key Exchanges:**
- [Important Q&A or discussions]

**Decisions Made:**
- [Any decisions with rationale]

**Lessons Learned:**
- [Gotchas, patterns, or insights discovered]

**Action Items:**
- [Follow-ups or TODOs mentioned]

Skip anything that is:
- Routine tool calls or file reads
- Content that's trivial or obvious
- Trivial back-and-forth or clarification exchanges

Only include sections that have actual content. If nothing is worth saving,
respond with exactly: FLUSH_OK

## Conversation Context

{context}"""

    response = ""

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=str(ROOT),
                allowed_tools=[],
                max_turns=2,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response += block.text
            elif isinstance(message, ResultMessage):
                pass
    except Exception as e:
        import traceback
        logging.error("Agent SDK error: %s\n%s", e, traceback.format_exc())
        response = f"FLUSH_ERROR: {type(e).__name__}: {e}"

    return response


COMPILE_AFTER_HOUR = 16  # 4 PM local time


def count_claude_instances() -> int:
    """Count running Claude Code CLI processes, excluding the bundled `claude`
    binary that flush.py itself spawns via the Agent SDK.

    Concurrent flush.py runs each launch
    `.venv/.../claude_agent_sdk/_bundled/claude` as a child during the LLM
    call. `pgrep -x claude` cannot distinguish that from a real Claude Code
    session — and on macOS `pgrep -a` does not emit the full command line —
    so we shell out to `ps` and inspect each process's argv for the bundled
    path.

    Returns -1 if `ps` is unavailable or fails (caller should treat as
    unknown and skip, to avoid running compile while other sessions are
    still active).
    """
    import subprocess as _sp

    try:
        result = _sp.run(
            ["ps", "-A", "-o", "pid=,comm=,args="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, _sp.TimeoutExpired, OSError):
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
        if comm.rsplit("/", 1)[-1] != "claude":
            continue
        if "claude_agent_sdk/_bundled" in args:
            continue
        count += 1
    return count


def maybe_trigger_compilation() -> None:
    """If it's past the compile hour, no other Claude instances are open, and
    today's log hasn't been compiled, run compile.py."""
    import subprocess as _sp

    now = datetime.now(timezone.utc).astimezone()
    if now.hour < COMPILE_AFTER_HOUR:
        return

    # Only compile when no Claude Code instances are still open. flush.py is a
    # detached background subprocess, so the session that triggered it has
    # typically exited by now; any remaining `claude` processes are other
    # active sessions we should not interrupt.
    instances = count_claude_instances()
    if instances != 0:
        if instances < 0:
            logging.info("Skipping compile: pgrep unavailable, cannot verify no Claude instances open")
        else:
            logging.info("Skipping compile: %d Claude instance(s) still open", instances)
        return

    # Skip if today's log has no unprocessed content past its last
    # `@compiled-through` marker. This is the source of truth — state.json
    # hash is incidental. Lets us auto-fire any number of times per day,
    # processing only the new sessions each time.
    today_log = f"{now.strftime('%Y-%m-%d')}.md"
    log_path = DAILY_DIR / today_log
    if log_path.exists():
        try:
            content = log_path.read_text(encoding="utf-8")
        except OSError:
            content = ""
        if content:
            # Inline regex to avoid importing compile.py (which loads the Agent SDK).
            import re as _re
            marker_re = _re.compile(r"<!--\s*@compiled-through:([^\s>]+)\s*-->")
            last_end = 0
            for m in marker_re.finditer(content):
                last_end = m.end()
            if not content[last_end:].strip():
                return  # nothing new since the last compile marker

    compile_script = SCRIPTS_DIR / "compile.py"
    if not compile_script.exists():
        return

    logging.info("End-of-day compilation triggered (after %d:00)", COMPILE_AFTER_HOUR)

    cmd = ["uv", "run", "--directory", str(ROOT), "python", str(compile_script)]

    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    # Hand the controlling-TTY path to the detached compile.py subprocess so
    # it can write progress messages even though `start_new_session=True`
    # severs /dev/tty access.
    env = os.environ.copy()
    tty_path = _resolve_tty_path()
    if tty_path:
        env["CLAUDE_MEMORY_TTY"] = tty_path

    try:
        log_handle = open(str(SCRIPTS_DIR / "compile.log"), "a")
        _sp.Popen(
            cmd,
            stdout=log_handle,
            stderr=_sp.STDOUT,
            cwd=str(ROOT),
            env=env,
            **kwargs,
        )
        notify_terminal("end-of-day compile triggered")
    except Exception as e:
        logging.error("Failed to spawn compile.py: %s", e)


def main():
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
    if "FLUSH_OK" in response:
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
