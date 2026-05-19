"""
SessionEnd hook - captures conversation transcript for memory extraction.

When a Claude Code session ends, this hook reads the transcript path from
stdin, extracts conversation context, and spawns flush.py as a background
process to extract knowledge into the daily log.

The hook itself does NO API calls - only local file I/O for speed (<10s).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Recursion guard: if we were spawned by flush.py (which calls Agent SDK,
# which runs Claude Code, which would fire this hook again), exit immediately.
if os.environ.get("CLAUDE_INVOKED_BY"):
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "daily"
SCRIPTS_DIR = ROOT / "scripts"
STATE_DIR = SCRIPTS_DIR

logging.basicConfig(
    filename=str(SCRIPTS_DIR / "flush.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [hook] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MAX_TURNS = 30
MAX_CONTEXT_CHARS = 15_000
MIN_TURNS_TO_FLUSH = 1

# Dedup guard: prevents the hook doing duplicate work when both project-local
# and user-global settings.json have it configured (Claude Code does not dedupe
# identical hook commands across scopes).
HOOK_DEDUP_WINDOW_SECONDS = 10
HOOK_DEDUP_FILE = SCRIPTS_DIR / "last-hook-fire.json"


def already_fired(session_id: str, event: str) -> bool:
    """Returns True if this (session_id, event) was handled within the dedup window."""
    try:
        data = json.loads(HOOK_DEDUP_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        data = {}

    key = f"{session_id}:{event}"
    now = time.time()
    last = data.get(key, 0)
    if now - last < HOOK_DEDUP_WINDOW_SECONDS:
        return True

    data[key] = now
    cutoff = now - 3600
    data = {k: v for k, v in data.items() if v > cutoff}
    try:
        HOOK_DEDUP_FILE.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass
    return False


def extract_conversation_context(transcript_path: Path) -> tuple[str, int]:
    """Read JSONL transcript and extract last ~N conversation turns as markdown."""
    turns: list[str] = []

    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = entry.get("message", {})
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = entry.get("role", "")
                content = entry.get("content", "")

            if role not in ("user", "assistant"):
                continue

            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)

            if isinstance(content, str) and content.strip():
                label = "User" if role == "user" else "Assistant"
                turns.append(f"**{label}:** {content.strip()}\n")

    recent = turns[-MAX_TURNS:]
    context = "\n".join(recent)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[-MAX_CONTEXT_CHARS:]
        boundary = context.find("\n**")
        if boundary > 0:
            context = context[boundary + 1 :]

    return context, len(recent)


def main() -> None:
    # Read hook input from stdin
    # Claude Code on Windows may pass paths with unescaped backslashes
    try:
        raw_input = sys.stdin.read()
        try:
            hook_input: dict = json.loads(raw_input)
        except json.JSONDecodeError:
            fixed_input = re.sub(r'(?<!\\)\\(?!["\\])', r'\\\\', raw_input)
            hook_input = json.loads(fixed_input)
    except (json.JSONDecodeError, ValueError, EOFError) as e:
        logging.error("Failed to parse stdin: %s", e)
        return

    session_id = hook_input.get("session_id", "unknown")
    source = hook_input.get("source", "unknown")
    transcript_path_str = hook_input.get("transcript_path", "")
    cwd_str = hook_input.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    project_key = Path(cwd_str).name or "unknown"

    logging.info("SessionEnd fired: session=%s source=%s project=%s", session_id, source, project_key)

    if already_fired(session_id, "session-end"):
        logging.info("SKIP: duplicate hook fire for session %s (project+global both configured)", session_id)
        return

    if not transcript_path_str or not isinstance(transcript_path_str, str):
        logging.info("SKIP: no transcript path")
        return

    transcript_path = Path(transcript_path_str)
    if not transcript_path.exists():
        logging.info("SKIP: transcript missing: %s", transcript_path_str)
        return

    # Extract conversation context in the hook (fast, no API calls)
    try:
        context, turn_count = extract_conversation_context(transcript_path)
    except Exception as e:
        logging.error("Context extraction failed: %s", e)
        return

    if not context.strip():
        logging.info("SKIP: empty context")
        return

    if turn_count < MIN_TURNS_TO_FLUSH:
        logging.info("SKIP: only %d turns (min %d)", turn_count, MIN_TURNS_TO_FLUSH)
        return

    # Write context to a temp file for the background process
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    context_file = STATE_DIR / f"session-flush-{session_id}-{timestamp}.md"
    context_file.write_text(context, encoding="utf-8")

    # Spawn flush.py as a background process
    flush_script = SCRIPTS_DIR / "flush.py"

    cmd = [
        "uv",
        "run",
        "--directory",
        str(ROOT),
        "python",
        str(flush_script),
        str(context_file),
        session_id,
        project_key,
        cwd_str,
    ]

    # On Windows, use CREATE_NO_WINDOW to avoid flash console window.
    # Do NOT use DETACHED_PROCESS — it breaks the Agent SDK's subprocess I/O.
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    # Resolve the user's terminal TTY NOW — while `claude` is still alive —
    # and pass it to the detached flush.py via env var. Once claude exits,
    # flush.py is reparented to launchd and loses the ancestry path back to
    # the real terminal device.
    env = os.environ.copy()
    tty_path = _resolve_user_tty()
    if tty_path:
        env["CLAUDE_MEMORY_TTY"] = tty_path
        logging.info("Resolved user TTY: %s", tty_path)
    else:
        logging.info("No user TTY resolved (notifications will silently no-op)")

    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
            env=env,
        )
        logging.info("Spawned flush.py for session %s (%d turns, %d chars)", session_id, turn_count, len(context))
    except Exception as e:
        logging.error("Failed to spawn flush.py: %s", e)


def _resolve_user_tty() -> str | None:
    """Walk ancestors to find a process attached to the user's real terminal.

    The hook itself has TTY=?? (Claude Code spawns hooks without a TTY), but
    the parent `claude` CLI does own the user's terminal. This must run
    BEFORE `claude` exits — once flush.py is detached, the PPID chain
    collapses to launchd and the TTY is unrecoverable.
    """
    pid = os.getpid()
    for _ in range(10):
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "ppid=,tty="],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split()
        if len(parts) < 2:
            return None
        try:
            ppid = int(parts[0])
        except ValueError:
            return None
        tty = parts[1]
        if tty and tty != "??":
            candidate = tty if tty.startswith("/") else f"/dev/{tty}"
            if os.path.exists(candidate):
                return candidate
        if ppid in (0, 1) or ppid == pid:
            return None
        pid = ppid
    return None


if __name__ == "__main__":
    main()
