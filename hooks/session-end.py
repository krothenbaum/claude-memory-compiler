"""Fast Claude SessionEnd adapter: normalize, enqueue, and return."""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Iterator


# This must precede imports of the capture/queue modules: those modules can
# create runtime state when their public entry points are used.
if os.environ.get("AI_MEMORY_INTERNAL_JOB") == "1" or "CLAUDE_INVOKED_BY" in os.environ:
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.capture import enqueue_hook_input
from scripts.transcripts import parse_claude_transcript, render_turns


MAX_TURNS = 30
MAX_CONTEXT_CHARS = 15_000
MIN_TURNS_TO_FLUSH = 1
LIVE_TRANSCRIPT_TAIL_BYTES = 1_000_000
MAX_LIVE_JSONL_RECORD_BYTES = 500_000


class LiveTranscriptRejected(ValueError):
    """A live transcript cannot be sliced without risking partial capture."""


def _runtime_root() -> Path:
    configured = os.environ.get("AI_MEMORY_HOME") or os.environ.get(
        "CLAUDE_MEMORY_HOME"
    )
    return Path(configured).expanduser() if configured else ROOT


def _logger() -> logging.Logger:
    logger = logging.getLogger("ai-memory-session-end")
    if logger.handlers:
        return logger
    try:
        log_dir = _runtime_root() / "scripts" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(
            log_dir / "hooks.log", encoding="utf-8"
        )
    except OSError:
        handler = logging.NullHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [session-end] %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _read_hook_input() -> dict[str, object]:
    """Read Claude JSON, retaining the legacy Windows-backslash recovery."""
    raw_input = sys.stdin.read()
    try:
        value = json.loads(raw_input)
    except json.JSONDecodeError:
        fixed_input = re.sub(r'(?<!\\)\\(?!["\\])', r"\\\\", raw_input)
        value = json.loads(fixed_input)
    if not isinstance(value, dict):
        raise ValueError("hook input must be a JSON object")
    return value


def _metadata_prefix(first_record: bytes) -> bytes:
    """Preserve bounded session metadata when the head falls outside the tail."""
    if not first_record or len(first_record) > MAX_LIVE_JSONL_RECORD_BYTES:
        return b""
    try:
        record = json.loads(first_record)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return b""
    if not isinstance(record, dict):
        return b""
    if record.get("type") == "session_meta" and isinstance(record.get("payload"), dict):
        preserved: dict[str, object] = {
            "type": "session_meta",
            "payload": {
                key: record["payload"][key]
                for key in ("id", "cwd")
                if key in record["payload"]
            },
        }
        if "timestamp" in record:
            preserved["timestamp"] = record["timestamp"]
    else:
        preserved = {
            key: record[key]
            for key in ("sessionId", "session_id", "cwd", "timestamp")
            if key in record
        }
    if not preserved:
        return b""
    return json.dumps(preserved, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _bounded_tail(source: Path) -> bytes:
    """Read a deterministic JSONL tail without retaining a partial first line."""
    size = source.stat().st_size
    with source.open("rb") as stream:
        first_record = stream.readline(MAX_LIVE_JSONL_RECORD_BYTES + 1).rstrip(b"\r\n")
        start = max(0, size - LIVE_TRANSCRIPT_TAIL_BYTES)
        stream.seek(start)
        tail = stream.read(LIVE_TRANSCRIPT_TAIL_BYTES)

    if start:
        boundary = tail.find(b"\n")
        tail = b"" if boundary < 0 else tail[boundary + 1 :]
        prefix = _metadata_prefix(first_record)
    else:
        prefix = b""

    lines = tail.splitlines(keepends=True)
    for line in lines:
        record = line.rstrip(b"\r\n")
        if len(record) > MAX_LIVE_JSONL_RECORD_BYTES:
            raise LiveTranscriptRejected("live transcript contains an oversized JSONL record")

    # A concurrently written final line is safe only when it is complete JSON.
    if lines and not lines[-1].endswith((b"\n", b"\r")):
        try:
            json.loads(lines[-1])
        except (json.JSONDecodeError, UnicodeDecodeError):
            lines.pop()
    return prefix + b"".join(lines)


@contextmanager
def bounded_transcript_slice(source: Path) -> Iterator[Path]:
    """Yield an owner-private bounded live JSONL slice and always remove it."""
    payload = _bounded_tail(source)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="ai-memory-live-", suffix=".jsonl"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        yield temporary
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        temporary.unlink(missing_ok=True)


def extract_conversation_context(
    transcript_path: Path, metadata: dict | None = None
) -> tuple[str, int]:
    """Preserve the characterized Claude live-slice rendering contract."""
    session = parse_claude_transcript(
        transcript_path,
        metadata or {},
        limits={"max_turns": MAX_TURNS},
    )
    context = render_turns(session)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[-MAX_CONTEXT_CHARS:]
        boundary = context.find("\n**")
        if boundary > 0:
            context = context[boundary + 1 :]
    return context, len(session.turns)


def _resolve_user_tty() -> str | None:
    """Find the terminal while the parent Claude process is still alive."""
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
            parent = int(parts[0])
        except ValueError:
            return None
        tty = parts[1]
        if tty and tty != "??":
            candidate = tty if tty.startswith("/") else f"/dev/{tty}"
            if os.path.exists(candidate):
                return candidate
        if parent in (0, 1) or parent == pid:
            return None
        pid = parent
    return None


def main() -> None:
    logger = _logger()
    try:
        hook_input = _read_hook_input()
    except (json.JSONDecodeError, ValueError, EOFError) as error:
        logger.error("failed to parse hook input: %s", error)
        return

    transcript_value = hook_input.get("transcript_path")
    if not isinstance(transcript_value, str) or not transcript_value:
        logger.info("skip: no transcript path")
        return
    transcript_path = Path(transcript_value).expanduser()
    if not transcript_path.is_file():
        logger.info("skip: transcript missing")
        return

    cwd = hook_input.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        cwd = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
        hook_input["cwd"] = cwd
    hook_input.setdefault("project", Path(cwd).name or "unknown")

    try:
        with bounded_transcript_slice(transcript_path) as live_slice:
            context, turn_count = extract_conversation_context(
                live_slice,
                {
                    "session_id": hook_input.get("session_id", ""),
                    "cwd": cwd,
                    "project": hook_input["project"],
                    "timestamp": hook_input.get("timestamp", ""),
                    "trigger": "session_end",
                },
            )
            if not context.strip() or turn_count < MIN_TURNS_TO_FLUSH:
                logger.info("skip: empty or too-short transcript")
                return

            tty_path = _resolve_user_tty()
            if tty_path:
                os.environ["CLAUDE_MEMORY_TTY"] = tty_path
            capture_input = dict(hook_input)
            capture_input["transcript_path"] = str(live_slice)
            outcome = enqueue_hook_input(
                capture_input,
                source_agent="claude",
                trigger="session_end",
                limits={"max_turns": MAX_TURNS, "max_chars": MAX_CONTEXT_CHARS},
            )
        logger.info("capture %s for session %s", outcome.status, outcome.job_id)
    except Exception as error:
        # Hooks are advisory. A capture failure must never block the host agent.
        logger.error("capture failed: %s", error)


if __name__ == "__main__":
    main()
