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
from typing import Callable, Iterator


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
MAX_LIVE_TRANSCRIPT_SCAN_BYTES = 16_000_000
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


def _validated_jsonl(records: bytes) -> bytes:
    """Return complete UTF-8 JSON-object records or reject the whole slice."""
    validated: list[bytes] = []
    for line in records.splitlines():
        if not line.strip():
            continue
        if len(line) > MAX_LIVE_JSONL_RECORD_BYTES:
            raise LiveTranscriptRejected(
                "live transcript contains an oversized JSONL record"
            )
        try:
            text = line.decode("utf-8", errors="strict")
            record = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LiveTranscriptRejected(
                "live transcript contains malformed JSONL"
            ) from error
        if not isinstance(record, dict):
            raise LiveTranscriptRejected(
                "live transcript JSONL records must be objects"
            )
        validated.append(line + b"\n")
    return b"".join(validated)


def _bounded_tail(
    source: Path,
    *,
    size: int,
    window_bytes: int,
    metadata_prefix: bytes,
) -> tuple[bytes, bool]:
    """Read and validate one deterministic tail window."""
    start = max(0, size - window_bytes)
    with source.open("rb") as stream:
        stream.seek(start)
        tail = stream.read(min(window_bytes, size))

    if start:
        boundary = tail.find(b"\n")
        tail = b"" if boundary < 0 else tail[boundary + 1 :]
        prefix = metadata_prefix
    else:
        prefix = b""
    return prefix + _validated_jsonl(tail), start == 0


def _write_private_slice(path: Path, payload: bytes, *, durable: bool) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        if durable:
            os.fsync(stream.fileno())


@contextmanager
def bounded_transcript_slice(
    source: Path,
    previewer: Callable[[Path], object],
) -> Iterator[tuple[Path, object]]:
    """Select a semantic tail under a hard 16 MB fail-closed scan budget.

    Windows expand geometrically until the shared normalizer finds a durable
    turn or the file start is reached. A larger file with no signal in the
    final 16 MB is pathological live input and is rejected for later recovery.
    """
    size = source.stat().st_size
    with source.open("rb") as stream:
        first_record = stream.readline(MAX_LIVE_JSONL_RECORD_BYTES + 1).rstrip(
            b"\r\n"
        )
    metadata_prefix = _metadata_prefix(first_record)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="ai-memory-live-", suffix=".jsonl"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        os.close(descriptor)
        window = min(LIVE_TRANSCRIPT_TAIL_BYTES, MAX_LIVE_TRANSCRIPT_SCAN_BYTES)
        while True:
            payload, reached_start = _bounded_tail(
                source,
                size=size,
                window_bytes=window,
                metadata_prefix=metadata_prefix,
            )
            _write_private_slice(temporary, payload, durable=False)
            preview = previewer(temporary)
            if getattr(preview, "turns", ()) or reached_start:
                _write_private_slice(temporary, payload, durable=True)
                yield temporary, preview
                return
            if window >= MAX_LIVE_TRANSCRIPT_SCAN_BYTES:
                raise LiveTranscriptRejected(
                    "no durable signal found within the 16 MB live scan budget"
                )
            window = min(window * 2, MAX_LIVE_TRANSCRIPT_SCAN_BYTES)
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
        metadata = {
            "session_id": hook_input.get("session_id", ""),
            "cwd": cwd,
            "project": hook_input["project"],
            "timestamp": hook_input.get("timestamp", ""),
            "trigger": "session_end",
        }

        def previewer(path: Path):
            return parse_claude_transcript(
                path, metadata, limits={"max_turns": MAX_TURNS}
            )

        with bounded_transcript_slice(transcript_path, previewer) as selected:
            live_slice, preview = selected
            context = render_turns(preview)
            if len(context) > MAX_CONTEXT_CHARS:
                context = context[-MAX_CONTEXT_CHARS:]
                boundary = context.find("\n**")
                if boundary > 0:
                    context = context[boundary + 1 :]
            turn_count = len(preview.turns)
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
