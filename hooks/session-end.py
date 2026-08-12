"""Fast Claude SessionEnd adapter: normalize, enqueue, and return."""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from typing import Callable, Iterator, Literal


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
MAX_LIVE_TRANSCRIPT_SCAN_BYTES = 16_000_000
MAX_LIVE_JSONL_RECORD_BYTES = 500_000
SEMANTIC_SCAN_CHUNK_BYTES = 500_000
HOOK_WORK_BUDGET_SECONDS = 2.25
MIN_CAPTURE_REMAINING_SECONDS = 0.75


class LiveTranscriptRejected(ValueError):
    """A live transcript cannot be sliced without risking partial capture."""


class HookDeadlineExceeded(TimeoutError):
    """The internal hook budget expired before durable enqueue began."""


def run_process_until_deadline(
    command: list[str],
    *,
    input_text: str,
    deadline: float,
    clock: Callable[[], float],
    on_timeout: Callable[[], None] | None = None,
) -> str:
    """Run a killable child and leave margin beneath the hook host timeout."""
    remaining = deadline - clock()
    if remaining <= 0:
        if on_timeout is not None:
            on_timeout()
        raise HookDeadlineExceeded("live hook work budget exhausted")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        close_fds=True,
    )
    try:
        stdout, _ = process.communicate(input_text, timeout=remaining)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        if on_timeout is not None:
            on_timeout()
        raise HookDeadlineExceeded("capture child exceeded hook deadline") from error
    if process.returncode != 0:
        if on_timeout is not None:
            on_timeout()
        raise RuntimeError(f"capture child exited {process.returncode}")
    return stdout


def _queue_path(root: Path) -> Path:
    configured = os.environ.get("AI_MEMORY_QUEUE_PATH")
    return Path(configured).expanduser() if configured else root / "scripts" / "jobs.sqlite3"


def _snapshot_is_referenced(path: Path, database: Path) -> bool | None:
    """Return None when queue state cannot be inspected safely."""
    if not database.is_file():
        return False
    try:
        uri = database.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=0.05) as connection:
            connection.execute("PRAGMA busy_timeout = 50")
            row = connection.execute(
                "SELECT 1 FROM jobs WHERE source_path = ? LIMIT 1", (str(path),)
            ).fetchone()
        return row is not None
    except sqlite3.OperationalError as error:
        if "no such table" in str(error).lower():
            return False
        return None


def cleanup_uncommitted_capture(token: str, *, root: Path | None = None) -> None:
    """Remove only token-owned snapshots proven absent from the durable queue."""
    memory_root = _runtime_root() if root is None else root
    spool = memory_root / "scripts" / "spool"
    if not spool.is_dir():
        return
    database = _queue_path(memory_root)
    patterns = (
        f"capture-{token}-*.jsonl",
        f"failed-claude-{token}-*.jsonl",
        f"failed-codex-{token}-*.jsonl",
    )
    for pattern in patterns:
        for candidate in spool.glob(pattern):
            try:
                info = candidate.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            if hasattr(os, "getuid") and info.st_uid != os.getuid():
                continue
            if _snapshot_is_referenced(candidate, database) is False:
                candidate.unlink(missing_ok=True)


def enqueue_capture_with_deadline(
    hook_input: dict[str, object],
    *,
    source_agent: Literal["claude", "codex"],
    trigger: Literal["session_end", "pre_compact"],
    limits: dict[str, int],
    deadline: float,
    clock: Callable[[], float],
) -> dict[str, object]:
    """Enqueue in a child so a blocked SQLite/copy phase is host-bounded."""
    require_time_remaining(deadline, clock)
    token = secrets.token_hex(16)
    remaining = deadline - clock()
    request = {
        "hook_input": hook_input,
        "source_agent": source_agent,
        "trigger": trigger,
        "limits": limits,
        "capture_token": token,
        "budget_seconds": max(0.01, remaining - 0.15),
    }
    output = run_process_until_deadline(
        [sys.executable, str(Path(__file__).resolve()), "--capture-child"],
        input_text=json.dumps(request, separators=(",", ":")),
        deadline=deadline,
        clock=clock,
        on_timeout=lambda: cleanup_uncommitted_capture(token),
    )
    value = json.loads(output)
    if not isinstance(value, dict):
        raise ValueError("capture child returned invalid output")
    return value


def _capture_child_main() -> None:
    request = json.loads(sys.stdin.read())
    if not isinstance(request, dict):
        raise ValueError("capture child input must be an object")
    budget = request.get("budget_seconds")
    token = request.get("capture_token")
    if not isinstance(budget, (int, float)) or budget <= 0:
        raise ValueError("capture child budget must be positive")
    if not isinstance(token, str) or not token:
        raise ValueError("capture child token is required")
    outcome = enqueue_hook_input(
        request["hook_input"],
        source_agent=request["source_agent"],
        trigger=request["trigger"],
        limits=request["limits"],
        deadline=time.monotonic() + budget,
        monotonic=time.monotonic,
        capture_token=token,
    )
    sys.stdout.write(
        json.dumps({"status": outcome.status, "job_id": outcome.job_id})
    )


def require_time_remaining(
    deadline: float,
    clock: Callable[[], float],
    minimum_seconds: float = 0.0,
) -> None:
    if clock() + minimum_seconds >= deadline:
        raise HookDeadlineExceeded("live hook work budget exhausted")


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


def _validated_chunks_backward(
    source: Path,
    size: int,
    *,
    deadline: float,
    clock: Callable[[], float],
) -> Iterator[tuple[int, bytes]]:
    """Yield validated chunks while reconstructing records across boundaries."""
    end = size
    incomplete = b""
    with source.open("rb") as stream:
        while end > 0:
            require_time_remaining(deadline, clock, MIN_CAPTURE_REMAINING_SECONDS)
            start = max(0, end - SEMANTIC_SCAN_CHUNK_BYTES)
            stream.seek(start)
            data = stream.read(end - start) + incomplete
            if start:
                boundary = data.find(b"\n")
                if boundary < 0:
                    incomplete = data
                    if len(incomplete) > MAX_LIVE_JSONL_RECORD_BYTES:
                        raise LiveTranscriptRejected(
                            "live transcript contains an oversized JSONL record"
                        )
                    end = start
                    continue
                incomplete = data[: boundary + 1]
                if len(incomplete.rstrip(b"\r\n")) > MAX_LIVE_JSONL_RECORD_BYTES:
                    raise LiveTranscriptRejected(
                        "live transcript contains an oversized JSONL record"
                    )
                complete = data[boundary + 1 :]
                complete_start = start + boundary + 1
            else:
                complete = data
                complete_start = 0
                incomplete = b""
            validated = _validated_jsonl(complete)
            if validated:
                yield complete_start, validated
            end = start


def _encoded_record(record: dict[str, object]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _record_message(
    role: object,
    content: object,
    timestamp: object,
) -> bytes:
    record: dict[str, object] = {"message": {"role": role, "content": content}}
    if isinstance(timestamp, str) and timestamp:
        record["timestamp"] = timestamp
    return _encoded_record(record)


def _codex_record(payload: dict[str, object], timestamp: object) -> bytes:
    record: dict[str, object] = {"type": "response_item", "payload": payload}
    if isinstance(timestamp, str) and timestamp:
        record["timestamp"] = timestamp
    return _encoded_record(record)


def _tool_basename(name: object) -> str:
    if not isinstance(name, str):
        return ""
    normalized = name.replace("::", ".").split(".")[-1]
    if "__" in normalized:
        normalized = normalized.split("__")[-1]
    return normalized.lower()


def _completed_agent_message(payload: dict[str, object]) -> dict[str, object] | None:
    author = payload.get("author")
    recipient = payload.get("recipient")
    content = payload.get("content")
    if not isinstance(author, str) or not isinstance(recipient, str):
        return None
    if not isinstance(content, list):
        return None
    expected = {
        "Message Type": "FINAL_ANSWER",
        "Task name": recipient,
        "Sender": author,
    }
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "input_text":
            continue
        text = block.get("text")
        if not isinstance(text, str):
            continue
        header, marker, finding = text.partition("\nPayload:\n")
        fields: dict[str, str] = {}
        for line in header.splitlines():
            name, separator, value = line.partition(": ")
            if separator:
                fields[name] = value
        if marker and finding.strip() and fields == expected:
            return {
                "type": "agent_message",
                "author": author,
                "recipient": recipient,
                "content": [{"type": "input_text", "text": text}],
            }
    return None


def _could_contain_semantic_record(
    record: dict[str, object], source_agent: Literal["claude", "codex"]
) -> bool:
    if source_agent == "claude":
        message = record.get("message")
        role = message.get("role") if isinstance(message, dict) else record.get("role")
        return role in ("user", "assistant")
    if record.get("type") != "response_item":
        return False
    payload = record.get("payload")
    return isinstance(payload, dict) and payload.get("type") in {
        "message",
        "agent_message",
        "function_call",
        "custom_tool_call",
        "function_call_output",
        "custom_tool_call_output",
    }


def _semantic_compaction(
    records: list[tuple[int, dict[str, object]]],
    *,
    source_agent: Literal["claude", "codex"],
) -> bytes:
    """Keep only records that the shared parser can turn into durable signal.

    Call declarations are collected across the whole scanned transcript before
    any result is retained. Reused IDs therefore fail closed without persisting
    their potentially private output.
    """
    entries: dict[int, bytes] = {}
    groups: list[tuple[int, set[int]]] = []
    calls: dict[str, list[tuple[str, int, bytes]]] = {}
    results: dict[str, list[tuple[int, bytes]]] = {}

    def add_group(offset: int, encoded: bytes) -> None:
        entries[offset] = encoded
        groups.append((offset, {offset}))

    for offset, record in records:
        record_offset = offset * 1_000
        timestamp = record.get("timestamp")
        if source_agent == "claude":
            message = record.get("message", {})
            if not isinstance(message, dict):
                role = record.get("role")
                content = record.get("content")
            else:
                role = message.get("role")
                content = message.get("content")
            if role not in ("user", "assistant"):
                continue
            if isinstance(content, str):
                if content.strip():
                    add_group(record_offset, _record_message(role, content, timestamp))
                continue
            if not isinstance(content, list):
                continue
            text_blocks: list[object] = []
            for block_index, block in enumerate(content):
                item_offset = record_offset + block_index + 1
                if isinstance(block, str):
                    if block.strip():
                        text_blocks.append(block)
                    continue
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        text_blocks.append({"type": "text", "text": text})
                elif block_type == "tool_use":
                    call_id = block.get("id")
                    name = block.get("name")
                    if not isinstance(call_id, str) or not isinstance(name, str):
                        continue
                    sanitized: dict[str, object] = {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": block.get("input", {})
                        if name == "AskUserQuestion"
                        else {},
                    }
                    calls.setdefault(call_id, []).append(
                        (
                            name,
                            item_offset,
                            _record_message(role, [sanitized], timestamp),
                        )
                    )
                elif block_type == "tool_result":
                    call_id = block.get("tool_use_id")
                    if not isinstance(call_id, str):
                        continue
                    sanitized = {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": block.get("content"),
                    }
                    results.setdefault(call_id, []).append(
                        (
                            item_offset,
                            _record_message(role, [sanitized], timestamp),
                        )
                    )
            if text_blocks:
                add_group(record_offset, _record_message(role, text_blocks, timestamp))
        else:
            if record.get("type") != "response_item":
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            payload_type = payload.get("type")
            if payload_type == "message":
                role = payload.get("role")
                expected = "input_text" if role == "user" else "output_text"
                if role not in ("user", "assistant"):
                    continue
                content = payload.get("content")
                if not isinstance(content, list):
                    continue
                blocks = [
                    {"type": expected, "text": block.get("text")}
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == expected
                    and isinstance(block.get("text"), str)
                    and block["text"].strip()
                ]
                if blocks:
                    add_group(
                        record_offset,
                        _codex_record(
                            {"type": "message", "role": role, "content": blocks},
                            timestamp,
                        ),
                    )
            elif payload_type == "agent_message":
                completed = _completed_agent_message(payload)
                if completed is not None:
                    add_group(record_offset, _codex_record(completed, timestamp))
            elif payload_type in ("function_call", "custom_tool_call"):
                call_id = payload.get("call_id") or payload.get("id")
                if not isinstance(call_id, str) or not call_id:
                    continue
                name = _tool_basename(payload.get("name"))
                sanitized = {
                    "type": payload_type,
                    "call_id": call_id,
                    "name": payload.get("name", ""),
                    "arguments": "{}",
                }
                calls.setdefault(call_id, []).append(
                    (name, record_offset, _codex_record(sanitized, timestamp))
                )
            elif payload_type in (
                "function_call_output",
                "custom_tool_call_output",
            ):
                call_id = payload.get("call_id") or payload.get("id")
                if not isinstance(call_id, str) or not call_id:
                    continue
                sanitized = {
                    "type": payload_type,
                    "call_id": call_id,
                    "output": payload.get("output", payload.get("content")),
                }
                results.setdefault(call_id, []).append(
                    (record_offset, _codex_record(sanitized, timestamp))
                )

    claude_allowed = {"AskUserQuestion", "Agent", "Task"}
    codex_allowed = {
        "ask_user_question",
        "askuserquestion",
        "request_user_input",
        "get_agent_result",
        "join_agent",
        "wait_agent",
        "wait_for_agent",
    }
    for call_id, declarations in calls.items():
        allowed = [
            declaration
            for declaration in declarations
            if declaration[0]
            in (claude_allowed if source_agent == "claude" else codex_allowed)
        ]
        if not allowed:
            continue
        if len(declarations) != 1:
            offsets: set[int] = set()
            for _name, offset, encoded in declarations:
                entries[offset] = encoded
                offsets.add(offset)
            groups.append((min(offsets), offsets))
            continue
        name, call_offset, call_record = declarations[0]
        matching_results = sorted(
            result for result in results.get(call_id, []) if result[0] > call_offset
        )
        if source_agent == "claude" and name == "AskUserQuestion":
            entries[call_offset] = call_record
            groups.append((call_offset, {call_offset}))
        if not matching_results:
            continue
        result_offset, result_record = matching_results[0]
        entries[call_offset] = call_record
        entries[result_offset] = result_record
        groups.append((call_offset, {call_offset, result_offset}))

    total = sum(len(value) for value in entries.values())
    for _group_offset, offsets in sorted(groups):
        if total <= MAX_LIVE_TRANSCRIPT_SCAN_BYTES:
            break
        for offset in offsets:
            removed = entries.pop(offset, None)
            if removed is not None:
                total -= len(removed)
    if total > MAX_LIVE_TRANSCRIPT_SCAN_BYTES:
        raise LiveTranscriptRejected(
            "compacted semantic snapshot exceeds the 16 MB safety bound"
        )
    return b"".join(entries[offset] for offset in sorted(entries))


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
    *,
    source_agent: Literal["claude", "codex"],
    deadline: float,
    clock: Callable[[], float],
) -> Iterator[tuple[Path, object]]:
    """Build a compact semantic snapshot while scanning backward under deadline.

    Raw routine-only records are discarded as each chunk is read, so the search
    is not limited by a fixed raw tail size. If a deadline prevents reaching
    the file start, selection fails closed; the original transcript remains the
    recovery source. The shared parser applies its final-30-turn contract.
    """
    require_time_remaining(deadline, clock)
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
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        os.close(descriptor)
        semantic_records: list[tuple[int, dict[str, object]]] = []
        for start, chunk in _validated_chunks_backward(
            source, size, deadline=deadline, clock=clock
        ):
            relative = 0
            for line in chunk.splitlines():
                record = json.loads(line)
                if _could_contain_semantic_record(record, source_agent):
                    semantic_records.append((start + relative, record))
                relative += len(line) + 1
        compacted = metadata_prefix + _semantic_compaction(
            semantic_records, source_agent=source_agent
        )
        if len(compacted) > MAX_LIVE_TRANSCRIPT_SCAN_BYTES:
            raise LiveTranscriptRejected(
                "compacted semantic snapshot exceeds the 16 MB safety bound"
            )
        _write_private_slice(temporary, compacted, durable=False)
        require_time_remaining(
            deadline, clock, MIN_CAPTURE_REMAINING_SECONDS
        )
        preview = previewer(temporary)
        _write_private_slice(temporary, compacted, durable=True)
        require_time_remaining(
            deadline, clock, MIN_CAPTURE_REMAINING_SECONDS
        )
        yield temporary, preview
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


def _resolve_user_tty(
    *,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> str | None:
    """Find the terminal while the parent Claude process is still alive."""
    tty_deadline = (
        min(deadline - MIN_CAPTURE_REMAINING_SECONDS, clock() + 0.15)
        if deadline is not None
        else None
    )
    pid = os.getpid()
    for _ in range(10):
        timeout = 2.0
        if tty_deadline is not None:
            remaining = tty_deadline - clock()
            if remaining <= 0:
                return None
            timeout = min(timeout, remaining)
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "ppid=,tty="],
                capture_output=True,
                text=True,
                timeout=timeout,
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


def main(clock: Callable[[], float] = time.monotonic) -> None:
    deadline = clock() + HOOK_WORK_BUDGET_SECONDS
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

        with bounded_transcript_slice(
            transcript_path,
            previewer,
            source_agent="claude",
            deadline=deadline,
            clock=clock,
        ) as selected:
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

            tty_path = _resolve_user_tty(deadline=deadline, clock=clock)
            if tty_path:
                os.environ["CLAUDE_MEMORY_TTY"] = tty_path
            require_time_remaining(
                deadline, clock, MIN_CAPTURE_REMAINING_SECONDS
            )
            capture_input = dict(hook_input)
            capture_input["transcript_path"] = str(live_slice)
            outcome = enqueue_capture_with_deadline(
                capture_input,
                source_agent="claude",
                trigger="session_end",
                limits={"max_turns": MAX_TURNS, "max_chars": MAX_CONTEXT_CHARS},
                deadline=deadline,
                clock=clock,
            )
        logger.info("capture %s for session %s", outcome.get("status"), outcome.get("job_id"))
    except Exception as error:
        # Hooks are advisory. A capture failure must never block the host agent.
        logger.error("capture failed: %s", error)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--capture-child":
        _capture_child_main()
    else:
        main()
