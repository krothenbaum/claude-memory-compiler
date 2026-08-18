"""Fast Claude SessionEnd adapter: normalize, enqueue, and return."""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import sqlite3
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal


# This must precede imports of the capture/queue modules: those modules can
# create runtime state when their public entry points are used.
if os.environ.get("AI_MEMORY_INTERNAL_JOB") == "1" or "CLAUDE_INVOKED_BY" in os.environ:
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.capture import enqueue_hook_input
from scripts.hook_logging import (
    classify_transcript_path,
    configure_hook_logger,
    log_hook_event,
)
from scripts.transcripts import parse_claude_transcript, render_turns
from scripts.utils import (
    open_secure_runtime_file,
    validate_secure_runtime_file,
)


MAX_TURNS = 30
MAX_CONTEXT_CHARS = 15_000
MIN_TURNS_TO_FLUSH = 1
MAX_LIVE_TRANSCRIPT_SCAN_BYTES = 16_000_000
MAX_LIVE_JSONL_RECORD_BYTES = 500_000
SEMANTIC_SCAN_CHUNK_BYTES = 500_000
MAX_SEMANTIC_CANDIDATE_RECORDS = 4_096
MAX_SEMANTIC_CANDIDATE_BYTES = 16_000_000
MAX_FALLBACK_SELF_CONTAINED_TURNS = MAX_TURNS * 2
RECORD_OFFSET_SCALE = 1_000_000
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
    return configure_hook_logger(
        "ai-memory-session-end", "session-end", _runtime_root()
    )


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


def _is_self_contained_turn(
    record: dict[str, object], source_agent: Literal["claude", "codex"]
) -> bool:
    """Whether this record yields a turn without any earlier call context."""
    if source_agent == "codex":
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return False
        if payload.get("type") == "agent_message":
            return _completed_agent_message(payload) is not None
        if payload.get("type") != "message":
            return False
        role = payload.get("role")
        expected = "input_text" if role == "user" else "output_text"
        content = payload.get("content")
        return role in ("user", "assistant") and isinstance(content, list) and any(
            isinstance(block, dict)
            and block.get("type") == expected
            and isinstance(block.get("text"), str)
            and bool(block["text"].strip())
            for block in content
        )

    message = record.get("message")
    role = message.get("role") if isinstance(message, dict) else record.get("role")
    content = (
        message.get("content") if isinstance(message, dict) else record.get("content")
    )
    if role not in ("user", "assistant"):
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    if any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    ):
        return False
    return any(
        (isinstance(block, str) and bool(block.strip()))
        or (
            isinstance(block, dict)
            and (
                (
                    block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    and bool(block["text"].strip())
                )
                or (
                    block.get("type") == "tool_use"
                    and block.get("name") == "AskUserQuestion"
                )
            )
        )
        for block in content
    )


def _dependency_ids(
    record: dict[str, object], source_agent: Literal["claude", "codex"]
) -> tuple[set[str], set[str]]:
    """Return call IDs and result IDs without retaining their payload bodies."""
    if source_agent == "codex":
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return set(), set()
        call_id = payload.get("call_id") or payload.get("id")
        if not isinstance(call_id, str) or not call_id:
            return set(), set()
        if payload.get("type") in {"function_call", "custom_tool_call"}:
            return {call_id}, set()
        if payload.get("type") in {
            "function_call_output",
            "custom_tool_call_output",
        }:
            return set(), {call_id}
        return set(), set()

    message = record.get("message")
    content = message.get("content") if isinstance(message, dict) else record.get(
        "content"
    )
    if not isinstance(content, list):
        return set(), set()
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
            call_ids.add(block["id"])
        elif block.get("type") == "tool_result" and isinstance(
            block.get("tool_use_id"), str
        ):
            result_ids.add(block["tool_use_id"])
    return call_ids, result_ids


def _json_output(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _has_text(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(
            _has_text(block.get("text"))
            if isinstance(block, dict) and block.get("type") == "text"
            else _has_text(block)
            for block in value
        )
    return False


def _answer_text(value: object) -> str:
    if isinstance(value, dict) and "answers" in value:
        return _answer_text(value["answers"])
    if isinstance(value, list):
        return ", ".join(filter(None, (_answer_text(item) for item in value)))
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return ""


def _choice_text(value: object) -> str:
    if isinstance(value, dict) and "answers" in value:
        return _choice_text(value["answers"])
    if isinstance(value, list):
        return ", ".join(
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        )
    return value.strip() if isinstance(value, str) else ""


def _completed_output(value: object) -> bool:
    parsed = _json_output(value)
    if isinstance(parsed, dict):
        if parsed.get("completed") not in (None, False) and _answer_text(
            parsed.get("completed")
        ).strip():
            return True
        if str(parsed.get("status", "")).lower() in {
            "complete",
            "completed",
            "done",
            "finished",
        }:
            return any(
                bool(_answer_text(parsed.get(key)).strip())
                for key in ("message", "result", "output", "final_answer", "finding")
            )
        return any(_completed_output(child) for child in parsed.values())
    if isinstance(parsed, list):
        return any(_completed_output(child) for child in parsed)
    if isinstance(parsed, str):
        lowered = parsed.lower()
        return "<subagent_notification>" in lowered and "completed" in lowered
    return False


def _result_is_durable(
    source_agent: Literal["claude", "codex"], name: str, encoded: bytes
) -> bool:
    record = json.loads(encoded)
    if source_agent == "claude":
        content = record["message"]["content"][0].get("content")
        if name == "AskUserQuestion":
            return _has_text(content)
        return _has_text(content) and not (
            isinstance(content, str) and content.startswith("Async agent launched")
        )
    output = record["payload"].get("output")
    parsed = _json_output(output)
    if name in {"ask_user_question", "askuserquestion", "request_user_input"}:
        if not isinstance(parsed, dict) or not isinstance(parsed.get("answers"), dict):
            return False
        return any(_choice_text(value) for value in parsed["answers"].values())
    return _completed_output(parsed)


def _merge_claude_entries(
    entries: dict[int, bytes],
    *,
    deadline: float,
    clock: Callable[[], float],
) -> dict[int, bytes]:
    merged: dict[int, dict[str, object]] = {}
    for entry_index, (offset, encoded) in enumerate(sorted(entries.items())):
        if entry_index % 64 == 0:
            require_time_remaining(deadline, clock, MIN_CAPTURE_REMAINING_SECONDS)
        parent = (offset // RECORD_OFFSET_SCALE) * RECORD_OFFSET_SCALE
        record = json.loads(encoded)
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        blocks = content if isinstance(content, list) else [content]
        if parent not in merged:
            merged[parent] = {
                "message": {"role": message.get("role"), "content": []}
            }
            if isinstance(record.get("timestamp"), str):
                merged[parent]["timestamp"] = record["timestamp"]
        merged_message = merged[parent]["message"]
        assert isinstance(merged_message, dict)
        merged_content = merged_message["content"]
        assert isinstance(merged_content, list)
        merged_content.extend(block for block in blocks if block is not None)
    return {offset: _encoded_record(record) for offset, record in merged.items()}


def _semantic_compaction(
    records: list[tuple[int, dict[str, object]]],
    *,
    source_agent: Literal["claude", "codex"],
    deadline: float,
    clock: Callable[[], float],
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

    for record_index, (offset, record) in enumerate(records):
        if record_index % 64 == 0:
            require_time_remaining(deadline, clock, MIN_CAPTURE_REMAINING_SECONDS)
        record_offset = offset * RECORD_OFFSET_SCALE
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
            for block_index, block in enumerate(content):
                item_offset = record_offset + block_index + 1
                if isinstance(block, str):
                    if block.strip():
                        add_group(
                            item_offset,
                            _record_message(role, [block], timestamp),
                        )
                    continue
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        add_group(
                            item_offset,
                            _record_message(
                                role,
                                [{"type": "text", "text": text}],
                                timestamp,
                            ),
                        )
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
        require_time_remaining(deadline, clock, MIN_CAPTURE_REMAINING_SECONDS)
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
        durable_result = next(
            (
                result
                for result in matching_results
                if _result_is_durable(source_agent, name, result[1])
            ),
            None,
        )
        if durable_result is None:
            continue
        result_offset, result_record = durable_result
        entries[call_offset] = call_record
        entries[result_offset] = result_record
        groups.append((call_offset, {call_offset, result_offset}))

    if source_agent == "claude":
        entries = _merge_claude_entries(
            entries, deadline=deadline, clock=clock
        )
        groups = [(offset, {offset}) for offset in entries]
    total = sum(len(value) for value in entries.values())
    for group_index, (_group_offset, offsets) in enumerate(sorted(groups)):
        if group_index % 64 == 0:
            require_time_remaining(deadline, clock, MIN_CAPTURE_REMAINING_SECONDS)
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


def _write_private_slice(descriptor: int, payload: bytes, *, durable: bool) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("incomplete live transcript slice write")
        remaining = remaining[written:]
    if durable:
        os.fsync(descriptor)


@contextmanager
def bounded_transcript_slice(
    source: Path,
    previewer: Callable[[Path], object],
    *,
    source_agent: Literal["claude", "codex"],
    memory_root: Path | str,
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
    with open_secure_runtime_file(memory_root) as (temporary, descriptor):
        semantic_records: list[tuple[int, dict[str, object]]] = []
        candidate_bytes = 0
        selection_complete = False
        self_contained_kept = 0
        dependency_call_ids: set[str] = set()
        for start, chunk in _validated_chunks_backward(
            source, size, deadline=deadline, clock=clock
        ):
            positioned: list[tuple[int, bytes]] = []
            relative = 0
            for line in chunk.splitlines():
                positioned.append((start + relative, line))
                relative += len(line) + 1
            for offset, line in reversed(positioned):
                require_time_remaining(
                    deadline, clock, MIN_CAPTURE_REMAINING_SECONDS
                )
                record = json.loads(line)
                if _could_contain_semantic_record(record, source_agent):
                    self_contained = _is_self_contained_turn(record, source_agent)
                    call_ids, result_ids = _dependency_ids(record, source_agent)
                    relevant_dependency = bool(
                        result_ids
                        or call_ids.intersection(dependency_call_ids)
                    )
                    if result_ids:
                        dependency_call_ids.update(result_ids)
                    if (
                        self_contained
                        and self_contained_kept
                        >= MAX_FALLBACK_SELF_CONTAINED_TURNS
                        and not relevant_dependency
                    ):
                        continue
                    if not self_contained and not relevant_dependency:
                        continue
                    semantic_records.append((offset, record))
                    candidate_bytes += len(line) + 1
                    if self_contained:
                        self_contained_kept += 1
                    if (
                        len(semantic_records) > MAX_SEMANTIC_CANDIDATE_RECORDS
                        or candidate_bytes > MAX_SEMANTIC_CANDIDATE_BYTES
                    ):
                        raise LiveTranscriptRejected(
                            "semantic candidate state exceeds live safety bound"
                        )
                    if (
                        not dependency_call_ids
                        and len(semantic_records) >= MAX_TURNS
                        and all(
                            _is_self_contained_turn(candidate, source_agent)
                            for _candidate_offset, candidate in semantic_records[
                                :MAX_TURNS
                            ]
                        )
                    ):
                        selection_complete = True
                        break
            if selection_complete:
                break
        compacted = metadata_prefix + _semantic_compaction(
            semantic_records,
            source_agent=source_agent,
            deadline=deadline,
            clock=clock,
        )
        if len(compacted) > MAX_LIVE_TRANSCRIPT_SCAN_BYTES:
            raise LiveTranscriptRejected(
                "compacted semantic snapshot exceeds the 16 MB safety bound"
            )
        _write_private_slice(descriptor, compacted, durable=False)
        validate_secure_runtime_file(temporary, descriptor)
        require_time_remaining(deadline, clock, MIN_CAPTURE_REMAINING_SECONDS)
        preview = previewer(temporary)
        validate_secure_runtime_file(temporary, descriptor)
        _write_private_slice(descriptor, compacted, durable=True)
        validate_secure_runtime_file(temporary, descriptor)
        require_time_remaining(deadline, clock, MIN_CAPTURE_REMAINING_SECONDS)
        yield temporary, preview


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
    except (json.JSONDecodeError, ValueError, EOFError):
        log_hook_event(
            logger,
            logging.ERROR,
            "malformed_input",
            "failed to parse hook input",
            source_agent="claude",
        )
        return

    transcript_value = hook_input.get("transcript_path")
    if not isinstance(transcript_value, str) or not transcript_value:
        log_hook_event(
            logger,
            logging.ERROR,
            "transcript_missing",
            "hook input did not include a transcript",
            source_agent="claude",
            session_id=hook_input.get("session_id"),
        )
        return
    transcript_path = Path(transcript_value).expanduser()
    transcript_event = classify_transcript_path(transcript_path)
    if transcript_event is not None:
        log_hook_event(
            logger,
            logging.ERROR,
            transcript_event,
            (
                "transcript is missing"
                if transcript_event == "transcript_missing"
                else "transcript is unreadable"
            ),
            source_agent="claude",
            session_id=hook_input.get("session_id"),
        )
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
            memory_root=_runtime_root(),
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
        log_hook_event(
            logger,
            logging.INFO,
            "capture_succeeded",
            f"capture {outcome.get('status')}",
            source_agent="claude",
            session_id=hook_input.get("session_id"),
        )
    except Exception as error:
        # Hooks are advisory. A capture failure must never block the host agent.
        event = (
            "queue_unavailable"
            if isinstance(error, sqlite3.Error)
            else "capture_failed"
        )
        log_hook_event(
            logger,
            logging.ERROR,
            event,
            (
                "queue unavailable during capture"
                if event == "queue_unavailable"
                else "capture failed"
            ),
            source_agent="claude",
            session_id=hook_input.get("session_id"),
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--capture-child":
        _capture_child_main()
    else:
        main()
