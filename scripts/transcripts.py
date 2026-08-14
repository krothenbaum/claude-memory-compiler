"""Normalize Claude and Codex transcripts into durable conversation context."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Iterable, Iterator, Literal, Mapping, Sequence


AgentName = Literal["claude", "codex"]
TriggerName = Literal["session_end", "pre_compact", "historical"]
TurnRole = Literal["user", "assistant"]
TurnKind = Literal["message", "decision", "subagent_finding"]

DEFAULT_TIMESTAMP = "1970-01-01T00:00:00Z"
MAX_FINDING_CHARS = 2_000
MAX_TRANSCRIPT_BYTES = 64 * 1024 * 1024
MAX_TRANSCRIPT_RECORD_BYTES = 2 * 1024 * 1024
MAX_TRANSCRIPT_RECORDS = 200_000

CLAUDE_SUBAGENT_TOOLS = {"Agent", "Task"}
CODEX_DECISION_TOOLS = {"ask_user_question", "askuserquestion", "request_user_input"}
CODEX_SUBAGENT_RESULT_TOOLS = {
    "get_agent_result",
    "join_agent",
    "wait_agent",
    "wait_for_agent",
}


@dataclass(frozen=True)
class Turn:
    role: TurnRole
    text: str
    kind: TurnKind = "message"
    timestamp: str | None = None


@dataclass(frozen=True)
class NormalizedSession:
    agent: AgentName
    session_id: str
    project: str
    cwd: str
    timestamp: str
    trigger: TriggerName
    turns: tuple[Turn, ...]
    source_path: str
    source_hash: str


class UnsafeTranscriptSource(ValueError):
    """A historical transcript could not be read as one confined snapshot."""


@dataclass(frozen=True)
class TranscriptSnapshot:
    path: Path
    records: tuple[dict, ...]
    size: int
    mtime_ns: int


def _safe_owner(info: os.stat_result) -> bool:
    return not hasattr(os, "getuid") or info.st_uid == os.getuid()


def _stable_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


_USE_DIR_FD = (
    os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and bool(getattr(os, "O_NOFOLLOW", 0))
)


def _link_or_reparse(info: os.stat_result) -> bool:
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse and attributes & reparse)


def _validate_directory(info: os.stat_result, label: str) -> None:
    if _link_or_reparse(info) or not stat.S_ISDIR(info.st_mode) or not _safe_owner(info):
        raise UnsafeTranscriptSource(f"unsafe transcript {label}")


def _validate_file(info: os.stat_result, max_bytes: int) -> None:
    if _link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
        raise UnsafeTranscriptSource("transcript must be a regular file")
    if not _safe_owner(info):
        raise UnsafeTranscriptSource("transcript has an unsafe owner")
    if info.st_nlink != 1:
        raise UnsafeTranscriptSource("transcript must not be hard-linked")
    if info.st_size > max_bytes:
        raise UnsafeTranscriptSource("transcript exceeds byte limit")


def read_transcript_snapshot(
    path: Path | str,
    *,
    root: Path | str,
    max_bytes: int = MAX_TRANSCRIPT_BYTES,
    max_record_bytes: int = MAX_TRANSCRIPT_RECORD_BYTES,
    max_records: int = MAX_TRANSCRIPT_RECORDS,
) -> TranscriptSnapshot:
    """Read one owner-controlled regular JSONL file exactly once under ``root``."""
    if max_bytes <= 0 or max_record_bytes <= 0 or max_records <= 0:
        raise ValueError("transcript limits must be positive")
    root_path = Path(root).expanduser().absolute()
    transcript_path = Path(path).expanduser().absolute()
    try:
        relative = transcript_path.relative_to(root_path)
    except ValueError as exc:
        raise UnsafeTranscriptSource("transcript is outside its discovery root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise UnsafeTranscriptSource("invalid transcript path")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    fallback_paths: list[tuple[Path, tuple[int, int, int, int, int]]] = []
    try:
        if _USE_DIR_FD:
            directory = os.open(root_path, directory_flags)
            descriptors.append(directory)
            _validate_directory(os.fstat(directory), "root")
            for component in relative.parts[:-1]:
                directory = os.open(component, directory_flags, dir_fd=directory)
                descriptors.append(directory)
                _validate_directory(os.fstat(directory), "directory")
            descriptor = os.open(relative.parts[-1], file_flags, dir_fd=directory)
        else:
            current_path = root_path
            info = os.lstat(current_path)
            _validate_directory(info, "root")
            fallback_paths.append((current_path, _stable_identity(info)))
            for component in relative.parts[:-1]:
                current_path = current_path / component
                info = os.lstat(current_path)
                _validate_directory(info, "directory")
                fallback_paths.append((current_path, _stable_identity(info)))
            candidate_info = os.lstat(transcript_path)
            _validate_file(candidate_info, max_bytes)
            descriptor = os.open(transcript_path, file_flags)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        _validate_file(before, max_bytes)
        if not _USE_DIR_FD and (
            before.st_dev != candidate_info.st_dev or before.st_ino != candidate_info.st_ino
        ):
            raise UnsafeTranscriptSource("transcript changed before it was opened")

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise UnsafeTranscriptSource("transcript exceeds byte limit")

        after = os.fstat(descriptor)
        current = (
            os.stat(relative.parts[-1], dir_fd=directory, follow_symlinks=False)
            if _USE_DIR_FD
            else os.lstat(transcript_path)
        )
        if (
            _stable_identity(before) != _stable_identity(after)
            or before.st_dev != current.st_dev
            or before.st_ino != current.st_ino
        ):
            raise UnsafeTranscriptSource("transcript changed while being read")
        for ancestor, identity in fallback_paths:
            if _stable_identity(os.lstat(ancestor)) != identity:
                raise UnsafeTranscriptSource("transcript directory changed while being read")
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as exc:
        if isinstance(exc, UnsafeTranscriptSource):
            raise
        raise UnsafeTranscriptSource("transcript could not be opened safely") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)

    records: list[dict] = []
    try:
        for raw_line in b"".join(chunks).splitlines():
            if not raw_line.strip():
                continue
            if len(raw_line) > max_record_bytes:
                raise UnsafeTranscriptSource("transcript record exceeds byte limit")
            if len(records) >= max_records:
                raise UnsafeTranscriptSource("transcript exceeds record limit")
            record = json.loads(raw_line.decode("utf-8"))
            if not isinstance(record, dict):
                raise UnsafeTranscriptSource("transcript record must be an object")
            records.append(record)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UnsafeTranscriptSource("transcript is not valid JSONL") from exc
    return TranscriptSnapshot(
        transcript_path,
        tuple(records),
        before.st_size,
        before.st_mtime_ns,
    )


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def consistent_codex_session_meta(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    metas: list[dict[str, object]] = []
    for record in records:
        payload = record.get("payload")
        if record.get("type") == "session_meta" and isinstance(payload, dict):
            metas.append(dict(payload))
    if not metas:
        return None
    identity = tuple(metas[0].get(field) for field in ("id", "cwd", "timestamp"))
    if any(
        tuple(meta.get(field) for field in ("id", "cwd", "timestamp")) != identity
        for meta in metas[1:]
    ):
        return None
    return metas[0]


def read_codex_session_meta(path: Path | str) -> dict[str, object] | None:
    """Return the first well-formed Codex ``session_meta`` payload.

    Historical discovery uses this small, side-effect-free probe to distinguish
    rollouts from malformed or unrelated JSONL files before doing full parsing.
    """
    return consistent_codex_session_meta(tuple(_read_jsonl(Path(path))))


def codex_transcript_is_well_formed(path: Path | str) -> bool:
    """Return false when any non-empty JSONL record is invalid or non-object."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    return False
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return True


def _nonempty_string(value: object) -> str:
    return value if isinstance(value, str) and value else ""


def _project_name(metadata: Mapping[str, object], cwd: str) -> str:
    explicit = _nonempty_string(metadata.get("project"))
    if explicit:
        return explicit
    return Path(cwd).name or "unknown"


def _trigger(metadata: Mapping[str, object]) -> TriggerName:
    value = metadata.get("trigger")
    if value in ("session_end", "pre_compact", "historical"):
        return value
    return "historical"


def _canonical_hash(
    *,
    agent: AgentName,
    session_id: str,
    project: str,
    cwd: str,
    turns: Sequence[Turn],
) -> str:
    """Hash durable identity and content, excluding delivery-only metadata."""
    canonical = {
        "agent": agent,
        "cwd": cwd,
        "project": project,
        "session_id": session_id,
        "turns": [asdict(turn) for turn in turns],
    }
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _make_session(
    *,
    agent: AgentName,
    path: Path,
    metadata: Mapping[str, object],
    turns: Sequence[Turn],
    fallback_session_id: str = "",
    fallback_cwd: str = "",
    fallback_timestamp: str = "",
) -> NormalizedSession:
    session_id = (
        _nonempty_string(metadata.get("session_id"))
        or fallback_session_id
        or path.stem
        or "unknown"
    )
    cwd = _nonempty_string(metadata.get("cwd")) or fallback_cwd
    project = _project_name(metadata, cwd)
    timestamp = (
        _nonempty_string(metadata.get("timestamp"))
        or fallback_timestamp
        or DEFAULT_TIMESTAMP
    )
    normalized_turns = tuple(turns)
    return NormalizedSession(
        agent=agent,
        session_id=session_id,
        project=project,
        cwd=cwd,
        timestamp=timestamp,
        trigger=_trigger(metadata),
        turns=normalized_turns,
        source_path=str(path),
        source_hash=_canonical_hash(
            agent=agent,
            session_id=session_id,
            project=project,
            cwd=cwd,
            turns=normalized_turns,
        ),
    )


def _limit_value(limits: object, name: str) -> int | None:
    if isinstance(limits, Mapping):
        value = limits.get(name)
    else:
        value = getattr(limits, name, None)
    if not isinstance(value, int):
        return None
    if value > 0 or (name == "max_chars" and value == 0):
        return value
    return None


def _render_turn(turn: Turn) -> str:
    label = "User" if turn.role == "user" else "Assistant"
    return f"**{label}:** {turn.text}\n"


def _turn_accumulator(limits: object) -> list[Turn] | deque[Turn]:
    max_turns = _limit_value(limits, "max_turns") if limits is not None else None
    return deque(maxlen=max_turns) if max_turns is not None else []


def _apply_limits(turns: Iterable[Turn], limits: object) -> tuple[Turn, ...]:
    limited = list(turns)
    max_turns = _limit_value(limits, "max_turns")
    if max_turns is not None:
        limited = limited[-max_turns:]

    max_chars = _limit_value(limits, "max_chars")
    if max_chars is None:
        return tuple(limited)
    if max_chars == 0:
        return ()

    while len(limited) > 1 and len(_render_turns(limited)) > max_chars:
        limited.pop(0)

    if limited and len(_render_turns(limited)) > max_chars:
        turn = limited[0]
        label = "User" if turn.role == "user" else "Assistant"
        framing_chars = len(f"**{label}:** \n")
        available = max(0, max_chars - framing_chars)
        if available == 0:
            return ()
        limited[0] = Turn(
            role=turn.role,
            text=turn.text[-available:],
            kind=turn.kind,
            timestamp=turn.timestamp,
        )
    return tuple(limited)


def _tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(_nonempty_string(block.get("text")))
    return "\n".join(parts)


def _remember_call(calls: dict[str, str | None], call_id: str, name: str) -> None:
    if call_id in calls:
        calls[call_id] = None
    else:
        calls[call_id] = name


def _duplicate_call_ids(call_ids: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for call_id in call_ids:
        if call_id in seen:
            duplicates.add(call_id)
        else:
            seen.add(call_id)
    return duplicates


def _claude_call_ids(records: Iterable[Mapping[str, object]]) -> Iterator[str]:
    for record in records:
        message = record.get("message", {})
        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
        else:
            role = record.get("role", "")
            content = record.get("content", "")
        if role not in ("user", "assistant") or not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                call_id = _nonempty_string(block.get("id"))
                if call_id:
                    yield call_id


def _render_ask_question(tool_input: object) -> str:
    questions = tool_input.get("questions", []) if isinstance(tool_input, dict) else []
    if not isinstance(questions, list):
        questions = []
    lines: list[str] = []
    for question_data in questions:
        if not isinstance(question_data, dict):
            continue
        header = _nonempty_string(question_data.get("header")).strip()
        question = _nonempty_string(question_data.get("question")).strip()
        if not question:
            continue
        raw_options = question_data.get("options", [])
        if not isinstance(raw_options, list):
            raw_options = []
        options = [
            _nonempty_string(option.get("label"))
            for option in raw_options
            if isinstance(option, dict) and _nonempty_string(option.get("label"))
        ]
        prefix = f"({header}) " if header else ""
        line = f"- {prefix}{question}".rstrip()
        if options:
            line += " | options: " + ", ".join(options)
        lines.append(line)
    if not lines:
        return ""
    return "\n".join(["[Decision requested]", *lines])


def _claude_tool_result(tool_name: str, content: object) -> tuple[str, TurnKind] | None:
    text = _tool_result_text(content).strip()
    if not text:
        return None
    if tool_name == "AskUserQuestion":
        return f"[Decision made] {text}", "decision"
    if tool_name in CLAUDE_SUBAGENT_TOOLS:
        if text.startswith("Async agent launched"):
            return None
        if len(text) > MAX_FINDING_CHARS:
            text = text[:MAX_FINDING_CHARS] + " …[truncated]"
        return f"[Subagent result] {text}", "subagent_finding"
    return None


def parse_claude_transcript(
    path: Path | str,
    metadata: Mapping[str, object],
    limits: object = None,
    *,
    records: Sequence[Mapping[str, object]] | None = None,
) -> NormalizedSession:
    """Normalize Claude JSONL while retaining only durable conversation signal."""
    transcript_path = Path(path)
    source_records = tuple(_read_jsonl(transcript_path)) if records is None else tuple(records)
    turns = _turn_accumulator(limits)
    tool_names: dict[str, str | None] = {
        call_id: None
        for call_id in _duplicate_call_ids(_claude_call_ids(source_records))
    }
    fallback_session_id = ""
    fallback_cwd = ""
    fallback_timestamp = ""

    for record in source_records:
        if not fallback_session_id:
            fallback_session_id = (
                _nonempty_string(record.get("sessionId"))
                or _nonempty_string(record.get("session_id"))
            )
        if not fallback_cwd:
            fallback_cwd = _nonempty_string(record.get("cwd"))
        if not fallback_timestamp:
            fallback_timestamp = _nonempty_string(record.get("timestamp"))
        message = record.get("message", {})
        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
        else:
            role = record.get("role", "")
            content = record.get("content", "")
        if role not in ("user", "assistant"):
            continue

        parts: list[str] = []
        kind: TurnKind = "message"
        if isinstance(content, str):
            if content.strip():
                parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    if block.strip():
                        parts.append(block)
                    continue
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = _nonempty_string(block.get("text"))
                    if text.strip():
                        parts.append(text)
                elif block_type == "tool_use":
                    tool_id = _nonempty_string(block.get("id"))
                    tool_name = _nonempty_string(block.get("name"))
                    if tool_id:
                        _remember_call(tool_names, tool_id, tool_name)
                    if tool_name == "AskUserQuestion":
                        rendered = _render_ask_question(block.get("input", {}))
                        if rendered:
                            parts.append(rendered)
                            kind = "decision"
                elif block_type == "tool_result":
                    tool_id = _nonempty_string(block.get("tool_use_id"))
                    tool_name = tool_names.get(tool_id)
                    rendered = (
                        _claude_tool_result(tool_name, block.get("content"))
                        if tool_name is not None
                        else None
                    )
                    if rendered is not None:
                        tool_names[tool_id] = None
                        text, result_kind = rendered
                        parts.append(text)
                        kind = result_kind

        text = "\n".join(part for part in parts if part and part.strip()).strip()
        if text:
            turns.append(
                Turn(
                    role=role,
                    text=text,
                    kind=kind,
                    timestamp=_nonempty_string(record.get("timestamp")) or None,
                )
            )

    return _make_session(
        agent="claude",
        path=transcript_path,
        metadata=metadata,
        turns=_apply_limits(turns, limits) if limits is not None else turns,
        fallback_session_id=fallback_session_id,
        fallback_cwd=fallback_cwd,
        fallback_timestamp=fallback_timestamp,
    )


def _tool_basename(name: str) -> str:
    normalized = name.replace("::", ".").split(".")[-1]
    if "__" in normalized:
        normalized = normalized.split("__")[-1]
    return normalized.lower()


def _json_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


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
        choices = [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
        return ", ".join(choices)
    if isinstance(value, str):
        return value.strip()
    return ""


def _render_codex_decision(output: object) -> str:
    parsed = _json_value(output)
    if isinstance(parsed, dict) and isinstance(parsed.get("answers"), dict):
        answers = []
        for question, value in parsed["answers"].items():
            answer = _choice_text(value)
            if answer:
                answers.append(f"{question}: {answer}")
        if answers:
            return "[Decision made] " + "; ".join(answers)
    return ""


def _completed_finding(value: object) -> str:
    parsed = _json_value(value)
    if isinstance(parsed, dict):
        completed = parsed.get("completed")
        if completed not in (None, False):
            text = _answer_text(completed).strip()
            if text:
                return text
        status = _nonempty_string(parsed.get("status")).lower()
        if status in {"complete", "completed", "done", "finished"}:
            for key in ("message", "result", "output", "final_answer", "finding"):
                text = _answer_text(parsed.get(key)).strip()
                if text:
                    return text
        for child in parsed.values():
            finding = _completed_finding(child)
            if finding:
                return finding
    elif isinstance(parsed, list):
        for child in parsed:
            finding = _completed_finding(child)
            if finding:
                return finding
    elif isinstance(parsed, str):
        lowered = parsed.lower()
        if "<subagent_notification>" in lowered and "completed" in lowered:
            return parsed
    return ""


def _parse_agent_message_header(header: str) -> dict[str, str] | None:
    expected_names = ("Message Type", "Task name", "Sender")
    lines = header.split("\n")
    if len(lines) != len(expected_names):
        return None
    fields: dict[str, str] = {}
    for line, expected_name in zip(lines, expected_names, strict=True):
        name, separator, value = line.partition(": ")
        if not separator or name != expected_name or not value:
            return None
        fields[name] = value
    return fields


def _codex_agent_message_finding(payload: Mapping[str, object]) -> str:
    author = _nonempty_string(payload.get("author"))
    recipient = _nonempty_string(payload.get("recipient"))
    content = payload.get("content")
    if not author or not recipient or not isinstance(content, list):
        return ""

    for block in content:
        if not isinstance(block, dict) or block.get("type") != "input_text":
            continue
        text = _nonempty_string(block.get("text"))
        header, marker, finding = text.partition("\nPayload:\n")
        fields = _parse_agent_message_header(header)
        if (
            marker
            and fields is not None
            and fields.get("Message Type") == "FINAL_ANSWER"
            and fields.get("Task name") == recipient
            and fields.get("Sender") == author
            and finding.strip()
        ):
            return finding.strip()
    return ""


def _codex_subagent_turn(value: object, timestamp: str | None) -> Turn | None:
    finding = _completed_finding(value).strip()
    if not finding:
        return None
    if len(finding) > MAX_FINDING_CHARS:
        finding = finding[:MAX_FINDING_CHARS] + " …[truncated]"
    return Turn(
        "assistant",
        f"[Subagent result] {finding}",
        "subagent_finding",
        timestamp,
    )


def _codex_call_ids(records: Iterable[Mapping[str, object]]) -> Iterator[str]:
    for record in records:
        payload = record.get("payload", {})
        if record.get("type") != "response_item" or not isinstance(payload, dict):
            continue
        if payload.get("type") not in {"function_call", "custom_tool_call"}:
            continue
        call_id = (
            _nonempty_string(payload.get("call_id"))
            or _nonempty_string(payload.get("id"))
        )
        if call_id:
            yield call_id


def parse_codex_transcript(
    path: Path | str,
    metadata: Mapping[str, object],
    limits: object = None,
    *,
    records: Sequence[Mapping[str, object]] | None = None,
) -> NormalizedSession:
    """Normalize Codex JSONL without retaining hidden or routine tool traffic."""
    transcript_path = Path(path)
    source_records = tuple(_read_jsonl(transcript_path)) if records is None else tuple(records)
    session_meta: dict = {}
    session_meta_timestamp = ""
    calls: dict[str, str | None] = {
        call_id: None
        for call_id in _duplicate_call_ids(_codex_call_ids(source_records))
    }
    turns = _turn_accumulator(limits)

    for record in source_records:
        record_type = record.get("type")
        payload = record.get("payload", {})
        if record_type == "session_meta" and isinstance(payload, dict) and not session_meta:
            session_meta = payload
            session_meta_timestamp = (
                _nonempty_string(payload.get("timestamp"))
                or _nonempty_string(record.get("timestamp"))
            )
            continue
        if record_type != "response_item" or not isinstance(payload, dict):
            continue

        payload_type = payload.get("type")
        timestamp = _nonempty_string(record.get("timestamp")) or None
        if payload_type == "message":
            role = payload.get("role")
            if role not in ("user", "assistant"):
                continue
            expected_block_type = "input_text" if role == "user" else "output_text"
            content = payload.get("content", [])
            if not isinstance(content, list):
                continue
            parts = [
                _nonempty_string(block.get("text"))
                for block in content
                if isinstance(block, dict) and block.get("type") == expected_block_type
            ]
            text = "\n".join(part for part in parts if part.strip()).strip()
            if text:
                turns.append(Turn(role=role, text=text, timestamp=timestamp))
        elif payload_type == "agent_message":
            finding = _codex_agent_message_finding(payload)
            if not finding:
                continue
            finding_turn = _codex_subagent_turn(
                {"completed": finding}, timestamp
            )
            if finding_turn is not None:
                turns.append(finding_turn)
        elif payload_type in {"function_call", "custom_tool_call"}:
            call_id = (
                _nonempty_string(payload.get("call_id"))
                or _nonempty_string(payload.get("id"))
            )
            if call_id:
                _remember_call(
                    calls,
                    call_id,
                    _tool_basename(_nonempty_string(payload.get("name"))),
                )
        elif payload_type in {"function_call_output", "custom_tool_call_output"}:
            call_id = (
                _nonempty_string(payload.get("call_id"))
                or _nonempty_string(payload.get("id"))
            )
            tool_name = calls.get(call_id)
            output = payload.get("output", payload.get("content"))
            if tool_name in CODEX_DECISION_TOOLS:
                text = _render_codex_decision(output)
                if text:
                    calls[call_id] = None
                    turns.append(Turn("user", text, "decision", timestamp))
            elif tool_name in CODEX_SUBAGENT_RESULT_TOOLS:
                finding_turn = _codex_subagent_turn(output, timestamp)
                if finding_turn is not None:
                    calls[call_id] = None
                    turns.append(finding_turn)

    return _make_session(
        agent="codex",
        path=transcript_path,
        metadata=metadata,
        turns=_apply_limits(turns, limits) if limits is not None else turns,
        fallback_session_id=_nonempty_string(session_meta.get("id")),
        fallback_cwd=_nonempty_string(session_meta.get("cwd")),
        fallback_timestamp=session_meta_timestamp,
    )


def _render_turns(turns: Sequence[Turn]) -> str:
    return "\n".join(_render_turn(turn) for turn in turns)


def render_turns(session: NormalizedSession) -> str:
    """Render normalized turns in the durable context format used by flush prompts."""
    return _render_turns(session.turns)


def _session_with_turns(
    session: NormalizedSession,
    turns: Sequence[Turn],
) -> NormalizedSession:
    normalized_turns = tuple(turns)
    return NormalizedSession(
        agent=session.agent,
        session_id=session.session_id,
        project=session.project,
        cwd=session.cwd,
        timestamp=session.timestamp,
        trigger=session.trigger,
        turns=normalized_turns,
        source_path=session.source_path,
        source_hash=_canonical_hash(
            agent=session.agent,
            session_id=session.session_id,
            project=session.project,
            cwd=session.cwd,
            turns=normalized_turns,
        ),
    )


def chunk_session(
    session: NormalizedSession, target_chars: int
) -> list[NormalizedSession]:
    """Split at user boundaries, keeping each assistant reply with its user turn."""
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if not session.turns:
        return []

    total_chars = sum(len(_render_turn(turn)) for turn in session.turns)
    if total_chars <= target_chars * 1.3:
        # A whole-session historical job must use the same normalized identity
        # as an equivalent live capture.  Derived chunk hashes are needed only
        # when a session actually splits.
        return [session]

    chunks: list[NormalizedSession] = []

    def append_chunk(turns: Sequence[Turn]) -> None:
        chunks.append(_session_with_turns(session, turns))

    current: list[Turn] = []
    current_chars = 0
    for turn in session.turns:
        if current and current_chars >= target_chars and turn.role == "user":
            append_chunk(current)
            current = []
            current_chars = 0
        current.append(turn)
        current_chars += len(_render_turn(turn))
    if current:
        append_chunk(current)
    return chunks
