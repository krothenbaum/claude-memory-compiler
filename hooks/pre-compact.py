"""Fast Claude PreCompact adapter: preserve the live slice in the queue."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

if os.environ.get("AI_MEMORY_INTERNAL_JOB") == "1" or "CLAUDE_INVOKED_BY" in os.environ:
    sys.exit(0)

_raw_queue_override = os.environ.get("AI_MEMORY_QUEUE_PATH")
_INVALID_QUEUE_OVERRIDE = False
if "AI_MEMORY_QUEUE_PATH" in os.environ:
    try:
        _INVALID_QUEUE_OVERRIDE = (
            not isinstance(_raw_queue_override, str)
            or not _raw_queue_override.strip()
            or not Path(_raw_queue_override).expanduser().is_absolute()
        )
    except (OSError, RuntimeError, ValueError):
        _INVALID_QUEUE_OVERRIDE = True
if _INVALID_QUEUE_OVERRIDE:
    os.environ.pop("AI_MEMORY_QUEUE_PATH", None)
del _raw_queue_override

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hook_logging import (
    classify_capture_error,
    classify_transcript_path,
    configure_hook_logger,
    log_hook_event,
)
from scripts.transcripts import parse_claude_transcript, render_turns

MAX_TURNS = 30
MAX_CONTEXT_CHARS = 15_000
MIN_TURNS_TO_FLUSH = 5
HOOK_WORK_BUDGET_SECONDS = 2.25


def _live_capture_helpers():
    path = Path(__file__).with_name("session-end.py")
    spec = importlib.util.spec_from_file_location("ai_memory_live_capture", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load live capture helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_root() -> Path:
    configured = os.environ.get("AI_MEMORY_HOME") or os.environ.get(
        "CLAUDE_MEMORY_HOME"
    )
    return Path(configured).expanduser() if configured else ROOT


def _logger() -> logging.Logger:
    return configure_hook_logger(
        "ai-memory-pre-compact", "pre-compact", _runtime_root()
    )


def _record_diagnostic(
    *,
    event: str,
    session_id: object,
    project: object,
    message: str,
    deadline: float,
    clock: Callable[[], float],
) -> bool:
    try:
        from scripts.status_health import record_hook_diagnostic

        return record_hook_diagnostic(
            _runtime_root(),
            event=event,
            source_agent="claude",
            session_id=session_id,
            project=project,
            message=message,
            deadline=deadline,
            clock=clock,
        )
    except Exception:
        return False


def _read_hook_input() -> dict[str, object]:
    raw_input = sys.stdin.read()
    try:
        value = json.loads(raw_input)
    except json.JSONDecodeError:
        fixed_input = re.sub(r'(?<!\\)\\(?!["\\])', r"\\\\", raw_input)
        value = json.loads(fixed_input)
    if not isinstance(value, dict):
        raise ValueError("hook input must be a JSON object")
    return value


def extract_conversation_context(
    transcript_path: Path, metadata: dict | None = None
) -> tuple[str, int]:
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


def main(clock: Callable[[], float] = time.monotonic) -> None:
    deadline = clock() + HOOK_WORK_BUDGET_SECONDS
    logger = _logger()
    if _INVALID_QUEUE_OVERRIDE:
        try:
            invalid_input = _read_hook_input()
            session_id = invalid_input.get("session_id")
        except (json.JSONDecodeError, ValueError, EOFError):
            session_id = None
        log_hook_event(
            logger,
            logging.ERROR,
            "queue_unavailable",
            "configured queue path is invalid",
            source_agent="claude",
            session_id=session_id,
        )
        return
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
        _record_diagnostic(
            event="malformed_input",
            session_id="unknown",
            project="unknown",
            message="failed to parse hook input",
            deadline=deadline,
            clock=clock,
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
        _record_diagnostic(
            event="transcript_missing",
            session_id=hook_input.get("session_id"),
            project=hook_input.get("project"),
            message="hook input did not include a transcript",
            deadline=deadline,
            clock=clock,
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
        _record_diagnostic(
            event=transcript_event,
            session_id=hook_input.get("session_id"),
            project=hook_input.get("project"),
            message=(
                "transcript is missing"
                if transcript_event == "transcript_missing"
                else "transcript is unreadable"
            ),
            deadline=deadline,
            clock=clock,
        )
        return

    cwd = hook_input.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        cwd = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
        hook_input["cwd"] = cwd
    hook_input.setdefault("project", Path(cwd).name or "unknown")

    try:
        helpers = _live_capture_helpers()
        metadata = {
            "session_id": hook_input.get("session_id", ""),
            "cwd": cwd,
            "project": hook_input["project"],
            "timestamp": hook_input.get("timestamp", ""),
            "trigger": "pre_compact",
        }

        def previewer(path: Path):
            return parse_claude_transcript(
                path, metadata, limits={"max_turns": MAX_TURNS}
            )

        with helpers.bounded_transcript_slice(
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
            helpers.require_time_remaining(
                deadline, clock, helpers.MIN_CAPTURE_REMAINING_SECONDS
            )
            capture_input = dict(hook_input)
            capture_input["transcript_path"] = str(live_slice)
            outcome = helpers.enqueue_capture_with_deadline(
                capture_input,
                source_agent="claude",
                trigger="pre_compact",
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
        event = classify_capture_error(error)
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
        if event == "capture_failed":
            _record_diagnostic(
                event=event,
                session_id=hook_input.get("session_id"),
                project=hook_input.get("project"),
                message="capture failed",
                deadline=deadline,
                clock=clock,
            )


if __name__ == "__main__":
    main()
