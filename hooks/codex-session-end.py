"""Fast Codex SessionEnd adapter with no model output or model calls."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
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
from scripts.transcripts import parse_codex_transcript

MAX_TURNS = 30
MAX_CONTEXT_CHARS = 15_000
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
        "ai-memory-codex-session-end", "codex-session-end", _runtime_root()
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
            source_agent="codex",
            session_id=session_id,
            project=project,
            message=message,
            deadline=deadline,
            clock=clock,
        )
    except Exception:
        return False


def main(clock: Callable[[], float] = time.monotonic) -> None:
    deadline = clock() + HOOK_WORK_BUDGET_SECONDS
    logger = _logger()
    if _INVALID_QUEUE_OVERRIDE:
        try:
            invalid_input = json.loads(sys.stdin.read())
            session_id = (
                invalid_input.get("session_id")
                if isinstance(invalid_input, dict)
                else None
            )
        except (json.JSONDecodeError, ValueError, EOFError):
            session_id = None
        log_hook_event(
            logger,
            logging.ERROR,
            "queue_unavailable",
            "configured queue path is invalid",
            source_agent="codex",
            session_id=session_id,
        )
        return
    try:
        value = json.loads(sys.stdin.read())
        if not isinstance(value, dict):
            raise ValueError("hook input must be a JSON object")
    except (json.JSONDecodeError, ValueError, EOFError):
        log_hook_event(
            logger,
            logging.ERROR,
            "malformed_input",
            "failed to parse hook input",
            source_agent="codex",
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

    transcript_value = value.get("transcript_path")
    if not isinstance(transcript_value, str) or not transcript_value:
        log_hook_event(
            logger,
            logging.ERROR,
            "transcript_missing",
            "hook input did not include a transcript",
            source_agent="codex",
            session_id=value.get("session_id"),
        )
        _record_diagnostic(
            event="transcript_missing",
            session_id=value.get("session_id"),
            project=value.get("project"),
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
            source_agent="codex",
            session_id=value.get("session_id"),
        )
        _record_diagnostic(
            event=transcript_event,
            session_id=value.get("session_id"),
            project=value.get("project"),
            message=(
                "transcript is missing"
                if transcript_event == "transcript_missing"
                else "transcript is unreadable"
            ),
            deadline=deadline,
            clock=clock,
        )
        return

    try:
        helpers = _live_capture_helpers()
        metadata = {
            "session_id": value.get("session_id", ""),
            "cwd": value.get("cwd", ""),
            "timestamp": value.get("timestamp", ""),
            "project": value.get("project", ""),
            "trigger": "session_end",
        }

        def previewer(path: Path):
            return parse_codex_transcript(
                path,
                metadata,
                limits={"max_turns": MAX_TURNS, "max_chars": MAX_CONTEXT_CHARS},
            )

        with helpers.bounded_transcript_slice(
            transcript_path,
            previewer,
            source_agent="codex",
            memory_root=_runtime_root(),
            deadline=deadline,
            clock=clock,
        ) as selected:
            live_slice, preview = selected
            if not preview.turns:
                logger.info("skip: empty normalized transcript")
                return
            helpers.require_time_remaining(
                deadline, clock, helpers.MIN_CAPTURE_REMAINING_SECONDS
            )
            capture_input = dict(value)
            capture_input["transcript_path"] = str(live_slice)
            outcome = helpers.enqueue_capture_with_deadline(
                capture_input,
                source_agent="codex",
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
            source_agent="codex",
            session_id=value.get("session_id"),
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
            source_agent="codex",
            session_id=value.get("session_id"),
        )
        if event == "capture_failed":
            _record_diagnostic(
                event=event,
                session_id=value.get("session_id"),
                project=value.get("project"),
                message="capture failed",
                deadline=deadline,
                clock=clock,
            )


if __name__ == "__main__":
    main()
