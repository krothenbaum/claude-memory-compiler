"""Fast Codex SessionEnd adapter with no model output or model calls."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sqlite3
import sys
import time
from collections.abc import Callable
from pathlib import Path

if os.environ.get("AI_MEMORY_INTERNAL_JOB") == "1" or "CLAUDE_INVOKED_BY" in os.environ:
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hook_logging import (
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


def main(clock: Callable[[], float] = time.monotonic) -> None:
    deadline = clock() + HOOK_WORK_BUDGET_SECONDS
    logger = _logger()
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
            source_agent="codex",
            session_id=value.get("session_id"),
        )


if __name__ == "__main__":
    main()
