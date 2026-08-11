"""Fast Codex SessionEnd adapter with no model output or model calls."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys


if os.environ.get("AI_MEMORY_INTERNAL_JOB") == "1" or "CLAUDE_INVOKED_BY" in os.environ:
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.capture import enqueue_hook_input
from scripts.transcripts import parse_codex_transcript


MAX_TURNS = 30
MAX_CONTEXT_CHARS = 15_000


def _runtime_root() -> Path:
    configured = os.environ.get("AI_MEMORY_HOME") or os.environ.get(
        "CLAUDE_MEMORY_HOME"
    )
    return Path(configured).expanduser() if configured else ROOT


def _logger() -> logging.Logger:
    logger = logging.getLogger("ai-memory-codex-session-end")
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
        logging.Formatter("%(asctime)s %(levelname)s [codex-session-end] %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def main() -> None:
    logger = _logger()
    try:
        value = json.loads(sys.stdin.read())
        if not isinstance(value, dict):
            raise ValueError("hook input must be a JSON object")
    except (json.JSONDecodeError, ValueError, EOFError) as error:
        logger.error("failed to parse hook input: %s", error)
        return

    transcript_value = value.get("transcript_path")
    if not isinstance(transcript_value, str) or not transcript_value:
        logger.info("skip: no transcript path")
        return
    transcript_path = Path(transcript_value).expanduser()
    if not transcript_path.is_file():
        logger.info("skip: transcript missing")
        return

    try:
        preview = parse_codex_transcript(
            transcript_path,
            {
                "session_id": value.get("session_id", ""),
                "cwd": value.get("cwd", ""),
                "timestamp": value.get("timestamp", ""),
                "project": value.get("project", ""),
                "trigger": "session_end",
            },
            limits={"max_turns": MAX_TURNS, "max_chars": MAX_CONTEXT_CHARS},
        )
        if not preview.turns:
            logger.info("skip: empty normalized transcript")
            return
        outcome = enqueue_hook_input(
            value,
            source_agent="codex",
            trigger="session_end",
            limits={"max_turns": MAX_TURNS, "max_chars": MAX_CONTEXT_CHARS},
        )
        logger.info("capture %s for session %s", outcome.status, outcome.job_id)
    except Exception as error:
        logger.error("capture failed: %s", error)


if __name__ == "__main__":
    main()
