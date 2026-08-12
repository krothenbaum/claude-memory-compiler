"""Fast Claude PreCompact adapter: preserve the live slice in the queue."""

from __future__ import annotations

import json
import importlib.util
import logging
import os
from pathlib import Path
import re
import sys


if os.environ.get("AI_MEMORY_INTERNAL_JOB") == "1" or "CLAUDE_INVOKED_BY" in os.environ:
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.capture import enqueue_hook_input
from scripts.transcripts import parse_claude_transcript, render_turns


MAX_TURNS = 30
MAX_CONTEXT_CHARS = 15_000
MIN_TURNS_TO_FLUSH = 5


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
    logger = logging.getLogger("ai-memory-pre-compact")
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
        logging.Formatter("%(asctime)s %(levelname)s [pre-compact] %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


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
        helpers = _live_capture_helpers()
        with helpers.bounded_transcript_slice(transcript_path) as live_slice:
            context, turn_count = extract_conversation_context(
                live_slice,
                {
                    "session_id": hook_input.get("session_id", ""),
                    "cwd": cwd,
                    "project": hook_input["project"],
                    "timestamp": hook_input.get("timestamp", ""),
                    "trigger": "pre_compact",
                },
            )
            if not context.strip() or turn_count < MIN_TURNS_TO_FLUSH:
                logger.info("skip: empty or too-short transcript")
                return
            capture_input = dict(hook_input)
            capture_input["transcript_path"] = str(live_slice)
            outcome = enqueue_hook_input(
                capture_input,
                source_agent="claude",
                trigger="pre_compact",
                limits={"max_turns": MAX_TURNS, "max_chars": MAX_CONTEXT_CHARS},
            )
        logger.info("capture %s for session %s", outcome.status, outcome.job_id)
    except Exception as error:
        logger.error("capture failed: %s", error)


if __name__ == "__main__":
    main()
