"""Fail-closed logging boundary shared by live hook entrypoints."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path

try:
    from .utils import open_secure_log_stream, prepare_secure_log_directory
except ImportError:  # Standalone execution with scripts/ on sys.path.
    from utils import open_secure_log_stream, prepare_secure_log_directory


class HookJsonFormatter(logging.Formatter):
    """Format one stable, exception-safe operational JSONL record."""

    def __init__(self, component: str) -> None:
        super().__init__()
        self.component = component

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, timezone.utc)
        value = {
            "timestamp": timestamp.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "level": record.levelname,
            "component": self.component,
            "event": "hook_log",
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class _HookLogHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        """Append one encoded JSON object with one operating-system write."""
        try:
            payload = (self.format(record) + "\n").encode("utf-8")
            written = os.write(self.stream.fileno(), payload)
            if written != len(payload):
                raise OSError("incomplete hook log append")
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            if self.stream is not None and not self.stream.closed:
                self.stream.close()
        finally:
            super().close()


def _remove_owned_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "_memory_hook_file", False):
            logger.removeHandler(handler)
            handler.close()


def configure_hook_logger(
    logger_name: str,
    label: str,
    memory_root: Path | str,
) -> logging.Logger:
    """Configure one isolated hook logger, degrading to a null sink on risk."""
    logger = logging.getLogger(logger_name)
    target = Path(os.path.abspath(Path(memory_root).expanduser())) / "scripts" / "logs" / "hooks.log"
    tagged = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_memory_hook_file", False)
    ]
    if len(tagged) == 1 and Path(tagged[0]._memory_log_path) == target:
        return logger

    _remove_owned_handlers(logger)
    try:
        prepare_secure_log_directory(memory_root)
        handler: logging.Handler = _HookLogHandler(open_secure_log_stream(target))
    except (OSError, ValueError):
        handler = logging.NullHandler()
    handler.setFormatter(HookJsonFormatter(label))
    handler._memory_hook_file = True  # type: ignore[attr-defined]
    handler._memory_log_path = str(target)  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
