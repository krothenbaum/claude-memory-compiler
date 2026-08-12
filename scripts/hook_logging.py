"""Fail-closed logging boundary shared by live hook entrypoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    from .utils import open_secure_log_stream, prepare_secure_log_directory
except ImportError:  # Standalone execution with scripts/ on sys.path.
    from utils import open_secure_log_stream, prepare_secure_log_directory


class _HookLogHandler(logging.StreamHandler):
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
    handler.setFormatter(
        logging.Formatter(f"%(asctime)s %(levelname)s [{label}] %(message)s")
    )
    handler._memory_hook_file = True  # type: ignore[attr-defined]
    handler._memory_log_path = str(target)  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
