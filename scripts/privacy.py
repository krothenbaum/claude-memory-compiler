"""Dependency-neutral validation for bounded persisted operational text."""

from __future__ import annotations

import re
from collections.abc import Mapping

MAX_PERSISTENCE_REASON_CHARS = 1_000
SECRET_ENV_NAMES = frozenset({"ANTHROPIC_API_KEY", "CLAUDE_API_KEY"})
SECRET_ENV_SUFFIXES = ("_TOKEN", "_API_KEY", "_SECRET", "_PASSWORD")

_OSC_PATTERN = re.compile(
    r"(?:\x1b\]|\x9d).*?(?:\x07|\x1b\\|\x9c|\Z)",
    re.DOTALL,
)
_CSI_PATTERN = re.compile(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]")
_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]+")


def strip_terminal_controls(value: object) -> str:
    """Remove terminal commands and replace remaining controls with spaces."""
    text = _OSC_PATTERN.sub("", str(value))
    text = _CSI_PATTERN.sub("", text)
    return _CONTROL_PATTERN.sub(" ", text)


def normalize_persistence_reason(
    reason: object,
    env: Mapping[str, str],
) -> str:
    """Bound and redact metadata persisted at operational failure boundaries."""
    normalized = " ".join(strip_terminal_controls(reason).split()) or "unspecified failure"
    secrets = {
        value
        for name, value in env.items()
        if value
        and (
            name in SECRET_ENV_NAMES
            or name.startswith(("OPENAI_", "AZURE_OPENAI_"))
            or name.endswith(SECRET_ENV_SUFFIXES)
        )
    }
    for secret in sorted(secrets, key=len, reverse=True):
        normalized = normalized.replace(secret, "[REDACTED]")
    return normalized[:MAX_PERSISTENCE_REASON_CHARS]
