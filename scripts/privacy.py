"""Dependency-neutral validation for bounded persisted operational text."""

from __future__ import annotations

from collections.abc import Mapping

MAX_PERSISTENCE_REASON_CHARS = 1_000
SECRET_ENV_NAMES = frozenset({"ANTHROPIC_API_KEY", "CLAUDE_API_KEY"})
SECRET_ENV_SUFFIXES = ("_TOKEN", "_API_KEY", "_SECRET", "_PASSWORD")


def normalize_persistence_reason(
    reason: object,
    env: Mapping[str, str],
) -> str:
    """Bound and redact metadata persisted at operational failure boundaries."""
    normalized = " ".join(str(reason).split()) or "unspecified failure"
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
