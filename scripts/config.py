"""Path constants and configuration for the personal knowledge base."""

import os
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

if __package__:
    from .providers import TaskKind
else:
    from providers import TaskKind


_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_COMPATIBILITY_WARNING_STATE = "_ai_memory_compatibility_warning_emitted"


@dataclass(frozen=True)
class MemoryConfig:
    """Validated, immutable runtime configuration for memory jobs."""

    root_dir: Path
    provider_order: tuple[str, ...]
    codex_luna_model: str
    codex_terra_model: str
    claude_model: str
    job_timeout_seconds: int
    internal_job: bool
    queue_path: Path
    worker_concurrency: int
    usage_estimate_only: bool

    @property
    def task_models(self) -> Mapping[TaskKind, str]:
        """Return configured Codex models for each task category."""
        return MappingProxyType(
            {
                TaskKind.EXTRACT: self.codex_luna_model,
                TaskKind.SEMANTIC_LINT: self.codex_luna_model,
                TaskKind.COMPILE: self.codex_terra_model,
                TaskKind.QUERY: self.codex_terra_model,
                TaskKind.CONNECTIONS: self.codex_terra_model,
                TaskKind.FILE_ANSWER: self.codex_terra_model,
            }
        )


def _resolved_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _resolve_root(env: Mapping[str, str]) -> Path:
    canonical_raw = env.get("AI_MEMORY_HOME")
    compatibility_raw = env.get("CLAUDE_MEMORY_HOME")

    if canonical_raw is not None and not canonical_raw.strip():
        raise ValueError("AI_MEMORY_HOME must not be empty")
    if compatibility_raw is not None and not compatibility_raw.strip():
        raise ValueError("CLAUDE_MEMORY_HOME must not be empty")

    canonical = _resolved_path(canonical_raw) if canonical_raw is not None else None
    compatibility = (
        _resolved_path(compatibility_raw) if compatibility_raw is not None else None
    )
    if canonical is not None and compatibility is not None and canonical != compatibility:
        raise ValueError(
            "AI_MEMORY_HOME and CLAUDE_MEMORY_HOME resolve to different paths"
        )
    if canonical is not None:
        return canonical
    if compatibility is not None:
        if not getattr(sys, _COMPATIBILITY_WARNING_STATE, False):
            warnings.warn(
                "CLAUDE_MEMORY_HOME is deprecated; use AI_MEMORY_HOME instead",
                DeprecationWarning,
                stacklevel=3,
            )
            setattr(sys, _COMPATIBILITY_WARNING_STATE, True)
        return compatibility
    return _REPOSITORY_ROOT


def _nonempty(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, default)
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _positive_integer(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name, "1" if default else "0")
    if raw not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return raw == "1"


def load_config(env: Mapping[str, str]) -> MemoryConfig:
    """Parse and validate memory configuration from an explicit environment."""
    root_dir = _resolve_root(env)

    provider_order_raw = env.get("AI_MEMORY_PROVIDER_ORDER", "codex,claude")
    provider_order = tuple(provider.strip() for provider in provider_order_raw.split(","))
    if provider_order != ("codex", "claude"):
        raise ValueError("AI_MEMORY_PROVIDER_ORDER must be codex,claude")

    queue_raw = env.get("AI_MEMORY_QUEUE_PATH")
    if queue_raw is None:
        queue_path = (root_dir / "scripts" / "jobs.sqlite3").resolve()
    else:
        if not queue_raw.strip():
            raise ValueError("AI_MEMORY_QUEUE_PATH must not be empty")
        expanded_queue = Path(queue_raw).expanduser()
        if not expanded_queue.is_absolute():
            raise ValueError("AI_MEMORY_QUEUE_PATH must be absolute")
        queue_path = expanded_queue.resolve()

    return MemoryConfig(
        root_dir=root_dir,
        provider_order=provider_order,
        codex_luna_model=_nonempty(
            env, "AI_MEMORY_CODEX_LUNA_MODEL", "gpt-5.6-luna"
        ),
        codex_terra_model=_nonempty(
            env, "AI_MEMORY_CODEX_TERRA_MODEL", "gpt-5.6-terra"
        ),
        claude_model=_nonempty(env, "AI_MEMORY_CLAUDE_MODEL", "claude-sonnet-5"),
        job_timeout_seconds=_positive_integer(
            env, "AI_MEMORY_JOB_TIMEOUT_SECONDS", 900
        ),
        internal_job=_boolean(env, "AI_MEMORY_INTERNAL_JOB", False),
        queue_path=queue_path,
        worker_concurrency=_positive_integer(
            env, "AI_MEMORY_WORKER_CONCURRENCY", 2
        ),
        usage_estimate_only=_boolean(env, "AI_MEMORY_USAGE_ESTIMATE_ONLY", False),
    )

# ── Paths ──────────────────────────────────────────────────────────────
CONFIG = load_config(os.environ)

ROOT_DIR = CONFIG.root_dir
DAILY_DIR = ROOT_DIR / "daily"
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"
CONCEPTS_DIR = KNOWLEDGE_DIR / "concepts"
CONNECTIONS_DIR = KNOWLEDGE_DIR / "connections"
QA_DIR = KNOWLEDGE_DIR / "qa"
REPORTS_DIR = ROOT_DIR / "reports"
SCRIPTS_DIR = ROOT_DIR / "scripts"
HOOKS_DIR = ROOT_DIR / "hooks"
AGENTS_FILE = ROOT_DIR / "AGENTS.md"

INDEX_FILE = KNOWLEDGE_DIR / "index.md"
LOG_FILE = KNOWLEDGE_DIR / "log.md"
STATE_FILE = SCRIPTS_DIR / "state.json"

# ── Generation settings ───────────────────────────────────────────────
PROVIDER_ORDER = CONFIG.provider_order
CODEX_LUNA_MODEL = CONFIG.codex_luna_model
CODEX_TERRA_MODEL = CONFIG.codex_terra_model
CLAUDE_MODEL = CONFIG.claude_model
JOB_TIMEOUT_SECONDS = CONFIG.job_timeout_seconds
INTERNAL_JOB = CONFIG.internal_job
QUEUE_PATH = CONFIG.queue_path
WORKER_CONCURRENCY = CONFIG.worker_concurrency
USAGE_ESTIMATE_ONLY = CONFIG.usage_estimate_only
TASK_MODELS = CONFIG.task_models

# ── Timezone ───────────────────────────────────────────────────────────
TIMEZONE = "America/Chicago"


def now_iso() -> str:
    """Current time in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def today_iso() -> str:
    """Current date in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
