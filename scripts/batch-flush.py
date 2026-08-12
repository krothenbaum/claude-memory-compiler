"""
Batch extraction of knowledge from historical Claude Code transcripts.

Reads JSONL transcript files, extracts conversation content with smart
chunking for large sessions, runs LLM extraction on each chunk, groups
results by date, and writes daily logs ready for compile.py.

Usage:
    uv run python scripts/batch-flush.py --dry-run           # preview what would be processed
    uv run python scripts/batch-flush.py                      # run full extraction
    uv run python scripts/batch-flush.py --compile            # extract + compile
    uv run python scripts/batch-flush.py --max-cost 5.00      # stop after $5 spent
    uv run python scripts/batch-flush.py --resume             # skip already-processed sessions
    uv run python scripts/batch-flush.py --dates 2026-04-11   # only specific dates
    uv run python scripts/batch-flush.py --all-projects       # seed every project on this machine
"""

from __future__ import annotations

import os

import argparse
import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from transcripts import (
    NormalizedSession,
    Turn as NormalizedTurn,
    chunk_session,
    codex_transcript_is_well_formed,
    parse_claude_transcript,
    parse_codex_transcript,
    read_codex_session_meta,
    render_turns,
)

from config import load_config
from providers import ClaudeProvider, CodexProvider, ProviderRouter, TaskKind
from scripts.queue import QueueRepository
from utils import append_daily_entry
from worker import MemoryWorker

DAILY_DIR = ROOT / "daily"
STATE_FILE = SCRIPTS_DIR / "state.json"
LOG_FILE = SCRIPTS_DIR / "flush.log"

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"


def default_transcripts_dir(cwd: Path | None = None) -> Path:
    """Auto-discover the transcripts directory from a working directory.

    Claude Code encodes a project's absolute path in the directory name by
    replacing ``/`` with ``-``. For a project at ``/Users/foo/bar``, transcripts
    live under ``~/.claude/projects/-Users-foo-bar/``.
    """
    base = (cwd or Path.cwd()).resolve()
    encoded = str(base).replace("/", "-")
    return CLAUDE_PROJECTS_DIR / encoded


DEFAULT_TRANSCRIPTS_DIR = default_transcripts_dir()


def _resolve_encoded_path(base: Path, parts: list[str]) -> Path | None:
    """Recursively try every '-' boundary as a '/' separator until one matches the filesystem."""
    if not parts:
        return base if base.is_dir() else None
    for end in range(1, len(parts) + 1):
        component = "-".join(parts[:end])
        candidate = base / component
        if candidate.is_dir():
            result = _resolve_encoded_path(candidate, parts[end:])
            if result is not None:
                return result
    return None


def decode_project_path(encoded_name: str) -> Path | None:
    """Best-effort reverse of Claude Code's '/' → '-' encoding via filesystem checks.

    The encoding loses information when path components themselves contain ``-``,
    so we walk filesystem candidates and return the first directory that exists.
    Returns None if no decoded path resolves to an existing directory (e.g.
    transcripts left behind after the project was moved or deleted).
    """
    if not encoded_name.startswith("-"):
        return None
    parts = encoded_name[1:].split("-")
    if not parts:
        return None
    return _resolve_encoded_path(Path("/"), parts)


MIN_FILE_SIZE = 5_000           # Skip files < 5KB
CHUNK_TARGET_CHARS = 25_000     # Target chunk size for LLM extraction
FLUSH_COST_ESTIMATE = 0.04      # Estimated cost per flush call

def configure_logging(*, dry_run: bool) -> None:
    """Configure interactive logging without creating files during imports/dry runs."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(
        getattr(handler, "_memory_batch_console", False)
        for handler in root_logger.handlers
    ):
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        console._memory_batch_console = True  # type: ignore[attr-defined]
        root_logger.addHandler(console)
    if dry_run or any(
        getattr(handler, "_memory_batch_file", False) for handler in root_logger.handlers
    ):
        return
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [batch] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    file_handler._memory_batch_file = True  # type: ignore[attr-defined]
    root_logger.addHandler(file_handler)


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class TranscriptInfo:
    path: Path
    size: int
    mtime: datetime
    date: str               # YYYY-MM-DD from mtime
    session_id: str         # UUID from filename

@dataclass
class Turn:
    role: str               # "user" or "assistant"
    text: str
    index: int

@dataclass
class Chunk:
    text: str
    char_count: int
    position: str           # "early", "mid", "late", "full"
    turn_range: tuple[int, int] = (0, 0)

@dataclass
class Extraction:
    session_id: str
    date: str
    time_str: str
    chunk_position: str
    content: str
    cost: float = 0.0


@dataclass
class Target:
    project_key: str
    project_cwd: str        # may be empty if the path could not be decoded
    transcripts_dir: Path


@dataclass(frozen=True)
class HistoricalSession:
    session: NormalizedSession
    path: Path
    date: str
    directory_date: str | None = None


@dataclass(frozen=True)
class CodexDiscovery:
    sessions: tuple[HistoricalSession, ...]
    malformed: tuple[Path, ...] = ()
    duplicates: tuple[Path, ...] = ()
    date_disagreements: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ImportReport:
    sessions: int
    chunks: int
    projects: tuple[str, ...]
    dates: tuple[str, ...]
    estimated_tokens: int
    enqueued: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    dead: int = 0


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _directory_date(root: Path, path: Path) -> str | None:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    for index in range(max(0, len(parts) - 3)):
        year, month, day = parts[index : index + 3]
        try:
            return date(int(year), int(month), int(day)).isoformat()
        except (TypeError, ValueError):
            continue
    return None


def _parse_codex_candidate(root: Path, path: Path) -> HistoricalSession | None:
    try:
        if not codex_transcript_is_well_formed(path):
            return None
        meta = read_codex_session_meta(path)
        if meta is None or any(
            not isinstance(meta.get(field), str) or not str(meta[field]).strip()
            for field in ("id", "cwd", "timestamp")
        ):
            return None
        timestamp = _parse_timestamp(str(meta["timestamp"]))
        if timestamp is None:
            return None
        normalized = parse_codex_transcript(path, {"trigger": "historical"})
    except (OSError, UnicodeError, ValueError):
        return None
    if not normalized.turns:
        return None
    return HistoricalSession(
        session=normalized,
        path=path,
        date=timestamp.date().isoformat(),
        directory_date=_directory_date(root, path),
    )


def discover_codex_sessions(
    sessions_root: Path | str = CODEX_SESSIONS_DIR, *, concurrency: int = 1
) -> CodexDiscovery:
    """Recursively discover valid local Codex rollouts with bounded parsing."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    root = Path(sessions_root).expanduser()
    if not root.is_dir():
        return CodexDiscovery(())

    paths = sorted(root.rglob("*.jsonl"))
    if concurrency == 1:
        candidates = [_parse_codex_candidate(root, path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            candidates = list(
                executor.map(lambda path: _parse_codex_candidate(root, path), paths)
            )

    sessions: list[HistoricalSession] = []
    malformed: list[Path] = []
    duplicates: list[Path] = []
    disagreements: list[Path] = []
    seen: set[tuple[str, str]] = set()
    for path, discovered in zip(paths, candidates, strict=True):
        if discovered is None:
            malformed.append(path)
            continue
        identity = (discovered.session.session_id, discovered.session.source_hash)
        if identity in seen:
            duplicates.append(path)
            continue
        seen.add(identity)
        sessions.append(discovered)
        if (
            discovered.directory_date is not None
            and discovered.directory_date != discovered.date
        ):
            disagreements.append(path)

    sessions.sort(key=lambda item: (item.session.timestamp, str(item.path)))
    return CodexDiscovery(
        tuple(sessions),
        tuple(malformed),
        tuple(duplicates),
        tuple(disagreements),
    )


# ── Project discovery ───────────────────────────────────────────────────

def discover_all_projects() -> list[Target]:
    """Walk ~/.claude/projects/ and return one Target per encoded project directory.

    Skips entries that don't look like Claude Code's path-encoded directories
    (i.e. don't start with ``-``); those are usually scratch dirs created by
    other tooling rather than transcript stores.
    """
    if not CLAUDE_PROJECTS_DIR.is_dir():
        return []

    targets: list[Target] = []
    for entry in sorted(CLAUDE_PROJECTS_DIR.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("-"):
            continue
        decoded = decode_project_path(entry.name)
        if decoded is not None:
            project_cwd = str(decoded)
            project_key = decoded.name
        else:
            # Project directory was moved/deleted; fall back to the encoded
            # tail segment as the project key and skip CWD tagging.
            project_cwd = ""
            tail = entry.name.lstrip("-").split("-")[-1]
            project_key = tail or entry.name
        targets.append(
            Target(
                project_key=project_key,
                project_cwd=project_cwd,
                transcripts_dir=entry,
            )
        )
    return targets


def resolve_single_target(args: argparse.Namespace) -> Target:
    """Build the single-project Target from CLI flags + cwd defaults."""
    project_cwd = (
        str(Path(args.project_cwd).expanduser().resolve()) if args.project_cwd else ""
    )
    project_key = (
        args.project_key
        or (Path(project_cwd).name if project_cwd else Path.cwd().name)
        or "unknown"
    )

    if args.transcripts_dir:
        transcripts_dir = Path(args.transcripts_dir).expanduser()
    elif project_cwd:
        transcripts_dir = default_transcripts_dir(Path(project_cwd))
    else:
        transcripts_dir = DEFAULT_TRANSCRIPTS_DIR

    return Target(
        project_key=project_key,
        project_cwd=project_cwd,
        transcripts_dir=transcripts_dir,
    )


# ── Transcript scanning ─────────────────────────────────────────────────

def scan_transcripts(transcripts_dir: Path) -> list[TranscriptInfo]:
    """Scan all top-level JSONL transcript files (skip subagent files)."""
    results = []
    if not transcripts_dir.exists():
        logging.error("Transcripts directory not found: %s", transcripts_dir)
        return results

    for f in transcripts_dir.glob("*.jsonl"):
        # Skip if inside a subagents directory
        if "subagents" in str(f):
            continue
        stat = f.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).astimezone()
        session_id = f.stem  # UUID
        results.append(TranscriptInfo(
            path=f,
            size=stat.st_size,
            mtime=mtime,
            date=mtime.strftime("%Y-%m-%d"),
            session_id=session_id,
        ))

    results.sort(key=lambda t: t.mtime)
    return results


def _parse_claude_candidate(
    target: Target, info: TranscriptInfo
) -> HistoricalSession | None:
    if info.size < MIN_FILE_SIZE:
        return None
    try:
        session = parse_claude_transcript(
            info.path,
            {
                "session_id": info.session_id,
                "cwd": target.project_cwd,
                "project": target.project_key,
                "timestamp": info.mtime.isoformat(),
                "trigger": "historical",
            },
        )
    except (OSError, UnicodeError, ValueError):
        return None
    if not session.turns:
        return None
    return HistoricalSession(session, info.path, info.date, info.date)


def discover_claude_sessions(
    targets: Sequence[Target], *, concurrency: int = 1
) -> list[HistoricalSession]:
    """Normalize historical Claude transcripts with bounded parsing."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    candidates = [
        (target, info)
        for target in targets
        for info in scan_transcripts(target.transcripts_dir)
    ]
    if concurrency == 1:
        parsed = [_parse_claude_candidate(*candidate) for candidate in candidates]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            parsed = list(
                executor.map(lambda item: _parse_claude_candidate(*item), candidates)
            )

    discovered: list[HistoricalSession] = []
    seen: set[tuple[str, str]] = set()
    for historical in parsed:
        if historical is None:
            continue
        identity = (historical.session.session_id, historical.session.source_hash)
        if identity in seen:
            continue
        seen.add(identity)
        discovered.append(historical)
    return sorted(discovered, key=lambda item: (item.session.timestamp, str(item.path)))


def filter_historical_sessions(
    sessions: Sequence[HistoricalSession], args: argparse.Namespace
) -> list[HistoricalSession]:
    selected_dates = set(args.dates.split(",")) if args.dates else None
    return [
        item
        for item in sessions
        if (selected_dates is None or item.date in selected_dates)
        and (args.from_date is None or item.date >= args.from_date)
        and (args.to_date is None or item.date <= args.to_date)
    ]


def _plan_chunks(
    sessions: Sequence[HistoricalSession], max_cost: float | Decimal | None
) -> list[NormalizedSession]:
    planned = [
        chunk
        for historical in sessions
        for chunk in chunk_session(historical.session, CHUNK_TARGET_CHARS)
    ]
    if max_cost is not None:
        budget = Decimal(str(max_cost))
        estimate = Decimal(str(FLUSH_COST_ESTIMATE))
        planned = planned[: max(0, int(budget // estimate))]
    return planned


def _historical_writer(memory_home: Path):
    def write(job, text: str) -> Path | None:
        if text.strip() == "FLUSH_OK":
            return None
        payload = job.payload
        timestamp = _parse_timestamp(str(payload.get("timestamp", "")))
        return append_daily_entry(
            memory_home,
            text,
            project_key=job.project,
            cwd=job.cwd,
            agent=job.source_agent,
            capture_identity=hashlib.sha256(
                "\0".join(
                    (job.kind, job.source_agent, job.session_id, job.source_hash)
                ).encode()
            ).hexdigest(),
            now=timestamp,
        )

    return write


class _BoundedRouter:
    def __init__(self, router: object, semaphore: asyncio.Semaphore) -> None:
        self._router = router
        self._semaphore = semaphore

    async def generate_text(self, request):
        async with self._semaphore:
            return await self._router.generate_text(request)


def _default_router(config=None) -> ProviderRouter:
    config = config or load_config(os.environ)
    return ProviderRouter(
        CodexProvider(task_models=config.task_models),
        ClaudeProvider(model=config.claude_model),
    )


def _config_for_home(memory_home: Path):
    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(memory_home)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    environment.pop("AI_MEMORY_QUEUE_PATH", None)
    return load_config(environment)


def _identity(session: NormalizedSession) -> tuple[str, str, str, str]:
    return ("capture", session.agent, session.session_id, session.source_hash)


class QueueSnapshotUnstable(RuntimeError):
    """The live queue changed while a read-only snapshot was being captured."""


def _snapshot_signature(paths: Sequence[Path]) -> tuple[tuple[object, ...], ...]:
    signature: list[tuple[object, ...]] = []
    for path in paths:
        try:
            info = path.stat()
        except FileNotFoundError:
            signature.append((False,))
        else:
            signature.append(
                (
                    True,
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                )
            )
    return tuple(signature)


def _copy_stable_queue_snapshot(queue_path: Path, destination: Path) -> None:
    source_paths = (queue_path, Path(f"{queue_path}-wal"))
    destination_paths = (destination, Path(f"{destination}-wal"))
    for _ in range(3):
        before = _snapshot_signature(source_paths)
        for target in destination_paths:
            target.unlink(missing_ok=True)
        for source, target in zip(source_paths, destination_paths, strict=True):
            if source.is_file():
                shutil.copyfile(source, target)
        after = _snapshot_signature(source_paths)
        if before == after:
            return
    raise QueueSnapshotUnstable("queue changed during read-only snapshot")


def _read_queue_identities(queue_path: Path) -> dict[tuple[str, str, str, str], str]:
    if not queue_path.is_file() or queue_path.is_symlink():
        return {}
    try:
        with tempfile.TemporaryDirectory(prefix="memory-queue-preview-") as directory:
            snapshot = Path(directory) / "jobs.sqlite3"
            _copy_stable_queue_snapshot(queue_path, snapshot)
            connection = sqlite3.connect(snapshot)
            try:
                rows = connection.execute(
                    "SELECT kind, source_agent, session_id, source_hash, status FROM jobs"
                ).fetchall()
            finally:
                connection.close()
    except (OSError, sqlite3.DatabaseError):
        return {}
    return {
        (kind, agent, session_id, source_hash): status
        for kind, agent, session_id, source_hash, status in rows
    }


def _read_legacy_processed_sessions(state_path: Path) -> set[str]:
    if not state_path.is_file():
        return set()
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        processed = state.get("batch_flush", {}).get("processed_sessions", {})
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        return set()
    return set(processed) if isinstance(processed, dict) else set()


def _read_legacy_total_cost(state_path: Path) -> Decimal:
    if not state_path.is_file():
        return Decimal(0)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        value = state.get("batch_flush", {}).get("total_cost", 0)
        total = Decimal(str(value))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        AttributeError,
        InvalidOperation,
    ):
        return Decimal(0)
    if not total.is_finite() or total < 0:
        return Decimal(0)
    return total


def _dedup_plan(
    planned: Sequence[NormalizedSession], memory_home: Path, *, resume: bool
) -> tuple[list[NormalizedSession], int, int]:
    queue_identities = _read_queue_identities(
        memory_home / "scripts" / "jobs.sqlite3"
    )
    legacy = (
        _read_legacy_processed_sessions(memory_home / "scripts" / "state.json")
        if resume
        else set()
    )
    retained: list[NormalizedSession] = []
    skipped = 0
    would_create = 0
    seen: set[tuple[str, str, str, str]] = set()
    for session in planned:
        identity = _identity(session)
        if identity in seen:
            skipped += 1
            continue
        seen.add(identity)
        status = queue_identities.get(identity)
        legacy_match = (
            resume and session.agent == "claude" and session.session_id in legacy
        )
        if status is not None or legacy_match:
            skipped += 1
            if resume and (status == "succeeded" or legacy_match):
                continue
        else:
            would_create += 1
        retained.append(session)
    return retained, skipped, would_create


def _print_import_report(report: ImportReport, config) -> None:
    print(f"sessions: {report.sessions}")
    print(f"chunks: {report.chunks}")
    print(f"projects: {', '.join(report.projects) if report.projects else '(none)'}")
    print(f"dates: {', '.join(report.dates) if report.dates else '(none)'}")
    print(
        f"models: {config.task_models[TaskKind.EXTRACT]} "
        f"(Claude fallback: {config.claude_model})"
    )
    print(f"estimated tokens: {report.estimated_tokens}")
    print(f"enqueued: {report.enqueued}")
    print(f"succeeded: {report.succeeded}")
    print(f"skipped: {report.skipped}")
    print(f"failed: {report.failed}")
    print(f"dead: {report.dead}")


async def execute_historical_import(
    sessions: Sequence[HistoricalSession],
    args: argparse.Namespace,
    *,
    memory_home: Path | str,
    router: object | None = None,
) -> ImportReport:
    """Plan, enqueue, and drain historical captures through the live queue contract."""
    home = Path(memory_home).expanduser().resolve()
    config = _config_for_home(home)
    selected = filter_historical_sessions(sessions, args)
    cost_budget: Decimal | None = None
    if args.max_cost is not None:
        accumulated = _read_legacy_total_cost(home / "scripts" / "state.json")
        cost_budget = max(Decimal(0), Decimal(str(args.max_cost)) - accumulated)
    all_planned = _plan_chunks(selected, cost_budget)
    planned, deduplicated, would_create = _dedup_plan(
        all_planned, home, resume=args.resume
    )
    report = ImportReport(
        sessions=len(selected),
        chunks=len(planned),
        projects=tuple(sorted({item.session.project for item in selected})),
        dates=tuple(sorted({item.date for item in selected})),
        estimated_tokens=sum(
            max(1, (len(render_turns(chunk)) + 3) // 4) for chunk in planned
        ),
        enqueued=would_create if args.dry_run else 0,
        skipped=deduplicated,
    )
    if args.dry_run:
        _print_import_report(report, config)
        return report

    queue_path = home / "scripts" / "jobs.sqlite3"
    repository = QueueRepository(queue_path)
    try:
        enqueued = [repository.enqueue_capture(chunk) for chunk in planned]
        created = sum(result.created for result in enqueued)
        skipped = max(deduplicated, len(enqueued) - created)
        if planned:
            bounded_router = _BoundedRouter(
                router or _default_router(config), asyncio.Semaphore(args.concurrency)
            )
            workers = [
                MemoryWorker(
                    repository,
                    bounded_router,
                    daily_writer=_historical_writer(home),
                    owner=f"historical-{index}",
                    lock_path=home / "scripts" / "memory-worker.lock",
                )
                for index in range(args.concurrency)
            ]
            await asyncio.gather(*(worker.drain() for worker in workers))
        statuses = [repository.get_job(result.job_id).status for result in enqueued]
        report = replace(
            report,
            enqueued=created,
            succeeded=statuses.count("succeeded"),
            skipped=skipped,
            failed=statuses.count("failed"),
            dead=statuses.count("dead"),
        )
    finally:
        repository.close()
    _print_import_report(report, config)
    return report


# ── Conversation extraction ──────────────────────────────────────────────

def extract_full_conversation(transcript_path: Path) -> list[Turn]:
    """Normalize ALL Claude turns for historical chunking."""
    session = parse_claude_transcript(
        transcript_path,
        {"session_id": transcript_path.stem, "trigger": "historical"},
    )
    return [
        Turn(role=turn.role, text=turn.text, index=index)
        for index, turn in enumerate(session.turns)
    ]


def extract_tool_summary(transcript_path: Path, project_cwd: str = "") -> str:
    """Extract a brief summary of tool usage (files edited, commands run)."""
    edited_files: set[str] = set()
    commands: list[str] = []

    # Strip the project directory prefix from edited file paths so the summary
    # stays compact and project-relative.
    prefix = project_cwd.rstrip("/") + "/" if project_cwd else ""

    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = entry.get("message", {})
            if not isinstance(msg, dict):
                continue

            content = msg.get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue

                tool_name = block.get("name", "")
                tool_input = block.get("input", {})

                if tool_name in ("Edit", "Write") and isinstance(tool_input, dict):
                    fp = tool_input.get("file_path", "")
                    if fp:
                        if prefix and fp.startswith(prefix):
                            fp = fp[len(prefix):]
                        edited_files.add(fp)
                elif tool_name == "Bash" and isinstance(tool_input, dict):
                    cmd = tool_input.get("command", "")
                    if cmd and len(cmd) < 100:
                        commands.append(cmd)

    parts = []
    if edited_files:
        files_list = sorted(edited_files)[:15]  # Cap at 15
        parts.append(f"Files edited: {', '.join(files_list)}")
    if commands:
        # Show unique command prefixes
        unique_cmds = list(dict.fromkeys(cmd.split()[0] for cmd in commands if cmd.split()))[:10]
        parts.append(f"Commands used: {', '.join(unique_cmds)}")

    return "; ".join(parts) if parts else ""


# ── Chunking ─────────────────────────────────────────────────────────────

def chunk_conversation(turns: list[Turn], target_chars: int = CHUNK_TARGET_CHARS) -> list[Chunk]:
    """Delegate user-boundary chunking to the normalized transcript module."""
    if not turns:
        return []

    normalized = NormalizedSession(
        agent="claude",
        session_id="historical",
        project="unknown",
        cwd="",
        timestamp="",
        trigger="historical",
        turns=tuple(NormalizedTurn(turn.role, turn.text) for turn in turns),
        source_path="",
        source_hash="",
    )
    normalized_chunks = chunk_session(normalized, target_chars)
    chunks: list[Chunk] = []
    cursor = 0
    for normalized_chunk in normalized_chunks:
        chunk_turns = turns[cursor : cursor + len(normalized_chunk.turns)]
        text = render_turns(normalized_chunk)
        chunks.append(
            Chunk(
                text=text,
                char_count=len(text),
                position="",
                turn_range=(chunk_turns[0].index, chunk_turns[-1].index),
            )
        )
        cursor += len(normalized_chunk.turns)

    # Assign position labels
    n = len(chunks)
    for i, chunk in enumerate(chunks):
        if n == 1:
            chunk.position = "full"
        elif i == 0:
            chunk.position = "early"
        elif i == n - 1:
            chunk.position = "late"
        else:
            chunk.position = "mid"

    return chunks


# ── LLM flush ────────────────────────────────────────────────────────────

async def flush_chunk(
    chunk_text: str,
    session_meta: str,
    project_key: str,
    project_cwd: str,
) -> str:
    """Call Claude Agent SDK to extract knowledge from one chunk."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    project_block = f"**Project:** {project_key}"
    if project_cwd:
        project_block += f"\n**CWD:** {project_cwd}"

    prompt = f"""Review the conversation context below and respond with a concise summary
of important items that should be preserved in the daily log.
Do NOT use any tools — just return plain text.

This conversation took place in the following project:

{project_block}

Treat the project key as the canonical scope for everything you extract. Anything
project-specific (e.g. a coding pattern, a bug, a decision) should be described
as belonging to "{project_key}" so it can be filtered later by project.

{session_meta}

Format your response as a structured daily log entry with these sections:

**Context:** [One line about what the user was working on]

**Key Exchanges:**
- [Important Q&A or discussions]

**Decisions Made:**
- [Any decisions with rationale]

**Lessons Learned:**
- [Gotchas, patterns, or insights discovered]

**Action Items:**
- [Follow-ups or TODOs mentioned]

Skip anything that is:
- Routine tool calls or file reads
- Content that's trivial or obvious
- Trivial back-and-forth or clarification exchanges

Only include sections that have actual content. If nothing is worth saving,
respond with exactly: FLUSH_OK

## Conversation Context

{chunk_text}"""

    response = ""

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=str(ROOT),
                allowed_tools=[],
                max_turns=2,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response += block.text
            elif isinstance(message, ResultMessage):
                pass
    except Exception as e:
        import traceback
        logging.error("Agent SDK error: %s\n%s", e, traceback.format_exc())
        response = f"FLUSH_ERROR: {type(e).__name__}: {e}"

    return response


# ── Daily log writing ────────────────────────────────────────────────────

def write_daily_logs(
    extractions: list[Extraction],
    project_key: str,
    project_cwd: str,
) -> list[Path]:
    """Group extractions by date and write to daily log files.

    Matches the format used by flush.py's append_to_daily_log so retrieval
    can scope by project key.
    """
    grouped: dict[str, list[Extraction]] = {}
    for ext in extractions:
        grouped.setdefault(ext.date, []).append(ext)

    metadata_lines = [f"**Project:** {project_key}"]
    if project_cwd:
        metadata_lines.append(f"**CWD:** {project_cwd}")
    metadata_block = "\n".join(metadata_lines)

    written: list[Path] = []
    for date_str in sorted(grouped.keys()):
        log_path = DAILY_DIR / f"{date_str}.md"
        DAILY_DIR.mkdir(parents=True, exist_ok=True)

        if not log_path.exists():
            log_path.write_text(
                f"# Daily Log: {date_str}\n\n## Sessions\n\n## Memory Maintenance\n\n",
                encoding="utf-8",
            )

        with open(log_path, "a", encoding="utf-8") as f:
            for ext in grouped[date_str]:
                if "FLUSH_OK" in ext.content:
                    continue  # Skip empty extractions

                position_label = (
                    f" [{ext.chunk_position}]" if ext.chunk_position != "full" else ""
                )

                if "FLUSH_ERROR" in ext.content:
                    section = "Memory Flush"
                else:
                    section = "Session"

                header = (
                    f"### {section} [{project_key}] ({ext.time_str}){position_label}"
                )
                f.write(f"{header}\n\n{metadata_block}\n\n{ext.content}\n\n")

        written.append(log_path)

    return written


# ── State management ─────────────────────────────────────────────────────

def load_batch_state() -> dict:
    """Load batch flush state from state.json."""
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return state.get("batch_flush", {})
        except (json.JSONDecodeError, OSError):
            pass
    return {"processed_sessions": {}, "total_cost": 0.0}


def save_batch_state(batch_state: dict) -> None:
    """Save batch flush state to state.json."""
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}
    else:
        state = {}

    state["batch_flush"] = batch_state
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ── Per-project worker ──────────────────────────────────────────────────

async def process_one_project(
    args: argparse.Namespace,
    target: Target,
    batch_state: dict,
    starting_cost: float,
    label_prefix: str = "",
) -> tuple[float, bool]:
    """Process one project's transcripts. Writes its daily logs.

    Returns ``(cumulative_cost, halted_due_to_cost)``. ``halted_due_to_cost``
    is True if --max-cost stopped extraction before all sessions completed,
    signalling the orchestrator to skip remaining projects.
    """
    transcripts_dir = target.transcripts_dir
    project_key = target.project_key
    project_cwd = target.project_cwd

    transcripts = scan_transcripts(transcripts_dir)
    if not transcripts:
        logging.info("%sNo transcripts found in %s — skipping", label_prefix, transcripts_dir)
        return starting_cost, False

    if args.dates:
        date_set = set(args.dates.split(","))
        transcripts = [t for t in transcripts if t.date in date_set]

    skipped_tiny = [t for t in transcripts if t.size < MIN_FILE_SIZE]
    transcripts = [t for t in transcripts if t.size >= MIN_FILE_SIZE]

    processed = batch_state.setdefault("processed_sessions", {})

    if args.resume:
        before = len(transcripts)
        transcripts = [t for t in transcripts if t.session_id not in processed]
        logging.info(
            "%sResume: skipping %d already-processed sessions",
            label_prefix, before - len(transcripts),
        )

    if not transcripts:
        logging.info("%sNothing to process for [%s]", label_prefix, project_key)
        return starting_cost, False

    dates = sorted(set(t.date for t in transcripts))

    logging.info("")
    logging.info("%s=== Batch Flush Summary ===", label_prefix)
    logging.info("%sProject key: %s", label_prefix, project_key)
    if project_cwd:
        logging.info("%sProject cwd: %s", label_prefix, project_cwd)
    logging.info("%sTranscripts dir: %s", label_prefix, transcripts_dir)
    logging.info("%sTotal sessions found: %d", label_prefix, len(transcripts) + len(skipped_tiny))
    logging.info("%sSkipping %d tiny sessions (<5KB)", label_prefix, len(skipped_tiny))
    logging.info("%sProcessing: %d sessions across %d dates", label_prefix, len(transcripts), len(dates))
    logging.info(
        "%sDate range: %s to %s",
        label_prefix,
        dates[0] if dates else "n/a",
        dates[-1] if dates else "n/a",
    )
    logging.info("")

    total_chunks = 0
    session_plans: list[tuple[TranscriptInfo, int]] = []
    for t in transcripts:
        turns = extract_full_conversation(t.path)
        total_text = sum(len(turn.text) for turn in turns)
        est_chunks = max(1, total_text // CHUNK_TARGET_CHARS)
        session_plans.append((t, est_chunks))
        total_chunks += est_chunks

        if args.dry_run:
            logging.info(
                "%s  %s | %s | %6.1fKB | %4d turns | %6dK chars | ~%d chunks",
                label_prefix, t.date, t.session_id[:8], t.size / 1024, len(turns),
                total_text // 1000, est_chunks,
            )

    est_cost = total_chunks * FLUSH_COST_ESTIMATE
    logging.info("")
    logging.info("%sEstimated: %d chunks, ~$%.2f flush cost", label_prefix, total_chunks, est_cost)

    if args.dry_run:
        logging.info("%s(dry run — no LLM calls made)", label_prefix)
        return starting_cost, False

    # Process sessions
    all_extractions: list[Extraction] = []
    cumulative_cost = starting_cost
    chunk_num = 0
    halted = False

    for t, _est_chunks in session_plans:
        if args.max_cost and cumulative_cost >= args.max_cost:
            logging.info(
                "%sCost limit reached ($%.2f >= $%.2f), stopping",
                label_prefix, cumulative_cost, args.max_cost,
            )
            halted = True
            break

        logging.info("")
        logging.info("%sProcessing session %s (%s, %.1fKB)...",
                     label_prefix, t.session_id[:8], t.date, t.size / 1024)

        turns = extract_full_conversation(t.path)
        if not turns:
            logging.info("%s  No text turns found, skipping", label_prefix)
            continue

        tool_summary = extract_tool_summary(t.path, project_cwd)
        chunks = chunk_conversation(turns)

        logging.info("%s  %d turns, %d chunks", label_prefix, len(turns), len(chunks))

        for i, chunk in enumerate(chunks):
            chunk_num += 1

            if args.max_cost and cumulative_cost >= args.max_cost:
                halted = True
                break

            meta_parts = [f"Session date: {t.date}"]
            if len(chunks) > 1:
                meta_parts.append(f"Part {i + 1}/{len(chunks)} ({chunk.position})")
            if tool_summary:
                meta_parts.append(f"Tool activity: {tool_summary}")
            session_meta = "\n".join(meta_parts)

            logging.info(
                "%s  [%d/%d] Flushing chunk %d/%d (%s, %dK chars) — $%.2f spent",
                label_prefix, chunk_num, total_chunks, i + 1, len(chunks),
                chunk.position, chunk.char_count // 1000, cumulative_cost,
            )

            response = await flush_chunk(chunk.text, session_meta, project_key, project_cwd)
            cost = FLUSH_COST_ESTIMATE
            cumulative_cost += cost

            all_extractions.append(Extraction(
                session_id=t.session_id,
                date=t.date,
                time_str=t.mtime.strftime("%H:%M"),
                chunk_position=chunk.position,
                content=response,
                cost=cost,
            ))

            await asyncio.sleep(0.5)

        # Mark session as processed
        processed[t.session_id] = {
            "chunks": len(chunks),
            "cost": len(chunks) * FLUSH_COST_ESTIMATE,
            "flushed_at": datetime.now(timezone.utc).isoformat(),
            "project_key": project_key,
        }
        batch_state["processed_sessions"] = processed
        batch_state["total_cost"] = cumulative_cost
        save_batch_state(batch_state)

        if halted:
            break

    meaningful = [
        e for e in all_extractions
        if "FLUSH_OK" not in e.content and "FLUSH_ERROR" not in e.content
    ]
    logging.info("")
    logging.info("%s=== Writing Daily Logs ===", label_prefix)
    logging.info(
        "%sTotal extractions: %d (%d meaningful)",
        label_prefix, len(all_extractions), len(meaningful),
    )

    written = write_daily_logs(all_extractions, project_key, project_cwd)
    logging.info("%sWrote %d daily log files", label_prefix, len(written))
    for p in written:
        logging.info("%s  %s", label_prefix, p.name)

    return cumulative_cost, halted


# ── Orchestrator ─────────────────────────────────────────────────────────

async def run_batch(args: argparse.Namespace) -> ImportReport:
    """Discover selected agent sources and process them through the durable queue."""
    sessions: list[HistoricalSession] = []
    if args.source in {"claude", "all"}:
        if args.all_projects:
            if args.transcripts_dir or args.project_cwd or args.project_key:
                logging.warning(
                    "--all-projects ignores --transcripts-dir/--project-cwd/--project-key"
                )
            targets = discover_all_projects()
        else:
            targets = [resolve_single_target(args)]
        sessions.extend(discover_claude_sessions(targets, concurrency=args.concurrency))

    if args.source in {"codex", "all"}:
        codex_root = (
            Path(args.codex_sessions_dir).expanduser()
            if args.codex_sessions_dir
            else Path.home() / ".codex" / "sessions"
        )
        discovery = discover_codex_sessions(codex_root, concurrency=args.concurrency)
        sessions.extend(discovery.sessions)
        for path in discovery.malformed:
            logging.warning("Skipping malformed Codex transcript: %s", path)
        for path in discovery.duplicates:
            logging.info("Skipping duplicate Codex transcript: %s", path)
        for path in discovery.date_disagreements:
            logging.warning("Codex transcript timestamp disagrees with directory date: %s", path)

    config = load_config(os.environ)
    report = await execute_historical_import(
        sessions,
        args,
        memory_home=config.root_dir,
    )

    if args.compile and not args.dry_run and not report.failed and not report.dead:
        logging.info("")
        logging.info("=== Triggering Compilation ===")
        import subprocess
        cmd = ["uv", "run", "--directory", str(ROOT), "python", str(SCRIPTS_DIR / "compile.py"), "--all"]
        logging.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT))
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _date_string(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch extract knowledge from historical transcripts")
    parser.add_argument(
        "--source", choices=("claude", "codex", "all"), default="claude",
        help="Historical transcript source (default: claude)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be processed")
    parser.add_argument("--compile", action="store_true", help="Run compile.py after extraction")
    parser.add_argument("--max-cost", type=float, default=None, help="Stop after spending this much ($, global across all projects in --all-projects mode)")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed sessions")
    parser.add_argument("--dates", type=str, default=None, help="Comma-separated dates to process (YYYY-MM-DD)")
    parser.add_argument("--from-date", type=_date_string, default=None)
    parser.add_argument("--to-date", type=_date_string, default=None)
    parser.add_argument("--concurrency", type=_positive_int, default=2)
    parser.add_argument(
        "--codex-sessions-dir", type=str, default=None,
        help="Codex rollout root (defaults to ~/.codex/sessions)",
    )
    parser.add_argument(
        "--all-projects", action="store_true",
        help=f"Seed every project under {CLAUDE_PROJECTS_DIR}/. Overrides --transcripts-dir/--project-cwd/--project-key.",
    )
    parser.add_argument(
        "--transcripts-dir", type=str, default=None,
        help="Directory containing JSONL transcripts (defaults to ~/.claude/projects/<encoded-cwd>/)",
    )
    parser.add_argument(
        "--project-cwd", type=str, default=None,
        help="Working directory of the project being seeded; controls transcript auto-discovery and tool-path stripping. Defaults to current cwd.",
    )
    parser.add_argument(
        "--project-key", type=str, default=None,
        help="Project key for daily-log tagging (defaults to basename of --project-cwd or current cwd, matching the live hook).",
    )
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_cost is not None and args.max_cost <= 0:
        parser.error("--max-cost must be positive")
    if args.source in {"codex", "all"} and args.max_cost is not None:
        parser.error("--max-cost is legacy Claude-only accounting and cannot include Codex")
    if args.dates:
        values = args.dates.split(",")
        if not values or any(not value for value in values):
            parser.error("--dates requires comma-separated YYYY-MM-DD values")
        try:
            args.dates = ",".join(_date_string(value) for value in values)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    if args.from_date and args.to_date and args.from_date > args.to_date:
        parser.error("--from-date must not be after --to-date")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_cli_args(argv)
    configure_logging(dry_run=args.dry_run)
    previous_guard = os.environ.get("CLAUDE_INVOKED_BY")
    os.environ["CLAUDE_INVOKED_BY"] = previous_guard or "batch_flush"
    try:
        report = asyncio.run(run_batch(args))
    finally:
        if previous_guard is None:
            os.environ.pop("CLAUDE_INVOKED_BY", None)
        else:
            os.environ["CLAUDE_INVOKED_BY"] = previous_guard
    return 1 if report.failed or report.dead else 0


if __name__ == "__main__":
    raise SystemExit(main())
