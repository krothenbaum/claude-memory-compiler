"""Fast transcript snapshot and enqueue boundary used by live capture adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Callable, Literal, Mapping


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import load_config
from scripts.queue import EnqueueResult, Job, QueueRepository
from scripts.transcripts import parse_claude_transcript, parse_codex_transcript


CAPTURE_DB_BUSY_TIMEOUT_MS = 250


class UnsafeSpoolError(ValueError):
    """Raised when a queue-owned snapshot path fails local safety checks."""


@dataclass(frozen=True)
class CaptureOutcome:
    """Discriminated result for an enqueue, deduplication, or guarded skip."""

    status: Literal["enqueued", "deduplicated", "skipped"]
    job: Job | None = None
    reason: str | None = None

    @classmethod
    def from_enqueue(cls, result: EnqueueResult) -> "CaptureOutcome":
        return cls("enqueued" if result.created else "deduplicated", result.job)

    @property
    def created(self) -> bool:
        return self.status == "enqueued"

    @property
    def inserted(self) -> bool:
        return self.created

    @property
    def job_id(self) -> int | None:
        return self.job.id if self.job is not None else None


def _guarded_outcome(env: Mapping[str, str]) -> CaptureOutcome | None:
    if env.get("AI_MEMORY_INTERNAL_JOB") == "1":
        return CaptureOutcome("skipped", reason="internal_job")
    if "CLAUDE_INVOKED_BY" in env:
        return CaptureOutcome("skipped", reason="legacy_internal_job")
    return None


def launch_worker(memory_home: Path | str) -> None:
    """Start a detached, non-interactive worker drain and return immediately."""
    root = Path(memory_home).expanduser().resolve()
    command = [
        "uv",
        "run",
        "--directory",
        str(root),
        "python",
        str(root / "scripts" / "worker.py"),
        "--drain",
    ]
    options: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if platform.system() == "Windows":
        options["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        options["start_new_session"] = True
    subprocess.Popen(command, **options)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        if platform.system() == "Windows":
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError:
        if platform.system() != "Windows":
            raise
    finally:
        os.close(descriptor)


def _safe_spool_directory(root: Path) -> Path:
    if root.is_symlink():
        raise UnsafeSpoolError("memory root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise UnsafeSpoolError("memory root must be a real directory")

    scripts_dir = root / "scripts"
    if scripts_dir.is_symlink():
        raise UnsafeSpoolError("scripts spool component must not be a symlink")
    scripts_dir.mkdir(exist_ok=True)
    if scripts_dir.is_symlink() or not scripts_dir.is_dir():
        raise UnsafeSpoolError("scripts spool component must be a real directory")

    spool_dir = scripts_dir / "spool"
    if spool_dir.is_symlink():
        raise UnsafeSpoolError("spool component must not be a symlink")
    spool_dir.mkdir(exist_ok=True, mode=0o700)
    if spool_dir.is_symlink() or not spool_dir.is_dir():
        raise UnsafeSpoolError("spool component must be a real directory")
    if spool_dir.resolve() != root.resolve() / "scripts" / "spool":
        raise UnsafeSpoolError("spool directory escapes the memory root")
    try:
        spool_dir.chmod(0o700)
    except OSError:
        pass
    return spool_dir


def _private_spool_copy(source: Path, spool_dir: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="capture-", suffix=".jsonl", dir=spool_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        _fsync_file(temporary)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _snapshot_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_existing_snapshot(
    path: Path,
    *,
    spool_dir: Path,
    expected_digest: str,
    expected_size: int,
) -> None:
    if path.parent != spool_dir or path.parent.resolve() != spool_dir.resolve():
        raise UnsafeSpoolError("snapshot destination escapes the spool directory")
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise UnsafeSpoolError("snapshot destination is not a regular private file")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise UnsafeSpoolError("snapshot destination has an unsafe owner")
    if info.st_nlink != 1 or stat.S_IMODE(info.st_mode) & 0o077:
        raise UnsafeSpoolError("snapshot destination has unsafe identity or permissions")
    if info.st_size != expected_size or _snapshot_digest(path) != expected_digest:
        raise UnsafeSpoolError("snapshot destination content does not match capture")


def _publish_snapshot(temporary: Path, destination: Path) -> None:
    """Publish once without allowing concurrent equivalent captures to overwrite."""
    spool_dir = temporary.parent
    expected_size = temporary.stat().st_size
    expected_digest = _snapshot_digest(temporary)
    try:
        os.link(temporary, destination)
    except FileExistsError:
        _validate_existing_snapshot(
            destination,
            spool_dir=spool_dir,
            expected_digest=expected_digest,
            expected_size=expected_size,
        )
        temporary.unlink(missing_ok=True)
    except OSError:
        try:
            with temporary.open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            try:
                destination.chmod(0o600)
            except OSError:
                pass
        except FileExistsError:
            _validate_existing_snapshot(
                destination,
                spool_dir=spool_dir,
                expected_digest=expected_digest,
                expected_size=expected_size,
            )
            temporary.unlink(missing_ok=True)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        else:
            temporary.unlink(missing_ok=True)
    else:
        temporary.unlink(missing_ok=True)
    _validate_existing_snapshot(
        destination,
        spool_dir=spool_dir,
        expected_digest=expected_digest,
        expected_size=expected_size,
    )
    _fsync_directory(spool_dir)


def _retain_failed_snapshot(temporary: Path, source_agent: str) -> None:
    digest = _snapshot_digest(temporary)
    destination = temporary.with_name(f"failed-{source_agent}-{digest}.jsonl")
    _publish_snapshot(temporary, destination)


def capture_transcript(
    transcript_path: Path | str,
    *,
    source_agent: str,
    metadata: Mapping[str, object],
    memory_home: Path | str | None = None,
    queue: QueueRepository | None = None,
    launcher: Callable[[Path], None] = launch_worker,
    clock: Callable[[], datetime] | None = None,
    limits: object = None,
    env: Mapping[str, str] | None = None,
) -> CaptureOutcome:
    """Snapshot, normalize, enqueue, and wake a worker without invoking a model."""
    source_env = os.environ if env is None else env
    guarded = _guarded_outcome(source_env)
    if guarded is not None:
        return guarded
    if source_agent not in {"claude", "codex"}:
        raise ValueError("source_agent must be claude or codex")
    source = Path(transcript_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("transcript_path must name a regular file")

    if memory_home is None:
        root = load_config(source_env).root_dir
    else:
        root = Path(os.path.abspath(Path(memory_home).expanduser()))
    spool_dir = _safe_spool_directory(root)
    temporary = _private_spool_copy(source, spool_dir)
    snapshot_digest = _snapshot_digest(temporary)
    snapshot_size = temporary.stat().st_size
    parser = parse_claude_transcript if source_agent == "claude" else parse_codex_transcript
    try:
        normalized = parser(temporary, metadata, limits=limits)
        final_snapshot = temporary.with_name(
            f"{source_agent}-{normalized.source_hash}.jsonl"
        )
        _publish_snapshot(temporary, final_snapshot)
        normalized = replace(normalized, source_path=str(final_snapshot))

        owns_queue = queue is None
        if queue is None:
            queue_config = load_config(
                {
                    **source_env,
                    "AI_MEMORY_HOME": str(root),
                    "CLAUDE_MEMORY_HOME": str(root),
                }
            )
            repository = QueueRepository(
                queue_config.queue_path,
                busy_timeout_ms=CAPTURE_DB_BUSY_TIMEOUT_MS,
                **({"clock": clock} if clock is not None else {}),
            )
        else:
            repository = queue
        try:
            _validate_existing_snapshot(
                final_snapshot,
                spool_dir=spool_dir,
                expected_digest=snapshot_digest,
                expected_size=snapshot_size,
            )
            result = repository.enqueue_capture(normalized)
        finally:
            if owns_queue:
                repository.close()
        launcher(root)
        return CaptureOutcome.from_enqueue(result)
    except BaseException:
        # A private snapshot is the recovery boundary even when parsing fails.
        if temporary.exists():
            try:
                _retain_failed_snapshot(temporary, source_agent)
            except Exception:
                # Keep the already-private random temporary file when publishing fails.
                pass
        raise


def enqueue_hook_input(
    hook_input: Mapping[str, object],
    *,
    source_agent: str,
    trigger: str,
    memory_home: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    **kwargs: object,
) -> CaptureOutcome:
    """Resolve the common hook payload fields and enqueue its transcript."""
    source_env = os.environ if env is None else env
    guarded = _guarded_outcome(source_env)
    if guarded is not None:
        return guarded
    transcript = hook_input.get("transcript_path")
    if not isinstance(transcript, str) or not transcript:
        raise ValueError("hook input has no transcript_path")
    metadata = {
        "session_id": hook_input.get("session_id", ""),
        "cwd": hook_input.get("cwd", ""),
        "timestamp": hook_input.get("timestamp", ""),
        "project": hook_input.get("project", ""),
        "trigger": trigger,
    }
    return capture_transcript(
        transcript,
        source_agent=source_agent,
        metadata=metadata,
        memory_home=memory_home,
        env=source_env,
        **kwargs,
    )
