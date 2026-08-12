"""Fast transcript snapshot and enqueue boundary used by live capture adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Literal, Mapping


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import load_config
from scripts.queue import EnqueueResult, Job, QueueRepository
from scripts.transcripts import parse_claude_transcript, parse_codex_transcript


CAPTURE_DB_BUSY_TIMEOUT_MS = 250
SNAPSHOT_LINK_RETRY_ATTEMPTS = 3
SNAPSHOT_LINK_RETRY_SECONDS = 0.01
_snapshot_retry_wait = time.sleep


class UnsafeSpoolError(ValueError):
    """Raised when a queue-owned snapshot path fails local safety checks."""


class CaptureDeadlineExceeded(TimeoutError):
    """Raised before queue commit when a bounded live capture runs out of time."""


def _check_deadline(
    deadline: float | None,
    monotonic: Callable[[], float],
) -> None:
    if deadline is not None and monotonic() >= deadline:
        raise CaptureDeadlineExceeded("capture deadline exhausted")


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


def _private_spool_copy(
    source: Path,
    spool_dir: Path,
    *,
    deadline: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    capture_token: str | None = None,
) -> Path:
    prefix = f"capture-{capture_token}-" if capture_token else "capture-"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=prefix, suffix=".jsonl", dir=spool_dir
    )
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        _check_deadline(deadline, monotonic)
        with source.open("rb") as input_stream, os.fdopen(
            descriptor, "wb"
        ) as output_stream:
            while True:
                _check_deadline(deadline, monotonic)
                block = input_stream.read(1024 * 1024)
                if not block:
                    break
                output_stream.write(block)
            output_stream.flush()
        _fsync_file(temporary)
        _check_deadline(deadline, monotonic)
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        return temporary
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _snapshot_digest(
    path: Path,
    *,
    deadline: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            _check_deadline(deadline, monotonic)
            digest.update(block)
    _check_deadline(deadline, monotonic)
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
    for attempt in range(SNAPSHOT_LINK_RETRY_ATTEMPTS):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise UnsafeSpoolError("snapshot destination is not a regular private file")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise UnsafeSpoolError("snapshot destination has an unsafe owner")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise UnsafeSpoolError("snapshot destination has unsafe permissions")
        if info.st_size != expected_size or _snapshot_digest(path) != expected_digest:
            raise UnsafeSpoolError("snapshot destination content does not match capture")
        if info.st_nlink == 1:
            return
        if info.st_nlink == 2 and attempt + 1 < SNAPSHOT_LINK_RETRY_ATTEMPTS:
            _snapshot_retry_wait(SNAPSHOT_LINK_RETRY_SECONDS)
            continue
        raise UnsafeSpoolError("snapshot destination has unsafe identity")


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


def _retain_failed_snapshot(
    temporary: Path,
    source_agent: str,
    capture_token: str | None = None,
) -> None:
    digest = _snapshot_digest(temporary)
    owner = f"-{capture_token}" if capture_token else ""
    destination = temporary.with_name(
        f"failed-{source_agent}{owner}-{digest}.jsonl"
    )
    _publish_snapshot(temporary, destination)


def _detached_process_options() -> dict[str, object]:
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
    return options


def _start_detached_worker_wake(root: Path) -> None:
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--wake-worker", str(root)],
        **_detached_process_options(),
    )


def _start_independent_callable_wake(
    launcher: Callable[[Path], None], root: Path
) -> bool:
    """Start an injected wake outside this process when POSIX fork is available."""
    if (
        threading.current_thread() is not threading.main_thread()
        or threading.active_count() != 1
    ):
        return False
    fork = getattr(os, "fork", None)
    if fork is None:
        return False
    try:
        child = fork()
    except OSError:
        return False
    if child == 0:  # pragma: no cover - exercised through subprocess marker test.
        try:
            os.setsid()
            grandchild = fork()
            if grandchild:
                os._exit(0)
            descriptor = os.open(os.devnull, os.O_RDWR)
            try:
                for standard_descriptor in (0, 1, 2):
                    os.dup2(descriptor, standard_descriptor)
            finally:
                if descriptor > 2:
                    os.close(descriptor)
            try:
                launcher(root)
            except Exception:
                pass
        finally:
            os._exit(0)
    try:
        os.waitpid(child, 0)
    except OSError:
        pass
    return True


def _wake_after_commit(
    launcher: Callable[[Path], None],
    root: Path,
    *,
    deadline: float | None,
    monotonic: Callable[[], float],
) -> None:
    """Best-effort worker wake that can never roll back a committed capture."""
    if launcher is launch_worker:
        try:
            _start_detached_worker_wake(root)
        except Exception:
            pass
        return
    if deadline is None:
        try:
            launcher(root)
        except Exception:
            pass
        return

    if _start_independent_callable_wake(launcher, root):
        return

    def wake() -> None:
        try:
            launcher(root)
        except Exception:
            pass

    thread = threading.Thread(target=wake, daemon=True)
    thread.start()
    thread.join(timeout=max(0.0, deadline - monotonic()))


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
    deadline: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    capture_token: str | None = None,
) -> CaptureOutcome:
    """Snapshot, normalize, enqueue, and wake a worker without invoking a model."""
    source_env = os.environ if env is None else env
    guarded = _guarded_outcome(source_env)
    if guarded is not None:
        return guarded
    if source_agent not in {"claude", "codex"}:
        raise ValueError("source_agent must be claude or codex")
    if capture_token is not None and (
        not capture_token
        or len(capture_token) > 64
        or any(not (character.isalnum() or character in "-_") for character in capture_token)
    ):
        raise ValueError("capture_token must contain only letters, digits, '-' or '_'")
    _check_deadline(deadline, monotonic)
    source = Path(transcript_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("transcript_path must name a regular file")

    if memory_home is None:
        root = load_config(source_env).root_dir
    else:
        root = Path(os.path.abspath(Path(memory_home).expanduser()))
    _check_deadline(deadline, monotonic)
    spool_dir = _safe_spool_directory(root)
    temporary = _private_spool_copy(
        source,
        spool_dir,
        deadline=deadline,
        monotonic=monotonic,
        capture_token=capture_token,
    )
    committed_result: EnqueueResult | None = None
    try:
        snapshot_digest = _snapshot_digest(
            temporary, deadline=deadline, monotonic=monotonic
        )
        snapshot_size = temporary.stat().st_size
        parser = (
            parse_claude_transcript
            if source_agent == "claude"
            else parse_codex_transcript
        )
        _check_deadline(deadline, monotonic)
        normalized = parser(temporary, metadata, limits=limits)
        _check_deadline(deadline, monotonic)
        if deadline is None:
            final_snapshot = temporary.with_name(
                f"{source_agent}-{normalized.source_hash}.jsonl"
            )
            _publish_snapshot(temporary, final_snapshot)
        else:
            # Deadline captures retain a unique ownership-tagged snapshot.
            # This lets a timed-out parent clean only its own uncommitted file.
            final_snapshot = temporary
            _fsync_directory(spool_dir)
        normalized = replace(normalized, source_path=str(final_snapshot))

        owns_queue = queue is None
        if queue is None:
            _check_deadline(deadline, monotonic)
            queue_config = load_config(
                {
                    **source_env,
                    "AI_MEMORY_HOME": str(root),
                    "CLAUDE_MEMORY_HOME": str(root),
                }
            )
            busy_timeout_ms = CAPTURE_DB_BUSY_TIMEOUT_MS
            if deadline is not None:
                remaining_seconds = deadline - monotonic()
                if remaining_seconds <= 0:
                    raise CaptureDeadlineExceeded("capture deadline exhausted")
                remaining_ms = int(remaining_seconds * 1_000)
                busy_timeout_ms = max(
                    1, min(CAPTURE_DB_BUSY_TIMEOUT_MS, remaining_ms)
                )
            repository_options = {
                "busy_timeout_ms": busy_timeout_ms,
                **({"clock": clock} if clock is not None else {}),
            }
            for attempt in range(25):
                try:
                    repository = QueueRepository(
                        queue_config.queue_path,
                        memory_home=queue_config.root_dir,
                        **repository_options,
                    )
                    break
                except (FileExistsError, sqlite3.OperationalError) as error:
                    transient = isinstance(error, FileExistsError) or any(
                        marker in str(error).lower()
                        for marker in ("locked", "busy")
                    )
                    if not transient or attempt == 24:
                        raise
                    _check_deadline(deadline, monotonic)
                    time.sleep(0.01)
            else:  # pragma: no cover - the bounded loop always breaks or raises.
                raise RuntimeError("queue open retry loop exhausted")
        else:
            repository = queue
        try:
            _check_deadline(deadline, monotonic)
            if deadline is None:
                _validate_existing_snapshot(
                    final_snapshot,
                    spool_dir=spool_dir,
                    expected_digest=snapshot_digest,
                    expected_size=snapshot_size,
                )
            result = repository.enqueue_capture(normalized)
            committed_result = result
        finally:
            if owns_queue:
                repository.close()
        if deadline is not None and not result.created:
            final_snapshot.unlink(missing_ok=True)
        _wake_after_commit(
            launcher,
            root,
            deadline=deadline,
            monotonic=monotonic,
        )
        return CaptureOutcome.from_enqueue(result)
    except CaptureDeadlineExceeded:
        if committed_result is None and temporary.exists():
            temporary.unlink(missing_ok=True)
            try:
                _fsync_directory(spool_dir)
            except OSError:
                pass
        raise
    except BaseException:
        if committed_result is not None:
            if deadline is not None and not committed_result.created:
                temporary.unlink(missing_ok=True)
            _wake_after_commit(
                launcher,
                root,
                deadline=deadline,
                monotonic=monotonic,
            )
            return CaptureOutcome.from_enqueue(committed_result)
        # A private snapshot is the recovery boundary even when parsing fails.
        if temporary.exists():
            try:
                _retain_failed_snapshot(temporary, source_agent, capture_token)
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


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--wake-worker":
    launch_worker(sys.argv[2])
