"""Fast transcript snapshot and enqueue boundary used by live capture adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import os
from pathlib import Path
import platform
import shutil
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


def _private_spool_copy(source: Path, spool_dir: Path) -> Path:
    spool_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        spool_dir.chmod(0o700)
    except OSError:
        pass
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
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish_snapshot(temporary: Path, destination: Path) -> None:
    """Publish once without allowing concurrent equivalent captures to overwrite."""
    try:
        os.link(temporary, destination)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        return
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
            temporary.unlink(missing_ok=True)
            return
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        temporary.unlink(missing_ok=True)
    else:
        temporary.unlink(missing_ok=True)


def _retain_failed_snapshot(temporary: Path, source_agent: str) -> None:
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
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
        root = load_config(os.environ).root_dir
    else:
        root = Path(memory_home).expanduser().resolve()
    temporary = _private_spool_copy(source, root / "scripts" / "spool")
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
                **({"clock": clock} if clock is not None else {}),
            )
        else:
            repository = queue
        try:
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
