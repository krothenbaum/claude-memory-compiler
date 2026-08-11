"""Detached drain worker for durable memory jobs."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timedelta, timezone
import inspect
import os
from pathlib import Path
import random
import sys
from typing import Awaitable, Callable
import uuid


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import load_config
from scripts.providers import (
    ClaudeProvider,
    CodexProvider,
    ProviderRouter,
    RoutedResult,
    TaskKind,
    TextRequest,
)
from scripts.queue import Job, QueueRepository


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DailyWriterUnavailable(RuntimeError):
    """Raised until Task 6 provides the serialized daily writer."""


def daily_writer_boundary(job: Job, text: str) -> None:
    """Task 6 replaces this boundary with a writer-lock-backed implementation."""
    raise DailyWriterUnavailable(
        f"daily writer is not configured for capture job {job.id}"
    )


class SingletonDrainLock:
    """A process-owned lock file with conservative stale-owner recovery."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self._token = f"{os.getpid()}:{uuid.uuid4()}"
        self._owned = False

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except (PermissionError, OSError):
            return True
        return True

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                try:
                    owner = self.path.read_text(encoding="utf-8").strip()
                    pid = int(owner.partition(":")[0])
                except (OSError, ValueError):
                    pid = -1
                if self._pid_is_alive(pid):
                    return False
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    return False
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(self._token)
                stream.flush()
                os.fsync(stream.fileno())
            self._owned = True
            return True
        return False

    def release(self) -> None:
        if not self._owned:
            return
        try:
            if self.path.read_text(encoding="utf-8").strip() == self._token:
                self.path.unlink(missing_ok=True)
        finally:
            self._owned = False

    def __enter__(self) -> "SingletonDrainLock":
        if not self.acquire():
            raise RuntimeError("worker drain lock is already owned")
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class MemoryWorker:
    """Claim and process jobs while preserving retry and attempt history."""

    def __init__(
        self,
        queue: QueueRepository,
        router: object,
        *,
        daily_writer: Callable[[Job, str], object] = daily_writer_boundary,
        clock: Callable[[], datetime] = _utc_now,
        owner: str | None = None,
        lock_path: Path | str | None = None,
        lease_seconds: int = 1_020,
        retry_base_seconds: int = 5,
        retry_max_seconds: int = 900,
        jitter: Callable[[], float] = random.random,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.queue = queue
        self.router = router
        self.daily_writer = daily_writer
        self.clock = clock
        self.owner = owner or str(uuid.uuid4())
        self.lock_path = Path(lock_path or ROOT / "scripts" / "memory-worker.lock")
        self.lease_seconds = lease_seconds
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.jitter = jitter
        self.sleeper = sleeper

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _renew(self, job_id: int) -> bool:
        return self.queue.renew(
            job_id,
            self.owner,
            self._now() + timedelta(seconds=self.lease_seconds),
        )

    async def _heartbeat(self, job_id: int, stopped: asyncio.Event) -> None:
        interval = max(1.0, self.lease_seconds / 3)
        while not stopped.is_set():
            await self.sleeper(interval)
            if stopped.is_set():
                return
            if not self._renew(job_id):
                return

    def _retry_at(self, job: Job) -> datetime:
        exponent = max(0, job.attempt_count - 1)
        base = min(self.retry_max_seconds, self.retry_base_seconds * (2**exponent))
        jitter_seconds = min(base * 0.25, max(0.0, self.jitter()) * base * 0.25)
        return self._now() + timedelta(seconds=base + jitter_seconds)

    @staticmethod
    def _failure_reason(result: RoutedResult) -> str:
        return "; ".join(
            f"{attempt.provider}:{attempt.outcome}:{attempt.reason or attempt.outcome}"
            for attempt in result.attempts
        ) or f"{result.provider}:{result.outcome}:{result.reason or result.outcome}"

    async def _write(self, job: Job, text: str) -> None:
        outcome = self.daily_writer(job, text)
        if inspect.isawaitable(outcome):
            await outcome

    async def process(self, job: Job) -> None:
        stopped = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(job.id, stopped))
        try:
            request = TextRequest(
                task=TaskKind.EXTRACT,
                prompt=job.payload.get("rendered_context", ""),
                cwd=Path(job.cwd or ROOT),
                timeout_seconds=max(1, self.lease_seconds - 60),
            )
            result = await self.router.generate_text(request)
            for attempt in result.attempts:
                self.queue.record_attempt(job.id, attempt)
            if result.outcome != "success":
                self.queue.retry(
                    job.id,
                    self.owner,
                    self._failure_reason(result),
                    self._retry_at(job),
                )
                return
            if not self._renew(job.id):
                raise RuntimeError("capture lease was lost before daily dispatch")
            await self._write(job, result.text)
            self.queue.complete(job.id, self.owner)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            current = self.queue.get_job(job.id)
            if current.status == "leased" and current.lease_owner == self.owner:
                self.queue.retry(
                    job.id,
                    self.owner,
                    str(exc) or type(exc).__name__,
                    self._retry_at(current),
                )
        finally:
            stopped.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

    async def drain(self) -> int:
        processed = 0
        self.queue.recover_stale(self._now())
        while True:
            job = self.queue.claim_next(
                self.owner, self._now(), self.lease_seconds
            )
            if job is None:
                return processed
            await self.process(job)
            processed += 1

    async def run_drain(self) -> int:
        lock = SingletonDrainLock(self.lock_path)
        if not lock.acquire():
            return 0
        try:
            return await self.drain()
        finally:
            lock.release()


def _default_worker() -> tuple[MemoryWorker, QueueRepository]:
    config = load_config(os.environ)
    repository = QueueRepository(config.queue_path)
    router = ProviderRouter(
        CodexProvider(task_models=config.task_models),
        ClaudeProvider(model=config.claude_model),
    )
    worker = MemoryWorker(
        repository,
        router,
        lock_path=config.root_dir / "scripts" / "memory-worker.lock",
        lease_seconds=config.job_timeout_seconds + 120,
    )
    return worker, repository


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drain durable memory jobs")
    parser.add_argument(
        "--drain", action="store_true", help="process available jobs and exit"
    )
    args = parser.parse_args(argv)
    if not args.drain:
        parser.error("--drain is required")
    worker, repository = _default_worker()
    try:
        asyncio.run(worker.run_drain())
    finally:
        repository.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
