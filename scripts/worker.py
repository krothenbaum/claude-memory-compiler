"""Detached drain worker for durable memory jobs."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timedelta, timezone
import errno
import inspect
import os
from pathlib import Path
import random
import stat
import sys
from typing import Awaitable, Callable
import uuid

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows branch.
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX branch.
    msvcrt = None


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import load_config
from scripts.providers import (
    ClaudeProvider,
    CodexProvider,
    ProviderResult,
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


class LeaseLostError(RuntimeError):
    """Raised when the active worker can no longer prove lease ownership."""


class SingletonDrainLock:
    """Cross-platform OS file lock; file contents are diagnostic only."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(os.path.abspath(Path(path).expanduser()))
        self._token = f"{os.getpid()}:{uuid.uuid4()}"
        self._descriptor: int | None = None

    @staticmethod
    def _try_os_lock(descriptor: int) -> bool:
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt is not None:  # pragma: no cover - Windows branch.
                os.lseek(descriptor, 0, os.SEEK_SET)
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                    os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - unsupported Python platform.
                raise RuntimeError("no supported OS file-lock implementation")
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        return True

    @staticmethod
    def _unlock(descriptor: int) -> None:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        elif msvcrt is not None:  # pragma: no cover - Windows branch.
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)

    def acquire(self) -> bool:
        if self._descriptor is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.is_symlink() or self.path.is_symlink():
            raise ValueError("worker lock path must not be a symlink")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            os.close(descriptor)
            raise ValueError("worker lock path must be a regular file")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            os.close(descriptor)
            raise ValueError("worker lock path has an unsafe owner")
        if info.st_nlink != 1:
            os.close(descriptor)
            raise ValueError("worker lock path must not be hard-linked")
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        if not self._try_os_lock(descriptor):
            os.close(descriptor)
            return False
        try:
            payload = self._token.encode()
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, payload)
            os.fsync(descriptor)
        except BaseException:
            self._unlock(descriptor)
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return True

    def release(self) -> None:
        if self._descriptor is None:
            return
        try:
            self._unlock(self._descriptor)
        finally:
            os.close(self._descriptor)
            self._descriptor = None

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
        router: object | None = None,
        *,
        router_factory: Callable[[Callable[[ProviderResult], None]], object] | None = None,
        daily_writer: Callable[[Job, str], object] = daily_writer_boundary,
        clock: Callable[[], datetime] = _utc_now,
        owner: str | None = None,
        lock_path: Path | str | None = None,
        lease_seconds: int = 1_020,
        provider_timeout_seconds: int | None = None,
        retry_base_seconds: int = 5,
        retry_max_seconds: int = 900,
        max_idle_sleep_seconds: float = 1.0,
        jitter: Callable[[], float] = random.random,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
        heartbeat_sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if provider_timeout_seconds is not None and provider_timeout_seconds <= 0:
            raise ValueError("provider_timeout_seconds must be positive")
        if max_idle_sleep_seconds <= 0:
            raise ValueError("max_idle_sleep_seconds must be positive")
        if (router is None) == (router_factory is None):
            raise ValueError("provide exactly one of router or router_factory")
        self.queue = queue
        self.router = router
        self.router_factory = router_factory
        self.daily_writer = daily_writer
        self.clock = clock
        self.owner = owner or str(uuid.uuid4())
        self.lock_path = Path(lock_path or ROOT / "scripts" / "memory-worker.lock")
        self.lease_seconds = lease_seconds
        self.provider_timeout_seconds = provider_timeout_seconds or max(
            1, lease_seconds - 120
        )
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.max_idle_sleep_seconds = max_idle_sleep_seconds
        self.jitter = jitter
        self.sleeper = sleeper
        self.heartbeat_sleeper = heartbeat_sleeper

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

    def _require_lease(self, job_id: int) -> None:
        try:
            owned = self._renew(job_id)
        except Exception as exc:
            raise LeaseLostError("lease renewal failed") from exc
        if not owned:
            raise LeaseLostError("lease ownership was lost")

    async def _heartbeat(self, job_id: int) -> None:
        interval = max(1.0, self.lease_seconds / 3)
        while True:
            await self.heartbeat_sleeper(interval)
            self._require_lease(job_id)

    async def _run_with_lease(self, job_id: int, awaitable: Awaitable[object]) -> object:
        operation = asyncio.create_task(awaitable)
        heartbeat = asyncio.create_task(self._heartbeat(job_id))
        try:
            done, _ = await asyncio.wait(
                {operation, heartbeat}, return_when=asyncio.FIRST_COMPLETED
            )
        except BaseException:
            operation.cancel()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await operation
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat
            raise
        if heartbeat in done:
            operation.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await operation
            try:
                await heartbeat
            except LeaseLostError:
                raise
            except Exception as exc:
                raise LeaseLostError("lease heartbeat failed") from exc
            raise LeaseLostError("lease heartbeat stopped unexpectedly")
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
        return await operation

    def _retry_at(self, job: Job) -> datetime:
        exponent = min(30, max(0, job.attempt_count - 1))
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
        try:
            request = TextRequest(
                task=TaskKind.EXTRACT,
                prompt=job.payload.get("rendered_context", ""),
                cwd=Path(job.cwd or ROOT),
                timeout_seconds=self.provider_timeout_seconds,
            )
            eagerly_persisted = self.router_factory is not None
            if self.router_factory is not None:
                router = self.router_factory(
                    lambda attempt: self.queue.record_attempt(job.id, attempt)
                )
            else:
                router = self.router
            result = await self._run_with_lease(job.id, router.generate_text(request))
            if not eagerly_persisted:
                for attempt in result.attempts:
                    self.queue.record_attempt(job.id, attempt)
            if result.outcome != "success":
                self._require_lease(job.id)
                self.queue.retry(
                    job.id,
                    self.owner,
                    self._failure_reason(result),
                    self._retry_at(job),
                )
                return
            self._require_lease(job.id)
            await self._run_with_lease(job.id, self._write(job, result.text))
            self._require_lease(job.id)
            self.queue.complete(job.id, self.owner)
        except LeaseLostError:
            return
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

    async def _drain(self, release_if_idle: Callable[[], None] | None) -> int:
        processed = 0
        while True:
            self.queue.recover_stale(self._now())
            job = self.queue.claim_next(
                self.owner, self._now(), self.lease_seconds
            )
            if job is None:
                wake_at = self.queue.next_wake_at()
                if wake_at is None:
                    if release_if_idle is None:
                        return processed
                    if self.queue.release_worker_lock_if_idle(release_if_idle):
                        return processed
                    continue
                delay = max(0.0, (wake_at - self._now()).total_seconds())
                await self.sleeper(min(delay, self.max_idle_sleep_seconds))
                continue
            await self.process(job)
            processed += 1

    async def drain(self) -> int:
        return await self._drain(None)

    async def run_drain(self) -> int:
        lock = SingletonDrainLock(self.lock_path)
        if not lock.acquire():
            return 0
        try:
            return await self._drain(lock.release)
        finally:
            lock.release()


def _default_worker() -> tuple[MemoryWorker, QueueRepository]:
    config = load_config(os.environ)
    repository = QueueRepository(config.queue_path)
    codex = CodexProvider(task_models=config.task_models)
    claude = ClaudeProvider(model=config.claude_model)
    worker = MemoryWorker(
        repository,
        router_factory=lambda callback: ProviderRouter(
            codex, claude, attempt_callback=callback
        ),
        lock_path=config.root_dir / "scripts" / "memory-worker.lock",
        lease_seconds=config.job_timeout_seconds + 120,
        provider_timeout_seconds=config.job_timeout_seconds,
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
