"""Detached drain worker for durable memory jobs."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import logging
import os
from pathlib import Path
import random
import sys
from typing import Awaitable, Callable, Protocol, cast
import uuid


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import load_config
from scripts.providers import (
    AttemptCallback,
    AttemptStartCallback,
    ClaudeProvider,
    CodexProvider,
    ProviderResult,
    ProviderRouter,
    RoutedResult,
    TaskKind,
    TextRequest,
)
from scripts.queue import Job, LeaseOwnershipError, QueueRepository
from scripts.staging import recover_incomplete_apply
from scripts.status_store import EventLevel, ProviderName
from scripts.utils import ExclusiveFileLock, append_daily_entry
from scripts.flush import build_flush_prompt, maybe_trigger_compilation


LOGGER = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def daily_writer_boundary(job: Job, text: str) -> None:
    """Append an extracted capture through the shared serialized writer."""
    config = load_config(os.environ)
    identity_material = "\0".join(
        (
            str(getattr(job, "kind", "capture")),
            str(job.source_agent),
            str(getattr(job, "session_id", job.id)),
            str(getattr(job, "source_hash", "")),
        )
    ).encode()
    append_daily_entry(
        config.root_dir,
        text,
        section="Session",
        project_key=job.project,
        cwd=job.cwd,
        agent=job.source_agent,
        capture_identity=hashlib.sha256(identity_material).hexdigest(),
    )


class LeaseLostError(RuntimeError):
    """Raised when the active worker can no longer prove lease ownership."""


class TextRouter(Protocol):
    """Text routing surface used by the live extraction worker."""

    async def generate_text(self, request: TextRequest) -> RoutedResult: ...


class SingletonDrainLock(ExclusiveFileLock):
    """Cross-platform OS file lock; file contents are diagnostic only."""

    def __init__(self, path: Path | str) -> None:
        super().__init__(path, blocking=False)


class MemoryWorker:
    """Claim and process jobs while preserving retry and attempt history."""

    def __init__(
        self,
        queue: QueueRepository,
        router: object | None = None,
        *,
        router_factory: Callable[..., object] | None = None,
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
        concurrency: int = 1,
        startup_recovery: Callable[[], object] | None = None,
        end_of_day_scheduler: Callable[[], object] | None = None,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if provider_timeout_seconds is not None and provider_timeout_seconds <= 0:
            raise ValueError("provider_timeout_seconds must be positive")
        if max_idle_sleep_seconds <= 0:
            raise ValueError("max_idle_sleep_seconds must be positive")
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
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
        self.concurrency = concurrency
        self.startup_recovery = startup_recovery
        self.end_of_day_scheduler = end_of_day_scheduler
        self._writer_lock = asyncio.Lock()

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
        if inspect.iscoroutinefunction(self.daily_writer):
            outcome = self.daily_writer(job, text)
        else:
            outcome = await asyncio.to_thread(self.daily_writer, job, text)
        if inspect.isawaitable(outcome):
            await outcome

    def _append_status_event(
        self,
        job: Job,
        phase: str,
        *,
        provider: ProviderName | None = None,
        level: EventLevel = "info",
        message: str | None = None,
        details: dict[str, int] | None = None,
    ) -> bool:
        try:
            self.queue.append_job_event(
                job.id,
                self.owner,
                phase,
                expected_attempt_count=job.attempt_count,
                provider=provider,
                level=level,
                message=message,
                details=details,
            )
        except LeaseOwnershipError as exc:
            raise LeaseLostError("lease ownership was lost") from exc
        except Exception:
            # Informative events may be omitted when their isolated transaction
            # fails. Queue complete/retry remains the authoritative terminal
            # transition, so continuing cannot create two terminal outcomes.
            LOGGER.exception("Failed to append %s status event", phase)
            return False
        return True

    async def _provider_started(
        self,
        job: Job,
        provider: ProviderName,
        _model: str,
        _task: TaskKind,
    ) -> None:
        async with self._writer_lock:
            self._append_status_event(
                job,
                f"{provider}_started",
                provider=provider,
            )

    async def _provider_ended(self, job: Job, attempt: ProviderResult) -> None:
        succeeded = attempt.outcome == "success"
        phase = f"{attempt.provider}_{'succeeded' if succeeded else 'failed'}"
        level = "info" if succeeded else (
            "warning" if attempt.provider == "codex" else "error"
        )
        async with self._writer_lock:
            self._append_status_event(
                job,
                phase,
                provider=attempt.provider,
                level=level,
                message=None if succeeded else (attempt.reason or attempt.outcome),
                details={"elapsed_ms": max(0, attempt.elapsed_ms)},
            )
            self.queue.record_attempt(job.id, attempt)

    def _build_router(self, job: Job) -> TextRouter:
        if self.router_factory is None:
            if self.router is None:  # Defensive: constructor validates this.
                raise RuntimeError("worker router is unavailable")
            return cast(TextRouter, self.router)

        attempt_callback: AttemptCallback = lambda attempt: self._provider_ended(
            job, attempt
        )
        start_callback: AttemptStartCallback = (
            lambda provider, model, task: self._provider_started(
                job, provider, model, task
            )
        )
        try:
            signature = inspect.signature(self.router_factory)
        except (TypeError, ValueError):
            return cast(
                TextRouter, self.router_factory(attempt_callback, start_callback)
            )
        try:
            signature.bind(attempt_callback, start_callback)
        except TypeError:
            # Compatibility for existing factories that only accept attempt-end
            # persistence. Live production uses both callbacks.
            signature.bind(attempt_callback)
            return cast(TextRouter, self.router_factory(attempt_callback))
        return cast(
            TextRouter, self.router_factory(attempt_callback, start_callback)
        )

    async def _run_serialized_with_lease(
        self,
        job_id: int,
        operation: Callable[[], Awaitable[object]],
    ) -> object:
        async def serialized() -> object:
            async with self._writer_lock:
                return await operation()

        return await self._run_with_lease(job_id, serialized())

    async def _complete_success(
        self,
        job: Job,
        text: str,
        attempts: tuple[ProviderResult, ...],
        *,
        claude_fallback: bool,
    ) -> None:
        async def write_and_complete() -> None:
            self._require_lease(job.id)
            for attempt in attempts:
                self.queue.record_attempt(job.id, attempt)
            self._append_status_event(
                job,
                "daily_log_write_started",
                details={"chars_saved": len(text)},
            )
            await self._write(job, text)
            self._require_lease(job.id)
            summary = f"Saved {len(text):,} characters"
            if claude_fallback:
                summary += " through Claude fallback"
            self.queue.complete(job.id, self.owner, summary=summary)

        await self._run_serialized_with_lease(job.id, write_and_complete)

    async def _retry_failure(
        self,
        job: Job,
        reason: str,
        attempts: tuple[ProviderResult, ...] = (),
    ) -> None:
        async def retry() -> None:
            self._require_lease(job.id)
            for attempt in attempts:
                self.queue.record_attempt(job.id, attempt)
            self.queue.retry(
                job.id,
                self.owner,
                reason,
                self._retry_at(job),
            )

        await self._run_serialized_with_lease(job.id, retry)

    async def process(self, job: Job) -> bool:
        try:
            request = TextRequest(
                task=TaskKind.EXTRACT,
                prompt=build_flush_prompt(
                    job.payload.get("rendered_context", ""), job.project, job.cwd
                ),
                cwd=Path(job.cwd or ROOT),
                timeout_seconds=self.provider_timeout_seconds,
            )
            eagerly_persisted = self.router_factory is not None
            router = self._build_router(job)
            result = cast(
                RoutedResult,
                await self._run_with_lease(job.id, router.generate_text(request)),
            )
            attempts = () if eagerly_persisted else result.attempts
            if result.outcome != "success":
                await self._retry_failure(
                    job, self._failure_reason(result), attempts
                )
                return False
            await self._complete_success(
                job,
                result.text,
                attempts,
                claude_fallback=(
                    result.provider == "claude" and result.fallback_reason is not None
                ),
            )
            return True
        except LeaseLostError:
            return False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            current = self.queue.get_job(job.id)
            if current.status == "leased" and current.lease_owner == self.owner:
                try:
                    await self._retry_failure(
                        current, str(exc) or type(exc).__name__
                    )
                except LeaseLostError:
                    return False
            return False

    async def _drain(
        self, release_if_idle: Callable[[], None] | None
    ) -> tuple[int, bool]:
        processed = 0
        wrote_daily_entry = False
        active: set[asyncio.Task[bool]] = set()
        try:
            while True:
                self.queue.recover_stale(self._now())
                while len(active) < self.concurrency:
                    job = self.queue.claim_next(
                        self.owner, self._now(), self.lease_seconds
                    )
                    if job is None:
                        break
                    active.add(asyncio.create_task(self.process(job)))
                if active:
                    done, active = await asyncio.wait(
                        active, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        wrote_daily_entry = await task or wrote_daily_entry
                        processed += 1
                    continue

                wake_at = self.queue.next_wake_at()
                if wake_at is None:
                    if release_if_idle is None:
                        return processed, wrote_daily_entry
                    if self.queue.release_worker_lock_if_idle(release_if_idle):
                        return processed, wrote_daily_entry
                    continue
                delay = max(0.0, (wake_at - self._now()).total_seconds())
                await self.sleeper(min(delay, self.max_idle_sleep_seconds))
        finally:
            for task in active:
                task.cancel()
            for task in active:
                with suppress(asyncio.CancelledError, Exception):
                    await task

    async def drain(self) -> int:
        processed, _ = await self._drain(None)
        return processed

    async def run_drain(self) -> int:
        lock = SingletonDrainLock(self.lock_path)
        if not lock.acquire():
            return 0
        try:
            if self.startup_recovery is not None:
                self.startup_recovery()
            self.queue.sync_usage_records()
            processed, wrote_daily_entry = await self._drain(lock.release)
            if wrote_daily_entry and self.end_of_day_scheduler is not None:
                try:
                    outcome = self.end_of_day_scheduler()
                    if inspect.isawaitable(outcome):
                        await outcome
                except Exception:
                    # Extraction is already durable. Scheduling is best effort
                    # and must not turn a successful queue job into a failure.
                    logging.exception("End-of-day compilation scheduling failed")
            return processed
        finally:
            lock.release()


def _default_worker() -> tuple[MemoryWorker, QueueRepository]:
    config = load_config(os.environ)
    repository = QueueRepository(
        config.queue_path,
        memory_home=config.root_dir,
        sync_usage=False,
    )
    codex = CodexProvider(task_models=config.task_models)
    claude = ClaudeProvider(model=config.claude_model)
    worker = MemoryWorker(
        repository,
        router_factory=lambda callback, start_callback: ProviderRouter(
            codex,
            claude,
            attempt_start_callback=start_callback,
            attempt_callback=callback,
        ),
        lock_path=config.root_dir / "scripts" / "memory-worker.lock",
        lease_seconds=config.job_timeout_seconds + 120,
        provider_timeout_seconds=config.job_timeout_seconds,
        concurrency=config.worker_concurrency,
        startup_recovery=lambda: recover_incomplete_apply(config.root_dir),
        end_of_day_scheduler=lambda: maybe_trigger_compilation(
            memory_home=config.root_dir
        ),
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
