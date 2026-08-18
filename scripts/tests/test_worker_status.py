from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import UTC, datetime, timedelta

from providers import ProviderResult, ProviderRouter, TaskKind
from transcripts import NormalizedSession, Turn
from worker import MemoryWorker

from scripts.queue import Job, QueueRepository


NOW = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)


class DeterministicProvider:
    def __init__(self, provider, result, *, during_call=None):
        self.provider = provider
        self.result = result
        self.during_call = during_call
        self.calls = []
        if provider == "codex":
            self._task_models = {TaskKind.EXTRACT: result.model}
        else:
            self._model = result.model

    async def generate_text(self, request):
        self.calls.append(request)
        if self.during_call is not None:
            self.during_call()
        return self.result

    async def edit_workspace(self, request):  # pragma: no cover - worker uses text
        raise AssertionError("workspace provider must not run")


def _session(tmp_path, *, session_id="session-1"):
    return NormalizedSession(
        agent="claude",
        session_id=session_id,
        project="memory",
        cwd=str(tmp_path),
        timestamp=NOW.isoformat(),
        trigger="session_end",
        turns=(Turn("user", "Remember provider status"),),
        source_path=str(tmp_path / f"{session_id}.jsonl"),
        source_hash=f"hash-{session_id}",
    )


def _router_factory(codex, claude):
    def factory(attempt_callback, attempt_start_callback):
        return ProviderRouter(
            codex,
            claude,
            attempt_start_callback=attempt_start_callback,
            attempt_callback=attempt_callback,
        )

    return factory


def _phases(repository, job_id):
    run = repository.status_run_for_job(job_id)
    return [event.phase for event in repository.status_events(run.id)]


def _events(repository, job_id):
    run = repository.status_run_for_job(job_id)
    return repository.status_events(run.id)


def _claim(repository, owner="worker", now=NOW, lease_seconds=60) -> Job:
    job = repository.claim_next(owner, now, lease_seconds)
    assert job is not None
    return job


def test_direct_codex_success_emits_exact_phases_and_summary(tmp_path):
    text = "x" * 1_234
    codex = DeterministicProvider(
        "codex",
        ProviderResult(
            "codex", "luna", TaskKind.EXTRACT, "success", text=text, elapsed_ms=17
        ),
    )
    claude = DeterministicProvider(
        "claude",
        ProviderResult("claude", "sonnet", TaskKind.EXTRACT, "success", text="unused"),
    )
    writes = []

    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        queued = repository.enqueue_capture(_session(tmp_path))
        job = _claim(repository)
        worker = MemoryWorker(
            repository,
            router_factory=_router_factory(codex, claude),
            daily_writer=lambda job, saved: writes.append((job.id, saved)),
            clock=lambda: NOW,
            owner="worker",
        )

        assert asyncio.run(worker.process(job)) is True

        events = _events(repository, queued.job_id)
        assert [event.phase for event in events] == [
            "queued",
            "worker_claimed",
            "codex_started",
            "codex_succeeded",
            "daily_log_write_started",
            "succeeded",
        ]
        assert [event.phase for event in events].count("worker_claimed") == 1
        assert [event.phase for event in events].count("succeeded") == 1
        assert events[2].provider == "codex"
        assert events[2].attempt == 1
        assert events[3].provider == "codex"
        assert events[3].details == {"elapsed_ms": 17}
        assert events[4].attempt == 1
        assert repository.status_run_for_job(queued.job_id).summary == (
            "Saved 1,234 characters"
        )
        assert writes == [(queued.job_id, text)]
        assert [(item.provider, item.outcome) for item in repository.attempts_for(queued.job_id)] == [
            ("codex", "success")
        ]
        assert claude.calls == []


def test_claude_fallback_emits_warning_then_success_and_fallback_summary(tmp_path):
    codex = DeterministicProvider(
        "codex",
        ProviderResult(
            "codex",
            "luna",
            TaskKind.EXTRACT,
            "capacity",
            reason="usage limit reached",
            elapsed_ms=5,
        ),
    )
    text = "fallback"
    claude = DeterministicProvider(
        "claude",
        ProviderResult(
            "claude", "sonnet", TaskKind.EXTRACT, "success", text=text, elapsed_ms=9
        ),
    )

    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        queued = repository.enqueue_capture(_session(tmp_path))
        job = _claim(repository)
        worker = MemoryWorker(
            repository,
            router_factory=_router_factory(codex, claude),
            daily_writer=lambda *_: None,
            clock=lambda: NOW,
            owner="worker",
        )

        assert asyncio.run(worker.process(job)) is True

        events = _events(repository, queued.job_id)
        assert [event.phase for event in events] == [
            "queued",
            "worker_claimed",
            "codex_started",
            "codex_failed",
            "claude_started",
            "claude_succeeded",
            "daily_log_write_started",
            "succeeded",
        ]
        assert events[3].level == "warning"
        assert events[3].message == "usage limit reached"
        assert events[3].details == {"elapsed_ms": 5}
        assert events[5].level == "info"
        assert events[5].details == {"elapsed_ms": 9}
        assert repository.status_run_for_job(queued.job_id).summary == (
            "Saved 8 characters through Claude fallback"
        )


def test_both_provider_failures_emit_error_then_retry(tmp_path):
    codex = DeterministicProvider(
        "codex",
        ProviderResult(
            "codex", "luna", TaskKind.EXTRACT, "timeout", reason="timed out"
        ),
    )
    claude = DeterministicProvider(
        "claude",
        ProviderResult(
            "claude", "sonnet", TaskKind.EXTRACT, "error", reason="SDK failed"
        ),
    )

    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, max_attempts=2, sync_usage=False
    ) as repository:
        queued = repository.enqueue_capture(_session(tmp_path))
        job = _claim(repository)
        worker = MemoryWorker(
            repository,
            router_factory=_router_factory(codex, claude),
            daily_writer=lambda *_: (_ for _ in ()).throw(
                AssertionError("daily writer must not run")
            ),
            clock=lambda: NOW,
            owner="worker",
        )

        assert asyncio.run(worker.process(job)) is False

        assert _phases(repository, queued.job_id) == [
            "queued",
            "worker_claimed",
            "codex_started",
            "codex_failed",
            "claude_started",
            "claude_failed",
            "retry_wait",
        ]
        events = _events(repository, queued.job_id)
        assert events[3].level == "warning"
        assert events[5].level == "error"
        assert repository.get_job(queued.job_id).status == "failed"


def test_daily_writer_failure_retries_after_write_started_without_success(tmp_path):
    codex = DeterministicProvider(
        "codex",
        ProviderResult(
            "codex", "luna", TaskKind.EXTRACT, "success", text="daily"
        ),
    )
    claude = DeterministicProvider(
        "claude",
        ProviderResult("claude", "sonnet", TaskKind.EXTRACT, "success", text="unused"),
    )

    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, max_attempts=2, sync_usage=False
    ) as repository:
        queued = repository.enqueue_capture(_session(tmp_path))
        job = _claim(repository)
        worker = MemoryWorker(
            repository,
            router_factory=_router_factory(codex, claude),
            daily_writer=lambda *_: (_ for _ in ()).throw(OSError("disk full")),
            clock=lambda: NOW,
            owner="worker",
        )

        assert asyncio.run(worker.process(job)) is False

        assert _phases(repository, queued.job_id) == [
            "queued",
            "worker_claimed",
            "codex_started",
            "codex_succeeded",
            "daily_log_write_started",
            "retry_wait",
        ]
        assert repository.get_job(queued.job_id).status == "failed"
        assert repository.status_run_for_job(queued.job_id).summary == "disk full"


def test_stale_attempt_rejection_stops_later_provider_events(tmp_path):
    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, max_attempts=3, sync_usage=False
    ) as repository:
        queued = repository.enqueue_capture(_session(tmp_path))
        job = _claim(repository, lease_seconds=30)

        def replace_owner():
            repository.recover_stale(NOW + timedelta(seconds=31))
            repository.claim_next("replacement", NOW + timedelta(seconds=31), 30)

        codex = DeterministicProvider(
            "codex",
            ProviderResult(
                "codex", "luna", TaskKind.EXTRACT, "success", text="must not write"
            ),
            during_call=replace_owner,
        )
        claude = DeterministicProvider(
            "claude",
            ProviderResult(
                "claude", "sonnet", TaskKind.EXTRACT, "success", text="unused"
            ),
        )
        writes = []
        worker = MemoryWorker(
            repository,
            router_factory=_router_factory(codex, claude),
            daily_writer=lambda *_: writes.append(True),
            clock=lambda: NOW,
            owner="worker",
        )

        assert asyncio.run(worker.process(job)) is False

        assert _phases(repository, queued.job_id) == [
            "queued",
            "worker_claimed",
            "codex_started",
            "recovery_pending",
            "worker_claimed",
        ]
        assert repository.get_job(queued.job_id).lease_owner == "replacement"
        assert repository.attempts_for(queued.job_id) == []
        assert writes == []


def test_nonownership_status_failure_is_logged_without_conflicting_terminal_state(
    tmp_path, caplog
):
    codex = DeterministicProvider(
        "codex",
        ProviderResult(
            "codex", "luna", TaskKind.EXTRACT, "success", text="daily"
        ),
    )
    claude = DeterministicProvider(
        "claude",
        ProviderResult("claude", "sonnet", TaskKind.EXTRACT, "success", text="unused"),
    )

    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        queued = repository.enqueue_capture(_session(tmp_path))
        job = _claim(repository)
        original = repository.append_job_event

        def fail_start(job_id, owner, phase, **kwargs):
            if phase == "codex_started":
                raise sqlite3.OperationalError("status unavailable")
            return original(job_id, owner, phase, **kwargs)

        repository.append_job_event = fail_start
        worker = MemoryWorker(
            repository,
            router_factory=_router_factory(codex, claude),
            daily_writer=lambda *_: None,
            clock=lambda: NOW,
            owner="worker",
        )

        with caplog.at_level(logging.ERROR):
            assert asyncio.run(worker.process(job)) is True

        assert _phases(repository, queued.job_id) == [
            "queued",
            "worker_claimed",
            "codex_succeeded",
            "daily_log_write_started",
            "succeeded",
        ]
        assert repository.get_job(queued.job_id).status == "succeeded"
        assert "Failed to append codex_started status event" in caplog.text
