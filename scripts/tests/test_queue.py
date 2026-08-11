from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import stat
import threading

import pytest

from providers import ProviderResult, TaskKind
from scripts.queue import LeaseOwnershipError, QueueRepository
from transcripts import NormalizedSession, Turn


NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def session(tmp_path, *, agent="claude", session_id="session-1", source_hash="hash-1"):
    return NormalizedSession(
        agent=agent,
        session_id=session_id,
        project="memory",
        cwd=str(tmp_path),
        timestamp=NOW.isoformat(),
        trigger="session_end",
        turns=(Turn("user", "Keep this durable"), Turn("assistant", "Done")),
        source_path=str(tmp_path / "source.jsonl"),
        source_hash=source_hash,
    )


@pytest.fixture
def repository(tmp_path):
    with QueueRepository(tmp_path / "jobs.sqlite3", clock=lambda: NOW) as repository:
        yield repository


def test_migration_creates_exact_queue_contract_in_wal_mode(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW):
        pass

    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    assert {row[1] for row in connection.execute("PRAGMA table_info(jobs)")} == {
        "id", "kind", "source_agent", "session_id", "project", "cwd",
        "trigger", "source_path", "source_hash", "payload_json", "status",
        "attempt_count", "available_at", "lease_owner", "lease_expires_at",
        "last_error", "created_at", "updated_at", "completed_at",
    }
    assert {row[1] for row in connection.execute("PRAGMA table_info(provider_attempts)")} == {
        "id", "job_id", "provider", "model", "task", "started_at", "ended_at",
        "outcome", "reason", "input_tokens", "output_tokens", "elapsed_ms",
    }
    indexes = {row[1] for row in connection.execute("PRAGMA index_list(jobs)")}
    assert {"jobs_status_available_idx", "jobs_lease_expiry_idx"} <= indexes


def test_same_agent_session_and_hash_deduplicates(repository, tmp_path):
    first = repository.enqueue_capture(session(tmp_path))
    second = repository.enqueue_capture(session(tmp_path))

    assert first.created is True
    assert second.created is False
    assert second.job_id == first.job_id
    assert repository.count_jobs() == 1


def test_same_session_id_from_different_agents_is_distinct(repository, tmp_path):
    claude = repository.enqueue_capture(session(tmp_path, agent="claude"))
    codex = repository.enqueue_capture(session(tmp_path, agent="codex"))

    assert claude.job_id != codex.job_id
    assert repository.count_jobs() == 2


def test_fallback_provider_is_an_attempt_not_a_new_job(repository, tmp_path):
    queued = repository.enqueue_capture(session(tmp_path))
    for provider, outcome in (("codex", "capacity"), ("claude", "success")):
        repository.record_attempt(
            queued.job_id,
            ProviderResult(
                provider=provider,
                model="test-model",
                task=TaskKind.EXTRACT,
                outcome=outcome,
                elapsed_ms=25,
                reason="capacity" if outcome == "capacity" else None,
            ),
        )

    assert repository.count_jobs() == 1
    assert [attempt.provider for attempt in repository.attempts_for(queued.job_id)] == [
        "codex", "claude"
    ]


def test_claim_is_atomic_and_cannot_be_claimed_twice(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW) as first, QueueRepository(
        path, clock=lambda: NOW
    ) as second:
        job_id = first.enqueue_capture(session(tmp_path)).job_id
        claimed = first.claim_next("worker-a", NOW, 120)

        assert claimed is not None and claimed.id == job_id
        assert claimed.status == "leased"
        assert claimed.attempt_count == 1
        assert second.claim_next("worker-b", NOW, 120) is None


def test_concurrent_claimers_observe_exactly_one_lease(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW) as repository:
        job_id = repository.enqueue_capture(session(tmp_path)).job_id
    ready = threading.Barrier(3)

    def claim(owner):
        with QueueRepository(path, clock=lambda: NOW) as contender:
            ready.wait()
            job = contender.claim_next(owner, NOW, 120)
            return job.id if job is not None else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(claim, owner) for owner in ("worker-a", "worker-b")]
        ready.wait()
        claimed = [future.result(timeout=2) for future in futures]

    assert sorted(result for result in claimed if result is not None) == [job_id]
    assert claimed.count(None) == 1


def test_renew_requires_the_current_owner(repository, tmp_path):
    job = repository.enqueue_capture(session(tmp_path))
    claimed = repository.claim_next("worker-a", NOW, 120)
    later = NOW + timedelta(minutes=10)

    assert claimed is not None
    assert repository.renew(job.job_id, "worker-b", later) is False
    assert repository.renew(job.job_id, "worker-a", later) is True
    assert repository.get_job(job.job_id).lease_expires_at == later


def test_recover_stale_makes_expired_lease_retryable(repository, tmp_path):
    job_id = repository.enqueue_capture(session(tmp_path)).job_id
    repository.claim_next("dead-worker", NOW, 30)

    assert repository.recover_stale(NOW + timedelta(seconds=31)) == 1
    recovered = repository.get_job(job_id)
    assert recovered.status == "failed"
    assert recovered.lease_owner is None
    assert recovered.lease_expires_at is None
    assert repository.claim_next("new-worker", NOW + timedelta(seconds=31), 30).id == job_id


def test_retry_respects_backoff_and_owner(repository, tmp_path):
    job_id = repository.enqueue_capture(session(tmp_path)).job_id
    repository.claim_next("worker", NOW, 30)
    available = NOW + timedelta(minutes=5)

    with pytest.raises(LeaseOwnershipError):
        repository.retry(job_id, "other", "no", available)
    repository.retry(job_id, "worker", "both failed", available)

    assert repository.claim_next("worker", NOW + timedelta(minutes=4), 30) is None
    assert repository.claim_next("worker", available, 30).id == job_id


def test_complete_and_dead_letter_transitions(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW, max_attempts=2) as repository:
        success_id = repository.enqueue_capture(session(tmp_path, session_id="ok")).job_id
        repository.claim_next("worker", NOW, 30)
        repository.complete(success_id, "worker")
        succeeded = repository.get_job(success_id)
        assert succeeded.status == "succeeded"
        assert succeeded.completed_at == NOW

        dead_id = repository.enqueue_capture(session(tmp_path, session_id="dead")).job_id
        repository.claim_next("worker", NOW, 30)
        repository.retry(dead_id, "worker", "first", NOW)
        repository.claim_next("worker", NOW, 30)
        repository.retry(dead_id, "worker", "second", NOW)
        dead = repository.get_job(dead_id)
        assert dead.status == "dead"
        assert dead.completed_at == NOW
        assert dead.last_error == "second"


def test_queue_database_and_sidecars_are_owner_only_with_typical_umask(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    previous = os.umask(0o022)
    try:
        with QueueRepository(path, clock=lambda: NOW) as repository:
            repository.enqueue_capture(session(tmp_path))
            present = [candidate for candidate in (
                path,
                Path(f"{path}-wal"),
                Path(f"{path}-shm"),
            ) if candidate.exists()]
            assert len(present) == 3
            assert all(stat.S_IMODE(candidate.stat().st_mode) == 0o600 for candidate in present)
    finally:
        os.umask(previous)


def test_queue_rejects_symlink_database_target(tmp_path):
    real = tmp_path / "attacker.sqlite3"
    real.write_bytes(b"attacker bytes")
    target = tmp_path / "jobs.sqlite3"
    target.symlink_to(real)

    with pytest.raises(ValueError, match="symlink"):
        QueueRepository(target)

    assert real.read_bytes() == b"attacker bytes"


def test_persistence_errors_and_attempt_reasons_redact_credentials(tmp_path):
    secret = "credential-value-never-persist"
    with QueueRepository(
        tmp_path / "jobs.sqlite3",
        clock=lambda: NOW,
        redaction_env={"OPENAI_API_KEY": secret},
    ) as repository:
        job_id = repository.enqueue_capture(session(tmp_path)).job_id
        repository.claim_next("worker", NOW, 30)
        repository.record_attempt(
            job_id,
            ProviderResult(
                "codex",
                "luna",
                TaskKind.EXTRACT,
                "error",
                reason=f"provider exposed {secret}\nacross lines",
            ),
        )
        repository.retry(job_id, "worker", f"retry exposed {secret}\nacross lines", NOW)

        assert secret not in repository.get_job(job_id).last_error
        assert "[REDACTED]" in repository.get_job(job_id).last_error
        assert secret not in repository.attempts_for(job_id)[0].reason
        assert "[REDACTED]" in repository.attempts_for(job_id)[0].reason


def test_persistence_redacts_generic_credential_suffixes_only(tmp_path):
    credentials = {
        "GITHUB_TOKEN": "github-secret-value",
        "SERVICE_API_KEY": "service-secret-value",
        "DB_PASSWORD": "database-secret-value",
        "OTHER_SECRET": "other-secret-value",
        "UNRELATED_SETTING": "public-setting-value",
    }
    with QueueRepository(
        tmp_path / "jobs.sqlite3",
        clock=lambda: NOW,
        redaction_env=credentials,
    ) as repository:
        job_id = repository.enqueue_capture(session(tmp_path)).job_id
        repository.claim_next("worker", NOW, 30)
        reason = " | ".join(credentials.values())
        repository.retry(job_id, "worker", reason, NOW)

        persisted = repository.get_job(job_id).last_error
        assert persisted.count("[REDACTED]") == 4
        assert all(value not in persisted for value in list(credentials.values())[:4])
        assert credentials["UNRELATED_SETTING"] in persisted
