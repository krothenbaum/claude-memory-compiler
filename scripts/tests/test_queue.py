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
import scripts.queue as queue_module
import scripts.utils as utils_module
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
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    assert {row[1] for row in connection.execute("PRAGMA table_info(jobs)")} == {
        "id", "kind", "source_agent", "session_id", "project", "cwd",
        "trigger", "source_path", "source_hash", "payload_json", "status",
        "attempt_count", "available_at", "lease_owner", "lease_expires_at",
        "last_error", "created_at", "updated_at", "completed_at",
    }
    assert {row[1] for row in connection.execute("PRAGMA table_info(provider_attempts)")} == {
        "id", "job_id", "provider", "model", "task", "started_at", "ended_at",
        "outcome", "reason", "input_tokens", "output_tokens", "elapsed_ms",
        "legacy_cost_usd",
    }
    assert connection.execute(
        "SELECT length(value) FROM queue_metadata WHERE key = 'queue_id'"
    ).fetchone() == (32,)
    indexes = {row[1] for row in connection.execute("PRAGMA index_list(jobs)")}
    assert {"jobs_status_available_idx", "jobs_lease_expiry_idx"} <= indexes


def test_same_agent_session_and_hash_deduplicates(repository, tmp_path):
    first = repository.enqueue_capture(session(tmp_path))
    second = repository.enqueue_capture(session(tmp_path))

    assert first.created is True
    assert second.created is False
    assert second.job_id == first.job_id
    assert repository.count_jobs() == 1
    run = repository.status_run_for_job(first.job_id)
    assert run.job_id == first.job_id
    assert run.state == "queued"
    assert run.phase == "queued"
    assert run.started_at == NOW
    assert [event.phase for event in repository.status_events(run.id)] == ["queued"]


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
        run = first.status_run_for_job(job_id)
        assert run.state == "running"
        assert run.phase == "worker_claimed"
        assert [event.phase for event in first.status_events(run.id)] == [
            "queued",
            "worker_claimed",
        ]
        assert second.claim_next("worker-b", NOW, 120) is None


def test_claim_rolls_back_when_status_event_insert_fails(repository, tmp_path):
    job_id = repository.enqueue_capture(session(tmp_path)).job_id
    repository._connection.execute(
        """
        CREATE TRIGGER reject_worker_claimed_event
        BEFORE INSERT ON status_events
        WHEN NEW.phase = 'worker_claimed'
        BEGIN
            SELECT RAISE(ABORT, 'controlled claim status failure');
        END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="controlled claim status failure"):
        repository.claim_next("worker", NOW, 30)

    job = repository.get_job(job_id)
    run = repository.status_run_for_job(job_id)
    assert job.status == "pending"
    assert job.attempt_count == 0
    assert job.lease_owner is None
    assert run.state == "queued"
    assert run.phase == "queued"
    assert [event.phase for event in repository.status_events(run.id)] == ["queued"]


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
    run = repository.status_run_for_job(job_id)
    assert run.state == "retrying"
    assert run.phase == "recovery_pending"
    assert run.error == "worker lease expired"
    assert run.completed_at is None
    assert repository.status_events(run.id)[-1].phase == "recovery_pending"
    assert repository.claim_next("new-worker", NOW + timedelta(seconds=31), 30).id == job_id


def test_recover_stale_dead_letters_jobs_at_the_attempt_limit(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW, max_attempts=1) as repository:
        job_id = repository.enqueue_capture(session(tmp_path)).job_id
        repository.claim_next("dead-worker", NOW, 30)

        assert repository.recover_stale(NOW + timedelta(seconds=31)) == 1

        run = repository.status_run_for_job(job_id)
        assert repository.get_job(job_id).status == "dead"
        assert run.state == "dead"
        assert run.phase == "dead"
        assert run.error == "worker lease expired"
        assert run.completed_at == NOW + timedelta(seconds=31)
        assert repository.status_events(run.id)[-1].phase == "dead"


def test_multi_row_stale_recovery_rolls_back_every_job_on_status_failure(
    repository, tmp_path
):
    first_id = repository.enqueue_capture(
        session(tmp_path, session_id="first", source_hash="first-hash")
    ).job_id
    second_id = repository.enqueue_capture(
        session(tmp_path, session_id="second", source_hash="second-hash")
    ).job_id
    repository.claim_next("first-owner", NOW, 30)
    repository.claim_next("second-owner", NOW, 30)
    second_run = repository.status_run_for_job(second_id)
    repository._connection.execute(
        f"""
        CREATE TRIGGER reject_second_recovery_event
        BEFORE INSERT ON status_events
        WHEN NEW.phase = 'recovery_pending' AND NEW.run_id = {second_run.id}
        BEGIN
            SELECT RAISE(ABORT, 'controlled stale recovery status failure');
        END
        """
    )

    with pytest.raises(
        sqlite3.IntegrityError,
        match="controlled stale recovery status failure",
    ):
        repository.recover_stale(NOW + timedelta(seconds=31))

    for job_id, owner in (
        (first_id, "first-owner"),
        (second_id, "second-owner"),
    ):
        job = repository.get_job(job_id)
        run = repository.status_run_for_job(job_id)
        assert job.status == "leased"
        assert job.lease_owner == owner
        assert run.state == "running"
        assert run.phase == "worker_claimed"
        assert [event.phase for event in repository.status_events(run.id)] == [
            "queued",
            "worker_claimed",
        ]


def test_retry_respects_backoff_and_owner(repository, tmp_path):
    job_id = repository.enqueue_capture(session(tmp_path)).job_id
    repository.claim_next("worker", NOW, 30)
    available = NOW + timedelta(minutes=5)

    with pytest.raises(LeaseOwnershipError):
        repository.retry(job_id, "other", "no", available)
    repository.retry(job_id, "worker", "both failed", available)

    run = repository.status_run_for_job(job_id)
    assert run.state == "retrying"
    assert run.phase == "retry_wait"
    assert run.error == "both failed"
    assert run.summary == "both failed"
    retry_event = repository.status_events(run.id)[-1]
    assert retry_event.phase == "retry_wait"
    assert retry_event.level == "warning"
    assert retry_event.details == {"retry_at": available.isoformat(timespec="microseconds")}

    assert repository.claim_next("worker", NOW + timedelta(minutes=4), 30) is None
    assert repository.claim_next("worker", available, 30).id == job_id


def test_retry_rolls_back_when_status_event_insert_fails(repository, tmp_path):
    job_id = repository.enqueue_capture(session(tmp_path)).job_id
    repository.claim_next("worker", NOW, 30)
    repository._connection.execute(
        """
        CREATE TRIGGER reject_retry_event
        BEFORE INSERT ON status_events
        WHEN NEW.phase = 'retry_wait'
        BEGIN
            SELECT RAISE(ABORT, 'controlled retry status failure');
        END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="controlled retry status failure"):
        repository.retry(job_id, "worker", "provider failed", NOW)

    job = repository.get_job(job_id)
    run = repository.status_run_for_job(job_id)
    assert job.status == "leased"
    assert job.lease_owner == "worker"
    assert job.last_error is None
    assert run.state == "running"
    assert run.phase == "worker_claimed"
    assert [event.phase for event in repository.status_events(run.id)] == [
        "queued",
        "worker_claimed",
    ]


def test_job_events_use_authoritative_owner_and_attempt(repository, tmp_path):
    job_id = repository.enqueue_capture(session(tmp_path)).job_id
    repository.claim_next("worker", NOW, 30)

    repository.append_job_event(
        job_id,
        "worker",
        "codex_started",
        expected_attempt_count=1,
        provider="codex",
    )

    run = repository.status_run_for_job(job_id)
    event = repository.status_events(run.id)[-1]
    assert run.state == "running"
    assert run.phase == "codex_started"
    assert event.phase == "codex_started"
    assert event.attempt == 1


def test_stale_attempt_cannot_append_after_recovery_and_reclaim(repository, tmp_path):
    job_id = repository.enqueue_capture(session(tmp_path)).job_id
    repository.claim_next("worker", NOW, 30)
    repository.recover_stale(NOW + timedelta(seconds=31))
    repository.claim_next("worker", NOW + timedelta(seconds=31), 30)

    with pytest.raises(LeaseOwnershipError):
        repository.append_job_event(
            job_id,
            "worker",
            "codex_succeeded",
            expected_attempt_count=1,
            provider="codex",
        )

    run = repository.status_run_for_job(job_id)
    assert run.state == "running"
    assert run.phase == "worker_claimed"
    repository.append_job_event(
        job_id,
        "worker",
        "codex_started",
        expected_attempt_count=2,
        provider="codex",
    )
    assert repository.status_events(run.id)[-1].attempt == 2


def test_terminal_job_rejects_late_provider_success(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW, max_attempts=1) as repository:
        job_id = repository.enqueue_capture(session(tmp_path)).job_id
        repository.claim_next("worker", NOW, 30)
        repository.retry(job_id, "worker", "terminal failure", NOW)

        with pytest.raises(LeaseOwnershipError):
            repository.append_job_event(
                job_id,
                "worker",
                "codex_succeeded",
                expected_attempt_count=1,
                provider="codex",
            )

        run = repository.status_run_for_job(job_id)
        assert run.state == "dead"
        assert run.phase == "dead"
        assert repository.status_events(run.id)[-1].phase == "dead"


def test_job_event_failure_rolls_back_phase_and_event(repository, tmp_path):
    job_id = repository.enqueue_capture(session(tmp_path)).job_id
    repository.claim_next("worker", NOW, 30)
    repository._connection.execute(
        """
        CREATE TRIGGER reject_codex_started_event
        BEFORE INSERT ON status_events
        WHEN NEW.phase = 'codex_started'
        BEGIN
            SELECT RAISE(ABORT, 'controlled job event failure');
        END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="controlled job event failure"):
        repository.append_job_event(
            job_id,
            "worker",
            "codex_started",
            expected_attempt_count=1,
            provider="codex",
        )

    run = repository.status_run_for_job(job_id)
    assert repository.get_job(job_id).status == "leased"
    assert run.state == "running"
    assert run.phase == "worker_claimed"
    assert [event.phase for event in repository.status_events(run.id)] == [
        "queued",
        "worker_claimed",
    ]


def test_complete_and_dead_letter_transitions(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW, max_attempts=2) as repository:
        success_id = repository.enqueue_capture(session(tmp_path, session_id="ok")).job_id
        repository.claim_next("worker", NOW, 30)
        repository.complete(success_id, "worker", summary="Saved 1,842 characters")
        succeeded = repository.get_job(success_id)
        assert succeeded.status == "succeeded"
        assert succeeded.completed_at == NOW
        success_run = repository.status_run_for_job(success_id)
        assert success_run.state == "succeeded"
        assert success_run.phase == "succeeded"
        assert success_run.summary == "Saved 1,842 characters"
        assert success_run.completed_at == NOW

        dead_id = repository.enqueue_capture(session(tmp_path, session_id="dead")).job_id
        repository.claim_next("worker", NOW, 30)
        repository.retry(dead_id, "worker", "first", NOW)
        repository.claim_next("worker", NOW, 30)
        repository.retry(dead_id, "worker", "second", NOW)
        dead = repository.get_job(dead_id)
        assert dead.status == "dead"
        assert dead.completed_at == NOW
        assert dead.last_error == "second"
        dead_run = repository.status_run_for_job(dead_id)
        assert dead_run.state == "dead"
        assert dead_run.phase == "dead"
        assert dead_run.error == "second"
        assert dead_run.summary == "second"
        assert dead_run.completed_at == NOW


def test_status_event_failure_rolls_back_queue_completion(repository, tmp_path):
    job_id = repository.enqueue_capture(session(tmp_path)).job_id
    repository.claim_next("worker", NOW, 30)
    repository._connection.execute(
        """
        CREATE TRIGGER reject_succeeded_status_event
        BEFORE INSERT ON status_events
        WHEN NEW.phase = 'succeeded'
        BEGIN
            SELECT RAISE(ABORT, 'controlled status insert failure');
        END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="controlled status insert failure"):
        repository.complete(job_id, "worker", summary="must roll back")

    job = repository.get_job(job_id)
    run = repository.status_run_for_job(job_id)
    assert job.status == "leased"
    assert job.lease_owner == "worker"
    assert run.state == "running"
    assert run.phase == "worker_claimed"
    assert run.summary is None
    assert [event.phase for event in repository.status_events(run.id)] == [
        "queued",
        "worker_claimed",
    ]


def test_nested_enqueue_status_failure_preserves_the_caller_transaction(
    repository, tmp_path
):
    repository._connection.execute("BEGIN IMMEDIATE")
    repository._connection.execute(
        "INSERT INTO queue_metadata(key, value) VALUES ('caller_sentinel', 'preserved')"
    )
    repository._connection.execute(
        """
        CREATE TEMP TRIGGER reject_nested_queued_event
        BEFORE INSERT ON status_events
        WHEN NEW.phase = 'queued'
        BEGIN
            SELECT RAISE(ABORT, 'controlled nested enqueue status failure');
        END
        """
    )

    with pytest.raises(
        sqlite3.IntegrityError,
        match="controlled nested enqueue status failure",
    ):
        repository.enqueue_capture(session(tmp_path))

    assert repository._connection.in_transaction is True
    assert repository._connection.execute(
        "SELECT value FROM queue_metadata WHERE key = 'caller_sentinel'"
    ).fetchone()[0] == "preserved"
    assert repository.count_jobs() == 0
    assert repository._connection.execute(
        "SELECT count(*) FROM status_runs"
    ).fetchone()[0] == 0
    repository._connection.execute("COMMIT")


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


def test_windows_queue_secures_main_wal_shm_and_dashboard_accepts(
    tmp_path, monkeypatch
):
    path = tmp_path / "jobs.sqlite3"
    secured: list[Path] = []
    monkeypatch.setattr(queue_module, "_windows_acl_required", lambda: True, raising=False)
    monkeypatch.setattr(
        queue_module,
        "_secure_windows_queue_file",
        lambda candidate: secured.append(Path(candidate)),
        raising=False,
    )

    with QueueRepository(path, clock=lambda: NOW, sync_usage=False):
        present = tuple(
            candidate
            for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
            if candidate.exists()
        )
        assert len(present) == 3
        assert set(present) <= set(secured)

        monkeypatch.setattr(utils_module, "_PRIVATE_STATE_DIR_FD_SUPPORTED", False)
        monkeypatch.setattr(utils_module, "_windows_acl_required", lambda: True)
        monkeypatch.setattr(
            utils_module,
            "_validate_windows_inherited_directory",
            lambda _parent: None,
        )

        def validate_secured(_descriptor, candidate):
            assert Path(candidate) in secured

        monkeypatch.setattr(
            utils_module,
            "_validate_windows_owner_only_file_descriptor",
            validate_secured,
        )
        identities = tuple(
            utils_module.inspect_secure_read_file(candidate) for candidate in present
        )

    with QueueRepository(path, clock=lambda: NOW, sync_usage=False):
        reopened = tuple(
            candidate
            for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
            if candidate.exists()
        )
        assert len(reopened) == 3
        assert set(reopened) <= set(secured)

    assert {identity.path for identity in identities} == set(present)
    assert all(secured.count(candidate) >= 2 for candidate in reopened)


def test_windows_queue_acl_failure_rejects_insecure_sidecar(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    monkeypatch.setattr(queue_module, "_windows_acl_required", lambda: True, raising=False)

    def reject_wal(candidate):
        if str(candidate).endswith("-wal"):
            raise PermissionError("could not establish owner-only queue ACL")

    monkeypatch.setattr(
        queue_module,
        "_secure_windows_queue_file",
        reject_wal,
        raising=False,
    )

    with pytest.raises(PermissionError, match="owner-only queue ACL"):
        QueueRepository(path, clock=lambda: NOW, sync_usage=False)


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
