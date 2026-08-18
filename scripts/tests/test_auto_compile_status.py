from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import compile as compile_module
import flush as flush_module
import pytest

from scripts.queue import QueueRepository

NOW = datetime(2026, 8, 18, 18, 0, tzinfo=UTC)
OWNER = "1" * 64
WATCHDOG = "2" * 64
SUCCESSOR = "3" * 64
FIRST = "a" * 64
SECOND = "b" * 64
LOG_NAME = "2026-08-18.md"


def schedule(repository, fingerprint=FIRST, *, lease_seconds=30):
    return repository.schedule_auto_compile(
        OWNER,
        WATCHDOG,
        fingerprint,
        log_name=LOG_NAME,
        required_marker_prefix=("prior",),
        now=NOW,
        expires_at=NOW + timedelta(seconds=lease_seconds),
    )


def reservation(repository):
    row = repository._connection.execute(
        "SELECT value FROM queue_metadata WHERE key = 'auto_compile_reservation'"
    ).fetchone()
    return json.loads(row["value"]) if row is not None else None


def test_schedule_creates_stable_compile_run_and_reuses_equivalent_requests(tmp_path):
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        assert schedule(repository) == ("owner", "watchdog")
        first = reservation(repository)
        run = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{FIRST}"
        )
        assert run is not None
        assert first["status_run_id"] == run.id
        assert run.state == "queued"
        assert [event.phase for event in repository.status_events(run.id)] == [
            "reserved"
        ]

        assert schedule(repository) == ()
        equivalent = reservation(repository)
        assert equivalent["status_run_id"] == run.id
        assert equivalent.get("pending_status_run_id", run.id) == run.id
        assert repository._connection.execute(
            "SELECT count(*) FROM status_runs"
        ).fetchone()[0] == 1


def test_new_pending_fingerprint_gets_a_distinct_compile_run(tmp_path):
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        schedule(repository)
        schedule(repository, SECOND)

        current = reservation(repository)
        first = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{FIRST}"
        )
        second = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{SECOND}"
        )
        assert first is not None and second is not None
        assert first.id != second.id
        assert current["status_run_id"] == first.id
        assert current["pending_status_run_id"] == second.id


def test_takeover_reuses_run_and_emits_one_recovery_event(tmp_path):
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        schedule(repository)
        original = reservation(repository)["status_run_id"]

        status, claimed = repository.poll_auto_compile_watcher(
            WATCHDOG,
            SUCCESSOR,
            lambda _reservation: (
                "uncompiled",
                {
                    "fingerprint": FIRST,
                    "log_name": LOG_NAME,
                    "required_marker_prefix": ["prior"],
                },
            ),
            lambda _token: None,
            predecessor_token=None,
            now=NOW + timedelta(seconds=31),
            watcher_expires_at=NOW + timedelta(seconds=60),
            owner_expires_at=NOW + timedelta(seconds=60),
        )

        assert (status, claimed) == ("claimed", FIRST)
        current = reservation(repository)
        assert current["status_run_id"] == original
        run = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{FIRST}"
        )
        assert run is not None
        assert run.state == "running"
        assert [event.phase for event in repository.status_events(run.id)] == [
            "reserved",
            "generation_recovered",
        ]


def test_failure_retry_and_marker_success_are_atomic_with_reservation(tmp_path):
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        schedule(repository)
        run_id = reservation(repository)["status_run_id"]
        repository.transition_operation_run(run_id, "running", "staging_started")

        outcome = repository.record_auto_compile_failure(
            OWNER,
            FIRST,
            "validation_failed",
            lambda: ("uncompiled", FIRST, ("prior",)),
            now=NOW,
            expires_at=NOW + timedelta(seconds=30),
            max_attempts=3,
            retry_base_seconds=5,
        )
        assert outcome == "retry_wait"
        retrying = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{FIRST}"
        )
        assert retrying is not None
        assert retrying.state == "retrying"
        assert retrying.phase == "retry_wait"
        assert reservation(repository)["status"] == "retry_wait"

        repository.transition_operation_run(
            run_id, "running", "generation_recovered"
        )
        next_fingerprint = repository.finish_auto_compile_generation(
            OWNER,
            FIRST,
            lambda: ("covered", None, ("prior", "compiled")),
            now=NOW + timedelta(seconds=10),
            expires_at=NOW + timedelta(seconds=40),
        )
        assert next_fingerprint is None
        completed = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{FIRST}"
        )
        assert completed is not None
        assert completed.state == "succeeded"
        assert completed.phase == "succeeded"
        assert reservation(repository) is None


def test_automatic_status_environment_is_reservation_bound_and_has_no_tty(tmp_path):
    with QueueRepository(tmp_path / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        schedule(repository)
        run_id = reservation(repository)["status_run_id"]
        assert repository.active_auto_compile_status_run(run_id, now=NOW) is not None
        assert repository.active_auto_compile_status_run(run_id + 1, now=NOW) is None

    environment = flush_module._auto_compile_environment(tmp_path, run_id)
    assert environment["AI_MEMORY_STATUS_RUN_ID"] == str(run_id)
    assert "CLAUDE_MEMORY_TTY" not in environment


def test_automatic_compile_records_stage_provider_validation_and_apply_phases(tmp_path):
    root = tmp_path / "memory"
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        schedule(repository, lease_seconds=365 * 24 * 60 * 60)
        run_id = reservation(repository)["status_run_id"]

    for phase, details in (
        ("staging_started", None),
        ("provider_started", None),
        ("validation_started", None),
        ("validation_started", None),
        ("apply_started", {"changed_files": 4}),
    ):
        assert compile_module._record_automatic_phase(
            root, run_id, phase, details=details
        )

    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        run = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{FIRST}"
        )
        assert run is not None
        assert run.state == "running"
        assert [event.phase for event in repository.status_events(run.id)] == [
            "reserved",
            "staging_started",
            "provider_started",
            "validation_started",
            "validation_started",
            "apply_started",
        ]
        assert repository.status_events(run.id)[-1].details == {"changed_files": 4}


def test_exhausted_compile_failure_marks_run_dead_atomically(tmp_path):
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        schedule(repository)
        run_id = reservation(repository)["status_run_id"]
        repository.transition_operation_run(run_id, "running", "staging_started")

        outcome = repository.record_auto_compile_failure(
            OWNER,
            FIRST,
            "provider_failed",
            lambda: ("uncompiled", FIRST, ("prior",)),
            now=NOW,
            expires_at=NOW + timedelta(seconds=30),
            max_attempts=1,
            retry_base_seconds=5,
        )

        assert outcome == "failed"
        run = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{FIRST}"
        )
        assert run is not None
        assert run.state == "dead"
        assert run.phase == "dead"
        assert reservation(repository)["status"] == "failed"


def test_exit_75_marker_completion_terminalizes_run_before_deleting_reservation(
    tmp_path,
):
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        schedule(repository)
        run_id = reservation(repository)["status_run_id"]
        repository.transition_operation_run(run_id, "running", "staging_started")

        assert repository.defer_auto_compile_generation(
            OWNER,
            FIRST,
            lambda: ("covered", None, ("prior", "compiled")),
            compiler_lock_held=False,
            now=NOW + timedelta(seconds=5),
            expires_at=NOW + timedelta(seconds=35),
        ) is None

        run = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{FIRST}"
        )
        assert run is not None
        assert run.state == "succeeded"
        assert run.phase == "succeeded"
        assert run.completed_at == NOW + timedelta(seconds=5)
        assert reservation(repository) is None


def test_watchdog_marker_completion_terminalizes_run_before_deleting_reservation(
    tmp_path,
):
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        schedule(repository)
        run_id = reservation(repository)["status_run_id"]
        repository.transition_operation_run(run_id, "running", "staging_started")

        status, fingerprint = repository.poll_auto_compile_watcher(
            WATCHDOG,
            SUCCESSOR,
            lambda _reservation: ("covered", None),
            lambda _token: None,
            predecessor_token=None,
            now=NOW + timedelta(seconds=31),
            watcher_expires_at=NOW + timedelta(seconds=60),
            owner_expires_at=NOW + timedelta(seconds=60),
        )

        assert (status, fingerprint) == ("done", None)
        run = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{FIRST}"
        )
        assert run is not None
        assert run.state == "succeeded"
        assert run.phase == "succeeded"
        assert run.completed_at == NOW + timedelta(seconds=31)
        assert reservation(repository) is None


def test_changed_content_failure_creates_fresh_reserved_generation(tmp_path):
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        schedule(repository)
        first_run_id = reservation(repository)["status_run_id"]
        repository.transition_operation_run(
            first_run_id, "running", "staging_started"
        )

        outcome = repository.record_auto_compile_failure(
            OWNER,
            FIRST,
            "provider_failed",
            lambda: ("uncompiled", SECOND, ("prior",)),
            now=NOW + timedelta(seconds=5),
            expires_at=NOW + timedelta(seconds=35),
            max_attempts=3,
            retry_base_seconds=5,
        )

        assert outcome == "retry_wait"
        current = reservation(repository)
        assert current["fingerprint"] == SECOND
        assert "attempt_count" not in current
        first = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{FIRST}"
        )
        second = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{SECOND}"
        )
        assert first is not None and second is not None
        assert first.state == "failed"
        assert first.phase == "failed"
        assert second.state == "queued"
        assert second.phase == "reserved"
        assert second.summary is None
        assert second.error is None
        assert [event.phase for event in repository.status_events(second.id)] == [
            "reserved"
        ]


def test_takeover_replaces_active_and_pending_with_one_recovered_generation(tmp_path):
    third = "c" * 64
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        schedule(repository)
        schedule(repository, SECOND)
        before = reservation(repository)
        active_id = before["status_run_id"]
        pending_id = before["pending_status_run_id"]

        status, claimed = repository.poll_auto_compile_watcher(
            WATCHDOG,
            SUCCESSOR,
            lambda _reservation: (
                "uncompiled",
                {
                    "fingerprint": third,
                    "log_name": LOG_NAME,
                    "required_marker_prefix": ["prior"],
                },
            ),
            lambda _token: None,
            predecessor_token=None,
            now=NOW + timedelta(seconds=31),
            watcher_expires_at=NOW + timedelta(seconds=60),
            owner_expires_at=NOW + timedelta(seconds=60),
        )

        assert (status, claimed) == ("claimed", third)
        current = reservation(repository)
        replacement = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{third}"
        )
        assert replacement is not None
        assert current["status_run_id"] == replacement.id
        assert replacement.state == "running"
        assert replacement.phase == "generation_recovered"
        assert repository._connection.execute(
            "SELECT state FROM status_runs WHERE id = ?", (active_id,)
        ).fetchone()[0] == "failed"
        assert repository._connection.execute(
            "SELECT state FROM status_runs WHERE id = ?", (pending_id,)
        ).fetchone()[0] == "failed"


@pytest.mark.parametrize("terminal_state", ["failed", "dead"])
def test_reschedule_same_fingerprint_never_relinks_terminal_run(
    tmp_path, terminal_state
):
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        schedule(repository)
        first = reservation(repository)
        first_id = first["status_run_id"]
        repository.transition_operation_run(first_id, terminal_state, terminal_state)
        repository.release_auto_compile(OWNER, FIRST)

        assert repository.schedule_auto_compile(
            "4" * 64,
            "5" * 64,
            FIRST,
            log_name=LOG_NAME,
            required_marker_prefix=("prior",),
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(seconds=31),
        ) == ("owner", "watchdog")

        current = reservation(repository)
        assert current["status_run_id"] != first_id
        rerun = repository._connection.execute(
            "SELECT operation_key, state, phase FROM status_runs WHERE id = ?",
            (current["status_run_id"],),
        ).fetchone()
        assert rerun[0].startswith(f"auto-compile:{LOG_NAME}:{FIRST}:")
        assert tuple(rerun[1:]) == ("queued", "reserved")


def test_legacy_reservation_is_backfilled_with_matching_nonterminal_run(tmp_path):
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        assert repository.reserve_auto_compile(
            OWNER,
            FIRST,
            log_name=LOG_NAME,
            now=NOW,
            expires_at=NOW + timedelta(seconds=30),
        )

        owned = repository.auto_compile_reservation(OWNER, now=NOW)
        assert owned is not None
        run = repository.status_run_for_operation(
            f"auto-compile:{LOG_NAME}:{FIRST}"
        )
        assert run is not None
        assert owned[5] == run.id
        assert run.state == "queued"


@pytest.mark.parametrize("removal", ["rollback", "release"])
def test_reservation_removal_terminalizes_all_linked_runs(tmp_path, removal):
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        schedule(repository)
        if removal == "release":
            schedule(repository, SECOND)
        linked = reservation(repository)
        if removal == "rollback":
            assert repository.rollback_auto_compile_schedule(
                OWNER, WATCHDOG, now=NOW + timedelta(seconds=1)
            )
        else:
            assert repository.release_auto_compile(OWNER, FIRST)

        run_ids = {linked["status_run_id"]}
        if "pending_status_run_id" in linked:
            run_ids.add(linked["pending_status_run_id"])
        for run_id in run_ids:
            assert repository._connection.execute(
                "SELECT state FROM status_runs WHERE id = ?", (run_id,)
            ).fetchone()[0] == "failed"


def test_manual_compile_has_no_automatic_status_run(monkeypatch):
    monkeypatch.delenv("AI_MEMORY_AUTO_COMPILE", raising=False)
    monkeypatch.delenv("AI_MEMORY_STATUS_RUN_ID", raising=False)

    assert compile_module._automatic_status_run_id() is None


def test_untrusted_automatic_status_run_id_is_ignored(monkeypatch):
    monkeypatch.setenv("AI_MEMORY_AUTO_COMPILE", "1")
    monkeypatch.setenv("AI_MEMORY_STATUS_RUN_ID", "not-an-id")

    assert compile_module._automatic_status_run_id() is None


def test_trusted_automatic_status_run_id_is_parsed():
    assert compile_module._automatic_status_run_id(
        {
            "AI_MEMORY_AUTO_COMPILE": "1",
            "AI_MEMORY_STATUS_RUN_ID": "42",
        }
    ) == 42
