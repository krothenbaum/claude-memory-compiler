from __future__ import annotations

import json
import os
import sqlite3
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest
from transcripts import NormalizedSession, Turn

import scripts.queue as queue_module
import scripts.status_store as status_store_module
import scripts.utils as utils_module
from scripts.queue import QueueRepository
from scripts.status_store import (
    ALLOWED_PHASES,
    EventLevel,
    ProviderName,
    RunState,
    StatusEvent,
    StatusRun,
    normalize_details,
    normalize_status_reason,
    normalize_summary,
)

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)


def _version_2_session() -> NormalizedSession:
    return NormalizedSession(
        agent="claude",
        session_id="session-41",
        project="memory",
        cwd="/memory",
        timestamp=NOW.isoformat(),
        trigger="session_end",
        turns=(Turn("user", "legacy capture"),),
        source_path="/memory/source.jsonl",
        source_hash="hash-41",
    )


def _create_version_2_queue(path, *, incompatible_status_view: bool = False) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            source_agent TEXT NOT NULL,
            session_id TEXT NOT NULL,
            project TEXT NOT NULL,
            cwd TEXT NOT NULL,
            trigger TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL,
            available_at TEXT NOT NULL,
            lease_owner TEXT,
            lease_expires_at TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            UNIQUE (kind, source_agent, session_id, source_hash)
        );
        CREATE TABLE provider_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            task TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL,
            outcome TEXT NOT NULL,
            reason TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            elapsed_ms INTEGER NOT NULL,
            legacy_cost_usd REAL
        );
        CREATE TABLE queue_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO queue_metadata(key, value) VALUES ('queue_id', 'version-two-queue');
        INSERT INTO jobs (
            id, kind, source_agent, session_id, project, cwd, trigger, source_path,
            source_hash, payload_json, status, attempt_count, available_at,
            created_at, updated_at
        ) VALUES (
            41, 'capture', 'claude', 'session-41', 'memory', '/memory',
            'session_end', '/memory/source.jsonl', 'hash-41', '{}', 'succeeded', 1,
            '2026-08-18T12:00:00+00:00', '2026-08-18T12:00:00+00:00',
            '2026-08-18T12:01:00+00:00'
        );
        INSERT INTO provider_attempts (
            id, job_id, provider, model, task, started_at, ended_at, outcome,
            reason, input_tokens, output_tokens, elapsed_ms, legacy_cost_usd
        ) VALUES (
            73, 41, 'codex', 'gpt-5.6-luna', 'extract',
            '2026-08-18T12:00:00+00:00', '2026-08-18T12:00:01+00:00',
            'success', NULL, 10, 20, 1000, NULL
        );
        PRAGMA user_version = 2;
        """
    )
    if incompatible_status_view:
        connection.execute("CREATE VIEW status_events AS SELECT 1 AS id")
    connection.commit()
    connection.close()
    path.chmod(0o600)


def _create_version_1_queue(path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            source_agent TEXT NOT NULL,
            session_id TEXT NOT NULL,
            project TEXT NOT NULL,
            cwd TEXT NOT NULL,
            trigger TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL,
            available_at TEXT NOT NULL,
            lease_owner TEXT,
            lease_expires_at TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            UNIQUE (kind, source_agent, session_id, source_hash)
        );
        CREATE TABLE provider_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            task TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL,
            outcome TEXT NOT NULL,
            reason TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            elapsed_ms INTEGER NOT NULL
        );
        PRAGMA user_version = 1;
        """
    )
    connection.commit()
    connection.close()


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _status_schema_signature(connection: sqlite3.Connection) -> dict[str, object]:
    signature: dict[str, object] = {}
    for table in ("status_runs", "status_events"):
        signature[f"{table}:table_info"] = tuple(
            tuple(row) for row in connection.execute(f"PRAGMA table_info({table})")
        )
        signature[f"{table}:foreign_keys"] = tuple(
            tuple(row) for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        )
        signature[f"{table}:indexes"] = tuple(
            tuple(row) for row in connection.execute(f"PRAGMA index_list({table})")
        )
        signature[f"{table}:index_sql"] = tuple(
            tuple(row)
            for row in connection.execute(
                """
                SELECT name, sql FROM sqlite_master
                WHERE type = 'index' AND tbl_name = ?
                ORDER BY name
                """,
                (table,),
            )
        )
    return signature


def test_version_2_migration_preserves_jobs_and_attempts(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    _create_version_2_queue(path)

    with QueueRepository(path, sync_usage=False) as repository:
        assert repository._connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert [
            tuple(row)
            for row in repository._connection.execute(
                "SELECT id, session_id, status FROM jobs"
            )
        ] == [(41, "session-41", "succeeded")]
        assert [
            tuple(row)
            for row in repository._connection.execute(
                "SELECT id, job_id, provider, outcome FROM provider_attempts"
            )
        ] == [(73, 41, "codex", "success")]
        assert {"status_runs", "status_events"} <= _table_names(repository._connection)


def test_claiming_a_migrated_active_job_creates_its_status_timeline(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    _create_version_2_queue(path)
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE jobs SET status = 'pending', attempt_count = 0, completed_at = NULL"
    )
    connection.commit()
    connection.close()

    with QueueRepository(path, clock=lambda: NOW, sync_usage=False) as repository:
        claimed = repository.claim_next("worker", NOW, 30)

        assert claimed is not None
        run = repository.status_run_for_job(claimed.id)
        assert run.state == "running"
        assert run.phase == "worker_claimed"
        assert [event.phase for event in repository.status_events(run.id)] == [
            "queued",
            "worker_claimed",
        ]


def test_deduplicated_enqueue_lazily_adds_status_to_a_migrated_active_job(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    _create_version_2_queue(path)
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE jobs SET status = 'pending', attempt_count = 0, completed_at = NULL"
    )
    connection.commit()
    connection.close()

    with QueueRepository(path, clock=lambda: NOW, sync_usage=False) as repository:
        result = repository.enqueue_capture(_version_2_session())

        assert result.created is False
        assert result.job_id == 41
        assert repository.get_job(41).payload_json == "{}"
        run = repository.status_run_for_job(41)
        assert run.state == "queued"
        assert [event.phase for event in repository.status_events(run.id)] == [
            "queued"
        ]
        assert repository._connection.execute(
            "SELECT count(*) FROM status_runs WHERE job_id = 41"
        ).fetchone()[0] == 1

        repository.claim_next("worker", NOW, 30)
        repository.append_job_event(
            41,
            "worker",
            "codex_started",
            expected_attempt_count=1,
            provider="codex",
        )
        assert repository.status_run_for_job(41).phase == "codex_started"
        assert [event.phase for event in repository.status_events(run.id)] == [
            "queued",
            "worker_claimed",
            "codex_started",
        ]


@pytest.mark.parametrize(
    (
        "job_status",
        "expected_state",
        "expected_phase",
        "last_error",
        "completed_at",
    ),
    [
        ("pending", "queued", "queued", None, None),
        ("leased", "running", "worker_claimed", "prior failure", None),
        ("failed", "retrying", "retry_wait", "retry failure", None),
        ("succeeded", "succeeded", "succeeded", None, NOW),
        ("dead", "dead", "dead", "terminal failure", NOW),
    ],
)
def test_deduplicated_migrated_jobs_synthesize_their_authoritative_status(
    tmp_path,
    job_status,
    expected_state,
    expected_phase,
    last_error,
    completed_at,
):
    path = tmp_path / "jobs.sqlite3"
    _create_version_2_queue(path)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        UPDATE jobs
        SET status = ?, attempt_count = 1, available_at = ?, lease_owner = ?,
            lease_expires_at = ?, last_error = ?, completed_at = ?
        """,
        (
            job_status,
            NOW.isoformat(),
            "worker" if job_status == "leased" else None,
            (
                (NOW.replace(minute=1)).isoformat()
                if job_status == "leased"
                else None
            ),
            last_error,
            completed_at.isoformat() if completed_at is not None else None,
        ),
    )
    connection.commit()
    connection.close()

    with QueueRepository(path, clock=lambda: NOW, sync_usage=False) as repository:
        result = repository.enqueue_capture(_version_2_session())

        assert result.created is False
        run = repository.status_run_for_job(result.job_id)
        assert run.state == expected_state
        assert run.phase == expected_phase
        assert run.error == (
            last_error if job_status in {"failed", "dead"} else None
        )
        assert run.summary == (
            last_error if job_status in {"failed", "dead"} else None
        )
        assert run.completed_at == completed_at
        events = repository.status_events(run.id)
        assert [event.phase for event in events] == [expected_phase]
        assert events[0].message == (
            last_error if job_status in {"failed", "dead"} else None
        )


def test_concurrent_version_2_openers_apply_status_migration_once(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    _create_version_2_queue(path)
    both_observed_v2 = threading.Barrier(2)

    def synchronize_version_read(self, version):
        if version == 2:
            both_observed_v2.wait(timeout=5)

    monkeypatch.setattr(
        queue_module.QueueRepository,
        "_migration_version_observed",
        synchronize_version_read,
    )

    def open_repository(_):
        with QueueRepository(path, sync_usage=False) as repository:
            return repository._connection.execute("PRAGMA user_version").fetchone()[0]

    with ThreadPoolExecutor(max_workers=2) as executor:
        versions = list(executor.map(open_repository, range(2)))

    assert versions == [3, 3]
    with QueueRepository(path, sync_usage=False) as repository:
        assert {"status_runs", "status_events"} <= _table_names(repository._connection)


def test_concurrent_version_1_openers_apply_each_migration_step_once(
    tmp_path, monkeypatch
):
    path = tmp_path / "jobs.sqlite3"
    _create_version_1_queue(path)
    both_observed_v1 = threading.Barrier(2)

    def synchronize_version_read(self, version):
        if version == 1:
            both_observed_v1.wait(timeout=5)

    monkeypatch.setattr(
        queue_module.QueueRepository,
        "_migration_version_observed",
        synchronize_version_read,
    )

    def open_repository(_):
        with QueueRepository(path, sync_usage=False) as repository:
            columns = {
                row[1]
                for row in repository._connection.execute(
                    "PRAGMA table_info(provider_attempts)"
                )
            }
            return (
                repository._connection.execute("PRAGMA user_version").fetchone()[0],
                "legacy_cost_usd" in columns,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(open_repository, range(2)))

    assert results == [(3, True), (3, True)]


def test_fresh_queue_contains_version_3_status_schema(tmp_path):
    path = tmp_path / "jobs.sqlite3"

    with QueueRepository(path, sync_usage=False) as repository:
        assert repository._connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert {"status_runs", "status_events"} <= _table_names(repository._connection)
        indexes = {
            row[0]
            for row in repository._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert {
            "status_runs_state_updated_idx",
            "status_events_run_id_id_idx",
        } <= indexes


def test_fresh_and_version_2_migrated_status_schemas_are_identical(tmp_path):
    fresh_path = tmp_path / "fresh.sqlite3"
    migrated_path = tmp_path / "migrated.sqlite3"
    _create_version_2_queue(migrated_path)

    with QueueRepository(fresh_path, sync_usage=False) as fresh, QueueRepository(
        migrated_path, sync_usage=False
    ) as migrated:
        assert _status_schema_signature(fresh._connection) == _status_schema_signature(
            migrated._connection
        )


@pytest.mark.parametrize("queue_origin", ["fresh", "version_2"])
def test_fresh_and_migrated_status_schemas_enforce_the_same_constraints(
    tmp_path, queue_origin
):
    path = tmp_path / f"{queue_origin}.sqlite3"
    if queue_origin == "version_2":
        _create_version_2_queue(path)

    with QueueRepository(path, sync_usage=False) as repository:
        connection = repository._connection
        job_id = 41 if queue_origin == "version_2" else None
        operation_key = None if job_id is not None else "compile:invalid-state"
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
            connection.execute(
                """
                INSERT INTO status_runs (
                    job_id, operation_key, kind, source_agent, session_id, project,
                    state, phase, started_at, updated_at
                ) VALUES (?, ?, 'compile', 'system', 'session', 'memory',
                    'invalid', 'queued', '2026-08-18T12:00:00+00:00',
                    '2026-08-18T12:00:00+00:00')
                """,
                (job_id, operation_key),
            )

        run_id = connection.execute(
            """
            INSERT INTO status_runs (
                operation_key, kind, source_agent, session_id, project,
                state, phase, started_at, updated_at
            ) VALUES (?, 'compile', 'system', 'session', 'memory',
                'queued', 'queued', '2026-08-18T12:00:00+00:00',
                '2026-08-18T12:00:00+00:00')
            """,
            (f"compile:{queue_origin}",),
        ).lastrowid
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
            connection.execute(
                """
                INSERT INTO status_events (run_id, phase, level, created_at)
                VALUES (?, 'queued', 'debug', '2026-08-18T12:00:00+00:00')
                """,
                (run_id,),
            )


def test_version_2_migration_rolls_back_all_status_changes_on_failure(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    _create_version_2_queue(path, incompatible_status_view=True)

    with pytest.raises(sqlite3.OperationalError, match="status_events"):
        QueueRepository(path, sync_usage=False)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert "status_runs" not in _table_names(connection)
        assert connection.execute("SELECT id FROM jobs").fetchall() == [(41,)]
        assert connection.execute("SELECT id FROM provider_attempts").fetchall() == [(73,)]
    finally:
        connection.close()


def test_status_schema_enforces_foreign_keys_and_cascades(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    _create_version_2_queue(path)

    with QueueRepository(path, sync_usage=False) as repository:
        connection = repository._connection
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                """
                INSERT INTO status_runs (
                    job_id, kind, source_agent, session_id, project, state, phase,
                    started_at, updated_at
                ) VALUES (999, 'capture', 'claude', 'missing', 'memory',
                    'queued', 'queued', '2026-08-18T12:00:00+00:00',
                    '2026-08-18T12:00:00+00:00')
                """
            )
        run_id = connection.execute(
            """
            INSERT INTO status_runs (
                job_id, kind, source_agent, session_id, project, state, phase,
                started_at, updated_at
            ) VALUES (41, 'capture', 'claude', 'session-41', 'memory', 'queued',
                'queued', '2026-08-18T12:00:00+00:00',
                '2026-08-18T12:00:00+00:00')
            """
        ).lastrowid
        connection.execute(
            """
            INSERT INTO status_events (run_id, phase, level, created_at)
            VALUES (?, 'queued', 'info', '2026-08-18T12:00:00+00:00')
            """,
            (run_id,),
        )

        connection.execute("DELETE FROM jobs WHERE id = 41")

        assert connection.execute("SELECT count(*) FROM status_runs").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM status_events").fetchone()[0] == 0


def test_status_runs_allow_only_one_run_per_queue_job(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    _create_version_2_queue(path)

    with QueueRepository(path, sync_usage=False) as repository:
        values = (
            41,
            "capture",
            "claude",
            "session-41",
            "memory",
            "queued",
            "queued",
            "2026-08-18T12:00:00+00:00",
            "2026-08-18T12:00:00+00:00",
        )
        sql = """
            INSERT INTO status_runs (
                job_id, kind, source_agent, session_id, project, state, phase,
                started_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        repository._connection.execute(sql, values)

        with pytest.raises(sqlite3.IntegrityError, match="status_runs.job_id"):
            repository._connection.execute(sql, values)


def test_status_runs_allow_only_one_run_per_compile_operation(tmp_path):
    path = tmp_path / "jobs.sqlite3"

    with QueueRepository(path, sync_usage=False) as repository:
        values = (
            "auto-compile:2026-08-18:abc",
            "compile",
            "system",
            "2026-08-18",
            "memory",
            "queued",
            "reserved",
            "2026-08-18T16:00:00+00:00",
            "2026-08-18T16:00:00+00:00",
        )
        sql = """
            INSERT INTO status_runs (
                operation_key, kind, source_agent, session_id, project, state, phase,
                started_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        repository._connection.execute(sql, values)

        with pytest.raises(sqlite3.IntegrityError, match="status_runs.operation_key"):
            repository._connection.execute(sql, values)


@pytest.mark.parametrize(
    ("job_id", "operation_key"),
    [(None, None), (41, "auto-compile:2026-08-18:abc")],
)
def test_status_run_requires_exactly_one_operation_identity(
    tmp_path, job_id, operation_key
):
    path = tmp_path / "jobs.sqlite3"
    _create_version_2_queue(path)

    with (
        QueueRepository(path, sync_usage=False) as repository,
        pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"),
    ):
        repository._connection.execute(
            """
            INSERT INTO status_runs (
                job_id, operation_key, kind, source_agent, session_id, project,
                state, phase, started_at, updated_at
            ) VALUES (?, ?, 'capture', 'claude', 'session-41', 'memory',
                'queued', 'queued', '2026-08-18T12:00:00+00:00',
                '2026-08-18T12:00:00+00:00')
            """,
            (job_id, operation_key),
        )


def test_queue_rejects_a_newer_schema_version(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 4")
    connection.close()

    with pytest.raises(
        RuntimeError,
        match="queue schema 4 is newer than supported version 3",
    ):
        QueueRepository(path, sync_usage=False)


def test_status_domain_types_are_immutable_and_copy_event_details():
    run = StatusRun(
        id=1,
        job_id=41,
        operation_key=None,
        kind="capture",
        source_agent="claude",
        session_id="session-41",
        project="memory",
        state="running",
        phase="codex_started",
        summary=None,
        error=None,
        started_at=NOW,
        updated_at=NOW,
        completed_at=None,
        redaction_env={},
    )
    source_details = {"chars_saved": 120}
    event = StatusEvent(
        id=2,
        run_id=1,
        phase="daily_log_write_started",
        level="info",
        provider=None,
        attempt=1,
        message="Writing daily log",
        details=source_details,
        created_at=NOW,
        redaction_env={},
    )
    source_details["chars_saved"] = 999

    with pytest.raises(FrozenInstanceError):
        run.phase = "succeeded"
    with pytest.raises(TypeError):
        event.details["chars_saved"] = 999
    assert event.details == {"chars_saved": 120}


def test_runtime_status_vocabularies_match_the_literal_types():
    for state in get_args(RunState):
        assert _status_run(state=state).state == state
    for level in get_args(EventLevel):
        assert _status_event(level=level).level == level
    for provider in get_args(ProviderName):
        assert _status_event(provider=provider).provider == provider


def test_allowed_phases_cover_flush_and_compile_lifecycles():
    assert ALLOWED_PHASES == frozenset(
        {
            "queued",
            "worker_claimed",
            "codex_started",
            "codex_succeeded",
            "codex_failed",
            "claude_started",
            "claude_succeeded",
            "claude_failed",
            "daily_log_write_started",
            "retry_wait",
            "recovery_pending",
            "succeeded",
            "failed",
            "dead",
            "reserved",
            "staging_started",
            "provider_started",
            "validation_started",
            "apply_started",
            "generation_recovered",
        }
    )


def test_status_reasons_are_bounded_redacted_and_single_line():
    secret = "credential-value-never-persist"
    result = normalize_status_reason(
        f"provider exposed {secret}\n" + ("x" * 2_000),
        {"OPENAI_API_KEY": secret},
    )

    assert secret not in result
    assert "[REDACTED]" in result
    assert "\n" not in result
    assert len(result) == 1_000
    assert normalize_status_reason(None, {}) is None


def test_queue_and_status_share_a_dependency_neutral_redaction_utility():
    from scripts.privacy import normalize_persistence_reason as privacy_normalize
    from scripts.queue import normalize_persistence_reason as queue_normalize

    assert queue_normalize is privacy_normalize


@pytest.mark.parametrize("module_name", ["status_store", "queue"])
def test_status_privacy_imports_work_in_direct_script_mode(module_name):
    root = Path(__file__).resolve().parents[2]
    code = f"""
import importlib
import sys
from pathlib import Path

root = Path({str(Path(__file__).resolve().parents[2])!r})
sys.path[:] = [path for path in sys.path if Path(path or '.').resolve() != root]
sys.path.insert(0, str(root / 'scripts'))
module = importlib.import_module({module_name!r})
privacy = importlib.import_module('privacy')
assert module.normalize_persistence_reason is privacy.normalize_persistence_reason
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_summaries_are_bounded_and_normalized_without_fabricating_text():
    assert normalize_summary("  Saved\n  42 characters  ", {}) == "Saved 42 characters"
    assert normalize_summary("x" * 1_001, {}) == "x" * 1_000
    assert normalize_summary(" \n ", {}) is None
    assert normalize_summary(None, {}) is None


def _status_run(**overrides):
    values = {
        "id": 1,
        "job_id": 41,
        "operation_key": None,
        "kind": "capture",
        "source_agent": "claude",
        "session_id": "session-41",
        "project": "memory",
        "state": "running",
        "phase": "codex_started",
        "summary": None,
        "error": None,
        "started_at": NOW,
        "updated_at": NOW,
        "completed_at": None,
        "redaction_env": {},
    }
    values.update(overrides)
    return StatusRun(**values)


def _status_event(**overrides):
    values = {
        "id": 2,
        "run_id": 1,
        "phase": "codex_started",
        "level": "info",
        "provider": "codex",
        "attempt": 1,
        "message": None,
        "details": {},
        "created_at": NOW,
        "redaction_env": {},
    }
    values.update(overrides)
    return StatusEvent(**values)


def test_status_domain_constructors_preserve_the_original_call_contract():
    run_values = dict(_status_run().__dict__)
    event = _status_event()
    event_values = dict(event.__dict__)
    run_values.pop("redaction_env", None)
    event_values.pop("redaction_env", None)
    event_values["details"] = dict(event.details)

    assert StatusRun(**run_values).state == "running"
    assert StatusEvent(**event_values).level == "info"


@pytest.mark.parametrize("state", ["pending", "leased", "complete", "unknown"])
def test_status_run_rejects_invalid_state(state):
    with pytest.raises(ValueError, match="state"):
        _status_run(state=state)


@pytest.mark.parametrize("phase", ["", "extracting", "provider_output"])
def test_status_records_reject_invalid_phases(phase):
    with pytest.raises(ValueError, match="phase"):
        _status_run(phase=phase)
    with pytest.raises(ValueError, match="phase"):
        _status_event(phase=phase)


@pytest.mark.parametrize("level", ["debug", "critical", ""])
def test_status_event_rejects_invalid_level(level):
    with pytest.raises(ValueError, match="level"):
        _status_event(level=level)


@pytest.mark.parametrize("attempt", [0, -1, True, "1"])
def test_status_event_rejects_invalid_attempt(attempt):
    with pytest.raises(ValueError, match="attempt"):
        _status_event(attempt=attempt)


@pytest.mark.parametrize(
    "provider",
    ["", "system", "openai", "credential-value-never-persist"],
)
def test_status_event_rejects_unapproved_provider_names(provider):
    with pytest.raises(ValueError, match="provider"):
        _status_event(provider=provider)


def test_status_records_normalize_and_redact_all_persisted_text():
    secret = "credential-value-never-persist"
    env = {"OPENAI_API_KEY": secret}
    run = _status_run(
        summary=f" Saved {secret}\n" + ("s" * 2_000),
        error=f" Failed with {secret}\n" + ("e" * 2_000),
        redaction_env=env,
    )
    event = _status_event(
        message=f" Provider exposed {secret}\n" + ("m" * 2_000),
        redaction_env=env,
    )

    for text in (run.summary, run.error, event.message):
        assert text is not None
        assert secret not in text
        assert "[REDACTED]" in text
        assert "\n" not in text
        assert len(text) == 1_000


def test_optional_event_message_normalizes_whitespace_to_none():
    assert _status_event(message=" \n\t ").message is None


def test_operation_runs_are_idempotent_and_have_a_queryable_timeline(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(
        path,
        clock=lambda: NOW,
        redaction_env={"OPENAI_API_KEY": "operation-secret"},
        sync_usage=False,
    ) as repository:
        first = repository.create_operation_run(
            "auto-compile:2026-08-18:abc",
            kind="compile",
            source_agent="system",
            session_id="2026-08-18",
            project="memory",
        )
        second = repository.create_operation_run(
            "auto-compile:2026-08-18:abc",
            kind="compile",
            source_agent="system",
            session_id="2026-08-18",
            project="memory",
        )

        assert first == second
        assert first.operation_key is not None
        assert repository.status_run_for_operation(first.operation_key) == first
        assert [event.phase for event in repository.status_events(first.id)] == [
            "reserved"
        ]

        repository.transition_operation_run(
            first.id,
            "running",
            "staging_started",
        )
        repository.append_operation_event(
            first.id,
            "provider_started",
            provider="codex",
            attempt=1,
            message="  Calling operation-secret\nnow  ",
            details={"elapsed_ms": 0},
        )
        running = repository.status_run_for_operation(first.operation_key)
        assert running is not None
        assert running.phase == "provider_started"
        assert running.state == "running"
        assert [event.phase for event in repository.status_events(first.id)] == [
            "reserved",
            "staging_started",
            "provider_started",
        ]
        assert repository.status_events(first.id)[-1].message == "Calling [REDACTED] now"


def test_operation_writer_rejects_unapproved_provider_before_persistence(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    secret = "credential-value-never-persist"
    with QueueRepository(
        path,
        clock=lambda: NOW,
        redaction_env={"OPENAI_API_KEY": secret},
        sync_usage=False,
    ) as repository:
        run = repository.create_operation_run(
            "auto-compile:2026-08-18:provider",
            kind="compile",
            source_agent="system",
            session_id="2026-08-18",
            project="memory",
        )

        with pytest.raises(ValueError, match="provider"):
            repository.append_operation_event(
                run.id,
                "provider_started",
                provider=secret,
            )

        unchanged = repository.status_run_for_operation(run.operation_key)
        assert unchanged is not None
        assert unchanged.phase == "reserved"
        assert [event.phase for event in repository.status_events(run.id)] == [
            "reserved"
        ]

        with pytest.raises(ValueError, match="provider"):
            repository.transition_operation_run(
                run.id,
                "running",
                "provider_started",
                provider=secret,
            )
        assert repository.status_run_for_operation(run.operation_key) == unchanged


def test_status_event_readback_rejects_an_unapproved_provider(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW, sync_usage=False) as repository:
        run = repository.create_operation_run(
            "auto-compile:2026-08-18:invalid-readback",
            kind="compile",
            source_agent="system",
            session_id="2026-08-18",
            project="memory",
        )
        repository._connection.execute(
            """
            INSERT INTO status_events (
                run_id, phase, level, provider, details_json, created_at
            ) VALUES (?, 'provider_started', 'info', ?, '{}', ?)
            """,
            (run.id, "provider-secret", NOW.isoformat()),
        )

        with pytest.raises(ValueError, match="provider"):
            repository.status_events(run.id)


def test_operation_transition_updates_summary_error_and_completion_atomically(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW, sync_usage=False) as repository:
        run = repository.create_operation_run(
            "auto-compile:2026-08-18:def",
            kind="compile",
            source_agent="system",
            session_id="2026-08-18",
            project="memory",
        )

        assert run.operation_key is not None
        repository.transition_operation_run(
            run.id,
            "running",
            "staging_started",
        )
        repository.transition_operation_run(
            run.id,
            "succeeded",
            "succeeded",
            summary=" Updated 6 articles ",
            message="Compile complete",
            details={"changed_files": 6},
        )

        completed = repository.status_run_for_operation(run.operation_key)
        assert completed is not None
        assert completed.state == "succeeded"
        assert completed.phase == "succeeded"
        assert completed.summary == "Updated 6 articles"
        assert completed.error is None
        assert completed.completed_at == NOW
        terminal = repository.status_events(run.id)[-1]
        assert terminal.phase == "succeeded"
        assert terminal.message == "Compile complete"
        assert terminal.details == {"changed_files": 6}


def test_operation_events_and_transitions_reject_queued_success_contradictions(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW, sync_usage=False) as repository:
        run = repository.create_operation_run(
            "auto-compile:2026-08-18:queued-contradiction",
            kind="compile",
            source_agent="system",
            session_id="",
            project="memory",
        )

        with pytest.raises(ValueError, match="phase"):
            repository.append_operation_event(run.id, "succeeded")
        with pytest.raises(ValueError, match="transition"):
            repository.transition_operation_run(run.id, "succeeded", "succeeded")

        unchanged = repository.status_run_for_operation(run.operation_key)
        assert unchanged == run
        assert [event.phase for event in repository.status_events(run.id)] == [
            "reserved"
        ]


def test_terminal_operation_rejects_backwards_transitions_and_new_events(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW, sync_usage=False) as repository:
        run = repository.create_operation_run(
            "auto-compile:2026-08-18:terminal",
            kind="compile",
            source_agent="system",
            session_id="",
            project="memory",
        )
        repository.transition_operation_run(run.id, "running", "staging_started")
        repository.transition_operation_run(run.id, "succeeded", "succeeded")

        with pytest.raises(ValueError, match="terminal"):
            repository.transition_operation_run(run.id, "running", "staging_started")
        with pytest.raises(ValueError, match="terminal"):
            repository.append_operation_event(run.id, "provider_started")

        completed = repository.status_run_for_operation(run.operation_key)
        assert completed is not None
        assert completed.state == "succeeded"
        assert completed.phase == "succeeded"
        assert [event.phase for event in repository.status_events(run.id)] == [
            "reserved",
            "staging_started",
            "succeeded",
        ]


def test_compile_operation_can_run_retry_recover_and_succeed(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW, sync_usage=False) as repository:
        run = repository.create_operation_run(
            "auto-compile:2026-08-18:valid-lifecycle",
            kind="compile",
            source_agent="system",
            session_id="",
            project="memory",
        )
        repository.transition_operation_run(run.id, "running", "staging_started")
        for phase in (
            "provider_started",
            "validation_started",
            "apply_started",
        ):
            repository.append_operation_event(run.id, phase)
        repository.transition_operation_run(
            run.id,
            "retrying",
            "retry_wait",
            error="retry compile",
            level="warning",
        )
        repository.transition_operation_run(
            run.id,
            "running",
            "generation_recovered",
        )
        repository.transition_operation_run(
            run.id,
            "succeeded",
            "succeeded",
            summary="Updated 4 articles",
        )

        completed = repository.status_run_for_operation(run.operation_key)
        assert completed is not None
        assert completed.state == "succeeded"
        assert completed.phase == "succeeded"
        assert completed.summary == "Updated 4 articles"
        assert completed.error is None
        assert [event.phase for event in repository.status_events(run.id)] == [
            "reserved",
            "staging_started",
            "provider_started",
            "validation_started",
            "apply_started",
            "retry_wait",
            "generation_recovered",
            "succeeded",
        ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation_key", " leading"),
        ("operation_key", "contains\ncontrol"),
        ("operation_key", "x" * 513),
        ("kind", "unsupported"),
        ("source_agent", "worker"),
        ("session_id", ""),
        ("session_id", "session\ncontrol"),
        ("session_id", "x" * 257),
        ("project", " trailing "),
        ("project", "project\tcontrol"),
        ("project", "x" * 257),
    ],
)
def test_operation_identity_rejects_noncanonical_or_unsupported_values(
    tmp_path, field, value
):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW, sync_usage=False) as repository:
        identity = {
            "operation_key": "auto-compile:2026-08-18:identity",
            "kind": "compile",
            "source_agent": "claude",
            "session_id": "session-1",
            "project": "memory",
        }
        identity[field] = value
        operation_key = identity.pop("operation_key")

        with pytest.raises(ValueError, match=field):
            repository.create_operation_run(operation_key, **identity)

        assert repository._connection.execute(
            "SELECT count(*) FROM status_runs"
        ).fetchone()[0] == 0


def test_operation_identity_rejects_configured_secrets(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    secret = "operation-secret-value"
    with QueueRepository(
        path,
        clock=lambda: NOW,
        redaction_env={"SERVICE_TOKEN": secret},
        sync_usage=False,
    ) as repository, pytest.raises(ValueError, match="operation_key"):
        repository.create_operation_run(
            f"auto-compile:{secret}",
            kind="compile",
            source_agent="system",
            session_id="",
            project="memory",
        )


def test_system_operation_allows_an_empty_session_id(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW, sync_usage=False) as repository:
        run = repository.create_operation_run(
            "auto-compile:2026-08-18:system-no-session",
            kind="compile",
            source_agent="system",
            session_id="",
            project="memory",
        )

        assert run.session_id == ""


def test_concurrent_operation_creation_reuses_one_run_and_reserved_event(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, sync_usage=False):
        pass
    ready = threading.Barrier(2)

    def create(_):
        with QueueRepository(path, clock=lambda: NOW, sync_usage=False) as repository:
            ready.wait(timeout=2)
            return repository.create_operation_run(
                "auto-compile:2026-08-18:concurrent",
                kind="compile",
                source_agent="system",
                session_id="",
                project="memory",
            ).id

    with ThreadPoolExecutor(max_workers=2) as executor:
        run_ids = list(executor.map(create, range(2)))

    assert run_ids[0] == run_ids[1]
    with QueueRepository(path, sync_usage=False) as repository:
        assert [
            event.phase for event in repository.status_events(run_ids[0])
        ] == ["reserved"]


def test_idempotent_operation_creation_still_validates_requested_phase(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: NOW, sync_usage=False) as repository:
        repository.create_operation_run(
            "auto-compile:2026-08-18:phase",
            kind="compile",
            source_agent="system",
            session_id="2026-08-18",
            project="memory",
        )

        with pytest.raises(ValueError, match="phase"):
            repository.create_operation_run(
                "auto-compile:2026-08-18:phase",
                kind="compile",
                source_agent="system",
                session_id="2026-08-18",
                project="memory",
                phase="provider_output",
            )


def test_details_accept_only_safe_scalar_json_metadata():
    details = normalize_details(
        {
            "chars_saved": 42,
            "changed_files": 3,
            "retry_at": "2026-08-18T12:05:00+00:00",
            "elapsed_ms": 1250,
        }
    )

    assert details == {
        "chars_saved": 42,
        "changed_files": 3,
        "retry_at": "2026-08-18T12:05:00.000000+00:00",
        "elapsed_ms": 1250,
    }
    with pytest.raises(TypeError):
        details["chars_saved"] = 43


def test_details_normalize_timezone_aware_retry_at_to_utc():
    details = normalize_details({"retry_at": "2026-08-18T05:05:00-07:00"})

    assert details["retry_at"] == "2026-08-18T12:05:00.000000+00:00"


@pytest.mark.parametrize("key", ["chars_saved", "changed_files", "elapsed_ms"])
@pytest.mark.parametrize("value", [-1, True, False, 1.5, "12", None])
def test_count_and_duration_details_require_nonnegative_integers(key, value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        normalize_details({key: value})


@pytest.mark.parametrize(
    "value",
    [
        None,
        123,
        True,
        "not-a-timestamp",
        "2026-08-18T12:05:00",
        "2026-08-18x12:05:00+00:00",
        "2026-08-18T12:05:00+00:00 sk-proj-secret-value",
        "0001-01-01T00:00:00+23:59",
    ],
)
def test_retry_at_requires_a_timezone_aware_iso_timestamp(value):
    with pytest.raises(ValueError, match="timezone-aware ISO-8601"):
        normalize_details({"retry_at": value})


@pytest.mark.parametrize("value", [["secret"], {"nested": "secret"}])
def test_details_reject_non_scalar_values(value):
    with pytest.raises(ValueError, match="scalar JSON"):
        normalize_details({"chars_saved": value})


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "model_output",
        "arbitrary",
        "prompt",
        "transcript",
        "content",
        "output",
        "rendered_context",
    ],
)
def test_details_reject_unapproved_and_sensitive_keys(key):
    with pytest.raises(ValueError, match="not permitted"):
        normalize_details({key: "private data"})


def test_details_reject_non_finite_numbers_and_oversized_metadata():
    with pytest.raises(ValueError, match="finite"):
        normalize_details({"elapsed_ms": float("nan")})
    with pytest.raises(ValueError, match="at most 32"):
        normalize_details({f"field_{index}": index for index in range(33)})
    with pytest.raises(ValueError, match="at most 1000"):
        normalize_details({"retry_at": "x" * 1_001})


READ_NOW = datetime(2026, 8, 18, 18, 0, tzinfo=UTC)


def _insert_projection_run(
    connection: sqlite3.Connection,
    *,
    run_id: int,
    state: str,
    phase: str,
    updated_at: datetime,
    project: str,
    summary: str | None = None,
    error: str | None = None,
    kind: str = "capture",
) -> None:
    connection.execute(
        """
        INSERT INTO status_runs (
            id, operation_key, kind, source_agent, session_id, project, state,
            phase, summary, error, started_at, updated_at, completed_at
        ) VALUES (?, ?, ?, 'system', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            f"projection:{run_id}",
            kind,
            f"session-{run_id}",
            project,
            state,
            phase,
            summary,
            error,
            (updated_at - timedelta(minutes=1)).isoformat(),
            updated_at.isoformat(),
            updated_at.isoformat() if state in {"succeeded", "failed", "dead"} else None,
        ),
    )


def _create_projection_queue(path: Path) -> None:
    with QueueRepository(path, sync_usage=False) as repository:
        connection = repository._connection
        _insert_projection_run(
            connection,
            run_id=1,
            state="running",
            phase="codex_started",
            updated_at=READ_NOW - timedelta(minutes=1),
            project="active-project",
            summary="Extracting",
        )
        _insert_projection_run(
            connection,
            run_id=2,
            state="retrying",
            phase="retry_wait",
            updated_at=READ_NOW - timedelta(minutes=2),
            project="retry-project",
            error="retry later",
        )
        _insert_projection_run(
            connection,
            run_id=3,
            state="failed",
            phase="failed",
            updated_at=READ_NOW - timedelta(days=30),
            project="attention-project",
            error="old but unacknowledged",
        )
        _insert_projection_run(
            connection,
            run_id=4,
            state="failed",
            phase="failed",
            updated_at=READ_NOW - timedelta(days=1),
            project="acknowledged-project",
            error="reviewed failure",
        )
        _insert_projection_run(
            connection,
            run_id=5,
            state="succeeded",
            phase="succeeded",
            updated_at=READ_NOW - timedelta(days=2),
            project="recent-project",
            summary="Saved 42 characters",
        )
        _insert_projection_run(
            connection,
            run_id=6,
            state="succeeded",
            phase="succeeded",
            updated_at=READ_NOW - timedelta(days=8),
            project="old-project",
            summary="Old successful result",
        )
        _insert_projection_run(
            connection,
            run_id=7,
            state="running",
            phase="validation_started",
            updated_at=READ_NOW - timedelta(seconds=30),
            project="memory",
            summary="Validating staged changes",
            kind="compile",
        )
        connection.execute(
            """
            INSERT INTO status_events (
                run_id, phase, level, provider, attempt, message,
                details_json, created_at
            ) VALUES (
                1, 'codex_started', 'info', 'codex', 1, 'Calling provider',
                '{"elapsed_ms":125}', '2026-08-18T17:59:00+00:00'
            )
            """
        )
        connection.commit()


def test_snapshot_groups_active_attention_and_seven_day_recent(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    _create_projection_queue(path)
    observer = status_store_module.ObserverState(
        version=1,
        acknowledged_run_ids=frozenset({4}),
    )

    snapshot = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=observer,
    )

    assert [run.id for run in snapshot.active] == [1, 2]
    assert [run.id for run in snapshot.attention] == [3]
    assert [run.id for run in snapshot.recent] == [4, 5]
    assert snapshot.compile.run is not None
    assert snapshot.compile.run.id == 7
    assert snapshot.health_alerts == ()


def test_snapshot_query_searches_older_approved_history(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    _create_projection_queue(path)
    observer = status_store_module.ObserverState.empty()

    snapshot = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=observer,
        query="old-project",
    )

    assert snapshot.active == ()
    assert snapshot.attention == ()
    assert [run.id for run in snapshot.recent] == [6]


def test_default_snapshot_filters_old_successes_in_sql(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    _create_projection_queue(path)
    original_from_row = status_store_module.status_run_from_row
    observed_ids: list[int] = []

    def record_rows(row, **kwargs):
        observed_ids.append(row["id"])
        return original_from_row(row, **kwargs)

    monkeypatch.setattr(status_store_module, "status_run_from_row", record_rows)

    default = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState.empty(),
    )

    assert 6 not in observed_ids
    assert all(run.id != 6 for run in default.recent)
    observed_ids.clear()

    searched = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState.empty(),
        query="old-project",
    )

    assert 6 in observed_ids
    assert [run.id for run in searched.recent] == [6]


def test_recent_sql_filter_includes_equivalent_exact_boundary_timestamps(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    _create_projection_queue(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            UPDATE status_runs
            SET updated_at = ?, completed_at = ?
            WHERE id = 5
            """,
            (
                "2026-08-11T18:00:00+00:00",
                "2026-08-11T18:00:00+00:00",
            ),
        )
        connection.execute(
            """
            UPDATE status_runs
            SET updated_at = ?, completed_at = ?
            WHERE id = 6
            """,
            (
                "2026-08-11T11:00:00-07:00",
                "2026-08-11T11:00:00-07:00",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    snapshot = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState.empty(),
    )

    assert {run.id for run in snapshot.recent} >= {5, 6}


def test_snapshot_injects_health_alerts_without_mutating_them(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    _create_projection_queue(path)
    alert = status_store_module.HealthAlert(
        created_at=READ_NOW,
        level="error",
        message="Hook could not reach SQLite",
    )

    snapshot = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState.empty(),
        health_alerts=(alert,),
    )

    assert snapshot.health_alerts == (alert,)
    with pytest.raises(FrozenInstanceError):
        snapshot.health_alerts = ()


def test_read_run_details_returns_immutable_safe_timeline(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    _create_projection_queue(path)

    details = status_store_module.read_run_details(path, 1)

    assert details.run.id == 1
    assert details.timeline_available is True
    assert [event.phase for event in details.events] == ["codex_started"]
    assert details.events[0].details == {"elapsed_ms": 125}
    with pytest.raises(TypeError):
        details.events[0].details["elapsed_ms"] = 500


def test_run_details_maps_unknown_provider_attempt_outcome_to_unknown(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: READ_NOW, sync_usage=False) as repository:
        result = repository.enqueue_capture(_version_2_session())
        run = repository.status_run_for_job(result.job_id)
        repository._connection.execute(
            """
            INSERT INTO provider_attempts (
                job_id, provider, model, task, started_at, ended_at,
                outcome, elapsed_ms
            ) VALUES (?, 'codex', 'gpt-5.6-luna', 'extract', ?, ?, ?, 1)
            """,
            (
                result.job_id,
                READ_NOW.isoformat(),
                READ_NOW.isoformat(),
                "credential-like-unknown-outcome",
            ),
        )
        repository._connection.commit()

    details = status_store_module.read_run_details(path, run.id)

    assert details.provider_attempts[0].outcome == "unknown"


def test_missing_database_is_a_typed_diagnostic_and_is_not_created(tmp_path):
    path = tmp_path / "missing.sqlite3"

    with pytest.raises(status_store_module.StatusDatabaseUnavailable) as caught:
        status_store_module.read_snapshot(
            path,
            now=READ_NOW,
            observer_state=status_store_module.ObserverState.empty(),
        )

    assert caught.value.path == path.resolve()
    assert not path.exists()


def test_status_read_rejects_symlinked_queue_path(tmp_path):
    actual = tmp_path / "actual.sqlite3"
    _create_projection_queue(actual)
    link = tmp_path / "linked.sqlite3"
    try:
        link.symlink_to(actual)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(status_store_module.StatusReadError, match="unsafe"):
        status_store_module.read_snapshot(
            link,
            now=READ_NOW,
            observer_state=status_store_module.ObserverState.empty(),
        )


@pytest.mark.skipif(not hasattr(os, "link"), reason="hard links unavailable")
def test_status_read_rejects_hardlinked_queue_path(tmp_path):
    actual = tmp_path / "actual.sqlite3"
    _create_projection_queue(actual)
    link = tmp_path / "hardlinked.sqlite3"
    os.link(actual, link)

    with pytest.raises(status_store_module.StatusReadError, match="unsafe"):
        status_store_module.read_snapshot(
            link,
            now=READ_NOW,
            observer_state=status_store_module.ObserverState.empty(),
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits required")
@pytest.mark.parametrize("mode", [0o644, 0o666])
def test_status_read_rejects_insecure_queue_mode(tmp_path, mode):
    path = tmp_path / "jobs.sqlite3"
    _create_projection_queue(path)
    path.chmod(mode)

    with pytest.raises(status_store_module.StatusReadError, match="unsafe"):
        status_store_module.read_snapshot(
            path,
            now=READ_NOW,
            observer_state=status_store_module.ObserverState.empty(),
        )


def test_status_read_rejects_unsafe_windows_queue_acl(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    _create_projection_queue(path)
    monkeypatch.setattr(utils_module, "_PRIVATE_STATE_DIR_FD_SUPPORTED", False)
    monkeypatch.setattr(utils_module, "_windows_acl_required", lambda: True)
    monkeypatch.setattr(
        utils_module,
        "_validate_windows_inherited_directory",
        lambda _path: None,
        raising=False,
    )
    monkeypatch.setattr(
        utils_module,
        "_validate_windows_owner_only_file_descriptor",
        lambda *_: (_ for _ in ()).throw(
            PermissionError("queue ACL is unsafe")
        ),
    )

    with pytest.raises(status_store_module.StatusReadError, match="unsafe"):
        status_store_module.read_snapshot(
            path,
            now=READ_NOW,
            observer_state=status_store_module.ObserverState.empty(),
        )


def test_status_read_rejects_queue_swapped_before_secure_open(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    replacement = tmp_path / "replacement.sqlite3"
    backup = tmp_path / "original.sqlite3"
    _create_projection_queue(path)
    _create_projection_queue(replacement)
    real_open = utils_module.os.open
    swapped = False

    def swap_before_open(candidate, *args, **kwargs):
        nonlocal swapped
        if candidate == path.name and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            path.rename(backup)
            replacement.rename(path)
        return real_open(candidate, *args, **kwargs)

    monkeypatch.setattr(utils_module.os, "open", swap_before_open)

    with pytest.raises(status_store_module.StatusReadError, match="unsafe"):
        status_store_module.read_snapshot(
            path,
            now=READ_NOW,
            observer_state=status_store_module.ObserverState.empty(),
        )


def test_version_2_queue_is_synthesized_without_migration_or_writes(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    _create_version_2_queue(path)
    before = path.read_bytes()

    snapshot = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState.empty(),
    )

    assert len(snapshot.recent) == 1
    legacy = snapshot.recent[0]
    assert legacy.job_id == 41
    assert legacy.timeline_available is False
    assert status_store_module.read_run_details(path, legacy.id).timeline_available is False
    assert path.read_bytes() == before
    connection = sqlite3.connect(path)
    try:
        assert "status_runs" not in _table_names(connection)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        connection.close()


def test_legacy_projection_replaces_unsafe_identity_and_excludes_it_from_search(
    tmp_path, monkeypatch
):
    path = tmp_path / "legacy-private.sqlite3"
    secret = "credential-value-never-display"
    monkeypatch.setenv("AI_MEMORY_VIEW_TOKEN", secret)
    _create_version_2_queue(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            UPDATE jobs
            SET kind = ?, source_agent = ?, session_id = ?, project = ?
            WHERE id = 41
            """,
            (
                f"capture-{secret}",
                "claude\ncontrol",
                "s" * 300,
                f"project-{secret}",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    snapshot = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState.empty(),
    )
    legacy = snapshot.recent[0]

    assert legacy.kind == "unknown"
    assert legacy.source_agent == "unknown"
    assert legacy.session_id == "unknown"
    assert legacy.project == "unknown"
    for query in (secret, "control", "s" * 100):
        searched = status_store_module.read_snapshot(
            path,
            now=READ_NOW,
            observer_state=status_store_module.ObserverState.empty(),
            query=query,
        )
        assert searched.active == ()
        assert searched.attention == ()
        assert searched.recent == ()


@pytest.mark.parametrize(
    "operation_key",
    [
        "compile-credential-value-never-display",
        "x" * 513,
        "compile\ncontrol",
    ],
)
def test_projection_removes_unsafe_operation_keys(
    tmp_path, monkeypatch, operation_key
):
    path = tmp_path / "operation-private.sqlite3"
    monkeypatch.setenv("AI_MEMORY_VIEW_TOKEN", "credential-value-never-display")
    _create_projection_queue(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE status_runs SET operation_key = ? WHERE id = 7",
            (operation_key,),
        )
        connection.commit()
    finally:
        connection.close()

    snapshot = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState.empty(),
    )
    details = status_store_module.read_run_details(path, 7)
    searched = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState.empty(),
        query="credential-value-never-display",
    )

    assert snapshot.compile.run is not None
    assert snapshot.compile.run.operation_key is None
    assert details.run.operation_key is None
    assert searched.active == ()
    assert searched.attention == ()
    assert searched.recent == ()


@pytest.mark.parametrize(
    "component",
    [
        "hook\nprivate",
        "x" * 257,
        "component-credential-value-never-display",
    ],
)
def test_health_alert_rejects_unsafe_components(component, monkeypatch):
    monkeypatch.setenv("AI_MEMORY_VIEW_TOKEN", "credential-value-never-display")

    with pytest.raises(ValueError, match="component"):
        status_store_module.HealthAlert(
            created_at=READ_NOW,
            level="error",
            message="Hook failed",
            component=component,
        )


def test_health_alert_redacts_configured_secrets_by_default(monkeypatch):
    secret = "credential-value-never-display"
    monkeypatch.setenv("AI_MEMORY_VIEW_TOKEN", secret)

    alert = status_store_module.HealthAlert(
        created_at=READ_NOW,
        level="error",
        message=f"Hook exposed {secret}",
    )

    assert alert.message == "Hook exposed [REDACTED]"


def test_snapshot_synthesizes_uninstrumented_v3_jobs_without_payload_search(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, sync_usage=False) as repository:
        repository._connection.execute(
            """
            INSERT INTO jobs (
                id, kind, source_agent, session_id, project, cwd, trigger,
                source_path, source_hash, payload_json, status, attempt_count,
                available_at, created_at, updated_at, completed_at
            ) VALUES (
                88, 'capture', 'claude', 'legacy-session', 'legacy-project',
                '/memory', 'session_end', '/memory/private.jsonl', 'hash-88',
                '{"rendered_context":"private-search-needle"}', 'succeeded', 1,
                '2026-08-18T12:00:00+00:00', '2026-08-18T12:00:00+00:00',
                '2026-08-18T12:01:00+00:00', '2026-08-18T12:01:00+00:00'
            )
            """
        )
        repository._connection.commit()

    default = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState.empty(),
    )
    private_search = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState.empty(),
        query="private-search-needle",
    )

    assert [run.job_id for run in default.recent] == [88]
    assert default.recent[0].timeline_available is False
    assert private_search.active == ()
    assert private_search.attention == ()
    assert private_search.recent == ()


def test_load_observer_state_returns_empty_for_missing_or_malformed_files(tmp_path):
    path = tmp_path / "scripts" / "status-view.json"

    assert status_store_module.load_observer_state(path) == (
        status_store_module.ObserverState.empty()
    )
    assert not path.exists()

    path.parent.mkdir()
    path.write_text('{"version":1,"acknowledged_run_ids":"not-a-list"}')
    path.chmod(0o600)

    assert status_store_module.load_observer_state(path) == (
        status_store_module.ObserverState.empty()
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 2, "acknowledged_run_ids": [1]},
        {"version": 1, "acknowledged_run_ids": [0]},
        {"version": 1, "acknowledged_run_ids": [True]},
        {"version": 1, "acknowledged_run_ids": [1], "extra": "field"},
    ],
)
def test_load_observer_state_requires_the_exact_version_1_schema(tmp_path, payload):
    path = tmp_path / "status-view.json"
    path.write_text(json.dumps(payload))
    path.chmod(0o600)

    assert status_store_module.load_observer_state(path) == (
        status_store_module.ObserverState.empty()
    )


def test_load_observer_state_rejects_oversized_or_nonprivate_files(tmp_path):
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (status_store_module.MAX_OBSERVER_STATE_BYTES + 1))
    oversized.chmod(0o600)
    public = tmp_path / "public.json"
    public.write_text('{"version":1,"acknowledged_run_ids":[1]}')
    public.chmod(0o644)

    assert status_store_module.load_observer_state(oversized) == (
        status_store_module.ObserverState.empty()
    )
    assert status_store_module.load_observer_state(public) == (
        status_store_module.ObserverState.empty()
    )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_load_observer_state_rejects_fifo_without_opening_it(tmp_path, monkeypatch):
    path = tmp_path / "status-view.json"
    os.mkfifo(path, 0o600)
    real_open = utils_module.os.open

    def reject_fifo_open(candidate, *args, **kwargs):
        if Path(candidate) == path:
            pytest.fail("observer reader attempted to open a FIFO")
        return real_open(candidate, *args, **kwargs)

    monkeypatch.setattr(utils_module.os, "open", reject_fifo_open)

    assert status_store_module.load_observer_state(path) == (
        status_store_module.ObserverState.empty()
    )


def test_load_observer_state_rejects_target_swapped_before_open(tmp_path, monkeypatch):
    path = tmp_path / "status-view.json"
    backup = tmp_path / "original.json"
    replacement = tmp_path / "replacement.json"
    path.write_text('{"version":1,"acknowledged_run_ids":[1]}')
    replacement.write_text('{"version":1,"acknowledged_run_ids":[99]}')
    path.chmod(0o600)
    replacement.chmod(0o600)
    real_open = utils_module.os.open
    swapped = False

    def swap_before_open(candidate, *args, **kwargs):
        nonlocal swapped
        is_target = Path(candidate) == path or (
            candidate == path.name and kwargs.get("dir_fd") is not None
        )
        if is_target and not swapped:
            swapped = True
            path.rename(backup)
            replacement.rename(path)
        return real_open(candidate, *args, **kwargs)

    monkeypatch.setattr(utils_module.os, "open", swap_before_open)

    assert status_store_module.load_observer_state(path) == (
        status_store_module.ObserverState.empty()
    )
    assert json.loads(backup.read_text())["acknowledged_run_ids"] == [1]


def test_load_observer_state_rejects_symlink_without_following(tmp_path):
    target = tmp_path / "actual.json"
    target.write_text('{"version":1,"acknowledged_run_ids":[7]}')
    target.chmod(0o600)
    link = tmp_path / "status-view.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")

    assert status_store_module.load_observer_state(link) == (
        status_store_module.ObserverState.empty()
    )


def test_observer_read_and_write_reject_symlinked_parent(tmp_path):
    real_parent = tmp_path / "real-scripts"
    real_parent.mkdir()
    target = real_parent / "status-view.json"
    status_store_module.acknowledge_run(target, 1)
    linked_parent = tmp_path / "scripts"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")

    assert status_store_module.load_observer_state(linked_parent / target.name) == (
        status_store_module.ObserverState.empty()
    )
    with pytest.raises(ValueError, match="linked ancestor"):
        status_store_module.acknowledge_run(linked_parent / target.name, 2)
    assert status_store_module.load_observer_state(target).acknowledged_run_ids == {
        1
    }


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits required")
def test_observer_read_rejects_group_or_world_writable_parent(tmp_path):
    parent = tmp_path / "scripts"
    parent.mkdir()
    target = parent / "status-view.json"
    status_store_module.acknowledge_run(target, 1)
    parent.chmod(0o777)

    assert status_store_module.load_observer_state(target) == (
        status_store_module.ObserverState.empty()
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits required")
def test_observer_write_rejects_writable_parent_and_preserves_prior(tmp_path):
    parent = tmp_path / "scripts"
    parent.mkdir()
    target = parent / "status-view.json"
    status_store_module.acknowledge_run(target, 1)
    prior = target.read_bytes()
    parent.chmod(0o777)

    with pytest.raises(ValueError, match="unsafe permissions"):
        status_store_module.acknowledge_run(target, 2)

    assert target.read_bytes() == prior


def _force_windows_observer_fallback(
    monkeypatch,
    *,
    validate_directory,
    validate_file,
    secure_directory=lambda _path, *, owner_only: None,
    secure_file=lambda _descriptor, _path: None,
):
    monkeypatch.setattr(utils_module, "_PRIVATE_STATE_DIR_FD_SUPPORTED", False)
    monkeypatch.setattr(utils_module, "_windows_acl_required", lambda: True)
    monkeypatch.setattr(
        utils_module,
        "_validate_windows_owner_only_directory",
        validate_directory,
        raising=False,
    )
    monkeypatch.setattr(
        utils_module,
        "_validate_windows_owner_only_file_descriptor",
        validate_file,
        raising=False,
    )
    monkeypatch.setattr(
        utils_module,
        "_secure_windows_runtime_directory",
        secure_directory,
    )
    monkeypatch.setattr(utils_module, "_secure_windows_runtime_file", secure_file)


def test_windows_observer_read_rejects_nonprivate_parent_acl(tmp_path, monkeypatch):
    parent = tmp_path / "scripts"
    parent.mkdir()
    target = parent / "status-view.json"
    target.write_text('{"version":1,"acknowledged_run_ids":[1]}')
    target.chmod(0o600)
    _force_windows_observer_fallback(
        monkeypatch,
        validate_directory=lambda _path: (_ for _ in ()).throw(
            PermissionError("directory ACL is not owner-only")
        ),
        validate_file=lambda *_: pytest.fail("unsafe parent should stop file open"),
    )

    assert status_store_module.load_observer_state(target) == (
        status_store_module.ObserverState.empty()
    )


def test_windows_observer_read_rejects_nonprivate_file_acl(tmp_path, monkeypatch):
    parent = tmp_path / "scripts"
    parent.mkdir()
    target = parent / "status-view.json"
    target.write_text('{"version":1,"acknowledged_run_ids":[1]}')
    target.chmod(0o600)
    _force_windows_observer_fallback(
        monkeypatch,
        validate_directory=lambda _path: None,
        validate_file=lambda *_: (_ for _ in ()).throw(
            PermissionError("file ACL is not owner-only")
        ),
    )

    assert status_store_module.load_observer_state(target) == (
        status_store_module.ObserverState.empty()
    )


@pytest.mark.parametrize("unsafe_boundary", ["parent", "file"])
def test_windows_observer_write_rejects_unsafe_acl_and_preserves_prior(
    tmp_path, monkeypatch, unsafe_boundary
):
    parent = tmp_path / "scripts"
    parent.mkdir()
    target = parent / "status-view.json"
    status_store_module.acknowledge_run(target, 1)
    prior = target.read_bytes()

    def validate_directory(_path):
        if unsafe_boundary == "parent":
            raise PermissionError("directory ACL is not owner-only")

    def validate_file(_descriptor, _path):
        if unsafe_boundary == "file":
            raise PermissionError("file ACL is not owner-only")

    _force_windows_observer_fallback(
        monkeypatch,
        validate_directory=validate_directory,
        validate_file=validate_file,
    )

    with pytest.raises(PermissionError, match="ACL is not owner-only"):
        status_store_module.acknowledge_run(target, 2)

    assert target.read_bytes() == prior


def test_windows_observer_write_secures_new_parent_temp_and_final_file(
    tmp_path, monkeypatch
):
    target = tmp_path / "memory" / "scripts" / "status-view.json"
    secured_directories: list[tuple[Path, bool]] = []
    secured_files: list[Path] = []
    validated_directories: list[Path] = []
    validated_files: list[Path] = []
    _force_windows_observer_fallback(
        monkeypatch,
        validate_directory=lambda path: validated_directories.append(Path(path)),
        validate_file=lambda _descriptor, path: validated_files.append(Path(path)),
        secure_directory=lambda path, *, owner_only: secured_directories.append(
            (Path(path), owner_only)
        ),
        secure_file=lambda _descriptor, path: secured_files.append(Path(path)),
    )

    state = status_store_module.acknowledge_run(target, 7)

    assert state.acknowledged_run_ids == {7}
    assert secured_directories == [(target.parent, True)]
    assert any(path.suffix == ".tmp" for path in secured_files)
    assert target.parent in validated_directories
    assert target in validated_files
    assert status_store_module.load_observer_state(target) == state


class _ObserverAclApi:
    def __init__(self, *, owner_only: bool):
        self.owner_only = owner_only
        self.closed: list[str] = []

    def open_directory(self, _path):
        return "directory"

    def open_file(self, _path, *, access):
        assert access
        return "file"

    def close(self, handle):
        self.closed.append(handle)

    def identity(self, _handle):
        return (1, 10)

    def is_reparse(self, _handle):
        return False

    def inspect(self, _handle):
        return SimpleNamespace(is_owner_only=self.owner_only)


def test_windows_observer_directory_acl_validator_rejects_broad_access(
    tmp_path, monkeypatch
):
    from scripts import windows_acl

    api = _ObserverAclApi(owner_only=False)
    monkeypatch.setattr(windows_acl, "_active_api", lambda _api: api)

    with pytest.raises(PermissionError, match="directory ACL is not owner-only"):
        utils_module._validate_windows_owner_only_directory(tmp_path)

    assert api.closed == ["directory"]


def test_windows_observer_file_acl_validator_rejects_broad_access(
    tmp_path, monkeypatch
):
    from scripts import windows_acl

    api = _ObserverAclApi(owner_only=False)
    monkeypatch.setattr(windows_acl, "_active_api", lambda _api: api)
    monkeypatch.setattr(
        utils_module,
        "msvcrt",
        SimpleNamespace(get_osfhandle=lambda _descriptor: "borrowed"),
    )

    with pytest.raises(PermissionError, match="file ACL is not owner-only"):
        utils_module._validate_windows_owner_only_file_descriptor(
            7,
            tmp_path / "status-view.json",
        )

    assert api.closed == ["file"]


def test_acknowledge_run_atomically_writes_private_exact_state(tmp_path):
    path = tmp_path / "scripts" / "status-view.json"

    first = status_store_module.acknowledge_run(path, 19)
    second = status_store_module.acknowledge_run(path, 12)
    duplicate = status_store_module.acknowledge_run(path, 19)

    assert first.acknowledged_run_ids == frozenset({19})
    assert second.acknowledged_run_ids == frozenset({12, 19})
    assert duplicate == second
    assert json.loads(path.read_text()) == {
        "version": 1,
        "acknowledged_run_ids": [12, 19],
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_acknowledge_run_refuses_unsafe_existing_target(tmp_path):
    target = tmp_path / "actual.json"
    target.write_text('{"version":1,"acknowledged_run_ids":[]}')
    target.chmod(0o600)
    link = tmp_path / "status-view.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(ValueError, match="private regular file"):
        status_store_module.acknowledge_run(link, 1)

    assert target.read_text() == '{"version":1,"acknowledged_run_ids":[]}'


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory descriptors required")
def test_atomic_observer_write_rejects_parent_swap_and_preserves_prior_state(
    tmp_path, monkeypatch
):
    parent = tmp_path / "safe" / "scripts"
    parent.mkdir(parents=True)
    target = parent / "status-view.json"
    status_store_module.acknowledge_run(target, 1)
    moved_parent = tmp_path / "moved-scripts"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    real_open = utils_module.os.open
    swapped = False

    def swap_parent_before_open(candidate, flags, *args, **kwargs):
        nonlocal swapped
        if (
            Path(candidate) == parent
            and flags & getattr(os, "O_DIRECTORY", 0)
            and not swapped
        ):
            swapped = True
            parent.rename(moved_parent)
            parent.symlink_to(attacker, target_is_directory=True)
        return real_open(candidate, flags, *args, **kwargs)

    monkeypatch.setattr(utils_module.os, "open", swap_parent_before_open)

    with pytest.raises((OSError, ValueError)):
        status_store_module.acknowledge_run(target, 2)

    assert json.loads((moved_parent / target.name).read_text()) == {
        "version": 1,
        "acknowledged_run_ids": [1],
    }
    assert not (attacker / target.name).exists()


@pytest.mark.parametrize("changed_identity_call", [3, 4, 5])
def test_atomic_observer_write_rejects_temp_or_destination_identity_swap(
    tmp_path, monkeypatch, changed_identity_call
):
    target = tmp_path / "status-view.json"
    status_store_module.acknowledge_run(target, 1)
    prior = target.read_bytes()
    real_same_identity = utils_module._same_file_identity
    calls = 0

    def changed_identity(first, second):
        nonlocal calls
        calls += 1
        if calls == changed_identity_call:
            return False
        return real_same_identity(first, second)

    monkeypatch.setattr(utils_module, "_same_file_identity", changed_identity)
    replacement = b'{"version":1,"acknowledged_run_ids":[1,2]}'

    with pytest.raises(ValueError, match="identity changed"):
        utils_module.atomic_write_private_file(
            target,
            replacement,
            max_bytes=status_store_module.MAX_OBSERVER_STATE_BYTES,
        )

    assert target.read_bytes() == prior


def test_atomic_observer_write_keeps_live_state_single_linked_and_readable(
    tmp_path, monkeypatch
):
    target = tmp_path / "status-view.json"
    status_store_module.acknowledge_run(target, 1)
    real_replace = utils_module.os.replace
    observations: list[tuple[int, frozenset[int]]] = []

    def inspect_live_state(source, destination, *args, **kwargs):
        if destination == target.name and kwargs.get("dst_dir_fd") is not None:
            info = os.stat(
                target.name,
                dir_fd=kwargs["dst_dir_fd"],
                follow_symlinks=False,
            )
            observed = status_store_module.load_observer_state(target)
            observations.append((info.st_nlink, observed.acknowledged_run_ids))
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(utils_module.os, "replace", inspect_live_state)

    status_store_module.acknowledge_run(target, 2)

    assert observations == [(1, frozenset({1}))]
    assert target.stat().st_nlink == 1
    assert status_store_module.load_observer_state(target).acknowledged_run_ids == {
        1,
        2,
    }


def test_interrupted_atomic_observer_write_never_links_or_corrupts_live_state(
    tmp_path, monkeypatch
):
    target = tmp_path / "status-view.json"
    status_store_module.acknowledge_run(target, 1)
    observed_links: list[int] = []

    def interrupt_before_replace(_source, destination, *args, **kwargs):
        if destination == target.name and kwargs.get("dst_dir_fd") is not None:
            info = os.stat(
                target.name,
                dir_fd=kwargs["dst_dir_fd"],
                follow_symlinks=False,
            )
            observed_links.append(info.st_nlink)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(utils_module.os, "replace", interrupt_before_replace)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        status_store_module.acknowledge_run(target, 2)

    assert observed_links == [1]
    assert target.stat().st_nlink == 1
    assert status_store_module.load_observer_state(target).acknowledged_run_ids == {
        1
    }


def test_acknowledge_run_supports_synthetic_legacy_run_identifiers(tmp_path):
    path = tmp_path / "status-view.json"

    state = status_store_module.acknowledge_run(path, -41)

    assert state.acknowledged_run_ids == frozenset({-41})
    assert status_store_module.load_observer_state(path) == state


@pytest.mark.parametrize("run_id", [0, True, "1"])
def test_acknowledge_run_rejects_invalid_identifiers(tmp_path, run_id):
    with pytest.raises(ValueError, match="nonzero integer"):
        status_store_module.acknowledge_run(tmp_path / "status-view.json", run_id)


def test_status_view_file_is_gitignored():
    ignore = (Path(__file__).resolve().parents[2] / ".gitignore").read_text()

    assert "scripts/status-state/" in ignore.splitlines()


def test_observer_state_path_uses_a_dedicated_private_runtime_directory(tmp_path):
    assert status_store_module.observer_state_path(tmp_path) == (
        tmp_path.resolve() / "scripts" / "status-state" / "status-view.json"
    )


@pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/var").is_symlink(),
    reason="macOS /var alias required",
)
def test_macos_var_alias_supports_private_state_and_custom_queue(tmp_path):
    actual_root = tmp_path.resolve()
    try:
        relative = actual_root.relative_to("/private/var")
    except ValueError:
        pytest.skip("temporary directory is not under /private/var")
    alias_root = Path("/var") / relative
    state_path = status_store_module.observer_state_path(alias_root)

    state = status_store_module.acknowledge_run(state_path, 7)

    assert state.acknowledged_run_ids == {7}
    assert status_store_module.load_observer_state(
        status_store_module.observer_state_path(actual_root)
    ) == state

    actual_queue = actual_root / "custom" / "jobs.sqlite3"
    with QueueRepository(actual_queue, sync_usage=False):
        pass
    alias_queue = Path("/var") / actual_queue.relative_to("/private/var")

    snapshot = status_store_module.read_snapshot(
        alias_queue,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState.empty(),
    )

    assert snapshot.active == ()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits required")
def test_observer_state_private_child_allows_normal_repository_scripts_acl(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    scripts.chmod(0o777)
    path = status_store_module.observer_state_path(tmp_path)

    state = status_store_module.acknowledge_run(path, 1)

    assert state.acknowledged_run_ids == {1}
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_concurrent_acknowledgments_do_not_lose_run_ids(tmp_path, monkeypatch):
    path = status_store_module.observer_state_path(tmp_path)
    original_load = status_store_module._load_observer_state_unlocked
    first_loaded = threading.Event()
    release_first = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def synchronized_load(candidate):
        nonlocal calls
        state = original_load(candidate)
        with calls_lock:
            calls += 1
            position = calls
        if position == 1:
            first_loaded.set()
            assert release_first.wait(timeout=5)
        return state

    monkeypatch.setattr(
        status_store_module,
        "_load_observer_state_unlocked",
        synchronized_load,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(status_store_module.acknowledge_run, path, 11)
        assert first_loaded.wait(timeout=5)
        second = executor.submit(status_store_module.acknowledge_run, path, 19)
        release_first.set()
        results = [first.result(timeout=5), second.result(timeout=5)]
    monkeypatch.setattr(
        status_store_module,
        "_load_observer_state_unlocked",
        original_load,
    )

    assert sorted(len(result.acknowledged_run_ids) for result in results) == [1, 2]
    assert set().union(*(result.acknowledged_run_ids for result in results)) == {
        11,
        19,
    }
    assert status_store_module.load_observer_state(path).acknowledged_run_ids == {
        11,
        19,
    }


def test_acknowledgments_are_deterministically_pruned_and_keep_current_id(tmp_path):
    path = status_store_module.observer_state_path(tmp_path)
    existing = list(range(1, status_store_module.MAX_ACKNOWLEDGED_RUN_IDS + 1))
    path.parent.mkdir(parents=True, mode=0o700)
    path.write_text(
        json.dumps({"version": 1, "acknowledged_run_ids": existing}),
        encoding="utf-8",
    )
    path.chmod(0o600)

    state = status_store_module.acknowledge_run(path, -50_000)

    assert len(state.acknowledged_run_ids) == (
        status_store_module.MAX_ACKNOWLEDGED_RUN_IDS
    )
    assert -50_000 in state.acknowledged_run_ids
    assert 1 not in state.acknowledged_run_ids
    assert path.stat().st_size <= status_store_module.MAX_OBSERVER_STATE_BYTES


@pytest.mark.parametrize(
    ("hour", "queue_active", "sessions", "daily_state", "reservation", "expected"),
    [
        (15, 0, 0, "uncompiled", None, "before_window"),
        (16, 2, 0, "uncompiled", None, "waiting_queue"),
        (16, 0, 2, "uncompiled", None, "waiting_sessions"),
        (16, 0, 0, "unreadable", None, "unavailable"),
        (16, 0, 0, "covered", None, "complete"),
        (16, 0, 0, "uncompiled", None, "ready"),
        (16, 0, 0, "uncompiled", "retry_wait", "retrying"),
    ],
)
def test_compile_readiness_uses_pure_injected_probes(
    hour, queue_active, sessions, daily_state, reservation, expected
):
    probes = status_store_module.CompileReadinessProbes(
        local_now=lambda: datetime(2026, 8, 18, hour, 0, tzinfo=UTC),
        session_count=lambda: sessions,
        daily_state=lambda: daily_state,
        reservation_state=lambda: reservation,
    )

    status = status_store_module.project_compile_status(
        compile_run=None,
        queue_active_count=queue_active,
        probes=probes,
    )

    assert status.state == expected
    assert status.ready is (expected == "ready")


def test_compile_status_prefers_an_active_authoritative_compile_run():
    running = _status_run(
        id=91,
        job_id=None,
        operation_key="compile:91",
        kind="compile",
        source_agent="system",
        session_id="2026-08-18",
        project="memory",
        state="running",
        phase="validation_started",
        summary="Validating staged changes",
        error=None,
        completed_at=None,
    )
    probes = status_store_module.CompileReadinessProbes(
        local_now=lambda: pytest.fail("compile run should avoid readiness probes"),
        session_count=lambda: pytest.fail("compile run should avoid readiness probes"),
        daily_state=lambda: pytest.fail("compile run should avoid readiness probes"),
        reservation_state=lambda: pytest.fail("compile run should avoid readiness probes"),
    )

    status = status_store_module.project_compile_status(
        compile_run=running,
        queue_active_count=0,
        probes=probes,
    )

    assert status.state == "running"
    assert status.summary == "Validating staged changes"
    assert status.run == running


@pytest.mark.parametrize("terminal_state", ["succeeded", "failed"])
def test_old_terminal_compile_run_does_not_suppress_current_readiness(terminal_state):
    terminal = _status_run(
        id=92,
        job_id=None,
        operation_key="compile:yesterday",
        kind="compile",
        source_agent="system",
        session_id="2026-08-17",
        project="memory",
        state=terminal_state,
        phase=terminal_state,
        summary="Yesterday's compile",
        error="Yesterday failed" if terminal_state == "failed" else None,
        started_at=READ_NOW - timedelta(days=1),
        updated_at=READ_NOW - timedelta(days=1),
        completed_at=READ_NOW - timedelta(days=1),
    )
    probes = status_store_module.CompileReadinessProbes(
        local_now=lambda: datetime.fromisoformat("2026-08-18T17:00:00-07:00"),
        session_count=lambda: 0,
        daily_state=lambda: "uncompiled",
        reservation_state=lambda: None,
    )

    status = status_store_module.project_compile_status(
        compile_run=terminal,
        queue_active_count=0,
        probes=probes,
    )

    assert status.state == "ready"
    assert status.ready is True
    assert status.run == terminal


def test_old_terminal_compile_run_does_not_suppress_queue_or_session_waits():
    terminal = _status_run(
        id=93,
        job_id=None,
        operation_key="compile:old",
        kind="compile",
        source_agent="system",
        session_id="2026-08-10",
        project="memory",
        state="succeeded",
        phase="succeeded",
        completed_at=READ_NOW - timedelta(days=8),
        started_at=READ_NOW - timedelta(days=8),
        updated_at=READ_NOW - timedelta(days=8),
    )
    probes = status_store_module.CompileReadinessProbes(
        local_now=lambda: datetime.fromisoformat("2026-08-18T17:00:00-07:00"),
        session_count=lambda: 2,
        daily_state=lambda: "uncompiled",
        reservation_state=lambda: None,
    )

    waiting_queue = status_store_module.project_compile_status(
        compile_run=terminal,
        queue_active_count=3,
        probes=probes,
    )
    waiting_sessions = status_store_module.project_compile_status(
        compile_run=terminal,
        queue_active_count=0,
        probes=probes,
    )

    assert waiting_queue.state == "waiting_queue"
    assert waiting_sessions.state == "waiting_sessions"


def test_snapshot_keeps_old_compile_metadata_outside_recent(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, sync_usage=False) as repository:
        _insert_projection_run(
            repository._connection,
            run_id=99,
            state="succeeded",
            phase="succeeded",
            updated_at=READ_NOW - timedelta(days=8),
            project="memory",
            summary="Updated 6 articles",
            kind="compile",
        )
        repository._connection.commit()
    monkeypatch.setattr(status_store_module, "_default_session_count", lambda: 0)
    monkeypatch.setattr(
        status_store_module,
        "_default_daily_state",
        lambda _path, _now: "uncompiled",
    )

    snapshot = status_store_module.read_snapshot(
        path,
        now=datetime.fromisoformat("2026-08-18T17:00:00-07:00"),
        observer_state=status_store_module.ObserverState.empty(),
    )

    assert snapshot.compile.state == "ready"
    assert snapshot.compile.run is not None
    assert snapshot.compile.run.id == 99
    assert snapshot.compile.run.summary == "Updated 6 articles"
    assert all(run.id != 99 for run in snapshot.recent)


def test_snapshot_reads_active_compile_reservation_without_queue_mutation(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, sync_usage=False) as repository:
        repository._connection.execute(
            """
            INSERT OR REPLACE INTO queue_metadata(key, value)
            VALUES (
                'auto_compile_reservation',
                '{"status":"retry_wait","expires_at":"2026-08-19T00:00:00+00:00"}'
            )
            """
        )
        repository._connection.commit()

    snapshot = status_store_module.read_snapshot(
        path,
        now=datetime(2026, 8, 18, 17, 0, tzinfo=UTC),
        observer_state=status_store_module.ObserverState.empty(),
    )

    assert snapshot.compile.state == "retrying"
    assert snapshot.compile.run is None


def test_snapshot_ignores_an_expired_compile_reservation(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, sync_usage=False) as repository:
        repository._connection.execute(
            """
            INSERT OR REPLACE INTO queue_metadata(key, value)
            VALUES (
                'auto_compile_reservation',
                '{"status":"retry_wait","expires_at":"2026-08-18T12:00:00+00:00"}'
            )
            """
        )
        repository._connection.commit()
    monkeypatch.setattr(status_store_module, "_default_session_count", lambda: 0)
    monkeypatch.setattr(
        status_store_module,
        "_default_daily_state",
        lambda _path, _now: "uncompiled",
    )

    snapshot = status_store_module.read_snapshot(
        path,
        now=datetime.fromisoformat("2026-08-18T17:00:00-07:00"),
        observer_state=status_store_module.ObserverState.empty(),
    )

    assert snapshot.compile.state == "ready"


def test_custom_queue_uses_explicit_memory_home_for_daily_readiness(
    tmp_path, monkeypatch
):
    memory_home = tmp_path / "memory"
    custom_queue = tmp_path / "runtime" / "custom.sqlite3"
    with QueueRepository(custom_queue, memory_home=memory_home, sync_usage=False):
        pass
    daily = memory_home / "daily"
    daily.mkdir(parents=True)
    daily.joinpath("2026-08-18.md").write_text("# Daily Log\n\nUncompiled session\n")
    monkeypatch.setattr(status_store_module, "_default_session_count", lambda: 0)
    now = datetime.fromisoformat("2026-08-18T17:00:00-07:00")

    without_root = status_store_module.read_snapshot(
        custom_queue,
        now=now,
        observer_state=status_store_module.ObserverState.empty(),
    )
    with_root = status_store_module.read_snapshot(
        custom_queue,
        now=now,
        observer_state=status_store_module.ObserverState.empty(),
        memory_home=memory_home,
    )

    assert without_root.compile.state == "unavailable"
    assert with_root.compile.state == "ready"


def test_snapshot_projection_queries_share_one_read_transaction(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, sync_usage=False):
        pass
    original_read_runs = status_store_module._read_runs

    def insert_reservation():
        writer = sqlite3.connect(path)
        try:
            writer.execute(
                """
                INSERT OR REPLACE INTO queue_metadata(key, value)
                VALUES (
                    'auto_compile_reservation',
                    '{"status":"retry_wait","expires_at":"2026-08-20T00:00:00+00:00"}'
                )
                """
            )
            writer.commit()
        finally:
            writer.close()

    def synchronized_read_runs(*args, **kwargs):
        runs = original_read_runs(*args, **kwargs)
        writer = threading.Thread(target=insert_reservation)
        writer.start()
        writer.join(timeout=5)
        assert not writer.is_alive()
        return runs

    monkeypatch.setattr(status_store_module, "_read_runs", synchronized_read_runs)

    snapshot = status_store_module.read_snapshot(
        path,
        now=datetime.fromisoformat("2026-08-18T15:00:00-07:00"),
        observer_state=status_store_module.ObserverState.empty(),
    )

    assert snapshot.compile.state == "before_window"


def test_run_details_queries_share_one_read_transaction(tmp_path, monkeypatch):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, clock=lambda: READ_NOW, sync_usage=False) as repository:
        result = repository.enqueue_capture(_version_2_session())
        run = repository.status_run_for_job(result.job_id)
    original_from_row = status_store_module.status_run_from_row
    inserted = False

    def add_event_and_attempt():
        writer = sqlite3.connect(path)
        try:
            writer.execute(
                """
                INSERT INTO status_events (
                    run_id, phase, level, details_json, created_at
                ) VALUES (?, 'codex_started', 'info', '{}', ?)
                """,
                (run.id, READ_NOW.isoformat()),
            )
            writer.execute(
                """
                INSERT INTO provider_attempts (
                    job_id, provider, model, task, started_at, ended_at,
                    outcome, elapsed_ms
                ) VALUES (?, 'codex', 'gpt-5.6-luna', 'extract', ?, ?, 'success', 1)
                """,
                (result.job_id, READ_NOW.isoformat(), READ_NOW.isoformat()),
            )
            writer.commit()
        finally:
            writer.close()

    def synchronized_from_row(*args, **kwargs):
        nonlocal inserted
        projected = original_from_row(*args, **kwargs)
        if not inserted:
            inserted = True
            writer = threading.Thread(target=add_event_and_attempt)
            writer.start()
            writer.join(timeout=5)
            assert not writer.is_alive()
        return projected

    monkeypatch.setattr(
        status_store_module,
        "status_run_from_row",
        synchronized_from_row,
    )

    details = status_store_module.read_run_details(path, run.id)

    assert [event.phase for event in details.events] == ["queued"]
    assert details.provider_attempts == ()
def test_bounded_snapshot_rejects_invalid_max_runs(tmp_path):
    for maximum in (0, -1, True, 10_001):
        with pytest.raises(ValueError, match="max_runs"):
            status_store_module.read_snapshot(
                tmp_path / "missing.sqlite3",
                now=READ_NOW,
                observer_state=status_store_module.ObserverState.empty(),
            max_runs=maximum,
        )


def test_bounded_legacy_ack_priority_keeps_older_unacknowledged_attention(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, sync_usage=False) as repository:
        for job_id in range(1, 5):
            timestamp = (READ_NOW - timedelta(minutes=job_id)).isoformat()
            repository._connection.execute(
                """INSERT INTO jobs(id,kind,source_agent,session_id,project,cwd,trigger,
                source_path,source_hash,payload_json,status,attempt_count,available_at,
                created_at,updated_at,completed_at) VALUES(?, 'capture','claude',?,
                'legacy','/tmp','end','/tmp/x',?,'{}','dead',1,?,?,?,?)""",
                (job_id, f"s-{job_id}", f"h-{job_id}", timestamp, timestamp, timestamp, timestamp),
            )
        repository._connection.commit()
    snapshot = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState(1, frozenset({-1, -2, -3})),
        max_runs=2,
    )
    assert -4 in {run.id for run in snapshot.attention}
    assert sum(map(len, (snapshot.active, snapshot.attention, snapshot.recent))) <= 2
    assert snapshot.has_more


def test_bounded_attention_reserves_modern_and_legacy_sources(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, sync_usage=False) as repository:
        for index in range(3):
            _insert_projection_run(
                repository._connection,
                run_id=100 + index,
                state="failed",
                phase="failed",
                updated_at=READ_NOW - timedelta(seconds=index),
                project="modern",
            )
            timestamp = (READ_NOW - timedelta(days=1, seconds=index)).isoformat()
            repository._connection.execute(
                """INSERT INTO jobs(id,kind,source_agent,session_id,project,cwd,trigger,
                source_path,source_hash,payload_json,status,attempt_count,available_at,
                created_at,updated_at,completed_at) VALUES(?, 'capture','claude',?,
                'legacy','/tmp','end','/tmp/x',?,'{}','dead',1,?,?,?,?)""",
                (index + 1, f"l-{index}", f"lh-{index}", timestamp, timestamp, timestamp, timestamp),
            )
        repository._connection.commit()
    snapshot = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState.empty(),
        max_runs=2,
    )
    assert {run.id > 0 for run in snapshot.attention} == {True, False}
    assert len(snapshot.attention) == 2
    assert snapshot.has_more


def test_bounded_candidate_queries_limit_projection_materialization(
    tmp_path, monkeypatch
):
    path = tmp_path / "jobs.sqlite3"
    with QueueRepository(path, sync_usage=False) as repository:
        for index in range(120):
            _insert_projection_run(
                repository._connection,
                run_id=20_000 + index,
                state="running" if index % 2 else "failed",
                phase="worker_claimed" if index % 2 else "failed",
                updated_at=READ_NOW - timedelta(seconds=index),
                project="bounded",
            )
        repository._connection.commit()
    materialized = 0
    statements: list[str] = []
    original_from_row = status_store_module.status_run_from_row
    original_open = status_store_module._open_read_only_database

    def count_row(row, **kwargs):
        nonlocal materialized
        materialized += 1
        return original_from_row(row, **kwargs)

    def traced_open(candidate):
        connection = original_open(candidate)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(status_store_module, "status_run_from_row", count_row)
    monkeypatch.setattr(status_store_module, "_open_read_only_database", traced_open)
    snapshot = status_store_module.read_snapshot(
        path,
        now=READ_NOW,
        observer_state=status_store_module.ObserverState.empty(),
        max_runs=2,
    )
    assert materialized <= 18
    assert len(snapshot.active) + len(snapshot.attention) + len(snapshot.recent) == 2
    assert snapshot.has_more
    assert sum(" LIMIT 3" in statement for statement in statements) >= 6
