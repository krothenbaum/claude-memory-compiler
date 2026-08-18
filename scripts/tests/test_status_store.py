from __future__ import annotations

import sqlite3
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from scripts.queue import QueueRepository
from scripts.status_store import (
    ALLOWED_PHASES,
    StatusEvent,
    StatusRun,
    normalize_details,
    normalize_status_reason,
    normalize_summary,
)

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


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


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


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

    with QueueRepository(path, sync_usage=False) as repository:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
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
    )
    source_details["chars_saved"] = 999

    with pytest.raises(FrozenInstanceError):
        run.phase = "succeeded"
    with pytest.raises(TypeError):
        event.details["chars_saved"] = 999
    assert event.details == {"chars_saved": 120}


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


def test_summaries_are_bounded_and_normalized_without_fabricating_text():
    assert normalize_summary("  Saved\n  42 characters  ") == "Saved 42 characters"
    assert normalize_summary("x" * 1_001) == "x" * 1_000
    assert normalize_summary(" \n ") is None
    assert normalize_summary(None) is None


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
        "retry_at": "2026-08-18T12:05:00+00:00",
        "elapsed_ms": 1250,
    }
    with pytest.raises(TypeError):
        details["chars_saved"] = 43


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
