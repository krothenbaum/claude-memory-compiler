from __future__ import annotations

import sqlite3
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

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

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)


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
