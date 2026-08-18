from __future__ import annotations

import importlib
import importlib.util
import io
import json
import logging
import os
import sqlite3
import stat
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import hook_logging

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / "hooks"
NOW = datetime(2026, 8, 18, 18, 0, tzinfo=UTC)


def _load_hook(name: str):
    path = HOOKS / name
    spec = importlib.util.spec_from_file_location(f"health_{name.replace('-', '_')}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RecordHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _record_logger(name: str) -> tuple[logging.Logger, _RecordHandler]:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = _RecordHandler()
    logger.addHandler(handler)
    return logger, handler


@pytest.mark.parametrize(
    ("hook_name", "source_agent"),
    [("session-end.py", "claude"), ("codex-session-end.py", "codex")],
)
def test_malformed_hook_input_records_precise_error_event(
    monkeypatch, hook_name, source_agent
):
    hook = _load_hook(hook_name)
    logger, handler = _record_logger(f"malformed-{source_agent}")
    monkeypatch.setattr(hook, "_logger", lambda: logger)
    monkeypatch.setattr(sys, "stdin", io.StringIO("{"))

    hook.main(clock=lambda: 0.0)

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.levelno == logging.ERROR
    assert record.hook_event == "malformed_input"
    assert record.source_agent == source_agent
    assert record.session_id is None


@pytest.mark.parametrize(
    ("hook_name", "source_agent"),
    [("session-end.py", "claude"), ("codex-session-end.py", "codex")],
)
def test_missing_transcript_records_error_without_exposing_path(
    tmp_path, monkeypatch, hook_name, source_agent
):
    hook = _load_hook(hook_name)
    logger, handler = _record_logger(f"missing-{source_agent}")
    secret_path = tmp_path / "credential-private-transcript.jsonl"
    monkeypatch.setattr(hook, "_logger", lambda: logger)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "session_id": "session-missing",
                    "transcript_path": str(secret_path),
                }
            )
        ),
    )

    hook.main(clock=lambda: 0.0)

    record = handler.records[0]
    assert record.levelno == logging.ERROR
    assert record.hook_event == "transcript_missing"
    assert record.source_agent == source_agent
    assert record.session_id == "session-missing"
    assert str(secret_path) not in record.getMessage()


@pytest.mark.parametrize(
    ("hook_name", "source_agent"),
    [("session-end.py", "claude"), ("codex-session-end.py", "codex")],
)
def test_nonregular_transcript_records_unreadable_event(
    tmp_path, monkeypatch, hook_name, source_agent
):
    hook = _load_hook(hook_name)
    logger, handler = _record_logger(f"unreadable-{source_agent}")
    transcript_directory = tmp_path / "transcript-directory"
    transcript_directory.mkdir()
    monkeypatch.setattr(hook, "_logger", lambda: logger)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "session_id": "session-unreadable",
                    "transcript_path": str(transcript_directory),
                }
            )
        ),
    )

    hook.main(clock=lambda: 0.0)

    record = handler.records[0]
    assert record.levelno == logging.ERROR
    assert record.hook_event == "transcript_unreadable"
    assert record.session_id == "session-unreadable"


def test_pre_enqueue_failure_records_capture_failed_with_session(tmp_path, monkeypatch):
    hook = _load_hook("session-end.py")
    logger, handler = _record_logger("pre-enqueue-failure")
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    slice_path = tmp_path / "slice.jsonl"
    slice_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(hook, "_logger", lambda: logger)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({
        "session_id": "session-before-queue",
        "transcript_path": str(transcript),
        "cwd": str(tmp_path),
    })))

    @contextmanager
    def selected(*_args, **_kwargs):
        yield slice_path, SimpleNamespace(turns=(object(),))

    monkeypatch.setattr(hook, "bounded_transcript_slice", selected)
    monkeypatch.setattr(hook, "render_turns", lambda _preview: "bounded context")
    monkeypatch.setattr(hook, "_resolve_user_tty", lambda **_kwargs: None)
    monkeypatch.setattr(
        hook,
        "enqueue_capture_with_deadline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("before enqueue")),
    )

    hook.main(clock=lambda: 0.0)

    record = handler.records[-1]
    assert record.hook_event == "capture_failed"
    assert record.session_id == "session-before-queue"


def test_unavailable_queue_records_queue_unavailable(tmp_path, monkeypatch):
    hook = _load_hook("session-end.py")
    logger, handler = _record_logger("queue-unavailable")
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(hook, "_logger", lambda: logger)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({
        "session_id": "session-db",
        "transcript_path": str(transcript),
        "cwd": str(tmp_path),
    })))

    @contextmanager
    def selected(*_args, **_kwargs):
        yield transcript, SimpleNamespace(turns=(object(),))

    monkeypatch.setattr(hook, "bounded_transcript_slice", selected)
    monkeypatch.setattr(hook, "render_turns", lambda _preview: "context")
    monkeypatch.setattr(hook, "_resolve_user_tty", lambda **_kwargs: None)
    monkeypatch.setattr(
        hook,
        "enqueue_capture_with_deadline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is unavailable")
        ),
    )

    hook.main(clock=lambda: 0.0)

    assert handler.records[-1].hook_event == "queue_unavailable"


def test_structured_hook_log_redacts_bounds_and_stays_one_line(tmp_path, monkeypatch):
    secret = "credential-value-never-log"
    monkeypatch.setenv("AI_MEMORY_VIEW_TOKEN", secret)
    logger = hook_logging.configure_hook_logger("health-json", "session-end", tmp_path)
    try:
        hook_logging.log_hook_event(
            logger,
            logging.ERROR,
            "capture_failed",
            f"failure {secret}\nwith control\x00" + "x" * 2_000,
            source_agent="claude",
            session_id=f"session-{secret}",
        )
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    path = tmp_path / "scripts" / "logs" / "hooks.log"
    lines = path.read_bytes().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "capture_failed"
    assert record["source_agent"] == "claude"
    assert secret not in json.dumps(record)
    assert "\n" not in record["message"]
    assert "\x00" not in record["message"]
    assert len(record["message"]) <= 1_000
    assert "session_id" not in record
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def _health_module():
    return importlib.import_module("scripts.status_health")


def _write_hook_log(memory_home: Path, records: list[object]) -> Path:
    logs = memory_home / "scripts" / "logs"
    logs.mkdir(parents=True, mode=0o700)
    logs.chmod(0o700)
    path = logs / "hooks.log"
    path.write_text(
        "".join(
            (json.dumps(record, separators=(",", ":")) if not isinstance(record, str) else record)
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _error_record(
    *,
    timestamp: datetime = NOW,
    component: str = "session-end",
    event: str = "capture_failed",
    message: str = "capture failed",
    session_id: str | None = "session-1",
) -> dict[str, object]:
    record: dict[str, object] = {
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "level": "ERROR",
        "component": component,
        "event": event,
        "logger": "ai-memory-session-end",
        "message": message,
    }
    if session_id is not None:
        record["session_id"] = session_id
    return record


def test_recent_hook_alerts_filter_dedupe_redact_and_sort(tmp_path, monkeypatch):
    health = _health_module()
    secret = "credential-value-never-display"
    monkeypatch.setenv("AI_MEMORY_VIEW_TOKEN", secret)
    duplicate = _error_record(message=f"failed {secret}\nnext")
    _write_hook_log(
        tmp_path,
        [
            duplicate,
            duplicate,
            _error_record(session_id="session-2"),
            _error_record(timestamp=NOW - timedelta(days=2), message="old"),
            _error_record(component="unknown-hook", message="unknown"),
            {**_error_record(message="warning"), "level": "WARNING"},
            "not-json",
        ],
    )

    alerts = health.read_recent_hook_alerts(tmp_path, now=NOW)

    assert len(alerts) == 2
    assert all(alert.level == "error" for alert in alerts)
    assert all(alert.component == "session-end" for alert in alerts)
    assert secret not in " ".join(alert.message for alert in alerts)
    assert all("\n" not in alert.message for alert in alerts)


def test_recent_hook_alerts_reads_bounded_tail_and_discards_partial_first_line(tmp_path):
    health = _health_module()
    valid = _error_record(message="tail failure")
    path = _write_hook_log(tmp_path, ["x" * 300, valid])

    alerts = health.read_recent_hook_alerts(tmp_path, now=NOW, max_bytes=220)

    assert [alert.message for alert in alerts] == ["tail failure"]
    assert path.stat().st_size > 220


@pytest.mark.parametrize("attack", ["missing", "symlink", "hardlink", "public"])
def test_recent_hook_alerts_unsafe_or_missing_log_degrades_empty(
    tmp_path, attack
):
    health = _health_module()
    if attack == "missing":
        path = tmp_path / "scripts" / "logs" / "hooks.log"
    else:
        path = _write_hook_log(tmp_path, [_error_record()])
        if attack == "symlink":
            external = tmp_path / "external.log"
            path.rename(external)
            path.symlink_to(external)
        elif attack == "hardlink":
            os.link(path, tmp_path / "second-link.log")
        else:
            path.chmod(0o644)

    assert health.read_recent_hook_alerts(tmp_path, now=NOW) == ()
    if attack == "missing":
        assert not path.exists()


def test_recent_hook_alerts_rejects_naive_now(tmp_path):
    health = _health_module()

    with pytest.raises(ValueError, match="timezone-aware"):
        health.read_recent_hook_alerts(
            tmp_path,
            now=NOW.replace(tzinfo=None),
        )
