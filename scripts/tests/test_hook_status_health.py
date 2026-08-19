from __future__ import annotations

import importlib
import importlib.util
import io
import json
import logging
import os
import sqlite3
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import capture as capture_module
from scripts import hook_logging
from scripts.queue import QueueRepository

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


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _diagnostic_runs(memory_home: Path) -> list[sqlite3.Row]:
    path = memory_home / "scripts" / "jobs.sqlite3"
    if not path.exists():
        return []
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            "SELECT * FROM status_runs ORDER BY id"
        ).fetchall()
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("case", "expected_event"),
    [
        ("malformed", "malformed_input"),
        ("missing", "transcript_missing"),
        ("unreadable", "transcript_unreadable"),
    ],
)
def test_precompact_structured_input_errors_are_visible_to_health_reader(
    tmp_path, monkeypatch, case, expected_event
):
    hook = _load_hook("pre-compact.py")
    memory_home = tmp_path / "memory"
    monkeypatch.setenv("AI_MEMORY_HOME", str(memory_home))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    if case == "malformed":
        payload = "{"
    elif case == "missing":
        payload = json.dumps(
            {
                "session_id": "precompact-session",
                "cwd": str(tmp_path / "cwd-precompact-project"),
            }
        )
    else:
        transcript = tmp_path / "transcript-directory"
        transcript.mkdir()
        payload = json.dumps(
            {
                "session_id": "precompact-session",
                "transcript_path": str(transcript),
            }
        )
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    logger = hook._logger()
    try:
        hook.main(clock=lambda: 0.0)
    finally:
        _close_logger(logger)

    records = [
        json.loads(line)
        for line in (memory_home / "scripts" / "logs" / "hooks.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[-1]["event"] == expected_event
    alerts = _health_module().read_recent_hook_alerts(
        memory_home,
        now=datetime.now(UTC),
    )
    assert alerts == ()
    runs = _diagnostic_runs(memory_home)
    assert len(runs) == 1
    assert runs[0]["state"] == "failed"
    assert runs[0]["phase"] == "failed"
    if case == "missing":
        assert runs[0]["project"] == "cwd-precompact-project"


@pytest.mark.parametrize(
    ("result", "expected_event", "is_error"),
    [
        (ValueError("capture problem"), "capture_failed", False),
        (
            capture_module.CaptureQueueUnavailableError("queue denied"),
            "queue_unavailable",
            True,
        ),
        ({"status": "created", "job_id": 4}, "capture_succeeded", False),
    ],
)
def test_precompact_structured_capture_outcomes_are_visible(
    tmp_path, monkeypatch, result, expected_event, is_error
):
    hook = _load_hook("pre-compact.py")
    memory_home = tmp_path / "memory"
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("AI_MEMORY_HOME", str(memory_home))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "session_id": "precompact-capture",
                    "transcript_path": str(transcript),
                    "cwd": str(tmp_path),
                }
            )
        ),
    )

    @contextmanager
    def selected(*_args, **_kwargs):
        yield transcript, SimpleNamespace(turns=(object(),) * 5)

    def enqueue(*_args, **_kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    real_helpers = hook._live_capture_helpers()
    helpers = SimpleNamespace(
        bounded_transcript_slice=selected,
        require_time_remaining=lambda *_args, **_kwargs: None,
        enqueue_capture_with_deadline=enqueue,
        persist_hook_diagnostic_with_deadline=(
            real_helpers.persist_hook_diagnostic_with_deadline
        ),
        MIN_CAPTURE_REMAINING_SECONDS=0.75,
    )
    monkeypatch.setattr(hook, "_live_capture_helpers", lambda: helpers)
    monkeypatch.setattr(hook, "render_turns", lambda _preview: "context")
    logger = hook._logger()
    try:
        hook.main(clock=lambda: 0.0)
    finally:
        _close_logger(logger)

    records = [
        json.loads(line)
        for line in (memory_home / "scripts" / "logs" / "hooks.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[-1]["event"] == expected_event
    alerts = _health_module().read_recent_hook_alerts(
        memory_home,
        now=datetime.now(UTC),
    )
    assert bool(alerts) is is_error
    runs = _diagnostic_runs(memory_home)
    assert len(runs) == int(expected_event == "capture_failed")


@pytest.mark.parametrize(
    ("hook_name", "source_agent"),
    [("session-end.py", "claude"), ("codex-session-end.py", "codex")],
)
def test_malformed_hook_input_records_precise_error_event(
    tmp_path, monkeypatch, hook_name, source_agent
):
    hook = _load_hook(hook_name)
    logger, handler = _record_logger(f"malformed-{source_agent}")
    monkeypatch.setattr(hook, "_logger", lambda: logger)
    monkeypatch.setenv("AI_MEMORY_HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    monkeypatch.setattr(sys, "stdin", io.StringIO("{"))

    hook.main(clock=lambda: 0.0)

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.levelno == logging.ERROR
    assert record.hook_event == "malformed_input"
    assert record.source_agent == source_agent
    assert record.session_id is None
    runs = _diagnostic_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["source_agent"] == source_agent


@pytest.mark.parametrize(
    "hook_name",
    ["session-end.py", "codex-session-end.py", "pre-compact.py"],
)
@pytest.mark.parametrize(
    "configured_queue",
    [
        "",
        "relative/private-queue.sqlite3",
        "~definitely-no-such-user/jobs.sqlite3",
    ],
)
def test_hook_quarantines_invalid_queue_override_before_scripts_import(
    tmp_path, hook_name, configured_queue
):
    memory_home = tmp_path / "memory"
    environment = os.environ.copy()
    environment.pop("CLAUDE_MEMORY_HOME", None)
    environment.update(
        {
            "AI_MEMORY_HOME": str(memory_home),
            "AI_MEMORY_QUEUE_PATH": configured_queue,
        }
    )
    result = subprocess.run(
        [sys.executable, str(HOOKS / hook_name)],
        input=json.dumps(
            {
                "session_id": "invalid-queue-session",
                "transcript_path": "private-transcript-path",
            }
        ),
        text=True,
        capture_output=True,
        timeout=3,
        env=environment,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    records = [
        json.loads(line)
        for line in (memory_home / "scripts" / "logs" / "hooks.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 1
    assert records[0] == {
        **records[0],
        "level": "ERROR",
        "event": "queue_unavailable",
        "session_id": "invalid-queue-session",
        "message": "configured queue path is invalid",
    }
    assert configured_queue not in json.dumps(records[0]) or configured_queue == ""
    assert "private-transcript-path" not in json.dumps(records[0])
    assert not (memory_home / "scripts" / "jobs.sqlite3").exists()


@pytest.mark.parametrize(
    "hook_name",
    ["session-end.py", "codex-session-end.py", "pre-compact.py"],
)
def test_internal_job_guard_precedes_invalid_queue_quarantine(tmp_path, hook_name):
    memory_home = tmp_path / "must-not-exist"
    environment = os.environ.copy()
    environment.pop("CLAUDE_MEMORY_HOME", None)
    environment.update(
        {
            "AI_MEMORY_HOME": str(memory_home),
            "AI_MEMORY_QUEUE_PATH": "~definitely-no-such-user/jobs.sqlite3",
            "AI_MEMORY_INTERNAL_JOB": "1",
        }
    )

    result = subprocess.run(
        [sys.executable, str(HOOKS / hook_name)],
        input="{}",
        text=True,
        capture_output=True,
        timeout=3,
        env=environment,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert not memory_home.exists()


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
    monkeypatch.setenv("AI_MEMORY_HOME", str(tmp_path / "memory"))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                    {
                        "session_id": "session-missing",
                        "transcript_path": str(secret_path),
                        "cwd": str(tmp_path / "cwd-hook-project"),
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
    runs = _diagnostic_runs(tmp_path / "memory")
    assert len(runs) == 1
    assert runs[0]["project"] == "cwd-hook-project"


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
    monkeypatch.setenv("AI_MEMORY_HOME", str(tmp_path / "memory"))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
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
    assert len(_diagnostic_runs(tmp_path / "memory")) == 1


def test_pre_enqueue_failure_records_capture_failed_with_session(tmp_path, monkeypatch):
    hook = _load_hook("session-end.py")
    logger, handler = _record_logger("pre-enqueue-failure")
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    slice_path = tmp_path / "slice.jsonl"
    slice_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(hook, "_logger", lambda: logger)
    monkeypatch.setenv("AI_MEMORY_HOME", str(tmp_path / "memory"))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
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
    assert len(_diagnostic_runs(tmp_path / "memory")) == 1


def test_codex_capture_failure_records_diagnostic(tmp_path, monkeypatch):
    hook = _load_hook("codex-session-end.py")
    logger, handler = _record_logger("codex-capture-diagnostic")
    memory_home = tmp_path / "memory"
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("AI_MEMORY_HOME", str(memory_home))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    monkeypatch.setattr(hook, "_logger", lambda: logger)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "session_id": "codex-capture",
                    "transcript_path": str(transcript),
                    "project": "memory",
                }
            )
        ),
    )

    @contextmanager
    def selected(*_args, **_kwargs):
        yield transcript, SimpleNamespace(turns=(object(),))

    helpers = SimpleNamespace(
        bounded_transcript_slice=selected,
        require_time_remaining=lambda *_args, **_kwargs: None,
        enqueue_capture_with_deadline=lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(ValueError("capture failed")),
        persist_hook_diagnostic_with_deadline=(
            _load_hook("session-end.py").persist_hook_diagnostic_with_deadline
        ),
        MIN_CAPTURE_REMAINING_SECONDS=0.75,
    )
    monkeypatch.setattr(hook, "_live_capture_helpers", lambda: helpers)

    hook.main(clock=lambda: 0.0)

    assert handler.records[-1].hook_event == "capture_failed"
    runs = _diagnostic_runs(memory_home)
    assert len(runs) == 1
    assert runs[0]["source_agent"] == "codex"


def test_unavailable_queue_records_queue_unavailable(tmp_path, monkeypatch):
    hook = _load_hook("session-end.py")
    logger, handler = _record_logger("queue-unavailable")
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(hook, "_logger", lambda: logger)
    monkeypatch.setenv("AI_MEMORY_HOME", str(tmp_path / "memory"))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
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
            capture_module.CaptureQueueUnavailableError("database is unavailable")
        ),
    )

    hook.main(clock=lambda: 0.0)

    assert handler.records[-1].hook_event == "queue_unavailable"
    assert _diagnostic_runs(tmp_path / "memory") == []


@pytest.mark.parametrize(
    ("hook_name", "fixture_name", "source_agent"),
    [
        ("session-end.py", "claude-basic.jsonl", "claude"),
        ("codex-session-end.py", "codex-basic.jsonl", "codex"),
    ],
)
@pytest.mark.parametrize(
    "queue_attack",
    [
        "directory",
        "symlink",
        "hardlink",
        "newer_schema",
        pytest.param(
            "unsafe_mode",
            marks=pytest.mark.skipif(os.name == "nt", reason="POSIX mode required"),
        ),
        pytest.param(
            "permission",
            marks=pytest.mark.skipif(os.name == "nt", reason="POSIX mode required"),
        ),
    ],
)
def test_real_capture_child_classifies_unsafe_queue_boundary(
    tmp_path, hook_name, fixture_name, source_agent, queue_attack
):
    memory_home = tmp_path / "memory"
    queue_parent_to_restore: Path | None = None
    if queue_attack == "directory":
        invalid_queue = tmp_path / "queue-is-a-directory"
        invalid_queue.mkdir()
    elif queue_attack == "symlink":
        target = tmp_path / "queue-target.sqlite3"
        target.write_bytes(b"")
        target.chmod(0o600)
        invalid_queue = tmp_path / "queue-symlink.sqlite3"
        invalid_queue.symlink_to(target)
    elif queue_attack == "hardlink":
        target = tmp_path / "queue-target.sqlite3"
        target.write_bytes(b"")
        target.chmod(0o600)
        invalid_queue = tmp_path / "queue-hardlink.sqlite3"
        os.link(target, invalid_queue)
    elif queue_attack == "unsafe_mode":
        invalid_queue = tmp_path / "queue-public.sqlite3"
        invalid_queue.write_bytes(b"")
        invalid_queue.chmod(0o644)
    elif queue_attack == "newer_schema":
        invalid_queue = tmp_path / "queue-newer.sqlite3"
        connection = sqlite3.connect(invalid_queue)
        connection.execute("PRAGMA user_version = 999")
        connection.close()
        invalid_queue.chmod(0o600)
    else:
        queue_parent_to_restore = tmp_path / "read-only-queue-parent"
        queue_parent_to_restore.mkdir()
        queue_parent_to_restore.chmod(0o500)
        invalid_queue = queue_parent_to_restore / "jobs.sqlite3"
    transcript = (
        Path(__file__).resolve().parent / "fixtures" / "transcripts" / fixture_name
    )
    environment = os.environ.copy()
    environment.pop("CLAUDE_MEMORY_HOME", None)
    environment.update(
        {
            "AI_MEMORY_HOME": str(memory_home),
            "AI_MEMORY_QUEUE_PATH": str(invalid_queue),
        }
    )
    try:
        result = subprocess.run(
            [sys.executable, str(HOOKS / hook_name)],
            input=json.dumps(
                {
                    "session_id": f"real-child-{source_agent}",
                    "transcript_path": str(transcript),
                    "cwd": str(tmp_path),
                }
            ),
            text=True,
            capture_output=True,
            timeout=3,
            env=environment,
            check=False,
        )
    finally:
        if queue_parent_to_restore is not None:
            queue_parent_to_restore.chmod(0o700)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    records = [
        json.loads(line)
        for line in (memory_home / "scripts" / "logs" / "hooks.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[-1] == {
        **records[-1],
        "level": "ERROR",
        "event": "queue_unavailable",
        "source_agent": source_agent,
        "session_id": f"real-child-{source_agent}",
        "message": "queue unavailable during capture",
    }
    assert str(transcript) not in json.dumps(records[-1])
    assert _diagnostic_runs(memory_home) == []


def test_structured_hook_log_redacts_bounds_and_stays_one_line(tmp_path, monkeypatch):
    secret = "credential-value-never-log"
    occurrence_id = "e" * 32
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
            occurrence_id=occurrence_id,
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
    assert record["occurrence_id"] == occurrence_id
    assert secret not in json.dumps(record)
    assert "\n" not in record["message"]
    assert "\x00" not in record["message"]
    assert len(record["message"]) <= 1_000
    assert "session_id" not in record
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("failure", ["format", "write"])
def test_hook_log_handler_failure_never_writes_record_to_stdio(
    tmp_path, monkeypatch, capsys, failure
):
    stream = (tmp_path / "hook.log").open("a", encoding="utf-8")
    handler = hook_logging._HookLogHandler(stream)
    handler.setFormatter(hook_logging.HookJsonFormatter("session-end"))
    record = logging.LogRecord(
        "private-hook",
        logging.ERROR,
        __file__,
        1,
        "secret /private/tmp/transcript.jsonl",
        (),
        None,
    )
    if failure == "format":
        monkeypatch.setattr(
            handler,
            "format",
            lambda _record: (_ for _ in ()).throw(ValueError("format failed")),
        )
    else:
        monkeypatch.setattr(
            hook_logging.os,
            "write",
            lambda *_args: (_ for _ in ()).throw(OSError("disk failed")),
        )
    try:
        handler.emit(record)
    finally:
        handler.close()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_hook_logger_retries_transient_secure_open_once(tmp_path, monkeypatch):
    logger_name = "transient-hook-open"
    logger = logging.getLogger(logger_name)
    _close_logger(logger)
    real_open = hook_logging.open_secure_log_stream
    attempts = 0

    def transient_open(path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("concurrent first-file creation")
        return real_open(path)

    monkeypatch.setattr(hook_logging, "open_secure_log_stream", transient_open)
    try:
        configured = hook_logging.configure_hook_logger(
            logger_name,
            "session-end",
            tmp_path,
        )
        configured.info("retry succeeded")
    finally:
        _close_logger(logger)

    assert attempts == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "scripts" / "logs" / "hooks.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["message"] for record in records] == ["retry succeeded"]


@pytest.mark.parametrize(
    ("error", "expected_event"),
    [
        (ValueError("database appears in unrelated content"), "capture_failed"),
        (PermissionError("unrelated permission denied"), "capture_failed"),
        (
            capture_module.CaptureQueueUnavailableError(
                "custom queue permission denied"
            ),
            "queue_unavailable",
        ),
    ],
)
def test_capture_child_uses_typed_error_classification(
    monkeypatch, error, expected_event
):
    hook = _load_hook("session-end.py")
    monkeypatch.setattr(
        hook,
        "enqueue_hook_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "hook_input": {"transcript_path": "private"},
                    "source_agent": "claude",
                    "trigger": "session_end",
                    "limits": {},
                    "capture_token": "token",
                    "budget_seconds": 1.0,
                }
            )
        ),
    )
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    hook._capture_child_main()

    envelope = json.loads(output.getvalue())
    assert envelope["event"] == expected_event
    assert "database appears" not in output.getvalue()


def test_legacy_hook_log_sanitizes_credentials_paths_and_controls(
    tmp_path, monkeypatch
):
    secret = "credential-value-never-log"
    monkeypatch.setenv("AI_MEMORY_VIEW_TOKEN", secret)
    logger = hook_logging.configure_hook_logger("legacy-health", "pre-compact", tmp_path)
    try:
        logger.error(
            "RuntimeError at /private/tmp/transcript.jsonl and "
            r"C:\Users\name\session.jsonl "
            f"with {secret}\ncontrol\x00"
        )
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    record = json.loads(
        (tmp_path / "scripts" / "logs" / "hooks.log").read_text(encoding="utf-8")
    )
    serialized = json.dumps(record)
    assert secret not in serialized
    assert "/private/tmp/transcript.jsonl" not in serialized
    assert r"C:\Users\name\session.jsonl" not in serialized
    assert "\n" not in record["message"]
    assert "\x00" not in record["message"]
    assert "RuntimeError" in record["message"]
    assert len(record["message"]) <= 1_000


@pytest.mark.parametrize(
    ("message", "forbidden"),
    [
        (
            'RuntimeError opening "/Users/alice/My Project/secret transcript.jsonl": denied',
            "secret transcript.jsonl",
        ),
        (
            "RuntimeError opening /Users/alice/My Project/secret transcript.jsonl, denied",
            "My Project",
        ),
        (
            r"RuntimeError opening 'C:\Users\alice\My Project\secret transcript.jsonl': denied",
            "secret transcript.jsonl",
        ),
        (
            r"RuntimeError opening C:\Users\alice\My Project\secret transcript.jsonl; denied",
            "My Project",
        ),
        (
            r"RuntimeError opening \\server\private share\secret-transcript.jsonl",
            "server",
        ),
    ],
)
def test_legacy_hook_log_scrubs_spaced_and_unc_absolute_paths(
    tmp_path, message, forbidden
):
    logger = hook_logging.configure_hook_logger(
        f"legacy-path-{abs(hash(message))}",
        "pre-compact",
        tmp_path,
    )
    try:
        logger.error("%s", message)
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    record = json.loads(
        (tmp_path / "scripts" / "logs" / "hooks.log").read_text(encoding="utf-8")
    )
    assert record["message"].startswith("RuntimeError")
    assert "[PATH]" in record["message"]
    assert forbidden not in record["message"]
    assert "\n" not in record["message"]
    assert len(record["message"]) <= 1_000


@pytest.mark.parametrize(
    "secret",
    ["line\nsecret", "tab\tsecret", "repeated   space secret"],
)
def test_legacy_hook_log_redacts_whitespace_normalized_credentials(
    tmp_path, monkeypatch, secret
):
    monkeypatch.setenv("AI_MEMORY_VIEW_TOKEN", secret)
    logger = hook_logging.configure_hook_logger(
        f"legacy-secret-{abs(hash(secret))}",
        "pre-compact",
        tmp_path,
    )
    try:
        logger.error("RuntimeError credential=%s", secret)
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    record = json.loads(
        (tmp_path / "scripts" / "logs" / "hooks.log").read_text(encoding="utf-8")
    )
    canonical_secret = " ".join(secret.split())
    assert canonical_secret not in record["message"]
    assert "[REDACTED]" in record["message"]
    assert "\n" not in record["message"]
    assert "\t" not in record["message"]


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
    occurrence_id: str | None = None,
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
    if occurrence_id is not None:
        record["occurrence_id"] = occurrence_id
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
            _error_record(message=duplicate["message"], session_id="session-2"),
            _error_record(timestamp=NOW - timedelta(days=2), message="old"),
            _error_record(component="unknown-hook", message="unknown"),
            {**_error_record(message="warning"), "level": "WARNING"},
            "not-json",
        ],
    )

    alerts = health.read_recent_hook_alerts(tmp_path, now=NOW)

    assert len(alerts) == 1
    assert all(alert.level == "error" for alert in alerts)
    assert all(alert.component == "session-end" for alert in alerts)
    assert secret not in " ".join(alert.message for alert in alerts)
    assert all("\n" not in alert.message for alert in alerts)


def test_recent_hook_alerts_caps_results_deterministically(tmp_path):
    health = _health_module()
    records = [
        _error_record(message=f"failure-{index:03d}", session_id=f"session-{index}")
        for index in range(105)
    ]
    _write_hook_log(tmp_path, records)

    first = health.read_recent_hook_alerts(tmp_path, now=NOW)
    second = health.read_recent_hook_alerts(tmp_path, now=NOW)

    assert len(first) == 100
    assert first == second
    assert first[0].message == "failure-104"


def test_recent_hook_alerts_computes_redaction_environment_once(
    tmp_path, monkeypatch
):
    health = _health_module()
    _write_hook_log(
        tmp_path,
        [_error_record(message="first"), _error_record(message="second")],
    )
    original = health._canonical_redaction_env
    calls = 0

    def counted_environment():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(health, "_canonical_redaction_env", counted_environment)

    assert len(health.read_recent_hook_alerts(tmp_path, now=NOW)) == 2
    assert calls == 1


def test_recent_hook_alerts_reads_bounded_tail_and_discards_partial_first_line(tmp_path):
    health = _health_module()
    valid = _error_record(message="tail failure")
    path = _write_hook_log(tmp_path, ["x" * 300, valid])

    alerts = health.read_recent_hook_alerts(tmp_path, now=NOW, max_bytes=220)

    assert [alert.message for alert in alerts] == ["tail failure"]
    assert path.stat().st_size > 220


def test_recent_hook_alerts_discards_unterminated_final_record(tmp_path):
    health = _health_module()
    path = _write_hook_log(tmp_path, [])
    encoded = json.dumps(_error_record(message="torn failure")).encode("utf-8")
    path.write_bytes(encoded)
    path.chmod(0o600)

    assert health.read_recent_hook_alerts(tmp_path, now=NOW) == ()

    path.write_bytes(encoded + b"\n")
    path.chmod(0o600)
    assert [
        alert.message for alert in health.read_recent_hook_alerts(tmp_path, now=NOW)
    ] == ["torn failure"]


@pytest.mark.parametrize(
    "malformed",
    [
        '{"value":' + "9" * 5_000 + "}",
        "[" * 2_000 + "0" + "]" * 2_000,
    ],
)
def test_recent_hook_alerts_ignores_pathological_json(tmp_path, malformed):
    health = _health_module()
    _write_hook_log(tmp_path, [malformed])

    assert health.read_recent_hook_alerts(tmp_path, now=NOW) == ()


@pytest.mark.parametrize(
    "timestamp",
    ["0001-01-01T00:00:00+23:59", "9999-12-31T23:59:59-23:59"],
)
def test_recent_hook_alerts_ignores_timestamp_utc_overflow(tmp_path, timestamp):
    health = _health_module()
    record = _error_record()
    record["timestamp"] = timestamp
    _write_hook_log(tmp_path, [record])

    assert health.read_recent_hook_alerts(tmp_path, now=NOW) == ()


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


def test_record_hook_diagnostic_persists_safe_terminal_run(tmp_path, monkeypatch):
    health = _health_module()
    secret = "credential-value-never-persist"
    monkeypatch.setenv("AI_MEMORY_VIEW_TOKEN", secret)

    recorded = health.record_hook_diagnostic(
        tmp_path,
        event="malformed_input",
        source_agent="claude",
        session_id=f"session-{secret}\nprivate",
        project="",
        message=f"Malformed {secret}\ninput",
        deadline=10.0,
        clock=lambda: 0.0,
        token_factory=lambda: "raw-private-token",
    )

    assert recorded is True
    with QueueRepository(
        tmp_path / "scripts" / "jobs.sqlite3",
        sync_usage=False,
    ) as repository:
        rows = repository._connection.execute("SELECT * FROM status_runs").fetchall()
        assert len(rows) == 1
        run = repository.status_run_for_operation(rows[0]["operation_key"])
        assert run is not None
        assert run.state == "failed"
        assert run.source_agent == "claude"
        assert run.session_id == "unknown"
        assert run.project == "unknown"
        assert secret not in (run.error or "")
        assert "raw-private-token" not in (run.operation_key or "")
        assert len(repository.status_events(run.id)) == 1
        assert repository._connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


def test_diagnostic_child_is_killed_by_parent_deadline_without_partial_run(tmp_path):
    hook = _load_hook("session-end.py")
    queue_path = tmp_path / "scripts" / "jobs.sqlite3"
    with QueueRepository(queue_path, sync_usage=False):
        pass
    code = (
        "import sqlite3,sys,time; "
        "db=sqlite3.connect(sys.argv[1], isolation_level=None); "
        "db.execute('BEGIN IMMEDIATE'); "
        "db.execute(\"INSERT INTO status_runs "
        "(operation_key,kind,source_agent,session_id,project,state,phase,"
        "started_at,updated_at,completed_at) VALUES "
        "('hook-diagnostic:capture_failed:" + "a" * 32 + "','capture','claude',"
        "'deadline-session','memory','failed','failed',"
        "'2026-08-18T00:00:00+00:00','2026-08-18T00:00:00+00:00',"
        "'2026-08-18T00:00:00+00:00')\"); "
        "time.sleep(2)"
    )
    started = time.monotonic()

    recorded = hook.persist_hook_diagnostic_with_deadline(
        tmp_path,
        event="capture_failed",
        source_agent="claude",
        session_id="deadline-session",
        project="memory",
        message="capture failed",
        occurrence_id="a" * 32,
        deadline=started + 0.3,
        clock=time.monotonic,
        command=[sys.executable, "-c", code, str(queue_path)],
    )

    assert recorded is False
    assert time.monotonic() - started < 0.7
    with QueueRepository(queue_path, sync_usage=False) as repository:
        assert (
            repository.status_run_for_operation(
                "hook-diagnostic:capture_failed:" + "a" * 32
            )
            is None
        )


def test_normal_diagnostic_child_persists_matching_occurrence(tmp_path):
    hook = _load_hook("session-end.py")
    occurrence_id = "b" * 32

    assert hook.persist_hook_diagnostic_with_deadline(
        tmp_path,
        event="capture_failed",
        source_agent="claude",
        session_id="child-session",
        project="memory",
        message="capture failed",
        occurrence_id=occurrence_id,
        deadline=time.monotonic() + 2,
        clock=time.monotonic,
    )

    with QueueRepository(
        tmp_path / "scripts" / "jobs.sqlite3",
        sync_usage=False,
    ) as repository:
        run = repository.status_run_for_operation(
            f"hook-diagnostic:capture_failed:{occurrence_id}"
        )
        assert run is not None
        assert run.state == "failed"


def test_health_alert_suppressed_only_when_durable_occurrence_exists(tmp_path):
    health = _health_module()
    occurrence_id = "c" * 32
    record = _error_record(
        message="capture failed",
        occurrence_id=occurrence_id,
    )
    _write_hook_log(tmp_path, [record])

    assert len(health.read_recent_hook_alerts(tmp_path, now=NOW)) == 1
    assert not (tmp_path / "scripts" / "jobs.sqlite3").exists()

    assert health.record_hook_diagnostic(
        tmp_path,
        event="capture_failed",
        source_agent="claude",
        session_id="session-1",
        project="memory",
        message="capture failed",
        occurrence_id=occurrence_id,
    )

    assert health.read_recent_hook_alerts(tmp_path, now=NOW) == ()


def test_health_suppression_is_per_occurrence_before_display_dedup(tmp_path):
    health = _health_module()
    first_occurrence = "1" * 32
    second_occurrence = "2" * 32
    _write_hook_log(
        tmp_path,
        [
            _error_record(message="same failure", occurrence_id=first_occurrence),
            _error_record(message="same failure", occurrence_id=second_occurrence),
        ],
    )

    assert len(health.read_recent_hook_alerts(tmp_path, now=NOW)) == 1
    assert health.record_hook_diagnostic(
        tmp_path,
        event="capture_failed",
        source_agent="claude",
        session_id="first",
        project="memory",
        message="same failure",
        occurrence_id=first_occurrence,
    )
    assert len(health.read_recent_hook_alerts(tmp_path, now=NOW)) == 1
    assert health.record_hook_diagnostic(
        tmp_path,
        event="capture_failed",
        source_agent="claude",
        session_id="second",
        project="memory",
        message="same failure",
        occurrence_id=second_occurrence,
    )
    assert health.read_recent_hook_alerts(tmp_path, now=NOW) == ()


def test_health_suppression_keeps_unpersisted_different_message(tmp_path):
    health = _health_module()
    persisted_occurrence = "3" * 32
    fallback_occurrence = "4" * 32
    _write_hook_log(
        tmp_path,
        [
            _error_record(message="persisted failure", occurrence_id=persisted_occurrence),
            _error_record(message="fallback failure", occurrence_id=fallback_occurrence),
        ],
    )
    assert health.record_hook_diagnostic(
        tmp_path,
        event="capture_failed",
        source_agent="codex",
        session_id="persisted",
        project="memory",
        message="persisted failure",
        occurrence_id=persisted_occurrence,
    )

    alerts = health.read_recent_hook_alerts(tmp_path, now=NOW)

    assert [alert.message for alert in alerts] == ["fallback failure"]


@pytest.mark.parametrize("event", ["queue_unavailable", "capture_succeeded", "hook_log"])
def test_record_hook_diagnostic_ignores_log_only_or_nonfailure_events(tmp_path, event):
    health = _health_module()

    assert health.record_hook_diagnostic(
        tmp_path,
        event=event,
        source_agent="claude",
        session_id="session",
        project="memory",
        message="failure",
        deadline=10.0,
        clock=lambda: 0.0,
    ) is False
    assert not (tmp_path / "scripts" / "jobs.sqlite3").exists()

def test_record_hook_diagnostic_deadline_and_persistence_failure_are_advisory(
    tmp_path, monkeypatch
):
    health = _health_module()
    assert health.record_hook_diagnostic(
        tmp_path,
        event="capture_failed",
        source_agent="claude",
        session_id="session",
        project="memory",
        message="failure",
        deadline=0.05,
        clock=lambda: 0.0,
    ) is False
    assert not (tmp_path / "scripts" / "jobs.sqlite3").exists()

    monkeypatch.setattr(
        health,
        "QueueRepository",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("diagnostic unavailable")
        ),
        raising=False,
    )
    assert health.record_hook_diagnostic(
        tmp_path,
        event="capture_failed",
        source_agent="claude",
        session_id="session",
        project="memory",
        message="failure",
        deadline=10.0,
        clock=lambda: 0.0,
    ) is False


@pytest.mark.parametrize("failure_mode", ["persistence", "deadline"])
def test_hook_diagnostic_failure_or_deadline_does_not_change_hook_outcome(
    tmp_path, monkeypatch, failure_mode
):
    hook = _load_hook("session-end.py")
    logger, handler = _record_logger(f"diagnostic-{failure_mode}")
    memory_home = tmp_path / "memory"
    monkeypatch.setenv("AI_MEMORY_HOME", str(memory_home))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    monkeypatch.setattr(hook, "_logger", lambda: logger)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"session_id": "advisory"})),
    )
    if failure_mode == "persistence":
        monkeypatch.setattr(
            hook,
            "persist_hook_diagnostic_with_deadline",
            lambda *_args, **_kwargs: False,
        )
        clock = lambda: 0.0
    else:
        ticks = iter([0.0, 2.2])
        clock = lambda: next(ticks)

    hook.main(clock=clock)

    assert handler.records[-1].hook_event == "transcript_missing"
    assert not (memory_home / "scripts" / "jobs.sqlite3").exists()


def test_concurrent_hook_diagnostic_occurrences_are_distinct(tmp_path):
    health = _health_module()

    def record(index):
        return health.record_hook_diagnostic(
            tmp_path,
            event="capture_failed",
            source_agent="codex",
            session_id=f"session-{index}",
            project="memory",
            message="capture failed",
            deadline=10.0,
            clock=lambda: 0.0,
            token_factory=lambda: f"occurrence-{index}",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert all(executor.map(record, range(8)))

    with QueueRepository(
        tmp_path / "scripts" / "jobs.sqlite3",
        sync_usage=False,
    ) as repository:
        runs = repository._connection.execute(
            "SELECT id, operation_key FROM status_runs ORDER BY id"
        ).fetchall()
        assert len(runs) == 8
        assert len({row["operation_key"] for row in runs}) == 8
        event_count = repository._connection.execute(
            "SELECT count(*) FROM status_events"
        ).fetchone()[0]
        assert event_count == 8
