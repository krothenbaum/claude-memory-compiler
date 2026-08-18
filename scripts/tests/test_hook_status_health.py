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
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import capture as capture_module
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


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


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
        payload = json.dumps({"session_id": "precompact-session"})
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
    assert len(alerts) == 1
    assert alerts[0].component == "pre-compact"


@pytest.mark.parametrize(
    ("result", "expected_event", "is_error"),
    [
        (ValueError("capture problem"), "capture_failed", True),
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

    helpers = SimpleNamespace(
        bounded_transcript_slice=selected,
        require_time_remaining=lambda *_args, **_kwargs: None,
        enqueue_capture_with_deadline=enqueue,
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
            capture_module.CaptureQueueUnavailableError("database is unavailable")
        ),
    )

    hook.main(clock=lambda: 0.0)

    assert handler.records[-1].hook_event == "queue_unavailable"


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
