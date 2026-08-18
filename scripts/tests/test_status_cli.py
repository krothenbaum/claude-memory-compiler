from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
from rich.cells import cell_len
from status_store import (
    CompileStatus,
    ObserverState,
    RunState,
    StatusRun,
    StatusSnapshot,
)

from scripts.queue import QueueRepository

NOW = datetime(2026, 8, 18, 19, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[2]


class TTYBuffer(StringIO):
    def isatty(self):
        return True


def _run(
    run_id,
    *,
    kind="capture",
    source_agent="claude",
    project="memory",
    state: RunState = "succeeded",
    phase="succeeded",
    summary=None,
    error=None,
    age=timedelta(minutes=1),
):
    updated = NOW - age
    return StatusRun(
        id=run_id,
        job_id=run_id if kind == "capture" else None,
        operation_key=None if kind == "capture" else f"{kind}:{run_id}",
        kind=kind,
        source_agent=source_agent,
        session_id=f"session-{run_id}",
        project=project,
        state=state,
        phase=phase,
        summary=summary,
        error=error,
        started_at=updated - timedelta(minutes=1),
        updated_at=updated,
        completed_at=updated if state in {"succeeded", "failed", "dead"} else None,
    )


def _install_snapshot(monkeypatch, snapshot):
    import status as status_cli

    monkeypatch.setattr(status_cli, "read_snapshot", lambda *_args, **_kwargs: snapshot)
    return status_cli


def test_empty_snapshot_prints_all_groups_without_ansi(tmp_path, monkeypatch):
    snapshot = StatusSnapshot(
        active=(),
        attention=(),
        recent=(),
        compile=CompileStatus(
            state="before_window",
            summary="Next automatic compile window begins at 16:00",
        ),
        health_alerts=(),
    )
    status_cli = _install_snapshot(monkeypatch, snapshot)
    output = StringIO()

    result = status_cli.main(
        ["--snapshot"],
        env={"AI_MEMORY_HOME": str(tmp_path)},
        now=NOW,
        output=output,
        terminal_width=100,
    )

    assert result == 0
    assert output.getvalue() == (
        "MEMORY STATUS · 2026-08-18 19:00 UTC\n"
        "\n"
        "ACTIVE\n"
        "  — None\n"
        "\n"
        "NEEDS ATTENTION\n"
        "  — None\n"
        "\n"
        "RECENT\n"
        "  — None\n"
        "\n"
        "END-OF-DAY COMPILE\n"
        "  ○ before window · Next automatic compile window begins at 16:00\n"
    )
    assert "\x1b[" not in output.getvalue()


def test_snapshot_golden_covers_success_fallback_retry_failure_and_compile(tmp_path, monkeypatch):
    compile_run = _run(
        5,
        kind="compile",
        source_agent="system",
        project="memory",
        state="running",
        phase="validation_started",
        summary="Validating staged changes",
        age=timedelta(seconds=30),
    )
    snapshot = StatusSnapshot(
        active=(
            _run(
                1,
                project="retry-project",
                state="retrying",
                phase="retry_wait",
                error="capacity exhausted",
                age=timedelta(minutes=2),
            ),
        ),
        attention=(
            _run(
                2,
                kind="query_file",
                source_agent="codex",
                project="query-project",
                state="failed",
                phase="failed",
                error="invalid output",
                age=timedelta(hours=1),
            ),
        ),
        recent=(
            _run(
                3,
                project="direct-project",
                summary="Saved 1,234 characters",
                age=timedelta(minutes=5),
            ),
            _run(
                4,
                source_agent="codex",
                project="fallback-project",
                summary="Saved 8 characters through Claude fallback",
                age=timedelta(hours=2),
            ),
        ),
        compile=CompileStatus(
            state="running",
            summary="Validating staged changes",
            run=compile_run,
        ),
        health_alerts=(),
    )
    status_cli = _install_snapshot(monkeypatch, snapshot)
    output = StringIO()

    result = status_cli.main(
        ["--snapshot"],
        env={"AI_MEMORY_HOME": str(tmp_path)},
        now=NOW,
        output=output,
        terminal_width=160,
    )

    assert result == 0
    assert output.getvalue() == (
        "MEMORY STATUS · 2026-08-18 19:00 UTC\n"
        "\n"
        "ACTIVE\n"
        "  ↻ Capture · Claude · retry-project · retry wait · capacity exhausted · 2m ago\n"
        "\n"
        "NEEDS ATTENTION\n"
        "  ! Filed answer · Codex · query-project · failed · invalid output · 1h ago\n"
        "\n"
        "RECENT\n"
        "  ✓ Capture · Claude · direct-project · succeeded · Saved 1,234 characters · 5m ago\n"
        "  ✓ Capture · Codex · fallback-project · succeeded · Saved 8 characters through Claude fallback · 2h ago\n"
        "\n"
        "END-OF-DAY COMPILE\n"
        "  ● running · Validating staged changes\n"
        "  ↳ Compile · System · memory · validation started · Validating staged changes · 30s ago\n"
    )


@pytest.mark.parametrize(
    ("state", "icon"),
    [
        ("ready", "○"),
        ("retrying", "↻"),
        ("complete", "✓"),
        ("failed", "!"),
    ],
)
def test_compile_states_have_stable_icons(tmp_path, monkeypatch, state, icon):
    snapshot = StatusSnapshot(
        active=(),
        attention=(),
        recent=(),
        compile=CompileStatus(state=state, summary=f"Compile is {state}"),
        health_alerts=(),
    )
    status_cli = _install_snapshot(monkeypatch, snapshot)
    output = StringIO()

    assert (
        status_cli.main(
            ["--snapshot"],
            env={"AI_MEMORY_HOME": str(tmp_path)},
            now=NOW,
            output=output,
            terminal_width=100,
        )
        == 0
    )

    assert f"  {icon} {state} · Compile is {state}\n" in output.getvalue()


def test_snapshot_calls_read_only_projection_contract(tmp_path, monkeypatch):
    import status as status_cli

    queue_path = tmp_path / "custom.sqlite3"
    config = SimpleNamespace(root_dir=tmp_path, queue_path=queue_path)
    observer = ObserverState.empty()
    snapshot = StatusSnapshot(
        active=(),
        attention=(),
        recent=(),
        compile=CompileStatus(state="ready", summary="Automatic compile is ready"),
        health_alerts=(),
    )
    observed = {}
    monkeypatch.setattr(status_cli, "load_config", lambda env: config)
    monkeypatch.setattr(status_cli, "observer_state_path", lambda root: root / "view.json")
    monkeypatch.setattr(status_cli, "load_observer_state", lambda path: observer)

    def read(path, **kwargs):
        observed["path"] = path
        observed.update(kwargs)
        return snapshot

    monkeypatch.setattr(status_cli, "read_snapshot", read)

    assert status_cli.main(["--snapshot"], env={}, now=NOW, output=StringIO()) == 0
    assert observed == {
        "path": queue_path,
        "now": NOW,
        "observer_state": observer,
        "memory_home": tmp_path,
        "health_alerts": (),
    }


def test_missing_database_is_diagnostic_nonzero_and_never_created(tmp_path):
    import status as status_cli

    queue_path = tmp_path / "missing.sqlite3"
    output = StringIO()

    result = status_cli.main(
        ["--snapshot"],
        env={
            "AI_MEMORY_HOME": str(tmp_path),
            "AI_MEMORY_QUEUE_PATH": str(queue_path),
        },
        now=NOW,
        output=output,
    )

    assert result == 2
    assert output.getvalue() == (
        f"MEMORY STATUS\n\nInspection failed: status database is unavailable: {queue_path}\n"
    )
    assert not queue_path.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits required")
def test_unsafe_database_is_diagnostic_nonzero(tmp_path):
    import status as status_cli

    queue_path = tmp_path / "unsafe.sqlite3"
    with QueueRepository(queue_path, sync_usage=False):
        pass
    queue_path.chmod(0o644)
    output = StringIO()

    result = status_cli.main(
        ["--snapshot"],
        env={
            "AI_MEMORY_HOME": str(tmp_path),
            "AI_MEMORY_QUEUE_PATH": str(queue_path),
        },
        now=NOW,
        output=output,
    )

    assert result == 2
    assert "Inspection failed: status database path is unsafe:" in output.getvalue()


def test_color_requires_tty_and_respects_both_disable_controls(tmp_path, monkeypatch):
    snapshot = StatusSnapshot(
        active=(),
        attention=(),
        recent=(_run(1, summary="Saved 5 characters", age=timedelta(minutes=1)),),
        compile=CompileStatus(state="complete", summary="Compiled"),
        health_alerts=(),
    )
    status_cli = _install_snapshot(monkeypatch, snapshot)

    tty = TTYBuffer()
    assert (
        status_cli.main(
            ["--snapshot"],
            env={"AI_MEMORY_HOME": str(tmp_path)},
            now=NOW,
            output=tty,
        )
        == 0
    )
    assert "\x1b[" in tty.getvalue()

    for argv, env in (
        (["--snapshot", "--no-color"], {"AI_MEMORY_HOME": str(tmp_path)}),
        (["--snapshot"], {"AI_MEMORY_HOME": str(tmp_path), "NO_COLOR": "1"}),
    ):
        disabled = TTYBuffer()
        assert status_cli.main(argv, env=env, now=NOW, output=disabled) == 0
        assert "\x1b[" not in disabled.getvalue()

    pipe = StringIO()
    assert (
        status_cli.main(
            ["--snapshot"],
            env={"AI_MEMORY_HOME": str(tmp_path)},
            now=NOW,
            output=pipe,
        )
        == 0
    )
    assert "\x1b[" not in pipe.getvalue()


def test_terminal_width_truncates_rows_deterministically(tmp_path, monkeypatch):
    snapshot = StatusSnapshot(
        active=(),
        attention=(),
        recent=(
            _run(
                1,
                project="project-with-a-very-long-name",
                summary="Saved a very long result that cannot fit",
            ),
        ),
        compile=CompileStatus(state="complete", summary="Compiled"),
        health_alerts=(),
    )
    status_cli = _install_snapshot(monkeypatch, snapshot)
    output = StringIO()

    assert (
        status_cli.main(
            ["--snapshot"],
            env={"AI_MEMORY_HOME": str(tmp_path)},
            now=NOW,
            output=output,
            terminal_width=48,
        )
        == 0
    )

    assert all(len(line) <= 48 for line in output.getvalue().splitlines())
    assert "…" in output.getvalue()


def test_default_mode_lazy_loads_interactive_dashboard(monkeypatch):
    import status as status_cli

    explicit_env = {
        "AI_MEMORY_HOME": "/memory/root",
        "AI_MEMORY_QUEUE_PATH": "/memory/runtime/jobs.sqlite3",
    }
    calls = []
    module = ModuleType("status_app")
    module.__dict__["run_dashboard"] = (
        lambda *, no_color, env: calls.append((no_color, env)) or 7
    )
    monkeypatch.setitem(sys.modules, "status_app", module)

    assert status_cli.main(["--no-color"], env=explicit_env) == 7
    assert calls == [(True, explicit_env)]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("safe\x1b[31mred\x1b[0mtext", "saferedtext"),
        ("safe\x1b]8;;https://unsafe.example\x07click\x1b]8;;\x07", "safeclick"),
        ("left\rright", "left right"),
        ("left\bright", "left right"),
        ("left\x85right", "left right"),
        ("left\nright", "left right"),
    ],
)
def test_safe_terminal_text_strips_escape_and_control_sequences(value, expected):
    import status as status_cli

    assert status_cli._safe_terminal_text(value) == expected


def test_every_dynamic_snapshot_field_is_terminal_safe(tmp_path, monkeypatch):
    malicious = SimpleNamespace(
        id=1,
        job_id=1,
        operation_key=None,
        kind="cap\x1b[31mture",
        source_agent="claude\x9b31m",
        session_id="session",
        project="project\rforged\nline\bback\x85c1",
        state="running",
        phase="codex_started\x1b]0;owned\x07",
        summary="working\x1b[2Jstill",
        error=None,
        started_at=NOW,
        updated_at=NOW,
        completed_at=None,
    )
    snapshot = StatusSnapshot(
        active=(cast(StatusRun, malicious),),
        attention=(),
        recent=(),
        compile=CompileStatus(
            state="ready\x1b[31m",
            summary="compile\nforged\x1b]8;;unsafe\x07text\x1b]8;;\x07",
        ),
        health_alerts=(),
    )
    status_cli = _install_snapshot(monkeypatch, snapshot)
    output = StringIO()

    assert (
        status_cli.main(
            ["--snapshot"],
            env={"AI_MEMORY_HOME": str(tmp_path)},
            now=NOW,
            output=output,
        )
        == 0
    )

    rendered = output.getvalue()
    assert len(rendered.splitlines()) == 13
    assert "owned" not in rendered
    assert "unsafe" not in rendered
    assert "project forged line back c1" in rendered
    assert all(
        character == "\n" or not (ord(character) < 32 or 0x7F <= ord(character) <= 0x9F)
        for character in rendered
    )


def test_diagnostic_text_is_terminal_safe(monkeypatch):
    import status as status_cli

    monkeypatch.setattr(
        status_cli,
        "load_config",
        lambda _env: (_ for _ in ()).throw(
            ValueError("bad\rpath\nforged\x1b[31mred\x1b]0;owned\x07")
        ),
    )
    output = StringIO()

    assert status_cli.main(["--snapshot"], env={}, now=NOW, output=output) == 2
    assert output.getvalue() == "MEMORY STATUS\n\nInspection failed: bad path forgedred\n"


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, str(ROOT / "scripts" / "status.py"), "--snapshot"],
        [sys.executable, "-m", "scripts.status", "--snapshot"],
    ],
    ids=["direct", "module"],
)
def test_invalid_environment_is_guarded_without_traceback(command):
    env = dict(os.environ)
    env["AI_MEMORY_PROVIDER_ORDER"] = "bad"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == (
        "MEMORY STATUS\n\nInspection failed: AI_MEMORY_PROVIDER_ORDER must be codex,claude\n"
    )
    assert completed.stderr == ""


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, str(ROOT / "scripts" / "status.py")],
        [sys.executable, "-m", "scripts.status"],
    ],
    ids=["direct-default", "module-default"],
)
def test_invalid_environment_in_default_mode_is_guarded_without_traceback(command):
    env = dict(os.environ)
    env["AI_MEMORY_PROVIDER_ORDER"] = "bad"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == (
        "MEMORY STATUS\n\nInspection failed: AI_MEMORY_PROVIDER_ORDER must be codex,claude\n"
    )
    assert completed.stderr == ""


def test_ascii_output_uses_encoding_safe_semantic_glyphs(tmp_path, monkeypatch):
    snapshot = StatusSnapshot(
        active=(_run(1, state="running", phase="codex_started", summary="Working"),),
        attention=(_run(2, state="failed", phase="failed", error="Failed"),),
        recent=(_run(3, state="succeeded", phase="succeeded", summary="Saved"),),
        compile=CompileStatus(state="complete", summary="Compiled"),
        health_alerts=(),
    )
    status_cli = _install_snapshot(monkeypatch, snapshot)
    raw = BytesIO()
    output = TextIOWrapper(raw, encoding="ascii", errors="strict", write_through=True)

    assert (
        status_cli.main(
            ["--snapshot", "--no-color"],
            env={"AI_MEMORY_HOME": str(tmp_path)},
            now=NOW,
            output=output,
        )
        == 0
    )

    rendered = raw.getvalue().decode("ascii")
    assert "MEMORY STATUS - 2026-08-18 19:00 UTC" in rendered
    assert "  * Capture | Claude | memory | codex started | Working | 1m ago" in rendered
    assert "  ! Capture | Claude | memory | failed | Failed | 1m ago" in rendered
    assert "  + Capture | Claude | memory | succeeded | Saved | 1m ago" in rendered
    assert "  + complete | Compiled" in rendered


def test_fit_uses_terminal_cells_and_preserves_graphemes():
    import status as status_cli

    fitted = status_cli._fit("界e\u0301界e\u0301界", 7)

    assert fitted == "界e\u0301界e\u0301…"
    assert cell_len(fitted) == 7


def test_pipe_output_uses_full_width_policy(tmp_path, monkeypatch):
    summary = "result-" + ("x" * 300)
    snapshot = StatusSnapshot(
        active=(),
        attention=(),
        recent=(_run(1, summary=summary),),
        compile=CompileStatus(state="complete", summary="Compiled"),
        health_alerts=(),
    )
    status_cli = _install_snapshot(monkeypatch, snapshot)
    output = StringIO()

    assert (
        status_cli.main(
            ["--snapshot"],
            env={"AI_MEMORY_HOME": str(tmp_path)},
            now=NOW,
            output=output,
        )
        == 0
    )

    assert summary in output.getvalue()
    assert "…" not in output.getvalue()
