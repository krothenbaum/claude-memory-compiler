from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from functools import wraps

import pytest

from scripts.status_app import StatusDashboard, run_dashboard
from scripts.status_store import (
    CompileStatus,
    ObserverState,
    RunDetails,
    StatusDatabaseUnavailable,
    StatusDataInvalid,
    StatusEvent,
    StatusRun,
    StatusSnapshot,
)

NOW = datetime(2026, 8, 18, 18, 0, tzinfo=UTC)


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


def run(run_id, state, phase, *, kind="capture", project="memory", error=None):
    return StatusRun(
        id=run_id,
        job_id=run_id,
        operation_key=None,
        kind=kind,
        source_agent="claude",
        session_id=f"session-{run_id}",
        project=project,
        state=state,
        phase=phase,
        summary="Saved knowledge" if state == "succeeded" else None,
        error=error,
        started_at=NOW,
        updated_at=NOW,
        completed_at=NOW if state in {"succeeded", "failed", "dead"} else None,
    )


def snapshot():
    active = run(1, "running", "provider_started")
    failed = run(2, "failed", "failed", error="sanitized failure")
    recent = run(3, "succeeded", "succeeded")
    compile_run = run(4, "running", "validation_started", kind="compile")
    return StatusSnapshot(
        active=(active,),
        attention=(failed,),
        recent=(recent,),
        compile=CompileStatus("compiling", "Validating staged changes", compile_run),
        health_alerts=(),
    )


def details_for(run_id):
    selected = next(
        item
        for item in (
            *snapshot().active,
            *snapshot().attention,
            *snapshot().recent,
            snapshot().compile.run,
        )
        if item is not None
        if item.id == run_id
    )
    event = StatusEvent(
        id=1,
        run_id=run_id,
        phase=selected.phase,
        level="info",
        provider="codex",
        attempt=1,
        message="Safe event",
        details={},
        created_at=NOW,
    )
    return RunDetails(selected, (event,), (), True)


def app(tmp_path, reader=lambda *_args, **_kwargs: snapshot()):
    return StatusDashboard(
        tmp_path / "jobs.sqlite3",
        memory_home=tmp_path,
        snapshot_reader=reader,
        details_reader=lambda _path, run_id: details_for(run_id),
        observer_loader=lambda _path: ObserverState.empty(),
        acknowledger=lambda _path, _run_id: ObserverState.empty(),
        clock=lambda: NOW,
    )


@async_test
@pytest.mark.parametrize(
    ("size", "layout_class"),
    [((120, 40), "wide"), ((80, 40), "stacked"), ((60, 30), "compact")],
)
async def test_dashboard_structure_and_responsive_layout(tmp_path, size, layout_class):
    dashboard = app(tmp_path)
    async with dashboard.run_test(size=size) as pilot:
        await pilot.pause()
        assert dashboard.has_class(layout_class)
        for selector in (
            "#app-header",
            "#health-banner",
            "#run-list",
            "#details-pane",
            "#compile-panel",
            "#status-footer",
        ):
            assert dashboard.query_one(selector)
        assert "1 active" in str(dashboard.query_one("#app-header").render())


@async_test
async def test_refresh_keeps_last_good_snapshot_on_sqlite_contention(tmp_path):
    calls = 0

    def reader(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return snapshot()
        raise sqlite3.OperationalError("database is locked")

    dashboard = app(tmp_path, reader)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        await dashboard.refresh_snapshot()
        assert dashboard.snapshot == snapshot()
        assert dashboard.query_one("#delayed-banner").display is True


@async_test
async def test_missing_database_shows_persistent_diagnostic_without_creation(tmp_path):
    queue_path = tmp_path / "missing.sqlite3"

    def reader(*_args, **_kwargs):
        raise StatusDatabaseUnavailable(queue_path, "missing database")

    dashboard = app(tmp_path, reader)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        assert dashboard.query_one("#diagnostic").display is True
        assert "missing.sqlite3" in str(dashboard.query_one("#diagnostic").render())
    assert not queue_path.exists()


@async_test
async def test_selection_persists_by_run_id_across_regrouping(tmp_path):
    dashboard = app(tmp_path)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        dashboard.selected_run_id = 2
        dashboard.preferred_run_id = 2
        await dashboard.refresh_snapshot()
        assert dashboard.selected_run_id == 2


@async_test
async def test_keyboard_selection_filter_and_acknowledgment(tmp_path):
    queries = []
    acknowledged = []

    def reader(*_args, **kwargs):
        queries.append(kwargs["query"])
        return snapshot()

    dashboard = StatusDashboard(
        tmp_path / "jobs.sqlite3",
        memory_home=tmp_path,
        snapshot_reader=reader,
        details_reader=lambda _path, run_id: details_for(run_id),
        observer_loader=lambda _path: ObserverState.empty(),
        acknowledger=lambda _path, run_id: (
            acknowledged.append(run_id)
            or ObserverState(1, frozenset({run_id}))
        ),
        clock=lambda: NOW,
    )
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        assert dashboard.selected_run_id == 1
        await pilot.press("j")
        await pilot.pause()
        assert dashboard.selected_run_id == 2
        await pilot.press("a")
        await pilot.pause()
        assert acknowledged == [2]
        await pilot.press("/")
        await pilot.press("m", "e", "m", "o", "r", "y", "enter")
        assert queries[-1] == "memory"
        await pilot.press("escape")
        assert dashboard.filter_query == ""
        await pilot.press("k")
        await pilot.pause()
        assert dashboard.selected_run_id == 1


@async_test
async def test_compact_enter_opens_and_escape_closes_details_overlay(tmp_path):
    dashboard = app(tmp_path)
    async with dashboard.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert dashboard.screen.query_one("#details-overlay")
        await pilot.press("escape")
        assert dashboard.screen.id == "_default"


@async_test
async def test_no_color_preserves_icons_and_labels(tmp_path):
    dashboard = StatusDashboard(
        tmp_path / "jobs.sqlite3",
        memory_home=tmp_path,
        snapshot_reader=lambda *_args, **_kwargs: snapshot(),
        details_reader=lambda _path, run_id: details_for(run_id),
        observer_loader=lambda _path: ObserverState.empty(),
        acknowledger=lambda _path, _run_id: ObserverState.empty(),
        clock=lambda: NOW,
        no_color=True,
    )
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        assert dashboard.has_class("nocolor")
        rendered = " ".join(
            str(row.render()) for row in dashboard.query(".run-row")
        )
        assert any(icon in rendered for icon in ("●", "◉"))
        assert "✓" in rendered and "✗" in rendered
        assert "provider_started" in rendered


@async_test
async def test_live_resize_changes_visible_column_contract(tmp_path):
    dashboard = app(tmp_path)
    async with dashboard.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        wide = str(dashboard.query_one("#run-positive-1").render())
        assert "session-1" in wide and "provider" in wide and "elapsed" in wide
        await pilot.resize_terminal(80, 40)
        await pilot.pause()
        stacked = str(dashboard.query_one("#run-positive-1").render())
        assert "session-1" not in stacked and "provider" in stacked
        await pilot.resize_terminal(60, 30)
        await pilot.pause()
        compact = str(dashboard.query_one("#run-positive-1").render())
        assert "provider=" not in compact and "elapsed=" not in compact
        assert "memory" in compact and "running" in compact


@async_test
async def test_quit_binding_exits_cleanly(tmp_path):
    dashboard = app(tmp_path)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
    assert not dashboard.is_running


def test_public_dashboard_runner_returns_zero_after_clean_exit(monkeypatch):
    exits = []
    monkeypatch.setattr(StatusDashboard, "run", lambda self: exits.append(self))

    assert run_dashboard(no_color=True) == 0
    assert exits and exits[0].no_color is True


@async_test
async def test_wrapped_sqlite_busy_keeps_last_good_frame_and_initial_busy_diagnoses(
    tmp_path,
):
    calls = 0

    def reader(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return snapshot()
        cause = sqlite3.OperationalError("database is locked")
        raise StatusDataInvalid(tmp_path / "jobs.sqlite3", "invalid") from cause

    dashboard = app(tmp_path, reader)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        await dashboard.refresh_snapshot()
        assert dashboard.snapshot is not None
        assert dashboard.query_one("#delayed-banner").display is True

    def initially_busy(*_args, **_kwargs):
        raise StatusDataInvalid(tmp_path / "missing.sqlite3", "busy") from (
            sqlite3.OperationalError("database is locked")
        )

    initial = app(tmp_path, initially_busy)
    async with initial.run_test() as pilot:
        await pilot.pause()
        assert initial.snapshot is None
        assert initial.query_one("#delayed-banner").display is True
        assert initial.query_one("#diagnostic").display is True
    assert not (tmp_path / "missing.sqlite3").exists()


@async_test
async def test_filter_fallback_restores_preferred_selection_when_cleared(tmp_path):
    def reader(*_args, **kwargs):
        value = snapshot()
        if kwargs["query"] == "success":
            return StatusSnapshot(
                active=(),
                attention=(),
                recent=value.recent,
                compile=value.compile,
                health_alerts=(),
            )
        return value

    dashboard = app(tmp_path, reader)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        await pilot.press("j")
        await pilot.pause()
        assert dashboard.preferred_run_id == 2
        dashboard.filter_query = "success"
        await dashboard.refresh_snapshot()
        assert dashboard.selected_run_id == 3
        dashboard.filter_query = ""
        await dashboard.refresh_snapshot()
        assert dashboard.selected_run_id == 2


@async_test
async def test_compile_panel_is_not_duplicated_and_participates_in_navigation(tmp_path):
    dashboard = app(tmp_path)
    async with dashboard.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert len(dashboard.query("#run-positive-4")) == 0
        for _ in range(3):
            await pilot.press("j")
            await pilot.pause()
        assert dashboard.selected_run_id == 4
        assert dashboard.query_one("#compile-panel").has_class("selected")
        await pilot.press("enter")
        assert "Run 4" in str(dashboard.query_one("#details").render())

    compact = app(tmp_path)
    async with compact.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        for _ in range(3):
            await pilot.press("j")
            await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert compact.screen.query_one("#details-overlay")


@async_test
async def test_only_active_rows_animate_between_refresh_ticks(tmp_path):
    dashboard = app(tmp_path)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        active_before = str(dashboard.query_one("#run-positive-1").render())
        terminal_before = str(dashboard.query_one("#run-positive-3").render())
        await dashboard.refresh_snapshot()
        active_after = str(dashboard.query_one("#run-positive-1").render())
        terminal_after = str(dashboard.query_one("#run-positive-3").render())
        assert active_before != active_after
        assert terminal_before == terminal_after


@async_test
async def test_no_color_environment_presence_and_constructor_are_additive(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NO_COLOR", "")
    dashboard = app(tmp_path)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        assert dashboard.no_color is True
        assert dashboard.has_class("nocolor")
        assert dashboard.screen.has_class("nocolor")
        assert dashboard.query_one("#run-positive-1").has_class("state-running")

    explicit = StatusDashboard(
        tmp_path / "jobs.sqlite3",
        memory_home=tmp_path,
        snapshot_reader=lambda *_args, **_kwargs: snapshot(),
        details_reader=lambda _path, run_id: details_for(run_id),
        observer_loader=lambda _path: ObserverState.empty(),
        acknowledger=lambda _path, _run_id: ObserverState.empty(),
        clock=lambda: NOW,
        no_color=True,
    )
    assert explicit.no_color is True


@async_test
async def test_signed_ids_and_concurrent_refreshes_do_not_duplicate_rows(tmp_path):
    value = snapshot()
    mixed = StatusSnapshot(
        active=(value.active[0], replace(value.active[0], id=-1, job_id=1)),
        attention=value.attention,
        recent=value.recent,
        compile=value.compile,
        health_alerts=(),
    )
    dashboard = app(tmp_path, lambda *_args, **_kwargs: mixed)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        await asyncio.gather(
            dashboard.refresh_snapshot(),
            dashboard.refresh_snapshot(),
            dashboard._render_snapshot(),
        )
        assert dashboard.query_one("#run-positive-1")
        assert dashboard.query_one("#run-legacy-1")
        assert len(dashboard.query(".run-row")) == 4


@async_test
async def test_non_busy_operational_error_and_ack_failure_are_diagnostic(tmp_path):
    def broken(*_args, **_kwargs):
        raise sqlite3.OperationalError("no such table")

    diagnostic = app(tmp_path, broken)
    async with diagnostic.run_test() as pilot:
        await pilot.pause()
        assert diagnostic.query_one("#diagnostic").display is True
        assert diagnostic.query_one("#delayed-banner").display is False

    dashboard = StatusDashboard(
        tmp_path / "jobs.sqlite3",
        memory_home=tmp_path,
        snapshot_reader=lambda *_args, **_kwargs: snapshot(),
        details_reader=lambda _path, run_id: details_for(run_id),
        observer_loader=lambda _path: ObserverState.empty(),
        acknowledger=lambda *_args: (_ for _ in ()).throw(PermissionError("denied")),
        clock=lambda: NOW,
    )
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        await pilot.press("j", "a")
        await pilot.pause()
        assert dashboard.snapshot is not None
        assert "denied" in str(dashboard.query_one("#diagnostic").render())
