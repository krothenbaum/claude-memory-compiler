from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from functools import wraps

import pytest

from scripts.status_app import StatusDashboard
from scripts.status_store import (
    CompileStatus,
    ObserverState,
    RunDetails,
    StatusDatabaseUnavailable,
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
        for item in (*snapshot().active, *snapshot().attention, *snapshot().recent)
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
        await dashboard.refresh_snapshot()
        assert dashboard.selected_run_id == 2
