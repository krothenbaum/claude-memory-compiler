from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from functools import wraps

import pytest
from rich.text import Text
from textual.content import Content

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
@pytest.mark.parametrize(
    ("active", "attention", "recent", "expected"),
    [
        (
            (run(10, "queued", "queued"), run(11, "running", "provider_started")),
            (run(12, "dead", "dead", error="failed"),),
            (run(13, "succeeded", "succeeded"),),
            11,
        ),
        (
            (run(10, "queued", "queued"), run(14, "retrying", "retry_wait")),
            (run(12, "dead", "dead", error="failed"),),
            (run(13, "succeeded", "succeeded"),),
            12,
        ),
        (
            (run(10, "queued", "queued"), run(14, "retrying", "retry_wait")),
            (),
            (run(13, "succeeded", "succeeded"),),
            13,
        ),
        (
            (run(10, "queued", "queued"), run(14, "retrying", "retry_wait")),
            (),
            (),
            10,
        ),
    ],
)
async def test_initial_selection_uses_explicit_triage_priority(
    tmp_path, active, attention, recent, expected
):
    value = StatusSnapshot(
        active=active,
        attention=attention,
        recent=recent,
        compile=CompileStatus("before_window", "Not ready"),
        health_alerts=(),
    )
    runs = (*active, *attention, *recent)
    dashboard = StatusDashboard(
        tmp_path / "jobs.sqlite3",
        memory_home=tmp_path,
        snapshot_reader=lambda *_args, **_kwargs: value,
        details_reader=lambda _path, run_id: RunDetails(
            next(item for item in runs if item.id == run_id), (), (), True
        ),
        observer_loader=lambda _path: ObserverState.empty(),
        clock=lambda: NOW,
    )

    async with dashboard.run_test() as pilot:
        await pilot.pause()
        assert dashboard.selected_run_id == expected


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
        assert "session-1" in wide and "Claude" in wide
        assert "provider=" not in wide and "elapsed=" not in wide
        await pilot.resize_terminal(80, 40)
        await pilot.pause()
        stacked = str(dashboard.query_one("#run-positive-1").render())
        assert "session-1" not in stacked and "Claude" in stacked
        assert "provider=" not in stacked and "elapsed=" not in stacked
        await pilot.resize_terminal(60, 30)
        await pilot.pause()
        compact = str(dashboard.query_one("#run-positive-1").render())
        assert "provider=" not in compact and "elapsed=" not in compact
        assert "memory" in compact and "running" in compact
        assert "Claude" not in compact


@async_test
async def test_source_agent_labels_distinguish_rows_without_color(tmp_path):
    rows = (
        replace(run(21, "running", "provider_started"), source_agent="claude"),
        replace(run(22, "queued", "queued"), source_agent="codex"),
        replace(run(23, "retrying", "retry_wait"), source_agent="system"),
    )
    value = StatusSnapshot(
        active=rows,
        attention=(),
        recent=(),
        compile=CompileStatus("before_window", "Not ready"),
        health_alerts=(),
    )
    dashboard = StatusDashboard(
        tmp_path / "jobs.sqlite3",
        memory_home=tmp_path,
        snapshot_reader=lambda *_args, **_kwargs: value,
        details_reader=lambda _path, run_id: RunDetails(
            next(item for item in rows if item.id == run_id), (), (), True
        ),
        observer_loader=lambda _path: ObserverState.empty(),
        clock=lambda: NOW,
        no_color=True,
    )

    async with dashboard.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        rendered = {}
        for run_id in (21, 22, 23):
            content = dashboard.query_one(f"#run-positive-{run_id}").render()
            assert isinstance(content, Content)
            rendered[run_id] = content.plain
        assert "Claude" in rendered[21]
        assert "Codex" in rendered[22]
        assert "System" in rendered[23]
        assert all("provider=" not in row for row in rendered.values())


@async_test
async def test_rows_style_only_icon_and_state_and_render_no_terminal_controls(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    unsafe = "saved\x1b[31mred\x1b[0m\x1b]0;title\x07\rline\bback\x85c1"
    active = run(20, "running", "provider_started")
    active = replace(active, summary=unsafe)
    value = StatusSnapshot(
        active=(active,),
        attention=(),
        recent=(),
        compile=CompileStatus("before_window", "Not ready"),
        health_alerts=(),
    )
    details = RunDetails(
        active,
        (
            StatusEvent(
                id=20,
                run_id=20,
                phase="provider_started",
                level="info",
                provider="codex",
                attempt=1,
                message=unsafe,
                details={},
                created_at=NOW,
            ),
        ),
        (),
        True,
    )
    dashboard = StatusDashboard(
        tmp_path / "jobs.sqlite3",
        memory_home=tmp_path,
        snapshot_reader=lambda *_args, **_kwargs: value,
        details_reader=lambda *_args: details,
        observer_loader=lambda _path: ObserverState.empty(),
        clock=lambda: NOW,
    )

    async with dashboard.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        rendered = dashboard.query_one("#run-positive-20").render()
        assert isinstance(rendered, Content)
        rich_row = dashboard._row_text(active, "●")
        assert isinstance(rich_row, Text)
        styled = "".join(
            rich_row.plain[span.start : span.end] for span in rich_row.spans
        )
        assert any(icon in styled for icon in ("●", "◉"))
        assert "running" in styled
        assert "memory" not in styled
        assert "result=" not in styled
        rendered_details = dashboard.query_one("#details").render()
        assert isinstance(rendered_details, Content)
        captured = rendered.plain + rendered_details.plain
        assert "provider=—" not in captured and "elapsed=—" not in captured
        assert "\x1b" not in captured
        assert all(
            character == "\n"
            or not (ord(character) < 32 or 0x7F <= ord(character) <= 0x9F)
            for character in captured
        )


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
async def test_reconcile_identical_snapshot_preserves_widget_identity_and_scroll(
    tmp_path, monkeypatch
):
    value = replace(
        snapshot(),
        recent=tuple(run(run_id, "succeeded", "succeeded") for run_id in range(3, 23)),
    )
    dashboard = app(tmp_path, lambda *_args, **_kwargs: value)
    async with dashboard.run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        sections_before = tuple(dashboard.query(".run-section"))
        labels_before = tuple(dashboard.query(".section-label"))
        rows_before = {
            row.run_id: row for row in dashboard.query(".run-row")  # type: ignore[attr-defined]
        }
        run_list = dashboard.query_one("#run-list")
        run_list.scroll_to(y=1, animate=False)
        await pilot.pause()
        scroll_before = run_list.scroll_y
        assert scroll_before > 0
        dashboard.selected_run_id = 2
        dashboard.preferred_run_id = 2
        dashboard._update_selection_classes()

        mounts = 0
        removals = 0
        container_type = type(dashboard.query_one("#active-runs"))
        row_type = type(rows_before[1])
        original_mount = container_type.mount
        original_remove = row_type.remove

        def tracked_mount(self, *widgets, **kwargs):
            nonlocal mounts
            mounts += len(widgets)
            return original_mount(self, *widgets, **kwargs)

        def tracked_remove(self):
            nonlocal removals
            removals += 1
            return original_remove(self)

        monkeypatch.setattr(container_type, "mount", tracked_mount)
        monkeypatch.setattr(row_type, "remove", tracked_remove)
        await dashboard.refresh_snapshot()

        assert len(sections_before) == 3
        assert tuple(dashboard.query(".run-section")) == sections_before
        assert tuple(dashboard.query(".section-label")) == labels_before
        assert {
            row.run_id: row for row in dashboard.query(".run-row")  # type: ignore[attr-defined]
        } == rows_before
        assert mounts == 0 and removals == 0
        assert dashboard.selected_run_id == 2
        assert rows_before[2].has_class("selected")
        assert run_list.scroll_y == scroll_before


@async_test
async def test_spinner_updates_only_active_row_without_replacing_widgets(tmp_path):
    dashboard = app(tmp_path)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        active = dashboard.query_one("#run-positive-1")
        recent = dashboard.query_one("#run-positive-3")
        active_before = active.render()
        recent_before = recent.render()
        recent_signature = recent.view_signature  # type: ignore[attr-defined]
        recent_updates = recent.view_update_count  # type: ignore[attr-defined]

        await dashboard.refresh_snapshot()

        assert dashboard.query_one("#run-positive-1") is active
        assert dashboard.query_one("#run-positive-3") is recent
        assert active.render() != active_before
        assert recent.render() is recent_before
        assert recent.view_signature == recent_signature  # type: ignore[attr-defined]
        assert recent.view_update_count == recent_updates  # type: ignore[attr-defined]


@async_test
async def test_reconcile_adds_and_removes_only_changed_row(tmp_path, monkeypatch):
    current = snapshot()

    def reader(*_args, **_kwargs):
        return current

    dashboard = app(tmp_path, reader)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        stable = dashboard.query_one("#run-positive-3")
        mounted_ids = []
        removed_ids = []
        container_type = type(dashboard.query_one("#recent-runs"))
        row_type = type(stable)
        original_mount = container_type.mount
        original_remove = row_type.remove

        def tracked_mount(self, *widgets, **kwargs):
            mounted_ids.extend(widget.id for widget in widgets)
            return original_mount(self, *widgets, **kwargs)

        def tracked_remove(self):
            removed_ids.append(self.run_id)
            return original_remove(self)

        monkeypatch.setattr(container_type, "mount", tracked_mount)
        monkeypatch.setattr(row_type, "remove", tracked_remove)
        added = run(5, "succeeded", "succeeded")
        current = replace(current, recent=(*current.recent, added))
        await dashboard.refresh_snapshot()

        added_row = dashboard.query_one("#run-positive-5")
        assert dashboard.query_one("#run-positive-3") is stable
        assert added_row.run_id == 5  # type: ignore[attr-defined]
        assert mounted_ids == ["run-positive-5"]
        assert removed_ids == []

        mounted_ids.clear()
        current = replace(current, recent=(added,))
        await dashboard.refresh_snapshot()

        assert len(dashboard.query("#run-positive-3")) == 0
        assert dashboard.query_one("#run-positive-5") is added_row
        assert dashboard.query_one("#run-positive-1")
        assert dashboard.query_one("#run-positive-2")
        assert mounted_ids == []
        assert removed_ids == [3]


@async_test
async def test_reconcile_group_transition_keeps_unrelated_sections_and_rows(tmp_path):
    current = snapshot()

    def reader(*_args, **_kwargs):
        return current

    dashboard = app(tmp_path, reader)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        sections = tuple(dashboard.query(".run-section"))
        active = dashboard.query_one("#run-positive-1")
        recent = dashboard.query_one("#run-positive-3")
        failed = current.attention[0]
        current = replace(
            current,
            attention=(),
            recent=(
                replace(failed, state="succeeded", phase="succeeded", error=None),
                *current.recent,
            ),
        )
        dashboard.selected_run_id = 3
        dashboard.preferred_run_id = 3
        await dashboard.refresh_snapshot()

        assert tuple(dashboard.query(".run-section")) == sections
        assert dashboard.query_one("#run-positive-1") is active
        assert dashboard.query_one("#run-positive-3") is recent
        assert dashboard.selected_run_id == 3
        assert recent.has_class("selected")


@async_test
async def test_reconcile_reorders_rows_in_place_within_one_section(tmp_path):
    first = run(3, "succeeded", "succeeded")
    second = run(5, "succeeded", "succeeded")
    current = replace(snapshot(), recent=(first, second))

    def reader(*_args, **_kwargs):
        return current

    dashboard = app(tmp_path, reader)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        first_row = dashboard.query_one("#run-positive-3")
        second_row = dashboard.query_one("#run-positive-5")
        current = replace(current, recent=(second, first))
        await dashboard.refresh_snapshot()

        rows = tuple(dashboard.query_one("#recent-runs").query(".run-row"))
        assert rows == (second_row, first_row)
        assert dashboard.query_one("#run-positive-3") is first_row
        assert dashboard.query_one("#run-positive-5") is second_row


@async_test
async def test_filter_and_resize_preserve_rows_that_remain_visible(tmp_path):
    def reader(*_args, **kwargs):
        value = snapshot()
        if kwargs["query"]:
            return replace(value, attention=())
        return value

    dashboard = app(tmp_path, reader)
    async with dashboard.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        active = dashboard.query_one("#run-positive-1")
        recent = dashboard.query_one("#run-positive-3")
        dashboard.filter_query = "safe"
        await dashboard.refresh_snapshot()
        assert dashboard.query_one("#run-positive-1") is active
        assert dashboard.query_one("#run-positive-3") is recent

        await pilot.resize_terminal(60, 30)
        await pilot.pause()
        assert dashboard.query_one("#run-positive-1") is active
        assert dashboard.query_one("#run-positive-3") is recent
        assert "Claude" not in str(active.render())


@async_test
async def test_reconcile_truncation_notice_updates_only_when_state_changes(tmp_path):
    current = replace(snapshot(), has_more=False)

    def reader(*_args, **_kwargs):
        return current

    dashboard = app(tmp_path, reader)
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        notice = dashboard.query_one("#truncation-notice")
        signature = notice.render()
        await dashboard.refresh_snapshot()
        assert dashboard.query_one("#truncation-notice") is notice
        assert notice.render() is signature
        assert notice.display is False

        current = replace(current, has_more=True)
        await dashboard.refresh_snapshot()
        assert dashboard.query_one("#truncation-notice") is notice
        assert notice.display is True


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
    unsafe = "failure\x1b[31mred\x1b[0m\x1b]0;title\x07\rforged\bback\x85c1"
    unsafe_path = tmp_path / "queue\x1b[32mgreen\x1b[0m\x1b]0;path-title\x07\rline"

    def broken(*_args, **_kwargs):
        raise StatusDataInvalid(unsafe_path, unsafe)

    diagnostic = app(tmp_path, broken)
    async with diagnostic.run_test() as pilot:
        await pilot.pause()
        assert diagnostic.query_one("#diagnostic").display is True
        assert diagnostic.query_one("#delayed-banner").display is False
        rendered = diagnostic.query_one("#diagnostic").render()
        assert isinstance(rendered, Content)
        assert "failure" in rendered.plain and "queuegreen" in rendered.plain
        assert len(rendered.plain.splitlines()) == 1
        assert "\x1b" not in rendered.plain
        assert all(
            not (ord(character) < 32 or 0x7F <= ord(character) <= 0x9F)
            for character in rendered.plain
        )

    dashboard = StatusDashboard(
        tmp_path / "jobs.sqlite3",
        memory_home=tmp_path,
        snapshot_reader=lambda *_args, **_kwargs: snapshot(),
        details_reader=lambda _path, run_id: details_for(run_id),
        observer_loader=lambda _path: ObserverState.empty(),
        acknowledger=lambda *_args: (_ for _ in ()).throw(PermissionError(unsafe)),
        clock=lambda: NOW,
    )
    async with dashboard.run_test() as pilot:
        await pilot.pause()
        await pilot.press("j", "a")
        await pilot.pause()
        assert dashboard.snapshot is not None
        rendered = dashboard.query_one("#diagnostic").render()
        assert isinstance(rendered, Content)
        assert "failure" in rendered.plain
        assert len(rendered.plain.splitlines()) == 1
        assert "\x1b" not in rendered.plain
        assert all(
            not (ord(character) < 32 or 0x7F <= ord(character) <= 0x9F)
            for character in rendered.plain
        )
