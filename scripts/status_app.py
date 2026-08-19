"""Read-only Textual dashboard for durable memory job status."""

from __future__ import annotations

import asyncio
import importlib
import os
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, Static

try:
    from scripts.config import load_config
    from scripts.privacy import normalize_persistence_reason
    from scripts.status_store import (
        HealthAlert,
        ObserverState,
        RunDetails,
        StatusReadError,
        StatusRun,
        StatusSnapshot,
        acknowledge_run,
        load_observer_state,
        observer_state_path,
        read_run_details,
        read_snapshot,
    )
except ImportError:  # Direct execution with scripts/ on sys.path.
    from config import load_config
    from privacy import normalize_persistence_reason
    from status_store import (
        HealthAlert,
        ObserverState,
        RunDetails,
        StatusReadError,
        StatusRun,
        StatusSnapshot,
        acknowledge_run,
        load_observer_state,
        observer_state_path,
        read_run_details,
        read_snapshot,
    )

SnapshotReader = Callable[..., StatusSnapshot]
DetailsReader = Callable[[Path, int], RunDetails]
HealthLoader = Callable[[], tuple[HealthAlert, ...]]
MAX_RENDERED_RUNS = 200
MAX_VISIBLE_HEALTH_ALERTS = 2
_SOURCE_LABELS = {"claude": "Claude", "codex": "Codex", "system": "System"}


def _diagnostic_text(value: object) -> str:
    """Return bounded, redacted terminal-safe diagnostic text."""
    return normalize_persistence_reason(value, os.environ)


def read_recent_hook_alerts(*args, **kwargs) -> tuple[HealthAlert, ...]:
    """Import hook health only when a configured dashboard refreshes."""
    module_name = f"{__package__}.status_health" if __package__ else "status_health"
    return importlib.import_module(module_name).read_recent_hook_alerts(*args, **kwargs)


class DetailsOverlay(ModalScreen[None]):
    CSS = "#details-overlay { width: 90%; height: 80%; border: round cyan; padding: 1; }"
    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "close", "Close"),
        ("enter", "close", "Close"),
    ]

    def __init__(self, content: Text) -> None:
        super().__init__()
        self.content = content

    def compose(self) -> ComposeResult:
        yield Static(self.content, id="details-overlay")

    def action_close(self) -> None:
        self.dismiss()


class RunRow(Static):
    """A keyed run row that updates only when its visible view changes."""

    def __init__(
        self,
        run: StatusRun,
        content: Text,
        *,
        widget_id: str,
        content_signature: tuple[object, ...],
        selected: bool,
    ) -> None:
        super().__init__(
            content,
            id=widget_id,
            classes=f"run-row state-{run.state}",
        )
        self.run_id = run.id
        self._state = run.state
        self._content_signature = content_signature
        self._selected = selected
        self.view_signature = (content_signature, selected)
        self.view_update_count = 1
        self.set_class(selected, "selected")

    def update_view(
        self,
        run: StatusRun,
        content: Text,
        *,
        content_signature: tuple[object, ...],
        selected: bool,
    ) -> None:
        """Apply only changed content, state classes, or selection."""
        if content_signature != self._content_signature:
            if run.state != self._state:
                self.remove_class(f"state-{self._state}")
                self.add_class(f"state-{run.state}")
                self._state = run.state
            self.update(content)
            self._content_signature = content_signature
            self.view_update_count += 1
        self.set_selected(selected)

    def set_selected(self, selected: bool) -> None:
        if selected != self._selected:
            self.set_class(selected, "selected")
            self._selected = selected
        self.view_signature = (self._content_signature, self._selected)


class StatusDashboard(App[None]):
    """Triage dashboard that observes, but never controls, execution."""

    CSS = """
    Screen { layout: vertical; }
    #app-header { height: 3; padding: 0 1; text-style: bold; color: ansi_bright_cyan; }
    #health-banner, #delayed-banner, #diagnostic { height: auto; padding: 0 1; }
    #health-banner {
        color: yellow;
        max-height: 3;
        overflow-y: hidden;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    #delayed-banner { color: yellow; display: none; }
    #diagnostic { color: red; display: none; }
    #filter-input { display: none; height: 3; }
    #main-grid { height: 1fr; }
    #runs-pane, #details-pane { border: round $panel; padding: 0 1; }
    #runs-pane { width: 1fr; }
    #details-pane { width: 1fr; }
    #run-list { height: 1fr; }
    .run-section, .run-rows { height: auto; }
    .section-label { color: $text-muted; text-style: bold; margin-top: 1; }
    .run-row { height: 1; padding: 0 1; }
    .run-row.selected { text-style: bold reverse; }
    #compile-panel { height: 3; border: round $panel; padding: 0 1; }
    #compile-panel.selected { border: round ansi_bright_cyan; text-style: bold reverse; }
    #status-footer { height: 1; }
    StatusDashboard.stacked #main-grid { layout: vertical; }
    StatusDashboard.stacked #runs-pane { width: 1fr; height: 1fr; }
    StatusDashboard.stacked #details-pane { width: 1fr; height: 1fr; }
    StatusDashboard.compact #main-grid { layout: vertical; }
    StatusDashboard.compact #details-pane { display: none; }
    StatusDashboard.compact #runs-pane { width: 1fr; height: 1fr; }
    StatusDashboard.nocolor #app-header,
    StatusDashboard.nocolor .run-row,
    StatusDashboard.nocolor #health-banner,
    StatusDashboard.nocolor #delayed-banner,
    StatusDashboard.nocolor #diagnostic { color: $text; }
    StatusDashboard.nocolor .run-row.selected { text-style: bold; }
    StatusDashboard.nocolor #runs-pane,
    StatusDashboard.nocolor #details-pane,
    StatusDashboard.nocolor #compile-panel { border: round #808080; }
    StatusDashboard.nocolor #compile-panel.selected { border: round #808080; text-style: bold; }
    StatusDashboard.nocolor Footer { color: $text; background: transparent; }
    DetailsOverlay.nocolor #details-overlay { border: round #808080; color: $text; background: transparent; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("?", "help", "Help", priority=True),
        Binding("j", "next_run", "Next", priority=True),
        Binding("down", "next_run", "Next", priority=True),
        Binding("k", "previous_run", "Previous", priority=True),
        Binding("up", "previous_run", "Previous", priority=True),
        Binding("enter", "details", "Details"),
        Binding("/", "filter", "Filter", priority=True),
        Binding("escape", "escape", "Clear", priority=True),
        Binding("a", "acknowledge", "Acknowledge", priority=True),
    ]

    def __init__(
        self,
        queue_path: Path,
        *,
        memory_home: Path,
        snapshot_reader: SnapshotReader = read_snapshot,
        details_reader: DetailsReader = read_run_details,
        observer_loader: Callable[[Path], ObserverState] = load_observer_state,
        acknowledger: Callable[[Path, int], ObserverState] = acknowledge_run,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        health_loader: HealthLoader | None = None,
        no_color: bool | None = None,
    ) -> None:
        super().__init__()
        self.queue_path = Path(queue_path)
        self.memory_home = Path(memory_home)
        self._snapshot_reader = snapshot_reader
        self._details_reader = details_reader
        self._observer_loader = observer_loader
        self._acknowledger = acknowledger
        self._clock = clock
        self._health_loader = health_loader or (
            lambda: read_recent_hook_alerts(self.memory_home, now=self._clock())
        )
        self.no_color = bool(no_color or "NO_COLOR" in os.environ)
        self.observer_path = observer_state_path(self.memory_home)
        try:
            self.observer_state = observer_loader(self.observer_path)
        except (OSError, ValueError):
            self.observer_state = ObserverState.empty()
        self.snapshot: StatusSnapshot | None = None
        self.selected_run_id: int | None = None
        self.preferred_run_id: int | None = None
        self.filter_query = ""
        self._spinner_frame = 0
        self._render_lock = asyncio.Lock()
        self._render_dirty = False
        self._refresh_generation = 0
        self._detail_generation = 0
        self._details_run_id: int | None = None
        self._run_rows: dict[int, RunRow] = {}
        self._run_groups: dict[int, str] = {}

    def compose(self) -> ComposeResult:
        yield Static(id="app-header")
        yield Static(id="health-banner")
        yield Static("Status refresh delayed", id="delayed-banner")
        yield Static(id="diagnostic")
        yield Input(
            placeholder="Filter status history", id="filter-input", disabled=True
        )
        with Horizontal(id="main-grid"):
            with Vertical(id="runs-pane"), VerticalScroll(id="run-list"):
                with Vertical(id="active-section", classes="run-section"):
                    yield Static("ACTIVE", classes="section-label")
                    yield Vertical(id="active-runs", classes="run-rows")
                with Vertical(id="attention-section", classes="run-section"):
                    yield Static("NEEDS ATTENTION", classes="section-label")
                    yield Vertical(id="attention-runs", classes="run-rows")
                with Vertical(id="recent-section", classes="run-section"):
                    yield Static("RECENT", classes="section-label")
                    yield Vertical(id="recent-runs", classes="run-rows")
                yield Static(
                    f"Showing first {MAX_RENDERED_RUNS} matching runs — refine filter",
                    id="truncation-notice",
                    classes="section-label",
                )
            with Vertical(id="details-pane"):
                yield Static("Select a run", id="details")
        yield Static(id="compile-panel")
        yield Footer(id="status-footer")

    async def on_mount(self) -> None:
        if self.no_color:
            self.add_class("nocolor")
            self.screen.add_class("nocolor")
        self._set_layout(self.size.width)
        await self.refresh_snapshot()
        self.set_interval(1.0, self._automatic_refresh)

    async def _automatic_refresh(self) -> None:
        if not self.filter_query:
            await self.refresh_snapshot()

    def on_resize(self, event) -> None:
        self._set_layout(event.size.width)
        if self.snapshot is not None:
            self.call_later(self._render_snapshot)

    def _set_layout(self, width: int) -> None:
        self.remove_class("wide", "stacked", "compact")
        self.add_class("wide" if width >= 100 else "stacked" if width >= 70 else "compact")

    def _all_runs(self) -> tuple[StatusRun, ...]:
        if self.snapshot is None:
            return ()
        grouped = tuple(
            run
            for run in (
                *self.snapshot.active,
                *self.snapshot.attention,
                *self.snapshot.recent,
            )
            if run.kind != "compile"
        )
        compile_run = self.snapshot.compile.run
        return (*grouped, compile_run) if compile_run is not None else grouped

    def _choose_selection(self) -> None:
        runs = self._all_runs()
        visible = {run.id for run in runs}
        if self.preferred_run_id in visible:
            self.selected_run_id = self.preferred_run_id
            return
        if self.selected_run_id not in visible:
            assert self.snapshot is not None
            candidates = (
                tuple(
                    run for run in self.snapshot.active if run.state == "running"
                )
                or self.snapshot.attention
                or self.snapshot.recent
                or tuple(
                    run
                    for run in self.snapshot.active
                    if run.state in {"queued", "retrying"}
                )
                or runs
            )
            self.selected_run_id = candidates[0].id if candidates else None
        if self.preferred_run_id is None:
            self.preferred_run_id = self.selected_run_id

    @staticmethod
    def _is_busy_error(error: BaseException) -> bool:
        pending: list[BaseException] = [error]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if isinstance(current, sqlite3.OperationalError):
                code = getattr(current, "sqlite_errorcode", None)
                primary = code & 0xFF if isinstance(code, int) else None
                if primary in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                    return True
                message = str(current).lower()
                if "database is locked" in message or "database is busy" in message:
                    return True
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
        return False

    async def refresh_snapshot(self) -> None:
        self._refresh_generation += 1
        generation = self._refresh_generation
        current = self._clock()
        try:
            health_alerts = await asyncio.to_thread(self._health_loader)
        except (OSError, RuntimeError, ValueError):
            health_alerts = ()
        try:
            result = await asyncio.to_thread(
                self._snapshot_reader,
                self.queue_path,
                now=current,
                observer_state=self.observer_state,
                query=self.filter_query,
                health_alerts=health_alerts,
                memory_home=self.memory_home,
                max_runs=MAX_RENDERED_RUNS,
            )
        except (StatusReadError, sqlite3.Error, OSError, ValueError) as error:
            if generation != self._refresh_generation:
                return
            if self._is_busy_error(error):
                self.query_one("#delayed-banner", Static).display = True
                if self.snapshot is None:
                    diagnostic = self.query_one("#diagnostic", Static)
                    diagnostic.update(Text("Status refresh delayed: database is busy"))
                    diagnostic.display = True
                else:
                    self.query_one("#diagnostic", Static).display = False
                return
            self.query_one("#delayed-banner", Static).display = False
            diagnostic = self.query_one("#diagnostic", Static)
            path = getattr(error, "path", self.queue_path)
            diagnostic.update(
                Text(
                    f"Status unavailable: {_diagnostic_text(path)} — "
                    f"{_diagnostic_text(error)}"
                )
            )
            diagnostic.display = True
            return
        if generation != self._refresh_generation:
            return
        self.snapshot = result
        self._spinner_frame = (self._spinner_frame + 1) % 2
        self.query_one("#delayed-banner", Static).display = False
        self.query_one("#diagnostic", Static).display = False
        self._choose_selection()
        await self._render_snapshot()

    async def _render_snapshot(self) -> None:
        if self._render_lock.locked():
            self._render_dirty = True
            return
        async with self._render_lock:
            while True:
                self._render_dirty = False
                await self._render_snapshot_unlocked()
                if not self._render_dirty:
                    break

    @staticmethod
    def _row_id(run_id: int) -> str:
        return f"run-positive-{run_id}" if run_id > 0 else f"run-legacy-{abs(run_id)}"

    @staticmethod
    def _run_icon(run: StatusRun, spinner_frame: int) -> str:
        return {
            "running": ("●", "◉")[spinner_frame],
            "queued": "◌",
            "retrying": "↻",
            "succeeded": "✓",
            "failed": "✗",
            "dead": "✗",
        }[run.state]

    def _row_content_signature(self, run: StatusRun, icon: str) -> tuple[object, ...]:
        return (
            icon,
            run.project,
            run.source_agent,
            run.state,
            run.phase,
            run.summary,
            run.error,
            run.session_id,
            "compact"
            if self.has_class("compact")
            else "wide"
            if self.has_class("wide")
            else "stacked",
            self.no_color,
        )

    def _visible_run_groups(self) -> tuple[dict[str, tuple[StatusRun, ...]], bool]:
        assert self.snapshot is not None
        rendered_count = 0
        truncated = self.snapshot.has_more
        groups: dict[str, tuple[StatusRun, ...]] = {}
        for group, runs in (
            ("active", self.snapshot.active),
            ("attention", self.snapshot.attention),
            ("recent", self.snapshot.recent),
        ):
            visible: list[StatusRun] = []
            for run in (item for item in runs if item.kind != "compile"):
                if rendered_count >= MAX_RENDERED_RUNS:
                    truncated = True
                    break
                visible.append(run)
                rendered_count += 1
            groups[group] = tuple(visible)
        return groups, truncated

    async def _reconcile_run_rows(
        self, groups: Mapping[str, tuple[StatusRun, ...]]
    ) -> None:
        desired_groups = {
            run.id: group for group, runs in groups.items() for run in runs
        }
        for run_id, row in tuple(self._run_rows.items()):
            if (
                run_id not in desired_groups
                or self._run_groups.get(run_id) != desired_groups[run_id]
            ):
                await row.remove()
                self._run_rows.pop(run_id, None)
                self._run_groups.pop(run_id, None)

        for group, runs in groups.items():
            container = self.query_one(f"#{group}-runs", Vertical)
            for run in runs:
                icon = self._run_icon(run, self._spinner_frame)
                signature = self._row_content_signature(run, icon)
                selected = run.id == self.selected_run_id
                row = self._run_rows.get(run.id)
                if row is None:
                    row = RunRow(
                        run,
                        self._row_text(run, icon),
                        widget_id=self._row_id(run.id),
                        content_signature=signature,
                        selected=selected,
                    )
                    await container.mount(row)
                    self._run_rows[run.id] = row
                    self._run_groups[run.id] = group
                else:
                    row.update_view(
                        run,
                        self._row_text(run, icon),
                        content_signature=signature,
                        selected=selected,
                    )

            desired_rows = [self._run_rows[run.id] for run in runs]
            for index, row in enumerate(desired_rows):
                current = list(container.children)
                if current[index] is not row:
                    container.move_child(row, before=current[index])

    async def _render_snapshot_unlocked(self) -> None:
        assert self.snapshot is not None
        active_count = (
            len(self.snapshot.active)
            if self.snapshot.active_count is None
            else self.snapshot.active_count
        )
        attention_count = (
            len(self.snapshot.attention)
            if self.snapshot.attention_count is None
            else self.snapshot.attention_count
        )
        self.query_one("#app-header", Static).update(
            Text(
                f"AI Memory  ● watching    {active_count} active   "
                f"{attention_count} needs attention"
            )
        )
        alerts = self.snapshot.health_alerts
        health = self.query_one("#health-banner", Static)
        visible_alerts = alerts[:MAX_VISIBLE_HEALTH_ALERTS]
        health_lines = [alert.message for alert in visible_alerts]
        hidden_alerts = len(alerts) - len(visible_alerts)
        if hidden_alerts:
            health_lines.append(f"+{hidden_alerts} more")
        health.update(Text("\n".join(health_lines)))
        health.display = bool(alerts)
        run_list = self.query_one("#run-list", VerticalScroll)
        scroll_y = run_list.scroll_y
        groups, truncated = self._visible_run_groups()
        await self._reconcile_run_rows(groups)
        notice = self.query_one("#truncation-notice", Static)
        if notice.display != truncated:
            notice.display = truncated
        run_list.scroll_to(y=scroll_y, animate=False)
        compile_status = self.snapshot.compile
        self.query_one("#compile-panel", Static).update(
            Text(f"End-of-day compile  {compile_status.state}: {compile_status.summary}")
        )
        self.query_one("#compile-panel", Static).set_class(
            compile_status.run is not None, "selectable"
        )
        self.query_one("#compile-panel", Static).set_class(
            compile_status.run is not None
            and compile_status.run.id == self.selected_run_id,
            "selected",
        )
        await self._render_details()

    def _row_text(self, run: StatusRun, icon: str) -> Text:
        semantic_style = None if self.no_color else {
            "queued": "dim",
            "running": "cyan",
            "retrying": "yellow",
            "succeeded": "green",
            "failed": "red",
            "dead": "red",
        }[run.state]
        result = run.summary or run.error or "—"
        row = Text()
        row.append(icon, style=semantic_style)
        row.append(f" {run.project}")
        if not self.has_class("compact"):
            source = _SOURCE_LABELS.get(run.source_agent, run.source_agent.title())
            row.append(f" {source}")
        row.append(" ")
        row.append(run.state, style=semantic_style)
        row.append(f" {run.phase} result={result}")
        if self.has_class("wide"):
            row.append(f" session={run.session_id}")
        return row

    async def _render_details(self) -> None:
        target = self.query_one("#details", Static)
        if self.selected_run_id is None:
            target.update(Text("No runs"))
            self._details_run_id = None
            return
        selected_run_id = self.selected_run_id
        self._detail_generation += 1
        generation = self._detail_generation
        try:
            details = await asyncio.to_thread(
                self._details_reader, self.queue_path, selected_run_id
            )
        except (KeyError, OSError, StatusReadError, ValueError):
            if generation != self._detail_generation or selected_run_id != self.selected_run_id:
                return
            target.update(Text("Details unavailable"))
            return
        if generation != self._detail_generation or selected_run_id != self.selected_run_id:
            return
        run = details.run
        lines = [
            f"{run.source_agent} / {run.project}",
            f"Run {run.id} · {run.kind}",
            f"State: {run.state} · Phase: {run.phase}",
            f"Session: {run.session_id}",
            f"Started: {run.started_at.isoformat()}",
        ]
        if run.error:
            lines.append(f"Error: {run.error}")
        for event in details.events:
            provider = f" · {event.provider}" if event.provider else ""
            message = f" · {event.message}" if event.message else ""
            detail_text = (
                " · " + ", ".join(f"{key}={value}" for key, value in event.details.items())
                if event.details
                else ""
            )
            lines.append(
                f"{event.created_at:%H:%M:%S}  {event.phase}{provider}{message}{detail_text}"
            )
        for attempt in details.provider_attempts:
            lines.append(
                f"Attempt {attempt.provider}: {attempt.outcome} · {attempt.elapsed_ms}ms"
            )
        if not details.timeline_available:
            lines.append("Fine-grained timeline unavailable")
        target.update(Text("\n".join(lines)))
        self._details_run_id = selected_run_id

    async def action_refresh(self) -> None:
        await self.refresh_snapshot()

    def _move_selection(self, delta: int) -> None:
        runs = self._all_runs()
        if not runs:
            return
        ids = [run.id for run in runs]
        index = ids.index(self.selected_run_id) if self.selected_run_id in ids else 0
        self.selected_run_id = ids[(index + delta) % len(ids)]
        self.preferred_run_id = self.selected_run_id
        self._update_selection_classes()
        self.call_later(self._render_details)
        selector = f"#{self._row_id(self.selected_run_id)}"
        rows = self.query(selector)
        if rows:
            rows.first().scroll_visible(animate=False)

    def _update_selection_classes(self) -> None:
        for row in self.query(".run-row"):
            if isinstance(row, RunRow):
                row.set_selected(row.run_id == self.selected_run_id)
        if self.snapshot is not None:
            compile_run = self.snapshot.compile.run
            self.query_one("#compile-panel", Static).set_class(
                compile_run is not None and compile_run.id == self.selected_run_id,
                "selected",
            )

    def action_next_run(self) -> None:
        self._move_selection(1)

    def action_previous_run(self) -> None:
        self._move_selection(-1)

    async def action_details(self) -> None:
        if not self.has_class("compact") or self.selected_run_id is None:
            return
        selected = self.selected_run_id
        await self._render_details()
        if selected != self.selected_run_id or self._details_run_id != selected:
            return
        details = self.query_one("#details", Static).render()
        overlay = DetailsOverlay(Text(str(details)))
        if self.no_color:
            overlay.add_class("nocolor")
        self.push_screen(overlay)

    def action_filter(self) -> None:
        field = self.query_one("#filter-input", Input)
        field.display = True
        field.disabled = False
        field.value = self.filter_query
        field.focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "filter-input":
            return
        self.filter_query = event.value.strip()
        event.input.display = False
        event.input.disabled = True
        await self.refresh_snapshot()

    async def action_escape(self) -> None:
        if isinstance(self.screen, DetailsOverlay):
            self.pop_screen()
            return
        field = self.query_one("#filter-input", Input)
        field.display = False
        field.disabled = True
        if self.filter_query:
            self.filter_query = ""
            await self.refresh_snapshot()

    async def action_acknowledge(self) -> None:
        if self.snapshot is None or self.selected_run_id is None:
            return
        if self.selected_run_id not in {run.id for run in self.snapshot.attention}:
            return
        try:
            self.observer_state = await asyncio.to_thread(
                self._acknowledger, self.observer_path, self.selected_run_id
            )
        except (OSError, PermissionError, sqlite3.Error, ValueError) as error:
            diagnostic = self.query_one("#diagnostic", Static)
            diagnostic.update(
                Text(f"Could not acknowledge failure: {_diagnostic_text(error)}")
            )
            diagnostic.display = True
            return
        await self.refresh_snapshot()

    def action_help(self) -> None:
        self.notify("↑/↓ or j/k select · Enter details · / filter · a acknowledge · q quit")


def run_dashboard(
    *,
    no_color: bool,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run the default read-only dashboard and return after a clean exit."""
    source_env = os.environ if env is None else env
    config = load_config(source_env)
    StatusDashboard(
        config.queue_path,
        memory_home=config.root_dir,
        snapshot_reader=read_snapshot,
        details_reader=read_run_details,
        observer_loader=load_observer_state,
        acknowledger=acknowledge_run,
        no_color=no_color or "NO_COLOR" in source_env,
    ).run()
    return 0
