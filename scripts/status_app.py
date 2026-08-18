"""Read-only Textual dashboard for durable memory job status."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, Static

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

SnapshotReader = Callable[..., StatusSnapshot]
DetailsReader = Callable[[Path, int], RunDetails]


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


class StatusDashboard(App[None]):
    """Triage dashboard that observes, but never controls, execution."""

    CSS = """
    Screen { layout: vertical; }
    #app-header { height: 3; padding: 0 1; text-style: bold; color: ansi_bright_cyan; }
    #health-banner, #delayed-banner, #diagnostic { height: auto; padding: 0 1; }
    #health-banner { color: yellow; }
    #delayed-banner { color: yellow; display: none; }
    #diagnostic { color: red; display: none; }
    #filter-input { display: none; height: 3; }
    #main-grid { height: 1fr; }
    #runs-pane, #details-pane { border: round $panel; padding: 0 1; }
    #runs-pane { width: 1fr; }
    #details-pane { width: 1fr; }
    #run-list { height: 1fr; }
    .section-label { color: $text-muted; text-style: bold; margin-top: 1; }
    .run-row { height: 1; padding: 0 1; }
    .run-row.selected { text-style: bold reverse; }
    .state-running { color: cyan; }
    .state-succeeded { color: green; }
    .state-retrying { color: yellow; }
    .state-failed, .state-dead { color: red; }
    .state-queued { color: $text-muted; }
    #compile-panel { height: 3; border: round $panel; padding: 0 1; }
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
        health_loader: Callable[[], tuple[HealthAlert, ...]] = tuple,
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
        self._health_loader = health_loader
        self.no_color = bool(os.environ.get("NO_COLOR")) if no_color is None else no_color
        self.observer_path = observer_state_path(self.memory_home)
        try:
            self.observer_state = observer_loader(self.observer_path)
        except (OSError, ValueError):
            self.observer_state = ObserverState.empty()
        self.snapshot: StatusSnapshot | None = None
        self.selected_run_id: int | None = None
        self.filter_query = ""

    def compose(self) -> ComposeResult:
        yield Static(id="app-header")
        yield Static(id="health-banner")
        yield Static("Status refresh delayed", id="delayed-banner")
        yield Static(id="diagnostic")
        yield Input(
            placeholder="Filter status history", id="filter-input", disabled=True
        )
        with Horizontal(id="main-grid"):
            with Vertical(id="runs-pane"):
                yield VerticalScroll(id="run-list")
            with Vertical(id="details-pane"):
                yield Static("Select a run", id="details")
        yield Static(id="compile-panel")
        yield Footer(id="status-footer")

    async def on_mount(self) -> None:
        if self.no_color:
            self.add_class("nocolor")
        self._set_layout(self.size.width)
        await self.refresh_snapshot()
        self.set_interval(1.0, self.refresh_snapshot)

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
        return (*self.snapshot.active, *self.snapshot.attention, *self.snapshot.recent)

    def _choose_selection(self) -> None:
        runs = self._all_runs()
        if self.selected_run_id in {run.id for run in runs}:
            return
        self.selected_run_id = runs[0].id if runs else None

    async def refresh_snapshot(self) -> None:
        try:
            result = self._snapshot_reader(
                self.queue_path,
                now=self._clock(),
                observer_state=self.observer_state,
                query=self.filter_query,
                health_alerts=self._health_loader(),
                memory_home=self.memory_home,
            )
        except sqlite3.OperationalError:
            if self.snapshot is not None:
                self.query_one("#delayed-banner", Static).display = True
            return
        except (StatusReadError, OSError, ValueError) as error:
            diagnostic = self.query_one("#diagnostic", Static)
            path = getattr(error, "path", self.queue_path)
            diagnostic.update(Text(f"Status unavailable: {path} — {error}"))
            diagnostic.display = True
            return
        self.snapshot = result
        self.query_one("#delayed-banner", Static).display = False
        self.query_one("#diagnostic", Static).display = False
        self._choose_selection()
        await self._render_snapshot()

    async def _render_snapshot(self) -> None:
        assert self.snapshot is not None
        self.query_one("#app-header", Static).update(
            Text(
                f"AI Memory  ● watching    {len(self.snapshot.active)} active   "
                f"{len(self.snapshot.attention)} needs attention"
            )
        )
        alerts = self.snapshot.health_alerts
        health = self.query_one("#health-banner", Static)
        health.update(Text(" · ".join(alert.message for alert in alerts)))
        health.display = bool(alerts)
        run_list = self.query_one("#run-list", VerticalScroll)
        await run_list.remove_children()
        for label, runs in (
            ("ACTIVE", self.snapshot.active),
            ("NEEDS ATTENTION", self.snapshot.attention),
            ("RECENT", self.snapshot.recent),
        ):
            await run_list.mount(Static(label, classes="section-label"))
            for run in runs:
                icon = {
                    "running": "●",
                    "queued": "◌",
                    "retrying": "↻",
                    "succeeded": "✓",
                    "failed": "✗",
                    "dead": "✗",
                }[run.state]
                row = Static(
                    Text(self._row_text(run, icon)),
                    id=f"run-{abs(run.id)}",
                    classes=f"run-row state-{run.state}",
                )
                row.run_id = run.id  # type: ignore[attr-defined]
                row.set_class(run.id == self.selected_run_id, "selected")
                await run_list.mount(row)
        compile_status = self.snapshot.compile
        self.query_one("#compile-panel", Static).update(
            Text(f"End-of-day compile  {compile_status.state}: {compile_status.summary}")
        )
        self.query_one("#compile-panel", Static).set_class(
            compile_status.run is not None, "selectable"
        )
        self._render_details()

    def _row_text(self, run: StatusRun, icon: str) -> str:
        result = run.summary or run.error or "—"
        if self.has_class("wide"):
            return (
                f"{icon} {run.project} {run.state} {run.phase} result={result} "
                f"session={run.session_id} provider=— elapsed=—"
            )
        if self.has_class("stacked"):
            return (
                f"{icon} {run.project} {run.state} {run.phase} result={result} "
                "provider=— elapsed=—"
            )
        return f"{icon} {run.project} {run.state} {run.phase} result={result}"

    def _render_details(self) -> None:
        target = self.query_one("#details", Static)
        if self.selected_run_id is None:
            target.update(Text("No runs"))
            return
        try:
            details = self._details_reader(self.queue_path, self.selected_run_id)
        except (KeyError, OSError, StatusReadError, ValueError):
            target.update(Text("Details unavailable"))
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
            lines.append(f"{event.created_at:%H:%M:%S}  {event.phase}{provider}")
        target.update(Text("\n".join(lines)))

    async def action_refresh(self) -> None:
        await self.refresh_snapshot()

    def _move_selection(self, delta: int) -> None:
        runs = self._all_runs()
        if not runs:
            return
        ids = [run.id for run in runs]
        index = ids.index(self.selected_run_id) if self.selected_run_id in ids else 0
        self.selected_run_id = ids[(index + delta) % len(ids)]
        self.call_later(self._render_snapshot)

    def action_next_run(self) -> None:
        self._move_selection(1)

    def action_previous_run(self) -> None:
        self._move_selection(-1)

    def action_details(self) -> None:
        if not self.has_class("compact") or self.selected_run_id is None:
            self._render_details()
            return
        self._render_details()
        details = self.query_one("#details", Static).render()
        self.push_screen(DetailsOverlay(Text(str(details))))

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
        self.observer_state = self._acknowledger(
            self.observer_path, self.selected_run_id
        )
        await self.refresh_snapshot()

    def action_help(self) -> None:
        self.notify("↑/↓ or j/k select · Enter details · / filter · a acknowledge · q quit")
