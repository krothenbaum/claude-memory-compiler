from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from io import StringIO
from types import SimpleNamespace
from typing import Literal

from scripts.status_app import StatusDashboard
from scripts.providers import ProviderResult, ProviderRouter, TextRequest
from scripts.queue import QueueRepository
from scripts.status_store import (
    CompileStatus,
    HealthAlert,
    ObserverState,
    StatusSnapshot,
    read_run_details,
    read_snapshot,
)
from scripts.transcripts import NormalizedSession, Turn
from scripts.worker import MemoryWorker


NOW = datetime(2026, 8, 18, 19, 0, tzinfo=UTC)


def _empty_snapshot(*, health_alerts=()) -> StatusSnapshot:
    return StatusSnapshot(
        active=(),
        attention=(),
        recent=(),
        compile=CompileStatus("before_window", "Next compile window begins at 16:00"),
        health_alerts=health_alerts,
    )


def _alert() -> HealthAlert:
    return HealthAlert(
        created_at=NOW,
        level="error",
        message="Codex hook could not enqueue capture",
        component="codex-session-end",
    )


def test_snapshot_loads_hook_health_once_and_renders_it(tmp_path, monkeypatch):
    import status as status_cli

    config = SimpleNamespace(root_dir=tmp_path, queue_path=tmp_path / "jobs.sqlite3")
    observer = ObserverState.empty()
    alert = _alert()
    health_calls = []
    projection_calls = []
    monkeypatch.setattr(status_cli, "load_config", lambda _env: config)
    monkeypatch.setattr(status_cli, "observer_state_path", lambda root: root / "view.json")
    monkeypatch.setattr(status_cli, "load_observer_state", lambda _path: observer)

    def read_health(root, *, now):
        health_calls.append((root, now))
        return (alert,)

    def read_projection(path, **kwargs):
        projection_calls.append((path, kwargs))
        return _empty_snapshot(health_alerts=kwargs["health_alerts"])

    monkeypatch.setattr(status_cli, "read_recent_hook_alerts", read_health)
    monkeypatch.setattr(status_cli, "read_snapshot", read_projection)
    output = StringIO()

    assert status_cli.main(["--snapshot"], env={}, now=NOW, output=output) == 0

    assert health_calls == [(tmp_path, NOW)]
    assert projection_calls == [
        (
            config.queue_path,
            {
                "now": NOW,
                "observer_state": observer,
                "memory_home": tmp_path,
                "health_alerts": (alert,),
            },
        )
    ]
    assert "HEALTH\n" in output.getvalue()
    assert "Codex hook could not enqueue capture" in output.getvalue()


def test_snapshot_health_failure_is_bounded_and_does_not_block_projection(
    tmp_path, monkeypatch
):
    import status as status_cli

    config = SimpleNamespace(root_dir=tmp_path, queue_path=tmp_path / "jobs.sqlite3")
    observed = []
    monkeypatch.setattr(status_cli, "load_config", lambda _env: config)
    monkeypatch.setattr(
        status_cli,
        "load_observer_state",
        lambda _path: ObserverState.empty(),
    )
    monkeypatch.setattr(
        status_cli,
        "read_recent_hook_alerts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("health unavailable")),
    )

    def read_projection(_path, **kwargs):
        observed.append(kwargs["health_alerts"])
        return _empty_snapshot(health_alerts=kwargs["health_alerts"])

    monkeypatch.setattr(status_cli, "read_snapshot", read_projection)
    output = StringIO()

    assert status_cli.main(["--snapshot"], env={}, now=NOW, output=output) == 0
    assert observed == [()]
    assert "Inspection failed" not in output.getvalue()


def test_dashboard_refresh_loads_health_and_passes_it_to_projection(tmp_path):
    alert = _alert()
    health_calls = []
    projection_calls = []

    def read_health(root, *, now):
        health_calls.append((root, now))
        return (alert,)

    def read_projection(path, **kwargs):
        projection_calls.append((path, kwargs))
        return _empty_snapshot(health_alerts=kwargs["health_alerts"])

    dashboard = StatusDashboard(
        tmp_path / "jobs.sqlite3",
        memory_home=tmp_path,
        snapshot_reader=read_projection,
        observer_loader=lambda _path: ObserverState.empty(),
        health_loader=read_health,
        clock=lambda: NOW,
    )

    async def exercise() -> None:
        async with dashboard.run_test() as pilot:
            await pilot.pause()
            assert dashboard.snapshot is not None
            assert dashboard.snapshot.health_alerts == (alert,)
            assert "Codex hook could not enqueue capture" in str(
                dashboard.query_one("#health-banner").render()
            )

    asyncio.run(exercise())

    assert health_calls == [(tmp_path, NOW)]
    assert projection_calls[0][0] == tmp_path / "jobs.sqlite3"
    assert projection_calls[0][1]["health_alerts"] == (alert,)
    assert projection_calls[0][1]["memory_home"] == tmp_path
    assert projection_calls[0][1]["max_runs"] == 200


def test_dashboard_health_failure_is_bounded_and_refreshes_without_alerts(tmp_path):
    observed = []

    def read_projection(_path, **kwargs):
        observed.append(kwargs["health_alerts"])
        return _empty_snapshot(health_alerts=kwargs["health_alerts"])

    dashboard = StatusDashboard(
        tmp_path / "jobs.sqlite3",
        memory_home=tmp_path,
        snapshot_reader=read_projection,
        observer_loader=lambda _path: ObserverState.empty(),
        health_loader=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("invalid health data")
        ),
        clock=lambda: NOW,
    )

    async def exercise() -> None:
        async with dashboard.run_test() as pilot:
            await pilot.pause()
            assert dashboard.snapshot is not None

    asyncio.run(exercise())
    assert observed == [()]


def test_public_dashboard_runner_composes_all_read_only_dependencies(
    tmp_path, monkeypatch
):
    import scripts.status_app as status_app

    queue_path = tmp_path / "scripts" / "jobs.sqlite3"
    config = SimpleNamespace(root_dir=tmp_path, queue_path=queue_path)
    constructed = []

    class Dashboard:
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))

        def run(self):
            return None

    monkeypatch.setattr(status_app, "load_config", lambda _env: config)
    monkeypatch.setattr(status_app, "StatusDashboard", Dashboard)

    assert status_app.run_dashboard(no_color=True) == 0
    assert constructed == [
        (
            (queue_path,),
            {
                "memory_home": tmp_path,
                "snapshot_reader": status_app.read_snapshot,
                "details_reader": status_app.read_run_details,
                "observer_loader": status_app.load_observer_state,
                "acknowledger": status_app.acknowledge_run,
                "health_loader": status_app.read_recent_hook_alerts,
                "no_color": True,
            },
        )
    ]


def test_persisted_worker_status_survives_observer_attach_detach_and_reopen(tmp_path):
    root = tmp_path / "memory"
    queue_path = root / "scripts" / "jobs.sqlite3"
    current = [NOW]
    writes = []
    live_started = asyncio.Event()
    live_release = asyncio.Event()

    class Provider:
        def __init__(self, name: Literal["codex", "claude"]):
            self.name = name
            self._model = f"test-{name}"

        async def generate_text(self, request: TextRequest) -> ProviderResult:
            if "LIVE" in request.prompt and self.name == "codex":
                live_started.set()
                await live_release.wait()
            if "FAIL" in request.prompt:
                outcome: Literal["success", "capacity", "error"] = "error"
                text, reason = "", f"{self.name} unavailable"
            elif "FALLBACK" in request.prompt and self.name == "codex":
                outcome, text, reason = "capacity", "", "test capacity"
            else:
                outcome, text, reason = "success", f"{self.name} extraction", None
            return ProviderResult(
                provider=self.name,
                model=self._model,
                task=request.task,
                outcome=outcome,
                text=text,
                reason=reason,
                elapsed_ms=5,
            )

    codex = Provider("codex")
    claude = Provider("claude")

    def router_factory(start_callback, attempt_callback):
        return ProviderRouter(
            codex,
            claude,
            attempt_start_callback=start_callback,
            attempt_callback=attempt_callback,
        )

    def session(label, agent="claude"):
        return NormalizedSession(
            agent=agent,
            session_id=f"session-{label.casefold()}",
            project=f"project-{label.casefold()}",
            cwd=str(root),
            timestamp=current[0].isoformat(),
            trigger="session_end",
            turns=(Turn("user", label), Turn("assistant", "Acknowledged")),
            source_path=str(root / f"{label.casefold()}.jsonl"),
            source_hash=f"hash-{label.casefold()}",
        )

    async def exercise() -> None:
        with QueueRepository(
            queue_path,
            max_attempts=2,
            clock=lambda: current[0],
            memory_home=root,
            sync_usage=False,
        ) as queue:
            worker = MemoryWorker(
                queue,
                status_router_factory=router_factory,
                daily_writer=lambda job, text: writes.append((job.session_id, text)),
                clock=lambda: current[0],
                owner="integration-worker",
                lock_path=root / "scripts" / "worker.lock",
                retry_base_seconds=1,
                retry_max_seconds=1,
                jitter=lambda: 0,
            )

            for label, agent in (
                ("DIRECT", "claude"),
                ("FALLBACK", "codex"),
                ("FAIL", "claude"),
            ):
                queue.enqueue_capture(session(label, agent))

            for _ in range(3):
                claimed = queue.claim_next(worker.owner, current[0], 120)
                assert claimed is not None
                await worker.process(claimed)

            current[0] = current[0].replace(second=2)
            retry = queue.claim_next(worker.owner, current[0], 120)
            assert retry is not None and retry.session_id == "session-fail"
            await worker.process(retry)

            closed_snapshot = read_snapshot(
                queue_path,
                now=current[0],
                observer_state=ObserverState.empty(),
                memory_home=root,
                max_runs=200,
            )
            recent = {run.session_id: run for run in closed_snapshot.recent}
            attention = {run.session_id: run for run in closed_snapshot.attention}
            fallback_summary = recent["session-fallback"].summary
            assert fallback_summary is not None
            assert "through Claude fallback" in fallback_summary
            assert attention["session-fail"].state == "dead"
            failure_error = attention["session-fail"].error
            assert failure_error is not None
            assert "claude:error:claude unavailable" in failure_error

            fallback_details = read_run_details(
                queue_path, recent["session-fallback"].id
            )
            assert [attempt.provider for attempt in fallback_details.provider_attempts] == [
                "codex",
                "claude",
            ]
            assert [event.phase for event in fallback_details.events] == [
                "queued",
                "worker_claimed",
                "codex_started",
                "codex_failed",
                "claude_started",
                "claude_succeeded",
                "daily_log_write_started",
                "succeeded",
            ]

            queue.enqueue_capture(session("LIVE", "codex"))
            live_job = queue.claim_next(worker.owner, current[0], 120)
            assert live_job is not None and live_job.session_id == "session-live"
            dashboard = StatusDashboard(
                queue_path,
                memory_home=root,
                observer_loader=lambda _path: ObserverState.empty(),
                health_loader=lambda *_args, **_kwargs: (),
                clock=lambda: current[0],
            )
            worker_task = None
            async with dashboard.run_test() as pilot:
                await pilot.pause()
                worker_task = asyncio.create_task(worker.process(live_job))
                await asyncio.wait_for(live_started.wait(), timeout=2)
                await dashboard.refresh_snapshot()
                live_run = next(
                    run
                    for run in dashboard.snapshot.active
                    if run.session_id == "session-live"
                )
                assert live_run.phase == "codex_started"

            assert worker_task is not None and not worker_task.done()
            live_release.set()
            assert await worker_task is True

            reopened = read_snapshot(
                queue_path,
                now=current[0],
                observer_state=ObserverState.empty(),
                memory_home=root,
                max_runs=200,
            )
            final_live = next(
                run for run in reopened.recent if run.session_id == "session-live"
            )
            assert final_live.state == "succeeded"
            assert final_live.summary == "Saved 16 characters"

    asyncio.run(exercise())
    assert writes == [
        ("session-direct", "codex extraction"),
        ("session-fallback", "claude extraction"),
        ("session-live", "codex extraction"),
    ]
