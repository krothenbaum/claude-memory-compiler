from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

from capture import capture_transcript, launch_worker
from providers import ProviderResult, RoutedResult, TaskKind
from scripts.queue import QueueRepository
from worker import MemoryWorker, SingletonDrainLock


NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def write_claude_transcript(path: Path) -> None:
    path.write_text(
        json.dumps({"sessionId": "session-1", "cwd": "/tmp/project", "message": {
            "role": "user", "content": "Remember the queue"
        }}) + "\n",
        encoding="utf-8",
    )


def test_capture_snapshots_parses_enqueues_and_launches(tmp_path):
    source = tmp_path / "outside.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"
    (home / "scripts").mkdir(parents=True)
    launched = []

    result = capture_transcript(
        source,
        source_agent="claude",
        metadata={"trigger": "session_end"},
        memory_home=home,
        launcher=lambda root: launched.append(root),
        clock=lambda: NOW,
    )

    assert result.created is True
    assert Path(result.job.source_path).parent == home / "scripts" / "spool"
    assert Path(result.job.source_path).read_bytes() == source.read_bytes()
    assert launched == [home]


def test_capture_never_uses_session_id_as_a_spool_path(tmp_path):
    source = tmp_path / "outside.jsonl"
    source.write_text(
        json.dumps({"sessionId": "../../escape", "message": {
            "role": "user", "content": "Keep the snapshot private"
        }}) + "\n",
        encoding="utf-8",
    )
    home = tmp_path / "memory"
    (home / "scripts").mkdir(parents=True)

    result = capture_transcript(
        source,
        source_agent="claude",
        metadata={"trigger": "session_end"},
        memory_home=home,
        launcher=lambda _: None,
        clock=lambda: NOW,
    )

    snapshot = Path(result.job.source_path)
    assert snapshot.parent == home / "scripts" / "spool"
    assert snapshot.name.startswith("claude-")
    assert "escape" not in snapshot.name


def test_launch_worker_is_detached_with_closed_standard_streams(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    launch_worker(tmp_path)

    command, options = calls[0]
    assert command == [
        "uv", "run", "--directory", str(tmp_path), "python",
        str(tmp_path / "scripts" / "worker.py"), "--drain",
    ]
    assert options["stdin"] is subprocess.DEVNULL
    assert options["stdout"] is subprocess.DEVNULL
    assert options["stderr"] is subprocess.DEVNULL
    assert options["close_fds"] is True
    assert options.get("start_new_session") is True or options.get("creationflags", 0)


def test_singleton_drain_lock_rejects_a_second_healthy_owner(tmp_path):
    path = tmp_path / "memory-worker.lock"
    first = SingletonDrainLock(path)
    second = SingletonDrainLock(path)

    assert first.acquire() is True
    assert second.acquire() is False
    first.release()
    assert second.acquire() is True
    second.release()


class FakeRouter:
    def __init__(self, result):
        self.result = result
        self.requests = []

    async def generate_text(self, request):
        self.requests.append(request)
        return self.result


def routed(*attempts):
    final = attempts[-1]
    return RoutedResult.from_result(
        final,
        attempts,
        "codex:capacity:full" if len(attempts) == 2 else None,
    )


def enqueue_capture_job(repository, tmp_path):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    return capture_transcript(
        source,
        source_agent="claude",
        metadata={"trigger": "session_end"},
        memory_home=tmp_path,
        queue=repository,
        launcher=lambda _: None,
        clock=lambda: NOW,
    )


def test_worker_dispatches_success_and_records_every_attempt(tmp_path):
    with QueueRepository(tmp_path / "jobs.sqlite3", clock=lambda: NOW) as repository:
        queued = enqueue_capture_job(repository, tmp_path)
        codex = ProviderResult("codex", "luna", TaskKind.EXTRACT, "capacity", reason="full")
        claude = ProviderResult("claude", "sonnet", TaskKind.EXTRACT, "success", text="daily")
        writes = []
        worker = MemoryWorker(
            repository,
            FakeRouter(routed(codex, claude)),
            daily_writer=lambda job, text: writes.append((job.id, text)),
            clock=lambda: NOW,
            owner="worker",
            sleeper=lambda _: asyncio.sleep(0),
        )

        assert asyncio.run(worker.drain()) == 1
        assert writes == [(queued.job_id, "daily")]
        assert repository.get_job(queued.job_id).status == "succeeded"
        assert [a.provider for a in repository.attempts_for(queued.job_id)] == ["codex", "claude"]


def test_worker_retries_after_both_providers_fail_and_retains_spool(tmp_path):
    with QueueRepository(tmp_path / "jobs.sqlite3", clock=lambda: NOW) as repository:
        queued = enqueue_capture_job(repository, tmp_path)
        spool = Path(queued.job.source_path)
        codex = ProviderResult("codex", "luna", TaskKind.EXTRACT, "capacity", reason="full")
        claude = ProviderResult("claude", "sonnet", TaskKind.EXTRACT, "timeout", reason="slow")
        worker = MemoryWorker(
            repository,
            FakeRouter(routed(codex, claude)),
            daily_writer=lambda *_: (_ for _ in ()).throw(AssertionError("must not write")),
            clock=lambda: NOW,
            owner="worker",
            jitter=lambda: 0,
            sleeper=lambda _: asyncio.sleep(0),
        )

        assert asyncio.run(worker.drain()) == 1
        failed = repository.get_job(queued.job_id)
        assert failed.status == "failed"
        assert failed.available_at == NOW + timedelta(seconds=5)
        assert "claude:timeout:slow" in failed.last_error
        assert spool.exists()


def test_worker_exits_cleanly_when_another_drain_owns_lock(tmp_path):
    lock_path = tmp_path / "memory-worker.lock"
    held = SingletonDrainLock(lock_path)
    assert held.acquire()
    try:
        with QueueRepository(tmp_path / "jobs.sqlite3", clock=lambda: NOW) as repository:
            worker = MemoryWorker(
                repository,
                FakeRouter(None),
                daily_writer=lambda *_: None,
                clock=lambda: NOW,
                lock_path=lock_path,
            )
            assert asyncio.run(worker.run_drain()) == 0
    finally:
        held.release()
