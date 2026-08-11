from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows exercises the msvcrt branch.
    fcntl = None
import json
from pathlib import Path
import sqlite3
import stat
import subprocess
import threading
import time

import pytest

import capture as capture_module
from capture import capture_transcript, enqueue_hook_input, launch_worker
from providers import ProviderResult, ProviderRouter, RoutedResult, TaskKind
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


@pytest.mark.parametrize(
    ("guard", "reason"),
    [
        ({"AI_MEMORY_INTERNAL_JOB": "1"}, "internal_job"),
        ({"CLAUDE_INVOKED_BY": ""}, "legacy_internal_job"),
    ],
)
def test_capture_guard_skips_before_transcript_or_runtime_work(tmp_path, guard, reason):
    home = tmp_path / "must-not-exist"

    result = capture_transcript(
        tmp_path / "missing-transcript.jsonl",
        source_agent="claude",
        metadata={},
        memory_home=home,
        launcher=lambda _: (_ for _ in ()).throw(AssertionError("must not launch")),
        env=guard,
    )

    assert result.status == "skipped"
    assert result.reason == reason
    assert result.created is False
    assert result.job is None
    assert result.job_id is None
    assert not home.exists()


def test_hook_capture_guard_precedes_payload_resolution(tmp_path):
    result = enqueue_hook_input(
        {},
        source_agent="claude",
        trigger="session_end",
        memory_home=tmp_path / "must-not-exist",
        env={"AI_MEMORY_INTERNAL_JOB": "1"},
    )

    assert result.status == "skipped"
    assert result.job is None
    assert not (tmp_path / "must-not-exist").exists()


def test_codex_hook_capture_falls_back_to_transcript_session_metadata(tmp_path):
    source = tmp_path / "codex.jsonl"
    source.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-11T10:15:00Z",
                "type": "session_meta",
                "payload": {"id": "meta-session", "cwd": "/projects/meta-project"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Remember this"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = enqueue_hook_input(
        {
            "transcript_path": str(source),
            "reason": "user_exit",
            "source": "cli",
        },
        source_agent="codex",
        trigger="session_end",
        memory_home=tmp_path / "memory",
        launcher=lambda _: None,
        env={},
        clock=lambda: NOW,
    )

    assert result.job.session_id == "meta-session"
    assert result.job.cwd == "/projects/meta-project"
    assert result.job.project == "meta-project"


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


def test_parser_failure_retains_private_safe_snapshot_without_queue_or_launch(tmp_path):
    source = tmp_path / "sensitive-name.jsonl"
    source.write_bytes(b'\xff{"private":"content"}\n')
    home = tmp_path / "memory"
    launched = []

    with pytest.raises(UnicodeDecodeError):
        capture_transcript(
            source,
            source_agent="claude",
            metadata={"trigger": "session_end"},
            memory_home=home,
            launcher=lambda root: launched.append(root),
            env={},
        )

    retained = list((home / "scripts" / "spool").iterdir())
    assert len(retained) == 1
    assert retained[0].name.startswith("failed-claude-")
    assert "sensitive" not in retained[0].name
    assert retained[0].read_bytes() == source.read_bytes()
    assert stat.S_IMODE(retained[0].stat().st_mode) == 0o600
    assert not (home / "scripts" / "jobs.sqlite3").exists()
    assert launched == []


def test_capture_fails_closed_under_database_contention_before_hook_deadline(tmp_path):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"
    database = home / "scripts" / "jobs.sqlite3"
    with QueueRepository(database, clock=lambda: NOW):
        pass
    locker = sqlite3.connect(database, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    launched = []
    started = time.monotonic()
    try:
        with pytest.raises(sqlite3.OperationalError):
            capture_transcript(
                source,
                source_agent="claude",
                metadata={"trigger": "session_end"},
                memory_home=home,
                launcher=lambda root: launched.append(root),
                env={},
            )
    finally:
        elapsed = time.monotonic() - started
        locker.execute("ROLLBACK")
        locker.close()

    assert elapsed < 1.0
    assert launched == []
    assert list((home / "scripts" / "spool").glob("*.jsonl"))


def test_capture_rejects_symlinked_spool_component(tmp_path):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"
    outside = tmp_path / "outside"
    outside.mkdir()
    home.mkdir()
    (home / "scripts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        capture_transcript(
            source,
            source_agent="claude",
            metadata={"trigger": "session_end"},
            memory_home=home,
            launcher=lambda _: (_ for _ in ()).throw(AssertionError("must not launch")),
            env={},
        )

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("existing_kind", ["corrupt", "symlink"])
def test_capture_rejects_unsafe_existing_deterministic_snapshot(tmp_path, existing_kind):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"
    spool = home / "scripts" / "spool"
    spool.mkdir(parents=True)
    normalized = capture_module.parse_claude_transcript(
        source, {"trigger": "session_end"}
    )
    destination = spool / f"claude-{normalized.source_hash}.jsonl"
    attacker = tmp_path / "attacker.jsonl"
    attacker.write_bytes(b"attacker")
    if existing_kind == "symlink":
        destination.symlink_to(attacker)
    else:
        destination.write_bytes(b"attacker")

    with pytest.raises(ValueError, match="snapshot"):
        capture_transcript(
            source,
            source_agent="claude",
            metadata={"trigger": "session_end"},
            memory_home=home,
            launcher=lambda _: (_ for _ in ()).throw(AssertionError("must not launch")),
            env={},
        )

    assert attacker.read_bytes() == b"attacker"
    assert not (home / "scripts" / "jobs.sqlite3").exists()


def test_snapshot_is_fsynced_before_directory_and_database_enqueue(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"
    events = []

    monkeypatch.setattr(
        capture_module, "_fsync_file", lambda path: events.append(("file", Path(path)))
    )
    monkeypatch.setattr(
        capture_module,
        "_fsync_directory",
        lambda path: events.append(("directory", Path(path))),
    )

    class RecordingQueue:
        def enqueue_capture(self, normalized):
            events.append(("enqueue", Path(normalized.source_path)))
            return type("Result", (), {"created": True, "job": type(
                "Job", (), {"id": 1, "source_path": normalized.source_path}
            )()})()

    capture_transcript(
        source,
        source_agent="claude",
        metadata={"trigger": "session_end"},
        memory_home=home,
        queue=RecordingQueue(),
        launcher=lambda _: None,
        env={},
    )

    names = [event[0] for event in events]
    assert names.index("file") < names.index("directory") < names.index("enqueue")


def test_equivalent_concurrent_snapshot_link_settles_without_false_rejection(
    tmp_path, monkeypatch
):
    spool = tmp_path / "spool"
    spool.mkdir(mode=0o700)
    publisher_temporary = spool / "publisher.jsonl"
    publisher_temporary.write_bytes(b"same capture")
    publisher_temporary.chmod(0o600)
    destination = spool / "claude-hash.jsonl"
    destination.hardlink_to(publisher_temporary)
    contender_temporary = spool / "contender.jsonl"
    contender_temporary.write_bytes(b"same capture")
    contender_temporary.chmod(0o600)
    waits = []

    def finish_publisher(_seconds):
        waits.append(True)
        publisher_temporary.unlink()

    monkeypatch.setattr(capture_module, "_snapshot_retry_wait", finish_publisher)
    capture_module._publish_snapshot(contender_temporary, destination)

    assert waits == [True]
    assert destination.read_bytes() == b"same capture"
    assert destination.stat().st_nlink == 1
    assert not contender_temporary.exists()


def test_persistent_snapshot_hard_link_is_rejected_after_bounded_handshake(
    tmp_path, monkeypatch
):
    spool = tmp_path / "spool"
    spool.mkdir(mode=0o700)
    linked = spool / "linked.jsonl"
    linked.write_bytes(b"same capture")
    linked.chmod(0o600)
    destination = spool / "claude-hash.jsonl"
    destination.hardlink_to(linked)
    contender = spool / "contender.jsonl"
    contender.write_bytes(b"same capture")
    contender.chmod(0o600)
    waits = []
    monkeypatch.setattr(
        capture_module, "_snapshot_retry_wait", lambda seconds: waits.append(seconds)
    )

    with pytest.raises(ValueError, match="identity"):
        capture_module._publish_snapshot(contender, destination)

    assert waits
    assert contender.exists()


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


def test_singleton_lock_respects_os_lock_even_with_incomplete_metadata(tmp_path):
    if fcntl is None:
        pytest.skip("POSIX advisory-lock adversary")
    path = tmp_path / "memory-worker.lock"
    with path.open("w+") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        contender = SingletonDrainLock(path)
        assert contender.acquire() is False
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)


def test_singleton_lock_ignores_stale_pid_text_without_an_os_lock(tmp_path):
    path = tmp_path / "memory-worker.lock"
    path.write_text(f"{__import__('os').getpid()}:old-token", encoding="utf-8")
    lock = SingletonDrainLock(path)
    assert lock.acquire() is True
    lock.release()


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

        claimed = repository.claim_next("worker", NOW, worker.lease_seconds)
        assert claimed is not None
        asyncio.run(worker.process(claimed))
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


def test_worker_waits_for_future_retry_without_real_sleep(tmp_path):
    current = [NOW]
    sleeps = []

    async def advance(seconds):
        sleeps.append(seconds)
        current[0] += timedelta(seconds=seconds)

    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: current[0], max_attempts=2
    ) as repository:
        queued = enqueue_capture_job(repository, tmp_path)
        failure = ProviderResult(
            "codex", "luna", TaskKind.EXTRACT, "capacity", reason="full"
        )
        worker = MemoryWorker(
            repository,
            FakeRouter(routed(failure)),
            daily_writer=lambda *_: None,
            clock=lambda: current[0],
            owner="worker",
            jitter=lambda: 0,
            sleeper=advance,
        )

        assert asyncio.run(worker.drain()) == 2
        assert repository.get_job(queued.job_id).status == "dead"
        assert sleeps == [1.0] * 5


def test_idle_release_wins_before_enqueue_and_new_worker_can_take_lock(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    lock_path = tmp_path / "memory-worker.lock"
    owner_lock = SingletonDrainLock(lock_path)
    assert owner_lock.acquire()
    enqueue_started = threading.Event()
    enqueue_completed = threading.Event()

    def enqueue_after_release():
        enqueue_started.set()
        with QueueRepository(path, clock=lambda: NOW) as contender:
            contender.enqueue_capture(
                capture_module.parse_claude_transcript(
                    tmp_path / "source.jsonl", {"trigger": "session_end"}
                )
            )
        enqueue_completed.set()

    write_claude_transcript(tmp_path / "source.jsonl")
    with QueueRepository(path, clock=lambda: NOW) as owner_queue, ThreadPoolExecutor(
        max_workers=1
    ) as executor:
        future = None

        def release_while_serialized():
            nonlocal future
            owner_lock.release()
            future = executor.submit(enqueue_after_release)
            assert enqueue_started.wait(timeout=1)
            assert not enqueue_completed.is_set()

        assert owner_queue.release_worker_lock_if_idle(release_while_serialized) is True
        future.result(timeout=2)

    successor = SingletonDrainLock(lock_path)
    assert successor.acquire() is True
    successor.release()


def test_enqueue_commit_wins_before_idle_check_and_owner_keeps_lock(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    normalized = capture_module.parse_claude_transcript(
        source, {"trigger": "session_end"}
    )
    lock_path = tmp_path / "memory-worker.lock"
    owner_lock = SingletonDrainLock(lock_path)
    assert owner_lock.acquire()

    with QueueRepository(path, clock=lambda: NOW) as owner_queue, QueueRepository(
        path, clock=lambda: NOW
    ) as enqueuer:
        enqueuer._connection.execute("BEGIN IMMEDIATE")
        enqueuer.enqueue_capture(normalized)
        enqueuer._connection.execute("COMMIT")
        assert owner_queue.release_worker_lock_if_idle(owner_lock.release) is False

    contender = SingletonDrainLock(lock_path)
    assert contender.acquire() is False
    owner_lock.release()


def test_worker_schedules_unexpired_crashed_lease_then_recovers_it(tmp_path):
    current = [NOW]
    sleeps = []

    async def advance(seconds):
        sleeps.append(seconds)
        current[0] += timedelta(seconds=seconds)

    with QueueRepository(tmp_path / "jobs.sqlite3", clock=lambda: current[0]) as repository:
        queued = enqueue_capture_job(repository, tmp_path)
        repository.claim_next("crashed-worker", NOW, 5)
        success = ProviderResult(
            "codex", "luna", TaskKind.EXTRACT, "success", text="daily"
        )
        worker = MemoryWorker(
            repository,
            FakeRouter(routed(success)),
            daily_writer=lambda *_: None,
            clock=lambda: current[0],
            owner="recovery-worker",
            sleeper=advance,
        )

        assert asyncio.run(worker.drain()) == 1
        assert sleeps == [1.0] * 5
        assert repository.get_job(queued.job_id).status == "succeeded"


@pytest.mark.parametrize("renewal", [False, RuntimeError("database unavailable")])
def test_heartbeat_lease_loss_cancels_provider_without_stale_transition(tmp_path, renewal):
    cancelled = []

    class HangingRouter:
        async def generate_text(self, _request):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

    async def no_wait(_seconds):
        await asyncio.sleep(0)

    with QueueRepository(tmp_path / "jobs.sqlite3", clock=lambda: NOW) as repository:
        queued = enqueue_capture_job(repository, tmp_path)
        job = repository.claim_next("worker", NOW, 30)

        def renew(*_args):
            if isinstance(renewal, Exception):
                raise renewal
            return renewal

        repository.renew = renew
        worker = MemoryWorker(
            repository,
            HangingRouter(),
            daily_writer=lambda *_: (_ for _ in ()).throw(AssertionError("must not write")),
            clock=lambda: NOW,
            owner="worker",
            lease_seconds=30,
            sleeper=no_wait,
            heartbeat_sleeper=no_wait,
        )
        asyncio.run(asyncio.wait_for(worker.process(job), timeout=0.2))

        assert cancelled == [True]
        assert repository.get_job(queued.job_id).status == "leased"


def test_default_router_factory_persists_codex_before_hanging_fallback_is_cancelled(tmp_path):
    codex_attempt = ProviderResult(
        "codex", "luna", TaskKind.EXTRACT, "capacity", reason="full"
    )

    class Codex:
        async def generate_text(self, _request):
            return codex_attempt

    class HangingClaude:
        async def generate_text(self, _request):
            await asyncio.Event().wait()

    async def no_wait(_seconds):
        await asyncio.sleep(0)

    with QueueRepository(tmp_path / "jobs.sqlite3", clock=lambda: NOW) as repository:
        queued = enqueue_capture_job(repository, tmp_path)
        job = repository.claim_next("worker", NOW, 30)
        repository.renew = lambda *_: False
        worker = MemoryWorker(
            repository,
            router_factory=lambda callback: ProviderRouter(
                Codex(), HangingClaude(), attempt_callback=callback
            ),
            daily_writer=lambda *_: None,
            clock=lambda: NOW,
            owner="worker",
            lease_seconds=30,
            sleeper=no_wait,
            heartbeat_sleeper=no_wait,
        )

        asyncio.run(asyncio.wait_for(worker.process(job), timeout=0.2))

        attempts = repository.attempts_for(queued.job_id)
        assert [(attempt.provider, attempt.outcome) for attempt in attempts] == [
            ("codex", "capacity")
        ]


def test_heartbeat_lease_loss_cancels_writer_before_mutation(tmp_path):
    renewal_results = iter([True, False])
    writer_events = []
    heartbeat_sleeps = []

    async def blocking_writer(_job, _text):
        writer_events.append("started")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            writer_events.append("cancelled")
            raise

    async def no_wait(_seconds):
        heartbeat_sleeps.append(True)
        if len(heartbeat_sleeps) == 1:
            await asyncio.Event().wait()
        await asyncio.sleep(0)

    with QueueRepository(tmp_path / "jobs.sqlite3", clock=lambda: NOW) as repository:
        queued = enqueue_capture_job(repository, tmp_path)
        job = repository.claim_next("worker", NOW, 30)
        repository.renew = lambda *_: next(renewal_results)
        success = ProviderResult(
            "codex", "luna", TaskKind.EXTRACT, "success", text="daily"
        )
        worker = MemoryWorker(
            repository,
            FakeRouter(routed(success)),
            daily_writer=blocking_writer,
            clock=lambda: NOW,
            owner="worker",
            lease_seconds=30,
            heartbeat_sleeper=no_wait,
        )

        asyncio.run(asyncio.wait_for(worker.process(job), timeout=0.2))

        assert writer_events == ["started", "cancelled"]
        assert repository.get_job(queued.job_id).status == "leased"
