from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows exercises the msvcrt branch.
    fcntl = None
import json
import hashlib
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import threading
import time

import pytest

import capture as capture_module
import flush as flush_module
from capture import (
    CaptureDeadlineExceeded,
    capture_transcript,
    enqueue_hook_input,
    launch_worker,
)
from providers import ProviderResult, ProviderRouter, RoutedResult, TaskKind
from scripts.queue import QueueRepository
from scripts.utils import ExclusiveFileLock
import scripts.usage as usage_module
from worker import MemoryWorker, SingletonDrainLock


NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def _wait_for_text(path: Path, expected: str, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    actual = None
    while time.monotonic() < deadline:
        try:
            actual = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            actual = None
        if actual == expected:
            return
        time.sleep(0.01)
    assert actual == expected


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


def test_deadline_during_private_copy_removes_owned_partial_snapshot(tmp_path):
    source = tmp_path / "large.jsonl"
    source.write_text(
        json.dumps({"message": {"role": "user", "content": "x" * 2_000_000}})
        + "\n",
        encoding="utf-8",
    )
    home = tmp_path / "memory"
    ticks = iter([0.0, 0.5, 1.1])

    with pytest.raises(CaptureDeadlineExceeded):
        capture_transcript(
            source,
            source_agent="claude",
            metadata={"trigger": "session_end"},
            memory_home=home,
            launcher=lambda _: (_ for _ in ()).throw(
                AssertionError("must not launch")
            ),
            env={},
            deadline=1.0,
            monotonic=lambda: next(ticks),
        )

    assert not (home / "scripts" / "jobs.sqlite3").exists()
    assert list((home / "scripts" / "spool").glob("*.jsonl")) == []


def test_deadline_during_hash_removes_owned_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"

    def completed_copy(_source, spool_dir, **_kwargs):
        snapshot = spool_dir / "capture-hashowner-partial.jsonl"
        snapshot.write_bytes(source.read_bytes())
        snapshot.chmod(0o600)
        return snapshot

    monkeypatch.setattr(capture_module, "_private_spool_copy", completed_copy)
    ticks = iter([0.0, 0.0, 1.1])

    with pytest.raises(CaptureDeadlineExceeded):
        capture_transcript(
            source,
            source_agent="claude",
            metadata={"trigger": "session_end"},
            memory_home=home,
            launcher=lambda _: (_ for _ in ()).throw(
                AssertionError("must not launch")
            ),
            env={},
            deadline=1.0,
            monotonic=lambda: next(ticks),
            capture_token="hashowner",
        )

    assert list((home / "scripts" / "spool").glob("*.jsonl")) == []


def test_deadline_before_queue_open_leaves_no_database_or_snapshot(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"
    current = [0.0]
    real_load_config = capture_module.load_config

    def delayed_config(env):
        config = real_load_config(env)
        current[0] = 2.0
        return config

    monkeypatch.setattr(capture_module, "load_config", delayed_config)

    with pytest.raises(CaptureDeadlineExceeded):
        capture_transcript(
            source,
            source_agent="claude",
            metadata={"trigger": "session_end"},
            memory_home=home,
            launcher=lambda _: (_ for _ in ()).throw(
                AssertionError("must not launch")
            ),
            env={},
            deadline=1.0,
            monotonic=lambda: current[0],
            capture_token="prequeue",
        )

    assert not (home / "scripts" / "jobs.sqlite3").exists()
    assert list((home / "scripts" / "spool").glob("*.jsonl")) == []


def test_deadline_after_committed_enqueue_retains_job_and_attempts_wake(tmp_path):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"
    current = [0.0]
    wake_marker = tmp_path / "wake-marker"

    class CommittingQueue:
        def enqueue_capture(self, normalized):
            current[0] = 2.0
            job = type(
                "CommittedJob",
                (),
                {"id": 41, "source_path": normalized.source_path},
            )()
            return type("Committed", (), {"created": True, "job": job})()

    result = capture_transcript(
        source,
        source_agent="claude",
        metadata={"trigger": "session_end"},
        memory_home=home,
        queue=CommittingQueue(),
        launcher=lambda _root: wake_marker.write_text("woke", encoding="utf-8"),
        env={},
        deadline=1.0,
        monotonic=lambda: current[0],
        capture_token="postcommit",
    )

    assert result.created is True
    assert Path(result.job.source_path).exists()
    _wait_for_text(wake_marker, "woke")


def test_launcher_failure_after_commit_preserves_referenced_snapshot_and_dedup(
    tmp_path,
):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"

    first = capture_transcript(
        source,
        source_agent="claude",
        metadata={"session_id": "launch-failure", "trigger": "session_end"},
        memory_home=home,
        launcher=lambda _: (_ for _ in ()).throw(RuntimeError("launch failed")),
        env={},
        deadline=time.monotonic() + 5,
        monotonic=time.monotonic,
        capture_token="launchfailure",
    )

    assert first.created is True
    snapshot = Path(first.job.source_path)
    assert snapshot.exists()
    assert "failed-" not in snapshot.name
    assert list(snapshot.parent.glob("failed-*.jsonl")) == []

    second = capture_transcript(
        source,
        source_agent="claude",
        metadata={"session_id": "launch-failure", "trigger": "session_end"},
        memory_home=home,
        launcher=lambda _: None,
        env={},
        deadline=time.monotonic() + 5,
        monotonic=time.monotonic,
        capture_token="equivalent",
    )

    assert second.created is False
    assert second.job_id == first.job_id
    assert Path(second.job.source_path) == snapshot
    assert snapshot.exists()


def test_postcommit_deadline_attempts_wake_without_blocking_or_losing_job(tmp_path):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"
    current = [0.0]
    wake_marker = tmp_path / "blocked-wake-marker"

    class CommittingQueue:
        def enqueue_capture(self, normalized):
            current[0] = 2.0
            job = type(
                "CommittedJob",
                (),
                {"id": 42, "source_path": normalized.source_path},
            )()
            return type("Committed", (), {"created": True, "job": job})()

    def blocked_wake(_root):
        time.sleep(0.25)
        wake_marker.write_text("woke", encoding="utf-8")

    started = time.monotonic()
    result = capture_transcript(
        source,
        source_agent="claude",
        metadata={"trigger": "session_end"},
        memory_home=home,
        queue=CommittingQueue(),
        launcher=blocked_wake,
        env={},
        deadline=1.0,
        monotonic=lambda: current[0],
        capture_token="postcommitwake",
    )
    elapsed = time.monotonic() - started

    assert result.created is True
    assert elapsed < 0.2
    assert Path(result.job.source_path).exists()
    _wait_for_text(wake_marker, "woke")


def test_expired_deadline_wake_outlives_capture_process(tmp_path):
    marker = tmp_path / "wake-marker"
    code = (
        "import pathlib, sys, time\n"
        f"sys.path.insert(0, {str(Path(capture_module.__file__).parent)!r})\n"
        "from capture import _wake_after_commit\n"
        "marker = pathlib.Path(sys.argv[1])\n"
        "def delayed(_root):\n"
        "    time.sleep(0.25)\n"
        "    with marker.open('w', encoding='utf-8') as stream:\n"
        "        stream.write('wo')\n"
        "        stream.flush()\n"
        "        time.sleep(0.1)\n"
        "        stream.write('ke')\n"
        "_wake_after_commit(delayed, marker, deadline=0.0, monotonic=lambda: 1.0)\n"
    )
    started = time.monotonic()

    result = subprocess.run(
        [sys.executable, "-c", code, str(marker)],
        text=True,
        capture_output=True,
        timeout=1,
        check=False,
    )
    process_elapsed = time.monotonic() - started
    assert result.returncode == 0, result.stderr
    assert process_elapsed < 0.2
    _wait_for_text(marker, "woke")


def test_slow_deadline_wake_invokes_launcher_once(tmp_path):
    marker = tmp_path / "wake-count"

    def slow_wake(_root):
        time.sleep(0.1)
        with marker.open("a", encoding="utf-8") as stream:
            stream.write("wake\n")

    capture_module._wake_after_commit(
        slow_wake,
        tmp_path,
        deadline=time.monotonic() + 0.02,
        monotonic=time.monotonic,
    )

    _wait_for_text(marker, "wake\n")
    time.sleep(0.2)
    assert marker.read_text(encoding="utf-8") == "wake\n"


def test_default_wake_starts_cross_platform_detached_helper(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(capture_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        capture_module.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    capture_module._start_detached_worker_wake(tmp_path)

    command, options = calls[0]
    assert command == [
        sys.executable,
        str(Path(capture_module.__file__).resolve()),
        "--wake-worker",
        str(tmp_path),
    ]
    assert options["stdin"] is subprocess.DEVNULL
    assert options["stdout"] is subprocess.DEVNULL
    assert options["stderr"] is subprocess.DEVNULL
    assert options["close_fds"] is True
    assert "creationflags" in options
    assert "start_new_session" not in options


def test_expired_wake_fork_failure_does_not_escape_commit(monkeypatch, tmp_path):
    launched = []
    monkeypatch.setattr(
        capture_module.os,
        "fork",
        lambda: (_ for _ in ()).throw(OSError("fork unavailable")),
    )

    capture_module._wake_after_commit(
        lambda root: launched.append(root),
        tmp_path,
        deadline=0.0,
        monotonic=lambda: 1.0,
    )

    deadline = time.monotonic() + 0.2
    while not launched and time.monotonic() < deadline:
        time.sleep(0.01)
    assert launched == [tmp_path]


def test_deadline_wake_from_worker_thread_does_not_fork(monkeypatch, tmp_path):
    launched = []
    monkeypatch.setattr(
        capture_module.os,
        "fork",
        lambda: (_ for _ in ()).throw(AssertionError("must not fork from a thread")),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(
            capture_module._wake_after_commit,
            lambda root: launched.append(root),
            tmp_path,
            deadline=time.monotonic() + 1,
            monotonic=time.monotonic,
        ).result()

    assert launched == [tmp_path]


def test_queue_close_failure_after_commit_keeps_snapshot_and_attempts_wake(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"
    wake_marker = tmp_path / "close-failure-wake"

    class CloseFailureQueue:
        def __init__(self, *_args, **_kwargs):
            pass

        def enqueue_capture(self, normalized):
            job = type(
                "CommittedJob",
                (),
                {"id": 43, "source_path": normalized.source_path},
            )()
            return type("Committed", (), {"created": True, "job": job})()

        def close(self):
            raise RuntimeError("close failed after commit")

    monkeypatch.setattr(capture_module, "QueueRepository", CloseFailureQueue)

    result = capture_transcript(
        source,
        source_agent="claude",
        metadata={"trigger": "session_end"},
        memory_home=home,
        launcher=lambda root: wake_marker.write_text(str(root), encoding="utf-8"),
        env={},
        deadline=time.monotonic() + 5,
        monotonic=time.monotonic,
        capture_token="closefailure",
    )

    assert result.created is True
    assert Path(result.job.source_path).exists()
    _wait_for_text(wake_marker, str(home))


def test_deadline_capture_tokens_isolate_concurrent_equivalent_snapshots(tmp_path):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"

    def run(token):
        return capture_transcript(
            source,
            source_agent="claude",
            metadata={"trigger": "session_end"},
            memory_home=home,
            launcher=lambda _: None,
            env={},
            deadline=time.monotonic() + 5,
            monotonic=time.monotonic,
            capture_token=token,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(run, ("ownerone", "ownertwo")))

    assert sorted([first.created, second.created]) == [False, True]
    assert first.job_id == second.job_id
    snapshots = list((home / "scripts" / "spool").glob("*.jsonl"))
    assert snapshots == [Path(first.job.source_path)]
    assert sum(token in snapshots[0].name for token in ("ownerone", "ownertwo")) == 1


def test_deadline_capture_tolerates_windows_without_fchmod(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    monkeypatch.delattr(capture_module.os, "fchmod", raising=False)

    result = capture_transcript(
        source,
        source_agent="claude",
        metadata={"trigger": "session_end"},
        memory_home=tmp_path / "memory",
        launcher=lambda _: None,
        env={},
        deadline=time.monotonic() + 5,
        monotonic=time.monotonic,
        capture_token="windows",
    )

    assert result.created is True


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


def test_identical_failed_captures_from_two_tokens_deduplicate(tmp_path):
    source = tmp_path / "invalid.jsonl"
    source.write_bytes(b'\xff{"private":"content"}\n')
    home = tmp_path / "memory"

    for token in ("first-token", "second-token"):
        with pytest.raises(UnicodeDecodeError):
            capture_transcript(
                source,
                source_agent="claude",
                metadata={"trigger": "session_end"},
                memory_home=home,
                launcher=lambda _root: None,
                env={},
                capture_token=token,
            )

    retained = list((home / "scripts" / "spool").glob("failed-*.jsonl"))
    assert len(retained) == 1
    assert retained[0].name == (
        f"failed-claude-{hashlib.sha256(source.read_bytes()).hexdigest()}.jsonl"
    )
    assert retained[0].stat().st_nlink == 1
    assert stat.S_IMODE(retained[0].stat().st_mode) == 0o600


def test_distinct_failed_capture_content_is_preserved(tmp_path):
    home = tmp_path / "memory"
    for index in range(2):
        source = tmp_path / f"invalid-{index}.jsonl"
        source.write_bytes(bytes([0xFF, index]))
        with pytest.raises(UnicodeDecodeError):
            capture_transcript(
                source,
                source_agent="claude",
                metadata={"trigger": "session_end"},
                memory_home=home,
                launcher=lambda _root: None,
                env={},
                capture_token=f"token-{index}",
            )

    assert len(list((home / "scripts" / "spool").glob("failed-*.jsonl"))) == 2


def test_failed_snapshot_collision_with_tampered_content_fails_safely(tmp_path):
    source = tmp_path / "invalid.jsonl"
    source.write_bytes(b"\xffprivate")
    home = tmp_path / "memory"
    spool = home / "scripts" / "spool"
    spool.mkdir(parents=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    collision = spool / f"failed-claude-{digest}.jsonl"
    collision.write_bytes(b"tampered")
    collision.chmod(0o600)

    with pytest.raises(UnicodeDecodeError):
        capture_transcript(
            source,
            source_agent="claude",
            metadata={"trigger": "session_end"},
            memory_home=home,
            launcher=lambda _root: None,
            env={},
            capture_token="ordinary-failure",
        )

    assert collision.read_bytes() == b"tampered"
    random_recovery = list(spool.glob("capture-ordinary-failure-*.jsonl"))
    assert len(random_recovery) == 1
    assert random_recovery[0].read_bytes() == source.read_bytes()


def test_private_spool_copy_does_not_close_reused_descriptor(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    spool = tmp_path / "spool"
    spool.mkdir()
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep open", encoding="utf-8")
    reused: list[int] = []

    def fail_after_descriptor_transfer(_path):
        reused.append(os.open(unrelated, os.O_RDONLY))
        raise RuntimeError("fsync failed")

    monkeypatch.setattr(capture_module, "_fsync_file", fail_after_descriptor_transfer)

    with pytest.raises(RuntimeError, match="fsync failed"):
        capture_module._private_spool_copy(source, spool)

    assert os.fstat(reused[0]).st_size == len("keep open")
    os.close(reused[0])


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


def enqueue_capture_job(repository, tmp_path, *, session_id="session-1"):
    source = tmp_path / f"source-{session_id}.jsonl"
    source.write_text(
        json.dumps(
            {
                "sessionId": session_id,
                "cwd": "/tmp/project",
                "message": {"role": "user", "content": "Remember the queue"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return capture_transcript(
        source,
        source_agent="claude",
        metadata={"trigger": "session_end"},
        memory_home=tmp_path,
        queue=repository,
        launcher=lambda _: None,
        clock=lambda: NOW,
    )


def test_worker_bounds_parallel_model_jobs_and_serializes_durable_writes(tmp_path):
    async def exercise(repository):
        for index in range(3):
            enqueue_capture_job(repository, tmp_path, session_id=f"session-{index}")

        provider_active = 0
        provider_max = 0
        provider_started = asyncio.Event()
        release_provider = asyncio.Event()
        writer_active = 0
        writer_max = 0

        class BlockingRouter:
            async def generate_text(self, _request):
                nonlocal provider_active, provider_max
                provider_active += 1
                provider_max = max(provider_max, provider_active)
                if provider_active == 2:
                    provider_started.set()
                await release_provider.wait()
                provider_active -= 1
                result = ProviderResult(
                    "codex", "luna", TaskKind.EXTRACT, "success", text="daily"
                )
                return routed(result)

        async def observed_writer(_job, _text):
            nonlocal writer_active, writer_max
            writer_active += 1
            writer_max = max(writer_max, writer_active)
            await asyncio.sleep(0.01)
            writer_active -= 1

        worker = MemoryWorker(
            repository,
            BlockingRouter(),
            concurrency=2,
            daily_writer=observed_writer,
            clock=lambda: NOW,
            owner="worker",
            heartbeat_sleeper=lambda _delay: asyncio.sleep(3600),
        )
        draining = asyncio.create_task(worker.drain())
        await asyncio.wait_for(provider_started.wait(), timeout=0.5)
        await asyncio.sleep(0)
        assert provider_active == 2
        assert provider_max == 2
        release_provider.set()

        assert await asyncio.wait_for(draining, timeout=1) == 3
        assert provider_max == 2
        assert writer_max == 1
        assert all(
            repository.get_job(job_id).status == "succeeded"
            for job_id in range(1, 4)
        )

    with QueueRepository(tmp_path / "jobs.sqlite3", clock=lambda: NOW) as repository:
        asyncio.run(exercise(repository))


def test_worker_rejects_nonpositive_concurrency(tmp_path):
    with QueueRepository(tmp_path / "jobs.sqlite3", clock=lambda: NOW) as repository:
        with pytest.raises(ValueError, match="concurrency must be positive"):
            MemoryWorker(repository, FakeRouter(None), concurrency=0)


def test_default_worker_consumes_configured_live_concurrency(tmp_path, monkeypatch):
    import worker as worker_module
    from types import SimpleNamespace

    config = SimpleNamespace(
        root_dir=tmp_path,
        queue_path=tmp_path / "jobs.sqlite3",
        task_models={},
        claude_model="claude-test",
        job_timeout_seconds=30,
        worker_concurrency=3,
    )
    repository = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(worker_module, "load_config", lambda _env: config)
    recoveries = []
    monkeypatch.setattr(
        worker_module,
        "recover_incomplete_apply",
        lambda root: recoveries.append(root),
    )
    repository_options = {}

    def open_repository(*_args, **kwargs):
        repository_options.update(kwargs)
        return repository

    monkeypatch.setattr(worker_module, "QueueRepository", open_repository)
    monkeypatch.setattr(worker_module, "CodexProvider", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(worker_module, "ClaudeProvider", lambda **_kwargs: SimpleNamespace())

    configured, returned_repository = worker_module._default_worker()

    assert configured.concurrency == 3
    assert returned_repository is repository
    assert repository_options["sync_usage"] is False
    assert recoveries == []
    configured.startup_recovery()
    assert recoveries == [tmp_path]


def test_losing_worker_does_not_recover_apply_or_sync_usage(tmp_path):
    lock_path = tmp_path / "memory-worker.lock"
    held = SingletonDrainLock(lock_path)
    assert held.acquire()
    calls = []
    try:
        with QueueRepository(
            tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
        ) as repository:
            repository.sync_usage_records = lambda: calls.append("sync")
            worker = MemoryWorker(
                repository,
                FakeRouter(None),
                daily_writer=lambda *_: None,
                clock=lambda: NOW,
                lock_path=lock_path,
                startup_recovery=lambda: calls.append("recover-apply"),
            )
            assert asyncio.run(worker.run_drain()) == 0
    finally:
        held.release()

    assert calls == []


def test_winning_worker_recovers_apply_then_syncs_usage_before_queue_recovery(tmp_path):
    events = []
    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        real_recover = repository.recover_stale
        repository.sync_usage_records = lambda: events.append("sync")

        def observed_recover(now):
            events.append("recover")
            return real_recover(now)

        repository.recover_stale = observed_recover
        worker = MemoryWorker(
            repository,
            FakeRouter(None),
            daily_writer=lambda *_: None,
            clock=lambda: NOW,
            lock_path=tmp_path / "memory-worker.lock",
            startup_recovery=lambda: events.append("recover-apply"),
        )

        assert asyncio.run(worker.run_drain()) == 0

    assert events == ["recover-apply", "sync", "recover"]


def test_live_capture_queue_open_does_not_wait_for_usage_writer_lock(tmp_path):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"
    logs = home / "scripts" / "logs"
    logs.mkdir(parents=True)
    archive_payload = b'{}\n'
    archive_digest = hashlib.sha256(archive_payload).hexdigest()
    archive = logs / f"usage.archive-{archive_digest}.jsonl"
    archive.write_bytes(archive_payload)
    archive.chmod(0o600)
    lock = ExclusiveFileLock(home / "scripts" / "memory-writer.lock")
    assert lock.acquire()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        capture_transcript,
        source,
        source_agent="claude",
        metadata={"trigger": "session_end"},
        memory_home=home,
        launcher=lambda _root: None,
        env={},
        deadline=time.monotonic() + 2.0,
        monotonic=time.monotonic,
        capture_token="writer-lock",
    )
    try:
        result = future.result(timeout=0.5)
    finally:
        lock.release()
        executor.shutdown(wait=True)

    assert result.created is True


def test_live_capture_queue_open_does_not_read_usage_archives(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    write_claude_transcript(source)
    home = tmp_path / "memory"
    logs = home / "scripts" / "logs"
    logs.mkdir(parents=True)
    for index in range(3):
        payload = json.dumps({"archive": index}, separators=(",", ":")).encode() + b"\n"
        digest = hashlib.sha256(payload).hexdigest()
        archive = logs / f"usage.archive-{digest}.jsonl"
        archive.write_bytes(payload)
        archive.chmod(0o600)
    observed: list[Path] = []
    real_read = usage_module._read_private_log

    def observe_read(path):
        observed.append(Path(path))
        return real_read(path)

    monkeypatch.setattr(usage_module, "_read_private_log", observe_read)

    result = capture_transcript(
        source,
        source_agent="claude",
        metadata={"trigger": "session_end"},
        memory_home=home,
        launcher=lambda _root: None,
        env={},
        deadline=time.monotonic() + 2.0,
        monotonic=time.monotonic,
        capture_token="archive-scan",
    )

    assert result.created is True
    assert not [path for path in observed if path.name.startswith("usage.archive-")]


def test_cancelling_concurrent_drain_cancels_every_inflight_provider(tmp_path):
    async def exercise(repository):
        for index in range(2):
            enqueue_capture_job(repository, tmp_path, session_id=f"cancel-{index}")
        started = asyncio.Event()
        active = 0
        cancelled = []

        class HangingRouter:
            async def generate_text(self, _request):
                nonlocal active
                active += 1
                if active == 2:
                    started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.append(True)
                    raise

        worker = MemoryWorker(
            repository,
            HangingRouter(),
            concurrency=2,
            daily_writer=lambda *_args: None,
            clock=lambda: NOW,
            owner="worker",
            heartbeat_sleeper=lambda _delay: asyncio.sleep(3600),
        )
        draining = asyncio.create_task(worker.drain())
        await asyncio.wait_for(started.wait(), timeout=0.5)
        draining.cancel()
        with pytest.raises(asyncio.CancelledError):
            await draining

        assert cancelled == [True, True]
        assert [repository.get_job(job_id).status for job_id in (1, 2)] == [
            "leased",
            "leased",
        ]

    with QueueRepository(tmp_path / "jobs.sqlite3", clock=lambda: NOW) as repository:
        asyncio.run(exercise(repository))


def test_worker_renews_lease_while_waiting_for_serialized_writer(tmp_path):
    async def exercise(repository):
        for index in range(2):
            enqueue_capture_job(repository, tmp_path, session_id=f"lease-wait-{index}")
        first_writer_started = asyncio.Event()
        release_first_writer = asyncio.Event()
        second_lost_lease = asyncio.Event()
        writes = []
        renewal_counts = {1: 0, 2: 0}
        original_renew = repository.renew

        def renew(job_id, owner, expires_at):
            renewal_counts[job_id] += 1
            if job_id == 2 and renewal_counts[job_id] >= 2:
                second_lost_lease.set()
                return False
            return original_renew(job_id, owner, expires_at)

        repository.renew = renew

        class SuccessfulRouter:
            async def generate_text(self, _request):
                result = ProviderResult(
                    "codex", "luna", TaskKind.EXTRACT, "success", text="daily"
                )
                return routed(result)

        async def blocking_writer(job, _text):
            writes.append(job.id)
            if job.id == 1:
                first_writer_started.set()
                await release_first_writer.wait()

        async def short_heartbeat(_delay):
            await asyncio.sleep(0.01)

        worker = MemoryWorker(
            repository,
            SuccessfulRouter(),
            concurrency=2,
            daily_writer=blocking_writer,
            clock=lambda: NOW,
            owner="worker",
            heartbeat_sleeper=short_heartbeat,
        )
        draining = asyncio.create_task(worker.drain())
        await asyncio.wait_for(first_writer_started.wait(), timeout=0.5)
        await asyncio.wait_for(second_lost_lease.wait(), timeout=0.5)
        release_first_writer.set()
        while repository.get_job(1).status != "succeeded":
            await asyncio.sleep(0)
        draining.cancel()
        with pytest.raises(asyncio.CancelledError):
            await draining

        assert writes == [1]
        assert repository.get_job(1).status == "succeeded"
        assert repository.get_job(2).status == "leased"

    with QueueRepository(tmp_path / "jobs.sqlite3", clock=lambda: NOW) as repository:
        asyncio.run(exercise(repository))


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


def test_worker_triggers_end_of_day_scheduler_once_after_successful_idle_drain(
    tmp_path,
):
    lock_path = tmp_path / "memory-worker.lock"
    scheduled = []

    def schedule_after_release():
        contender = SingletonDrainLock(lock_path)
        acquired = contender.acquire()
        scheduled.append(acquired)
        if acquired:
            contender.release()

    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        enqueue_capture_job(repository, tmp_path)
        success = ProviderResult(
            "codex", "luna", TaskKind.EXTRACT, "success", text="daily"
        )
        worker = MemoryWorker(
            repository,
            FakeRouter(routed(success)),
            daily_writer=lambda *_: None,
            clock=lambda: NOW,
            owner="worker",
            lock_path=lock_path,
            end_of_day_scheduler=schedule_after_release,
        )

        assert asyncio.run(worker.run_drain()) == 1

    assert scheduled == [True]


def test_worker_does_not_trigger_end_of_day_scheduler_without_successful_write(
    tmp_path,
):
    scheduled = []
    with QueueRepository(
        tmp_path / "jobs.sqlite3",
        clock=lambda: NOW,
        max_attempts=1,
        sync_usage=False,
    ) as repository:
        enqueue_capture_job(repository, tmp_path)
        failure = ProviderResult(
            "codex", "luna", TaskKind.EXTRACT, "capacity", reason="full"
        )
        worker = MemoryWorker(
            repository,
            FakeRouter(routed(failure)),
            daily_writer=lambda *_: (_ for _ in ()).throw(
                AssertionError("must not write")
            ),
            clock=lambda: NOW,
            owner="worker",
            lock_path=tmp_path / "memory-worker.lock",
            end_of_day_scheduler=lambda: scheduled.append(True),
        )

        assert asyncio.run(worker.run_drain()) == 1

    assert scheduled == []


def test_worker_does_not_schedule_while_retry_work_remains(tmp_path):
    async def exercise(repository):
        enqueue_capture_job(repository, tmp_path)
        failure = ProviderResult(
            "codex", "luna", TaskKind.EXTRACT, "capacity", reason="full"
        )
        waiting = asyncio.Event()
        keep_waiting = asyncio.Event()
        scheduled = []

        async def blocked_retry_sleep(_seconds):
            waiting.set()
            await keep_waiting.wait()

        worker = MemoryWorker(
            repository,
            FakeRouter(routed(failure)),
            daily_writer=lambda *_: None,
            clock=lambda: NOW,
            owner="worker",
            lock_path=tmp_path / "memory-worker.lock",
            sleeper=blocked_retry_sleep,
            end_of_day_scheduler=lambda: scheduled.append(True),
        )
        draining = asyncio.create_task(worker.run_drain())
        await asyncio.wait_for(waiting.wait(), timeout=0.5)
        assert scheduled == []
        draining.cancel()
        with pytest.raises(asyncio.CancelledError):
            await draining

    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        asyncio.run(exercise(repository))


def test_agent_session_counter_includes_interactive_claude_and_codex_only(
    monkeypatch,
):
    process_table = "\n".join(
        (
            "101 claude /usr/local/bin/claude --resume session-1",
            "102 claude /opt/sdk/_bundled/claude --print",
            "103 codex /usr/local/bin/codex",
            "104 codex /usr/local/bin/codex --ask-for-approval never exec "
            "--skip-git-repo-check --ephemeral --ignore-user-config --ignore-rules "
            "--model gpt-5.6-luna --sandbox read-only --cd /tmp/project "
            "--output-last-message /tmp/ai-memory-codex-abc/last-message.txt -",
            "105 codex /usr/local/bin/codex --prompt 'please exec this command'",
            "106 python /usr/bin/python worker.py --drain",
            "107 codex /usr/local/bin/codex --ask-for-approval never exec "
            "--skip-git-repo-check --ephemeral --ignore-user-config --ignore-rules "
            "--model gpt-5.6-terra --sandbox workspace-write --cd /tmp/project "
            "--output-last-message "
            "/tmp/project/.ai-memory-last-message-0123456789abcdef0123456789abcdef.txt "
            "--output-schema /tmp/schema.json -",
        )
    )
    monkeypatch.setattr(
        flush_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=process_table, stderr=""
        ),
    )

    assert flush_module.count_interactive_agent_sessions() == 3


def test_agent_session_counter_does_not_trust_bundled_path_inside_claude_prompt(
    monkeypatch,
):
    process_table = "\n".join(
        (
            "101 claude /usr/local/bin/claude --prompt "
            "'debug /opt/sdk/_bundled/claude --print'",
            "102 claude /opt/sdk/_bundled/claude --print",
        )
    )
    monkeypatch.setattr(flush_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        flush_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=process_table, stderr=""
        ),
    )

    assert flush_module.count_interactive_agent_sessions() == 1


def test_windows_agent_session_counter_uses_native_bounded_process_listing(
    monkeypatch,
):
    processes = [
        (
            r"C:\Program Files\Claude\claude.exe",
            [r"C:\Program Files\Claude\claude.exe", "--resume", "session"],
        ),
        (
            r"C:\sdk\_bundled\claude.exe",
            [r"C:\sdk\_bundled\claude.exe", "--print"],
        ),
        (r"C:\Tools\codex.exe", [r"C:\Tools\codex.exe"]),
        (
            r"C:\Tools\codex.exe",
            [
                r"C:\Tools\codex.exe",
                "--ask-for-approval",
                "never",
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--model",
                "gpt-5.6-luna",
                "--sandbox",
                "read-only",
                "--cd",
                r"C:\memory",
                "--output-last-message",
                r"C:\Temp\ai-memory-codex-abc\last-message.txt",
                "--output-schema",
                r"C:\memory\schema.json",
                "-",
            ],
        ),
    ]
    monkeypatch.setattr(flush_module.sys, "platform", "win32")
    monkeypatch.setattr(flush_module, "_list_windows_processes", lambda: processes)
    monkeypatch.setattr(
        flush_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Windows must not invoke ps")
        ),
    )

    assert flush_module.count_interactive_agent_sessions() == 2


def test_windows_process_listing_uses_filtered_argv_command_without_shell(
    monkeypatch,
):
    calls = []
    payload = json.dumps(
        {
            "Name": "claude.exe",
            "ExecutablePath": r"C:\Tools\claude.exe",
            "CommandLine": r'"C:\Tools\claude.exe" --resume session',
        }
    )

    def run(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, payload, "")

    monkeypatch.setattr(flush_module.subprocess, "run", run)
    monkeypatch.setattr(
        flush_module,
        "_split_windows_command_line",
        lambda command_line: [command_line],
    )

    assert flush_module._list_windows_processes() == [
        (r"C:\Tools\claude.exe", [r'"C:\Tools\claude.exe" --resume session'])
    ]
    command, options = calls[0]
    assert command[:5] == [
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
    ]
    assert "-Filter" in command[5]
    assert "claude.exe" in command[5]
    assert "codex.exe" in command[5]
    assert options["timeout"] == 5
    assert "shell" not in options


def test_windows_process_listing_accepts_successful_empty_result(monkeypatch):
    monkeypatch.setattr(
        flush_module.subprocess,
        "run",
        lambda command, **_options: subprocess.CompletedProcess(command, 0, "", ""),
    )

    assert flush_module._list_windows_processes() == []


def test_windows_process_listing_fails_closed_for_inaccessible_matching_process(
    monkeypatch,
):
    payload = json.dumps(
        {
            "Name": "claude.exe",
            "ExecutablePath": None,
            "CommandLine": None,
        }
    )
    monkeypatch.setattr(
        flush_module.subprocess,
        "run",
        lambda command, **_options: subprocess.CompletedProcess(
            command, 0, payload, ""
        ),
    )

    with pytest.raises(OSError, match="inaccessible process"):
        flush_module._list_windows_processes()


def test_windows_agent_session_counter_fails_closed_on_enumeration_error(
    monkeypatch,
):
    monkeypatch.setattr(flush_module.sys, "platform", "win32")
    monkeypatch.setattr(
        flush_module,
        "_list_windows_processes",
        lambda: (_ for _ in ()).throw(OSError("CIM unavailable")),
    )

    assert flush_module.count_interactive_agent_sessions() == -1


def test_windows_agent_session_counter_fails_closed_on_timeout(monkeypatch):
    monkeypatch.setattr(flush_module.sys, "platform", "win32")
    monkeypatch.setattr(
        flush_module,
        "_list_windows_processes",
        lambda: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["powershell.exe"], 5)
        ),
    )

    assert flush_module.count_interactive_agent_sessions() == -1


@pytest.mark.parametrize(
    "arguments",
    [
        "--ask-for-approval never exec --ephemeral --output-last-message "
        "/tmp/ai-memory-codex-abc/last-message.txt -",
        "--ask-for-approval never exec --skip-git-repo-check --ephemeral "
        "--ignore-user-config --ignore-rules --model gpt-5.6-luna "
        "--sandbox read-only --cd /tmp/project --output-last-message /tmp/output -",
        "--ask-for-approval never exec --skip-git-repo-check --ephemeral "
        "--ignore-user-config --model gpt-5.6-luna --sandbox read-only "
        "--cd /tmp/project --output-last-message "
        "/tmp/ai-memory-codex-abc/last-message.txt -",
        "--ask-for-approval never exec --skip-git-repo-check --ephemeral "
        "--ignore-user-config --ignore-rules --model gpt-5.6-luna "
        "--sandbox read-only --cd /tmp/project --output-last-message "
        "/tmp/ai-memory-codex-abc/last-message.txt",
        "--prompt 'imitate --ask-for-approval never exec --skip-git-repo-check "
        "--ephemeral --ignore-user-config --ignore-rules --model gpt-5.6-luna "
        "--sandbox read-only --cd /tmp/project --output-last-message "
        "/tmp/ai-memory-codex-abc/last-message.txt -'",
        "--ask-for-approval never exec --skip-git-repo-check --ephemeral "
        "--ignore-user-config --ignore-rules --model gpt-5.6-terra "
        "--sandbox workspace-write --cd /tmp/project --output-last-message "
        "/tmp/elsewhere/.ai-memory-last-message-0123456789abcdef0123456789abcdef.txt -",
    ],
)
def test_agent_session_counter_treats_partial_provider_shapes_as_interactive(
    monkeypatch, arguments
):
    monkeypatch.setattr(
        flush_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"101 codex /usr/local/bin/codex {arguments}\n",
            stderr="",
        ),
    )

    assert flush_module.count_interactive_agent_sessions() == 1


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("missing"),
        subprocess.TimeoutExpired(["ps"], 5),
        OSError("unavailable"),
    ],
)
def test_agent_session_counter_fails_closed_when_process_inspection_fails(
    monkeypatch, failure
):
    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(flush_module.subprocess, "run", fail)

    assert flush_module.count_interactive_agent_sessions() == -1


def test_end_of_day_scheduler_requires_uncompiled_content_after_four_pm(
    tmp_path, monkeypatch
):
    root = tmp_path / "memory"
    scripts = root / "scripts"
    daily = root / "daily"
    scripts.mkdir(parents=True)
    daily.mkdir()
    (scripts / "compile.py").write_text("# compiler\n", encoding="utf-8")
    (scripts / "auto-compile.py").write_text("# coordinator\n", encoding="utf-8")
    log_path = daily / "2026-08-11.md"
    log_path.write_text(
        "# Daily\n\n<!-- @compiled-through:2026-08-11T16:01:00-07:00 -->\n",
        encoding="utf-8",
    )
    launches = []
    monkeypatch.setattr(
        flush_module, "count_interactive_agent_sessions", lambda: 0
    )
    monkeypatch.setattr(flush_module, "_resolve_tty_path", lambda: None)
    monkeypatch.setattr(flush_module, "notify_terminal", lambda _message: None)
    monkeypatch.setattr(
        flush_module.subprocess,
        "Popen",
        lambda command, **options: launches.append((command, options)),
    )

    assert (
        flush_module.maybe_trigger_compilation(
            memory_home=root,
            now=datetime(2026, 8, 11, 16, 30, tzinfo=timezone.utc),
        )
        is False
    )
    log_path.write_text(log_path.read_text() + "New session content.\n", encoding="utf-8")
    assert (
        flush_module.maybe_trigger_compilation(
            memory_home=root,
            now=datetime(2026, 8, 11, 16, 31, tzinfo=timezone.utc),
        )
        is True
    )

    assert len(launches) == 2
    command, options = next(
        launch for launch in launches if launch[0][2] == "owner"
    )
    assert command[:2] == [sys.executable, str(scripts / "auto-compile.py")]
    assert command[2] == "owner"
    assert command[3] == str(root.resolve())
    assert len(command[4]) == 64
    assert len(command[5]) == 64
    assert options["env"]["AI_MEMORY_HOME"] == str(root.resolve())
    assert options["env"]["AI_MEMORY_INTERNAL_JOB"] == "1"
    assert options["env"]["AI_MEMORY_AUTO_COMPILE"] == "1"


def test_first_auto_compile_request_atomically_spawns_owner_and_watchdog(
    tmp_path, monkeypatch
):
    root, _daily = _auto_compile_test_root(tmp_path)
    commands = []
    scheduling = [False]
    spawn_during_transaction = []
    real_schedule = QueueRepository.schedule_auto_compile

    def observed_schedule(repository, *args, **kwargs):
        scheduling[0] = True
        try:
            return real_schedule(repository, *args, **kwargs)
        finally:
            scheduling[0] = False

    _configure_auto_compile_test(monkeypatch, commands)
    monkeypatch.setattr(QueueRepository, "schedule_auto_compile", observed_schedule)
    monkeypatch.setattr(
        flush_module.subprocess,
        "Popen",
        lambda command, **_options: (
            spawn_during_transaction.append(scheduling[0]), commands.append(command)
        ),
    )

    assert _schedule_auto_compile(root) is True

    assert [command[2] for command in commands] == ["watchdog", "owner"]
    assert spawn_during_transaction == [True, True]
    watchdog_token = commands[0][-1]
    owner_token, fingerprint = commands[1][-2:]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["token"] == owner_token
    assert state["fingerprint"] == fingerprint
    assert state["watcher_token"] == watchdog_token
    assert state["required_marker_prefix"] == []


def test_initial_schedule_persists_readable_marker_prefix(tmp_path, monkeypatch):
    root, daily = _auto_compile_test_root(tmp_path)
    daily.write_text(
        "<!-- @compiled-through:first -->\nA\n", encoding="utf-8"
    )
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["required_marker_prefix"] == ["first"]


def test_first_owner_crash_is_taken_over_without_a_later_drain(tmp_path, monkeypatch):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    watchdog_token = commands[0][-1]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        state["expires_at"] = "2000-01-01T00:00:00.000000+00:00"
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )
    takeovers = []

    assert flush_module.run_auto_compile_watcher(
        root,
        watchdog_token,
        coordinator=lambda _root, token, fingerprint: (
            takeovers.append((token, fingerprint)) or True
        ),
        sleeper=lambda _seconds: (_ for _ in ()).throw(
            AssertionError("expired first owner must be claimed immediately")
        ),
        lock_probe=lambda _root: False,
    )
    assert takeovers[0][1] == flush_module._uncompiled_fingerprint(daily)


def test_watchdog_spawn_failure_rolls_back_both_first_roles_for_retry(
    tmp_path, monkeypatch
):
    root, _daily = _auto_compile_test_root(tmp_path)
    attempts = []
    monkeypatch.setattr(flush_module, "count_interactive_agent_sessions", lambda: 0)
    monkeypatch.setattr(flush_module, "_resolve_tty_path", lambda: None)
    monkeypatch.setattr(flush_module, "notify_terminal", lambda _message: None)

    def fail_watchdog(command, **_options):
        attempts.append(command)
        if command[2] == "watchdog":
            raise OSError("watchdog spawn failed")
        raise AssertionError("owner must not spawn without watchdog")

    monkeypatch.setattr(flush_module.subprocess, "Popen", fail_watchdog)
    monkeypatch.setattr(flush_module.time, "sleep", lambda _seconds: None)

    assert _schedule_auto_compile(root) is False
    assert len(attempts) == flush_module.AUTO_COMPILE_SPAWN_ATTEMPTS
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        assert _reservation(repository) is None


def test_watchdog_spawn_rollback_leaves_latest_generation_retryable(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    attempts = []
    monkeypatch.setattr(flush_module, "count_interactive_agent_sessions", lambda: 0)
    monkeypatch.setattr(flush_module, "_resolve_tty_path", lambda: None)
    monkeypatch.setattr(flush_module, "notify_terminal", lambda _message: None)
    monkeypatch.setattr(flush_module.time, "sleep", lambda _seconds: None)

    def fail_watchdog(command, **_options):
        attempts.append(command)
        raise OSError("watchdog spawn failed")

    monkeypatch.setattr(flush_module.subprocess, "Popen", fail_watchdog)

    assert _schedule_auto_compile(root) is False
    daily.write_text("A\nB\n", encoding="utf-8")
    monkeypatch.setattr(
        flush_module.subprocess,
        "Popen",
        lambda command, **_options: attempts.append(command),
    )
    assert _schedule_auto_compile(root) is True

    latest = flush_module._uncompiled_fingerprint(daily)
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["fingerprint"] == latest


def test_owner_spawn_failure_watchdog_bootstraps_recovery_without_later_drain(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    launches = []
    monkeypatch.setattr(flush_module, "count_interactive_agent_sessions", lambda: 0)
    monkeypatch.setattr(flush_module, "_resolve_tty_path", lambda: None)
    monkeypatch.setattr(flush_module, "notify_terminal", lambda _message: None)

    def spawn(command, **_options):
        launches.append(command)
        if command[2] == "owner":
            raise OSError("owner spawn failed")

    monkeypatch.setattr(flush_module.subprocess, "Popen", spawn)

    assert _schedule_auto_compile(root) is False

    assert [command[2] for command in launches] == ["watchdog", "owner"]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        assert _reservation(repository) is None
    bootstrap = launches[0]
    recovered = []
    assert flush_module.run_auto_compile_watchdog_bootstrap(
        root,
        bootstrap[8],
        bootstrap[4],
        bootstrap[5],
        bootstrap[6],
        tuple(json.loads(bootstrap[7])),
        coordinator=lambda _root, token, fingerprint: (
            recovered.append((token, fingerprint)) or True
        ),
        sleeper=lambda _seconds: None,
        lock_probe=lambda _root: False,
        clock=lambda: datetime.now(timezone.utc),
    )
    assert recovered == [(bootstrap[8], flush_module._uncompiled_fingerprint(daily))]


def test_schedule_spawn_then_commit_failure_rolls_back_and_children_see_no_role(
    tmp_path
):
    database = tmp_path / "jobs.sqlite3"
    owner = "1" * 64
    watchdog = "2" * 64
    fingerprint = "a" * 64
    spawned = []
    with QueueRepository(database, sync_usage=False) as repository:

        def spawn_then_rollback(roles, predecessor):
            assert repository._connection.in_transaction
            assert roles == ("owner", "watchdog")
            assert predecessor is None
            spawned.extend(roles)
            repository._connection.execute("ROLLBACK")

        with pytest.raises(sqlite3.OperationalError):
            repository.schedule_auto_compile(
                owner,
                watchdog,
                fingerprint,
                log_name="2026-08-11.md",
                now=NOW,
                expires_at=NOW + timedelta(minutes=2),
                launch_roles=spawn_then_rollback,
            )
    assert spawned == ["owner", "watchdog"]
    with QueueRepository(database, sync_usage=False) as repository:
        assert _reservation(repository) is None


def test_bootstrap_rejects_nonidle_queue_and_does_not_create_reservation(tmp_path):
    root, daily = _auto_compile_test_root(tmp_path)
    fingerprint = flush_module._uncompiled_fingerprint(daily)
    assert fingerprint is not None
    source = tmp_path / "busy.jsonl"
    write_claude_transcript(source)
    capture_transcript(
        source,
        source_agent="claude",
        metadata={"trigger": "session_end"},
        memory_home=root,
        launcher=lambda _root: None,
        clock=lambda: NOW,
    )
    assert not flush_module.run_auto_compile_watchdog_bootstrap(
        root,
        "2" * 64,
        "1" * 64,
        fingerprint,
        daily.name,
        (),
        coordinator=lambda *_args: (_ for _ in ()).throw(
            AssertionError("non-idle bootstrap must not coordinate")
        ),
        sleeper=lambda _seconds: None,
        lock_probe=lambda _root: False,
        clock=lambda: NOW,
    )
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        assert _reservation(repository) is None


def test_bootstrap_waits_past_five_seconds_for_parent_transaction(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    fingerprint = flush_module._uncompiled_fingerprint(daily)
    assert fingerprint is not None
    real_bootstrap = QueueRepository.bootstrap_auto_compile_watchdog
    bootstrap_calls = []

    def blocked_then_committed(self, *args, **kwargs):
        bootstrap_calls.append(True)
        if len(bootstrap_calls) == 1:
            raise sqlite3.OperationalError("parent transaction still owns writer lock")
        return real_bootstrap(self, *args, **kwargs)

    monkeypatch.setattr(
        QueueRepository,
        "bootstrap_auto_compile_watchdog",
        blocked_then_committed,
    )
    monkeypatch.setattr(
        flush_module, "_spawn_auto_compile_watchdog", lambda *_args: None
    )
    clock_calls = []

    def clock():
        current = NOW + timedelta(seconds=6 * len(clock_calls))
        clock_calls.append(current)
        return current

    sleeps = []
    coordinated = []

    def coordinate(_root, token, active_fingerprint):
        coordinated.append((token, active_fingerprint))
        with QueueRepository(
            root / "scripts/jobs.sqlite3", sync_usage=False
        ) as repository:
            repository.release_auto_compile(token, active_fingerprint)
        return True

    assert flush_module.run_auto_compile_watchdog_bootstrap(
        root,
        "2" * 64,
        "1" * 64,
        fingerprint,
        daily.name,
        (),
        coordinator=coordinate,
        sleeper=sleeps.append,
        lock_probe=lambda _root: False,
        clock=clock,
    )
    assert len(bootstrap_calls) == 2
    assert sleeps == [flush_module.AUTO_COMPILE_WATCHER_POLL_SECONDS]
    assert coordinated == [("2" * 64, fingerprint)]


@pytest.mark.parametrize("registration", ["replacement", "contender"])
def test_later_role_is_spawned_inside_its_registration_transaction(
    tmp_path, registration
):
    database = tmp_path / "jobs.sqlite3"
    owner = "1" * 64
    predecessor = "2" * 64
    candidate = "3" * 64
    fingerprint = "a" * 64
    later = "b" * 64
    launched = []
    with QueueRepository(database, sync_usage=False) as repository:
        assert repository.schedule_auto_compile(
            owner,
            predecessor,
            fingerprint,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(minutes=2),
        ) == ("owner", "watchdog")
        state = _reservation(repository)
        if registration == "replacement":
            state["watcher_expires_at"] = (NOW - timedelta(seconds=1)).isoformat()
        else:
            state["status"] = "queue_wait"
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )

        def launch(roles, predecessor_token):
            assert repository._connection.in_transaction
            current = _reservation(repository)
            if registration == "replacement":
                assert roles == ("watchdog",)
                assert current["watcher_token"] == candidate
                assert predecessor_token is None
            else:
                assert roles == ("contender",)
                assert current["contender_token"] == candidate
                assert predecessor_token == predecessor
            launched.append(roles)

        roles = repository.schedule_auto_compile(
            "4" * 64,
            candidate,
            later,
            log_name="2026-08-11.md",
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(minutes=3),
            launch_roles=launch,
        )
    assert launched == [roles]


def test_precommit_owner_starts_after_committed_registration_is_visible(
    tmp_path
):
    root, _daily = _auto_compile_test_root(tmp_path)
    owner = "1" * 64
    watchdog = "2" * 64
    fingerprint = "a" * 64
    with QueueRepository(
        root / "scripts/jobs.sqlite3", memory_home=root, sync_usage=False
    ) as repository:
        repository.schedule_auto_compile(
            owner,
            watchdog,
            fingerprint,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(minutes=2),
        )
    calls = []
    assert flush_module.run_auto_compile_owner_startup(
        root,
        owner,
        fingerprint,
        coordinator=lambda *_args: calls.append(True) or True,
        sleeper=lambda _seconds: (_ for _ in ()).throw(
            AssertionError("committed owner should start immediately")
        ),
        clock=lambda: NOW,
    )
    assert calls == [True]


def test_precommit_owner_waits_while_registration_is_not_yet_visible(
    tmp_path, monkeypatch
):
    root, _daily = _auto_compile_test_root(tmp_path)
    visible = iter([False, True])
    checks = []

    def registration(_repository, _token, role, *, predecessor_token):
        checks.append((role, predecessor_token))
        return next(visible)

    monkeypatch.setattr(
        QueueRepository, "auto_compile_role_registered", registration
    )
    sleeps = []
    coordinated = []
    assert flush_module.run_auto_compile_owner_startup(
        root,
        "1" * 64,
        "a" * 64,
        coordinator=lambda *_args: coordinated.append(True) or True,
        sleeper=sleeps.append,
        clock=lambda: NOW,
    )
    assert checks == [("owner", None), ("owner", None)]
    assert len(sleeps) == 1
    assert coordinated == [True]


def test_precommit_owner_exits_safely_when_registration_rolled_back(tmp_path):
    root, _daily = _auto_compile_test_root(tmp_path)
    ticks = iter(
        [
            NOW,
            NOW,
            NOW + timedelta(seconds=10),
        ]
    )
    assert not flush_module.run_auto_compile_owner_startup(
        root,
        "1" * 64,
        "a" * 64,
        coordinator=lambda *_args: (_ for _ in ()).throw(
            AssertionError("rolled-back owner must not coordinate")
        ),
        sleeper=lambda _seconds: None,
        clock=lambda: next(ticks),
    )


def test_precommit_watchdog_exits_safely_when_registration_rolled_back(tmp_path):
    root, _daily = _auto_compile_test_root(tmp_path)
    ticks = iter([NOW, NOW, NOW + timedelta(seconds=10)])
    assert not flush_module.run_auto_compile_watcher(
        root,
        "2" * 64,
        registration_required=True,
        coordinator=lambda *_args: (_ for _ in ()).throw(
            AssertionError("rolled-back watchdog must not coordinate")
        ),
        sleeper=lambda _seconds: None,
        lock_probe=lambda _root: False,
        clock=lambda: next(ticks),
    )


def test_end_of_day_scheduler_adds_one_watcher_across_repeated_callers(
    tmp_path, monkeypatch
):
    root = tmp_path / "memory"
    (root / "scripts").mkdir(parents=True)
    (root / "daily").mkdir()
    (root / "scripts/compile.py").write_text("# compiler\n", encoding="utf-8")
    (root / "scripts/auto-compile.py").write_text("# coordinator\n", encoding="utf-8")
    (root / "daily/2026-08-11.md").write_text("Uncompiled content.\n", encoding="utf-8")
    launches = []
    monkeypatch.setattr(flush_module, "count_interactive_agent_sessions", lambda: 0)
    monkeypatch.setattr(flush_module, "_resolve_tty_path", lambda: None)
    monkeypatch.setattr(flush_module, "notify_terminal", lambda _message: None)
    monkeypatch.setattr(
        flush_module.subprocess,
        "Popen",
        lambda command, **options: launches.append((command, options)),
    )
    now = datetime(2026, 8, 11, 16, 31, tzinfo=timezone.utc)

    assert flush_module.maybe_trigger_compilation(memory_home=root, now=now) is True
    assert flush_module.maybe_trigger_compilation(memory_home=root, now=now) is False

    assert [launch[0][2] for launch in launches] == ["watchdog", "owner"]


def test_end_of_day_scheduler_retries_watchdog_spawn_before_starting_owner(
    tmp_path, monkeypatch
):
    root = tmp_path / "memory"
    (root / "scripts").mkdir(parents=True)
    (root / "daily").mkdir()
    (root / "scripts/compile.py").write_text("# compiler\n", encoding="utf-8")
    (root / "scripts/auto-compile.py").write_text("# coordinator\n", encoding="utf-8")
    (root / "daily/2026-08-11.md").write_text("Uncompiled content.\n", encoding="utf-8")
    launches = []
    failures = [True, False, False]
    monkeypatch.setattr(flush_module, "count_interactive_agent_sessions", lambda: 0)
    monkeypatch.setattr(flush_module, "_resolve_tty_path", lambda: None)
    monkeypatch.setattr(flush_module, "notify_terminal", lambda _message: None)

    def spawn(command, **options):
        if failures.pop(0):
            raise OSError("spawn failed")
        launches.append((command, options))

    monkeypatch.setattr(flush_module.subprocess, "Popen", spawn)
    now = datetime(2026, 8, 11, 16, 31, tzinfo=timezone.utc)

    assert flush_module.maybe_trigger_compilation(memory_home=root, now=now) is True

    assert [command[0][2] for command in launches] == ["watchdog", "owner"]


def test_auto_compile_coordinator_cancels_if_job_arrives_after_reservation(
    tmp_path, monkeypatch
):
    root = tmp_path / "memory"
    (root / "scripts").mkdir(parents=True)
    (root / "daily").mkdir()
    (root / "scripts/compile.py").write_text("# compiler\n", encoding="utf-8")
    (root / "scripts/auto-compile.py").write_text("# coordinator\n", encoding="utf-8")
    (root / "daily/2026-08-11.md").write_text("Uncompiled content.\n", encoding="utf-8")
    coordinators = []
    compile_launches = []
    monkeypatch.setattr(flush_module, "count_interactive_agent_sessions", lambda: 0)
    monkeypatch.setattr(flush_module, "_resolve_tty_path", lambda: None)
    monkeypatch.setattr(flush_module, "notify_terminal", lambda _message: None)
    monkeypatch.setattr(
        flush_module.subprocess,
        "Popen",
        lambda command, **_options: coordinators.append(command),
    )
    now = datetime(2026, 8, 11, 16, 31, tzinfo=timezone.utc)
    assert flush_module.maybe_trigger_compilation(memory_home=root, now=now) is True
    token, fingerprint = coordinators[0][-2:]

    with QueueRepository(
        root / "scripts/jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        enqueue_capture_job(repository, root, session_id="arrived-before-launch")

    assert (
        flush_module.run_auto_compile_coordinator(
            root,
            token,
            fingerprint,
            compile_launcher=lambda *_args, **_kwargs: compile_launches.append(True),
        )
        is False
    )
    assert compile_launches == []


def test_queue_work_before_launch_defers_and_watchdog_resumes_without_new_drain(
    tmp_path, monkeypatch
):
    root, _daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    owner_token, fingerprint = _owner_command(commands)[-2:]
    watchdog_token = _watchdog_command(commands)[-1]
    launches = []
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        enqueue_capture_job(repository, root, session_id="arrived-before-launch")

    assert not flush_module.run_auto_compile_coordinator(
        root,
        owner_token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: launches.append(True),
    )
    assert launches == []
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        assert state["fingerprint"] == fingerprint
        assert state["watcher_token"] == watchdog_token
        repository._connection.execute("UPDATE jobs SET status = 'dead'")

    resumed = []

    def resume(_root, token, current):
        resumed.append((token, current))
        with QueueRepository(
            root / "scripts/jobs.sqlite3", sync_usage=False
        ) as repository:
            repository.release_auto_compile(token, current)
        return True

    assert flush_module.run_auto_compile_watcher(
        root,
        watchdog_token,
        coordinator=resume,
        sleeper=lambda _seconds: (_ for _ in ()).throw(
            AssertionError("deferred owner must be immediately reclaimable")
        ),
        lock_probe=lambda _root: False,
        clock=lambda: datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    assert resumed[0][1] == fingerprint


def test_queue_work_during_pending_promotion_preserves_latest_generation(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    owner_token, first = _owner_command(commands)[-2:]
    watchdog_token = _watchdog_command(commands)[-1]
    daily.write_text("A\nB\n", encoding="utf-8")
    latest = flush_module._uncompiled_fingerprint(daily)
    assert _schedule_auto_compile(root) is False
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        enqueue_capture_job(repository, root, session_id="arrived-before-promotion")

    assert not flush_module.run_auto_compile_coordinator(
        root,
        owner_token,
        first,
        compile_launcher=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("queue work must prevent compile launch")
        ),
    )
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        assert state["pending_fingerprint"] == latest
        repository._connection.execute("UPDATE jobs SET status = 'dead'")

    resumed = []

    def resume(_root, token, current):
        resumed.append(current)
        with QueueRepository(
            root / "scripts/jobs.sqlite3", sync_usage=False
        ) as repository:
            repository.release_auto_compile(token, current)
        return True

    assert flush_module.run_auto_compile_watcher(
        root,
        watchdog_token,
        coordinator=resume,
        sleeper=lambda _seconds: (_ for _ in ()).throw(
            AssertionError("latest generation must resume immediately")
        ),
        lock_probe=lambda _root: False,
        clock=lambda: datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    assert resumed == [latest]


def test_auto_compile_reservation_fingerprint_changes_with_uncompiled_content(
    tmp_path, monkeypatch
):
    root = tmp_path / "memory"
    (root / "scripts").mkdir(parents=True)
    (root / "daily").mkdir()
    (root / "scripts/compile.py").write_text("# compiler\n", encoding="utf-8")
    (root / "scripts/auto-compile.py").write_text("# coordinator\n", encoding="utf-8")
    daily = root / "daily/2026-08-11.md"
    daily.write_text("First content.\n", encoding="utf-8")
    commands = []
    monkeypatch.setattr(flush_module, "count_interactive_agent_sessions", lambda: 0)
    monkeypatch.setattr(flush_module, "_resolve_tty_path", lambda: None)
    monkeypatch.setattr(flush_module, "notify_terminal", lambda _message: None)
    monkeypatch.setattr(
        flush_module.subprocess,
        "Popen",
        lambda command, **_options: commands.append(command),
    )
    now = datetime(2026, 8, 11, 16, 31, tzinfo=timezone.utc)
    assert flush_module.maybe_trigger_compilation(memory_home=root, now=now) is True
    first_token, first_fingerprint = commands[-1][-2:]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        assert repository.release_auto_compile(first_token, first_fingerprint) is True
    daily.write_text("First content.\nSecond content.\n", encoding="utf-8")

    assert flush_module.maybe_trigger_compilation(memory_home=root, now=now) is True

    assert commands[-1][-1] != first_fingerprint


def test_ordinary_compile_failure_retains_durable_backoff_for_watchdog(
    tmp_path, monkeypatch
):
    root, _daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    owner = next(command for command in commands if command[2] == "owner")
    token, fingerprint = owner[-2:]

    class FailedCompile:
        def wait(self, timeout=None):
            return 1

    assert not flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: FailedCompile(),
    )
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["fingerprint"] == fingerprint
    assert state["attempt_count"] == 1
    assert state["last_error_class"] == "compile_exit_1"
    assert state["next_retry_at"]
    assert state["status"] == "retry_wait"


def test_compile_process_spawn_failure_uses_same_durable_retry_budget(
    tmp_path, monkeypatch
):
    root, _daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]

    assert not flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("cannot start compile child")
        ),
    )

    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["attempt_count"] == 1
    assert state["last_error_class"] == "compile_launch_OSError"
    assert state["status"] == "retry_wait"


@pytest.mark.parametrize("completed_path", ["provider", "stage", "deferred_apply"])
def test_exit_zero_without_marker_progress_uses_durable_retry_budget(
    tmp_path, monkeypatch, completed_path
):
    root, _daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]

    class NoProgressCompile:
        def wait(self, timeout=None):
            return 0

    assert not flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: NoProgressCompile(),
    ), completed_path

    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["fingerprint"] == fingerprint
    assert state["attempt_count"] == 1
    assert state["last_error_class"] == "compile_no_progress"
    assert state["status"] == "retry_wait"


def test_exit_zero_no_progress_then_watchdog_retry_with_marker_succeeds(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    owner_token, fingerprint = _owner_command(commands)[-2:]
    watchdog_token = _watchdog_command(commands)[-1]

    class NoProgressCompile:
        def wait(self, timeout=None):
            return 0

    assert not flush_module.run_auto_compile_coordinator(
        root,
        owner_token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: NoProgressCompile(),
    )
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        retry_at = datetime.fromisoformat(_reservation(repository)["next_retry_at"])

    class MarkerProgressCompile:
        def wait(self, timeout=None):
            daily.write_text(
                "A\n<!-- @compiled-through:no-article -->\n", encoding="utf-8"
            )
            return 0

    def retry(target, token, current):
        return flush_module.run_auto_compile_coordinator(
            target,
            token,
            current,
            compile_launcher=lambda *_args, **_kwargs: MarkerProgressCompile(),
        )

    assert flush_module.run_auto_compile_watcher(
        root,
        watchdog_token,
        coordinator=retry,
        sleeper=lambda _seconds: (_ for _ in ()).throw(
            AssertionError("retry deadline has arrived")
        ),
        lock_probe=lambda _root: False,
        clock=lambda: retry_at + timedelta(microseconds=1),
    )
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        assert _reservation(repository) is None


def test_exit_zero_no_progress_reaches_inspectable_failed_cap(
    tmp_path, monkeypatch
):
    root, _daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    monkeypatch.setattr(flush_module, "AUTO_COMPILE_MAX_ATTEMPTS", 1)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]

    class NoProgressCompile:
        def wait(self, timeout=None):
            return 0

    assert not flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: NoProgressCompile(),
    )
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["status"] == "failed"
    assert state["attempt_count"] == 1
    assert state["last_error_class"] == "compile_no_progress"


def test_exit_zero_without_marker_but_with_new_content_resets_generation(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]

    class ChangedWithoutProgress:
        def wait(self, timeout=None):
            daily.write_text("A\nB\n", encoding="utf-8")
            return 0

    assert not flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: ChangedWithoutProgress(),
    )
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["fingerprint"] == flush_module._uncompiled_fingerprint(daily)
    assert state["status"] == "retry_wait"
    assert "attempt_count" not in state
    assert "last_error_class" not in state


def test_exit_zero_with_no_article_marker_progress_is_success(tmp_path, monkeypatch):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]

    class MarkerOnlyCompile:
        def wait(self, timeout=None):
            daily.write_text(
                "A\n<!-- @compiled-through:no-article -->\n", encoding="utf-8"
            )
            return 0

    assert flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: MarkerOnlyCompile(),
    )
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        assert _reservation(repository) is None


def test_daily_compile_read_distinguishes_unreadable_covered_and_uncompiled(tmp_path):
    daily = tmp_path / "daily.md"
    unreadable = flush_module._read_daily_compile_state(daily)
    assert unreadable.status == "unreadable"
    assert unreadable.fingerprint is None

    daily.write_text(
        "<!-- @compiled-through:first -->\n", encoding="utf-8"
    )
    covered = flush_module._read_daily_compile_state(daily)
    assert covered.status == "covered"
    assert covered.markers == ("first",)
    assert covered.fingerprint is None

    daily.write_text(
        "<!-- @compiled-through:first -->\nA\n", encoding="utf-8"
    )
    uncompiled = flush_module._read_daily_compile_state(daily)
    assert uncompiled.status == "uncompiled"
    assert uncompiled.markers == ("first",)
    assert uncompiled.fingerprint == flush_module._uncompiled_fingerprint(daily)


def test_unreadable_failure_observation_never_deletes_reservation(tmp_path):
    owner = "1" * 64
    watchdog = "2" * 64
    fingerprint = "a" * 64
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        assert repository.schedule_auto_compile(
            owner,
            watchdog,
            fingerprint,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(minutes=2),
        ) == ("owner", "watchdog")
        assert repository.record_auto_compile_failure(
            owner,
            fingerprint,
            "compile_read_unreadable",
            lambda: ("unreadable", None),
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(minutes=2),
        ) == "retry_wait"
        state = _reservation(repository)
    assert state["fingerprint"] == fingerprint
    assert state["attempt_count"] == 1
    assert state["last_error_class"] == "compile_read_unreadable"


def test_prelaunch_unreadable_daily_log_uses_durable_failure(tmp_path, monkeypatch):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]
    daily.unlink()

    assert not flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unreadable source must not launch")
        ),
    )
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["fingerprint"] == fingerprint
    assert state["attempt_count"] == 1
    assert state["last_error_class"] == "compile_read_unreadable"


@pytest.mark.parametrize(
    "mutation",
    [
        "unreadable",
        "replacement",
        "truncation",
        "replacement_covered",
        "truncation_covered",
        "no_marker_covered",
    ],
)
def test_exit_zero_invalid_or_unreadable_marker_state_uses_backoff(
    tmp_path, monkeypatch, mutation
):
    root, daily = _auto_compile_test_root(tmp_path)
    daily.write_text(
        "<!-- @compiled-through:first -->\nA\n", encoding="utf-8"
    )
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]

    class InvalidProgressCompile:
        def wait(self, timeout=None):
            if mutation == "unreadable":
                daily.unlink()
            elif mutation == "replacement":
                daily.write_text(
                    "<!-- @compiled-through:replacement -->\nA\n",
                    encoding="utf-8",
                )
            elif mutation == "replacement_covered":
                daily.write_text(
                    "<!-- @compiled-through:replacement -->\n", encoding="utf-8"
                )
            elif mutation == "truncation_covered":
                daily.write_text("\n", encoding="utf-8")
            elif mutation == "no_marker_covered":
                daily.write_text("", encoding="utf-8")
            else:
                daily.write_text("A\n", encoding="utf-8")
            return 0

    assert not flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: InvalidProgressCompile(),
    )
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["attempt_count"] == 1
    assert state["status"] == "retry_wait"
    assert state["last_error_class"] in {
        "compile_read_unreadable",
        "compile_invalid_marker_progress",
    }


def test_finish_rereads_same_log_after_covered_read_and_preserves_later_append(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]
    real_finish = QueueRepository.finish_auto_compile_generation
    launches = []
    appended = False

    def finish_with_later_append(repository, *args, **kwargs):
        nonlocal appended
        if not appended:
            appended = True
            daily.write_text(
                "A\n<!-- @compiled-through:first -->\nB\n", encoding="utf-8"
            )
            latest = flush_module._uncompiled_fingerprint(daily)
            assert latest is not None
            with QueueRepository(
                root / "scripts/jobs.sqlite3", sync_usage=False
            ) as scheduler:
                scheduler.schedule_auto_compile(
                    "7" * 64,
                    "8" * 64,
                    latest,
                    log_name=daily.name,
                    now=NOW,
                    expires_at=NOW + timedelta(minutes=2),
                )
        return real_finish(repository, *args, **kwargs)

    monkeypatch.setattr(
        QueueRepository, "finish_auto_compile_generation", finish_with_later_append
    )

    class Compile:
        def wait(self, timeout=None):
            launches.append(True)
            marker = "first" if len(launches) == 1 else "second"
            daily.write_text(
                daily.read_text(encoding="utf-8")
                + f"<!-- @compiled-through:{marker} -->\n",
                encoding="utf-8",
            )
            return 0

    assert flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: Compile(),
    )
    assert launches == [True, True]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        assert _reservation(repository) is None


def test_covered_recovery_rereads_same_log_inside_finish_transaction(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]
    daily.write_text(
        "A\n<!-- @compiled-through:external -->\n", encoding="utf-8"
    )
    real_finish = QueueRepository.finish_auto_compile_generation
    appended = False

    def finish_with_later_append(repository, *args, **kwargs):
        nonlocal appended
        if not appended:
            appended = True
            daily.write_text(
                daily.read_text(encoding="utf-8") + "B\n", encoding="utf-8"
            )
            latest = flush_module._uncompiled_fingerprint(daily)
            assert latest is not None
            with QueueRepository(
                root / "scripts/jobs.sqlite3", sync_usage=False
            ) as scheduler:
                scheduler.schedule_auto_compile(
                    "7" * 64,
                    "8" * 64,
                    latest,
                    log_name=daily.name,
                    now=NOW,
                    expires_at=NOW + timedelta(minutes=2),
                )
        return real_finish(repository, *args, **kwargs)

    monkeypatch.setattr(
        QueueRepository, "finish_auto_compile_generation", finish_with_later_append
    )
    launches = []

    class Compile:
        def wait(self, timeout=None):
            launches.append(True)
            daily.write_text(
                daily.read_text(encoding="utf-8")
                + "<!-- @compiled-through:later -->\n",
                encoding="utf-8",
            )
            return 0

    assert flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: Compile(),
    )
    assert launches == [True]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        assert _reservation(repository) is None


def test_invalid_covered_marker_is_retained_for_watcher_retry(tmp_path, monkeypatch):
    root, daily = _auto_compile_test_root(tmp_path)
    daily.write_text(
        "<!-- @compiled-through:first -->\nA\n", encoding="utf-8"
    )
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]

    class InvalidCompile:
        def wait(self, timeout=None):
            daily.write_text(
                "<!-- @compiled-through:replacement -->\n", encoding="utf-8"
            )
            return 0

    assert not flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: InvalidCompile(),
    )
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        retry_at = datetime.fromisoformat(state["next_retry_at"])
        state["expires_at"] = retry_at.isoformat(timespec="microseconds")
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )
        status, claimed = repository.poll_auto_compile_watcher(
            state["watcher_token"],
            "9" * 64,
            lambda reservation: flush_module._observe_auto_compile_content(
                root, dict(reservation)
            ),
            lambda _token: None,
            predecessor_token=None,
            now=retry_at + timedelta(microseconds=1),
            watcher_expires_at=retry_at + timedelta(minutes=2),
            owner_expires_at=retry_at + timedelta(minutes=2),
        )
    assert (status, claimed) == ("claimed", fingerprint)


def test_ordinary_failure_then_watchdog_retry_succeeds_without_new_session(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    watchdog_token = next(
        command[-1] for command in commands if command[2] == "watchdog"
    )
    owner = next(command for command in commands if command[2] == "owner")
    token, fingerprint = owner[-2:]

    class FailedCompile:
        def wait(self, timeout=None):
            return 2

    assert not flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: FailedCompile(),
    )
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        retry_at = datetime.fromisoformat(_reservation(repository)["next_retry_at"])
    retries = []

    def retry(_root, retry_token, retry_fingerprint):
        retries.append((retry_token, retry_fingerprint))
        daily.write_text(
            "A\n<!-- @compiled-through:retry -->\n", encoding="utf-8"
        )
        with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
            return repository.finish_auto_compile_generation(
                retry_token,
                retry_fingerprint,
                lambda: ("covered", None),
                now=retry_at,
                expires_at=retry_at + timedelta(minutes=2),
            ) is None

    assert flush_module.run_auto_compile_watcher(
        root,
        watchdog_token,
        coordinator=retry,
        sleeper=lambda _seconds: (_ for _ in ()).throw(
            AssertionError("retry deadline has already arrived")
        ),
        lock_probe=lambda _root: False,
        clock=lambda: retry_at + timedelta(microseconds=1),
    )
    assert len(retries) == 1
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        assert _reservation(repository) is None


def test_repeated_compile_failures_reach_inspectable_retry_cap(tmp_path):
    token = "1" * 64
    watchdog = "2" * 64
    fingerprint = "a" * 64
    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        assert repository.schedule_auto_compile(
            token,
            watchdog,
            fingerprint,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(minutes=2),
        ) == ("owner", "watchdog")
        for attempt in range(1, flush_module.AUTO_COMPILE_MAX_ATTEMPTS + 1):
            result = repository.record_auto_compile_failure(
                token,
                fingerprint,
                "provider_failure_with_untrusted_details" * 20,
                lambda: ("uncompiled", fingerprint),
                now=NOW + timedelta(minutes=attempt),
                expires_at=NOW + timedelta(minutes=attempt + 2),
            )
    assert result == "failed"
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["status"] == "failed"
    assert state["attempt_count"] == flush_module.AUTO_COMPILE_MAX_ATTEMPTS
    assert len(state["last_error_class"]) <= 64
    assert "next_retry_at" not in state


def test_new_fingerprint_resets_capped_auto_compile_retry(tmp_path):
    owner = "1" * 64
    watchdog = "2" * 64
    old = "a" * 64
    new = "b" * 64
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        assert repository.schedule_auto_compile(
            owner,
            watchdog,
            old,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(minutes=2),
        ) == ("owner", "watchdog")
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (
                json.dumps(
                    {
                        **_reservation(repository),
                        "status": "failed",
                        "attempt_count": flush_module.AUTO_COMPILE_MAX_ATTEMPTS,
                    }
                ),
            ),
        )
        roles = repository.schedule_auto_compile(
            "3" * 64,
            "4" * 64,
            new,
            log_name="2026-08-11.md",
            now=NOW + timedelta(minutes=3),
            expires_at=NOW + timedelta(minutes=5),
        )
        state = _reservation(repository)
    assert roles == ("owner", "watchdog")
    assert state["fingerprint"] == new
    assert state.get("attempt_count", 0) == 0
    assert state.get("status") != "failed"


def test_content_change_during_failed_compile_starts_fresh_generation(tmp_path):
    owner = "1" * 64
    watchdog = "2" * 64
    old = "a" * 64
    new = "b" * 64
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        assert repository.schedule_auto_compile(
            owner,
            watchdog,
            old,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(minutes=2),
        ) == ("owner", "watchdog")
        assert repository.record_auto_compile_failure(
            owner,
            old,
            "compile_exit_1",
            lambda: ("uncompiled", new),
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(minutes=2),
        ) == "retry_wait"
        state = _reservation(repository)
    assert state["fingerprint"] == new
    assert state["next_retry_at"] == (NOW + timedelta(seconds=1)).isoformat(
        timespec="microseconds"
    )
    assert "attempt_count" not in state
    assert "last_error_class" not in state


def test_marker_covered_during_ordinary_failure_retains_retry_request(tmp_path):
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        assert repository.schedule_auto_compile(
            "1" * 64,
            "2" * 64,
            "a" * 64,
            log_name="2026-08-11.md",
            required_marker_prefix=("first",),
            now=NOW,
            expires_at=NOW + timedelta(minutes=2),
        ) == ("owner", "watchdog")
        assert repository.record_auto_compile_failure(
            "1" * 64,
            "a" * 64,
            "compile_exit_1",
            lambda: ("covered", None, ("first", "second")),
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(minutes=2),
        ) == "retry_wait"
        state = _reservation(repository)
        assert state["fingerprint"] == "a" * 64
        assert state["required_marker_prefix"] == ["first", "second"]
        assert state["attempt_count"] == 1


def test_watchdog_crash_during_backoff_can_be_replaced_without_losing_retry(
    tmp_path,
):
    owner = "1" * 64
    watchdog = "2" * 64
    fingerprint = "a" * 64
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        assert repository.schedule_auto_compile(
            owner,
            watchdog,
            fingerprint,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(minutes=2),
        ) == ("owner", "watchdog")
        assert repository.record_auto_compile_failure(
            owner,
            fingerprint,
            "compile_exit_1",
            lambda: ("uncompiled", fingerprint),
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(minutes=2),
        ) == "retry_wait"
        state = _reservation(repository)
        retry_at = datetime.fromisoformat(state["next_retry_at"])
        state["watcher_expires_at"] = "2000-01-01T00:00:00.000000+00:00"
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )
        assert repository.schedule_auto_compile(
            "3" * 64,
            "4" * 64,
            fingerprint,
            log_name="2026-08-11.md",
            now=NOW + timedelta(seconds=2),
            expires_at=NOW + timedelta(minutes=2),
        ) == ("watchdog",)
        recovered = _reservation(repository)
    assert recovered["watcher_token"] == "4" * 64
    assert datetime.fromisoformat(recovered["next_retry_at"]) == retry_at
    assert recovered["attempt_count"] == 1


def test_running_auto_compile_renews_reservation_lease(tmp_path, monkeypatch):
    root = tmp_path / "memory"
    (root / "scripts").mkdir(parents=True)
    (root / "daily").mkdir()
    (root / "scripts/compile.py").write_text("# compiler\n", encoding="utf-8")
    (root / "scripts/auto-compile.py").write_text("# coordinator\n", encoding="utf-8")
    (root / "daily/2026-08-11.md").write_text("Uncompiled content.\n", encoding="utf-8")
    commands = []
    monkeypatch.setattr(flush_module, "count_interactive_agent_sessions", lambda: 0)
    monkeypatch.setattr(flush_module, "_resolve_tty_path", lambda: None)
    monkeypatch.setattr(flush_module, "notify_terminal", lambda _message: None)
    monkeypatch.setattr(
        flush_module.subprocess,
        "Popen",
        lambda command, **_options: commands.append(command),
    )
    now = datetime(2026, 8, 11, 16, 31, tzinfo=timezone.utc)
    assert flush_module.maybe_trigger_compilation(memory_home=root, now=now) is True
    token, fingerprint = commands[-1][-2:]
    renewals = []
    real_renew = QueueRepository.renew_auto_compile

    def observed_renew(self, *args, **kwargs):
        renewals.append((args, kwargs))
        return real_renew(self, *args, **kwargs)

    monkeypatch.setattr(QueueRepository, "renew_auto_compile", observed_renew)

    class SlowFailedCompile:
        waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["compile"], timeout)
            return 1

    assert not flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: SlowFailedCompile(),
    )
    assert len(renewals) == 1


def test_end_of_day_scheduler_does_not_reserve_while_queue_has_active_work(
    tmp_path, monkeypatch
):
    root = tmp_path / "memory"
    (root / "scripts").mkdir(parents=True)
    (root / "daily").mkdir()
    (root / "scripts/compile.py").write_text("# compiler\n", encoding="utf-8")
    (root / "scripts/auto-compile.py").write_text("# coordinator\n", encoding="utf-8")
    (root / "daily/2026-08-11.md").write_text("Uncompiled content.\n", encoding="utf-8")
    with QueueRepository(
        root / "scripts/jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        enqueue_capture_job(repository, root, session_id="pending-before-reserve")
    monkeypatch.setattr(flush_module, "count_interactive_agent_sessions", lambda: 0)
    monkeypatch.setattr(
        flush_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not launch")
        ),
    )

    assert (
        flush_module.maybe_trigger_compilation(
            memory_home=root,
            now=datetime(2026, 8, 11, 16, 31, tzinfo=timezone.utc),
        )
        is False
    )


def test_auto_compile_reservation_renewal_prevents_overlap_and_expiry_recovers(
    tmp_path,
):
    first = "1" * 64
    second = "2" * 64
    fingerprint = "a" * 64
    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        assert repository.reserve_auto_compile(
            first,
            fingerprint,
            now=NOW,
            expires_at=NOW + timedelta(seconds=5),
        )
        assert repository.renew_auto_compile(
            first,
            fingerprint,
            now=NOW + timedelta(seconds=4),
            expires_at=NOW + timedelta(seconds=9),
        )
        assert not repository.reserve_auto_compile(
            second,
            fingerprint,
            now=NOW + timedelta(seconds=6),
            expires_at=NOW + timedelta(seconds=11),
        )
        assert repository.reserve_auto_compile(
            second,
            fingerprint,
            now=NOW + timedelta(seconds=10),
            expires_at=NOW + timedelta(seconds=15),
        )


def _auto_compile_test_root(tmp_path):
    root = tmp_path / "memory"
    (root / "scripts").mkdir(parents=True)
    (root / "daily").mkdir()
    (root / "scripts/compile.py").write_text("# compiler\n", encoding="utf-8")
    (root / "scripts/auto-compile.py").write_text(
        "# coordinator\n", encoding="utf-8"
    )
    daily = root / "daily/2026-08-11.md"
    daily.write_text("A\n", encoding="utf-8")
    return root, daily


def _configure_auto_compile_test(monkeypatch, commands):
    monkeypatch.setattr(flush_module, "count_interactive_agent_sessions", lambda: 0)
    monkeypatch.setattr(flush_module, "_resolve_tty_path", lambda: None)
    monkeypatch.setattr(flush_module, "notify_terminal", lambda _message: None)
    monkeypatch.setattr(
        flush_module.subprocess,
        "Popen",
        lambda command, **_options: commands.append(command),
    )


def _owner_command(commands):
    return next(command for command in commands if command[2] == "owner")


def _watchdog_command(commands):
    return next(command for command in commands if command[2] == "watchdog")


def _schedule_auto_compile(root):
    return flush_module.maybe_trigger_compilation(
        memory_home=root,
        now=datetime(2026, 8, 11, 16, 31, tzinfo=timezone.utc),
    )


def _reservation(repository):
    row = repository._connection.execute(
        "SELECT value FROM queue_metadata WHERE key = 'auto_compile_reservation'"
    ).fetchone()
    return json.loads(row["value"]) if row is not None else None


def test_active_reservation_persists_latest_pending_fingerprint(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, first = _owner_command(commands)[-2:]

    daily.write_text("A\nB\n", encoding="utf-8")
    second = flush_module._uncompiled_fingerprint(daily)
    assert _schedule_auto_compile(root) is False
    daily.write_text("A\nB\nC\n", encoding="utf-8")
    latest = flush_module._uncompiled_fingerprint(daily)
    assert _schedule_auto_compile(root) is False

    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        reservation = _reservation(repository)
    assert reservation["fingerprint"] == first
    assert reservation["pending_fingerprint"] == latest
    assert reservation["token"] == token
    assert reservation["watcher_token"] == _watchdog_command(commands)[-1]
    assert latest != second != first


def test_coordinator_promotes_pending_when_baseline_changes_before_launch(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]
    daily.write_text("A\nB\n", encoding="utf-8")
    assert _schedule_auto_compile(root) is False
    launches = []

    class SuccessfulCompile:
        def wait(self, timeout=None):
            daily.write_text(
                "A\nB\n<!-- @compiled-through:done -->\n", encoding="utf-8"
            )
            return 0

    assert flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: (
            launches.append(True) or SuccessfulCompile()
        ),
    )
    assert launches == [True]


def test_coordinator_coalesces_multiple_appends_into_one_follow_up_compile(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]
    launches = []

    class CompileProcess:
        def __init__(self, generation):
            self.generation = generation

        def wait(self, timeout=None):
            if self.generation == 1:
                daily.write_text("A\nB\n", encoding="utf-8")
                assert _schedule_auto_compile(root) is False
                daily.write_text("A\nB\nC\n", encoding="utf-8")
                assert _schedule_auto_compile(root) is False
                daily.write_text(
                    "A\n<!-- @compiled-through:first -->\nB\nC\n",
                    encoding="utf-8",
                )
            else:
                daily.write_text(
                    "A\n<!-- @compiled-through:first -->\nB\nC\n"
                    "<!-- @compiled-through:second -->\n",
                    encoding="utf-8",
                )
            return 0

    def launch(*_args, **_kwargs):
        launches.append(len(launches) + 1)
        return CompileProcess(launches[-1])

    assert flush_module.run_auto_compile_coordinator(
        root, token, fingerprint, compile_launcher=launch
    )
    assert launches == [1, 2]


def test_coordinator_does_not_follow_up_when_first_compile_covers_pending(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]
    launches = []

    class SuccessfulCompile:
        def wait(self, timeout=None):
            daily.write_text("A\nB\n", encoding="utf-8")
            assert _schedule_auto_compile(root) is False
            daily.write_text(
                "A\nB\n<!-- @compiled-through:all -->\n", encoding="utf-8"
            )
            return 0

    assert flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: (
            launches.append(True) or SuccessfulCompile()
        ),
    )
    assert launches == [True]


def test_coordinator_defers_changed_generation_after_first_compile_failure(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]
    launches = []

    class FailedCompile:
        def __init__(self, generation):
            self.generation = generation

        def wait(self, timeout=None):
            if self.generation == 1:
                daily.write_text("A\nB\n", encoding="utf-8")
                assert _schedule_auto_compile(root) is False
            return 1

    def launch(*_args, **_kwargs):
        launches.append(len(launches) + 1)
        return FailedCompile(launches[-1])

    assert not flush_module.run_auto_compile_coordinator(
        root, token, fingerprint, compile_launcher=launch
    )
    assert launches == [1]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["status"] == "retry_wait"
    assert state["fingerprint"] == flush_module._uncompiled_fingerprint(daily)


def test_coordinator_defers_handoff_when_new_queue_job_arrives(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]
    launches = []

    class SuccessfulCompile:
        def wait(self, timeout=None):
            daily.write_text("A\nB\n", encoding="utf-8")
            assert _schedule_auto_compile(root) is False
            with QueueRepository(
                root / "scripts/jobs.sqlite3", sync_usage=False
            ) as repository:
                enqueue_capture_job(
                    repository, root, session_id="arrived-during-handoff"
                )
            daily.write_text(
                "A\n<!-- @compiled-through:first -->\nB\n", encoding="utf-8"
            )
            return 0

    assert flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=lambda *_args, **_kwargs: (
            launches.append(True) or SuccessfulCompile()
        ),
    )
    assert launches == [True]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["status"] == "queue_wait"
    assert state["watcher_token"] == _watchdog_command(commands)[-1]
    assert state["pending_fingerprint"] != fingerprint


def test_expired_coordinator_with_pending_is_replaced_by_latest_reservation(tmp_path):
    first_token = "1" * 64
    replacement_token = "2" * 64
    first = "a" * 64
    pending = "b" * 64
    latest = "c" * 64
    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        assert repository.reserve_auto_compile(
            first_token,
            first,
            now=NOW,
            expires_at=NOW + timedelta(seconds=5),
        )
        assert not repository.reserve_auto_compile(
            "3" * 64,
            pending,
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(seconds=6),
        )
        assert repository.reserve_auto_compile(
            replacement_token,
            latest,
            now=NOW + timedelta(seconds=6),
            expires_at=NOW + timedelta(seconds=11),
        )
        assert _reservation(repository) == {
            "expires_at": "2026-08-11T12:00:11.000000+00:00",
            "fingerprint": latest,
            "token": replacement_token,
        }


def test_auto_compile_handoff_reads_current_fingerprint_inside_transaction(tmp_path):
    token = "1" * 64
    first = "a" * 64
    latest = "b" * 64
    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        assert repository.reserve_auto_compile(
            token,
            first,
            now=NOW,
            expires_at=NOW + timedelta(seconds=5),
        )

        def observe_current():
            assert repository._connection.in_transaction
            return "uncompiled", latest

        assert repository.finish_auto_compile_generation(
            token,
            first,
            observe_current,
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(seconds=6),
        ) == latest


def test_completed_owner_promotes_pending_fingerprint_from_a_different_log(tmp_path):
    token = "1" * 64
    first = "a" * 64
    pending = "b" * 64
    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        assert repository.request_auto_compile(
            token,
            first,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(seconds=5),
        ) == "owner"
        assert repository.request_auto_compile(
            "2" * 64,
            pending,
            log_name="2026-08-12.md",
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(seconds=6),
        ) == "watcher"

        assert repository.finish_auto_compile_generation(
            token,
            first,
            lambda: ("covered", None),
            now=NOW + timedelta(seconds=2),
            expires_at=NOW + timedelta(seconds=7),
        ) == pending
        state = _reservation(repository)
    assert state["fingerprint"] == pending
    assert state["log_name"] == "2026-08-12.md"
    assert "pending_fingerprint" not in state


def test_residual_active_content_preserves_pending_fingerprint_for_next_log(tmp_path):
    token = "1" * 64
    first = "a" * 64
    residual = "c" * 64
    pending = "b" * 64
    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        assert repository.request_auto_compile(
            token,
            first,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(seconds=5),
        ) == "owner"
        assert repository.request_auto_compile(
            "2" * 64,
            pending,
            log_name="2026-08-12.md",
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(seconds=6),
        ) == "watcher"

        assert repository.finish_auto_compile_generation(
            token,
            first,
            lambda: ("uncompiled", residual),
            now=NOW + timedelta(seconds=2),
            expires_at=NOW + timedelta(seconds=7),
        ) == residual
        state = _reservation(repository)
        assert state["pending_fingerprint"] == pending
        assert state["pending_log_name"] == "2026-08-12.md"

        assert repository.finish_auto_compile_generation(
            token,
            residual,
            lambda: ("covered", None),
            now=NOW + timedelta(seconds=3),
            expires_at=NOW + timedelta(seconds=8),
        ) == pending


def test_deferred_owner_promotes_pending_fingerprint_from_a_different_log(tmp_path):
    token = "1" * 64
    first = "a" * 64
    pending = "b" * 64
    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        assert repository.request_auto_compile(
            token,
            first,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(seconds=5),
        ) == "owner"
        assert repository.request_auto_compile(
            "2" * 64,
            pending,
            log_name="2026-08-12.md",
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(seconds=6),
        ) == "watcher"

        assert repository.defer_auto_compile_generation(
            token,
            first,
            lambda: ("covered", None),
            now=NOW + timedelta(seconds=2),
            expires_at=NOW + timedelta(seconds=7),
        ) == pending
        state = _reservation(repository)
    assert state["fingerprint"] == pending
    assert state["log_name"] == "2026-08-12.md"


def test_exit_75_continues_with_promoted_pending_log_from_next_day(
    tmp_path, monkeypatch
):
    root, first_log = _auto_compile_test_root(tmp_path)
    second_log = root / "daily/2026-08-12.md"
    second_log.write_text("B\n", encoding="utf-8")
    first = flush_module._uncompiled_fingerprint(first_log)
    pending = flush_module._uncompiled_fingerprint(second_log)
    token = "1" * 64
    lease_now = datetime.now(timezone.utc)
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        assert repository.request_auto_compile(
            token,
            first,
            log_name=first_log.name,
            now=lease_now,
            expires_at=lease_now + timedelta(minutes=5),
        ) == "owner"
        assert repository.request_auto_compile(
            "2" * 64,
            pending,
            log_name=second_log.name,
            now=lease_now + timedelta(seconds=1),
            expires_at=lease_now + timedelta(minutes=5),
        ) == "watcher"
    monkeypatch.setattr(flush_module, "count_interactive_agent_sessions", lambda: 0)
    launches = []
    compiler_lock_held = [False]

    class CompileProcess:
        def wait(self, timeout=None):
            return 75

    def defer(_seconds):
        if len(launches) == 1:
            first_log.write_text(
                "A\n<!-- @compiled-through:first -->\n", encoding="utf-8"
            )
        else:
            second_log.write_text(
                "B\n<!-- @compiled-through:second -->\n", encoding="utf-8"
            )
        compiler_lock_held[0] = False

    def launch(*_args, **_kwargs):
        launches.append(True)
        compiler_lock_held[0] = True
        return CompileProcess()

    assert flush_module.run_auto_compile_coordinator(
        root,
        token,
        first,
        compile_launcher=launch,
        sleeper=defer,
        lock_probe=lambda _root: compiler_lock_held[0],
    )
    assert launches == [True, True]


def test_later_idle_drain_reuses_exactly_one_durable_watchdog(tmp_path, monkeypatch):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)

    assert _schedule_auto_compile(root) is True
    daily.write_text("A\nB\n", encoding="utf-8")
    assert _schedule_auto_compile(root) is False
    daily.write_text("A\nB\nC\n", encoding="utf-8")
    assert _schedule_auto_compile(root) is False

    assert [command[2] for command in commands] == ["watchdog", "owner"]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        reservation = _reservation(repository)
    assert reservation["pending_fingerprint"] == flush_module._uncompiled_fingerprint(
        daily
    )
    assert reservation["watcher_token"] == _watchdog_command(commands)[-1]


def test_owner_or_watcher_role_is_selected_in_one_queue_transaction(tmp_path):
    first = "a" * 64
    latest = "b" * 64
    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        assert repository.request_auto_compile(
            "1" * 64,
            first,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(seconds=5),
        ) == "owner"
        assert repository.request_auto_compile(
            "2" * 64,
            latest,
            log_name="2026-08-11.md",
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(seconds=6),
        ) == "watcher"
        assert repository.request_auto_compile(
            "3" * 64,
            latest,
            log_name="2026-08-11.md",
            now=NOW + timedelta(seconds=2),
            expires_at=NOW + timedelta(seconds=7),
        ) is None
        state = _reservation(repository)
    assert state["token"] == "1" * 64
    assert state["watcher_token"] == "2" * 64
    assert state["pending_fingerprint"] == latest


def test_owner_spawn_failure_preserves_an_already_registered_watcher(tmp_path):
    owner = "1" * 64
    watcher = "2" * 64
    fingerprint = "a" * 64
    with QueueRepository(
        tmp_path / "jobs.sqlite3", clock=lambda: NOW, sync_usage=False
    ) as repository:
        assert repository.request_auto_compile(
            owner,
            fingerprint,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(seconds=5),
        ) == "owner"
        assert repository.request_auto_compile(
            watcher,
            fingerprint,
            log_name="2026-08-11.md",
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(seconds=6),
        ) == "watcher"

        assert repository.fail_auto_compile_owner_spawn(
            owner, fingerprint, now=NOW + timedelta(seconds=2)
        )
        state = _reservation(repository)
    assert state["watcher_token"] == watcher
    assert datetime.fromisoformat(state["expires_at"]) <= NOW + timedelta(seconds=2)


def test_expired_watcher_can_be_replaced_by_subsequent_idle_drain(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    daily.write_text("A\nB\n", encoding="utf-8")
    first_watcher = _watchdog_command(commands)[-1]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        state["watcher_expires_at"] = "2000-01-01T00:00:00.000000+00:00"
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )

    assert _schedule_auto_compile(root) is True
    assert commands[-1][2] == "watchdog"
    assert commands[-1][-1] != first_watcher


def test_watcher_takes_over_expired_owner_without_another_session(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    daily.write_text("A\nB\n", encoding="utf-8")
    watcher_token = _watchdog_command(commands)[-1]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        state["expires_at"] = "2000-01-01T00:00:00.000000+00:00"
        state["watcher_expires_at"] = "2999-01-01T00:00:00.000000+00:00"
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )
    takeovers = []

    assert flush_module.run_auto_compile_watcher(
        root,
        watcher_token,
        coordinator=lambda _root, token, fingerprint: (
            takeovers.append((token, fingerprint)) or True
        ),
        sleeper=lambda _seconds: (_ for _ in ()).throw(
            AssertionError("expired owner should be claimed immediately")
        ),
        lock_probe=lambda _root: False,
    )
    assert takeovers == [(watcher_token, flush_module._uncompiled_fingerprint(daily))]


def test_each_crashed_takeover_owner_has_an_independent_successor_watchdog(
    tmp_path, monkeypatch
):
    root, _daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    first_watchdog = _watchdog_command(commands)[-1]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        state["expires_at"] = "2000-01-01T00:00:00.000000+00:00"
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )

    class OwnerCrash(RuntimeError):
        pass

    def crash(*_args):
        raise OwnerCrash("takeover owner crashed")

    with pytest.raises(OwnerCrash):
        flush_module.run_auto_compile_watcher(
            root,
            first_watchdog,
            coordinator=crash,
            sleeper=lambda _seconds: (_ for _ in ()).throw(
                AssertionError("expired owner must be claimed immediately")
            ),
            lock_probe=lambda _root: False,
            clock=lambda: datetime.now(timezone.utc),
    )
    assert commands[-1][2] == "watchdog"
    second_watchdog = commands[-1][-2]
    assert commands[-1][-1] == first_watchdog
    assert second_watchdog != first_watchdog
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        assert state["token"] == first_watchdog
        assert state["watcher_token"] == second_watchdog
        state["expires_at"] = "2000-01-01T00:00:00.000000+00:00"
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )

    with pytest.raises(OwnerCrash):
        flush_module.run_auto_compile_watcher(
            root,
            second_watchdog,
            coordinator=crash,
            sleeper=lambda _seconds: (_ for _ in ()).throw(
                AssertionError("second crashed owner must be reclaimed immediately")
            ),
            lock_probe=lambda _root: False,
            clock=lambda: datetime.now(timezone.utc),
        )
    third_watchdog = commands[-1][-2]
    assert commands[-1][2] == "watchdog"
    assert commands[-1][-1] == second_watchdog
    assert third_watchdog not in {first_watchdog, second_watchdog}
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
    assert state["token"] == second_watchdog
    assert state["watcher_token"] == third_watchdog


@pytest.mark.parametrize(
    "failure",
    [sqlite3.OperationalError("commit failed"), SystemExit("pre-commit crash")],
)
def test_spawned_successor_recovers_takeover_after_transaction_rollback(
    tmp_path, failure
):
    owner = "1" * 64
    predecessor = "2" * 64
    successor = "3" * 64
    next_successor = "4" * 64
    fingerprint = "a" * 64
    database = tmp_path / "jobs.sqlite3"
    with QueueRepository(database, sync_usage=False) as repository:
        assert repository.schedule_auto_compile(
            owner,
            predecessor,
            fingerprint,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(seconds=2),
        ) == ("owner", "watchdog")

        spawned = []

        def spawn_then_fail(token):
            spawned.append(token)
            raise failure

        with pytest.raises(type(failure)):
            repository.poll_auto_compile_watcher(
                predecessor,
                successor,
                lambda _reservation: (
                    "uncompiled",
                    {"fingerprint": fingerprint, "log_name": "2026-08-11.md"},
                ),
                spawn_then_fail,
                predecessor_token=None,
                now=NOW + timedelta(seconds=3),
                watcher_expires_at=NOW + timedelta(seconds=8),
                owner_expires_at=NOW + timedelta(seconds=8),
            )
        state = _reservation(repository)
    assert spawned == [successor]
    assert state["watcher_token"] == predecessor
    assert state["token"] == owner

    next_spawns = []
    with QueueRepository(database, sync_usage=False) as repository:
        status, claimed = repository.poll_auto_compile_watcher(
            successor,
            next_successor,
            lambda _reservation: (
                "uncompiled",
                {"fingerprint": fingerprint, "log_name": "2026-08-11.md"},
            ),
            next_spawns.append,
            predecessor_token=predecessor,
            now=NOW + timedelta(seconds=4),
            watcher_expires_at=NOW + timedelta(seconds=9),
            owner_expires_at=NOW + timedelta(seconds=9),
        )
        recovered = _reservation(repository)
    assert (status, claimed) == ("claimed", fingerprint)
    assert next_spawns == [next_successor]
    assert recovered["token"] == successor
    assert recovered["watcher_token"] == next_successor


def test_successor_defers_while_predecessor_watcher_lease_is_live(tmp_path):
    owner = "1" * 64
    predecessor = "2" * 64
    successor = "3" * 64
    with QueueRepository(tmp_path / "jobs.sqlite3", sync_usage=False) as repository:
        assert repository.schedule_auto_compile(
            owner,
            predecessor,
            "a" * 64,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(seconds=10),
        ) == ("owner", "watchdog")
        status, claimed = repository.poll_auto_compile_watcher(
            successor,
            "4" * 64,
            lambda _reservation: (_ for _ in ()).throw(
                AssertionError("contender must not observe before predecessor expiry")
            ),
            lambda _token: (_ for _ in ()).throw(
                AssertionError("contender must not spawn before predecessor expiry")
            ),
            predecessor_token=predecessor,
            now=NOW + timedelta(seconds=1),
            watcher_expires_at=NOW + timedelta(seconds=11),
            owner_expires_at=NOW + timedelta(seconds=11),
        )
    assert (status, claimed) == ("wait", None)


def test_successor_recovers_when_rolled_back_predecessor_was_pending_contender(
    tmp_path,
):
    owner = "1" * 64
    watcher = "2" * 64
    contender = "3" * 64
    successor = "4" * 64
    next_successor = "5" * 64
    fingerprint = "a" * 64
    database = tmp_path / "jobs.sqlite3"
    with QueueRepository(database, sync_usage=False) as repository:
        assert repository.schedule_auto_compile(
            owner,
            watcher,
            fingerprint,
            log_name="2026-08-11.md",
            now=NOW,
            expires_at=NOW + timedelta(seconds=2),
        ) == ("owner", "watchdog")
        state = _reservation(repository)
        state.update(
            {
                "status": "queue_wait",
                "contender_token": contender,
                "contender_predecessor_token": watcher,
                "contender_expires_at": (
                    NOW + timedelta(seconds=4)
                ).isoformat(timespec="microseconds"),
            }
        )
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )

        def spawned_then_crashed(token):
            assert token == successor
            raise SystemExit("contender died before takeover commit")

        with pytest.raises(SystemExit):
            repository.poll_auto_compile_watcher(
                contender,
                successor,
                lambda _reservation: (
                    "uncompiled",
                    {"fingerprint": fingerprint, "log_name": "2026-08-11.md"},
                ),
                spawned_then_crashed,
                predecessor_token=watcher,
                now=NOW + timedelta(seconds=3),
                watcher_expires_at=NOW + timedelta(seconds=4),
                owner_expires_at=NOW + timedelta(seconds=8),
            )
        rolled_back = _reservation(repository)
    assert rolled_back["watcher_token"] == watcher
    assert rolled_back["contender_token"] == contender

    spawns = []
    with QueueRepository(database, sync_usage=False) as repository:
        status, claimed = repository.poll_auto_compile_watcher(
            successor,
            next_successor,
            lambda _reservation: (
                "uncompiled",
                {"fingerprint": fingerprint, "log_name": "2026-08-11.md"},
            ),
            spawns.append,
            predecessor_token=contender,
            now=NOW + timedelta(seconds=5),
            watcher_expires_at=NOW + timedelta(seconds=10),
            owner_expires_at=NOW + timedelta(seconds=10),
        )
        recovered = _reservation(repository)
    assert (status, claimed) == ("claimed", fingerprint)
    assert spawns == [next_successor]
    assert recovered["token"] == successor
    assert recovered["watcher_token"] == next_successor


@pytest.mark.parametrize("wait_status", ["queue_wait", "retry_wait"])
def test_later_drain_preprovisions_contender_that_survives_watcher_death(
    tmp_path, monkeypatch, wait_status
):
    root, _daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    predecessor = _watchdog_command(commands)[-1]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        state["status"] = wait_status
        state["expires_at"] = "2000-01-01T00:00:00.000000+00:00"
        state["watcher_expires_at"] = "2999-01-01T00:00:00.000000+00:00"
        if wait_status == "retry_wait":
            state["next_retry_at"] = "2000-01-01T00:00:00.000000+00:00"
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )

    assert _schedule_auto_compile(root) is True
    contender_command = commands[-1]
    assert contender_command[2] == "contender"
    contender = contender_command[-2]
    assert contender_command[-1] == predecessor
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        assert state["contender_token"] == contender
        state["watcher_expires_at"] = "2000-01-01T00:00:00.000000+00:00"
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )

    resumed = []

    def resume(_root, token, fingerprint):
        resumed.append((token, fingerprint))
        with QueueRepository(
            root / "scripts/jobs.sqlite3", sync_usage=False
        ) as repository:
            repository.release_auto_compile(token, fingerprint)
        return True

    assert flush_module.run_auto_compile_watcher(
        root,
        contender,
        predecessor_token=predecessor,
        coordinator=resume,
        sleeper=lambda _seconds: (_ for _ in ()).throw(
            AssertionError("pre-provisioned contender must take over after expiry")
        ),
        lock_probe=lambda _root: False,
        clock=lambda: datetime.now(timezone.utc),
    )
    assert resumed[0][0] == contender


def test_watcher_retries_after_sqlite_takeover_error(tmp_path, monkeypatch):
    root, _daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    watchdog = _watchdog_command(commands)[-1]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        state["expires_at"] = "2000-01-01T00:00:00.000000+00:00"
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )
    real_poll = QueueRepository.poll_auto_compile_watcher
    calls = []

    def flaky_poll(self, *args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise sqlite3.OperationalError("commit failed")
        return real_poll(self, *args, **kwargs)

    monkeypatch.setattr(QueueRepository, "poll_auto_compile_watcher", flaky_poll)

    def finish(_root, token, fingerprint):
        with QueueRepository(
            root / "scripts/jobs.sqlite3", sync_usage=False
        ) as repository:
            repository.release_auto_compile(token, fingerprint)
        return True

    assert flush_module.run_auto_compile_watcher(
        root,
        watchdog,
        coordinator=finish,
        sleeper=lambda _seconds: None,
        lock_probe=lambda _root: False,
        clock=lambda: datetime.now(timezone.utc),
    )
    assert len(calls) == 2


def test_delayed_live_watcher_can_renew_if_its_token_was_not_replaced(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    daily.write_text("A\nB\n", encoding="utf-8")
    watcher_token = _watchdog_command(commands)[-1]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        state["expires_at"] = "2000-01-01T00:00:00.000000+00:00"
        state["watcher_expires_at"] = "2000-01-01T00:00:00.000000+00:00"
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )
    takeovers = []

    assert flush_module.run_auto_compile_watcher(
        root,
        watcher_token,
        coordinator=lambda _root, token, fingerprint: (
            takeovers.append((token, fingerprint)) or True
        ),
        sleeper=lambda _seconds: (_ for _ in ()).throw(
            AssertionError("unreplaced watcher should reclaim immediately")
        ),
        lock_probe=lambda _root: False,
    )
    assert takeovers == [(watcher_token, flush_module._uncompiled_fingerprint(daily))]


@pytest.mark.parametrize("resolution", ["marker", "lock_release"])
def test_exit_75_defers_without_deleting_request(tmp_path, monkeypatch, resolution):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    token, fingerprint = _owner_command(commands)[-2:]
    launches = []
    lock_held = [True]
    defer_calls = []

    class CompileProcess:
        def __init__(self, returncode):
            self.returncode = returncode

        def wait(self, timeout=None):
            if self.returncode == 0:
                daily.write_text(
                    "A\n<!-- @compiled-through:retried -->\n", encoding="utf-8"
                )
            return self.returncode

    def launch(*_args, **_kwargs):
        launches.append(len(launches))
        return CompileProcess(75 if len(launches) == 1 else 0)

    def defer(_seconds):
        defer_calls.append(True)
        with QueueRepository(
            root / "scripts/jobs.sqlite3", sync_usage=False
        ) as repository:
            state = _reservation(repository)
        assert state["token"] == token
        assert state["fingerprint"] == fingerprint
        if resolution == "marker":
            if len(defer_calls) == 1:
                daily.write_text(
                    "A\n<!-- @compiled-through:orphan -->\n", encoding="utf-8"
                )
            else:
                daily.write_text("A\n", encoding="utf-8")
                lock_held[0] = False
        else:
            lock_held[0] = False

    assert flush_module.run_auto_compile_coordinator(
        root,
        token,
        fingerprint,
        compile_launcher=launch,
        sleeper=defer,
        lock_probe=lambda _root: lock_held[0],
    )
    assert launches == [0, 1]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        assert _reservation(repository) is None


def test_watcher_exits_cleanly_when_owner_finishes_and_clears_reservation(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    daily.write_text("A\nB\n", encoding="utf-8")
    watcher_token = _watchdog_command(commands)[-1]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        assert repository.release_auto_compile(
            state["token"], state["fingerprint"]
        )

    assert flush_module.run_auto_compile_watcher(
        root,
        watcher_token,
        coordinator=lambda *_args: (_ for _ in ()).throw(
            AssertionError("cleared watcher must not take over")
        ),
        sleeper=lambda _seconds: (_ for _ in ()).throw(
            AssertionError("cleared watcher must exit without polling")
        ),
        lock_probe=lambda _root: False,
    )


def test_watcher_waits_when_marker_is_visible_before_owner_lease_expires(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    daily.write_text("A\nB\n", encoding="utf-8")
    watcher_token = _watchdog_command(commands)[-1]
    daily.write_text(
        "A\nB\n<!-- @compiled-through:owner -->\n", encoding="utf-8"
    )

    class ExpectedWait(RuntimeError):
        pass

    with pytest.raises(ExpectedWait):
        flush_module.run_auto_compile_watcher(
            root,
            watcher_token,
            coordinator=lambda *_args: (_ for _ in ()).throw(
                AssertionError("covered content must not be taken over")
            ),
            sleeper=lambda _seconds: (_ for _ in ()).throw(ExpectedWait()),
            lock_probe=lambda _root: True,
        )
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        assert _reservation(repository) is not None


@pytest.mark.parametrize("blocker", ["owner", "compiler_lock"])
def test_watcher_never_clears_visible_marker_while_owner_or_compiler_is_live(
    tmp_path, blocker
):
    root, daily = _auto_compile_test_root(tmp_path)
    owner = "1" * 64
    watcher = "2" * 64
    successor = "3" * 64
    fingerprint = flush_module._uncompiled_fingerprint(daily)
    assert fingerprint is not None
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        repository.schedule_auto_compile(
            owner,
            watcher,
            fingerprint,
            log_name=daily.name,
            required_marker_prefix=(),
            now=(NOW if blocker == "owner" else NOW - timedelta(minutes=3)),
            expires_at=(
                NOW + timedelta(minutes=2)
                if blocker == "owner"
                else NOW - timedelta(microseconds=1)
            ),
        )
        daily.write_text(
            "A\n<!-- @compiled-through:staged -->\n", encoding="utf-8"
        )
        status, claimed = repository.poll_auto_compile_watcher(
            watcher,
            successor,
            lambda reservation: flush_module._observe_auto_compile_content(
                root, dict(reservation)
            ),
            lambda _token: (_ for _ in ()).throw(
                AssertionError("live compiler state must not be claimed")
            ),
            predecessor_token=None,
            compiler_lock_held=blocker == "compiler_lock",
            now=NOW,
            watcher_expires_at=NOW + timedelta(minutes=3),
            owner_expires_at=NOW + timedelta(minutes=3),
        )
        assert (status, claimed) == ("wait", None)
        assert _reservation(repository) is not None
        daily.write_text("A\n", encoding="utf-8")
        state = _reservation(repository)
        state["expires_at"] = (NOW - timedelta(microseconds=1)).isoformat()
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )
        launches = []
        status, claimed = repository.poll_auto_compile_watcher(
            watcher,
            successor,
            lambda reservation: flush_module._observe_auto_compile_content(
                root, dict(reservation)
            ),
            launches.append,
            predecessor_token=None,
            compiler_lock_held=False,
            now=NOW,
            watcher_expires_at=NOW + timedelta(minutes=3),
            owner_expires_at=NOW + timedelta(minutes=3),
        )
    assert (status, claimed) == ("claimed", fingerprint)
    assert launches == [successor]


def test_observer_retains_required_prefix_across_uncompiled_replacement(tmp_path):
    root, daily = _auto_compile_test_root(tmp_path)
    pending_log = root / "daily/2026-08-12.md"
    pending_log.write_text("Pending\n", encoding="utf-8")
    original = flush_module._uncompiled_fingerprint(daily)
    pending = flush_module._uncompiled_fingerprint(pending_log)
    assert original is not None
    assert pending is not None
    reservation = {
        "fingerprint": original,
        "log_name": daily.name,
        "required_marker_prefix": ["first"],
        "pending_fingerprint": pending,
        "pending_log_name": pending_log.name,
        "pending_required_marker_prefix": [],
    }
    daily.write_text(
        "<!-- @compiled-through:replacement -->\nB\n", encoding="utf-8"
    )
    status, observed = flush_module._observe_auto_compile_content(root, reservation)
    assert status == "uncompiled"
    assert observed == {
        "fingerprint": original,
        "log_name": daily.name,
        "required_marker_prefix": ["first"],
        "pending_fingerprint": pending,
        "pending_log_name": pending_log.name,
        "pending_required_marker_prefix": [],
    }


def test_orphan_takeover_keeps_exit_75_request_until_lock_releases(
    tmp_path, monkeypatch
):
    root, daily = _auto_compile_test_root(tmp_path)
    commands = []
    _configure_auto_compile_test(monkeypatch, commands)
    assert _schedule_auto_compile(root) is True
    daily.write_text("A\nB\n", encoding="utf-8")
    watcher_token = _watchdog_command(commands)[-1]
    with QueueRepository(root / "scripts/jobs.sqlite3", sync_usage=False) as repository:
        state = _reservation(repository)
        state["expires_at"] = "2000-01-01T00:00:00.000000+00:00"
        state["watcher_expires_at"] = "2999-01-01T00:00:00.000000+00:00"
        repository._connection.execute(
            "UPDATE queue_metadata SET value = ? "
            "WHERE key = 'auto_compile_reservation'",
            (json.dumps(state),),
        )
    launches = []
    lock_held = [True]

    class LockedCompile:
        def wait(self, timeout=None):
            return 75

    def deferred(_seconds):
        with QueueRepository(
            root / "scripts/jobs.sqlite3", sync_usage=False
        ) as repository:
            assert _reservation(repository)["token"] == watcher_token
        daily.write_text(
            "A\nB\n<!-- @compiled-through:orphan -->\n", encoding="utf-8"
        )
        lock_held[0] = False

    def takeover(target, token, fingerprint):
        return flush_module.run_auto_compile_coordinator(
            target,
            token,
            fingerprint,
            compile_launcher=lambda *_args, **_kwargs: (
                launches.append(True) or LockedCompile()
            ),
            sleeper=deferred,
            lock_probe=lambda _root: lock_held[0],
        )

    assert flush_module.run_auto_compile_watcher(
        root,
        watcher_token,
        coordinator=takeover,
        sleeper=lambda _seconds: (_ for _ in ()).throw(
            AssertionError("expired owner should be claimed immediately")
        ),
        lock_probe=lambda _root: False,
    )
    assert launches == [True]


def test_end_of_day_scheduler_rejects_symlinked_daily_log(tmp_path, monkeypatch):
    root = tmp_path / "memory"
    (root / "scripts").mkdir(parents=True)
    (root / "daily").mkdir()
    (root / "scripts/compile.py").write_text("# compiler\n", encoding="utf-8")
    (root / "scripts/auto-compile.py").write_text("# coordinator\n", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("Sensitive uncompiled content.\n", encoding="utf-8")
    (root / "daily/2026-08-11.md").symlink_to(outside)
    launches = []
    monkeypatch.setattr(flush_module, "count_interactive_agent_sessions", lambda: 0)
    monkeypatch.setattr(flush_module, "_resolve_tty_path", lambda: None)
    monkeypatch.setattr(flush_module, "notify_terminal", lambda _message: None)
    monkeypatch.setattr(
        flush_module.subprocess,
        "Popen",
        lambda command, **_kwargs: launches.append(command),
    )

    assert (
        flush_module.maybe_trigger_compilation(
            memory_home=root,
            now=datetime(2026, 8, 11, 16, 31, tzinfo=timezone.utc),
        )
        is False
    )
    assert launches == []


@pytest.mark.parametrize(
    ("hour", "sessions"),
    [(15, 0), (16, 1), (16, -1)],
)
def test_end_of_day_scheduler_skips_before_four_or_with_active_or_unknown_sessions(
    tmp_path, monkeypatch, hour, sessions
):
    root = tmp_path / "memory"
    (root / "scripts").mkdir(parents=True)
    (root / "daily").mkdir()
    (root / "scripts/compile.py").write_text("# compiler\n", encoding="utf-8")
    (root / "daily/2026-08-11.md").write_text("Uncompiled content.\n", encoding="utf-8")
    monkeypatch.setattr(
        flush_module, "count_interactive_agent_sessions", lambda: sessions
    )
    monkeypatch.setattr(
        flush_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not launch")
        ),
    )

    assert (
        flush_module.maybe_trigger_compilation(
            memory_home=root,
            now=datetime(2026, 8, 11, hour, 30, tzinfo=timezone.utc),
        )
        is False
    )


def test_legacy_flush_does_not_schedule_after_failed_extraction(
    tmp_path, monkeypatch
):
    context = tmp_path / "context.md"
    context.write_text("User: remember this", encoding="utf-8")
    scheduled = []

    async def failed_flush(*_args, **_kwargs):
        return "FLUSH_ERROR: provider unavailable"

    monkeypatch.setenv("AI_MEMORY_INTERNAL_JOB", "1")
    monkeypatch.setenv("CLAUDE_INVOKED_BY", "test")
    monkeypatch.setattr(flush_module, "configure_logging", lambda: None)
    monkeypatch.setattr(flush_module, "load_flush_state", lambda: {})
    monkeypatch.setattr(flush_module, "run_flush", failed_flush)
    monkeypatch.setattr(flush_module, "append_to_daily_log", lambda *_args: None)
    monkeypatch.setattr(flush_module, "notify_terminal", lambda _message: None)
    monkeypatch.setattr(
        flush_module,
        "maybe_trigger_compilation",
        lambda **_kwargs: scheduled.append(True),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["flush.py", str(context), "session-1", "project", str(tmp_path)],
    )

    flush_module.main()

    assert scheduled == []
    assert context.exists()


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
