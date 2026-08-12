from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from decimal import Decimal
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sqlite3
import stat
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from providers import ProviderResult, RoutedResult, TaskKind
from scripts.queue import QueueRepository


def load_batch_flush():
    path = Path(__file__).resolve().parents[1] / "batch-flush.py"
    spec = importlib.util.spec_from_file_location("task8_batch_flush", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def batch():
    return load_batch_flush()


def write_codex(path: Path, *, session_id: str, cwd: str, timestamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "timestamp": timestamp,
            "type": "session_meta",
            "payload": {"id": session_id, "cwd": cwd, "timestamp": timestamp},
        },
        {
            "timestamp": timestamp,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Keep the durable choice"}],
            },
        },
        {
            "timestamp": timestamp,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Choice recorded"}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def write_claude(path: Path, *, session_id: str, cwd: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "sessionId": session_id,
            "cwd": cwd,
            "timestamp": "2026-08-10T10:00:00Z",
            "message": {"role": "user", "content": "x" * 6_000},
        },
        {
            "sessionId": session_id,
            "cwd": cwd,
            "timestamp": "2026-08-10T10:00:01Z",
            "message": {"role": "assistant", "content": "done"},
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def manifest(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def metadata_manifest(root: Path) -> dict[str, tuple[str, int, int, int]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mode,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def copy_runtime_tree(tmp_path: Path) -> Path:
    source_root = tmp_path / "runtime"
    scripts_dir = Path(__file__).resolve().parents[1]
    shutil.copytree(
        scripts_dir,
        source_root / "scripts",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.log", "tests"),
    )
    return source_root


def subprocess_environment(memory_home: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["AI_MEMORY_HOME"] = str(memory_home)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPATH", None)
    return environment


def test_recursive_codex_discovery_uses_session_metadata_and_reports_date_disagreement(
    batch, tmp_path
):
    root = tmp_path / ".codex" / "sessions"
    write_codex(
        root / "2026" / "07" / "01" / "nested" / "rollout-a.jsonl",
        session_id="meta-session",
        cwd="/work/acme/project-one",
        timestamp="2026-07-03T04:05:06Z",
    )

    result = batch.discover_codex_sessions(root)

    assert len(result.sessions) == 1
    discovered = result.sessions[0]
    assert discovered.session.session_id == "meta-session"
    assert discovered.session.cwd == "/work/acme/project-one"
    assert discovered.session.project == "project-one"
    assert discovered.date == "2026-07-03"
    assert discovered.directory_date == "2026-07-01"
    assert result.date_disagreements == (discovered.path,)


def test_codex_discovery_skips_malformed_and_deduplicates_equivalent_rollouts(
    batch, tmp_path
):
    root = tmp_path / "sessions"
    first = root / "2026" / "08" / "10" / "a.jsonl"
    duplicate = root / "2026" / "08" / "10" / "nested" / "b.jsonl"
    malformed = root / "2026" / "08" / "10" / "broken.jsonl"
    write_codex(
        first,
        session_id="same",
        cwd="/repo/widget",
        timestamp="2026-08-10T10:00:00Z",
    )
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    duplicate.write_bytes(first.read_bytes())
    malformed.write_text("{definitely not json\n", encoding="utf-8")

    result = batch.discover_codex_sessions(root)

    assert len(result.sessions) == 1
    assert result.duplicates == (duplicate,)
    assert result.malformed == (malformed,)


def test_codex_discovery_rejects_external_symlink_without_reading_target(
    batch, tmp_path
):
    root = tmp_path / "sessions"
    root.mkdir()
    external = tmp_path / "external.jsonl"
    write_codex(
        external,
        session_id="outside",
        cwd="/outside/project",
        timestamp="2026-08-10T10:00:00Z",
    )
    sentinel = external.read_bytes()
    linked = root / "linked.jsonl"
    linked.symlink_to(external)

    result = batch.discover_codex_sessions(root)

    assert result.sessions == ()
    assert result.malformed == (linked,)
    assert external.read_bytes() == sentinel


def test_codex_discovery_fallback_rejects_symlinked_root(
    batch, tmp_path, monkeypatch
):
    real_root = tmp_path / "real-sessions"
    transcript = real_root / "rollout.jsonl"
    write_codex(
        transcript,
        session_id="outside",
        cwd="/outside/project",
        timestamp="2026-08-10T10:00:00Z",
    )
    linked_root = tmp_path / "linked-sessions"
    linked_root.symlink_to(real_root, target_is_directory=True)
    transcript_module = sys.modules[batch.read_transcript_snapshot.__module__]
    monkeypatch.setattr(transcript_module, "_USE_DIR_FD", False)

    result = batch.discover_codex_sessions(linked_root)

    assert result.sessions == ()
    assert result.malformed == (linked_root / "rollout.jsonl",)


def test_claude_discovery_rejects_external_symlink(batch, tmp_path):
    root = tmp_path / "claude"
    root.mkdir()
    external = tmp_path / "external-claude.jsonl"
    write_claude(external, session_id="outside", cwd="/outside/project")
    linked = root / "linked.jsonl"
    linked.symlink_to(external)
    target = batch.Target("project", "/trusted/project", root)

    assert batch.discover_claude_sessions([target]) == []


def test_codex_discovery_rejects_hardlinked_source_when_supported(batch, tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    external = tmp_path / "external.jsonl"
    write_codex(
        external,
        session_id="hardlinked",
        cwd="/outside/project",
        timestamp="2026-08-10T10:00:00Z",
    )
    linked = root / "linked.jsonl"
    try:
        os.link(external, linked)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")

    result = batch.discover_codex_sessions(root)

    assert result.sessions == ()
    assert result.malformed == (linked,)


def test_codex_discovery_accepts_repeated_identical_session_meta(batch, tmp_path):
    root = tmp_path / "sessions"
    transcript = root / "rollout.jsonl"
    write_codex(
        transcript,
        session_id="same-meta",
        cwd="/repo/project",
        timestamp="2026-08-10T10:00:00Z",
    )
    records = [json.loads(line) for line in transcript.read_text().splitlines()]
    records.insert(1, records[0])
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    result = batch.discover_codex_sessions(root)

    assert len(result.sessions) == 1
    assert result.malformed == ()


def test_codex_discovery_rejects_conflicting_session_meta(batch, tmp_path):
    root = tmp_path / "sessions"
    transcript = root / "rollout.jsonl"
    write_codex(
        transcript,
        session_id="first",
        cwd="/repo/first",
        timestamp="2026-08-10T10:00:00Z",
    )
    records = [json.loads(line) for line in transcript.read_text().splitlines()]
    conflicting = json.loads(json.dumps(records[0]))
    conflicting["payload"]["id"] = "second"
    conflicting["payload"]["cwd"] = "/repo/second"
    records.insert(1, conflicting)
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    result = batch.discover_codex_sessions(root)

    assert result.sessions == ()
    assert result.malformed == (transcript,)


@pytest.mark.parametrize("limit_name", ["MAX_TRANSCRIPT_BYTES", "MAX_TRANSCRIPT_RECORDS"])
def test_codex_discovery_rejects_bounded_source_limits(
    batch, tmp_path, monkeypatch, limit_name
):
    root = tmp_path / "sessions"
    transcript = root / "rollout.jsonl"
    write_codex(
        transcript,
        session_id="bounded",
        cwd="/repo/bounded",
        timestamp="2026-08-10T10:00:00Z",
    )
    monkeypatch.setattr(batch, limit_name, 1, raising=False)

    result = batch.discover_codex_sessions(root)

    assert result.sessions == ()
    assert result.malformed == (transcript,)


@pytest.mark.parametrize("change", ["grow", "replace"])
def test_codex_discovery_rejects_source_changed_during_single_snapshot(
    batch, tmp_path, monkeypatch, change
):
    root = tmp_path / "sessions"
    transcript = root / "rollout.jsonl"
    write_codex(
        transcript,
        session_id="stable",
        cwd="/repo/stable",
        timestamp="2026-08-10T10:00:00Z",
    )
    original_read = os.read
    changed = False

    def changing_read(descriptor, count):
        nonlocal changed
        data = original_read(descriptor, count)
        if data and not changed:
            changed = True
            if change == "grow":
                with transcript.open("ab") as stream:
                    stream.write(b"\n{}")
            else:
                replacement = transcript.with_suffix(".replacement")
                write_codex(
                    replacement,
                    session_id="replacement",
                    cwd="/repo/replacement",
                    timestamp="2026-08-11T10:00:00Z",
                )
                os.replace(replacement, transcript)
        return data

    monkeypatch.setattr(os, "read", changing_read)

    result = batch.discover_codex_sessions(root)

    assert changed
    assert result.sessions == ()
    assert result.malformed == (transcript,)


def test_historical_single_chunk_keeps_live_capture_resume_key(batch, tmp_path):
    transcript = tmp_path / "session.jsonl"
    write_codex(
        transcript,
        session_id="shared-id",
        cwd="/repo/shared",
        timestamp="2026-08-10T10:00:00Z",
    )
    live = batch.parse_codex_transcript(transcript, {"trigger": "session_end"})
    historical = batch.parse_codex_transcript(transcript, {"trigger": "historical"})

    chunks = batch.chunk_session(historical, batch.CHUNK_TARGET_CHARS)

    assert len(chunks) == 1
    assert chunks[0].source_hash == live.source_hash


def test_multi_chunk_resume_keys_are_distinct_and_stable(batch, tmp_path):
    transcript = tmp_path / "large.jsonl"
    timestamp = "2026-08-10T10:00:00Z"
    records = [
        {
            "timestamp": timestamp,
            "type": "session_meta",
            "payload": {"id": "large", "cwd": "/repo/large", "timestamp": timestamp},
        }
    ]
    for index in range(3):
        records.extend(
            [
                {
                    "timestamp": timestamp,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"topic-{index} " + "x" * 26_000,
                            }
                        ],
                    },
                },
                {
                    "timestamp": timestamp,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": f"answer-{index}"}],
                    },
                },
            ]
        )
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    session = batch.parse_codex_transcript(transcript, {"trigger": "historical"})

    first = batch.chunk_session(session, batch.CHUNK_TARGET_CHARS)
    second = batch.chunk_session(session, batch.CHUNK_TARGET_CHARS)

    assert len(first) == 3
    assert len({chunk.source_hash for chunk in first}) == 3
    assert [chunk.source_hash for chunk in first] == [chunk.source_hash for chunk in second]

    for index, chunk in enumerate(first):
        live_path = tmp_path / f"live-slice-{index}.jsonl"
        records = [
            {
                "type": "session_meta",
                "payload": {
                    "id": session.session_id,
                    "cwd": session.cwd,
                    "timestamp": session.timestamp,
                },
            }
        ]
        for turn in chunk.turns:
            records.append(
                {
                    "timestamp": turn.timestamp,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": turn.role,
                        "content": [
                            {
                                "type": (
                                    "input_text"
                                    if turn.role == "user"
                                    else "output_text"
                                ),
                                "text": turn.text,
                            }
                        ],
                    },
                }
            )
        live_path.write_text(
            "\n".join(json.dumps(record) for record in records), encoding="utf-8"
        )
        live_slice = batch.parse_codex_transcript(
            live_path, {"trigger": "session_end"}
        )
        assert chunk.source_hash == live_slice.source_hash


@pytest.mark.parametrize("missing", ["id", "cwd", "timestamp"])
def test_codex_discovery_requires_complete_session_meta(batch, tmp_path, missing):
    root = tmp_path / "sessions"
    valid = root / "2026" / "08" / "10" / "valid.jsonl"
    write_codex(
        valid,
        session_id="complete",
        cwd="/repo/complete",
        timestamp="2026-08-10T10:00:00Z",
    )
    incomplete = root / "2026" / "08" / "10" / f"missing-{missing}.jsonl"
    records = [json.loads(line) for line in valid.read_text().splitlines()]
    del records[0]["payload"][missing]
    incomplete.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    result = batch.discover_codex_sessions(root)

    assert tuple(item.path for item in result.sessions) == (valid,)
    assert result.malformed == (incomplete,)


def test_codex_parsing_is_concurrency_bounded_and_ordered(
    batch, tmp_path, monkeypatch
):
    root = tmp_path / "sessions"
    for index in range(5):
        write_codex(
            root / "2026" / "08" / "10" / f"{index}.jsonl",
            session_id=f"parse-{index}",
            cwd="/repo/parse",
            timestamp=f"2026-08-10T10:0{index}:00Z",
        )
    original = batch._parse_codex_candidate
    active = 0
    peak = 0
    lock = threading.Lock()

    def tracked(root_path, path):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        try:
            return original(root_path, path)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(batch, "_parse_codex_candidate", tracked)

    result = batch.discover_codex_sessions(root, concurrency=2)

    assert peak == 2
    assert [item.path.name for item in result.sessions] == [
        f"{i}.jsonl" for i in range(5)
    ]


@pytest.mark.parametrize("source", ["claude", "codex", "all"])
def test_cli_accepts_sources_and_date_filters(batch, source):
    args = batch.parse_cli_args(
        [
            "--source",
            source,
            "--from-date",
            "2026-07-01",
            "--to-date",
            "2026-07-31",
            "--dates",
            "2026-07-03,2026-07-04",
            "--concurrency",
            "3",
            "--dry-run",
        ]
    )
    assert args.source == source
    assert args.concurrency == 3


@pytest.mark.parametrize(
    "argv",
    [
        ["--from-date", "2026-08-02", "--to-date", "2026-08-01"],
        ["--from-date", "not-a-date"],
        ["--dates", "2026-08-01,nope"],
        ["--concurrency", "0"],
        ["--source", "codex", "--max-cost", "1"],
        ["--source", "all", "--max-cost", "1"],
    ],
)
def test_cli_rejects_invalid_ranges_concurrency_and_codex_dollar_budget(batch, argv):
    with pytest.raises(SystemExit):
        batch.parse_cli_args(argv)


@pytest.mark.parametrize(
    "value", ["nan", "NaN", "inf", "-inf", "0", "-1", "not-a-number"]
)
def test_cli_rejects_nonfinite_nonpositive_and_malformed_max_cost(
    batch, value, capsys
):
    with pytest.raises(SystemExit):
        batch.parse_cli_args(["--source", "claude", "--max-cost", value])

    assert "--max-cost" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("value", "capacity"),
    [
        ("0.039999999999999999", 0),
        ("0.11999999999999999999999999999", 2),
    ],
)
def test_cli_preserves_exact_max_cost_token_for_chunk_capacity(
    batch, tmp_path, value, capacity
):
    args = batch.parse_cli_args(
        ["--source", "claude", "--max-cost", value, "--dry-run"]
    )
    sessions = [
        replace(item, session=replace(item.session, agent="claude"))
        for item in make_discovered(batch, tmp_path, 3)
    ]

    report = asyncio.run(
        batch.execute_historical_import(
            sessions, args, memory_home=tmp_path / "memory", router=None
        )
    )

    assert args.max_cost == Decimal(value)
    assert report.chunks == capacity
    assert report.enqueued == capacity


def test_filter_uses_session_timestamp_not_directory_date(batch, tmp_path):
    root = tmp_path / "sessions"
    write_codex(
        root / "2026" / "01" / "01" / "rollout.jsonl",
        session_id="filter-me",
        cwd="/repo/filter",
        timestamp="2026-02-14T23:30:00-08:00",
    )
    discovered = batch.discover_codex_sessions(root).sessions
    args = batch.parse_cli_args(["--source", "codex", "--dates", "2026-02-14"])

    assert batch.filter_historical_sessions(discovered, args) == list(discovered)


def test_strict_dry_run_is_no_write_no_model_and_reports_plan(
    batch, tmp_path, capsys, monkeypatch
):
    memory_home = tmp_path / "memory"
    (memory_home / "scripts").mkdir(parents=True)
    sentinel = memory_home / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    sessions_root = tmp_path / ".codex" / "sessions"
    write_codex(
        sessions_root / "2026" / "08" / "10" / "rollout.jsonl",
        session_id="dry",
        cwd="/repo/dry-project",
        timestamp="2026-08-10T10:00:00Z",
    )
    before = manifest(memory_home)
    monkeypatch.setenv("AI_MEMORY_HOME", str(memory_home))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    monkeypatch.setattr(
        batch,
        "_default_router",
        lambda: (_ for _ in ()).throw(AssertionError("dry run built a provider")),
    )

    exit_code = batch.main(
        [
            "--source",
            "codex",
            "--codex-sessions-dir",
            str(sessions_root),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "sessions: 1" in output
    assert "chunks: 1" in output
    assert "projects: dry-project" in output
    assert "dates: 2026-08-10" in output
    assert "models: gpt-5.6-luna" in output
    assert "estimated tokens:" in output
    assert manifest(memory_home) == before
    assert not (memory_home / "scripts" / "jobs.sqlite3").exists()
    assert not (memory_home / "scripts" / "state.json").exists()


def test_batch_main_dry_run_subprocess_preserves_runtime_and_memory_manifests(
    tmp_path,
):
    source_root = copy_runtime_tree(tmp_path)
    memory_home = tmp_path / "memory"
    (memory_home / "scripts").mkdir(parents=True)
    (memory_home / "sentinel.txt").write_text("unchanged", encoding="utf-8")
    sessions_root = tmp_path / ".codex" / "sessions"
    write_codex(
        sessions_root / "2026" / "08" / "10" / "rollout.jsonl",
        session_id="dry-subprocess",
        cwd="/repo/dry-subprocess",
        timestamp="2026-08-10T10:00:00Z",
    )
    runtime_before = metadata_manifest(source_root)
    memory_before = metadata_manifest(memory_home)
    probe = """
import importlib.util
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / "scripts" / "batch-flush.py"
sys.path.insert(0, str(root))
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location("dry_run_batch_main", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

def forbidden_provider(*args, **kwargs):
    raise AssertionError("dry run constructed a provider")

def forbidden_queue(*args, **kwargs):
    raise AssertionError("dry run constructed a writable queue")

module._default_router = forbidden_provider
module.QueueRepository = forbidden_queue
raise SystemExit(module.main([
    "--source", "codex",
    "--codex-sessions-dir", sys.argv[2],
    "--dates", "2026-08-10",
    "--dry-run",
]))
"""

    result = subprocess.run(
        [sys.executable, "-c", probe, str(source_root), str(sessions_root)],
        cwd=tmp_path,
        env=subprocess_environment(memory_home),
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert metadata_manifest(source_root) == runtime_before
    assert metadata_manifest(memory_home) == memory_before
    assert not (source_root / "scripts" / "flush.log").exists()
    assert not (memory_home / "scripts" / "jobs.sqlite3").exists()
    assert not (memory_home / "scripts" / "state.json").exists()


def test_flush_logging_is_lazy_until_explicitly_configured(tmp_path):
    source_root = copy_runtime_tree(tmp_path)
    memory_home = tmp_path / "memory"
    memory_home.mkdir()
    probe = """
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
import scripts.flush as flush

log_path = root / "scripts" / "flush.log"
assert not log_path.exists()
flush.configure_logging()
assert log_path.exists()
"""

    result = subprocess.run(
        [sys.executable, "-c", probe, str(source_root)],
        cwd=tmp_path,
        env=subprocess_environment(memory_home),
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("artifact", ["missing", "corrupt-db", "corrupt-state"])
def test_resume_dry_run_read_only_lookup_tolerates_missing_and_corrupt_state(
    batch, tmp_path, artifact
):
    memory_home = tmp_path / "memory"
    (memory_home / "scripts").mkdir(parents=True)
    if artifact == "corrupt-db":
        (memory_home / "scripts" / "jobs.sqlite3").write_bytes(b"not sqlite")
    if artifact == "corrupt-state":
        (memory_home / "scripts" / "state.json").write_text("{broken", encoding="utf-8")
    before = manifest(memory_home)
    args = batch.parse_cli_args(["--source", "codex", "--resume", "--dry-run"])

    report = asyncio.run(
        batch.execute_historical_import(
            make_discovered(batch, tmp_path, 1),
            args,
            memory_home=memory_home,
            router=None,
        )
    )

    assert report.chunks == 1
    assert report.skipped == 0
    assert report.enqueued == 1
    assert manifest(memory_home) == before


def test_resume_dry_run_skips_completed_queue_identity_without_mutation(batch, tmp_path):
    memory_home = tmp_path / "memory"
    sessions = make_discovered(batch, tmp_path, 1)
    run_args = batch.parse_cli_args(["--source", "codex"])
    asyncio.run(
        batch.execute_historical_import(
            sessions, run_args, memory_home=memory_home, router=TrackingRouter()
        )
    )
    before = manifest(memory_home)
    dry_args = batch.parse_cli_args(
        ["--source", "codex", "--resume", "--dry-run"]
    )

    report = asyncio.run(
        batch.execute_historical_import(
            sessions, dry_args, memory_home=memory_home, router=None
        )
    )

    assert report.chunks == 0
    assert report.enqueued == 0
    assert report.preexisting == 1
    assert report.skipped == 0
    assert report.newly_enqueued == 0
    assert report.processed == 0
    assert manifest(memory_home) == before


def test_read_only_dedup_sees_succeeded_row_present_only_in_wal(batch, tmp_path):
    memory_home = tmp_path / "memory"
    session = make_discovered(batch, tmp_path, 1)[0].session
    queue_path = memory_home / "scripts" / "jobs.sqlite3"
    with QueueRepository(queue_path) as repository:
        repository._connection.execute("PRAGMA wal_autocheckpoint = 0")
        repository._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        queued = repository.enqueue_capture(session)
        claimed = repository.claim_next("test-worker", batch.datetime.now(batch.timezone.utc), 30)
        assert claimed is not None
        repository.complete(queued.job_id, "test-worker")
        wal_path = Path(f"{queue_path}-wal")
        assert wal_path.stat().st_size > 0
        base_only = sqlite3.connect(
            f"file:{queue_path.resolve().as_posix()}?mode=ro&immutable=1", uri=True
        )
        try:
            assert base_only.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        finally:
            base_only.close()
        before = metadata_manifest(memory_home)

        identities = batch._read_queue_identities(queue_path)

        assert identities[batch._identity(session)] == "succeeded"
        assert metadata_manifest(memory_home) == before


def test_read_only_dedup_fails_closed_when_queue_snapshot_never_stabilizes(
    batch, tmp_path, monkeypatch
):
    memory_home = tmp_path / "memory"
    session = make_discovered(batch, tmp_path, 1)[0].session
    queue_path = memory_home / "scripts" / "jobs.sqlite3"
    with QueueRepository(queue_path) as repository:
        repository.enqueue_capture(session)
    before = metadata_manifest(memory_home)
    counter = iter(range(100))
    monkeypatch.setattr(
        batch,
        "_snapshot_signature",
        lambda _paths: ((True, next(counter)),),
    )

    with pytest.raises(batch.QueueSnapshotUnstable):
        batch._read_queue_identities(queue_path)

    assert metadata_manifest(memory_home) == before


def test_resume_dry_run_honors_legacy_processed_session_state(batch, tmp_path):
    memory_home = tmp_path / "memory"
    scripts = memory_home / "scripts"
    scripts.mkdir(parents=True)
    sessions = make_discovered(batch, tmp_path, 1)
    sessions = [
        replace(sessions[0], session=replace(sessions[0].session, agent="claude"))
    ]
    (scripts / "state.json").write_text(
        json.dumps(
            {
                "batch_flush": {
                    "processed_sessions": {sessions[0].session.session_id: {"chunks": 1}}
                }
            }
        ),
        encoding="utf-8",
    )
    before = manifest(memory_home)
    args = batch.parse_cli_args(["--source", "claude", "--resume", "--dry-run"])

    report = asyncio.run(
        batch.execute_historical_import(
            sessions, args, memory_home=memory_home, router=None
        )
    )

    assert report.chunks == 0
    assert report.preexisting == 1
    assert report.skipped == 0
    assert manifest(memory_home) == before


def test_dry_run_reports_existing_pending_queue_identity_as_deduplicated(batch, tmp_path):
    memory_home = tmp_path / "memory"
    sessions = make_discovered(batch, tmp_path, 1)
    queue_path = memory_home / "scripts" / "jobs.sqlite3"
    with QueueRepository(queue_path) as repository:
        repository.enqueue_capture(sessions[0].session)
    before = manifest(memory_home)
    args = batch.parse_cli_args(["--source", "codex", "--dry-run"])

    report = asyncio.run(
        batch.execute_historical_import(
            sessions, args, memory_home=memory_home, router=None
        )
    )

    assert report.chunks == 1
    assert report.enqueued == 0
    assert report.preexisting == 1
    assert report.skipped == 0
    assert report.newly_enqueued == 0
    assert report.processed == 0
    assert manifest(memory_home) == before


def test_custom_queue_path_is_shared_by_live_and_historical_import(
    batch, tmp_path, monkeypatch
):
    memory_home = tmp_path / "memory"
    custom_queue = tmp_path / "private-queue" / "custom.sqlite3"
    sessions = make_discovered(batch, tmp_path, 1)
    with QueueRepository(custom_queue) as repository:
        repository.enqueue_capture(sessions[0].session)
    monkeypatch.setenv("AI_MEMORY_QUEUE_PATH", str(custom_queue))
    dry_args = batch.parse_cli_args(["--source", "codex", "--dry-run"])

    dry_report = asyncio.run(
        batch.execute_historical_import(
            sessions, dry_args, memory_home=memory_home, router=None
        )
    )

    assert dry_report.preexisting == 1
    assert dry_report.newly_enqueued == 0
    assert not (memory_home / "scripts" / "jobs.sqlite3").exists()

    router = TrackingRouter()
    report = asyncio.run(
        batch.execute_historical_import(
            sessions,
            batch.parse_cli_args(["--source", "codex"]),
            memory_home=memory_home,
            router=router,
        )
    )

    assert report.preexisting == 1
    assert report.processed == 1
    assert report.succeeded == 1
    assert router.calls == 1
    assert not (memory_home / "scripts" / "jobs.sqlite3").exists()


@pytest.mark.parametrize(
    ("source", "expected"),
    [("claude", ("claude",)), ("codex", ("codex",)), ("all", ("claude", "codex"))],
)
def test_run_batch_discovers_only_requested_sources(
    batch, source, expected, monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(
        batch,
        "discover_claude_sessions",
        lambda _targets, **_kwargs: calls.append("claude") or [],
    )
    monkeypatch.setattr(
        batch,
        "discover_codex_sessions",
        lambda _root, **_kwargs: calls.append("codex") or batch.CodexDiscovery(()),
    )
    monkeypatch.setattr(batch, "resolve_single_target", lambda _args: object())
    monkeypatch.setattr(
        batch,
        "load_config",
        lambda _env: SimpleNamespace(root_dir=tmp_path / "memory"),
    )
    monkeypatch.setattr(
        batch,
        "execute_historical_import",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=batch.ImportReport(0, 0, (), (), 0),
        ),
    )
    args = batch.parse_cli_args(["--source", source, "--dry-run"])

    asyncio.run(batch.run_batch(args))

    assert tuple(calls) == expected


@dataclass
class TrackingRouter:
    active: int = 0
    peak: int = 0
    calls: int = 0

    async def generate_text(self, request):
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        attempt = ProviderResult(
            provider="codex",
            model="gpt-5.6-luna",
            task=TaskKind.EXTRACT,
            outcome="success",
            text="**Context:** Imported historical session",
        )
        return RoutedResult.from_result(attempt, [attempt], None)


class WorkspaceCheckingRouter(TrackingRouter):
    def __init__(self, expected_source_cwd: str, forbidden_roots: tuple[Path, ...]):
        super().__init__()
        self.expected_source_cwd = expected_source_cwd
        self.forbidden_roots = forbidden_roots
        self.workspaces: list[Path] = []

    async def generate_text(self, request):
        workspace = request.cwd
        assert workspace.is_absolute()
        assert workspace.exists() and workspace.is_dir()
        assert not workspace.is_symlink()
        assert stat.S_IMODE(workspace.stat().st_mode) == 0o700
        assert list(workspace.iterdir()) == []
        assert all(
            workspace != root and root not in workspace.parents
            for root in self.forbidden_roots
        )
        assert f"Historical source CWD: {self.expected_source_cwd}" in request.prompt
        self.workspaces.append(workspace)
        return await super().generate_text(request)


class SentinelRouter:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def generate_text(self, request):
        self.calls += 1
        attempt = ProviderResult(
            provider="codex",
            model="gpt-5.6-luna",
            task=TaskKind.EXTRACT,
            outcome="success",
            text=self.text,
        )
        return RoutedResult.from_result(attempt, [attempt], None)


class FailedRouter:
    async def generate_text(self, request):
        attempt = ProviderResult(
            provider="claude",
            model="claude-test",
            task=TaskKind.EXTRACT,
            outcome="error",
            reason="synthetic failure",
        )
        return RoutedResult.from_result(attempt, [attempt], "codex:error:synthetic")


def make_discovered(batch, tmp_path: Path, count: int):
    root = tmp_path / "sessions"
    for index in range(count):
        write_codex(
            root / "2026" / "08" / "10" / f"{index}.jsonl",
            session_id=f"session-{index}",
            cwd=f"/repo/project-{index}",
            timestamp=f"2026-08-10T10:{index:02d}:00Z",
        )
    return list(batch.discover_codex_sessions(root).sessions)


def test_import_bounds_provider_concurrency_and_serializes_daily_writes(batch, tmp_path):
    memory_home = tmp_path / "memory"
    router = TrackingRouter()
    args = batch.parse_cli_args(["--source", "codex", "--concurrency", "2"])

    report = asyncio.run(
        batch.execute_historical_import(
            make_discovered(batch, tmp_path, 5),
            args,
            memory_home=memory_home,
            router=router,
        )
    )

    assert report.enqueued == 5
    assert report.succeeded == 5
    assert router.calls == 5
    assert router.peak == 2
    daily = (memory_home / "daily" / "2026-08-10.md").read_text(encoding="utf-8")
    assert daily.count("**Agent:** Codex") == 5


@pytest.mark.parametrize("source_kind", ["missing", "unrelated", "symlink"])
def test_historical_provider_uses_ephemeral_empty_workspace_not_source_cwd(
    batch, tmp_path, source_kind
):
    memory_home = tmp_path / "memory"
    source = tmp_path / "source-project"
    if source_kind == "unrelated":
        source.mkdir()
        (source / "secret.txt").write_text("do not expose", encoding="utf-8")
        source_cwd = str(source)
    elif source_kind == "symlink":
        target = tmp_path / "sensitive"
        target.mkdir()
        source.symlink_to(target, target_is_directory=True)
        source_cwd = str(source)
    else:
        source_cwd = str(source)
    discovered = make_discovered(batch, tmp_path, 1)[0]
    historical = replace(
        discovered,
        session=replace(discovered.session, cwd=source_cwd),
    )
    router = WorkspaceCheckingRouter(
        source_cwd,
        (memory_home.resolve(), source.resolve()),
    )

    report = asyncio.run(
        batch.execute_historical_import(
            [historical],
            batch.parse_cli_args(["--source", "codex"]),
            memory_home=memory_home,
            router=router,
        )
    )

    assert report.succeeded == 1
    assert len(router.workspaces) == 1
    assert not router.workspaces[0].exists()
    daily = (memory_home / "daily" / "2026-08-10.md").read_text(encoding="utf-8")
    assert f"**CWD:** {source_cwd}" in daily


def test_import_collapses_repeated_identical_chunks_before_estimate_and_enqueue(
    batch, tmp_path
):
    discovered = make_discovered(batch, tmp_path, 1)[0]
    repeated_turns = tuple(
        turn
        for _ in range(3)
        for turn in (
            batch.NormalizedTurn("user", "same request " + "x" * 26_000),
            batch.NormalizedTurn("assistant", "same response"),
        )
    )
    historical = replace(
        discovered,
        session=replace(discovered.session, turns=repeated_turns),
    )
    dry_home = tmp_path / "dry-memory"
    dry_args = batch.parse_cli_args(["--source", "codex", "--dry-run"])

    dry_report = asyncio.run(
        batch.execute_historical_import(
            [historical], dry_args, memory_home=dry_home, router=None
        )
    )
    unique_chunk = batch.chunk_session(
        historical.session, batch.CHUNK_TARGET_CHARS
    )[0]
    unique_tokens = max(1, (len(batch.render_turns(unique_chunk)) + 3) // 4)

    assert dry_report.chunks == 1
    assert dry_report.enqueued == 1
    assert dry_report.skipped == 2
    assert dry_report.estimated_tokens == unique_tokens

    memory_home = tmp_path / "memory"
    router = TrackingRouter()
    run_args = batch.parse_cli_args(["--source", "codex"])
    report = asyncio.run(
        batch.execute_historical_import(
            [historical], run_args, memory_home=memory_home, router=router
        )
    )

    with QueueRepository(memory_home / "scripts" / "jobs.sqlite3") as repository:
        job_count = repository._connection.execute(
            "SELECT COUNT(*) FROM jobs"
        ).fetchone()[0]
    daily = (memory_home / "daily" / "2026-08-10.md").read_text(encoding="utf-8")
    assert job_count == 1
    assert report.chunks == 1
    assert report.enqueued == 1
    assert report.succeeded == 1
    assert report.skipped == 2
    assert router.calls == 1
    assert daily.count("**Agent:** Codex") == 1


@pytest.mark.parametrize("resume", [False, True])
def test_claude_max_cost_subtracts_legacy_total_at_exact_boundary(
    batch, tmp_path, resume
):
    memory_home = tmp_path / "memory"
    scripts = memory_home / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "state.json").write_text(
        json.dumps({"batch_flush": {"total_cost": 0.04}}), encoding="utf-8"
    )
    sessions = [
        replace(item, session=replace(item.session, agent="claude"))
        for item in make_discovered(batch, tmp_path, 3)
    ]
    argv = ["--source", "claude", "--max-cost", "0.12", "--dry-run"]
    if resume:
        argv.append("--resume")

    report = asyncio.run(
        batch.execute_historical_import(
            sessions,
            batch.parse_cli_args(argv),
            memory_home=memory_home,
            router=None,
        )
    )

    assert report.chunks == 2
    assert report.enqueued == 2


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize(
    "state_value", ["0.040000000000000002", '"0.040000000000000002"']
)
def test_claude_max_cost_preserves_exact_legacy_state_arithmetic(
    batch, tmp_path, resume, state_value
):
    memory_home = tmp_path / "memory"
    scripts = memory_home / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "state.json").write_text(
        f'{{"batch_flush":{{"total_cost":{state_value}}}}}',
        encoding="utf-8",
    )
    sessions = [
        replace(item, session=replace(item.session, agent="claude"))
        for item in make_discovered(batch, tmp_path, 1)
    ]
    argv = [
        "--source",
        "claude",
        "--max-cost",
        "0.080000000000000001",
        "--dry-run",
    ]
    if resume:
        argv.append("--resume")

    report = asyncio.run(
        batch.execute_historical_import(
            sessions,
            batch.parse_cli_args(argv),
            memory_home=memory_home,
            router=None,
        )
    )

    assert report.chunks == 0
    assert report.enqueued == 0


@pytest.mark.parametrize("accumulated", [0.12, 0.13])
def test_claude_max_cost_processes_nothing_when_legacy_total_reaches_ceiling(
    batch, tmp_path, accumulated
):
    memory_home = tmp_path / "memory"
    scripts = memory_home / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "state.json").write_text(
        json.dumps({"batch_flush": {"total_cost": accumulated}}), encoding="utf-8"
    )
    sessions = [
        replace(item, session=replace(item.session, agent="claude"))
        for item in make_discovered(batch, tmp_path, 1)
    ]
    args = batch.parse_cli_args(
        ["--source", "claude", "--max-cost", "0.12", "--dry-run"]
    )

    report = asyncio.run(
        batch.execute_historical_import(
            sessions, args, memory_home=memory_home, router=None
        )
    )

    assert report.chunks == 0
    assert report.enqueued == 0
    assert report.estimated_tokens == 0


@pytest.mark.parametrize("legacy_state", [None, "{broken"])
def test_claude_max_cost_tolerates_missing_or_corrupt_legacy_state(
    batch, tmp_path, legacy_state
):
    memory_home = tmp_path / "memory"
    if legacy_state is not None:
        scripts = memory_home / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "state.json").write_text(legacy_state, encoding="utf-8")
    sessions = [
        replace(item, session=replace(item.session, agent="claude"))
        for item in make_discovered(batch, tmp_path, 2)
    ]
    args = batch.parse_cli_args(
        ["--source", "claude", "--max-cost", "0.04", "--dry-run"]
    )

    report = asyncio.run(
        batch.execute_historical_import(
            sessions, args, memory_home=memory_home, router=None
        )
    )

    assert report.chunks == 1
    assert report.enqueued == 1


@pytest.mark.parametrize("response", ["**Context:** captured", "FLUSH_OK"])
def test_claude_transcript_via_codex_does_not_fabricate_legacy_cost(
    batch, tmp_path, response
):
    memory_home = tmp_path / "memory"
    discovered = make_discovered(batch, tmp_path, 1)[0]
    historical = replace(
        discovered,
        session=replace(discovered.session, agent="claude"),
    )
    args = batch.parse_cli_args(["--source", "claude", "--resume"])

    first = asyncio.run(
        batch.execute_historical_import(
            [historical],
            args,
            memory_home=memory_home,
            router=SentinelRouter(response),
        )
    )
    second = asyncio.run(
        batch.execute_historical_import(
            [historical],
            args,
            memory_home=memory_home,
            router=SentinelRouter(response),
        )
    )
    state = json.loads(
        (memory_home / "scripts" / "state.json").read_text(encoding="utf-8"),
        parse_float=Decimal,
    )["batch_flush"]

    assert first.succeeded == 1
    assert second.preexisting == 1
    assert Decimal(str(state["total_cost"])) == Decimal("0")
    assert len(state["accounted_historical_jobs"]) == 1


def test_explicit_claude_legacy_cost_advances_total_exactly_once(batch, tmp_path):
    memory_home = tmp_path / "memory"
    discovered = make_discovered(batch, tmp_path, 1)[0]
    session = replace(discovered.session, agent="claude")
    historical = replace(discovered, session=session)
    queue_path = memory_home / "scripts" / "jobs.sqlite3"
    with QueueRepository(queue_path, memory_home=memory_home) as repository:
        queued = repository.enqueue_capture(session)
        claimed = repository.claim_next(
            "prior-worker", batch.datetime.now(batch.timezone.utc), 30
        )
        assert claimed is not None
        repository.record_attempt(
            queued.job_id,
            ProviderResult(
                provider="codex",
                model="gpt-5.6-terra",
                task=TaskKind.EXTRACT,
                outcome="capacity",
                reason="subscription full",
            ),
        )
        repository.record_attempt(
            queued.job_id,
            ProviderResult(
                provider="claude",
                model="claude-sonnet-5",
                task=TaskKind.EXTRACT,
                outcome="success",
            ),
            legacy_cost_usd=0.037,
        )
        repository.complete(queued.job_id, "prior-worker")
        batch._reconcile_legacy_claude_cost(
            repository, [historical.session], memory_home
        )
        batch._reconcile_legacy_claude_cost(
            repository, [historical.session], memory_home
        )

    state = json.loads(
        (memory_home / "scripts" / "state.json").read_text(encoding="utf-8")
    )["batch_flush"]
    assert Decimal(str(state["total_cost"])) == Decimal("0.037")
    assert list(state["accounted_historical_jobs"].values()) == ["0.037"]
    records = [
        json.loads(line)
        for line in (memory_home / "scripts/logs/usage.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(item["provider"], item["outcome"]) for item in records] == [
        ("codex", "capacity"),
        ("claude", "success"),
    ]
    assert records[-1]["fallback_reason"] == "codex:capacity:subscription full"
    assert records[-1]["cost_usd"] == 0.037


def test_failed_claude_job_does_not_advance_legacy_cost(batch, tmp_path, monkeypatch):
    original_repository = batch.QueueRepository
    monkeypatch.setattr(
        batch,
        "QueueRepository",
        lambda path, **kwargs: original_repository(path, max_attempts=1, **kwargs),
    )
    memory_home = tmp_path / "memory"
    discovered = make_discovered(batch, tmp_path, 1)[0]
    historical = replace(
        discovered,
        session=replace(discovered.session, agent="claude"),
    )

    report = asyncio.run(
        batch.execute_historical_import(
            [historical],
            batch.parse_cli_args(["--source", "claude"]),
            memory_home=memory_home,
            router=FailedRouter(),
        )
    )

    assert report.dead == 1
    assert not (memory_home / "scripts" / "state.json").exists()


def test_resume_repairs_cost_after_queue_success_before_state_commit(batch, tmp_path):
    memory_home = tmp_path / "memory"
    discovered = make_discovered(batch, tmp_path, 1)[0]
    session = replace(discovered.session, agent="claude")
    historical = replace(discovered, session=session)
    queue_path = memory_home / "scripts" / "jobs.sqlite3"
    with QueueRepository(queue_path) as repository:
        queued = repository.enqueue_capture(session)
        claimed = repository.claim_next("prior-worker", batch.datetime.now(batch.timezone.utc), 30)
        assert claimed is not None
        repository.complete(queued.job_id, "prior-worker")

    report = asyncio.run(
        batch.execute_historical_import(
            [historical],
            batch.parse_cli_args(["--source", "claude", "--resume"]),
            memory_home=memory_home,
            router=TrackingRouter(),
        )
    )
    state = json.loads(
        (memory_home / "scripts" / "state.json").read_text(encoding="utf-8")
    )["batch_flush"]

    assert report.preexisting == 1
    assert report.processed == 0
    assert Decimal(str(state["total_cost"])) == Decimal("0")


def test_resume_repairs_cost_before_enforcing_remaining_ceiling(batch, tmp_path):
    memory_home = tmp_path / "memory"
    sessions = [
        replace(item, session=replace(item.session, agent="claude"))
        for item in make_discovered(batch, tmp_path, 2)
    ]
    queue_path = memory_home / "scripts" / "jobs.sqlite3"
    with QueueRepository(queue_path) as repository:
        queued = repository.enqueue_capture(sessions[0].session)
        claimed = repository.claim_next("prior-worker", batch.datetime.now(batch.timezone.utc), 30)
        assert claimed is not None
        repository.complete(queued.job_id, "prior-worker")
    router = TrackingRouter()

    report = asyncio.run(
        batch.execute_historical_import(
            sessions,
            batch.parse_cli_args(
                ["--source", "claude", "--resume", "--max-cost", "0.04"]
            ),
            memory_home=memory_home,
            router=router,
        )
    )

    assert report.chunks == 1
    assert report.preexisting == 1
    assert report.newly_enqueued == 1
    assert router.calls == 1
    state = json.loads(
        (memory_home / "scripts" / "state.json").read_text(encoding="utf-8")
    )["batch_flush"]
    assert Decimal(str(state["total_cost"])) == Decimal("0")


def test_concurrent_claude_imports_do_not_lose_legacy_cost_updates(batch, tmp_path):
    memory_home = tmp_path / "memory"
    sessions = [
        replace(item, session=replace(item.session, agent="claude"))
        for item in make_discovered(batch, tmp_path, 2)
    ]
    args = batch.parse_cli_args(["--source", "claude", "--resume"])

    async def run_concurrently():
        return await asyncio.gather(
            batch.execute_historical_import(
                [sessions[0]], args, memory_home=memory_home, router=TrackingRouter()
            ),
            batch.execute_historical_import(
                [sessions[1]], args, memory_home=memory_home, router=TrackingRouter()
            ),
        )

    reports = asyncio.run(run_concurrently())
    state = json.loads(
        (memory_home / "scripts" / "state.json").read_text(encoding="utf-8")
    )["batch_flush"]

    assert sum(report.succeeded for report in reports) == 2
    assert Decimal(str(state["total_cost"])) == Decimal("0")
    assert len(state["accounted_historical_jobs"]) == 2


def test_cross_process_claude_max_cost_reservation_prevents_overspend(
    batch, tmp_path
):
    memory_home = tmp_path / "memory"
    sessions = make_discovered(batch, tmp_path, 2)
    gate = tmp_path / "go"
    # Keep the adversary focused on budget arbitration, not first-open schema
    # creation. Both importers still open and mutate the same live queue.
    with QueueRepository(memory_home / "scripts" / "jobs.sqlite3"):
        pass
    worker = r'''
import asyncio, importlib.util, json, sys, time
from pathlib import Path
from dataclasses import replace
from scripts.providers import ProviderResult, RoutedResult, TaskKind

batch_path, transcript_path, memory_home, gate = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("child_batch", batch_path)
batch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = batch
spec.loader.exec_module(batch)
session = batch.parse_codex_transcript(transcript_path, {"trigger": "historical"})
session = replace(session, agent="claude")
historical = batch.HistoricalSession(session, transcript_path, "2026-08-10")
class Router:
    async def generate_text(self, request):
        await asyncio.sleep(0.2)
        attempt = ProviderResult(provider="codex", model="fake", task=TaskKind.EXTRACT,
                                 outcome="success", text="**Context:** captured")
        return RoutedResult.from_result(attempt, [attempt], None)
while not gate.exists():
    time.sleep(0.005)
report = asyncio.run(batch.execute_historical_import(
    [historical], batch.parse_cli_args(["--source", "claude", "--resume", "--max-cost", "0.04"]),
    memory_home=memory_home, router=Router()))
print(json.dumps({"new": report.newly_enqueued, "succeeded": report.succeeded}))
'''
    batch_path = Path(batch.__file__)
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                worker,
                str(batch_path),
                str(item.path),
                str(memory_home),
                str(gate),
            ],
            cwd=batch_path.parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for item in sessions
    ]
    gate.touch()
    outputs = [process.communicate(timeout=20) for process in processes]

    assert [process.returncode for process in processes] == [0, 0], outputs
    with QueueRepository(memory_home / "scripts" / "jobs.sqlite3") as repository:
        job_count = repository._connection.execute(
            "SELECT COUNT(*) FROM jobs"
        ).fetchone()[0]
        succeeded = repository._connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = 'succeeded'"
        ).fetchone()[0]
    state = json.loads(
        (memory_home / "scripts" / "state.json").read_text(encoding="utf-8")
    )["batch_flush"]
    assert job_count == 1
    assert succeeded == 1
    assert Decimal(str(state["total_cost"])) == Decimal("0")


def test_abandoned_claude_cost_reservation_without_job_is_released(
    batch, tmp_path
):
    memory_home = tmp_path / "memory"
    scripts = memory_home / "scripts"
    scripts.mkdir(parents=True)
    abandoned = "a" * 64
    (scripts / "state.json").write_text(
        json.dumps(
            {
                "batch_flush": {
                    "total_cost": "0",
                    "historical_cost_reservations": {
                        abandoned: {
                            "cost": "0.04",
                            "pid": 999_999_999,
                            "created_at": "2000-01-01T00:00:00+00:00",
                            "expires_at": "2000-01-01T00:30:00+00:00",
                            "queue_identity": [
                                "capture",
                                "claude",
                                "crashed-session",
                                "b" * 64,
                            ],
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    discovered = make_discovered(batch, tmp_path, 1)[0]
    historical = replace(
        discovered, session=replace(discovered.session, agent="claude")
    )
    router = TrackingRouter()

    report = asyncio.run(
        batch.execute_historical_import(
            [historical],
            batch.parse_cli_args(
                ["--source", "claude", "--resume", "--max-cost", "0.04"]
            ),
            memory_home=memory_home,
            router=router,
        )
    )
    state = json.loads((scripts / "state.json").read_text(encoding="utf-8"))[
        "batch_flush"
    ]

    assert report.succeeded == 1
    assert Decimal(str(state["total_cost"])) == Decimal("0")
    assert abandoned not in state.get("historical_cost_reservations", {})


def test_resume_processes_stale_leased_job_left_by_crashed_cost_reservation(
    batch, tmp_path
):
    memory_home = tmp_path / "memory"
    discovered = make_discovered(batch, tmp_path, 1)[0]
    session = replace(discovered.session, agent="claude")
    historical = replace(discovered, session=session)
    queue_path = memory_home / "scripts" / "jobs.sqlite3"
    with QueueRepository(queue_path) as repository:
        repository.enqueue_capture(session)
        claimed = repository.claim_next(
            "crashed-worker", batch.datetime.now(batch.timezone.utc), 30
        )
        assert claimed is not None
        repository._connection.execute(
            "UPDATE jobs SET lease_expires_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", claimed.id),
        )
    identity = batch._legacy_accounting_identity(session)
    state_path = memory_home / "scripts" / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "batch_flush": {
                    "total_cost": "0",
                    "historical_cost_reservations": {
                        identity: {
                            "cost": "0.04",
                            "pid": 999_999_999,
                            "created_at": "2000-01-01T00:00:00+00:00",
                            "expires_at": "2000-01-01T00:30:00+00:00",
                            "queue_identity": list(batch._identity(session)),
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    router = TrackingRouter()

    report = asyncio.run(
        batch.execute_historical_import(
            [historical],
            batch.parse_cli_args(
                ["--source", "claude", "--resume", "--max-cost", "0.04"]
            ),
            memory_home=memory_home,
            router=router,
        )
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))["batch_flush"]

    assert report.preexisting == 1
    assert report.processed == 1
    assert report.succeeded == 1
    assert router.calls == 1
    assert Decimal(str(state["total_cost"])) == Decimal("0")
    assert state.get("historical_cost_reservations", {}) == {}


def test_failed_reserved_claude_job_releases_capacity(
    batch, tmp_path, monkeypatch
):
    original_repository = batch.QueueRepository
    monkeypatch.setattr(
        batch,
        "QueueRepository",
        lambda path, **kwargs: original_repository(path, max_attempts=1, **kwargs),
    )
    memory_home = tmp_path / "memory"
    discovered = make_discovered(batch, tmp_path, 1)[0]
    historical = replace(
        discovered, session=replace(discovered.session, agent="claude")
    )
    args = batch.parse_cli_args(
        ["--source", "claude", "--resume", "--max-cost", "0.04"]
    )

    failed = asyncio.run(
        batch.execute_historical_import(
            [historical], args, memory_home=memory_home, router=FailedRouter()
        )
    )
    state = json.loads(
        (memory_home / "scripts" / "state.json").read_text(encoding="utf-8")
    )["batch_flush"]

    assert failed.dead == 1
    assert Decimal(str(state["total_cost"])) == Decimal(0)
    assert state.get("historical_cost_reservations", {}) == {}


def test_legacy_cost_atomic_replace_failure_preserves_prior_state(
    batch, tmp_path, monkeypatch
):
    state_path = tmp_path / "memory" / "scripts" / "state.json"
    state_path.parent.mkdir(parents=True)
    original = b'{"batch_flush":{"total_cost":"1.00"}}'
    state_path.write_bytes(original)
    real_replace = os.replace

    def fail_target_replace(source, destination):
        if Path(destination) == state_path:
            raise OSError("synthetic replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_target_replace)

    with pytest.raises(OSError, match="synthetic replace failure"):
        batch._atomic_write_state(
            state_path, {"batch_flush": {"total_cost": "1.04"}}
        )

    assert state_path.read_bytes() == original
    assert list(state_path.parent.glob(".state.json.*.tmp")) == []


@pytest.mark.parametrize("operation", ["open", "fsync"])
def test_atomic_state_write_tolerates_unsupported_windows_directory_fsync(
    batch, tmp_path, monkeypatch, operation
):
    state_path = tmp_path / "scripts" / "state.json"
    state_path.parent.mkdir(parents=True)
    monkeypatch.setattr(batch, "WINDOWS", True, raising=False)
    if operation == "open":
        real_open = os.open

        def unsupported_open(path, flags, *args, **kwargs):
            if Path(path) == state_path.parent and flags & getattr(os, "O_DIRECTORY", 0):
                raise OSError(errno.EINVAL, "unsupported")
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", unsupported_open)
    else:
        real_fsync = os.fsync

        def unsupported_fsync(descriptor):
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError(errno.EINVAL, "unsupported")
            return real_fsync(descriptor)

        monkeypatch.setattr(os, "fsync", unsupported_fsync)

    batch._atomic_write_state(
        state_path, {"batch_flush": {"total_cost": "0.04"}}
    )

    assert json.loads(state_path.read_text())["batch_flush"]["total_cost"] == "0.04"


def test_atomic_state_write_propagates_real_posix_directory_fsync_failure(
    batch, tmp_path, monkeypatch
):
    state_path = tmp_path / "scripts" / "state.json"
    state_path.parent.mkdir(parents=True)
    monkeypatch.setattr(batch, "WINDOWS", False, raising=False)
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(errno.EIO, "storage failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="storage failure"):
        batch._atomic_write_state(
            state_path, {"batch_flush": {"total_cost": "0.04"}}
        )

    assert json.loads(state_path.read_text())["batch_flush"]["total_cost"] == "0.04"


@pytest.mark.parametrize("agent", ["claude", "codex"])
def test_flush_ok_completes_dedup_without_daily_write(batch, tmp_path, agent):
    memory_home = tmp_path / "memory"
    discovered = make_discovered(batch, tmp_path, 1)[0]
    session = replace(discovered.session, agent=agent)
    historical = replace(discovered, session=session)
    router = SentinelRouter("  FLUSH_OK\n")
    args = batch.parse_cli_args(["--source", agent, "--resume"])

    first = asyncio.run(
        batch.execute_historical_import(
            [historical], args, memory_home=memory_home, router=router
        )
    )
    second = asyncio.run(
        batch.execute_historical_import(
            [historical], args, memory_home=memory_home, router=router
        )
    )

    assert first.succeeded == 1
    assert second.preexisting == 1
    assert second.skipped == 0
    assert router.calls == 1
    assert not (memory_home / "daily").exists()


def test_dry_run_reports_models_from_environment(batch, tmp_path, monkeypatch, capsys):
    memory_home = tmp_path / "memory"
    monkeypatch.setenv("AI_MEMORY_CODEX_LUNA_MODEL", "custom-luna")
    monkeypatch.setenv("AI_MEMORY_CLAUDE_MODEL", "custom-claude")
    args = batch.parse_cli_args(["--source", "codex", "--dry-run"])

    asyncio.run(
        batch.execute_historical_import(
            make_discovered(batch, tmp_path, 1),
            args,
            memory_home=memory_home,
            router=None,
        )
    )

    assert (
        "models: custom-luna (Claude fallback: custom-claude)"
        in capsys.readouterr().out
    )


def test_report_output_distinguishes_plan_queue_and_outcomes(batch, tmp_path, capsys):
    report = batch.ImportReport(
        sessions=2,
        chunks=3,
        projects=("project",),
        dates=("2026-08-10",),
        estimated_tokens=100,
        enqueued=2,
        preexisting=1,
        newly_enqueued=2,
        processed=3,
        succeeded=2,
        failed=1,
    )
    config = batch._config_for_home(tmp_path / "memory")

    batch._print_import_report(report, config)

    output = capsys.readouterr().out
    assert "planned chunks: 3" in output
    assert "preexisting: 1" in output
    assert "newly enqueued: 2" in output
    assert "processed: 3" in output
    assert "succeeded: 2" in output
    assert "failed: 1" in output
    assert "\nenqueued:" not in output


def test_failed_import_reports_dead_job_and_main_returns_nonzero(
    batch, tmp_path, monkeypatch, capsys
):
    original_repository = batch.QueueRepository
    monkeypatch.setattr(
        batch,
        "QueueRepository",
        lambda path, **kwargs: original_repository(path, max_attempts=1, **kwargs),
    )
    args = batch.parse_cli_args(["--source", "codex"])
    report = asyncio.run(
        batch.execute_historical_import(
            make_discovered(batch, tmp_path, 1),
            args,
            memory_home=tmp_path / "memory",
            router=FailedRouter(),
        )
    )

    assert report.failed == 0
    assert report.dead == 1
    assert "failed: 0" in capsys.readouterr().out
    monkeypatch.setattr(
        batch,
        "run_batch",
        lambda _args: asyncio.sleep(0, result=report),
    )
    assert batch.main(["--source", "codex", "--dry-run"]) == 1


def test_resume_skips_completed_jobs_and_provider_fallback_does_not_change_key(
    batch, tmp_path
):
    memory_home = tmp_path / "memory"
    sessions = make_discovered(batch, tmp_path, 1)
    args = batch.parse_cli_args(["--source", "codex", "--resume"])
    first_router = TrackingRouter()
    first = asyncio.run(
        batch.execute_historical_import(
            sessions, args, memory_home=memory_home, router=first_router
        )
    )
    daily_path = memory_home / "daily" / "2026-08-10.md"
    daily_before = daily_path.read_bytes()
    queue_path = memory_home / "scripts" / "jobs.sqlite3"
    with QueueRepository(queue_path) as repository:
        job_id = repository._connection.execute("SELECT id FROM jobs").fetchone()[0]
        repository._connection.execute("DELETE FROM provider_attempts WHERE job_id = ?", (job_id,))
        repository.record_attempt(
            job_id,
            ProviderResult(
                provider="codex",
                model="gpt-5.6-luna",
                task=TaskKind.EXTRACT,
                outcome="capacity",
                reason="usage limit",
            ),
        )
        repository.record_attempt(
            job_id,
            ProviderResult(
                provider="claude",
                model="claude-sonnet-5",
                task=TaskKind.EXTRACT,
                outcome="success",
            ),
        )
    second_router = TrackingRouter()

    dry_args = batch.parse_cli_args(
        ["--source", "codex", "--resume", "--dry-run"]
    )
    before_dry_run = manifest(memory_home)
    dry_report = asyncio.run(
        batch.execute_historical_import(
            sessions, dry_args, memory_home=memory_home, router=None
        )
    )

    second = asyncio.run(
        batch.execute_historical_import(
            sessions, args, memory_home=memory_home, router=second_router
        )
    )

    connection = sqlite3.connect(queue_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    finally:
        connection.close()
    assert first.succeeded == 1
    assert second.enqueued == 0
    assert second.succeeded == 0
    assert second_router.calls == 0
    assert daily_path.read_bytes() == daily_before
    assert dry_report.preexisting == 1
    assert dry_report.skipped == 0
    assert manifest(memory_home) == before_dry_run
