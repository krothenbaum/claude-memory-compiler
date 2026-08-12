from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
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


def manifest(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


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


@pytest.mark.parametrize(
    ("source", "expected"),
    [("claude", ("claude",)), ("codex", ("codex",)), ("all", ("claude", "codex"))],
)
def test_run_batch_discovers_only_requested_sources(batch, source, expected, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        batch,
        "discover_claude_sessions",
        lambda _targets: calls.append("claude") or [],
    )
    monkeypatch.setattr(
        batch,
        "discover_codex_sessions",
        lambda _root: calls.append("codex") or batch.CodexDiscovery(()),
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
