from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import threading
from types import SimpleNamespace

import pytest

from providers import ProviderResult, RoutedResult, TaskKind
from scripts.queue import QueueRepository
from transcripts import NormalizedSession, Turn
import usage
import scripts.usage as queue_usage
import utils
import compile as compile_module
import connections
import lint
import query


NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def _attempt(
    provider: str = "codex",
    outcome: str = "success",
    *,
    reason: str | None = None,
    task: TaskKind = TaskKind.EXTRACT,
) -> ProviderResult:
    return ProviderResult(
        provider=provider,
        model="gpt-5.6-luna" if provider == "codex" else "claude-sonnet-5",
        task=task,
        outcome=outcome,
        input_tokens=123,
        output_tokens=45,
        elapsed_ms=678,
        reason=reason,
    )


def _session(home: Path) -> NormalizedSession:
    return NormalizedSession(
        agent="claude",
        session_id="session-1",
        project="memory",
        cwd=str(home),
        timestamp=NOW.isoformat(),
        trigger="historical",
        turns=(Turn("user", "Keep this"), Turn("assistant", "Saved")),
        source_path=str(home / "session.jsonl"),
        source_hash="hash-1",
    )


def _memory_home(tmp_path: Path) -> Path:
    home = tmp_path / "memory"
    for relative, content in {
        "AGENTS.md": "# Schema\n",
        "daily/2026-08-11.md": "# Daily\n\nA durable decision.\n",
        "knowledge/index.md": (
            "# Knowledge Base Index\n\n"
            "| Article | Project | Summary | Compiled From | Updated |\n"
            "|---|---|---|---|---|\n"
        ),
        "knowledge/log.md": "# Build Log\n",
        "scripts/state.json": '{"ingested":{},"query_count":0,"total_cost":1.25}',
    }.items():
        path = home / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return home


def _batch_module():
    path = Path(__file__).resolve().parents[1] / "batch-flush.py"
    spec = importlib.util.spec_from_file_location("task10_batch_flush", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _RoutedText:
    def __init__(self, routed: RoutedResult):
        self.routed = routed

    async def generate_text(self, _request):
        return self.routed


class _RoutedWorkspace:
    def __init__(self, routed: RoutedResult):
        self.routed = routed

    async def edit_workspace(self, _request, **_kwargs):
        return self.routed


def _write_file_answer(stage: Path) -> None:
    article = stage / "knowledge/qa/what-changed.md"
    article.parent.mkdir(parents=True, exist_ok=True)
    article.write_text(
        "---\ntitle: Q What changed\nquestion: What changed?\n"
        "consulted:\n  - concepts/example\nfiled: 2026-08-11\n---\n"
        "# Q What changed\n\n## Answer\nNothing.\n",
        encoding="utf-8",
    )
    with (stage / "knowledge/index.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "| [[qa/what-changed]] | memory | Answer | daily/2026-08-11.md | 2026-08-11 |\n"
        )
    with (stage / "knowledge/log.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## [2026-08-11T12:00:00+00:00] query (filed) | what changed\n"
            "- Filed to: [[qa/what-changed]]\n"
        )


class _ValidCompileRouter:
    async def edit_workspace(self, request):
        with (request.cwd / "knowledge/log.md").open("a", encoding="utf-8") as stream:
            stream.write(
                "\n## [2026-08-11T12:00:00+00:00] compile | 2026-08-11.md\n"
                "- Articles created: (none)\n"
            )
        attempt = ProviderResult(
            "codex", "gpt-5.6-terra", request.task, "success", text="done"
        )
        return RoutedResult.from_result(attempt, (attempt,), None)


def test_legacy_state_round_trip_preserves_unknown_fields_and_costs(tmp_path, monkeypatch):
    state_path = tmp_path / "scripts/state.json"
    state_path.parent.mkdir(parents=True)
    original = {
        "ingested": {
            "2026-04-01.md": {
                "hash": "abc",
                "compiled_at": "2026-04-01T12:00:00-05:00",
                "cost_usd": 0.52,
            }
        },
        "query_count": 3,
        "total_cost": 1.25,
        "future_field": {"preserve": True},
    }
    state_path.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(utils, "STATE_FILE", state_path)

    loaded = utils.load_state()
    utils.save_state(loaded)

    assert json.loads(state_path.read_text(encoding="utf-8")) == original


def test_malformed_legacy_state_is_rejected_without_overwrite(tmp_path, monkeypatch):
    state_path = tmp_path / "scripts/state.json"
    state_path.parent.mkdir(parents=True)
    original = b'{"ingested": '
    state_path.write_bytes(original)
    monkeypatch.setattr(utils, "STATE_FILE", state_path)

    with pytest.raises(json.JSONDecodeError):
        utils.load_state()

    assert state_path.read_bytes() == original


def test_save_state_rejects_symlink_without_locking_or_writing_outside_root(
    tmp_path, monkeypatch
):
    home = tmp_path / "memory"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text('{"safe":true}', encoding="utf-8")
    state_path = scripts / "state.json"
    state_path.symlink_to(outside)
    monkeypatch.setattr(utils, "STATE_FILE", state_path)

    with pytest.raises(ValueError, match="symlink"):
        utils.save_state({"total_cost": 99})

    assert outside.read_text(encoding="utf-8") == '{"safe":true}'
    assert not (home / "scripts/memory-writer.lock").exists()


def test_usage_record_contains_provider_outcome_tokens_and_optional_legacy_cost():
    record = usage.UsageRecord.from_attempt(
        _attempt("claude"),
        job_id=17,
        source_agent="claude",
        timestamp=NOW,
        fallback_reason="codex:capacity:subscription limit",
        legacy_cost_usd=0.04,
    )

    assert record.to_dict() == {
        "job_id": 17,
        "task": "extract",
        "source_agent": "claude",
        "provider": "claude",
        "model": "claude-sonnet-5",
        "outcome": "success",
        "fallback_reason": "codex:capacity:subscription limit",
        "input_tokens": 123,
        "output_tokens": 45,
        "elapsed_ms": 678,
        "timestamp": "2026-08-11T12:00:00+00:00",
        "cost_usd": 0.04,
    }


def test_codex_usage_never_fabricates_a_dollar_cost():
    record = usage.UsageRecord.from_attempt(
        _attempt("codex"),
        job_id=None,
        source_agent="system",
        timestamp=NOW,
        legacy_cost_usd=99.0,
    )

    assert "cost_usd" not in record.to_dict()


def test_usage_jsonl_is_compact_redacted_bounded_and_owner_only(tmp_path):
    secret = "sk-secret-value"
    noisy_reason = f"failed with {secret} " + "x" * 4_000
    record = usage.UsageRecord.from_attempt(
        _attempt("claude", "error", reason=noisy_reason),
        job_id=4,
        source_agent="codex",
        timestamp=NOW,
        fallback_reason=f"codex:error:{secret}",
    )

    path = usage.append_usage_record(
        tmp_path,
        record,
        env={"OPENAI_API_KEY": secret},
    )

    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n") and "\n " not in raw
    assert secret not in raw
    value = json.loads(raw)
    assert "[REDACTED]" in value["reason"]
    assert len(value["reason"]) <= usage.MAX_ERROR_CHARS
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o077 == 0


def test_concurrent_usage_appends_keep_every_json_line(tmp_path):
    def append(index: int) -> None:
        usage.append_usage_record(
            tmp_path,
            usage.UsageRecord.from_attempt(
                _attempt(),
                job_id=index,
                source_agent="codex",
                timestamp=NOW,
            ),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(40)))

    lines = (tmp_path / "scripts/logs/usage.jsonl").read_text().splitlines()
    assert len(lines) == 40
    assert {json.loads(line)["job_id"] for line in lines} == set(range(40))


def test_usage_log_rejects_symlink_without_touching_target(tmp_path):
    logs = tmp_path / "scripts/logs"
    logs.mkdir(parents=True)
    target = tmp_path / "outside.jsonl"
    target.write_text("safe\n", encoding="utf-8")
    (logs / "usage.jsonl").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        usage.append_usage_record(
            tmp_path,
            usage.UsageRecord.from_attempt(
                _attempt(), source_agent="system", timestamp=NOW
            ),
        )

    assert target.read_text(encoding="utf-8") == "safe\n"


@pytest.mark.parametrize("unsafe_ancestor", ["root", "scripts", "logs"])
def test_queue_open_rejects_symlinked_usage_ancestor_without_external_changes(
    tmp_path, unsafe_ancestor
):
    queue_path = tmp_path / "custom/jobs.sqlite3"
    with QueueRepository(queue_path, memory_home=tmp_path / "initial"):
        pass
    external = tmp_path / "external"
    for relative in ("usage.jsonl", "logs/usage.jsonl", "scripts/logs/usage.jsonl"):
        target = external / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'{"truncated":')
    def manifest() -> dict[Path, tuple[str, bytes | None]]:
        return {
            path.relative_to(external): (
                "directory" if path.is_dir() else "file",
                path.read_bytes() if path.is_file() else None,
            )
            for path in external.rglob("*")
        }

    before = manifest()
    home = tmp_path / "unsafe-memory"
    if unsafe_ancestor == "root":
        home.symlink_to(external, target_is_directory=True)
    elif unsafe_ancestor == "scripts":
        home.mkdir()
        (home / "scripts").symlink_to(external, target_is_directory=True)
    else:
        (home / "scripts").mkdir(parents=True)
        (home / "scripts/logs").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="usage.*(symlink|reparse)"):
        with QueueRepository(queue_path, memory_home=home):
            pass

    assert manifest() == before


def test_usage_ancestor_rejects_windows_reparse_attribute(monkeypatch):
    monkeypatch.setattr(
        usage.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400, raising=False
    )
    info = SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o700,
        st_file_attributes=0x400,
    )

    assert usage._link_or_reparse(info)


def test_valid_usage_append_does_not_rewrite_existing_log(tmp_path, monkeypatch):
    usage.append_usage_record(
        tmp_path,
        usage.UsageRecord.from_attempt(
            _attempt(), job_id=1, source_agent="system", timestamp=NOW
        ),
    )
    monkeypatch.setattr(
        usage,
        "_write_private",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("valid append rewrote the usage log")
        ),
    )

    usage.append_usage_record(
        tmp_path,
        usage.UsageRecord.from_attempt(
            _attempt(), job_id=2, source_agent="system", timestamp=NOW
        ),
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "scripts/logs/usage.jsonl").read_text().splitlines()
    ]
    assert [record["job_id"] for record in records] == [1, 2]


def test_usage_rotation_continues_projection_and_reopen_is_idempotent(
    tmp_path, monkeypatch
):
    home = tmp_path / "memory"
    queue_path = home / "scripts/jobs.sqlite3"
    monkeypatch.setattr(queue_usage, "MAX_USAGE_BYTES", 500)
    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW) as repository:
        job_id = repository.enqueue_capture(_session(home)).job_id
        for _ in range(8):
            repository.record_attempt(job_id, _attempt())

    def projected() -> list[dict[str, object]]:
        return [
            json.loads(line)
            for path in sorted((home / "scripts/logs").glob("usage*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    assert sorted(record["provider_attempt_id"] for record in projected()) == list(
        range(1, 9)
    )
    before = {path.name: path.read_bytes() for path in (home / "scripts/logs").iterdir()}
    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW):
        pass
    after = {path.name: path.read_bytes() for path in (home / "scripts/logs").iterdir()}
    assert after == before


def test_tampered_archive_is_quarantined_and_db_attempt_reprojects(
    tmp_path, monkeypatch
):
    home = tmp_path / "memory"
    queue_path = home / "scripts/jobs.sqlite3"
    monkeypatch.setattr(queue_usage, "MAX_USAGE_BYTES", 500)
    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW) as repository:
        job_id = repository.enqueue_capture(_session(home)).job_id
        for _ in range(2):
            repository.record_attempt(job_id, _attempt("codex"))

    archive = next((home / "scripts/logs").glob("usage.archive-*.jsonl"))
    original = archive.read_bytes()
    tampered = original.replace(b'"provider":"codex"', b'"provider":"evilx"', 1)
    assert len(tampered) == len(original) and tampered != original
    archive.write_bytes(tampered)
    tampered_attempt = json.loads(tampered.splitlines()[0])["provider_attempt_id"]

    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW):
        pass

    assert not archive.exists()
    quarantine = list((home / "scripts/logs").glob("usage.corrupt-*.jsonl"))
    assert len(quarantine) == 1
    assert quarantine[0].read_bytes() == tampered
    active = [
        json.loads(line)
        for line in (home / "scripts/logs/usage.jsonl").read_text().splitlines()
    ]
    repaired = [
        record for record in active
        if record.get("provider_attempt_id") == tampered_attempt
    ]
    assert len(repaired) == 1
    assert repaired[0]["provider"] == "codex"


def test_private_append_retries_short_and_interrupted_writes(tmp_path, monkeypatch):
    path = tmp_path / "usage.jsonl"
    payload = b'{"provider":"codex"}\n'
    real_write = os.write
    actions: list[int | str] = [3, "interrupt", 2, 100]

    def scripted_write(descriptor, data):
        action = actions.pop(0)
        if action == "interrupt":
            raise InterruptedError()
        count = min(action, len(data))
        return real_write(descriptor, data[:count])

    monkeypatch.setattr(usage.os, "write", scripted_write)

    usage._append_private(path, payload)

    assert path.read_bytes() == payload
    assert actions == []


def test_zero_or_failed_append_leaves_recoverable_torn_tail(tmp_path, monkeypatch):
    record = usage.UsageRecord.from_attempt(
        _attempt(), job_id=1, source_agent="system", timestamp=NOW
    )
    path = usage.append_usage_record(tmp_path, record)
    before = path.read_bytes()
    real_write = os.write
    calls = 0

    def torn_write(descriptor, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, data[:5])
        if calls == 2:
            raise OSError("disk stopped")
        return 0

    with monkeypatch.context() as patch:
        patch.setattr(usage.os, "write", torn_write)
        with pytest.raises(OSError, match="disk stopped"):
            usage._append_private(path, b'{"torn":true}\n')

    usage.recover_usage_log(tmp_path)
    assert path.read_bytes() == before
    assert len(list(path.parent.glob("usage.corrupt-*.jsonl"))) == 1

    with monkeypatch.context() as patch:
        patch.setattr(usage.os, "write", lambda *_args: 0)
        with pytest.raises(OSError, match="zero bytes"):
            usage._append_private(path, b"more\n")
    assert path.read_bytes() == before


def test_queue_attempt_is_source_of_truth_and_emits_observability_record(tmp_path):
    queue_path = tmp_path / "scripts/jobs.sqlite3"
    with QueueRepository(queue_path, clock=lambda: NOW) as repository:
        job_id = repository.enqueue_capture(_session(tmp_path)).job_id
        repository.record_attempt(
            job_id,
            _attempt("codex", "capacity", reason="subscription limit"),
        )
        repository.record_attempt(job_id, _attempt("claude"))

        attempts = repository.attempts_for(job_id)

    assert [attempt.provider for attempt in attempts] == ["codex", "claude"]
    records = [
        json.loads(line)
        for line in (tmp_path / "scripts/logs/usage.jsonl").read_text().splitlines()
    ]
    assert [record["provider"] for record in records] == ["codex", "claude"]
    assert records[0]["source_agent"] == "claude"
    assert records[1]["fallback_reason"] == "codex:capacity:subscription limit"


def test_usage_log_failure_does_not_erase_provider_attempt_source_of_truth(
    tmp_path, monkeypatch
):
    queue_path = tmp_path / "scripts/jobs.sqlite3"
    with QueueRepository(queue_path, clock=lambda: NOW) as repository:
        job_id = repository.enqueue_capture(_session(tmp_path)).job_id
        monkeypatch.setattr(
            "scripts.queue.append_usage_record",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
        )

        repository.record_attempt(job_id, _attempt())

        assert len(repository.attempts_for(job_id)) == 1


def test_queue_reopen_recovers_missing_usage_once_without_duplicates(tmp_path, monkeypatch):
    queue_path = tmp_path / "scripts/jobs.sqlite3"
    with monkeypatch.context() as patch:
        patch.setattr(
            "scripts.queue.append_usage_record",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
        )
        with QueueRepository(queue_path, clock=lambda: NOW) as repository:
            job_id = repository.enqueue_capture(_session(tmp_path)).job_id
            repository.record_attempt(job_id, _attempt())
    assert not (tmp_path / "scripts/logs/usage.jsonl").exists()

    with QueueRepository(queue_path, clock=lambda: NOW):
        pass
    with QueueRepository(queue_path, clock=lambda: NOW):
        pass

    lines = (tmp_path / "scripts/logs/usage.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["provider_attempt_id"] == 1


def test_routed_usage_records_fallback_without_changing_legacy_state(tmp_path):
    state_path = tmp_path / "scripts/state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text('{"total_cost":1.25,"query_count":3}', encoding="utf-8")
    codex = _attempt("codex", "timeout", reason="slow")
    claude = _attempt("claude")
    routed = RoutedResult.from_result(
        claude,
        (codex, claude),
        "codex:timeout:slow",
    )

    usage.record_routed_usage(tmp_path, routed, source_agent="system", timestamp=NOW)

    values = [
        json.loads(line)
        for line in (tmp_path / "scripts/logs/usage.jsonl").read_text().splitlines()
    ]
    assert len(values) == 2
    assert values[-1]["fallback_reason"] == "codex:timeout:slow"
    assert json.loads(state_path.read_text()) == {"total_cost": 1.25, "query_count": 3}


def test_usage_append_failure_preserves_previous_log(
    tmp_path, monkeypatch
):
    first = usage.UsageRecord.from_attempt(
        _attempt(), job_id=1, source_agent="system", timestamp=NOW
    )
    path = usage.append_usage_record(tmp_path, first)
    before = path.read_bytes()
    monkeypatch.setattr(
        usage,
        "_append_private",
        lambda *_args: (_ for _ in ()).throw(OSError("boom")),
    )

    with pytest.raises(OSError, match="boom"):
        usage.append_usage_record(
            tmp_path,
            usage.UsageRecord.from_attempt(
                _attempt(), job_id=2, source_agent="system", timestamp=NOW
            ),
        )

    assert path.read_bytes() == before
    assert list(path.parent.glob(".usage.jsonl.*.tmp")) == []


def test_read_only_query_records_both_attempts_without_changing_total_cost(tmp_path):
    home = _memory_home(tmp_path)
    codex = _attempt("codex", "capacity", reason="limit", task=TaskKind.QUERY)
    claude = _attempt("claude", task=TaskKind.QUERY)
    routed = RoutedResult.from_result(claude, (codex, claude), "codex:capacity:limit")

    answer = asyncio.run(
        query.run_query("What changed?", router=_RoutedText(routed), memory_home=home)
    )

    assert answer == ""
    records = [json.loads(line) for line in (home / "scripts/logs/usage.jsonl").read_text().splitlines()]
    assert [record["provider"] for record in records] == ["codex", "claude"]
    assert records[-1]["fallback_reason"] == "codex:capacity:limit"
    assert json.loads((home / "scripts/state.json").read_text())["total_cost"] == 1.25


def test_semantic_lint_records_usage_but_structural_functions_do_not(tmp_path):
    home = _memory_home(tmp_path)
    attempt = _attempt(task=TaskKind.SEMANTIC_LINT)
    routed = RoutedResult.from_result(attempt, (attempt,), None)

    assert asyncio.run(lint.check_contradictions(router=_RoutedText(routed), memory_home=home)) == []
    path = home / "scripts/logs/usage.jsonl"
    assert json.loads(path.read_text().splitlines()[0])["task"] == "semantic_lint"
    before = path.read_bytes()
    lint.check_broken_links()
    assert path.read_bytes() == before


def test_batch_extraction_records_source_agent_usage(tmp_path):
    home = _memory_home(tmp_path)
    batch = _batch_module()
    routed = RoutedResult.from_result(_attempt(), (_attempt(),), None)

    asyncio.run(
        batch.flush_chunk(
            "User: durable",
            "Session date: 2026-08-11",
            "memory",
            str(home),
            router=_RoutedText(routed),
            memory_home=home,
            source_agent="codex",
        )
    )

    record = json.loads((home / "scripts/logs/usage.jsonl").read_text().splitlines()[0])
    assert record["source_agent"] == "codex"


def test_failed_compile_records_attempts_without_fabricating_cost(tmp_path):
    home = _memory_home(tmp_path)
    state, baseline = query._state_with_baseline(home)
    codex = ProviderResult(
        "codex", "gpt-5.6-terra", TaskKind.COMPILE, "capacity", reason="limit"
    )
    claude = ProviderResult(
        "claude", "claude-sonnet-5", TaskKind.COMPILE, "timeout", reason="slow"
    )
    routed = RoutedResult.from_result(claude, (codex, claude), "codex:capacity:limit")

    result = asyncio.run(
        compile_module.compile_daily_log(
            home / "daily/2026-08-11.md",
            state,
            baseline,
            router=_RoutedWorkspace(routed),
            memory_home=home,
        )
    )

    assert result == 0.0
    records = [json.loads(line) for line in (home / "scripts/logs/usage.jsonl").read_text().splitlines()]
    assert [record["outcome"] for record in records] == ["capacity", "timeout"]
    assert all("cost_usd" not in record for record in records)


def test_compile_bookkeeping_preserves_legacy_ingested_extension_fields(tmp_path):
    home = _memory_home(tmp_path)
    daily = home / "daily/2026-08-11.md"
    daily.write_text(
        "# Daily\n\nA durable decision.\n\n"
        "<!-- @compiled-through:2026-08-11T00:00:00+00:00 -->\n",
        encoding="utf-8",
    )
    state = {
        "ingested": {
            daily.name: {
                "hash": "old",
                "compiled_at": "2026-04-01T12:00:00-05:00",
                "cost_usd": 0.52,
                "legacy_extension": "keep",
            }
        },
        "total_cost": 1.25,
    }
    (home / "scripts/state.json").write_text(json.dumps(state), encoding="utf-8")
    loaded, baseline = query._state_with_baseline(home)

    assert asyncio.run(
        compile_module.compile_daily_log(daily, loaded, baseline, memory_home=home)
    ) == 0.0

    saved = json.loads((home / "scripts/state.json").read_text())
    assert saved["ingested"][daily.name]["legacy_extension"] == "keep"
    assert saved["ingested"][daily.name]["cost_usd"] == 0.52
    assert saved["total_cost"] == 1.25


def test_new_compile_bookkeeping_does_not_fabricate_legacy_cost(tmp_path):
    home = _memory_home(tmp_path)
    state, baseline = query._state_with_baseline(home)

    asyncio.run(
        compile_module.compile_daily_log(
            home / "daily/2026-08-11.md",
            state,
            baseline,
            router=_ValidCompileRouter(),
            memory_home=home,
        )
    )

    saved = json.loads((home / "scripts/state.json").read_text())
    assert "cost_usd" not in saved["ingested"]["2026-08-11.md"]


def test_reconcile_new_entry_does_not_fabricate_legacy_cost(tmp_path, monkeypatch):
    home = _memory_home(tmp_path)
    (home / "knowledge/log.md").write_text(
        "# Build Log\n\n## [2026-08-11T12:00:00+00:00] compile | 2026-08-11.md\n",
        encoding="utf-8",
    )
    module_path = Path(__file__).resolve().parents[1] / "reconcile-state.py"
    spec = importlib.util.spec_from_file_location("task10_reconcile", module_path)
    assert spec is not None and spec.loader is not None
    reconcile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reconcile)
    monkeypatch.setattr(reconcile, "DAILY_DIR", home / "daily")
    monkeypatch.setattr(reconcile, "KNOWLEDGE_DIR", home / "knowledge")
    monkeypatch.setattr(reconcile, "LOG_MD_PATH", home / "knowledge/log.md")
    monkeypatch.setattr(reconcile, "load_state_with_baseline", lambda: query._state_with_baseline(home))
    monkeypatch.setattr("sys.argv", [str(module_path)])

    reconcile.main()

    saved = json.loads((home / "scripts/state.json").read_text())
    assert "cost_usd" not in saved["ingested"]["2026-08-11.md"]


def test_file_back_invalid_codex_stage_records_authoritative_fallback_usage(tmp_path):
    home = _memory_home(tmp_path)

    class Codex:
        async def edit_workspace(self, request):
            (request.cwd / "unexpected.txt").write_text("bad", encoding="utf-8")
            return ProviderResult(
                "codex", "gpt-5.6-terra", request.task, "success", text="bad",
                input_tokens=19, output_tokens=6, elapsed_ms=23,
            )

    class Claude:
        _model = "claude-sonnet-5"

        async def edit_workspace(self, request):
            _write_file_answer(request.cwd)
            return ProviderResult(
                "claude",
                "claude-sonnet-5",
                request.task,
                "success",
                text="fallback answer",
                input_tokens=7,
                output_tokens=3,
                elapsed_ms=8,
            )

    def factory(fallback_workspace_factory):
        from providers import ProviderRouter

        return ProviderRouter(
            Codex(), Claude(), fallback_workspace_factory=fallback_workspace_factory
        )

    assert asyncio.run(
        query.run_query(
            "What changed?",
            file_back=True,
            router_factory=factory,
            memory_home=home,
        )
    ) == "fallback answer"

    records = [json.loads(line) for line in (home / "scripts/logs/usage.jsonl").read_text().splitlines()]
    assert [(record["provider"], record["outcome"]) for record in records] == [
        ("codex", "invalid_output"),
        ("claude", "success"),
    ]
    assert (records[0]["input_tokens"], records[0]["output_tokens"]) == (19, 6)
    assert records[0]["elapsed_ms"] == 23
    assert records[-1]["fallback_reason"].startswith("codex:invalid_output:")
    assert (records[-1]["input_tokens"], records[-1]["output_tokens"]) == (7, 3)


def test_file_back_invalid_claude_fallback_records_authoritative_failure(tmp_path):
    home = _memory_home(tmp_path)

    class Codex:
        async def edit_workspace(self, request):
            (request.cwd / "unexpected.txt").write_text("bad", encoding="utf-8")
            return ProviderResult(
                "codex", "terra", request.task, "success",
                input_tokens=19, output_tokens=6, elapsed_ms=23,
            )

    class Claude:
        _model = "sonnet"

        async def edit_workspace(self, request):
            return ProviderResult("claude", "sonnet", request.task, "timeout", reason="slow")

    def factory(fallback_workspace_factory):
        from providers import ProviderRouter

        return ProviderRouter(Codex(), Claude(), fallback_workspace_factory=fallback_workspace_factory)

    answer = asyncio.run(
        query.run_query("What?", file_back=True, router_factory=factory, memory_home=home)
    )
    assert answer.startswith("Error querying knowledge base:")
    records = [json.loads(line) for line in (home / "scripts/logs/usage.jsonl").read_text().splitlines()]
    assert [(record["provider"], record["outcome"]) for record in records] == [
        ("codex", "invalid_output"),
        ("claude", "timeout"),
    ]


def test_file_back_invalid_successful_claude_stage_records_invalid_output(tmp_path):
    home = _memory_home(tmp_path)

    class Codex:
        async def edit_workspace(self, request):
            (request.cwd / "unexpected.txt").write_text("bad", encoding="utf-8")
            return ProviderResult(
                "codex", "terra", request.task, "success",
                input_tokens=19, output_tokens=6, elapsed_ms=23,
            )

    class Claude:
        _model = "sonnet"

        async def edit_workspace(self, request):
            return ProviderResult("claude", "sonnet", request.task, "success")

    def factory(fallback_workspace_factory):
        from providers import ProviderRouter

        return ProviderRouter(Codex(), Claude(), fallback_workspace_factory=fallback_workspace_factory)

    answer = asyncio.run(
        query.run_query("What?", file_back=True, router_factory=factory, memory_home=home)
    )
    assert answer.startswith("Error querying knowledge base:")
    records = [json.loads(line) for line in (home / "scripts/logs/usage.jsonl").read_text().splitlines()]
    assert [(record["provider"], record["outcome"]) for record in records] == [
        ("codex", "invalid_output"),
        ("claude", "invalid_output"),
    ]
    assert (records[0]["input_tokens"], records[0]["output_tokens"]) == (19, 6)
    assert records[0]["elapsed_ms"] == 23


@pytest.mark.parametrize("operation", ["compile", "connections"])
def test_invalid_successful_claude_workspace_records_authoritative_invalid_output(
    tmp_path, operation
):
    home = _memory_home(tmp_path)
    for slug in ("a", "b"):
        path = home / f"knowledge/concepts/{slug}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"---\ntitle: {slug}\nproject: memory\nsources:\n  - daily/2026-08-11.md\n"
            "created: 2026-08-11\nupdated: 2026-08-11\n---\n",
            encoding="utf-8",
        )

    class Codex:
        async def edit_workspace(self, request):
            (request.cwd / "unexpected.txt").write_text("bad", encoding="utf-8")
            return ProviderResult("codex", "terra", request.task, "success")

    class Claude:
        _model = "sonnet"

        async def edit_workspace(self, request):
            return ProviderResult(
                "claude",
                "sonnet",
                request.task,
                "success",
                input_tokens=11,
                output_tokens=5,
                elapsed_ms=17,
            )

    def factory(fallback_workspace_factory):
        from providers import ProviderRouter

        return ProviderRouter(Codex(), Claude(), fallback_workspace_factory=fallback_workspace_factory)

    if operation == "compile":
        state, baseline = query._state_with_baseline(home)
        asyncio.run(
            compile_module.compile_daily_log(
                home / "daily/2026-08-11.md",
                state,
                baseline,
                router_factory=factory,
                memory_home=home,
            )
        )
    else:
        asyncio.run(
            connections.synthesize_connections(
                [connections.Candidate("a", "b", [], 1.0)],
                router_factory=factory,
                memory_home=home,
            )
        )

    records = [json.loads(line) for line in (home / "scripts/logs/usage.jsonl").read_text().splitlines()]
    assert [(record["provider"], record["outcome"]) for record in records] == [
        ("codex", "invalid_output"),
        ("claude", "invalid_output"),
    ]
    assert records[-1]["fallback_reason"].startswith("codex:invalid_output:")
    assert records[-1]["model"] == "sonnet"
    assert (records[-1]["input_tokens"], records[-1]["output_tokens"]) == (11, 5)
    assert records[-1]["elapsed_ms"] == 17
    assert list((home / "scripts/staging").iterdir()) == []


@pytest.mark.parametrize("operation", ["query", "compile", "connections"])
def test_outer_router_claude_invalid_stage_reclassifies_only_final_attempt(
    tmp_path, operation
):
    home = _memory_home(tmp_path)
    for slug in ("a", "b"):
        path = home / f"knowledge/concepts/{slug}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"---\ntitle: {slug}\nproject: memory\nsources:\n  - daily/2026-08-11.md\n"
            "created: 2026-08-11\nupdated: 2026-08-11\n---\n",
            encoding="utf-8",
        )
    task = {
        "query": TaskKind.FILE_ANSWER,
        "compile": TaskKind.COMPILE,
        "connections": TaskKind.CONNECTIONS,
    }[operation]
    codex = ProviderResult(
        "codex", "terra", task, "capacity", input_tokens=2, output_tokens=1,
        elapsed_ms=4, reason="subscription full",
    )
    claude = ProviderResult(
        "claude", "sonnet", task, "success", input_tokens=13, output_tokens=8,
        elapsed_ms=21,
    )
    routed = RoutedResult.from_result(
        claude, (codex, claude), "codex:capacity:subscription full"
    )

    class InvalidOuterRouter:
        async def edit_workspace(self, request, **_kwargs):
            (request.cwd / "unexpected.txt").write_text("invalid", encoding="utf-8")
            return routed

    if operation == "query":
        answer = asyncio.run(
            query.run_query(
                "What?", file_back=True, router=InvalidOuterRouter(), memory_home=home
            )
        )
        assert answer.startswith("Error querying knowledge base:")
    elif operation == "compile":
        state, baseline = query._state_with_baseline(home)
        asyncio.run(
            compile_module.compile_daily_log(
                home / "daily/2026-08-11.md", state, baseline,
                router=InvalidOuterRouter(), memory_home=home,
            )
        )
    else:
        asyncio.run(
            connections.synthesize_connections(
                [connections.Candidate("a", "b", [], 1.0)],
                router=InvalidOuterRouter(), memory_home=home,
            )
        )

    records = [
        json.loads(line)
        for line in (home / "scripts/logs/usage.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert [(record["provider"], record["outcome"]) for record in records] == [
        ("codex", "capacity"),
        ("claude", "invalid_output"),
    ]
    assert records[0]["reason"] == "subscription full"
    assert records[-1]["model"] == "sonnet"
    assert records[-1]["task"] == task.value
    assert (records[-1]["input_tokens"], records[-1]["output_tokens"]) == (13, 8)
    assert records[-1]["elapsed_ms"] == 21
    assert records[-1]["fallback_reason"] == "codex:capacity:subscription full"
    assert list((home / "scripts/staging").iterdir()) == []


def test_custom_queue_projects_usage_only_to_canonical_memory_home(tmp_path):
    home = tmp_path / "memory"
    custom = tmp_path / "custom-queue/jobs.sqlite3"
    with QueueRepository(custom, memory_home=home, clock=lambda: NOW) as repository:
        job = repository.enqueue_capture(_session(home))
        repository.record_attempt(job.job_id, _attempt())

    assert (home / "scripts/logs/usage.jsonl").exists()
    assert not (custom.parent / "logs").exists()


def test_capture_safe_queue_open_defers_authoritative_attempt_projection(tmp_path):
    home = tmp_path / "memory"
    queue_path = home / "scripts/jobs.sqlite3"
    with QueueRepository(
        queue_path, memory_home=home, clock=lambda: NOW, sync_usage=False
    ) as repository:
        job = repository.enqueue_capture(_session(home))
        repository._append_attempt_usage = lambda _attempt_id: None
        repository.record_attempt(job.job_id, _attempt())

    usage_path = home / "scripts/logs/usage.jsonl"
    assert not usage_path.exists()
    with QueueRepository(
        queue_path, memory_home=home, clock=lambda: NOW, sync_usage=False
    ):
        pass
    assert not usage_path.exists()

    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW):
        pass
    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW):
        pass

    records = [json.loads(line) for line in usage_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["provider_attempt_id"] == 1


def test_capture_safe_queue_open_defers_corrupt_usage_recovery(tmp_path):
    home = tmp_path / "memory"
    queue_path = home / "scripts/jobs.sqlite3"
    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW) as repository:
        job = repository.enqueue_capture(_session(home))
        repository.record_attempt(job.job_id, _attempt())
    usage_path = home / "scripts/logs/usage.jsonl"
    usage_path.write_bytes(usage_path.read_bytes() + b'{"truncated":')

    with QueueRepository(
        queue_path, memory_home=home, clock=lambda: NOW, sync_usage=False
    ):
        pass

    assert usage_path.read_bytes().endswith(b'{"truncated":')
    assert list(usage_path.parent.glob("usage.corrupt-*.jsonl")) == []

    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW):
        pass

    assert not usage_path.read_bytes().endswith(b'{"truncated":')
    assert len(list(usage_path.parent.glob("usage.corrupt-*.jsonl"))) == 1


@pytest.mark.parametrize("corruption", [b'{"broken":', b'{bad}\n'])
def test_queue_reopen_quarantines_corrupt_usage_and_reprojects_once(tmp_path, corruption):
    home = tmp_path / "memory"
    queue_path = home / "scripts/jobs.sqlite3"
    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW) as repository:
        job = repository.enqueue_capture(_session(home))
        repository.record_attempt(job.job_id, _attempt())
    usage_path = home / "scripts/logs/usage.jsonl"
    valid = usage_path.read_bytes()
    usage_path.write_bytes(valid + corruption + valid)

    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW):
        pass
    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW):
        pass

    records = [json.loads(line) for line in usage_path.read_text().splitlines()]
    assert len(records) == 1
    quarantine = list((home / "scripts/logs").glob("usage.corrupt-*.jsonl"))
    assert len(quarantine) == 1
    assert corruption.strip() in quarantine[0].read_bytes()
    assert quarantine[0].stat().st_mode & 0o777 == 0o600


def test_concurrent_corrupt_usage_recovery_is_idempotent(tmp_path):
    home = tmp_path / "memory"
    queue_path = home / "scripts/jobs.sqlite3"
    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW) as repository:
        job = repository.enqueue_capture(_session(home))
        repository.record_attempt(job.job_id, _attempt())
    path = home / "scripts/logs/usage.jsonl"
    path.write_bytes(path.read_bytes() + b'{"truncated":')

    def reopen():
        with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW):
            pass

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: reopen(), range(2)))
    assert len(path.read_text().splitlines()) == 1
    assert len(list(path.parent.glob("usage.corrupt-*.jsonl"))) == 1


def test_recreated_queue_does_not_collide_with_old_attempt_ids(tmp_path):
    home = tmp_path / "memory"
    queue_path = home / "scripts/jobs.sqlite3"
    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW) as repository:
        job = repository.enqueue_capture(_session(home))
        repository.record_attempt(job.job_id, _attempt("codex"))
    for suffix in ("", "-wal", "-shm"):
        Path(f"{queue_path}{suffix}").unlink(missing_ok=True)
    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW) as repository:
        job = repository.enqueue_capture(_session(home))
        repository.record_attempt(job.job_id, _attempt("claude"))
    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW):
        pass

    records = [json.loads(line) for line in (home / "scripts/logs/usage.jsonl").read_text().splitlines()]
    assert [record["provider"] for record in records] == ["codex", "claude"]
    assert records[0]["queue_id"] != records[1]["queue_id"]


def test_v1_queue_migrates_atomically_and_preserves_attempts(tmp_path):
    queue_path = tmp_path / "memory/scripts/jobs.sqlite3"
    queue_path.parent.mkdir(parents=True)
    connection = __import__("sqlite3").connect(queue_path)
    connection.executescript(
        """
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY, kind TEXT, source_agent TEXT, session_id TEXT,
            project TEXT, cwd TEXT, trigger TEXT, source_path TEXT, source_hash TEXT,
            payload_json TEXT, status TEXT, attempt_count INTEGER, available_at TEXT,
            lease_owner TEXT, lease_expires_at TEXT, last_error TEXT, created_at TEXT,
            updated_at TEXT, completed_at TEXT
        );
        CREATE TABLE provider_attempts (
            id INTEGER PRIMARY KEY, job_id INTEGER, provider TEXT, model TEXT,
            task TEXT, started_at TEXT, ended_at TEXT, outcome TEXT, reason TEXT,
            input_tokens INTEGER, output_tokens INTEGER, elapsed_ms INTEGER
        );
        PRAGMA user_version = 1;
        """
    )
    connection.close()

    with QueueRepository(queue_path, memory_home=tmp_path / "memory") as repository:
        columns = {
            row[1] for row in repository._connection.execute(
                "PRAGMA table_info(provider_attempts)"
            )
        }
        assert "legacy_cost_usd" in columns
        assert len(repository.queue_id) == 32
        assert repository._connection.execute("PRAGMA user_version").fetchone()[0] == 2


def test_corrupt_usage_recovery_rejects_unsafe_existing_quarantine(tmp_path):
    home = tmp_path / "memory"
    queue_path = home / "scripts/jobs.sqlite3"
    with QueueRepository(queue_path, memory_home=home, clock=lambda: NOW) as repository:
        job = repository.enqueue_capture(_session(home))
        repository.record_attempt(job.job_id, _attempt())
    usage_path = home / "scripts/logs/usage.jsonl"
    corrupt = b'{"truncated":'
    usage_path.write_bytes(usage_path.read_bytes() + corrupt)
    digest = __import__("hashlib").sha256(corrupt).hexdigest()
    quarantine = usage_path.parent / f"usage.corrupt-{digest}.jsonl"
    outside = tmp_path / "outside"
    outside.write_bytes(b"safe")
    quarantine.symlink_to(outside)

    with pytest.raises(ValueError, match="quarantine"):
        usage.recover_usage_log(home)
    assert outside.read_bytes() == b"safe"


def test_update_state_preserves_concurrent_fields_and_retries(tmp_path, monkeypatch):
    state_path = tmp_path / "scripts/state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text('{"query_count":2,"unknown":"keep"}', encoding="utf-8")
    monkeypatch.setattr(utils, "STATE_FILE", state_path)

    utils.update_state(lambda state: state.__setitem__("last_lint", "now"))

    assert json.loads(state_path.read_text()) == {
        "query_count": 2, "unknown": "keep", "last_lint": "now"
    }


def test_update_state_retry_exhaustion_does_not_overwrite(tmp_path, monkeypatch):
    state_path = tmp_path / "scripts/state.json"
    state_path.parent.mkdir(parents=True)
    original = b'{"query_count":9,"unknown":"keep"}'
    state_path.write_bytes(original)
    monkeypatch.setattr(utils, "STATE_FILE", state_path)
    monkeypatch.setattr(
        utils, "capture_file_baseline", lambda _path: utils.FileBaseline(True, 1, "race")
    )

    with pytest.raises(RuntimeError, match="conflicted"):
        utils.update_state(lambda state: state.__setitem__("last_lint", "now"), max_attempts=2)
    assert state_path.read_bytes() == original


def test_connections_records_authoritative_usage(tmp_path):
    home = _memory_home(tmp_path)
    for slug in ("a", "b"):
        path = home / f"knowledge/concepts/{slug}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"---\ntitle: {slug}\nproject: memory\nsources:\n  - daily/2026-08-11.md\n"
            "created: 2026-08-11\nupdated: 2026-08-11\n---\n",
            encoding="utf-8",
        )
    attempt = ProviderResult(
        "codex", "gpt-5.6-terra", TaskKind.CONNECTIONS, "success", text="done"
    )
    routed = RoutedResult.from_result(attempt, (attempt,), None)

    class ValidConnectionsRouter:
        async def edit_workspace(self, request, **_kwargs):
            with (request.cwd / "knowledge/log.md").open("a", encoding="utf-8") as stream:
                stream.write(
                    "\n## [2026-08-11T12:00:00+00:00] connections | swanson-pass\n"
                    "- Candidates evaluated: 1\n"
                    "- Connections created: none\n"
                    "- Rejected (co-occurrence / too weak): concepts/a <-> concepts/b - weak\n"
                )
            return routed

    asyncio.run(
        connections.synthesize_connections(
            [connections.Candidate("a", "b", ["bridge"], 1.0)],
            router=ValidConnectionsRouter(),
            memory_home=home,
        )
    )

    record = json.loads((home / "scripts/logs/usage.jsonl").read_text().splitlines()[0])
    assert record["task"] == "connections"
