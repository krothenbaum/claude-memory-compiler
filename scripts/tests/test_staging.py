from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from staging import (
    ApplyBookkeeping,
    RetryableApplyError,
    StageValidationError,
    apply_validated_stage,
    create_stage,
    recover_incomplete_apply,
    snapshot_manifest,
    validate_stage,
)


ARTICLE = """---
title: "Original"
project: memory
sources:
  - "daily/2026-08-11.md"
---

# Original

## Key Points

- One
"""


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


@pytest.fixture
def memory_home(tmp_path: Path) -> Path:
    root = tmp_path / "memory"
    _write(root / "AGENTS.md", "# Schema\n")
    _write(root / "daily/2026-08-11.md", "# Daily\n\nSession source\n")
    _write(
        root / "knowledge/index.md",
        "# Index\n\n| Article | Project | Summary | Compiled From | Updated |\n"
        "|---|---|---|---|---|\n"
        "| [[concepts/original]] | memory | Original | daily/2026-08-11.md | 2026-08-11 |\n",
    )
    _write(root / "knowledge/log.md", "# Build Log\n")
    _write(root / "knowledge/concepts/original.md", ARTICLE)
    _write(root / "scripts/state.json", '{"ingested": {}}\n')
    return root


def _compile_stage(memory_home: Path, *, attempt_id: str = "attempt-1"):
    stage = create_stage(
        memory_home,
        "job-1",
        attempt_id,
        daily_source="daily/2026-08-11.md",
        relevant_articles=("knowledge/concepts/original.md",),
    )
    article = stage.root / "knowledge/concepts/original.md"
    article.write_text(ARTICLE.replace('title: "Original"', 'title: "Updated"'), encoding="utf-8")
    with (stage.root / "knowledge/index.md").open("a", encoding="utf-8") as handle:
        handle.write("| [[concepts/original]] | memory | Updated | daily/2026-08-11.md | 2026-08-11 |\n")
    with (stage.root / "knowledge/log.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "## [2026-08-11T12:00:00+00:00] compile | 2026-08-11.md\n"
            "- Articles updated: [[concepts/original]]\n"
        )
    return stage


def test_create_stage_copies_only_selected_files_and_records_private_manifest(memory_home):
    _write(memory_home / "daily/not-selected.md", "secret\n")
    _write(memory_home / "knowledge/concepts/not-selected.md", ARTICLE)

    stage = create_stage(
        memory_home,
        "job/../../unsafe",
        "attempt/../../unsafe",
        daily_source="daily/2026-08-11.md",
        relevant_articles=("knowledge/concepts/original.md",),
    )

    assert stage.root.parent == memory_home / "scripts/staging"
    assert stage.root.name == "job-unsafe-attempt-unsafe"
    assert set(stage.baseline) == {
        "AGENTS.md",
        "daily/2026-08-11.md",
        "knowledge/index.md",
        "knowledge/log.md",
        "knowledge/concepts/original.md",
        "scripts/state.json",
    }
    assert not (stage.root / "daily/not-selected.md").exists()
    assert not (stage.root / "knowledge/concepts/not-selected.md").exists()
    assert stat.S_IMODE(stage.root.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in stage.root.rglob("*")
        if path.is_file()
    )
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o700
        for path in stage.root.rglob("*")
        if path.is_dir()
    )
    assert all(entry.sha256 and entry.size >= 0 for entry in stage.baseline.values())


def test_create_stage_rejects_traversal_and_symlink_sources(memory_home, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (memory_home / "knowledge/concepts/link.md").symlink_to(outside)

    with pytest.raises(StageValidationError, match="relative|escape"):
        create_stage(memory_home, "1", "1", daily_source="../outside.md")
    with pytest.raises(StageValidationError, match="symlink"):
        create_stage(
            memory_home,
            "1",
            "2",
            daily_source="daily/2026-08-11.md",
            relevant_articles=("knowledge/concepts/link.md",),
        )


def test_manifest_reports_before_after_hashes_and_allowed_changes(memory_home):
    stage = _compile_stage(memory_home)
    validated = validate_stage(
        stage,
        allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
        task="compile",
    )

    assert set(validated.changed_paths) == {
        "knowledge/concepts/original.md",
        "knowledge/index.md",
        "knowledge/log.md",
    }
    assert validated.before["knowledge/concepts/original.md"].sha256 != validated.after[
        "knowledge/concepts/original.md"
    ].sha256


@pytest.mark.parametrize("bad_change", ["deletion", "daily", "unexpected", "invalid_utf8"])
def test_validation_rejects_unsafe_changes_without_touching_real_files(memory_home, bad_change):
    before = {path.relative_to(memory_home): path.read_bytes() for path in memory_home.rglob("*") if path.is_file()}
    stage = _compile_stage(memory_home)
    if bad_change == "deletion":
        (stage.root / "knowledge/index.md").unlink()
    elif bad_change == "daily":
        (stage.root / "daily/2026-08-11.md").write_text("model edit", encoding="utf-8")
    elif bad_change == "unexpected":
        _write(stage.root / "private/escape.md", "surprise")
    else:
        (stage.root / "knowledge/concepts/original.md").write_bytes(b"\xff\xfe")

    with pytest.raises(StageValidationError):
        validate_stage(
            stage,
            allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
            task="compile",
        )

    assert {path.relative_to(memory_home): path.read_bytes() for path in memory_home.rglob("*") if path.is_file() and "staging" not in path.parts} == before


def test_manifest_rejects_symlink_and_special_file(memory_home):
    stage = _compile_stage(memory_home)
    article = stage.root / "knowledge/concepts/original.md"
    article.unlink()
    article.symlink_to(memory_home / "knowledge/concepts/original.md")
    with pytest.raises(StageValidationError, match="symlink"):
        snapshot_manifest(stage.root)

    article.unlink()
    if hasattr(os, "mkfifo"):
        os.mkfifo(article)
        with pytest.raises(StageValidationError, match="special"):
            snapshot_manifest(stage.root)


def test_validation_rejects_malformed_utf8_in_unchanged_selected_source(memory_home):
    (memory_home / "daily/2026-08-11.md").write_bytes(b"\xff\xfe")
    stage = create_stage(
        memory_home,
        "job",
        "utf8",
        daily_source="daily/2026-08-11.md",
    )

    with pytest.raises(StageValidationError, match="UTF-8"):
        validate_stage(
            stage,
            allowed_paths=("knowledge/index.md", "knowledge/log.md"),
            task="compile",
        )


def test_validation_rejects_incompatible_staged_state(memory_home):
    (memory_home / "scripts/state.json").write_text("[]", encoding="utf-8")
    stage = create_stage(memory_home, "job", "state")

    with pytest.raises(StageValidationError, match="state"):
        validate_stage(
            stage,
            allowed_paths=("knowledge/index.md", "knowledge/log.md"),
            task="compile",
        )


def test_manifest_rejects_hard_linked_stage_file(memory_home, tmp_path):
    stage = _compile_stage(memory_home)
    article = stage.root / "knowledge/concepts/original.md"
    attacker = tmp_path / "attacker.md"
    attacker.hardlink_to(article)

    with pytest.raises(StageValidationError, match="hard-linked"):
        snapshot_manifest(stage.root)


def test_validation_records_owner_only_after_manifest(memory_home):
    stage = _compile_stage(memory_home)
    validated = validate_stage(
        stage,
        allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
        task="compile",
    )

    recorded = stage.root / ".stage-manifest-after.json"
    payload = json.loads(recorded.read_text(encoding="utf-8"))
    assert payload["knowledge/concepts/original.md"]["sha256"] == validated.after[
        "knowledge/concepts/original.md"
    ].sha256
    assert stat.S_IMODE(recorded.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("task", "article_path"),
    [
        ("file_answer", "knowledge/qa/answer.md"),
        ("connections", "knowledge/connections/a-and-b.md"),
    ],
)
def test_specialized_tasks_require_article_index_row_and_log_entry(memory_home, task, article_path):
    stage = create_stage(memory_home, "job", task, daily_source="daily/2026-08-11.md")
    article = ARTICLE.replace('title: "Original"', f'title: "{task}"')
    _write(stage.root / article_path, article)
    slug = article_path.removeprefix("knowledge/").removesuffix(".md")

    with pytest.raises(StageValidationError, match="index"):
        validate_stage(stage, allowed_paths=(article_path, "knowledge/index.md", "knowledge/log.md"), task=task)

    with (stage.root / "knowledge/index.md").open("a", encoding="utf-8") as handle:
        handle.write(f"| [[{slug}]] | memory | Added | daily/2026-08-11.md | 2026-08-11 |\n")
    with pytest.raises(StageValidationError, match="log"):
        validate_stage(stage, allowed_paths=(article_path, "knowledge/index.md", "knowledge/log.md"), task=task)

    with (stage.root / "knowledge/log.md").open("a", encoding="utf-8") as handle:
        handle.write(f"## [now] {task}\n- Filed to: [[{slug}]]\n")
    validated = validate_stage(stage, allowed_paths=(article_path, "knowledge/index.md", "knowledge/log.md"), task=task)
    assert article_path in validated.changed_paths


def test_compile_article_requires_both_index_and_build_log_changes(memory_home):
    stage = _compile_stage(memory_home)
    (stage.root / "knowledge/log.md").write_bytes(stage.baseline_bytes["knowledge/log.md"])
    with pytest.raises(StageValidationError, match="build log"):
        validate_stage(
            stage,
            allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
            task="compile",
        )


@pytest.mark.parametrize("malformed", ["index", "log"])
def test_validation_rejects_malformed_new_index_or_build_log_entries(memory_home, malformed):
    stage = create_stage(
        memory_home,
        "job",
        malformed,
        daily_source="daily/2026-08-11.md",
        relevant_articles=("knowledge/concepts/original.md",),
    )
    article = stage.root / "knowledge/concepts/original.md"
    article.write_text(ARTICLE.replace('title: "Original"', 'title: "Changed"'), encoding="utf-8")
    with (stage.root / "knowledge/index.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "broken [[concepts/original]] row\n"
            if malformed == "index"
            else "| [[concepts/original]] | memory | Changed | daily/2026-08-11.md | 2026-08-11 |\n"
        )
    with (stage.root / "knowledge/log.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "just text [[concepts/original]]\n"
            if malformed == "log"
            else "## [2026-08-11T12:00:00+00:00] compile | 2026-08-11.md\n"
            "- Articles updated: [[concepts/original]]\n"
        )

    with pytest.raises(StageValidationError, match=malformed.replace("index", "index row").replace("log", "build log")):
        validate_stage(
            stage,
            allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
            task="compile",
        )


def test_invalid_codex_stage_is_discarded_and_fallback_factory_gets_fresh_stage(memory_home):
    codex = _compile_stage(memory_home, attempt_id="codex")
    (codex.root / "daily/2026-08-11.md").write_text("contaminated", encoding="utf-8")
    with pytest.raises(StageValidationError):
        validate_stage(
            codex,
            allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
            task="compile",
        )

    claude = create_stage(
        memory_home,
        "job-1",
        "claude",
        daily_source="daily/2026-08-11.md",
        relevant_articles=("knowledge/concepts/original.md",),
    )
    assert claude.root != codex.root
    assert (claude.root / "daily/2026-08-11.md").read_text(encoding="utf-8") == "# Daily\n\nSession source\n"


@pytest.mark.parametrize("failure_step", range(1, 6))
def test_apply_failure_after_each_replace_restores_every_real_file(memory_home, failure_step):
    stage = _compile_stage(memory_home, attempt_id=f"failure-{failure_step}")
    validated = validate_stage(
        stage,
        allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
        task="compile",
    )
    tracked = [
        memory_home / "knowledge/concepts/original.md",
        memory_home / "knowledge/index.md",
        memory_home / "knowledge/log.md",
        memory_home / "daily/2026-08-11.md",
        memory_home / "scripts/state.json",
    ]
    before = {path: path.read_bytes() for path in tracked}

    def fail(step, _path):
        if step == failure_step:
            raise OSError(f"injected at {step}")

    bookkeeping = ApplyBookkeeping(
        compiled_marker_path="daily/2026-08-11.md",
        compiled_at="2026-08-11T12:00:00+00:00",
        state={"ingested": {"2026-08-11.md": {"hash": "new"}}},
        failure_injector=fail,
    )
    with pytest.raises(RetryableApplyError, match="rolled back"):
        apply_validated_stage(validated, stage.baseline, bookkeeping)

    assert {path: path.read_bytes() for path in tracked} == before
    assert list((memory_home / "scripts/memory-apply-journal").glob("*.json")) == []


def test_apply_commits_stage_marker_and_state_together(memory_home):
    stage = _compile_stage(memory_home)
    validated = validate_stage(
        stage,
        allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
        task="compile",
    )
    result = apply_validated_stage(
        validated,
        stage.baseline,
        ApplyBookkeeping(
            compiled_marker_path="daily/2026-08-11.md",
            compiled_at="2026-08-11T12:00:00+00:00",
            state={"ingested": {"2026-08-11.md": {"hash": "new"}}},
        ),
    )

    assert 'title: "Updated"' in (memory_home / "knowledge/concepts/original.md").read_text(encoding="utf-8")
    assert "@compiled-through:2026-08-11T12:00:00+00:00" in (memory_home / "daily/2026-08-11.md").read_text(encoding="utf-8")
    assert json.loads((memory_home / "scripts/state.json").read_text(encoding="utf-8"))["ingested"]
    assert set(result.changed_paths) >= set(validated.changed_paths)
    assert list((memory_home / "scripts/memory-apply-journal").glob("*.json")) == []


def test_apply_rechecks_real_baseline_before_writing(memory_home):
    stage = _compile_stage(memory_home)
    validated = validate_stage(
        stage,
        allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
        task="compile",
    )
    real_article = memory_home / "knowledge/concepts/original.md"
    real_article.write_text("concurrent owner edit", encoding="utf-8")

    with pytest.raises(RetryableApplyError, match="baseline"):
        apply_validated_stage(validated, stage.baseline, ApplyBookkeeping())
    assert real_article.read_text(encoding="utf-8") == "concurrent owner edit"


def test_recovery_restores_a_simulated_crash_journal(memory_home):
    article = memory_home / "knowledge/concepts/original.md"
    original = article.read_bytes()
    replacement = b"partially applied"
    article.write_bytes(replacement)
    journal_dir = memory_home / "scripts/memory-apply-journal"
    journal_dir.mkdir(mode=0o700)
    journal = journal_dir / "crash.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "state": "applying",
                "root": str(memory_home),
                "entries": [
                    {
                        "path": "knowledge/concepts/original.md",
                        "original": original.hex(),
                        "replacement": replacement.hex(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    journal.chmod(0o600)

    assert recover_incomplete_apply(memory_home) is True
    assert article.read_bytes() == original
    assert not journal.exists()
    assert recover_incomplete_apply(memory_home) is False


def test_daily_writer_uses_shared_lock_and_agent_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_MEMORY_HOME", str(tmp_path))
    # flush.py deliberately sets this recursion guard at import time. Let the
    # fixture restore the process environment after exercising its wrapper.
    monkeypatch.setenv("CLAUDE_INVOKED_BY", "test-sentinel")
    import flush

    path = flush.append_to_daily_log(
        "Durable result",
        project_key="memory",
        cwd="/tmp/memory",
        agent="codex",
        memory_home=tmp_path,
    )
    content = path.read_text(encoding="utf-8")
    assert "**Agent:** Codex" in content
    assert "**Project:** memory" in content
    assert "Durable result" in content
    assert (tmp_path / "scripts/memory-writer.lock").exists()


def test_worker_capture_success_uses_task6_daily_writer_boundary(tmp_path, monkeypatch):
    from worker import daily_writer_boundary

    monkeypatch.setenv("AI_MEMORY_HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    daily_writer_boundary(
        SimpleNamespace(id=1, project="memory", cwd="/tmp/memory", source_agent="codex"),
        "Captured by worker",
    )

    daily_files = list((tmp_path / "daily").glob("*.md"))
    assert len(daily_files) == 1
    assert "**Agent:** Codex" in daily_files[0].read_text(encoding="utf-8")


def test_daily_writer_rejects_symlink_without_mutating_target(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_INVOKED_BY", "test-sentinel")
    import flush

    today = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).astimezone()
    daily = tmp_path / "daily"
    daily.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("owner content", encoding="utf-8")
    (daily / f"{today.strftime('%Y-%m-%d')}.md").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        flush.append_to_daily_log("attacker append", memory_home=tmp_path)
    assert outside.read_text(encoding="utf-8") == "owner content"


def test_bookkeeping_is_immutable_between_validation_and_apply(memory_home):
    stage = _compile_stage(memory_home)
    validated = validate_stage(
        stage,
        allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
        task="compile",
    )
    tampered = replace(validated, changed_paths=validated.changed_paths + ("scripts/state.json",))
    with pytest.raises(StageValidationError):
        apply_validated_stage(tampered, stage.baseline, ApplyBookkeeping())
