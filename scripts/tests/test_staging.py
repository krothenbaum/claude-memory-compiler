from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import time
from types import SimpleNamespace

import pytest

from staging import (
    ApplyBookkeeping,
    RetryableApplyError,
    StageValidationError,
    apply_validated_stage,
    create_fallback_stage,
    create_stage,
    recover_incomplete_apply,
    snapshot_manifest,
    validate_stage,
)
from utils import ExclusiveFileLock, FileBaseline, capture_file_baseline


ARTICLE = """---
title: "Original"
project: memory
sources:
  - "daily/2026-08-11.md"
created: 2026-08-11
updated: 2026-08-11
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
        include_state=True,
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


def _schema_article(kind: str) -> tuple[str, str, str]:
    if kind == "concept":
        return "knowledge/concepts/scalar.md", ARTICLE, "compile"
    if kind == "connection":
        return (
            "knowledge/connections/scalar.md",
            ARTICLE.replace(
                'title: "Original"',
                'title: "Connection"\nconnects:\n  - concepts/a\n  - concepts/b',
            ),
            "connections",
        )
    return (
        "knowledge/qa/scalar.md",
        "---\ntitle: Q\nquestion: Why?\nconsulted:\n"
        "  - concepts/original\nfiled: 2026-08-11\n---\n# Q\n",
        "file_answer",
    )


_SCALAR_ORIGINALS = {
    "title": {'concept': '"Original"', "connection": '"Connection"', "qa": "Q"},
    "question": {"qa": "Why?"},
    "created": {"concept": "2026-08-11", "connection": "2026-08-11"},
    "updated": {"concept": "2026-08-11", "connection": "2026-08-11"},
    "filed": {"qa": "2026-08-11"},
}
_SCALAR_MATRIX = [
    (kind, field, bad)
    for kind, fields in {
        "concept": ("title", "created", "updated"),
        "connection": ("title", "created", "updated"),
        "qa": ("title", "question", "filed"),
    }.items()
    for field in fields
    for bad in ('""', "[not-a-scalar]", "{nested: value}")
]


def test_create_stage_copies_only_selected_files_and_records_private_manifest(memory_home):
    _write(memory_home / "daily/not-selected.md", "secret\n")
    _write(memory_home / "knowledge/concepts/not-selected.md", ARTICLE)

    stage = create_stage(
        memory_home,
        "job/../../unsafe",
        "attempt/../../unsafe",
        daily_source="daily/2026-08-11.md",
        relevant_articles=("knowledge/concepts/original.md",),
        include_state=True,
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


@pytest.mark.parametrize(
    "relative",
    [
        "AGENTS.md",
        "scripts/state.json",
        "scripts/secrets/token.txt",
        "knowledge/index.md",
        "knowledge/concepts/not-markdown.txt",
        "daily/2026-08-11.md",
    ],
)
def test_relevant_articles_accepts_only_markdown_article_paths(memory_home, relative):
    _write(memory_home / relative, "sensitive")
    with pytest.raises(StageValidationError, match="relevant article"):
        create_stage(
            memory_home,
            "job",
            relative,
            relevant_articles=(relative,),
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


def test_allowed_globs_are_anchored_at_stage_root(memory_home):
    stage = _compile_stage(memory_home)
    _write(stage.root / "x/knowledge/concepts/injected.md", ARTICLE)

    with pytest.raises(StageValidationError, match="outside allowlist"):
        validate_stage(
            stage,
            allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
            task="compile",
        )


def test_recursive_allowed_glob_is_root_anchored_and_explicit(memory_home):
    stage = _compile_stage(memory_home)
    nested = stage.root / "knowledge/concepts/nested/new.md"
    _write(nested, ARTICLE.replace('title: "Original"', 'title: "Nested"'))
    with (stage.root / "knowledge/index.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "| [[concepts/nested/new]] | memory | Nested | daily/2026-08-11.md | 2026-08-11 |\n"
        )
    with (stage.root / "knowledge/log.md").open("a", encoding="utf-8") as handle:
        handle.write("## [now] compile | source\n- Created: [[concepts/nested/new]]\n")

    validated = validate_stage(
        stage,
        allowed_paths=("knowledge/concepts/**", "knowledge/index.md", "knowledge/log.md"),
        task="compile",
    )
    assert "knowledge/concepts/nested/new.md" in validated.changed_paths


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
    stage = create_stage(memory_home, "job", "state", include_state=True)

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
    if task == "connections":
        article = ARTICLE.replace(
            'title: "Original"',
            f'title: "{task}"\nconnects:\n  - "concepts/original"\n  - "concepts/other"',
        )
    else:
        article = (
            "---\ntitle: Q\nquestion: Why?\nconsulted:\n"
            "  - concepts/original\nfiled: 2026-08-11\n---\n# Q\n"
        )
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


def test_schema_file_must_remain_unchanged_even_if_allowlisted(memory_home):
    stage = _compile_stage(memory_home)
    (stage.root / "AGENTS.md").write_text("changed schema", encoding="utf-8")
    with pytest.raises(StageValidationError, match="AGENTS"):
        validate_stage(
            stage,
            allowed_paths=(
                "AGENTS.md",
                "knowledge/concepts/*.md",
                "knowledge/index.md",
                "knowledge/log.md",
            ),
            task="compile",
        )


@pytest.mark.parametrize(
    ("relative", "body", "missing"),
    [
        (
            "knowledge/concepts/bad.md",
            ARTICLE.replace("sources:\n  - \"daily/2026-08-11.md\"\n", ""),
            "sources",
        ),
        ("knowledge/concepts/bad.md", ARTICLE.replace("created: 2026-08-11\n", ""), "created"),
        ("knowledge/concepts/bad.md", ARTICLE.replace("updated: 2026-08-11\n", ""), "updated"),
        ("knowledge/connections/bad.md", ARTICLE, "connects"),
        (
            "knowledge/qa/bad.md",
            "---\ntitle: Q\nquestion: Why?\nconsulted:\n"
            "  - concepts/original\nfiled: 2026-08-11\n---\n# Q\n",
            "unused",
        ),
    ],
)
def test_article_frontmatter_enforces_schema_required_keys(memory_home, relative, body, missing):
    stage = create_stage(memory_home, "job", missing)
    _write(stage.root / relative, body)
    slug = relative.removeprefix("knowledge/").removesuffix(".md")
    with (stage.root / "knowledge/index.md").open("a", encoding="utf-8") as handle:
        handle.write(f"| [[{slug}]] | memory | Added | daily/source.md | 2026-08-11 |\n")
    with (stage.root / "knowledge/log.md").open("a", encoding="utf-8") as handle:
        log_task = "file_answer" if relative.startswith("knowledge/qa/") else "compile"
        handle.write(f"## [now] {log_task} | source\n- Created: [[{slug}]]\n")

    if relative.startswith("knowledge/qa/"):
        validate_stage(
            stage,
            allowed_paths=(relative, "knowledge/index.md", "knowledge/log.md"),
            task="file_answer",
        )
    else:
        with pytest.raises(StageValidationError, match=missing):
            validate_stage(
                stage,
                allowed_paths=(relative, "knowledge/index.md", "knowledge/log.md"),
                task="compile",
            )


@pytest.mark.parametrize(
    ("relative", "body", "key"),
    [
        (
            "knowledge/concepts/bad.md",
            ARTICLE.replace('sources:\n  - "daily/2026-08-11.md"', "sources: daily/2026-08-11.md"),
            "sources",
        ),
        (
            "knowledge/connections/bad.md",
            ARTICLE.replace(
                'title: "Original"',
                'title: "Connection"\nconnects: concepts/original',
            ),
            "connects",
        ),
        (
            "knowledge/qa/bad.md",
            "---\ntitle: Q\nquestion: Why?\nconsulted: concepts/original\n"
            "filed: 2026-08-11\n---\n# Q\n",
            "consulted",
        ),
    ],
)
def test_frontmatter_list_fields_reject_scalar_values(memory_home, relative, body, key):
    stage = create_stage(memory_home, "job", f"scalar-{key}")
    _write(stage.root / relative, body)
    slug = relative.removeprefix("knowledge/").removesuffix(".md")
    with (stage.root / "knowledge/index.md").open("a", encoding="utf-8") as handle:
        handle.write(f"| [[{slug}]] | memory | Added | source | 2026-08-11 |\n")
    with (stage.root / "knowledge/log.md").open("a", encoding="utf-8") as handle:
        handle.write(f"## [now] compile | source\n- Created: [[{slug}]]\n")
    with pytest.raises(StageValidationError, match=key):
        validate_stage(
            stage,
            allowed_paths=(relative, "knowledge/index.md", "knowledge/log.md"),
            task="file_answer" if "/qa/" in relative else "compile",
        )


@pytest.mark.parametrize(("kind", "field", "bad_value"), _SCALAR_MATRIX)
def test_frontmatter_scalar_fields_reject_empty_list_and_map_values(
    memory_home, kind, field, bad_value
):
    relative, body, task = _schema_article(kind)
    original = _SCALAR_ORIGINALS[field][kind]
    body = body.replace(f"{field}: {original}", f"{field}: {bad_value}")
    stage = create_stage(memory_home, "job", f"{kind}-{field}-{bad_value}")
    _write(stage.root / relative, body)
    slug = relative.removeprefix("knowledge/").removesuffix(".md")
    with (stage.root / "knowledge/index.md").open("a", encoding="utf-8") as handle:
        handle.write(f"| [[{slug}]] | memory | Added | source | 2026-08-11 |\n")
    with (stage.root / "knowledge/log.md").open("a", encoding="utf-8") as handle:
        handle.write(f"## [now] {task} | source\n- Created: [[{slug}]]\n")
    with pytest.raises(StageValidationError, match=field):
        validate_stage(
            stage,
            allowed_paths=(relative, "knowledge/index.md", "knowledge/log.md"),
            task=task,
        )


@pytest.mark.parametrize("kind", ["concept", "connection"])
def test_project_accepts_nonempty_scalar_or_nonempty_list(memory_home, kind):
    relative, body, task = _schema_article(kind)
    body = body.replace("project: memory", "project: [memory, shared]")
    stage = create_stage(memory_home, "job", f"project-list-{kind}")
    _write(stage.root / relative, body)
    slug = relative.removeprefix("knowledge/").removesuffix(".md")
    with (stage.root / "knowledge/index.md").open("a", encoding="utf-8") as handle:
        handle.write(f"| [[{slug}]] | memory | Added | source | 2026-08-11 |\n")
    with (stage.root / "knowledge/log.md").open("a", encoding="utf-8") as handle:
        handle.write(f"## [now] {task} | source\n- Created: [[{slug}]]\n")
    validate_stage(
        stage,
        allowed_paths=(relative, "knowledge/index.md", "knowledge/log.md"),
        task=task,
    )


@pytest.mark.parametrize(("kind", "bad_value"), [("concept", '""'), ("connection", "{x: y}")])
def test_project_rejects_empty_or_mapping_values(memory_home, kind, bad_value):
    relative, body, task = _schema_article(kind)
    body = body.replace("project: memory", f"project: {bad_value}")
    stage = create_stage(memory_home, "job", f"bad-project-{kind}")
    _write(stage.root / relative, body)
    slug = relative.removeprefix("knowledge/").removesuffix(".md")
    with (stage.root / "knowledge/index.md").open("a", encoding="utf-8") as handle:
        handle.write(f"| [[{slug}]] | memory | Added | source | 2026-08-11 |\n")
    with (stage.root / "knowledge/log.md").open("a", encoding="utf-8") as handle:
        handle.write(f"## [now] {task} | source\n- Created: [[{slug}]]\n")
    with pytest.raises(StageValidationError, match="project"):
        validate_stage(
            stage,
            allowed_paths=(relative, "knowledge/index.md", "knowledge/log.md"),
            task=task,
        )


def test_unchanged_relevant_article_must_also_match_frontmatter_schema(memory_home):
    article = memory_home / "knowledge/concepts/original.md"
    article.write_text(ARTICLE.replace("updated: 2026-08-11\n", ""), encoding="utf-8")
    stage = create_stage(
        memory_home,
        "job",
        "unchanged-invalid",
        relevant_articles=("knowledge/concepts/original.md",),
    )
    with pytest.raises(StageValidationError, match="updated"):
        validate_stage(
            stage,
            allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
            task="compile",
        )


@pytest.mark.parametrize("missing", ["question", "consulted", "filed"])
def test_qa_frontmatter_requires_qa_schema(memory_home, missing):
    body = (
        "---\ntitle: Q\nquestion: Why?\nconsulted:\n"
        "  - concepts/original\nfiled: 2026-08-11\n---\n# Q\n"
    )
    if missing == "consulted":
        body = body.replace("consulted:\n  - concepts/original\n", "")
    else:
        body = body.replace(f"{missing}: {'Why?' if missing == 'question' else '2026-08-11'}\n", "")
    stage = create_stage(memory_home, "job", f"qa-{missing}")
    _write(stage.root / "knowledge/qa/bad.md", body)
    with (stage.root / "knowledge/index.md").open("a", encoding="utf-8") as handle:
        handle.write("| [[qa/bad]] | memory | Added | query | 2026-08-11 |\n")
    with (stage.root / "knowledge/log.md").open("a", encoding="utf-8") as handle:
        handle.write("## [now] query | source\n- Filed to: [[qa/bad]]\n")
    with pytest.raises(StageValidationError, match=missing):
        validate_stage(
            stage,
            allowed_paths=("knowledge/qa/*.md", "knowledge/index.md", "knowledge/log.md"),
            task="file_answer",
        )


def test_build_log_is_strictly_append_only(memory_home):
    stage = _compile_stage(memory_home)
    log_path = stage.root / "knowledge/log.md"
    log_path.write_text(
        "# Rewritten Build Log\n"
        "## [2026-08-11] compile | source\n"
        "- Articles updated: [[concepts/original]]\n",
        encoding="utf-8",
    )
    with pytest.raises(StageValidationError, match="append-only"):
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

    claude = create_fallback_stage(codex, attempt_id="claude")
    assert claude.root != codex.root
    assert not codex.root.exists()
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
        compiled_marker_baseline=capture_file_baseline(
            memory_home / "daily/2026-08-11.md"
        ),
        compiled_at="2026-08-11T12:00:00+00:00",
        state={"ingested": {"2026-08-11.md": {"hash": "new"}}},
        state_baseline=capture_file_baseline(memory_home / "scripts/state.json"),
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
            compiled_marker_baseline=capture_file_baseline(
                memory_home / "daily/2026-08-11.md"
            ),
            compiled_at="2026-08-11T12:00:00+00:00",
            state={"ingested": {"2026-08-11.md": {"hash": "new"}}},
            state_baseline=capture_file_baseline(memory_home / "scripts/state.json"),
        ),
    )

    assert 'title: "Updated"' in (memory_home / "knowledge/concepts/original.md").read_text(encoding="utf-8")
    assert "@compiled-through:2026-08-11T12:00:00+00:00" in (memory_home / "daily/2026-08-11.md").read_text(encoding="utf-8")
    assert json.loads((memory_home / "scripts/state.json").read_text(encoding="utf-8"))["ingested"]
    expected_hash = hashlib.sha256(
        (memory_home / "daily/2026-08-11.md").read_bytes()
    ).hexdigest()[:16]
    persisted_state = json.loads(
        (memory_home / "scripts/state.json").read_text(encoding="utf-8")
    )
    assert persisted_state["ingested"]["2026-08-11.md"]["hash"] == expected_hash
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


def test_daily_append_recovers_journal_before_appending(memory_home):
    from utils import append_daily_entry

    daily = memory_home / "daily/2026-08-11.md"
    original = daily.read_bytes()
    partial = b"partially applied"
    daily.write_bytes(partial)
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
                        "path": "daily/2026-08-11.md",
                        "original": original.hex(),
                        "replacement": partial.hex(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    journal.chmod(0o600)

    datetime_module = __import__("datetime")
    now = datetime_module.datetime(
        2026, 8, 11, 12, tzinfo=datetime_module.timezone.utc
    )
    append_daily_entry(memory_home, "capture after recovery", now=now)
    content = daily.read_bytes()
    assert content.startswith(original)
    assert b"capture after recovery" in content
    assert not journal.exists()


def test_state_compare_and_swap_rejects_concurrent_writer_without_data_loss(memory_home):
    baseline = capture_file_baseline(memory_home / "scripts/state.json")
    concurrent = {"ingested": {}, "concurrent_writer": True}
    (memory_home / "scripts/state.json").write_text(json.dumps(concurrent), encoding="utf-8")

    from staging import apply_host_bookkeeping

    with pytest.raises(RetryableApplyError, match="state baseline"):
        apply_host_bookkeeping(
            memory_home,
            ApplyBookkeeping(
                state={"ingested": {"stale": {}}},
                state_baseline=baseline,
            ),
        )
    persisted = json.loads(
        (memory_home / "scripts/state.json").read_text(encoding="utf-8")
    )
    assert persisted == concurrent


def test_compiled_marker_compare_and_swap_rejects_concurrent_daily_append(memory_home):
    from staging import apply_host_bookkeeping

    daily = memory_home / "daily/2026-08-11.md"
    daily_baseline = capture_file_baseline(daily)
    state = memory_home / "scripts/state.json"
    state_before = state.read_bytes()
    daily.write_text(
        daily.read_text(encoding="utf-8") + "\nconcurrent session\n",
        encoding="utf-8",
    )
    concurrent_bytes = daily.read_bytes()

    with pytest.raises(RetryableApplyError, match="compiled marker baseline"):
        apply_host_bookkeeping(
            memory_home,
            ApplyBookkeeping(
                compiled_marker_path="daily/2026-08-11.md",
                compiled_marker_baseline=daily_baseline,
                compiled_at="2026-08-11T12:00:00+00:00",
                state={
                    "ingested": {
                        "2026-08-11.md": {"hash": "must-be-derived-in-transaction"}
                    }
                },
                state_baseline=capture_file_baseline(state),
            ),
        )

    assert daily.read_bytes() == concurrent_bytes
    assert b"@compiled-through" not in concurrent_bytes
    assert state.read_bytes() == state_before


def test_state_cannot_bypass_compare_and_swap_through_stage_or_extra_updates(memory_home):
    stage = create_stage(memory_home, "job", "state-edit", include_state=True)
    (stage.root / "scripts/state.json").write_text('{"model": "edit"}', encoding="utf-8")
    with pytest.raises(StageValidationError, match="state.json"):
        validate_stage(
            stage,
            allowed_paths=("scripts/state.json",),
            task="compile",
        )

    from staging import apply_host_bookkeeping

    with pytest.raises(StageValidationError, match="state"):
        apply_host_bookkeeping(
            memory_home,
            ApplyBookkeeping(extra_updates={"scripts/state.json": b"{}"}),
        )


def test_reconcile_recovers_before_reading_state_or_build_log(monkeypatch):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "reconcile-state.py"
    spec = importlib.util.spec_from_file_location("reconcile_state_order_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    events = []
    monkeypatch.setattr(module, "recover_incomplete_apply", lambda _home: events.append("recover"))
    monkeypatch.setattr(
        module,
        "load_state_with_baseline",
        lambda: events.append("state") or ({"ingested": {}}, object()),
    )
    monkeypatch.setattr(
        module,
        "files_mentioned_in_log_md",
        lambda: events.append("log") or set(),
    )
    monkeypatch.setattr("sys.argv", [str(path)])

    module.main()
    assert events == ["recover", "state", "log"]


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


@pytest.mark.parametrize("attack", ["content", "symlink", "hardlink"])
def test_apply_reads_exact_validated_stage_bytes_without_toctou(
    memory_home, tmp_path, monkeypatch, attack
):
    import staging

    stage = _compile_stage(memory_home, attempt_id=f"toctou-{attack}")
    validated = validate_stage(
        stage,
        allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
        task="compile",
    )
    target = stage.root / "knowledge/concepts/original.md"
    outside = tmp_path / f"outside-{attack}.md"
    outside.write_text("attacker bytes", encoding="utf-8")
    original_reader = getattr(staging, "_read_safe_regular", lambda _root, rel: (stage.root / rel).read_bytes())
    attacked = False

    def mutate_after_snapshot(root, relative):
        nonlocal attacked
        if relative == "knowledge/concepts/original.md" and not attacked:
            attacked = True
            if attack == "content":
                target.write_text("changed after snapshot", encoding="utf-8")
            else:
                target.unlink()
                if attack == "symlink":
                    target.symlink_to(outside)
                else:
                    target.hardlink_to(outside)
        return original_reader(root, relative)

    monkeypatch.setattr(staging, "_read_safe_regular", mutate_after_snapshot, raising=False)
    real_before = (memory_home / "knowledge/concepts/original.md").read_bytes()
    with pytest.raises(StageValidationError, match="stage|symlink|hard-linked"):
        apply_validated_stage(validated, stage.baseline, ApplyBookkeeping())
    assert (memory_home / "knowledge/concepts/original.md").read_bytes() == real_before
    assert outside.read_text(encoding="utf-8") == "attacker bytes"


class SimulatedCrash(BaseException):
    pass


@pytest.mark.parametrize("transition", ["prepared", "applying", "complete"])
def test_journal_transition_crash_leaves_recoverable_atomic_record(
    memory_home, transition
):
    from staging import apply_host_bookkeeping

    state_path = memory_home / "scripts/state.json"
    original = state_path.read_bytes()

    def crash(phase):
        if phase == transition:
            raise SimulatedCrash(phase)

    with pytest.raises(SimulatedCrash):
        apply_host_bookkeeping(
            memory_home,
            ApplyBookkeeping(
                state={"ingested": {}, "transition": transition},
                state_baseline=capture_file_baseline(state_path),
                journal_transition_injector=crash,
            ),
        )

    journals = list((memory_home / "scripts/memory-apply-journal").glob("*.json"))
    if journals:
        payload = json.loads(journals[0].read_text(encoding="utf-8"))
        assert payload["state"] in {"prepared", "applying", "complete"}
    recover_incomplete_apply(memory_home)
    expected_new = (json.dumps({"ingested": {}, "transition": transition}, indent=2) + "\n").encode()
    assert state_path.read_bytes() in {original, expected_new}
    assert list((memory_home / "scripts/memory-apply-journal").glob("*")) == []


def test_recovery_removes_orphan_atomic_journal_temps(memory_home):
    directory = memory_home / "scripts/memory-apply-journal"
    directory.mkdir(mode=0o700)
    orphan = directory / ".apply-dead.json.partial.tmp"
    orphan.write_bytes(b"torn")
    orphan.chmod(0o600)

    assert recover_incomplete_apply(memory_home) is False
    assert not orphan.exists()


def test_failed_atomic_journal_publish_retains_prior_valid_record(memory_home, monkeypatch):
    import staging

    directory = memory_home / "scripts/memory-apply-journal"
    directory.mkdir(mode=0o700)
    journal = directory / "apply-test.json"
    prior = b'{"state":"prepared","version":1}\n'
    journal.write_bytes(prior)
    journal.chmod(0o600)
    real_replace = staging.os.replace

    def fail_journal_replace(source, destination):
        if Path(destination) == journal:
            raise OSError("simulated power loss before publish")
        return real_replace(source, destination)

    monkeypatch.setattr(staging.os, "replace", fail_journal_replace)
    with pytest.raises(OSError, match="power loss"):
        staging._atomic_journal_transition(journal, b'{"state":"applying"}\n')
    assert journal.read_bytes() == prior
    assert list(directory.glob(".*.tmp")) == []


def test_compile_without_article_requires_valid_append_only_build_log(memory_home):
    stage = create_stage(memory_home, "job", "no-article")
    with pytest.raises(StageValidationError, match="build log"):
        validate_stage(
            stage,
            allowed_paths=("knowledge/index.md", "knowledge/log.md"),
            task="compile",
        )

    with (stage.root / "knowledge/log.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "## [2026-08-11T12:00:00+00:00] compile | 2026-08-11.md\n"
            "- Articles created: (none)\n"
        )
    validate_stage(
        stage,
        allowed_paths=("knowledge/index.md", "knowledge/log.md"),
        task="compile",
    )


@pytest.mark.parametrize("corrupt", ["index", "log"])
def test_validation_rejects_total_index_or_log_corruption_without_article_changes(
    memory_home, corrupt
):
    stage = create_stage(memory_home, "job", f"corrupt-{corrupt}")
    (stage.root / "knowledge/index.md").write_text(
        "totally corrupt\n" if corrupt == "index" else stage.root.joinpath("knowledge/index.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (stage.root / "knowledge/log.md").write_text(
        "rewritten history\n"
        if corrupt == "log"
        else stage.root.joinpath("knowledge/log.md").read_text(encoding="utf-8")
        + "## [now] compile | source.md\n- Articles created: (none)\n",
        encoding="utf-8",
    )
    with pytest.raises(StageValidationError, match=corrupt):
        validate_stage(
            stage,
            allowed_paths=("knowledge/index.md", "knowledge/log.md"),
            task="compile",
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        '"unterminated',
        "'unterminated",
        "[one, 'unterminated]",
        "[one,,two]",
        "[one, two",
        "null",
        "~",
        "&anchor value",
        "*alias",
        "!tag value",
        "{nested: value}",
        "plain: mapping",
    ],
)
def test_frontmatter_safe_yaml_subset_rejects_ambiguous_constructs(
    memory_home, bad_value
):
    relative, body, task = _schema_article("concept")
    body = body.replace('title: "Original"', f"title: {bad_value}")
    stage = create_stage(memory_home, "job", f"yaml-{hashlib.sha256(bad_value.encode()).hexdigest()[:8]}")
    _write(stage.root / relative, body)
    slug = relative.removeprefix("knowledge/").removesuffix(".md")
    with (stage.root / "knowledge/index.md").open("a", encoding="utf-8") as handle:
        handle.write(f"| [[{slug}]] | memory | Added | source | 2026-08-11 |\n")
    with (stage.root / "knowledge/log.md").open("a", encoding="utf-8") as handle:
        handle.write(f"## [now] {task} | source\n- Created: [[{slug}]]\n")
    with pytest.raises(StageValidationError, match="frontmatter|title"):
        validate_stage(
            stage,
            allowed_paths=(relative, "knowledge/index.md", "knowledge/log.md"),
            task=task,
        )


def test_create_stage_rejects_scripts_ancestor_symlink_without_external_write(tmp_path):
    home = tmp_path / "memory"
    external = tmp_path / "external"
    home.mkdir()
    external.mkdir()
    _write(home / "AGENTS.md", "# schema\n")
    _write(home / "knowledge/index.md", "# Index\n")
    _write(home / "knowledge/log.md", "# Build Log\n")
    (home / "scripts").symlink_to(external, target_is_directory=True)

    with pytest.raises(StageValidationError, match="symlink"):
        create_stage(home, "job", "attempt")
    assert list(external.iterdir()) == []


def test_apply_rejects_stage_and_bookkeeping_destination_collision(memory_home):
    stage = _compile_stage(memory_home, attempt_id="collision")
    validated = validate_stage(
        stage,
        allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
        task="compile",
    )
    relative = "knowledge/concepts/original.md"
    with pytest.raises(StageValidationError, match="collision"):
        apply_validated_stage(
            validated,
            stage.baseline,
            ApplyBookkeeping(
                extra_updates={relative: b"host overwrite"},
                input_baselines={relative: capture_file_baseline(memory_home / relative)},
            ),
        )


@pytest.mark.parametrize("exists", [False, True])
def test_extra_updates_require_exact_destination_baseline(memory_home, exists):
    from staging import apply_host_bookkeeping

    relative = "scripts/provider-usage.json"
    destination = memory_home / relative
    if exists:
        destination.write_bytes(b"owner-current")
    with pytest.raises(StageValidationError, match="baseline"):
        apply_host_bookkeeping(
            memory_home,
            ApplyBookkeeping(extra_updates={relative: b"replacement"}),
        )

    baseline = capture_file_baseline(destination)
    if exists:
        destination.write_bytes(b"concurrent")
    else:
        destination.write_bytes(b"created concurrently")
    with pytest.raises(RetryableApplyError, match="baseline"):
        apply_host_bookkeeping(
            memory_home,
            ApplyBookkeeping(
                extra_updates={relative: b"replacement"},
                input_baselines={relative: baseline},
            ),
        )


def _capture_job() -> SimpleNamespace:
    return SimpleNamespace(
        id=7,
        kind="capture",
        source_agent="codex",
        session_id="private-session-id",
        source_hash="source-hash",
        project="memory",
        cwd="/tmp/memory",
    )


def test_worker_daily_capture_identity_makes_retry_exactly_once(tmp_path, monkeypatch):
    from worker import daily_writer_boundary

    monkeypatch.setenv("AI_MEMORY_HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    job = _capture_job()
    daily_writer_boundary(job, "persist once")
    daily_writer_boundary(job, "persist once")

    content = next((tmp_path / "daily").glob("*.md")).read_text(encoding="utf-8")
    assert content.count("persist once") == 1
    assert content.count("@capture-id:") == 1
    marker = content.split("@capture-id:", 1)[1].split(" ", 1)[0]
    assert len(marker) == 64
    assert job.session_id not in marker


def _successful_router(text):
    from providers import ProviderResult, RoutedResult, TaskKind

    result = ProviderResult("codex", "luna", TaskKind.EXTRACT, "success", text=text)

    class Router:
        async def generate_text(self, _request):
            return RoutedResult.from_result(result, (result,), None)

    return Router()


def test_worker_crash_after_append_before_queue_complete_retries_without_duplicate(
    tmp_path, monkeypatch
):
    from worker import MemoryWorker

    monkeypatch.setenv("AI_MEMORY_HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    job = _capture_job()
    job.payload = {"rendered_context": "capture"}
    job.attempt_count = 1
    job.status = "leased"
    job.lease_owner = "worker"

    class Queue:
        completes = 0

        def renew(self, *_args):
            return True

        def record_attempt(self, *_args):
            return None

        def complete(self, *_args):
            self.completes += 1
            if self.completes == 1:
                raise RuntimeError("crash after append")

        def get_job(self, _job_id):
            return job

        def retry(self, *_args):
            return None

    queue = Queue()
    worker = MemoryWorker(queue, _successful_router("crash-gap"), owner="worker")
    asyncio.run(worker.process(job))
    asyncio.run(worker.process(job))

    content = next((tmp_path / "daily").glob("*.md")).read_text(encoding="utf-8")
    assert queue.completes == 2
    assert content.count("crash-gap") == 1


def test_lease_loss_after_append_allows_thread_to_finish_without_duplicate(
    tmp_path, monkeypatch
):
    from worker import MemoryWorker, daily_writer_boundary

    monkeypatch.setenv("AI_MEMORY_HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    job = _capture_job()
    job.payload = {"rendered_context": "capture"}
    job.attempt_count = 1
    job.status = "leased"
    job.lease_owner = "worker"
    appended = threading.Event()
    renewals = iter((True, False))

    def append_then_pause(current_job, text):
        daily_writer_boundary(current_job, text)
        appended.set()
        time.sleep(0.05)

    class Queue:
        completed = False

        def renew(self, *_args):
            return next(renewals)

        def record_attempt(self, *_args):
            return None

        def complete(self, *_args):
            self.completed = True

        def get_job(self, _job_id):
            return job

        def retry(self, *_args):
            return None

    async def wait_for_append(_seconds):
        while not appended.is_set():
            await asyncio.sleep(0)

    queue = Queue()
    worker = MemoryWorker(
        queue,
        _successful_router("lease-gap"),
        daily_writer=append_then_pause,
        owner="worker",
        heartbeat_sleeper=wait_for_append,
    )
    asyncio.run(worker.process(job))
    daily_writer_boundary(job, "lease-gap")

    content = next((tmp_path / "daily").glob("*.md")).read_text(encoding="utf-8")
    assert queue.completed is False
    assert content.count("lease-gap") == 1


def test_sync_daily_writer_does_not_block_heartbeat_event_loop():
    from worker import MemoryWorker

    events = []

    def blocking_writer(_job, _text):
        time.sleep(0.05)
        events.append("writer-finished")

    worker = MemoryWorker(
        SimpleNamespace(),
        router=SimpleNamespace(),
        daily_writer=blocking_writer,
    )

    async def exercise():
        async def heartbeat_tick():
            await asyncio.sleep(0.005)
            events.append("heartbeat")

        await asyncio.gather(worker._write(_capture_job(), "text"), heartbeat_tick())

    asyncio.run(exercise())
    assert events == ["heartbeat", "writer-finished"]


def test_cancel_after_daily_append_then_retry_is_exactly_once(tmp_path, monkeypatch):
    from worker import MemoryWorker, daily_writer_boundary

    monkeypatch.setenv("AI_MEMORY_HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    appended = threading.Event()
    release = threading.Event()
    job = _capture_job()

    def append_then_block(current_job, text):
        daily_writer_boundary(current_job, text)
        appended.set()
        release.wait(timeout=1)

    worker = MemoryWorker(SimpleNamespace(), router=SimpleNamespace(), daily_writer=append_then_block)

    async def cancel_and_retry():
        first = asyncio.create_task(worker._write(job, "cancel-safe"))
        assert await asyncio.to_thread(appended.wait, 0.5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        await worker._write(job, "cancel-safe")

    asyncio.run(cancel_and_retry())
    content = next((tmp_path / "daily").glob("*.md")).read_text(encoding="utf-8")
    assert content.count("cancel-safe") == 1


def test_reconcile_dry_run_detects_journal_without_mutation(tmp_path, monkeypatch):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "reconcile-state.py"
    spec = importlib.util.spec_from_file_location("reconcile_state_dry_run_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    home = tmp_path / "memory"
    journal_dir = home / "scripts/memory-apply-journal"
    journal_dir.mkdir(parents=True)
    journal = journal_dir / "crash.json"
    journal.write_bytes(b'{"incomplete":true}\n')
    before = journal.read_bytes()
    monkeypatch.setattr(module, "DAILY_DIR", home / "daily")
    monkeypatch.setattr(
        module,
        "load_state_with_baseline",
        lambda: (_ for _ in ()).throw(AssertionError("must not read state")),
    )
    monkeypatch.setattr("sys.argv", [str(path), "--dry-run"])

    with pytest.raises(SystemExit):
        module.main()
    assert journal.read_bytes() == before
    assert not (home / "scripts/memory-writer.lock").exists()


def test_atomic_replace_preserves_existing_mode(memory_home):
    stage = _compile_stage(memory_home, attempt_id="mode")
    validated = validate_stage(
        stage,
        allowed_paths=("knowledge/concepts/*.md", "knowledge/index.md", "knowledge/log.md"),
        task="compile",
    )
    article = memory_home / "knowledge/concepts/original.md"
    article.chmod(0o640)
    apply_validated_stage(validated, stage.baseline, ApplyBookkeeping())
    assert stat.S_IMODE(article.stat().st_mode) == 0o640


def test_exclusive_file_lock_rejects_nested_acquire(tmp_path):
    lock = ExclusiveFileLock(tmp_path / "writer.lock")
    assert lock.acquire() is True
    try:
        with pytest.raises(RuntimeError, match="already acquired"):
            lock.acquire()
    finally:
        lock.release()


def test_create_stage_omits_state_by_default(memory_home):
    stage = create_stage(memory_home, "job", "no-state")
    assert "scripts/state.json" not in stage.baseline
    assert not (stage.root / "scripts/state.json").exists()


def test_new_daily_file_fsyncs_daily_directory_and_root(tmp_path, monkeypatch):
    import utils

    synced = []
    monkeypatch.setattr(utils, "_fsync_directory", lambda path: synced.append(Path(path)))
    timestamp = __import__("datetime").datetime(
        2026, 8, 11, 12, tzinfo=__import__("datetime").timezone.utc
    )
    utils.append_daily_entry(tmp_path, "durable", now=timestamp)
    assert synced == [tmp_path / "daily", tmp_path]
