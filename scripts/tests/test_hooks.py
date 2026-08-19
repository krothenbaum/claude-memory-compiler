"""Subprocess contract tests for the fast Claude and Codex hook adapters."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts.transcripts import parse_claude_transcript


ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / "hooks"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "transcripts"
FIXED_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
CODEX_0146_HELP = """\
Options:
      --dangerously-bypass-approvals-and-sandbox
          Skip all confirmation prompts and execute commands without sandboxing. EXTREMELY
          DANGEROUS. Intended solely for externally sandboxed environments.

      --dangerously-bypass-hook-trust
          Run enabled hooks without requiring persisted hook trust for this invocation. DANGEROUS.
          Intended only for automation that already vets hook sources.

  -C, --cd <DIR>
          Tell the agent which working root to use.
"""

HOOK_LOGGERS = [
    ("session-end.py", "ai-memory-session-end", "session-end"),
    ("pre-compact.py", "ai-memory-pre-compact", "pre-compact"),
    ("codex-session-end.py", "ai-memory-codex-session-end", "codex-session-end"),
]


def _close_hook_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _hook_log_records(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines
    return [json.loads(line) for line in lines]


def _assert_diagnostic_only(memory_home: Path, *, count: int = 1) -> None:
    connection = sqlite3.connect(memory_home / "scripts" / "jobs.sqlite3")
    try:
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM status_runs").fetchone()[0] == count
        assert connection.execute("SELECT count(*) FROM status_events").fetchone()[0] == count
        assert connection.execute(
            "SELECT count(*) FROM status_runs WHERE state = 'failed' AND phase = 'failed'"
        ).fetchone()[0] == count
    finally:
        connection.close()


def _tree_manifest(root: Path) -> dict[str, tuple[str, int, int, bytes | str | None]]:
    manifest: dict[str, tuple[str, int, int, bytes | str | None]] = {}
    for path in (root, *sorted(root.rglob("*"))):
        metadata = path.lstat()
        if stat.S_ISREG(metadata.st_mode):
            kind = "file"
            content: bytes | str | None = path.read_bytes()
        elif stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            content = None
        elif stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
            content = os.readlink(path)
        else:
            kind = "special"
            content = None
        manifest[str(path.relative_to(root))] = (
            kind,
            stat.S_IMODE(metadata.st_mode),
            metadata.st_mtime_ns,
            content,
        )
    return manifest


def _run_global_setup(home: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"HOME": str(home), "ZDOTDIR": str(home), "SHELL": "/bin/zsh"})
    return subprocess.run(
        ["bash", str(ROOT / "bin" / "setup-global.sh")],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )


def _codex_help_option_block(help_text: str, option: str) -> str:
    lines = help_text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != option:
            continue

        option_indent = len(line) - len(line.lstrip())
        block = [line]
        for following in lines[index + 1 :]:
            stripped = following.strip()
            indent = len(following) - len(following.lstrip())
            if stripped and indent <= option_indent:
                break
            block.append(following)
        return "\n".join(block)

    raise AssertionError(f"missing Codex help option: {option}")


def _assert_codex_hook_trust_help(help_text: str) -> None:
    block = _codex_help_option_block(
        help_text,
        "--dangerously-bypass-hook-trust",
    )
    assert "DANGEROUS" in block
    assert "automation" in block.lower()


def test_tree_manifest_detects_metadata_only_changes(tmp_path):
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"unchanged")
    sentinel.chmod(0o640)
    before = _tree_manifest(tmp_path)

    sentinel.chmod(0o600)

    assert _tree_manifest(tmp_path) != before


def _fake_uv(bin_dir: Path) -> Path:
    """Install a worker-launch stand-in that records the queued agent."""
    bin_dir.mkdir()
    executable = bin_dir / "uv"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sqlite3, sys\n"
        "root = pathlib.Path(sys.argv[sys.argv.index('--directory') + 1])\n"
        "database = root / 'scripts' / 'jobs.sqlite3'\n"
        "with sqlite3.connect(database) as connection:\n"
        "    agent = connection.execute(\n"
        "        'SELECT source_agent FROM jobs ORDER BY id DESC LIMIT 1'\n"
        "    ).fetchone()[0]\n"
        "label = {'claude': 'Claude Code', 'codex': 'Codex'}[agent]\n"
        "daily = root / 'daily'\n"
        "daily.mkdir(parents=True, exist_ok=True)\n"
        "(daily / 'fake-worker.md').write_text(f'**Agent:** {label}\\n')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _hook_env(
    memory_home: Path,
    *,
    fake_bin: Path | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["AI_MEMORY_HOME"] = str(memory_home)
    env.pop("CLAUDE_MEMORY_HOME", None)
    env.pop("CLAUDE_INVOKED_BY", None)
    env.pop("AI_MEMORY_INTERNAL_JOB", None)
    if fake_bin is not None:
        env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    if extra:
        env.update(extra)
    return env


def _run_hook(
    name: str,
    payload: dict[str, object],
    memory_home: Path,
    *,
    fake_bin: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(HOOKS / name)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=3,
        env=_hook_env(memory_home, fake_bin=fake_bin, extra=extra_env),
        check=False,
    )
    return result, time.monotonic() - started


def _job_rows(memory_home: Path) -> list[tuple[str, str, str]]:
    database = memory_home / "scripts" / "jobs.sqlite3"
    with sqlite3.connect(database) as connection:
        return connection.execute(
            "SELECT source_agent, session_id, trigger FROM jobs ORDER BY id"
        ).fetchall()


def _only_job_source_path(memory_home: Path) -> Path:
    database = memory_home / "scripts" / "jobs.sqlite3"
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT source_path FROM jobs").fetchall()
    assert len(rows) == 1
    return Path(rows[0][0])


def _wait_for(path: Path, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def test_codex_session_end_returns_under_three_seconds_and_enqueues(tmp_path):
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    _fake_uv(fake_bin)

    result, elapsed = _run_hook(
        "codex-session-end.py",
        {
            "hook_event_name": "SessionEnd",
            "session_id": "codex-hook-session",
            "transcript_path": str(FIXTURES / "codex-basic.jsonl"),
            "cwd": "/projects/codex-project",
            "reason": "user_exit",
            "source": "cli",
        },
        memory_home,
        fake_bin=fake_bin,
    )

    assert result.returncode == 0, result.stderr
    assert elapsed < 3
    assert result.stdout == ""
    assert _job_rows(memory_home) == [
        ("codex", "codex-hook-session", "session_end")
    ]
    fake_daily = memory_home / "daily" / "fake-worker.md"
    _wait_for(fake_daily)
    assert fake_daily.read_text(encoding="utf-8") == "**Agent:** Codex\n"


def test_claude_session_end_enqueues_without_model_call(tmp_path):
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    _fake_uv(fake_bin)

    result, elapsed = _run_hook(
        "session-end.py",
        {
            "session_id": "claude-hook-session",
            "transcript_path": str(FIXTURES / "claude-basic.jsonl"),
            "cwd": "/projects/claude-project",
            "source": "terminal",
        },
        memory_home,
        fake_bin=fake_bin,
    )

    assert result.returncode == 0, result.stderr
    assert elapsed < 3
    assert _job_rows(memory_home) == [
        ("claude", "claude-hook-session", "session_end")
    ]
    _wait_for(memory_home / "daily" / "fake-worker.md")
    assert "**Agent:** Claude Code" in (
        memory_home / "daily" / "fake-worker.md"
    ).read_text(encoding="utf-8")
    assert not list((memory_home / "scripts").glob("session-flush-*.md"))


def test_precompact_then_session_end_deduplicates_normalized_slice(tmp_path):
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    _fake_uv(fake_bin)
    payload = {
        "session_id": "same-session",
        "transcript_path": str(FIXTURES / "claude-decisions.jsonl"),
        "cwd": "/projects/shared-project",
    }

    precompact, _ = _run_hook(
        "pre-compact.py", payload, memory_home, fake_bin=fake_bin
    )
    session_end, _ = _run_hook(
        "session-end.py", payload, memory_home, fake_bin=fake_bin
    )

    assert precompact.returncode == session_end.returncode == 0
    rows = _job_rows(memory_home)
    assert len(rows) == 1
    assert rows[0][:2] == ("claude", "same-session")


def test_internal_job_guard_creates_no_jobs_or_spool_files(tmp_path):
    memory_home = tmp_path / "must-not-exist"
    result, elapsed = _run_hook(
        "session-end.py",
        {"transcript_path": str(FIXTURES / "claude-basic.jsonl")},
        memory_home,
        extra_env={"AI_MEMORY_INTERNAL_JOB": "1"},
    )

    assert result.returncode == 0
    assert elapsed < 3
    assert not memory_home.exists()


def test_internal_codex_job_creates_no_queue_or_session_rollout(tmp_path):
    memory_home = tmp_path / "must-not-exist"
    codex_home = tmp_path / "codex-must-not-exist"
    result, elapsed = _run_hook(
        "codex-session-end.py",
        {"transcript_path": str(FIXTURES / "codex-basic.jsonl")},
        memory_home,
        extra_env={
            "AI_MEMORY_INTERNAL_JOB": "1",
            "CODEX_HOME": str(codex_home),
        },
    )

    assert result.returncode == 0
    assert elapsed < 3
    assert not memory_home.exists()
    assert not codex_home.exists()


def test_missing_transcript_fails_closed_without_blocking_host(tmp_path):
    memory_home = tmp_path / "memory"
    for hook_name in ("session-end.py", "pre-compact.py", "codex-session-end.py"):
        result, elapsed = _run_hook(
            hook_name,
            {"session_id": "missing", "transcript_path": ""},
            memory_home,
        )
        assert result.returncode == 0, result.stderr
        assert elapsed < 3
    database = memory_home / "scripts" / "jobs.sqlite3"
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM status_runs").fetchone()[0] == 3
        assert connection.execute("SELECT count(*) FROM status_events").fetchone()[0] == 3
        assert connection.execute(
            "SELECT count(*) FROM status_runs WHERE state = 'failed' AND phase = 'failed'"
        ).fetchone()[0] == 3
    finally:
        connection.close()
    assert not (memory_home / "scripts" / "spool").exists()


def _write_large_valid_transcript(path: Path, agent: str) -> None:
    padding = "x" * 60_000
    with path.open("w", encoding="utf-8") as stream:
        if agent == "claude":
            stream.write(
                json.dumps(
                    {
                        "sessionId": "large-claude-session",
                        "cwd": "/projects/large-claude",
                    }
                )
                + "\n"
            )
            routine = {"type": "progress", "data": padding}
            final = {
                "message": {"role": "user", "content": "CLAUDE_TAIL_SIGNAL"}
            }
        else:
            stream.write(
                json.dumps(
                    {
                        "timestamp": "2026-08-11T10:15:00Z",
                        "type": "session_meta",
                        "payload": {
                            "id": "large-codex-session",
                            "cwd": "/projects/large-codex",
                        },
                    }
                )
                + "\n"
            )
            routine = {
                "type": "event_msg",
                "payload": {"type": "agent_reasoning", "text": padding},
            }
            final = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "CODEX_TAIL_SIGNAL"}
                    ],
                },
            }
        encoded_routine = json.dumps(routine) + "\n"
        for _ in range(220):
            stream.write(encoded_routine)
        stream.write(json.dumps(final) + "\n")


def test_large_valid_live_transcripts_use_a_bounded_private_tail(tmp_path):
    for agent, hook_name in (
        ("claude", "session-end.py"),
        ("codex", "codex-session-end.py"),
    ):
        case = tmp_path / agent
        case.mkdir()
        source = case / f"raw-private-session-name-{agent}.jsonl"
        _write_large_valid_transcript(source, agent)
        assert source.stat().st_size > 12_000_000
        memory_home = case / "memory"
        fake_bin = case / "bin"
        hook_tmp = case / "hook-tmp"
        hook_tmp.mkdir()
        _fake_uv(fake_bin)

        payload = {"transcript_path": str(source)}
        if agent == "claude":
            payload.update(
                {
                    "session_id": "large-claude-session",
                    "cwd": "/projects/large-claude",
                }
            )
        result, elapsed = _run_hook(
            hook_name,
            payload,
            memory_home,
            fake_bin=fake_bin,
            extra_env={"TMPDIR": str(hook_tmp)},
        )

        assert result.returncode == 0, result.stderr
        assert elapsed < 3
        snapshot = _only_job_source_path(memory_home)
        assert snapshot.stat().st_size <= 16_100_000
        assert stat.S_IMODE(snapshot.stat().st_mode) == 0o600
        assert "raw-private-session-name" not in snapshot.name
        snapshot_lines = snapshot.read_text(encoding="utf-8").splitlines()
        assert f"{agent.upper()}_TAIL_SIGNAL" in snapshot_lines[-1]
        assert all(isinstance(json.loads(line), dict) for line in snapshot_lines)
        assert list(hook_tmp.iterdir()) == []
        row = _job_rows(memory_home)[0]
        assert row[1] == f"large-{agent}-session"


def test_oversized_final_jsonl_record_fails_closed_without_partial_job(tmp_path):
    source = tmp_path / "raw-secret-name.jsonl"
    source.write_text(
        json.dumps(
            {"message": {"role": "user", "content": "x" * 2_000_000}}
        ),
        encoding="utf-8",
    )
    memory_home = tmp_path / "memory"
    hook_tmp = tmp_path / "hook-tmp"
    hook_tmp.mkdir()

    result, elapsed = _run_hook(
        "session-end.py",
        {
            "session_id": "oversized",
            "transcript_path": str(source),
            "cwd": "/projects/oversized",
        },
        memory_home,
        extra_env={"TMPDIR": str(hook_tmp)},
    )

    assert result.returncode == 0
    assert elapsed < 3
    _assert_diagnostic_only(memory_home)
    assert not (memory_home / "scripts" / "spool").exists()
    assert list(hook_tmp.iterdir()) == []


def test_oversized_record_before_valid_signal_fails_closed_without_deadline_spin(
    tmp_path,
):
    source = tmp_path / "oversized-before-signal.jsonl"
    source.write_text(
        json.dumps({"type": "progress", "data": "x" * 1_100_000})
        + "\n"
        + json.dumps(
            {"message": {"role": "user", "content": "MUST_NOT_ENQUEUE"}}
        )
        + "\n",
        encoding="utf-8",
    )
    memory_home = tmp_path / "memory"

    result, elapsed = _run_hook(
        "session-end.py",
        {"transcript_path": str(source)},
        memory_home,
    )

    assert result.returncode == 0
    assert elapsed < 1.0
    _assert_diagnostic_only(memory_home)
    assert not (memory_home / "scripts" / "spool").exists()


def _write_signal_before_routine_tail(
    path: Path,
    *,
    agent: str,
    routine_records: int,
) -> None:
    padding = "r" * 60_000
    with path.open("w", encoding="utf-8") as stream:
        if agent == "claude":
            stream.write(
                json.dumps(
                    {
                        "sessionId": "semantic-claude-session",
                        "cwd": "/projects/semantic-claude",
                    }
                )
                + "\n"
            )
            for index in range(5):
                stream.write(
                    json.dumps(
                        {
                            "message": {
                                "role": "user" if index % 2 == 0 else "assistant",
                                "content": f"LAST_DURABLE_SIGNAL_{index}",
                            }
                        }
                    )
                    + "\n"
                )
            routine = {"type": "progress", "data": padding}
        else:
            stream.write(
                json.dumps(
                    {
                        "timestamp": "2026-08-11T10:15:00Z",
                        "type": "session_meta",
                        "payload": {
                            "id": "semantic-codex-session",
                            "cwd": "/projects/semantic-codex",
                        },
                    }
                )
                + "\n"
            )
            stream.write(
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "LAST_DURABLE_SIGNAL_CODEX",
                                }
                            ],
                        },
                    }
                )
                + "\n"
            )
            routine = {
                "type": "event_msg",
                "payload": {"type": "agent_reasoning", "text": padding},
            }
        encoded = json.dumps(routine) + "\n"
        for _ in range(routine_records):
            stream.write(encoded)


def _only_job_payload(memory_home: Path) -> dict[str, object]:
    database = memory_home / "scripts" / "jobs.sqlite3"
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT payload_json FROM jobs").fetchall()
    assert len(rows) == 1
    return json.loads(rows[0][0])


def test_claude_semantic_tail_expansion_preserves_signal_and_dedup(tmp_path):
    source = tmp_path / "claude-semantic.jsonl"
    _write_signal_before_routine_tail(
        source, agent="claude", routine_records=65
    )
    assert source.stat().st_size > 3_700_000
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    hook_tmp = tmp_path / "hook-tmp"
    hook_tmp.mkdir()
    _fake_uv(fake_bin)
    payload = {
        "session_id": "semantic-claude-session",
        "transcript_path": str(source),
        "cwd": "/projects/semantic-claude",
    }

    precompact, precompact_elapsed = _run_hook(
        "pre-compact.py",
        payload,
        memory_home,
        fake_bin=fake_bin,
        extra_env={"TMPDIR": str(hook_tmp)},
    )
    session_end, session_end_elapsed = _run_hook(
        "session-end.py",
        payload,
        memory_home,
        fake_bin=fake_bin,
        extra_env={"TMPDIR": str(hook_tmp)},
    )

    assert precompact.returncode == session_end.returncode == 0
    assert precompact_elapsed < 3
    assert session_end_elapsed < 3
    assert len(_job_rows(memory_home)) == 1
    rendered = _only_job_payload(memory_home)["rendered_context"]
    assert "LAST_DURABLE_SIGNAL_0" in rendered
    assert "LAST_DURABLE_SIGNAL_4" in rendered
    snapshot = _only_job_source_path(memory_home)
    assert all(
        isinstance(json.loads(line), dict)
        for line in snapshot.read_text(encoding="utf-8").splitlines()
    )
    assert list(hook_tmp.iterdir()) == []


def test_semantic_expansion_preserves_four_older_turns_before_routine_tail(
    tmp_path,
):
    source = tmp_path / "claude-threshold-edge.jsonl"
    padding = "t" * 60_000
    with source.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "sessionId": "threshold-session",
                    "cwd": "/projects/threshold-project",
                }
            )
            + "\n"
        )
        for index in range(4):
            stream.write(
                json.dumps(
                    {
                        "message": {
                            "role": "user" if index % 2 == 0 else "assistant",
                            "content": f"OLDER_DURABLE_TURN_{index}",
                        }
                    }
                )
                + "\n"
            )
        routine = json.dumps({"type": "progress", "data": padding}) + "\n"
        for _ in range(65):
            stream.write(routine)
        stream.write(
            json.dumps(
                {"message": {"role": "user", "content": "FINAL_DURABLE_TURN"}}
            )
            + "\n"
        )
    assert source.stat().st_size > 3_700_000
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    hook_tmp = tmp_path / "hook-tmp"
    hook_tmp.mkdir()
    _fake_uv(fake_bin)
    payload = {
        "session_id": "threshold-session",
        "transcript_path": str(source),
        "cwd": "/projects/threshold-project",
    }

    precompact, precompact_elapsed = _run_hook(
        "pre-compact.py",
        payload,
        memory_home,
        fake_bin=fake_bin,
        extra_env={"TMPDIR": str(hook_tmp)},
    )
    session_end, session_end_elapsed = _run_hook(
        "session-end.py",
        payload,
        memory_home,
        fake_bin=fake_bin,
        extra_env={"TMPDIR": str(hook_tmp)},
    )

    assert precompact.returncode == session_end.returncode == 0
    assert precompact_elapsed < 3
    assert session_end_elapsed < 3
    assert len(_job_rows(memory_home)) == 1
    rendered = _only_job_payload(memory_home)["rendered_context"]
    for index in range(4):
        assert f"OLDER_DURABLE_TURN_{index}" in rendered
    assert "FINAL_DURABLE_TURN" in rendered
    assert list(hook_tmp.iterdir()) == []


def test_very_large_semantic_tail_completes_with_host_timeout_margin(tmp_path):
    source = tmp_path / "claude-19mb.jsonl"
    padding = "v" * 60_000
    routine = json.dumps({"type": "progress", "data": padding}) + "\n"
    with source.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "sessionId": "nineteen-mb-session",
                    "cwd": "/projects/nineteen-mb",
                }
            )
            + "\n"
        )
        for _ in range(130):
            stream.write(routine)
        stream.write(
            json.dumps(
                {
                    "message": {
                        "role": "user",
                        "content": "TWELVE_MB_BURIED_SIGNAL",
                    }
                }
            )
            + "\n"
        )
        for _ in range(200):
            stream.write(routine)
    assert source.stat().st_size > 19_000_000
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    hook_tmp = tmp_path / "hook-tmp"
    hook_tmp.mkdir()
    _fake_uv(fake_bin)

    result, elapsed = _run_hook(
        "session-end.py",
        {
            "session_id": "nineteen-mb-session",
            "transcript_path": str(source),
            "cwd": "/projects/nineteen-mb",
        },
        memory_home,
        fake_bin=fake_bin,
        extra_env={"TMPDIR": str(hook_tmp)},
    )

    assert result.returncode == 0, result.stderr
    assert elapsed < 2.75
    assert "TWELVE_MB_BURIED_SIGNAL" in _only_job_payload(memory_home)[
        "rendered_context"
    ]
    assert _only_job_source_path(memory_home).stat().st_size <= 16_100_000
    assert list(hook_tmp.iterdir()) == []


def test_backward_semantic_scan_finds_signal_beyond_raw_tail_cap(tmp_path):
    source = tmp_path / "claude-beyond-cap.jsonl"
    padding = "b" * 60_000
    routine = json.dumps({"type": "progress", "data": padding}) + "\n"
    with source.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "sessionId": "beyond-cap-session",
                    "cwd": "/projects/beyond-cap",
                }
            )
            + "\n"
        )
        for _ in range(30):
            stream.write(routine)
        stream.write(
            json.dumps(
                {
                    "message": {
                        "role": "user",
                        "content": "SIGNAL_BEYOND_SIXTEEN_MB",
                    }
                }
            )
            + "\n"
        )
        for _ in range(300):
            stream.write(routine)
    assert source.stat().st_size > 19_000_000
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    hook_tmp = tmp_path / "hook-tmp"
    hook_tmp.mkdir()
    _fake_uv(fake_bin)

    result, elapsed = _run_hook(
        "session-end.py",
        {
            "session_id": "beyond-cap-session",
            "transcript_path": str(source),
            "cwd": "/projects/beyond-cap",
        },
        memory_home,
        fake_bin=fake_bin,
        extra_env={"TMPDIR": str(hook_tmp)},
    )

    assert result.returncode == 0, result.stderr
    assert elapsed < 2.75
    assert "SIGNAL_BEYOND_SIXTEEN_MB" in _only_job_payload(memory_home)[
        "rendered_context"
    ]
    assert _only_job_source_path(memory_home).stat().st_size < 4_100_000
    assert list(hook_tmp.iterdir()) == []


@pytest.mark.parametrize(
    ("agent", "hook_name"),
    [("claude", "session-end.py"), ("codex", "codex-session-end.py")],
)
def test_semantic_compaction_pairs_calls_across_chunks_and_suppresses_reused_ids(
    tmp_path, agent, hook_name
):
    source = tmp_path / f"{agent}-cross-chunk.jsonl"
    routine_text = "p" * 60_000
    if agent == "claude":
        routine = {"type": "progress", "data": routine_text}

        def call(name, call_id, tool_input=None):
            return {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": tool_input or {},
                        }
                    ],
                }
            }

        def output(call_id, content):
            return {
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": content,
                        }
                    ],
                }
            }

        records = [
            call("Read", "reused"),
            *([routine] * 10),
            call("Agent", "reused"),
            *([routine] * 10),
            output("reused", "REUSED_PRIVATE_OUTPUT"),
            call(
                "AskUserQuestion",
                "ask",
                {
                    "questions": [
                        {"question": "Choose?", "options": [{"label": "A"}]}
                    ]
                },
            ),
            *([routine] * 10),
            output("ask", "CROSS_CHUNK_DECISION"),
            call("Task", "task"),
            *([routine] * 10),
            output("task", "CROSS_CHUNK_FINDING"),
        ]
        payload = {
            "session_id": "cross-chunk-claude",
            "transcript_path": str(source),
            "cwd": "/projects/cross-chunk",
        }
    else:
        routine = {
            "type": "event_msg",
            "payload": {"type": "agent_reasoning", "text": routine_text},
        }

        def call(name, call_id):
            return {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": name,
                    "call_id": call_id,
                    "arguments": "{}",
                },
            }

        def output(call_id, content):
            return {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(content),
                },
            }

        records = [
            call("shell", "reused"),
            *([routine] * 10),
            call("request_user_input", "reused"),
            *([routine] * 10),
            output("reused", {"answers": {"secret": "REUSED_PRIVATE_OUTPUT"}}),
            call("request_user_input", "ask"),
            *([routine] * 10),
            output("ask", {"answers": {"choice": "CROSS_CHUNK_DECISION"}}),
            call("wait_agent", "wait"),
            *([routine] * 10),
            output(
                "wait",
                {"status": "completed", "result": "CROSS_CHUNK_FINDING"},
            ),
            {
                "type": "response_item",
                "payload": {
                    "type": "agent_message",
                    "author": "/root/child",
                    "recipient": "/root",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Message Type: FINAL_ANSWER\n"
                                "Task name: /root\n"
                                "Sender: /root/child\n"
                                "Payload:\nCROSS_CHUNK_AGENT_MESSAGE"
                            ),
                        },
                        {"type": "input_text", "text": "AGENT_MESSAGE_PRIVATE"},
                    ],
                },
            },
        ]
        payload = {"transcript_path": str(source)}
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    hook_tmp = tmp_path / "hook-tmp"
    hook_tmp.mkdir()
    _fake_uv(fake_bin)

    result, elapsed = _run_hook(
        hook_name,
        payload,
        memory_home,
        fake_bin=fake_bin,
        extra_env={"TMPDIR": str(hook_tmp)},
    )

    assert result.returncode == 0, result.stderr
    assert elapsed < 2.75
    rendered = _only_job_payload(memory_home)["rendered_context"]
    assert "CROSS_CHUNK_DECISION" in rendered
    assert "CROSS_CHUNK_FINDING" in rendered
    assert "REUSED_PRIVATE_OUTPUT" not in rendered
    snapshot_text = _only_job_source_path(memory_home).read_text(encoding="utf-8")
    assert "REUSED_PRIVATE_OUTPUT" not in snapshot_text
    assert "AGENT_MESSAGE_PRIVATE" not in snapshot_text
    assert list(hook_tmp.iterdir()) == []


@pytest.mark.parametrize(
    ("agent", "hook_name"),
    [("claude", "session-end.py"), ("codex", "codex-session-end.py")],
)
def test_semantic_compaction_keeps_later_durable_result_after_empty_result(
    tmp_path, agent, hook_name
):
    source = tmp_path / f"{agent}-later-result.jsonl"
    if agent == "claude":
        records = [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "result-call",
                            "name": "Agent",
                            "input": {},
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "result-call",
                            "content": "",
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "result-call",
                            "content": "LATER_DURABLE_RESULT",
                        }
                    ],
                }
            },
        ]
        payload = {
            "session_id": "later-result",
            "transcript_path": str(source),
            "cwd": "/projects/later-result",
        }
    else:
        records = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "wait_agent",
                    "call_id": "result-call",
                    "arguments": "{}",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "result-call",
                    "output": "{}",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "result-call",
                    "output": json.dumps(
                        {"status": "completed", "result": "LATER_DURABLE_RESULT"}
                    ),
                },
            },
        ]
        payload = {"transcript_path": str(source)}
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    _fake_uv(fake_bin)

    result, _elapsed = _run_hook(
        hook_name, payload, memory_home, fake_bin=fake_bin
    )

    assert result.returncode == 0, result.stderr
    assert "LATER_DURABLE_RESULT" in _only_job_payload(memory_home)[
        "rendered_context"
    ]


def test_mixed_text_and_question_blocks_remain_one_turn_each(tmp_path):
    source = tmp_path / "mixed-blocks.jsonl"
    records = []
    for index in range(20):
        text_block = {"type": "text", "text": f"TEXT_{index:02d}"}
        question_block = {
            "type": "tool_use",
            "id": f"ask-{index}",
            "name": "AskUserQuestion",
            "input": {
                "questions": [
                    {
                        "question": f"QUESTION_{index:02d}?",
                        "options": [{"label": "Yes"}],
                    }
                ]
            },
        }
        records.append(
            {
                "message": {
                    "role": "assistant",
                    "content": [text_block, question_block]
                    if index % 2 == 0
                    else [question_block, text_block],
                }
            }
        )
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    _fake_uv(fake_bin)

    result, _elapsed = _run_hook(
        "session-end.py",
        {
            "session_id": "mixed-blocks",
            "transcript_path": str(source),
            "cwd": "/projects/mixed-blocks",
        },
        memory_home,
        fake_bin=fake_bin,
    )

    assert result.returncode == 0, result.stderr
    queued = _only_job_payload(memory_home)
    rendered = queued["rendered_context"]
    expected = parse_claude_transcript(
        source,
        {
            "session_id": "mixed-blocks",
            "cwd": "/projects/mixed-blocks",
            "trigger": "session_end",
        },
        limits={"max_turns": 30, "max_chars": 15_000},
    )
    assert "TEXT_00" in rendered
    assert "QUESTION_00" in rendered
    assert "TEXT_19" in rendered
    assert queued["turns"] == [asdict(turn) for turn in expected.turns]


def test_huge_semantic_suffix_stops_after_safe_final_thirty(tmp_path):
    source = tmp_path / "huge-semantic.jsonl"
    with source.open("w", encoding="utf-8") as stream:
        for index in range(150_000):
            stream.write(
                json.dumps(
                    {
                        "message": {
                            "role": "user",
                            "content": f"SEMANTIC_{index:06d}",
                        }
                    }
                )
                + "\n"
            )
    assert source.stat().st_size > 8_000_000
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    _fake_uv(fake_bin)

    result, elapsed = _run_hook(
        "session-end.py",
        {
            "session_id": "huge-semantic",
            "transcript_path": str(source),
            "cwd": "/projects/huge-semantic",
        },
        memory_home,
        fake_bin=fake_bin,
    )

    assert result.returncode == 0, result.stderr
    assert elapsed < 2.75
    payload = _only_job_payload(memory_home)
    assert len(payload["turns"]) == 30
    assert "SEMANTIC_149970" in payload["rendered_context"]
    assert "SEMANTIC_149999" in payload["rendered_context"]
    assert _only_job_source_path(memory_home).stat().st_size < 100_000


@pytest.mark.parametrize(
    ("agent", "kind", "hook_name"),
    [
        ("claude", "finding", "session-end.py"),
        ("codex", "decision", "codex-session-end.py"),
        ("codex", "finding", "codex-session-end.py"),
    ],
)
def test_long_history_retains_latest_resolved_tool_turn(
    tmp_path, agent, kind, hook_name
):
    source = tmp_path / f"{agent}-{kind}-long-history.jsonl"
    records = []
    for index in range(5_000):
        if agent == "claude":
            records.append(
                {"message": {"role": "user", "content": f"OLDER_{index:04d}"}}
            )
        else:
            records.append(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": f"OLDER_{index:04d}"}
                        ],
                    },
                }
            )
    if agent == "claude":
        records.extend(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "LATEST_MIXED_TEXT"},
                            {
                                "type": "tool_use",
                                "id": "latest-call",
                                "name": "Agent",
                                "input": {},
                            },
                        ],
                    }
                },
                {
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "latest-call",
                                "content": "LATEST_FINDING",
                            }
                        ],
                    }
                },
            ]
        )
        payload = {
            "session_id": "long-history",
            "transcript_path": str(source),
            "cwd": "/projects/long-history",
        }
    else:
        tool_name = "request_user_input" if kind == "decision" else "wait_agent"
        output = (
            {"answers": {"choice": "LATEST_DECISION"}}
            if kind == "decision"
            else {"status": "completed", "result": "LATEST_FINDING"}
        )
        records.extend(
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "name": tool_name,
                        "call_id": "latest-call",
                        "arguments": "{}",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "latest-call",
                        "output": json.dumps(output),
                    },
                },
            ]
        )
        payload = {"transcript_path": str(source)}
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    _fake_uv(fake_bin)

    result, elapsed = _run_hook(
        hook_name, payload, memory_home, fake_bin=fake_bin
    )

    assert result.returncode == 0, result.stderr
    assert elapsed < 2.75
    rendered = _only_job_payload(memory_home)["rendered_context"]
    assert "LATEST_FINDING" in rendered or "LATEST_DECISION" in rendered
    if agent == "claude":
        assert "LATEST_MIXED_TEXT" in rendered
    assert _only_job_source_path(memory_home).stat().st_size < 100_000


@pytest.mark.parametrize(
    ("agent", "hook_name"),
    [("claude", "session-end.py"), ("codex", "codex-session-end.py")],
)
def test_long_history_older_duplicate_id_suppresses_latest_result(
    tmp_path, agent, hook_name
):
    source = tmp_path / f"{agent}-old-duplicate.jsonl"
    if agent == "claude":
        older_call = {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "reused-call",
                        "name": "Read",
                        "input": {},
                    }
                ],
            }
        }
        latest_call = {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "reused-call",
                        "name": "Agent",
                        "input": {},
                    }
                ],
            }
        }
        latest_result = {
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "reused-call",
                        "content": "DUPLICATE_PRIVATE_RESULT",
                    }
                ],
            }
        }
        users = [
            {"message": {"role": "user", "content": f"SAFE_{index:04d}"}}
            for index in range(5_000)
        ]
        payload = {
            "session_id": "old-duplicate",
            "transcript_path": str(source),
            "cwd": "/projects/old-duplicate",
        }
    else:
        older_call = {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "shell",
                "call_id": "reused-call",
                "arguments": "{}",
            },
        }
        latest_call = {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "wait_agent",
                "call_id": "reused-call",
                "arguments": "{}",
            },
        }
        latest_result = {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "reused-call",
                "output": json.dumps(
                    {"status": "completed", "result": "DUPLICATE_PRIVATE_RESULT"}
                ),
            },
        }
        users = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"SAFE_{index:04d}"}],
                },
            }
            for index in range(5_000)
        ]
        payload = {"transcript_path": str(source)}
    source.write_text(
        "\n".join(
            json.dumps(record)
            for record in [older_call, *users, latest_call, latest_result]
        )
        + "\n",
        encoding="utf-8",
    )
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    _fake_uv(fake_bin)

    result, elapsed = _run_hook(
        hook_name, payload, memory_home, fake_bin=fake_bin
    )

    assert result.returncode == 0, result.stderr
    assert elapsed < 2.75
    rendered = _only_job_payload(memory_home)["rendered_context"]
    assert "SAFE_4999" in rendered
    assert "DUPLICATE_PRIVATE_RESULT" not in rendered
    assert "DUPLICATE_PRIVATE_RESULT" not in _only_job_source_path(
        memory_home
    ).read_text(encoding="utf-8")


def test_process_deadline_kills_blocking_enqueue_with_margin(tmp_path):
    hook = _load_hook("session-end.py")
    owned = tmp_path / "owned-partial.jsonl"
    owned.write_text("partial", encoding="utf-8")
    started = time.monotonic()

    with pytest.raises(hook.HookDeadlineExceeded):
        hook.run_process_until_deadline(
            [sys.executable, "-c", "import time; time.sleep(3.1)"],
            input_text="",
            deadline=time.monotonic() + 2.25,
            clock=time.monotonic,
            on_timeout=lambda: owned.unlink(missing_ok=True),
        )

    assert time.monotonic() - started < 2.75
    assert not owned.exists()


def test_process_nonzero_exit_preserves_failed_capture(tmp_path):
    hook = _load_hook("session-end.py")
    failed = tmp_path / "failed-claude-owner-digest.jsonl"
    failed.write_text("{}\n", encoding="utf-8")
    failed.chmod(0o600)

    with pytest.raises(RuntimeError, match="capture child exited 1"):
        hook.run_process_until_deadline(
            [sys.executable, "-c", "raise SystemExit(1)"],
            input_text="",
            deadline=time.monotonic() + 1.0,
            clock=time.monotonic,
            on_timeout=lambda: failed.unlink(missing_ok=True),
        )

    assert failed.exists()
    assert stat.S_IMODE(failed.stat().st_mode) == 0o600


def test_bounded_slice_does_not_close_reused_descriptor(tmp_path, monkeypatch):
    hook = _load_hook("session-end.py")
    source = tmp_path / "source.jsonl"
    source.write_text('{}\n', encoding="utf-8")
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep open", encoding="utf-8")
    reused: list[int] = []

    def fail_after_descriptor_transfer(*_args, **_kwargs):
        reused.append(os.open(unrelated, os.O_RDONLY))
        raise RuntimeError("selection failed")
        yield  # pragma: no cover

    monkeypatch.setattr(hook, "_validated_chunks_backward", fail_after_descriptor_transfer)

    with pytest.raises(RuntimeError, match="selection failed"):
        with hook.bounded_transcript_slice(
            source,
            lambda _path: None,
            source_agent="claude",
            memory_root=tmp_path / "memory",
            deadline=10.0,
            clock=lambda: 0.0,
        ):
            pass

    assert os.fstat(reused[0]).st_size == len("keep open")
    os.close(reused[0])


def test_timeout_cleanup_checks_owner_token_and_queue_reference(tmp_path):
    hook = _load_hook("session-end.py")
    root = tmp_path / "memory"
    spool = root / "scripts" / "spool"
    spool.mkdir(parents=True)
    removable = spool / "capture-timeoutowner-one.jsonl"
    failed = spool / "failed-claude-timeoutowner-two.jsonl"
    unrelated = spool / "capture-otherowner-three.jsonl"
    for path in (removable, failed, unrelated):
        path.write_text("{}\n", encoding="utf-8")
        path.chmod(0o600)

    hook.cleanup_uncommitted_capture("timeoutowner", root=root)

    assert not removable.exists()
    assert not failed.exists()
    assert unrelated.exists()

    referenced = spool / "capture-timeoutowner-referenced.jsonl"
    referenced.write_text("{}\n", encoding="utf-8")
    referenced.chmod(0o600)
    database = root / "scripts" / "jobs.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE jobs (source_path TEXT NOT NULL)")
        connection.execute("INSERT INTO jobs VALUES (?)", (str(referenced),))

    hook.cleanup_uncommitted_capture("timeoutowner", root=root)

    assert referenced.exists()


def test_private_slice_works_when_windows_has_no_fchmod(tmp_path, monkeypatch):
    hook = _load_hook("session-end.py")
    source = tmp_path / "windows.jsonl"
    source.write_text(
        json.dumps({"message": {"role": "user", "content": "WINDOWS_SIGNAL"}})
        + "\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "memory" / "scripts" / "runtime"
    runtime.mkdir(parents=True)
    runtime.chmod(0o700)
    monkeypatch.delattr(hook.os, "fchmod")

    with hook.bounded_transcript_slice(
        source,
        lambda path: hook.parse_claude_transcript(
            path, {"trigger": "session_end"}, limits={"max_turns": 30}
        ),
        source_agent="claude",
        memory_root=tmp_path / "memory",
        deadline=10.0,
        clock=lambda: 0.0,
    ) as selected:
        private_slice, preview = selected
        assert private_slice.exists()
        assert preview.turns[0].text == "WINDOWS_SIGNAL"

    assert not private_slice.exists()


def test_live_slice_uses_private_memory_root_runtime_directory(tmp_path):
    hook = _load_hook("session-end.py")
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"message": {"role": "user", "content": "MEMORY_ROOT_SIGNAL"}})
        + "\n",
        encoding="utf-8",
    )
    memory_home = tmp_path / "memory"

    with hook.bounded_transcript_slice(
        source,
        lambda path: path.read_text(encoding="utf-8"),
        source_agent="claude",
        memory_root=memory_home,
        deadline=10.0,
        clock=lambda: 0.0,
    ) as selected:
        private_slice, _preview = selected
        assert private_slice.parent == memory_home / "scripts" / "runtime"
        assert stat.S_IMODE(private_slice.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(private_slice.stat().st_mode) == 0o600

    assert not private_slice.exists()


@pytest.mark.parametrize("failure", [ValueError("preview failed"), asyncio.CancelledError()])
def test_live_slice_in_memory_root_is_removed_on_failure_or_cancellation(
    tmp_path, failure
):
    hook = _load_hook("session-end.py")
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"message": {"role": "user", "content": "SIGNAL"}}) + "\n",
        encoding="utf-8",
    )
    memory_home = tmp_path / "memory"
    observed: list[Path] = []

    def fail(path):
        observed.append(path)
        raise failure

    with pytest.raises(type(failure)):
        with hook.bounded_transcript_slice(
            source,
            fail,
            source_agent="claude",
            memory_root=memory_home,
            deadline=10.0,
            clock=lambda: 0.0,
        ):
            pass

    assert len(observed) == 1
    assert not observed[0].exists()


@pytest.mark.parametrize("linked_component", ["root", "scripts", "runtime"])
def test_live_slice_rejects_linked_memory_root_parents(tmp_path, linked_component):
    hook = _load_hook("session-end.py")
    source = tmp_path / "source.jsonl"
    source.write_text('{}\n', encoding="utf-8")
    memory_home = tmp_path / "memory"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("unchanged", encoding="utf-8")
    if linked_component == "root":
        memory_home.symlink_to(external, target_is_directory=True)
    else:
        memory_home.mkdir()
        scripts = memory_home / "scripts"
        if linked_component == "scripts":
            scripts.symlink_to(external, target_is_directory=True)
        else:
            scripts.mkdir()
            (scripts / "runtime").symlink_to(external, target_is_directory=True)

    before = _tree_manifest(external)
    with pytest.raises(ValueError):
        with hook.bounded_transcript_slice(
            source,
            lambda _path: None,
            source_agent="claude",
            memory_root=memory_home,
            deadline=10.0,
            clock=lambda: 0.0,
        ):
            pass

    assert _tree_manifest(external) == before


def test_cross_platform_fallback_rejects_reparse_runtime_component(
    tmp_path, monkeypatch
):
    import scripts.utils as memory_utils

    memory_home = tmp_path / "memory"
    runtime = memory_home / "scripts" / "runtime"
    runtime.mkdir(parents=True)
    runtime_info = runtime.lstat()
    real_link_or_reparse = memory_utils._link_or_reparse

    def mark_runtime_reparse(info):
        return (
            (info.st_dev, info.st_ino) == (runtime_info.st_dev, runtime_info.st_ino)
            or real_link_or_reparse(info)
        )

    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())
    monkeypatch.setattr(memory_utils, "_link_or_reparse", mark_runtime_reparse)

    with pytest.raises(ValueError, match="non-reparse"):
        with memory_utils.open_secure_runtime_file(memory_home):
            pass

    assert list(runtime.iterdir()) == []


def test_live_slice_fails_closed_when_runtime_chmod_fails(tmp_path, monkeypatch):
    import scripts.utils as memory_utils

    hook = _load_hook("session-end.py")
    source = tmp_path / "source.jsonl"
    source.write_text('{}\n', encoding="utf-8")
    real_fchmod = memory_utils.os.fchmod

    def deny_directory_chmod(descriptor, mode):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise PermissionError("runtime chmod denied")
        return real_fchmod(descriptor, mode)

    monkeypatch.setattr(memory_utils.os, "fchmod", deny_directory_chmod)

    with pytest.raises(PermissionError, match="runtime chmod denied"):
        with hook.bounded_transcript_slice(
            source,
            lambda _path: None,
            source_agent="claude",
            memory_root=tmp_path / "memory",
            deadline=10.0,
            clock=lambda: 0.0,
        ):
            pass


def test_live_slice_directory_swap_never_creates_in_external_target(
    tmp_path, monkeypatch
):
    hook = _load_hook("session-end.py")
    source = tmp_path / "source.jsonl"
    source.write_text('{}\n', encoding="utf-8")
    memory_home = tmp_path / "memory"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("unchanged", encoding="utf-8")
    observed: list[Path] = []
    import scripts.utils as memory_utils

    real_prepare = memory_utils._bootstrap_secure_runtime_directory_posix

    def swap_after_validation(root):
        runtime = real_prepare(root)
        runtime.rename(runtime.with_name("runtime-pinned"))
        runtime.symlink_to(external, target_is_directory=True)
        return runtime

    monkeypatch.setattr(
        memory_utils,
        "_bootstrap_secure_runtime_directory_posix",
        swap_after_validation,
    )
    before = _tree_manifest(external)

    with pytest.raises((OSError, ValueError)):
        with hook.bounded_transcript_slice(
            source,
            lambda path: observed.append(path.resolve()),
            source_agent="claude",
            memory_root=memory_home,
            deadline=10.0,
            clock=lambda: 0.0,
        ):
            pass

    assert observed == []
    assert _tree_manifest(external) == before


def test_runtime_bootstrap_helpers_are_private_implementation_details():
    import scripts.utils as memory_utils

    assert not hasattr(memory_utils, "prepare_secure_runtime_directory")
    assert not hasattr(memory_utils, "prepare_secure_runtime_directory_fallback")


def test_live_slice_rejects_previewer_symlink_without_overwriting_target(tmp_path):
    hook = _load_hook("session-end.py")
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"message": {"role": "user", "content": "SIGNAL"}}) + "\n",
        encoding="utf-8",
    )
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("do not overwrite", encoding="utf-8")

    def replace_with_symlink(path):
        path.unlink()
        path.symlink_to(sentinel)
        return None

    with pytest.raises(ValueError, match="identity|link"):
        with hook.bounded_transcript_slice(
            source,
            replace_with_symlink,
            source_agent="claude",
            memory_root=tmp_path / "memory",
            deadline=10.0,
            clock=lambda: 0.0,
        ):
            pass

    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"


@pytest.mark.parametrize("mutation", ["hardlink", "mode"])
def test_live_slice_rejects_previewer_file_identity_mutation(
    tmp_path, mutation
):
    hook = _load_hook("session-end.py")
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"message": {"role": "user", "content": "SIGNAL"}}) + "\n",
        encoding="utf-8",
    )

    def mutate(path):
        if mutation == "hardlink":
            os.link(path, tmp_path / "outside-link")
        else:
            path.chmod(0o666)
        return None

    with pytest.raises(ValueError, match="link|permission|mode"):
        with hook.bounded_transcript_slice(
            source,
            mutate,
            source_agent="claude",
            memory_root=tmp_path / "memory",
            deadline=10.0,
            clock=lambda: 0.0,
        ):
            pass


@pytest.mark.parametrize(
    "overrides",
    [
        {"st_mode": stat.S_IFREG | 0o666},
        {"st_mode": stat.S_IFREG | 0o600, "st_nlink": 2},
        {
            "st_mode": stat.S_IFREG | 0o600,
            "st_file_attributes": getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
        },
    ],
    ids=["mode", "hardlink", "reparse"],
)
def test_runtime_file_validator_rejects_unsafe_metadata(tmp_path, overrides):
    import scripts.utils as memory_utils

    values = {
        "st_mode": stat.S_IFREG | 0o600,
        "st_nlink": 1,
        "st_uid": os.getuid() if hasattr(os, "getuid") else 0,
        "st_dev": 1,
        "st_ino": 2,
        "st_file_attributes": 0,
    }
    values.update(overrides)
    validator = getattr(memory_utils, "_validate_runtime_file", None)
    assert validator is not None, "secure runtime file validator is missing"

    with pytest.raises(ValueError):
        validator(SimpleNamespace(**values), tmp_path / "slice.jsonl")


def test_live_slice_cross_platform_fallback_uses_existing_private_runtime(
    tmp_path, monkeypatch
):
    import scripts.utils as memory_utils

    hook = _load_hook("session-end.py")
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"message": {"role": "user", "content": "FALLBACK_SIGNAL"}})
        + "\n",
        encoding="utf-8",
    )
    memory_home = tmp_path / "memory"
    runtime = memory_home / "scripts" / "runtime"
    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())

    with hook.bounded_transcript_slice(
        source,
        lambda path: path.read_text(encoding="utf-8"),
        source_agent="claude",
        memory_root=memory_home,
        deadline=10.0,
        clock=lambda: 0.0,
    ) as selected:
        private_slice, preview = selected
        assert private_slice.parent == runtime
        assert "FALLBACK_SIGNAL" in preview

    assert list(runtime.iterdir()) == []
    assert stat.S_IMODE(runtime.stat().st_mode) == 0o700


def test_cross_platform_fallback_creates_fresh_private_runtime_tree(
    tmp_path, monkeypatch
):
    import scripts.utils as memory_utils

    memory_home = tmp_path / "memory"
    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())

    with memory_utils.open_secure_runtime_file(memory_home) as (path, descriptor):
        assert path.parent == memory_home / "scripts" / "runtime"
        assert os.fstat(descriptor).st_nlink == 1
        assert stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o600

    assert stat.S_IMODE((memory_home / "scripts" / "runtime").stat().st_mode) == 0o700
    assert list((memory_home / "scripts" / "runtime").iterdir()) == []


def test_cross_platform_fallback_creates_fresh_tree_without_directory_handles(
    tmp_path, monkeypatch
):
    import scripts.utils as memory_utils

    memory_home = tmp_path / "memory"

    def unsupported_chmod(*_args, **_kwargs):
        raise AssertionError("descriptorless fallback must not use path chmod")

    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())
    monkeypatch.setattr(memory_utils, "_open_runtime_directory_nofollow", lambda _path: None)
    monkeypatch.setattr(memory_utils.os, "chmod", unsupported_chmod)

    with memory_utils.open_secure_runtime_file(memory_home) as (path, descriptor):
        assert path.parent == memory_home / "scripts" / "runtime"
        assert os.fstat(descriptor).st_nlink == 1

    assert list((memory_home / "scripts" / "runtime").iterdir()) == []


def test_windows_fallback_establishes_acl_for_fresh_live_capture(tmp_path, monkeypatch):
    import scripts.utils as memory_utils

    hook = _load_hook("session-end.py")
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"message": {"role": "user", "content": "WINDOWS_SIGNAL"}})
        + "\n",
        encoding="utf-8",
    )
    memory_home = tmp_path / "memory"
    secured = []
    secured_files = []

    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())
    monkeypatch.setattr(memory_utils, "_open_runtime_directory_nofollow", lambda _path: None)
    monkeypatch.setattr(memory_utils, "_windows_acl_required", lambda: True)
    monkeypatch.setattr(
        memory_utils,
        "_secure_windows_runtime_directory",
        lambda path, *, owner_only: secured.append((path, owner_only)),
    )
    monkeypatch.setattr(
        memory_utils,
        "_secure_windows_runtime_file",
        lambda descriptor, path: secured_files.append((descriptor, path)),
    )

    with hook.bounded_transcript_slice(
        source,
        lambda path: path.read_text(encoding="utf-8"),
        source_agent="claude",
        memory_root=memory_home,
        deadline=10.0,
        clock=lambda: 0.0,
    ) as (_path, preview):
        assert "WINDOWS_SIGNAL" in preview

    assert secured == [
        (memory_home, False),
        (memory_home / "scripts", False),
        (memory_home / "scripts" / "runtime", True),
    ]
    assert len(secured_files) == 1
    assert secured_files[0][1].parent == memory_home / "scripts" / "runtime"


def test_windows_fallback_accepts_inherited_acl_on_existing_ancestry(
    tmp_path, monkeypatch
):
    import scripts.utils as memory_utils

    memory_home = tmp_path / "memory"
    scripts = memory_home / "scripts"
    scripts.mkdir(parents=True)

    secured = []

    def accept_inherited(path, *, owner_only):
        secured.append((path, owner_only))

    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())
    monkeypatch.setattr(memory_utils, "_open_runtime_directory_nofollow", lambda _path: None)
    monkeypatch.setattr(memory_utils, "_windows_acl_required", lambda: True)
    monkeypatch.setattr(
        memory_utils, "_secure_windows_runtime_directory", accept_inherited
    )
    monkeypatch.setattr(memory_utils, "_secure_windows_runtime_file", lambda *_: None)

    with memory_utils.open_secure_runtime_file(memory_home):
        pass

    assert secured == [
        (memory_home, False),
        (scripts, False),
        (scripts / "runtime", True),
    ]


def test_windows_fallback_fails_closed_when_acl_api_is_unavailable(
    tmp_path, monkeypatch
):
    import scripts.utils as memory_utils

    memory_home = tmp_path / "memory"

    def unavailable(_path, *, owner_only):
        assert owner_only is False
        raise PermissionError("Windows ACL API is unavailable")

    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())
    monkeypatch.setattr(memory_utils, "_open_runtime_directory_nofollow", lambda _path: None)
    monkeypatch.setattr(memory_utils, "_windows_acl_required", lambda: True)
    monkeypatch.setattr(memory_utils, "_secure_windows_runtime_directory", unavailable)

    with pytest.raises(PermissionError, match="ACL API is unavailable"):
        with memory_utils.open_secure_runtime_file(memory_home):
            pass

    assert not (memory_home / "scripts").exists()


def test_cross_platform_fallback_rejects_unsafe_owner_before_creation(
    tmp_path, monkeypatch
):
    import scripts.utils as memory_utils

    memory_home = tmp_path / "memory"
    real_uid = os.getuid()
    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())
    monkeypatch.setattr(memory_utils.os, "getuid", lambda: real_uid + 1)

    with pytest.raises(ValueError, match="unsafe owner"):
        with memory_utils.open_secure_runtime_file(memory_home):
            pass

    assert not memory_home.exists()


def test_cross_platform_fallback_rejects_group_writable_scripts(
    tmp_path, monkeypatch
):
    import scripts.utils as memory_utils

    memory_home = tmp_path / "memory"
    scripts = memory_home / "scripts"
    scripts.mkdir(parents=True)
    scripts.chmod(0o777)
    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())

    with pytest.raises(ValueError, match="unsafe permissions"):
        with memory_utils.open_secure_runtime_file(memory_home):
            pass

    assert not (scripts / "runtime").exists()


@pytest.mark.parametrize("component", ["root", "scripts", "runtime"])
def test_cross_platform_fallback_rejects_linked_ancestry(
    tmp_path, monkeypatch, component
):
    import scripts.utils as memory_utils

    memory_home = tmp_path / "memory"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("unchanged", encoding="utf-8")
    if component == "root":
        memory_home.symlink_to(external, target_is_directory=True)
    else:
        memory_home.mkdir()
        scripts = memory_home / "scripts"
        if component == "scripts":
            scripts.symlink_to(external, target_is_directory=True)
        else:
            scripts.mkdir()
            (scripts / "runtime").symlink_to(external, target_is_directory=True)
    before = _tree_manifest(external)
    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())

    with pytest.raises((OSError, ValueError)):
        with memory_utils.open_secure_runtime_file(memory_home):
            pass

    assert _tree_manifest(external) == before


def test_cross_platform_fallback_detects_parent_swap_without_outside_artifact(
    tmp_path, monkeypatch
):
    import scripts.utils as memory_utils

    memory_home = tmp_path / "memory"
    memory_home.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("unchanged", encoding="utf-8")
    before_names = sorted(path.relative_to(external) for path in external.rglob("*"))
    real_mkdir = memory_utils.os.mkdir

    def swap_then_mkdir(path, mode=0o777, *, dir_fd=None):
        candidate = Path(path)
        if candidate == memory_home / "scripts":
            memory_home.rename(tmp_path / "memory-pinned")
            memory_home.symlink_to(external, target_is_directory=True)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())
    monkeypatch.setattr(memory_utils.os, "mkdir", swap_then_mkdir)

    with pytest.raises(ValueError, match="identity changed"):
        with memory_utils.open_secure_runtime_file(memory_home):
            pass

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert sorted(path.relative_to(external) for path in external.rglob("*")) == before_names


def test_cross_platform_fallback_detects_slice_parent_swap_and_cleans_artifact(
    tmp_path, monkeypatch
):
    import scripts.utils as memory_utils

    memory_home = tmp_path / "memory"
    runtime = memory_home / "scripts" / "runtime"
    runtime.mkdir(parents=True)
    runtime.chmod(0o700)
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("unchanged", encoding="utf-8")
    before_names = sorted(path.relative_to(external) for path in external.rglob("*"))
    real_mkstemp = memory_utils.tempfile.mkstemp

    def swap_then_mkstemp(*, prefix, suffix, dir):
        runtime.rename(runtime.with_name("runtime-pinned"))
        runtime.symlink_to(external, target_is_directory=True)
        return real_mkstemp(prefix=prefix, suffix=suffix, dir=dir)

    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())
    monkeypatch.setattr(memory_utils.tempfile, "mkstemp", swap_then_mkstemp)

    with pytest.raises(ValueError, match="identity changed|non-symlink"):
        with memory_utils.open_secure_runtime_file(memory_home):
            pass

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert sorted(path.relative_to(external) for path in external.rglob("*")) == before_names


def test_cross_platform_fallback_creation_chmod_failure_is_closed(
    tmp_path, monkeypatch
):
    import scripts.utils as memory_utils

    memory_home = tmp_path / "memory"
    real_fchmod = memory_utils.os.fchmod

    def deny_runtime_chmod(descriptor, mode):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise PermissionError("fallback runtime chmod denied")
        return real_fchmod(descriptor, mode)

    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())
    monkeypatch.setattr(memory_utils.os, "fchmod", deny_runtime_chmod)

    with pytest.raises(PermissionError, match="fallback runtime chmod denied"):
        with memory_utils.open_secure_runtime_file(memory_home):
            pass

    runtime = memory_home / "scripts" / "runtime"
    assert not runtime.exists() or list(runtime.iterdir()) == []


def test_internal_deadline_fails_before_enqueue_and_cleans_artifacts(
    tmp_path, monkeypatch
):
    source = tmp_path / "deadline.jsonl"
    source.write_text(
        json.dumps({"message": {"role": "user", "content": "TIMEOUT_SIGNAL"}})
        + "\n",
        encoding="utf-8",
    )
    memory_home = tmp_path / "memory"
    hook = _load_hook("session-end.py")
    ticks = iter([0.0, 0.1, 0.2, 2.5])
    monkeypatch.setenv("AI_MEMORY_HOME", str(memory_home))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "session_id": "deadline-session",
                    "transcript_path": str(source),
                    "cwd": "/projects/deadline",
                }
            )
        ),
    )
    monkeypatch.setattr(
        hook,
        "enqueue_hook_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("deadline must prevent enqueue")
        ),
    )

    hook.main(clock=lambda: next(ticks))

    assert not (memory_home / "scripts" / "jobs.sqlite3").exists()
    assert not (memory_home / "scripts" / "spool").exists()
    assert list((memory_home / "scripts" / "runtime").iterdir()) == []


def test_codex_semantic_tail_expansion_preserves_signal_behind_larger_tail(tmp_path):
    source = tmp_path / "codex-semantic.jsonl"
    _write_signal_before_routine_tail(
        source, agent="codex", routine_records=150
    )
    assert source.stat().st_size > 8_500_000
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    hook_tmp = tmp_path / "hook-tmp"
    hook_tmp.mkdir()
    _fake_uv(fake_bin)

    result, elapsed = _run_hook(
        "codex-session-end.py",
        {"transcript_path": str(source)},
        memory_home,
        fake_bin=fake_bin,
        extra_env={"TMPDIR": str(hook_tmp)},
    )

    assert result.returncode == 0, result.stderr
    assert elapsed < 3
    assert "LAST_DURABLE_SIGNAL_CODEX" in _only_job_payload(memory_home)[
        "rendered_context"
    ]
    snapshot = _only_job_source_path(memory_home)
    assert snapshot.stat().st_size < 16_100_000
    assert all(
        isinstance(json.loads(line), dict)
        for line in snapshot.read_text(encoding="utf-8").splitlines()
    )
    assert list(hook_tmp.iterdir()) == []


@pytest.mark.parametrize(
    "malformed_record",
    [
        b'{"malformed":"unterminated"\n',
        b'{"invalid_utf8":"\xff"}\n',
        b'{"incomplete_final":',
    ],
    ids=["malformed-json", "invalid-utf8", "incomplete-final"],
)
def test_malformed_retained_record_fails_closed_without_artifacts(
    tmp_path, malformed_record
):
    source = tmp_path / "malformed-private-name.jsonl"
    source.write_bytes(
        json.dumps(
            {"message": {"role": "user", "content": "VALID_SIGNAL"}}
        ).encode()
        + b"\n"
        + malformed_record
    )
    memory_home = tmp_path / "memory"
    hook_tmp = tmp_path / "hook-tmp"
    hook_tmp.mkdir()

    result, elapsed = _run_hook(
        "session-end.py",
        {
            "session_id": "malformed",
            "transcript_path": str(source),
            "cwd": "/projects/malformed",
        },
        memory_home,
        extra_env={"TMPDIR": str(hook_tmp)},
    )

    assert result.returncode == 0
    assert elapsed < 3
    _assert_diagnostic_only(memory_home)
    assert not (memory_home / "scripts" / "spool").exists()
    assert list(hook_tmp.iterdir()) == []


def _load_hook(name: str):
    path = HOOKS / name
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("hook_name,logger_name,label", HOOK_LOGGERS)
def test_hook_logger_creates_private_log_and_preserves_host_handlers(
    tmp_path, monkeypatch, hook_name, logger_name, label
):
    hook = _load_hook(hook_name)
    memory_home = tmp_path / "memory"
    monkeypatch.setenv("AI_MEMORY_HOME", str(memory_home))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    logger = logging.getLogger(logger_name)
    _close_hook_handlers(logger)
    host_handler = logging.NullHandler()
    logger.addHandler(host_handler)
    try:
        first = hook._logger()
        second = hook._logger()
        first.info("private hook message")

        path = memory_home / "scripts" / "logs" / "hooks.log"
        assert first is second is logger
        assert host_handler in logger.handlers
        tagged = [
            handler
            for handler in logger.handlers
            if getattr(handler, "_memory_hook_file", False)
        ]
        assert len(tagged) == 1
        assert logger.propagate is False
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.stat().st_nlink == 1
        if hasattr(os, "getuid"):
            assert path.stat().st_uid == os.getuid()
        record = _hook_log_records(path)[0]
        assert set(record) == {
            "timestamp",
            "level",
            "component",
            "event",
            "logger",
            "message",
        }
        assert record["timestamp"].endswith("Z")
        datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        assert record == {
            **record,
            "level": "INFO",
            "component": label,
            "event": "hook_log",
            "logger": logger_name,
            "message": "private hook message",
        }
    finally:
        _close_hook_handlers(logger)


@pytest.mark.parametrize("hook_name,logger_name,_label", HOOK_LOGGERS)
@pytest.mark.parametrize(
    "attack",
    [
        "root-symlink",
        "scripts-symlink",
        "logs-symlink",
        "file-symlink",
        "file-hardlink",
    ],
)
def test_hook_logger_rejects_linked_paths_without_external_mutation(
    tmp_path, monkeypatch, hook_name, logger_name, _label, attack
):
    hook = _load_hook(hook_name)
    memory_home = tmp_path / "memory"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.log"
    sentinel.write_text("do not mutate", encoding="utf-8")
    sentinel.chmod(0o600)
    if attack == "root-symlink":
        memory_home.symlink_to(external, target_is_directory=True)
        scripts = memory_home / "scripts"
    else:
        memory_home.mkdir()
        scripts = memory_home / "scripts"
    if attack == "root-symlink":
        pass
    elif attack == "scripts-symlink":
        scripts.symlink_to(external, target_is_directory=True)
    else:
        scripts.mkdir()
        logs = scripts / "logs"
        if attack == "logs-symlink":
            logs.symlink_to(external, target_is_directory=True)
        else:
            logs.mkdir()
            target = logs / "hooks.log"
            if attack == "file-symlink":
                target.symlink_to(sentinel)
            else:
                os.link(sentinel, target)
    before = _tree_manifest(external)
    monkeypatch.setenv("AI_MEMORY_HOME", str(memory_home))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    logger = logging.getLogger(logger_name)
    _close_hook_handlers(logger)
    try:
        configured = hook._logger()
        configured.error("must not escape")

        assert _tree_manifest(external) == before
        assert len(configured.handlers) == 1
        assert isinstance(configured.handlers[0], logging.NullHandler)
        assert getattr(configured.handlers[0], "_memory_hook_file", False)
    finally:
        _close_hook_handlers(logger)


@pytest.mark.parametrize("hook_name,logger_name,_label", HOOK_LOGGERS)
def test_hook_logger_cross_platform_fallback_creates_private_file(
    tmp_path, monkeypatch, hook_name, logger_name, _label
):
    import scripts.utils as memory_utils

    hook = _load_hook(hook_name)
    memory_home = tmp_path / "memory"
    memory_home.mkdir()
    (memory_home / "scripts").mkdir()
    monkeypatch.setenv("AI_MEMORY_HOME", str(memory_home))
    monkeypatch.setattr(memory_utils.os, "supports_dir_fd", set())
    logger = logging.getLogger(logger_name)
    _close_hook_handlers(logger)
    try:
        hook._logger().warning("fallback message")

        path = memory_home / "scripts" / "logs" / "hooks.log"
        assert [record["message"] for record in _hook_log_records(path)] == [
            "fallback message"
        ]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.stat().st_nlink == 1
    finally:
        _close_hook_handlers(logger)


def test_secure_hook_logging_detects_windows_reparse_attribute(monkeypatch):
    import scripts.utils as memory_utils

    monkeypatch.setattr(
        memory_utils.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400, raising=False
    )
    info = type(
        "WindowsStat",
        (),
        {"st_mode": stat.S_IFDIR | 0o700, "st_file_attributes": 0x400},
    )()

    assert memory_utils._link_or_reparse(info)


def test_hook_jsonl_encodes_quotes_newlines_and_omits_exception_details(
    tmp_path, monkeypatch
):
    hook = _load_hook("session-end.py")
    memory_home = tmp_path / "memory"
    monkeypatch.setenv("AI_MEMORY_HOME", str(memory_home))
    logger = logging.getLogger("ai-memory-session-end")
    _close_hook_handlers(logger)
    try:
        try:
            raise RuntimeError("credential-must-not-leak")
        except RuntimeError:
            hook._logger().error('quoted "message"\nsecond line', exc_info=True)

        path = memory_home / "scripts" / "logs" / "hooks.log"
        physical_lines = path.read_text(encoding="utf-8").splitlines()
        assert len(physical_lines) == 1
        record = json.loads(physical_lines[0])
        assert record["message"] == 'quoted "message"\nsecond line'
        assert "credential-must-not-leak" not in physical_lines[0]
        assert "Traceback" not in physical_lines[0]
    finally:
        _close_hook_handlers(logger)


def test_hook_jsonl_concurrent_process_appends_remain_complete_records(tmp_path):
    memory_home = tmp_path / "memory"
    probe = """
import sys
from scripts.hook_logging import configure_hook_logger
logger = configure_hook_logger("concurrent-hook-test", "session-end", sys.argv[1])
logger.info('worker=%s quote=" newline=\\n', sys.argv[2])
"""

    def write_one(index: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", probe, str(memory_home), str(index)],
            cwd=ROOT,
            env=_hook_env(memory_home),
            capture_output=True,
            text=True,
            timeout=5,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write_one, range(24)))

    assert all(result.returncode == 0 for result in results)
    records = _hook_log_records(memory_home / "scripts" / "logs" / "hooks.log")
    assert len(records) == 24
    assert {record["message"] for record in records} == {
        f'worker={index} quote=" newline=\n' for index in range(24)
    }
    assert all(record["component"] == "session-end" for record in records)


@pytest.mark.parametrize("hook_name,logger_name,_label", HOOK_LOGGERS)
def test_hook_logger_replaces_only_its_handler_when_runtime_path_changes(
    tmp_path, monkeypatch, hook_name, logger_name, _label
):
    hook = _load_hook(hook_name)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    logger = logging.getLogger(logger_name)
    _close_hook_handlers(logger)
    host_handler = logging.NullHandler()
    logger.addHandler(host_handler)
    try:
        monkeypatch.setenv("AI_MEMORY_HOME", str(first_root))
        hook._logger().info("first path")
        monkeypatch.setenv("AI_MEMORY_HOME", str(second_root))
        hook._logger().info("second path")

        first_text = (first_root / "scripts" / "logs" / "hooks.log").read_text()
        second_text = (second_root / "scripts" / "logs" / "hooks.log").read_text()
        assert "first path" in first_text
        assert "second path" not in first_text
        assert "second path" in second_text
        assert host_handler in logger.handlers
        assert sum(
            bool(getattr(handler, "_memory_hook_file", False))
            for handler in logger.handlers
        ) == 1
    finally:
        _close_hook_handlers(logger)


def _write_retrieval_files(memory_home: Path) -> None:
    (memory_home / "knowledge").mkdir(parents=True)
    (memory_home / "daily").mkdir()
    (memory_home / "knowledge" / "index.md").write_text(
        "# Index\n\n"
        "| Title | Project |\n"
        "| --- | --- |\n"
        "| Alpha fact | alpha |\n"
        "| Global fact | global |\n"
        "| Beta secret | beta |\n",
        encoding="utf-8",
    )
    (memory_home / "daily" / "2026-08-11.md").write_text(
        "# Daily\n\n"
        "### Alpha\n**Project:** alpha\nALPHA_RECENT\n\n"
        "### Beta\n**Project:** beta\nBETA_RECENT\n",
        encoding="utf-8",
    )


def test_shared_retrieval_builder_returns_plain_project_scoped_context(tmp_path):
    memory_home = tmp_path / "memory"
    _write_retrieval_files(memory_home)
    session_start = _load_hook("session-start.py")

    context = session_start.build_context(
        "alpha", memory_home=memory_home, now=FIXED_NOW
    )

    assert "Alpha fact" in context
    assert "Global fact" in context
    assert "ALPHA_RECENT" in context
    assert "Beta secret" not in context
    assert "BETA_RECENT" not in context


def test_claude_session_start_wraps_shared_context_in_claude_shape(tmp_path):
    memory_home = tmp_path / "memory"
    _write_retrieval_files(memory_home)
    result, _ = _run_hook(
        "session-start.py",
        {"cwd": "/projects/alpha"},
        memory_home,
        extra_env={"CLAUDE_PROJECT_DIR": "/projects/alpha"},
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    context = output["hookSpecificOutput"]["additionalContext"]
    assert "Alpha fact" in context
    assert "Beta secret" not in context


def test_codex_session_start_wraps_shared_context_in_supported_shape(tmp_path):
    memory_home = tmp_path / "memory"
    _write_retrieval_files(memory_home)
    result, _ = _run_hook(
        "codex-session-start.py",
        {"hook_event_name": "SessionStart", "cwd": "/projects/alpha"},
        memory_home,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output == {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": output["hookSpecificOutput"]["additionalContext"],
        }
    }
    assert "Alpha fact" in output["hookSpecificOutput"]["additionalContext"]
    assert "Beta secret" not in output["hookSpecificOutput"]["additionalContext"]


def test_hook_examples_preserve_ten_second_capture_timeouts_and_are_opt_in():
    claude = json.loads((ROOT / ".claude" / "settings.json").read_text())
    for event in ("PreCompact", "SessionEnd"):
        command = claude["hooks"][event][0]["hooks"][0]
        assert command["timeout"] == 10
        assert "AI_MEMORY_HOME" in command["command"]
        assert "CLAUDE_MEMORY_HOME" in command["command"]

    codex = json.loads((ROOT / ".codex" / "hooks.json.example").read_text())
    assert set(codex["hooks"]) == {"SessionStart", "SessionEnd"}
    for event, timeout in (("SessionStart", 15), ("SessionEnd", 3)):
        groups = codex["hooks"][event]
        assert len(groups) == 1
        assert groups[0]["matcher"] == ""
        handlers = groups[0]["hooks"]
        assert len(handlers) == 1
        assert handlers[0]["type"] == "command"
        assert handlers[0]["timeout"] == timeout
        assert "AI_MEMORY_HOME" in handlers[0]["command"]
        assert "CLAUDE_MEMORY_HOME" in handlers[0]["command"]
    assert not (ROOT / ".codex" / "hooks.json").exists()


def test_codex_hook_commands_use_no_sync():
    codex = json.loads((ROOT / ".codex" / "hooks.json.example").read_text())

    for event in ("SessionStart", "SessionEnd"):
        command = codex["hooks"][event][0]["hooks"][0]["command"]
        assert "uv run --no-sync --directory" in command


def test_codex_session_end_starts_from_cold_cache_with_no_sync(tmp_path):
    real_uv = shutil.which("uv")
    assert real_uv is not None
    memory_home = tmp_path / "memory"
    fake_bin = tmp_path / "bin"
    _fake_uv(fake_bin)
    cold_cache = tmp_path / "cold-uv-cache"
    payload = {
        "hook_event_name": "SessionEnd",
        "session_id": "cold-cache-session",
        "transcript_path": str(FIXTURES / "codex-basic.jsonl"),
        "cwd": "/projects/cold-cache",
    }
    env = _hook_env(memory_home, fake_bin=fake_bin)
    env["UV_CACHE_DIR"] = str(cold_cache)
    env["UV_OFFLINE"] = "1"

    started = time.monotonic()
    result = subprocess.run(
        [
            real_uv,
            "run",
            "--no-sync",
            "--directory",
            str(ROOT),
            "python",
            str(HOOKS / "codex-session-end.py"),
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=3,
        env=env,
        check=False,
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 0, result.stderr
    assert elapsed < 3
    assert _job_rows(memory_home) == [
        ("codex", "cold-cache-session", "session_end")
    ]


def test_global_setup_only_prints_safe_merge_instructions(tmp_path):
    home = tmp_path / "home"
    claude = home / ".claude"
    codex = home / ".codex"
    claude.mkdir(parents=True)
    codex.mkdir()
    sentinels = {
        claude / "settings.json": (b'{"claude":"sentinel"}\n', 0o600),
        codex / "hooks.json": (b'{"codex":"sentinel"}\n', 0o640),
        claude / "unrelated.bin": (b"\x00claude-unrelated\xff", 0o400),
        codex / "unrelated.txt": (b"codex-unrelated\n", 0o444),
    }
    for path, (content, mode) in sentinels.items():
        path.write_bytes(content)
        path.chmod(mode)
    before = _tree_manifest(home)

    first = _run_global_setup(home)
    after_first = _tree_manifest(home)
    second = _run_global_setup(home)
    after_second = _tree_manifest(home)

    assert after_first == before
    assert after_second == before
    assert first.stdout == second.stdout
    assert f"in {home}/.zshrc" in first.stdout
    assert "AI_MEMORY_HOME" in first.stdout
    assert "~/.claude/settings.json" in first.stdout
    assert "~/.codex/hooks.json" in first.stdout
    assert "codex-cli 0.146.1 or newer" in first.stdout
    assert "uv sync" in first.stdout
    assert "--no-sync" in first.stdout
    assert first.stdout.count("do not replace") == 2


def test_codex_hook_setup_requires_interactive_trust_review(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    before = _tree_manifest(home)
    result = _run_global_setup(home)
    assert _tree_manifest(home) == before
    documents = {
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "AGENTS.md": (ROOT / "AGENTS.md").read_text(encoding="utf-8"),
        "setup output": result.stdout,
    }

    for name, content in documents.items():
        assert "launch Codex interactively" in content, name
        assert "new or changed hook commands and hashes" in content, name
        assert "only the vetted repository hooks" in content, name
        assert "enabled and trusted" in content, name

    assert "--dangerously-bypass-hook-trust" not in documents["README.md"]
    assert "--dangerously-bypass-hook-trust" not in documents["setup output"]
    assert "DANGEROUS" in documents["AGENTS.md"]
    assert "disposable Gate 2" in documents["AGENTS.md"]
    assert "Never persist" in documents["AGENTS.md"]


def test_codex_0146_help_marks_hook_trust_bypass_as_dangerous():
    _assert_codex_hook_trust_help(CODEX_0146_HELP)


def test_hook_trust_warning_must_be_adjacent_to_its_option():
    help_text = """\
Options:
      --dangerously-bypass-approvals-and-sandbox
          Skip confirmations. EXTREMELY DANGEROUS.

      --dangerously-bypass-hook-trust
          Run enabled hooks without persisted trust.
          Intended only for automation that already vets hook sources.

  -C, --cd <DIR>
          Set the working root.
"""

    with pytest.raises(AssertionError):
        _assert_codex_hook_trust_help(help_text)
