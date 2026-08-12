"""Subprocess contract tests for the fast Claude and Codex hook adapters."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / "hooks"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "transcripts"
FIXED_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


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
    assert not (memory_home / "scripts" / "jobs.sqlite3").exists()
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
    assert not (memory_home / "scripts" / "jobs.sqlite3").exists()
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
    assert not (memory_home / "scripts" / "jobs.sqlite3").exists()
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
    monkeypatch.delattr(hook.os, "fchmod")

    with hook.bounded_transcript_slice(
        source,
        lambda path: hook.parse_claude_transcript(
            path, {"trigger": "session_end"}, limits={"max_turns": 30}
        ),
        source_agent="claude",
        deadline=10.0,
        clock=lambda: 0.0,
    ) as selected:
        private_slice, preview = selected
        assert private_slice.exists()
        assert preview.turns[0].text == "WINDOWS_SIGNAL"

    assert not private_slice.exists()


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
    hook_tmp = tmp_path / "hook-tmp"
    hook_tmp.mkdir()
    hook = _load_hook("session-end.py")
    ticks = iter([0.0, 0.1, 0.2, 2.5])
    monkeypatch.setenv("AI_MEMORY_HOME", str(memory_home))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    monkeypatch.setattr(hook.tempfile, "tempdir", str(hook_tmp))
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
    assert list(hook_tmp.iterdir()) == []


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
    assert not (memory_home / "scripts" / "jobs.sqlite3").exists()
    assert not (memory_home / "scripts" / "spool").exists()
    assert list(hook_tmp.iterdir()) == []


def _load_hook(name: str):
    path = HOOKS / name
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert codex["hooks"]["SessionEnd"][0]["hooks"][0]["timeout"] == 3
    assert not (ROOT / ".codex" / "hooks.json").exists()


def test_global_setup_only_prints_safe_merge_instructions():
    result = subprocess.run(
        ["bash", str(ROOT / "bin" / "setup-global.sh")],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "AI_MEMORY_HOME" in result.stdout
    assert "~/.claude/settings.json" in result.stdout
    assert "~/.codex/hooks.json" in result.stdout
    assert "codex-cli 0.146.1 or newer" in result.stdout
    assert result.stdout.count("do not replace") == 2
