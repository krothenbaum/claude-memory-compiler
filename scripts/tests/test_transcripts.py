"""Provider-neutral transcript normalization tests."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from transcripts import (
    NormalizedSession,
    Turn,
    chunk_session,
    parse_claude_transcript,
    parse_codex_transcript,
    render_turns,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "transcripts"
CLAUDE_METADATA = {
    "session_id": "claude-session",
    "cwd": "/workspaces/card-app",
    "timestamp": "2026-08-10T15:00:00Z",
    "trigger": "session_end",
}


def test_normalized_session_is_immutable_and_uses_tuple_turns():
    session = parse_claude_transcript(
        FIXTURES / "claude-basic.jsonl", CLAUDE_METADATA
    )

    assert isinstance(session, NormalizedSession)
    assert isinstance(session.turns, tuple)
    assert session.turns == (
        Turn("user", "Plan the confirmation card"),
        Turn("assistant", "Proceeding to tickets"),
    )
    with pytest.raises(FrozenInstanceError):
        session.project = "different"  # type: ignore[misc]


def test_source_hash_is_canonical_and_ignores_delivery_metadata(tmp_path):
    source = FIXTURES / "claude-basic.jsonl"
    copied_source = tmp_path / "copied.jsonl"
    copied_source.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    pre_compact = parse_claude_transcript(
        source,
        {**CLAUDE_METADATA, "trigger": "pre_compact", "timestamp": "2026-08-10T15:01:00Z"},
    )
    session_end = parse_claude_transcript(
        copied_source,
        {**CLAUDE_METADATA, "trigger": "session_end", "timestamp": "2026-08-10T16:00:00Z"},
    )

    canonical = {
        "agent": "claude",
        "cwd": "/workspaces/card-app",
        "project": "card-app",
        "session_id": "claude-session",
        "turns": [
            {"kind": "message", "role": "user", "text": "Plan the confirmation card", "timestamp": None},
            {"kind": "message", "role": "assistant", "text": "Proceeding to tickets", "timestamp": None},
        ],
    }
    expected = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert pre_compact.source_hash == expected
    assert session_end.source_hash == expected
    assert pre_compact.trigger != session_end.trigger
    assert pre_compact.timestamp != session_end.timestamp
    assert pre_compact.source_path != session_end.source_path

    changed = tmp_path / "changed.jsonl"
    changed.write_text(
        source.read_text(encoding="utf-8").replace("Proceeding to tickets", "Changed"),
        encoding="utf-8",
    )
    assert parse_claude_transcript(changed, CLAUDE_METADATA).source_hash != expected


@pytest.mark.parametrize(
    ("metadata", "expected_project"),
    [
        ({"project": "explicit", "cwd": "/workspaces/card-app"}, "explicit"),
        ({"cwd": "/workspaces/card-app"}, "card-app"),
        ({"cwd": "/"}, "unknown"),
        ({}, "unknown"),
    ],
)
def test_project_fallback(metadata, expected_project):
    session = parse_claude_transcript(
        FIXTURES / "claude-basic.jsonl", metadata
    )
    assert session.project == expected_project


def test_timestamp_uses_stable_transcript_value(tmp_path):
    transcript = tmp_path / "timestamped.jsonl"
    transcript.write_text(
        '\n'.join(
            [
                '{"timestamp":"2026-08-10T15:00:00Z","message":{"role":"user","content":"First"}}',
                '{"timestamp":"2026-08-10T15:00:01Z","message":{"role":"assistant","content":"Second"}}',
            ]
        ),
        encoding="utf-8",
    )

    first = parse_claude_transcript(transcript, {})
    second = parse_claude_transcript(transcript, {})
    assert first.timestamp == second.timestamp == "2026-08-10T15:00:00Z"


def test_claude_basic_fixture_renders_exact_durable_context():
    session = parse_claude_transcript(
        FIXTURES / "claude-basic.jsonl", CLAUDE_METADATA
    )
    assert render_turns(session) == (
        "**User:** Plan the confirmation card\n\n"
        "**Assistant:** Proceeding to tickets\n"
    )


def test_claude_decision_fixture_renders_exact_durable_context():
    session = parse_claude_transcript(
        FIXTURES / "claude-decisions.jsonl", CLAUDE_METADATA
    )
    assert render_turns(session) == (
        "**Assistant:** Let me research Fabric.\n\n"
        "**User:** [Subagent result] SUBAGENT_FINDING: Fabric Avatar is already a rounded square in Encore.\n\n"
        "**User:** [Subagent result] TASK_FINDING: The confirmation card must wait for server resolution.\n\n"
        "**Assistant:** Here are the options.\n"
        "[Decision requested]\n"
        "- (Resolved state) When should the card switch to the Confirmed state? | options: On server resolution, Pill, Optimistic on click\n\n"
        "**User:** [Decision made] Your questions have been answered: \"When should the card switch to the Confirmed state?\"=\"On server resolution, Pill\". You can now continue.\n"
    )
    assert [turn.kind for turn in session.turns] == [
        "message",
        "subagent_finding",
        "subagent_finding",
        "decision",
        "decision",
    ]


def test_claude_live_limits_keep_recent_complete_turns():
    session = parse_claude_transcript(
        FIXTURES / "claude-decisions.jsonl",
        CLAUDE_METADATA,
        limits={"max_turns": 2, "max_chars": 1_000},
    )
    assert len(session.turns) == 2
    assert "Here are the options" in session.turns[0].text
    assert "Decision made" in session.turns[1].text


@pytest.mark.parametrize("max_chars", [0, 1, 5, len("**User:** \n")])
def test_max_chars_drops_turn_when_no_text_can_fit(tmp_path, max_chars):
    transcript = tmp_path / "one-turn.jsonl"
    transcript.write_text(
        json.dumps({"message": {"role": "user", "content": "abcdef"}}),
        encoding="utf-8",
    )

    session = parse_claude_transcript(
        transcript, {}, limits={"max_chars": max_chars}
    )

    assert session.turns == ()
    assert render_turns(session) == ""
    assert len(render_turns(session)) <= max_chars


@pytest.mark.parametrize(
    "max_chars", [len("**User:** \n") + 1, len("**User:** \n") + 5, 100]
)
def test_max_chars_is_a_hard_rendered_bound_when_text_fits(tmp_path, max_chars):
    transcript = tmp_path / "one-turn.jsonl"
    transcript.write_text(
        json.dumps({"message": {"role": "user", "content": "abcdef"}}),
        encoding="utf-8",
    )

    session = parse_claude_transcript(
        transcript, {}, limits={"max_chars": max_chars}
    )

    assert session.turns
    assert all(turn.text for turn in session.turns)
    assert len(render_turns(session)) <= max_chars


def test_codex_parser_keeps_user_and_assistant_text():
    session = parse_codex_transcript(FIXTURES / "codex-basic.jsonl", {})
    assert session.session_id == "codex-basic-session"
    assert session.cwd == "/workspaces/card-app"
    assert session.project == "card-app"
    assert session.timestamp == "2026-08-10T16:00:00Z"
    assert session.turns == (
        Turn("user", "Plan the confirmation card", timestamp="2026-08-10T16:00:02Z"),
        Turn("assistant", "Proceeding to tickets", timestamp="2026-08-10T16:00:03Z"),
    )


def test_codex_parser_excludes_developer_instructions():
    rendered = render_turns(
        parse_codex_transcript(FIXTURES / "codex-basic.jsonl", {})
    )
    assert "DEVELOPER_INSTRUCTIONS_SHOULD_NOT_APPEAR" not in rendered
    assert "SESSION_INSTRUCTIONS_SHOULD_NOT_APPEAR" not in rendered


def test_codex_parser_excludes_reasoning():
    rendered = render_turns(
        parse_codex_transcript(FIXTURES / "codex-basic.jsonl", {})
    )
    assert "HIDDEN_RESPONSE_REASONING_SHOULD_NOT_APPEAR" not in rendered
    assert "HIDDEN_EVENT_REASONING_SHOULD_NOT_APPEAR" not in rendered
    assert "SECRET_REASONING_TOKEN" not in rendered


def test_codex_parser_excludes_routine_tool_output():
    rendered = render_turns(
        parse_codex_transcript(FIXTURES / "codex-basic.jsonl", {})
    )
    assert "ROUTINE_TOOL_OUTPUT_SHOULD_NOT_APPEAR" not in rendered
    assert "UNKNOWN_TOOL_OUTPUT_SHOULD_NOT_APPEAR" not in rendered
    assert "TURN_CONTEXT_SHOULD_NOT_APPEAR" not in rendered
    assert "cat secrets.txt" not in rendered


def test_codex_parser_does_not_duplicate_event_messages():
    rendered = render_turns(
        parse_codex_transcript(FIXTURES / "codex-basic.jsonl", {})
    )
    assert rendered.count("Proceeding to tickets") == 1


def test_codex_parser_keeps_decisions_and_selected_subagent_findings():
    session = parse_codex_transcript(FIXTURES / "codex-decisions.jsonl", {})
    rendered = render_turns(session)
    assert "[Decision made] Rollout: Staged rollout" in rendered
    assert "[Subagent result] SUBAGENT_FINDING" in rendered
    assert "Fabric Avatar is already a rounded square in Encore" in rendered
    assert "agent-123" not in rendered
    assert "Wait completed" not in rendered
    assert "PROGRESS_UPDATE_SHOULD_NOT_APPEAR" not in rendered
    assert "ENCRYPTED_COLLAB_CONTENT_SHOULD_NOT_APPEAR" not in rendered
    assert "final-turn" not in rendered
    assert "ROUTINE_COLLAB_ACK_SHOULD_NOT_APPEAR" not in rendered
    assert rendered.count("SUBAGENT_FINDING") == 1
    assert rendered.count("Applied the selected rollout.") == 1
    assert [turn.kind for turn in session.turns] == [
        "decision",
        "subagent_finding",
        "message",
    ]


@pytest.mark.parametrize(
    "output",
    [
        {"error": "cancelled"},
        {"status": "cancelled"},
        {"message": "No answer was selected."},
        {"answers": {}},
        {"answers": {"Rollout": {"answers": []}}},
        {"answers": {"Rollout": {"answers": [False]}}},
    ],
)
def test_codex_parser_excludes_non_choice_request_user_input_outputs(
    tmp_path, output
):
    transcript = tmp_path / "non-choice.jsonl"
    records = [
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "request_user_input",
                "call_id": "choice-1",
                "arguments": "{}",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "choice-1",
                "output": json.dumps(output),
            },
        },
    ]
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    session = parse_codex_transcript(transcript, {})

    assert session.turns == ()
    assert render_turns(session) == ""


def test_chunk_session_preserves_turns_and_deterministic_provenance():
    session = parse_claude_transcript(
        FIXTURES / "claude-decisions.jsonl", CLAUDE_METADATA
    )
    chunks = chunk_session(session, target_chars=140)

    assert len(chunks) > 1
    assert tuple(turn for chunk in chunks for turn in chunk.turns) == session.turns
    assert all(chunk.source_path == session.source_path for chunk in chunks)
    assert all(chunk.trigger == session.trigger for chunk in chunks)
    assert [chunk.source_hash for chunk in chunks] == [
        chunk.source_hash for chunk in chunk_session(session, target_chars=140)
    ]


def test_chunk_session_keeps_assistant_reply_with_oversize_user_turn():
    session = parse_claude_transcript(
        FIXTURES / "claude-basic.jsonl", CLAUDE_METADATA
    )
    oversized = replace(
        session,
        turns=(
            Turn("user", "x" * 200),
            Turn("assistant", "done"),
            Turn("user", "next topic"),
        ),
    )
    chunks = chunk_session(oversized, target_chars=50)
    assert chunks[0].turns == (
        Turn("user", "x" * 200),
        Turn("assistant", "done"),
    )
    assert chunks[1].turns == (Turn("user", "next topic"),)
    assert tuple(turn for chunk in chunks for turn in chunk.turns) == oversized.turns


def test_batch_parser_delegates_full_claude_normalization(monkeypatch):
    batch_path = Path(__file__).resolve().parents[1] / "batch-flush.py"
    spec = importlib.util.spec_from_file_location("batch_flush", batch_path)
    batch_flush = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, batch_flush)
    monkeypatch.setenv("CLAUDE_INVOKED_BY", os.environ.get("CLAUDE_INVOKED_BY", "test"))
    spec.loader.exec_module(batch_flush)

    turns = batch_flush.extract_full_conversation(
        FIXTURES / "claude-decisions.jsonl"
    )
    rendered = "\n".join(turn.text for turn in turns)
    assert "[Decision requested]" in rendered
    assert "[Decision made]" in rendered
    assert "[Subagent result]" in rendered
    assert "SHOULD_NOT_APPEAR" not in rendered
    assert len(turns) == 5
