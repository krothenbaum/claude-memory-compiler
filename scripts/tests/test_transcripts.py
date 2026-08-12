"""Provider-neutral transcript normalization tests."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref

import pytest
import transcripts as transcripts_module

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


def test_claude_embedded_metadata_is_stable_across_renamed_copies(tmp_path):
    records = [
        {
            "sessionId": None,
            "session_id": 7,
            "cwd": [],
            "timestamp": 99,
            "message": {"role": "user", "content": "Before metadata"},
        },
        {
            "sessionId": "embedded-session",
            "cwd": "/workspaces/card-app",
            "timestamp": "2026-08-10T15:00:00Z",
            "message": {"role": "user", "content": "Embedded metadata"},
        },
        {
            "session_id": "later-conflict",
            "cwd": "/workspaces/wrong-project",
            "timestamp": "2026-08-10T16:00:00Z",
            "message": {"role": "assistant", "content": "Later turn"},
        },
    ]
    serialized = "\n".join(json.dumps(record) for record in records)
    first_path = tmp_path / "renamed-one.jsonl"
    second_path = tmp_path / "renamed-two.jsonl"
    first_path.write_text(serialized, encoding="utf-8")
    second_path.write_text(serialized, encoding="utf-8")

    first = parse_claude_transcript(first_path, {})
    second = parse_claude_transcript(second_path, {})

    assert first.session_id == second.session_id == "embedded-session"
    assert first.cwd == second.cwd == "/workspaces/card-app"
    assert first.project == second.project == "card-app"
    assert first.timestamp == second.timestamp == "2026-08-10T15:00:00Z"
    assert first.source_path != second.source_path
    assert first.source_hash == second.source_hash


def test_claude_caller_metadata_overrides_embedded_metadata(tmp_path):
    transcript = tmp_path / "embedded.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "sessionId": "embedded-session",
                "cwd": "/workspaces/embedded",
                "timestamp": "2026-08-10T15:00:00Z",
                "message": {"role": "user", "content": "Hello"},
            }
        ),
        encoding="utf-8",
    )

    session = parse_claude_transcript(
        transcript,
        {
            "session_id": "caller-session",
            "cwd": "/workspaces/caller",
            "timestamp": "2026-08-11T15:00:00Z",
        },
    )

    assert session.session_id == "caller-session"
    assert session.cwd == "/workspaces/caller"
    assert session.project == "caller"
    assert session.timestamp == "2026-08-11T15:00:00Z"


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


def test_claude_malformed_questions_and_options_do_not_abort_later_turns(tmp_path):
    transcript = tmp_path / "malformed-questions.jsonl"
    records = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "null-questions",
                        "name": "AskUserQuestion",
                        "input": {"questions": None},
                    }
                ],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "scalar-questions",
                        "name": "AskUserQuestion",
                        "input": {"questions": 7},
                    }
                ],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "mixed-questions",
                        "name": "AskUserQuestion",
                        "input": {
                            "questions": [
                                None,
                                "bad question",
                                {
                                    "question": "Valid without options?",
                                    "options": None,
                                },
                                {"question": "", "options": []},
                                {
                                    "question": "Good mixed options?",
                                    "options": [
                                        None,
                                        3,
                                        {"label": "Yes"},
                                        {"label": None},
                                    ],
                                },
                            ]
                        },
                    }
                ],
            }
        },
        {"message": {"role": "user", "content": "AFTER_MALFORMED"}},
    ]
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    try:
        session = parse_claude_transcript(transcript, {})
    except (TypeError, AttributeError) as error:
        pytest.fail(f"malformed question data aborted parsing: {error}")

    rendered = render_turns(session)
    assert "Valid without options?" in rendered
    assert "Good mixed options? | options: Yes" in rendered
    assert "bad question" not in rendered
    assert "AFTER_MALFORMED" in rendered


@pytest.mark.parametrize(
    "tool_input",
    [
        None,
        {},
        {"questions": None},
        {"questions": []},
        {
            "questions": [
                None,
                "bad",
                {},
                {"question": ""},
                {"question": "   ", "options": []},
            ]
        },
    ],
)
def test_claude_empty_decision_requests_emit_no_signal(tmp_path, tool_input):
    transcript = tmp_path / "empty-decision.jsonl"
    records = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "empty-decision",
                        "name": "AskUserQuestion",
                        "input": tool_input,
                    }
                ],
            }
        },
        {"message": {"role": "user", "content": "VALID_LATER_TURN"}},
    ]
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    session = parse_claude_transcript(transcript, {})
    rendered = render_turns(session)

    assert session.turns == (Turn("user", "VALID_LATER_TURN"),)
    assert "[Decision requested]" not in rendered


def _claude_tool_use(name, call_id="reused-id"):
    block = {"type": "tool_use", "id": call_id, "name": name, "input": {}}
    if name == "AskUserQuestion":
        block["input"] = {
            "questions": [{"question": "Choose?", "options": [{"label": "A"}]}]
        }
    return block


@pytest.mark.parametrize(
    ("calls", "output_before"),
    [
        (("Read", "AskUserQuestion"), False),
        (("AskUserQuestion", "Read"), False),
        (("AskUserQuestion", "AskUserQuestion"), False),
        (("Read", "Read"), False),
        (("AskUserQuestion",), True),
    ],
    ids=[
        "routine-then-decision",
        "decision-then-routine",
        "same-decision",
        "same-routine",
        "output-before-call",
    ],
)
def test_claude_reused_or_unknown_tool_ids_exclude_outputs(
    tmp_path, calls, output_before
):
    transcript = tmp_path / "claude-reused-call-id.jsonl"
    output_record = {
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "reused-id",
                    "content": "AMBIGUOUS_RESULT_SHOULD_NOT_APPEAR",
                }
            ],
        }
    }
    call_record = {
        "message": {
            "role": "assistant",
            "content": [_claude_tool_use(name) for name in calls],
        }
    }
    records = (
        [output_record, call_record] if output_before else [call_record, output_record]
    )
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    rendered = render_turns(parse_claude_transcript(transcript, {}))

    assert "AMBIGUOUS_RESULT_SHOULD_NOT_APPEAR" not in rendered
    assert "[Decision made]" not in rendered


def test_claude_later_reuse_invalidates_an_earlier_tool_output(tmp_path):
    transcript = tmp_path / "claude-late-reused-call-id.jsonl"
    records = [
        {
            "message": {
                "role": "assistant",
                "content": [_claude_tool_use("AskUserQuestion")],
            }
        },
        {
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "reused-id",
                        "content": "EARLY_AMBIGUOUS_RESULT_SHOULD_NOT_APPEAR",
                    }
                ],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": [_claude_tool_use("Read")],
            }
        },
    ]
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    rendered = render_turns(parse_claude_transcript(transcript, {}))

    assert "EARLY_AMBIGUOUS_RESULT_SHOULD_NOT_APPEAR" not in rendered
    assert "[Decision made]" not in rendered


@pytest.mark.parametrize(
    ("tool_name", "first_output", "replayed_output", "expected_kind"),
    [
        (
            "AskUserQuestion",
            "FIRST_DECISION",
            "REPLAYED_DECISION_SHOULD_NOT_APPEAR",
            "decision",
        ),
        (
            "Agent",
            "FIRST_FINDING",
            "REPLAYED_FINDING_SHOULD_NOT_APPEAR",
            "subagent_finding",
        ),
    ],
)
def test_claude_accepts_only_first_durable_output_per_call(
    tmp_path, tool_name, first_output, replayed_output, expected_kind
):
    transcript = tmp_path / "claude-replayed-output.jsonl"
    records = [
        {
            "message": {
                "role": "assistant",
                "content": [_claude_tool_use(tool_name, call_id="unique-id")],
            }
        },
        {
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "unique-id",
                        "content": first_output,
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
                        "tool_use_id": "unique-id",
                        "content": replayed_output,
                    }
                ],
            }
        },
    ]
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    session = parse_claude_transcript(transcript, {})
    rendered = render_turns(session)

    durable_outputs = [
        turn
        for turn in session.turns
        if first_output in turn.text or replayed_output in turn.text
    ]
    assert len(durable_outputs) == 1
    assert durable_outputs[0].kind == expected_kind
    assert first_output in rendered
    assert replayed_output not in rendered


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


@pytest.mark.parametrize("agent", ["claude", "codex"])
def test_positive_max_turns_bounds_live_turn_accumulation(
    tmp_path, monkeypatch, agent
):
    transcript = tmp_path / f"many-{agent}-turns.jsonl"
    records = []
    for index in range(100):
        role = "user" if index % 2 == 0 else "assistant"
        text = f"turn-{index}"
        if agent == "claude":
            records.append({"message": {"role": role, "content": text}})
        else:
            block_type = "input_text" if role == "user" else "output_text"
            records.append(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": role,
                        "content": [{"type": block_type, "text": text}],
                    },
                }
            )
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    parser = parse_claude_transcript if agent == "claude" else parse_codex_transcript
    full = parser(transcript, {})

    original_turn = Turn
    counts = {"live": 0, "peak": 0}

    def tracking_turn(*args, **kwargs):
        turn = original_turn(*args, **kwargs)
        counts["live"] += 1
        counts["peak"] = max(counts["peak"], counts["live"])

        def released():
            counts["live"] -= 1

        weakref.finalize(turn, released)
        return turn

    monkeypatch.setattr(transcripts_module, "Turn", tracking_turn)

    limited = parser(transcript, {}, limits={"max_turns": 3})

    assert limited.turns == full.turns[-3:]
    assert counts["peak"] <= 4


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


def _parse_collaboration_envelope(tmp_path, sender_lines):
    transcript = tmp_path / "collaboration-envelope.jsonl"
    envelope_lines = [
        "Message Type: FINAL_ANSWER",
        "Task name: /root",
        *sender_lines,
        "Payload:",
        "EXACT_SENDER_FINDING",
    ]
    record = {
        "type": "response_item",
        "payload": {
            "type": "agent_message",
            "id": "collaboration-result",
            "author": "/root/research",
            "recipient": "/root",
            "content": [
                {"type": "input_text", "text": "\n".join(envelope_lines)}
            ],
        },
    }
    transcript.write_text(json.dumps(record), encoding="utf-8")
    return parse_codex_transcript(transcript, {})


def test_codex_collaboration_sender_exactly_matches_author(tmp_path):
    session = _parse_collaboration_envelope(
        tmp_path, ["Sender: /root/research"]
    )

    assert session.turns == (
        Turn(
            "assistant",
            "[Subagent result] EXACT_SENDER_FINDING",
            "subagent_finding",
        ),
    )


@pytest.mark.parametrize(
    "sender_lines",
    [
        ["Sender: /evil/root/research"],
        ["Sender: /root/research-evil"],
        ["Sender: /root/research", "Sender: /root/research"],
        [],
    ],
    ids=["prefix", "suffix", "duplicate", "missing"],
)
def test_codex_collaboration_rejects_inexact_sender_fields(
    tmp_path, sender_lines
):
    session = _parse_collaboration_envelope(tmp_path, sender_lines)

    assert session.turns == ()
    assert render_turns(session) == ""


@pytest.mark.parametrize(
    "header_lines",
    [
        ["Message Type: FINAL_ANSWER", "Sender: /root/research"],
        [
            "Message Type: FINAL_ANSWER",
            "Task name: /root/other",
            "Sender: /root/research",
        ],
        [
            "Message Type: FINAL_ANSWER",
            "Task name: /root",
            "Sender: /root/research",
            "Unexpected: value",
        ],
        [
            "Message Type: FINAL_ANSWER",
            "Sender: /root/research",
            "Task name: /root",
        ],
    ],
    ids=["missing-task", "mismatched-task", "extra-field", "wrong-order"],
)
def test_codex_collaboration_requires_exact_canonical_header(
    tmp_path, header_lines
):
    transcript = tmp_path / "invalid-collaboration-header.jsonl"
    record = {
        "type": "response_item",
        "payload": {
            "type": "agent_message",
            "author": "/root/research",
            "recipient": "/root",
            "content": [
                {
                    "type": "input_text",
                    "text": "\n".join(
                        [*header_lines, "Payload:", "SHOULD_NOT_APPEAR"]
                    ),
                }
            ],
        },
    }
    transcript.write_text(json.dumps(record), encoding="utf-8")

    session = parse_codex_transcript(transcript, {})

    assert session.turns == ()
    assert "SHOULD_NOT_APPEAR" not in render_turns(session)


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


@pytest.mark.parametrize(
    ("calls", "output_before"),
    [
        (("exec_command", "request_user_input"), False),
        (("request_user_input", "exec_command"), False),
        (("request_user_input", "request_user_input"), False),
        (("exec_command", "exec_command"), False),
        (("request_user_input",), True),
    ],
    ids=[
        "routine-then-decision",
        "decision-then-routine",
        "same-decision",
        "same-routine",
        "output-before-call",
    ],
)
@pytest.mark.parametrize(
    ("call_type", "output_type"),
    [
        ("function_call", "function_call_output"),
        ("custom_tool_call", "custom_tool_call_output"),
    ],
)
def test_codex_reused_or_unknown_call_ids_exclude_outputs(
    tmp_path, calls, output_before, call_type, output_type
):
    transcript = tmp_path / "codex-reused-call-id.jsonl"
    call_records = [
        {
            "type": "response_item",
            "payload": {
                "type": call_type,
                "name": name,
                "call_id": "reused-id",
                "arguments": "{}",
            },
        }
        for name in calls
    ]
    output_record = {
        "type": "response_item",
        "payload": {
            "type": output_type,
            "call_id": "reused-id",
            "output": json.dumps(
                {"answers": {"Choice": {"answers": ["LEAKED_CHOICE"]}}}
            ),
        },
    }
    records = (
        [output_record, *call_records]
        if output_before
        else [*call_records, output_record]
    )
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    session = parse_codex_transcript(transcript, {})

    assert session.turns == ()
    assert "LEAKED_CHOICE" not in render_turns(session)


@pytest.mark.parametrize(
    ("call_type", "output_type"),
    [
        ("function_call", "function_call_output"),
        ("custom_tool_call", "custom_tool_call_output"),
    ],
)
def test_codex_later_reuse_invalidates_an_earlier_call_output(
    tmp_path, call_type, output_type
):
    transcript = tmp_path / "codex-late-reused-call-id.jsonl"
    records = [
        {
            "type": "response_item",
            "payload": {
                "type": call_type,
                "name": "request_user_input",
                "call_id": "reused-id",
                "arguments": "{}",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": output_type,
                "call_id": "reused-id",
                "output": json.dumps(
                    {"answers": {"Choice": {"answers": ["LEAKED_CHOICE"]}}}
                ),
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": call_type,
                "name": "exec_command",
                "call_id": "reused-id",
                "arguments": "{}",
            },
        },
    ]
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    session = parse_codex_transcript(transcript, {})

    assert session.turns == ()
    assert "LEAKED_CHOICE" not in render_turns(session)


@pytest.mark.parametrize(
    ("tool_name", "first_output", "replayed_output", "expected_kind"),
    [
        (
            "request_user_input",
            {"answers": {"Choice": {"answers": ["FIRST_DECISION"]}}},
            {
                "answers": {
                    "Choice": {"answers": ["REPLAYED_DECISION_SHOULD_NOT_APPEAR"]}
                }
            },
            "decision",
        ),
        (
            "wait_agent",
            {"status": "completed", "message": "FIRST_FINDING"},
            {
                "status": "completed",
                "message": "REPLAYED_FINDING_SHOULD_NOT_APPEAR",
            },
            "subagent_finding",
        ),
    ],
)
@pytest.mark.parametrize(
    ("call_type", "output_type"),
    [
        ("function_call", "function_call_output"),
        ("custom_tool_call", "custom_tool_call_output"),
    ],
)
def test_codex_accepts_only_first_durable_output_per_call(
    tmp_path,
    tool_name,
    first_output,
    replayed_output,
    expected_kind,
    call_type,
    output_type,
):
    transcript = tmp_path / "codex-replayed-output.jsonl"
    records = [
        {
            "type": "response_item",
            "payload": {
                "type": call_type,
                "name": tool_name,
                "call_id": "unique-id",
                "arguments": "{}",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": output_type,
                "call_id": "unique-id",
                "output": json.dumps(first_output),
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": output_type,
                "call_id": "unique-id",
                "output": json.dumps(replayed_output),
            },
        },
    ]
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    session = parse_codex_transcript(transcript, {})
    rendered = render_turns(session)

    assert len(session.turns) == 1
    assert session.turns[0].kind == expected_kind
    assert "FIRST_" in rendered
    assert "REPLAYED_" not in rendered


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


def test_chunk_session_deduplicates_identical_normalized_slices(tmp_path):
    transcript = tmp_path / "repeated.jsonl"
    records = []
    for _ in range(3):
        records.extend(
            [
                {"message": {"role": "user", "content": "same user"}},
                {"message": {"role": "assistant", "content": "same assistant"}},
            ]
        )
    transcript.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    session = parse_claude_transcript(
        transcript,
        {"session_id": "repeated-session", "cwd": "/workspaces/repeated"},
    )
    parent_hash = session.source_hash

    chunks = chunk_session(session, target_chars=1)
    repeated = chunk_session(session, target_chars=1)

    assert len(chunks) == 3
    assert all(chunk.turns == chunks[0].turns for chunk in chunks)
    assert tuple(turn for chunk in chunks for turn in chunk.turns) == session.turns
    assert len({chunk.source_hash for chunk in chunks}) == 1
    assert [chunk.source_hash for chunk in chunks] == [
        chunk.source_hash for chunk in repeated
    ]
    assert session.source_hash == parent_hash


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


def test_batch_flush_imports_from_arbitrary_cwd_without_pythonpath(tmp_path):
    batch_path = Path(__file__).resolve().parents[1] / "batch-flush.py"
    probe = """
import importlib.util
from pathlib import Path
import sys

path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("isolated_batch_flush", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.ROOT == path.parent.parent
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(batch_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
