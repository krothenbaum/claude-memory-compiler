"""Tests for the SessionEnd transcript extractor.

`extract_conversation_context` lives in hooks/session-end.py, whose filename has
a hyphen and so cannot be imported normally. Load it by path.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

HOOK_PATH = Path(__file__).resolve().parents[2] / "hooks" / "session-end.py"


def _load_hook_module():
    # The module has a top-level recursion guard that calls sys.exit(0) when
    # CLAUDE_INVOKED_BY is set. Clear it so import succeeds under pytest.
    os.environ.pop("CLAUDE_INVOKED_BY", None)
    spec = importlib.util.spec_from_file_location("session_end_hook", HOOK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def hook():
    return _load_hook_module()


def _write_transcript(path: Path, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _tool_heavy_entries() -> list[dict]:
    """A transcript whose substance lives in tools: a subagent result and an
    AskUserQuestion decision, plus routine Read noise that must be dropped."""
    return [
        {"message": {"role": "user", "content": "Plan the confirmation card."}},
        {"message": {"role": "assistant", "content": [
            {"type": "text", "text": "Let me research Fabric."},
            {"type": "tool_use", "id": "t1", "name": "Agent",
             "input": {"description": "Research Fabric", "subagent_type": "Explore",
                       "prompt": "..."}},
        ]}},
        {"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": "SUBAGENT_FINDING: Fabric Avatar is already a rounded "
                        "square in Encore."},
        ]}},
        {"message": {"role": "assistant", "content": [
            {"type": "text", "text": "Here are the options."},
            {"type": "tool_use", "id": "t2", "name": "AskUserQuestion",
             "input": {"questions": [
                 {"question": "When should the card switch to the Confirmed state?",
                  "header": "Resolved state",
                  "multiSelect": False,
                  "options": [
                      {"label": "On server resolution, Pill", "description": "..."},
                      {"label": "Optimistic on click", "description": "..."},
                  ]},
             ]}},
        ]}},
        {"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t2",
             "content": 'Your questions have been answered: "When should the card '
                        'switch to the Confirmed state?"="On server resolution, '
                        'Pill". You can now continue.'},
        ]}},
        # Routine Read whose large result must NOT leak into the flush context.
        {"message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t3", "name": "Read",
             "input": {"file_path": "/x"}},
        ]}},
        {"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t3",
             "content": "line1 SHOULD_NOT_APPEAR big file dump"},
        ]}},
        {"message": {"role": "assistant", "content": [
            {"type": "text", "text": "Great, proceeding to tickets."},
        ]}},
    ]


def test_decisions_and_subagent_results_survive(hook, tmp_path):
    t = tmp_path / "transcript.jsonl"
    _write_transcript(t, _tool_heavy_entries())

    context, turn_count = hook.extract_conversation_context(t)

    # The AskUserQuestion decision (question header + chosen answer) is preserved.
    assert "Resolved state" in context
    assert "On server resolution, Pill" in context
    # The subagent's finding is preserved.
    assert "SUBAGENT_FINDING" in context
    # Plain conversation text still works.
    assert "Plan the confirmation card." in context
    assert "Great, proceeding to tickets." in context
    # Routine tool output (a Read result) is excluded.
    assert "SHOULD_NOT_APPEAR" not in context
    # Turn count includes the tool-bearing turns.
    assert turn_count >= 5


def test_routine_only_tool_output_is_dropped(hook, tmp_path):
    entries = [
        {"message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "r1", "name": "Bash",
             "input": {"command": "ls"}},
        ]}},
        {"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "r1",
             "content": "file dump NOPE"},
        ]}},
    ]
    t = tmp_path / "routine.jsonl"
    _write_transcript(t, entries)

    context, turn_count = hook.extract_conversation_context(t)

    assert "NOPE" not in context
    assert turn_count == 0


def test_async_agent_launch_metadata_is_dropped(hook, tmp_path):
    """The async 'Agent launched' tool_result is bookkeeping, not a finding."""
    entries = [
        {"message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "a1", "name": "Agent",
             "input": {"description": "x", "subagent_type": "Explore"}},
        ]}},
        {"message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a1",
             "content": "Async agent launched successfully. agentId: abc123 "
                        "internal metadata do not quote"},
        ]}},
    ]
    t = tmp_path / "async.jsonl"
    _write_transcript(t, entries)

    context, _ = hook.extract_conversation_context(t)

    assert "Async agent launched" not in context
    assert "agentId" not in context
