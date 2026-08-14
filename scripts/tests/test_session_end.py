"""Tests for the SessionEnd transcript extractor.

`extract_conversation_context` lives in hooks/session-end.py, whose filename has
a hyphen and so cannot be imported normally. Load it by path.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from transcripts import parse_claude_transcript, render_turns

HOOK_PATH = Path(__file__).resolve().parents[2] / "hooks" / "session-end.py"
PRE_COMPACT_PATH = Path(__file__).resolve().parents[2] / "hooks" / "pre-compact.py"


def _load_hook_module():
    # The module has a top-level recursion guard that calls sys.exit(0) when
    # CLAUDE_INVOKED_BY is set. Clear it so import succeeds under pytest.
    os.environ.pop("CLAUDE_INVOKED_BY", None)
    spec = importlib.util.spec_from_file_location("session_end_hook", HOOK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_pre_compact_module():
    os.environ.pop("CLAUDE_INVOKED_BY", None)
    spec = importlib.util.spec_from_file_location("pre_compact_hook", PRE_COMPACT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def hook():
    return _load_hook_module()


@pytest.fixture
def fixture_dir():
    return Path(__file__).resolve().parent / "fixtures"


def test_claude_fixture_preserves_plain_turns(hook, fixture_dir):
    context, count = hook.extract_conversation_context(
        fixture_dir / "transcripts/claude-basic.jsonl"
    )
    assert "Plan the confirmation card" in context
    assert "Proceeding to tickets" in context
    assert count == 2


def test_claude_fixture_preserves_decisions_and_findings(hook, fixture_dir):
    context, count = hook.extract_conversation_context(
        fixture_dir / "transcripts/claude-decisions.jsonl"
    )
    assert "[Decision requested]" in context
    assert "Resolved state" in context
    decision_result = next(
        turn for turn in context.split("\n\n") if "[Decision made]" in turn
    )
    assert decision_result.startswith(
        "**User:** [Decision made] Your questions have been answered"
    )
    assert "On server resolution, Pill" in decision_result
    assert "[Subagent result]" in context
    assert "SUBAGENT_FINDING" in context
    assert "TASK_FINDING" in context
    assert "SHOULD_NOT_APPEAR" not in context
    assert "Async agent launched" not in context
    assert "agentId" not in context
    assert count == 5


def test_session_end_render_is_shared_parser_equivalent(hook, fixture_dir):
    transcript = fixture_dir / "transcripts/claude-decisions.jsonl"
    context, count = hook.extract_conversation_context(transcript)
    normalized = parse_claude_transcript(
        transcript,
        {},
        limits={
            "max_turns": hook.MAX_TURNS,
            "max_chars": hook.MAX_CONTEXT_CHARS,
        },
    )
    assert context == render_turns(normalized)
    assert count == len(normalized.turns)


def test_pre_compact_uses_same_high_signal_normalization(fixture_dir):
    hook = _load_pre_compact_module()
    transcript = fixture_dir / "transcripts/claude-decisions.jsonl"
    context, count = hook.extract_conversation_context(transcript)

    assert "[Decision requested]" in context
    assert "[Decision made]" in context
    assert "[Subagent result]" in context
    assert "SHOULD_NOT_APPEAR" not in context
    assert count == 5


def test_session_end_preserves_legacy_tail_for_single_oversize_turn(hook, tmp_path):
    transcript = tmp_path / "oversize.jsonl"
    transcript.write_text(
        json.dumps({"message": {"role": "user", "content": "x" * 16_000}}),
        encoding="utf-8",
    )

    context, count = hook.extract_conversation_context(transcript)

    assert context == "x" * (hook.MAX_CONTEXT_CHARS - 1) + "\n"
    assert count == 1
