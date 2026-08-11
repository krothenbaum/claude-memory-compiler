"""Tests for the SessionEnd transcript extractor.

`extract_conversation_context` lives in hooks/session-end.py, whose filename has
a hyphen and so cannot be imported normally. Load it by path.
"""

import importlib.util
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
    context, _ = hook.extract_conversation_context(
        fixture_dir / "transcripts/claude-decisions.jsonl"
    )
    assert "[Decision requested]" in context
    assert "Resolved state" in context
    assert "[Decision made]" in context
    assert "On server resolution, Pill" in context
    assert "[Subagent result]" in context
    assert "SUBAGENT_FINDING" in context
    assert "TASK_FINDING" in context
    assert "SHOULD_NOT_APPEAR" not in context
    assert "Async agent launched" not in context
    assert "agentId" not in context
