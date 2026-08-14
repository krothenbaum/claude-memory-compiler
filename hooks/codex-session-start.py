"""Codex SessionStart adapter over the shared local retrieval builder."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys


if os.environ.get("AI_MEMORY_INTERNAL_JOB") == "1" or "CLAUDE_INVOKED_BY" in os.environ:
    sys.exit(0)


def _shared_retrieval():
    path = Path(__file__).with_name("session-start.py")
    spec = importlib.util.spec_from_file_location("ai_memory_session_start", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load shared retrieval adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    try:
        value = json.loads(sys.stdin.read())
        if not isinstance(value, dict):
            raise ValueError("hook input must be a JSON object")
        shared = _shared_retrieval()
        project_key = shared.get_project_key(value)
        context = shared.build_context(project_key)
    except (json.JSONDecodeError, ValueError, OSError, ImportError):
        return

    # Codex 0.146.1 consumes the same command-hook SessionStart envelope.
    print(json.dumps(shared.session_start_output(context)))


if __name__ == "__main__":
    main()
