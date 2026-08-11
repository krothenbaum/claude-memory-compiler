"""
SessionStart hook - injects knowledge base context into every conversation.

This is the "context injection" layer. When Claude Code starts a session,
this hook reads the knowledge base index and recent daily log, then injects
them as additional context so Claude always "remembers" what it has learned.

The injection is scoped to the current project (basename of the session's
working directory) so each session only sees knowledge relevant to the repo
it was opened in, plus anything tagged `global`.

Configure in .claude/settings.json:
{
    "hooks": {
        "SessionStart": [{
            "matcher": "",
            "command": "uv run python hooks/session-start.py"
        }]
    }
}
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Internal provider jobs must not inject memory back into themselves. Keep the
# check before any operation that could inspect or write runtime state.
if os.environ.get("AI_MEMORY_INTERNAL_JOB") == "1" or "CLAUDE_INVOKED_BY" in os.environ:
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent

MAX_CONTEXT_CHARS = 20_000
MAX_LOG_LINES = 60


def resolve_memory_home(memory_home: Path | str | None = None) -> Path:
    """Resolve the retrieval root without mutating configuration or runtime state."""
    if memory_home is not None:
        return Path(memory_home).expanduser()
    canonical = os.environ.get("AI_MEMORY_HOME")
    compatibility = os.environ.get("CLAUDE_MEMORY_HOME")
    if canonical and compatibility:
        if Path(canonical).expanduser().resolve() != Path(compatibility).expanduser().resolve():
            raise ValueError("AI_MEMORY_HOME and CLAUDE_MEMORY_HOME resolve differently")
    configured = canonical or compatibility
    return Path(configured).expanduser() if configured else ROOT


def get_project_key(hook_input: dict[str, object] | None = None) -> str:
    """Detect the current project from the session's working directory."""
    payload_cwd = (hook_input or {}).get("cwd")
    cwd = (
        payload_cwd
        if isinstance(payload_cwd, str) and payload_cwd
        else os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    )
    return Path(cwd).name or "unknown"


def _split_md_row(line: str) -> list[str]:
    """Split a markdown table row into trimmed cell strings."""
    parts = line.split("|")
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return [p.strip() for p in parts]


def _row_matches_project(project_cell: str, project_key: str) -> bool:
    """Match a row's Project column value against the current project_key."""
    cleaned = project_cell.strip().strip("[]")
    projects = [p.strip().strip("'\"") for p in cleaned.split(",") if p.strip()]
    return project_key in projects or "global" in projects


def filter_index(index_content: str, project_key: str) -> str:
    """Keep header/separator rows plus rows where the Project column matches.

    Falls back to the unfiltered index if no Project column exists (legacy index).
    """
    lines = index_content.splitlines()

    header_idx = None
    project_col = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("|"):
            cells = _split_md_row(line)
            if "Project" in cells:
                project_col = cells.index("Project")
                header_idx = i
                break

    if project_col is None:
        return index_content

    out_lines: list[str] = []
    kept_data_rows = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("|"):
            out_lines.append(line)
            continue

        # Always keep header and the separator row immediately after it
        if i == header_idx or i == header_idx + 1:
            out_lines.append(line)
            continue

        cells = _split_md_row(line)
        if project_col >= len(cells):
            continue
        if _row_matches_project(cells[project_col], project_key):
            out_lines.append(line)
            kept_data_rows += 1

    if kept_data_rows == 0:
        out_lines.append("")
        out_lines.append(f"_(no articles tagged with project `{project_key}` or `global` yet)_")

    return "\n".join(out_lines)


def filter_daily_log(log_content: str, project_key: str) -> str:
    """Keep session entries matching project_key (or 'global', or untagged legacy entries)."""
    parts = log_content.split("\n### ")
    if len(parts) <= 1:
        return log_content

    head = parts[0]
    sessions = parts[1:]

    project_prefix = "**Project:**"
    kept: list[str] = []
    for block in sessions:
        proj = None
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith(project_prefix):
                proj = stripped[len(project_prefix):].strip()
                break

        if proj is None:
            # Legacy entry without project metadata — include it
            kept.append(block)
        elif proj == project_key or proj == "global":
            kept.append(block)

    if not kept:
        return f"{head}\n\n_(no recent sessions for project `{project_key}`)_"

    return head + "\n### " + "\n### ".join(kept)


def get_recent_log(
    project_key: str,
    *,
    memory_home: Path | str | None = None,
    now: datetime | None = None,
) -> str:
    """Read the most recent daily log (today or yesterday), filtered by project."""
    today = (now or datetime.now(timezone.utc)).astimezone()
    daily_dir = resolve_memory_home(memory_home) / "daily"

    for offset in range(2):
        date = today - timedelta(days=offset)
        log_path = daily_dir / f"{date.strftime('%Y-%m-%d')}.md"
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8")
            filtered = filter_daily_log(content, project_key)
            lines = filtered.splitlines()
            recent = lines[-MAX_LOG_LINES:] if len(lines) > MAX_LOG_LINES else lines
            return "\n".join(recent)

    return "(no recent daily log)"


def build_context(
    project_key: str,
    *,
    memory_home: Path | str | None = None,
    now: datetime | None = None,
) -> str:
    """Assemble the project-scoped context to inject into the conversation."""
    parts = []

    today = (now or datetime.now(timezone.utc)).astimezone()
    root = resolve_memory_home(memory_home)
    index_file = root / "knowledge" / "index.md"
    parts.append(
        f"## Today\n{today.strftime('%A, %B %d, %Y')}\n\n**Project:** {project_key}"
    )

    if index_file.exists():
        index_content = index_file.read_text(encoding="utf-8")
        filtered_index = filter_index(index_content, project_key)
        parts.append(f"## Knowledge Base Index (scoped to `{project_key}` + `global`)\n\n{filtered_index}")
    else:
        parts.append("## Knowledge Base Index\n\n(empty - no articles compiled yet)")

    recent_log = get_recent_log(project_key, memory_home=root, now=today)
    parts.append(f"## Recent Daily Log (scoped to `{project_key}`)\n\n{recent_log}")

    context = "\n\n---\n\n".join(parts)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n...(truncated)"

    return context


def session_start_output(context: str) -> dict[str, object]:
    """Return the supported SessionStart command-hook response envelope."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }


def _read_optional_input() -> dict[str, object]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("hook input must be a JSON object")
    return value


def main():
    try:
        hook_input = _read_optional_input()
        project_key = get_project_key(hook_input)
        context = build_context(project_key)
    except (json.JSONDecodeError, ValueError, OSError):
        return

    print(json.dumps(session_start_output(context)))


if __name__ == "__main__":
    main()
