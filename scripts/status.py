"""Read-only memory status command with snapshot and interactive modes."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TextIO, cast

if __package__:
    from .config import load_config
    from .status_store import (
        StatusReadError,
        StatusRun,
        StatusSnapshot,
        load_observer_state,
        observer_state_path,
        read_snapshot,
    )
else:
    from config import load_config
    from status_store import (
        StatusReadError,
        StatusRun,
        StatusSnapshot,
        load_observer_state,
        observer_state_path,
        read_snapshot,
    )


_KIND_LABELS = {
    "capture": "Capture",
    "compile": "Compile",
    "query_file": "Filed answer",
    "connections": "Connections",
    "semantic_lint": "Semantic lint",
}
_SOURCE_LABELS = {"claude": "Claude", "codex": "Codex", "system": "System"}
_STATE_ICONS = {
    "queued": "○",
    "running": "●",
    "retrying": "↻",
    "succeeded": "✓",
    "failed": "!",
    "dead": "!",
}
_COMPILE_ICONS = {
    "running": "●",
    "retrying": "↻",
    "complete": "✓",
    "failed": "!",
    "unavailable": "!",
}
_RESET = "\x1b[0m"
_TITLE_COLOR = "\x1b[1;36m"
_HEADING_COLOR = "\x1b[1;34m"
_STATE_COLORS = {
    "queued": "\x1b[36m",
    "running": "\x1b[36m",
    "retrying": "\x1b[33m",
    "succeeded": "\x1b[32m",
    "complete": "\x1b[32m",
    "failed": "\x1b[31m",
    "dead": "\x1b[31m",
    "unavailable": "\x1b[31m",
}


def _fit(line: str, width: int) -> str:
    width = max(1, width)
    if len(line) <= width:
        return line
    return "…" if width == 1 else line[: width - 1] + "…"


def _relative_time(updated_at: datetime, now: datetime) -> str:
    elapsed = max(
        0,
        int((now.astimezone(UTC) - updated_at.astimezone(UTC)).total_seconds()),
    )
    if elapsed < 60:
        return f"{elapsed}s ago"
    if elapsed < 3_600:
        return f"{elapsed // 60}m ago"
    if elapsed < 86_400:
        return f"{elapsed // 3_600}h ago"
    return f"{elapsed // 86_400}d ago"


def _run_line(
    run: StatusRun,
    *,
    now: datetime,
    prefix: str = "  ",
    include_state_icon: bool = True,
) -> str:
    icon = f"{_STATE_ICONS.get(run.state, '○')} " if include_state_icon else ""
    label = _KIND_LABELS.get(run.kind, run.kind.replace("_", " ").title())
    source = _SOURCE_LABELS.get(run.source_agent, run.source_agent.title())
    phase = run.phase.replace("_", " ")
    result = run.summary or run.error or run.state.replace("_", " ")
    return (
        f"{prefix}{icon}{label} · {source} · {run.project} · {phase} · "
        f"{result} · {_relative_time(run.updated_at, now)}"
    )


def _colorize(line: str, code: str | None, *, enabled: bool) -> str:
    if not enabled or code is None:
        return line
    return f"{code}{line}{_RESET}"


def _render_snapshot(
    snapshot: StatusSnapshot,
    *,
    now: datetime,
    width: int,
    color: bool,
) -> str:
    lines: list[tuple[str, str | None]] = [
        (f"MEMORY STATUS · {now.astimezone(UTC):%Y-%m-%d %H:%M} UTC", _TITLE_COLOR)
    ]
    for heading, runs in (
        ("ACTIVE", snapshot.active),
        ("NEEDS ATTENTION", snapshot.attention),
        ("RECENT", snapshot.recent),
    ):
        visible_runs = tuple(run for run in runs if run.kind != "compile")
        lines.extend((("", None), (heading, _HEADING_COLOR)))
        if visible_runs:
            lines.extend(
                (
                    _run_line(run, now=now),
                    _STATE_COLORS.get(run.state),
                )
                for run in visible_runs
            )
        else:
            lines.append(("  — None", None))
    compile_label = snapshot.compile.state.replace("_", " ")
    compile_icon = _COMPILE_ICONS.get(snapshot.compile.state, "○")
    lines.extend(
        (
            ("", None),
            ("END-OF-DAY COMPILE", _HEADING_COLOR),
            (
                f"  {compile_icon} {compile_label} · {snapshot.compile.summary}",
                _STATE_COLORS.get(snapshot.compile.state),
            ),
        )
    )
    if snapshot.compile.run is not None:
        lines.append(
            (
                _run_line(
                    snapshot.compile.run,
                    now=now,
                    prefix="  ↳ ",
                    include_state_icon=False,
                ),
                _STATE_COLORS.get(snapshot.compile.run.state),
            )
        )
    return (
        "\n".join(_colorize(_fit(line, width), code, enabled=color) for line, code in lines) + "\n"
    )


def _run_dashboard(*, no_color: bool) -> int:
    module_name = f"{__package__}.status_app" if __package__ else "status_app"
    run_dashboard = cast(
        Callable[..., int | None],
        importlib.import_module(module_name).run_dashboard,
    )
    result = run_dashboard(no_color=no_color)
    return 0 if result is None else int(result)


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    output: TextIO | None = None,
    terminal_width: int | None = None,
) -> int:
    """Run the interactive dashboard or print one deterministic snapshot."""
    parser = argparse.ArgumentParser(description="Show memory compiler status")
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="print one status snapshot instead of watching interactively",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable semantic terminal colors",
    )
    args = parser.parse_args(argv)
    if not args.snapshot:
        return _run_dashboard(no_color=args.no_color)

    source_env = os.environ if env is None else env
    stream = sys.stdout if output is None else output
    current = datetime.now(UTC) if now is None else now
    width = terminal_width or shutil.get_terminal_size((100, 24)).columns
    is_tty = bool(getattr(stream, "isatty", lambda: False)())
    color = is_tty and not args.no_color and "NO_COLOR" not in source_env
    try:
        config = load_config(source_env)
        observer_state = load_observer_state(observer_state_path(config.root_dir))
        snapshot = read_snapshot(
            config.queue_path,
            now=current,
            observer_state=observer_state,
            memory_home=config.root_dir,
            health_alerts=(),
        )
    except (StatusReadError, OSError, ValueError) as error:
        stream.write(f"MEMORY STATUS\n\nInspection failed: {error}\n")
        return 2

    stream.write(_render_snapshot(snapshot, now=current, width=width, color=color))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
