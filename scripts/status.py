"""Read-only memory status command with snapshot and interactive modes."""

from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TextIO, cast

from rich.cells import cell_len, split_text

if TYPE_CHECKING:
    if __package__:
        from .status_store import StatusRun, StatusSnapshot
    else:
        from status_store import StatusRun, StatusSnapshot


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
_OSC_PATTERN = re.compile(
    r"(?:\x1b\]|\x9d).*?(?:\x07|\x1b\\|\x9c|\Z)",
    re.DOTALL,
)
_CSI_PATTERN = re.compile(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]")
_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
_UNICODE_GLYPH_PROBE = "·○●↻✓↳—…"


def load_config(env: Mapping[str, str]):
    """Import configuration lazily so invalid environments remain diagnosable."""
    module_name = f"{__package__}.config" if __package__ else "config"
    return importlib.import_module(module_name).load_config(env)


def _status_store_module() -> Any:
    module_name = f"{__package__}.status_store" if __package__ else "status_store"
    return importlib.import_module(module_name)


def observer_state_path(memory_home):
    return _status_store_module().observer_state_path(memory_home)


def load_observer_state(path):
    return _status_store_module().load_observer_state(path)


def read_snapshot(*args, **kwargs):
    return _status_store_module().read_snapshot(*args, **kwargs)


def read_recent_hook_alerts(*args, **kwargs):
    """Import hook health lazily after configuration has been validated."""
    module_name = f"{__package__}.status_health" if __package__ else "status_health"
    return importlib.import_module(module_name).read_recent_hook_alerts(*args, **kwargs)


def _safe_terminal_text(value: object) -> str:
    """Remove terminal controls while retaining readable single-line text."""
    text = _OSC_PATTERN.sub("", str(value))
    text = _CSI_PATTERN.sub("", text)
    text = _CONTROL_PATTERN.sub(" ", text)
    return " ".join(text.split())


def _output_encoding(stream: TextIO) -> str:
    return getattr(stream, "encoding", None) or "utf-8"


def _supports_unicode_glyphs(encoding: str) -> bool:
    try:
        _UNICODE_GLYPH_PROBE.encode(encoding, errors="strict")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _encoding_safe_text(text: str, encoding: str) -> str:
    try:
        return text.encode(encoding, errors="replace").decode(encoding)
    except LookupError:
        return text.encode("ascii", errors="replace").decode("ascii")


def _write_output(stream: TextIO, text: str, *, encoding: str) -> None:
    stream.write(_encoding_safe_text(text, encoding))


def _fit(line: str, width: int | None, ellipsis: str = "…") -> str:
    if width is None or cell_len(line) <= width:
        return line
    width = max(1, width)
    ellipsis_width = cell_len(ellipsis)
    if width <= ellipsis_width:
        return split_text(ellipsis, width)[0]
    prefix, _ = split_text(line, width - ellipsis_width)
    return prefix + ellipsis


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
    unicode_glyphs: bool,
    prefix: str = "  ",
    include_state_icon: bool = True,
) -> str:
    state = _safe_terminal_text(run.state)
    icons = (
        _STATE_ICONS
        if unicode_glyphs
        else {
            "queued": "o",
            "running": "*",
            "retrying": "~",
            "succeeded": "+",
            "failed": "!",
            "dead": "!",
        }
    )
    icon = f"{icons.get(state, 'o')} " if include_state_icon else ""
    kind = _safe_terminal_text(run.kind)
    label = _KIND_LABELS.get(kind, kind.replace("_", " ").title())
    agent = _safe_terminal_text(run.source_agent)
    source = _SOURCE_LABELS.get(agent, agent.title())
    project = _safe_terminal_text(run.project)
    phase = _safe_terminal_text(run.phase).replace("_", " ")
    result = _safe_terminal_text(run.summary or run.error or state.replace("_", " "))
    separator = " · " if unicode_glyphs else " | "
    return (
        f"{prefix}{icon}{label}{separator}{source}{separator}{project}{separator}"
        f"{phase}{separator}{result}{separator}{_relative_time(run.updated_at, now)}"
    )


def _colorize(line: str, code: str | None, *, enabled: bool) -> str:
    if not enabled or code is None:
        return line
    return f"{code}{line}{_RESET}"


def _render_snapshot(
    snapshot: StatusSnapshot,
    *,
    now: datetime,
    width: int | None,
    color: bool,
    unicode_glyphs: bool,
) -> str:
    title_separator = " · " if unicode_glyphs else " - "
    empty_label = "— None" if unicode_glyphs else "- None"
    ellipsis = "…" if unicode_glyphs else "..."
    lines: list[tuple[str, str | None]] = [
        (
            f"MEMORY STATUS{title_separator}{now.astimezone(UTC):%Y-%m-%d %H:%M} UTC",
            _TITLE_COLOR,
        )
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
                    _run_line(run, now=now, unicode_glyphs=unicode_glyphs),
                    _STATE_COLORS.get(run.state),
                )
                for run in visible_runs
            )
        else:
            lines.append((f"  {empty_label}", None))
    compile_state = _safe_terminal_text(snapshot.compile.state)
    compile_label = compile_state.replace("_", " ")
    compile_icons = (
        _COMPILE_ICONS
        if unicode_glyphs
        else {
            "running": "*",
            "retrying": "~",
            "complete": "+",
            "failed": "!",
            "unavailable": "!",
        }
    )
    compile_icon = compile_icons.get(compile_state, "○" if unicode_glyphs else "o")
    separator = " · " if unicode_glyphs else " | "
    compile_summary = _safe_terminal_text(snapshot.compile.summary)
    lines.extend(
        (
            ("", None),
            ("END-OF-DAY COMPILE", _HEADING_COLOR),
            (
                f"  {compile_icon} {compile_label}{separator}{compile_summary}",
                _STATE_COLORS.get(compile_state),
            ),
        )
    )
    if snapshot.compile.run is not None:
        lines.append(
            (
                _run_line(
                    snapshot.compile.run,
                    now=now,
                    unicode_glyphs=unicode_glyphs,
                    prefix="  ↳ " if unicode_glyphs else "  -> ",
                    include_state_icon=False,
                ),
                _STATE_COLORS.get(snapshot.compile.run.state),
            )
        )
    if snapshot.health_alerts:
        lines.extend((("", None), ("HEALTH", _HEADING_COLOR)))
        lines.extend(
            (
                f"  ! {_safe_terminal_text(alert.message)}",
                _STATE_COLORS["failed"],
            )
            for alert in snapshot.health_alerts
        )
    return (
        "\n".join(
            _colorize(_fit(line, width, ellipsis), code, enabled=color) for line, code in lines
        )
        + "\n"
    )


def _run_dashboard(*, no_color: bool, env: Mapping[str, str]) -> int:
    module_name = f"{__package__}.status_app" if __package__ else "status_app"
    run_dashboard = cast(
        Callable[..., int | None],
        importlib.import_module(module_name).run_dashboard,
    )
    result = run_dashboard(no_color=no_color, env=env)
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
    source_env = os.environ if env is None else env
    stream = sys.stdout if output is None else output
    if not args.snapshot:
        try:
            return _run_dashboard(no_color=args.no_color, env=source_env)
        except (OSError, RuntimeError, ValueError) as error:
            diagnostic = _safe_terminal_text(error)
            _write_output(
                stream,
                f"MEMORY STATUS\n\nInspection failed: {diagnostic}\n",
                encoding=_output_encoding(stream),
            )
            return 2

    current = datetime.now(UTC) if now is None else now
    is_tty = bool(getattr(stream, "isatty", lambda: False)())
    width = (
        terminal_width
        if terminal_width is not None
        else (shutil.get_terminal_size((100, 24)).columns if is_tty else None)
    )
    encoding = _output_encoding(stream)
    unicode_glyphs = _supports_unicode_glyphs(encoding)
    color = is_tty and not args.no_color and "NO_COLOR" not in source_env
    try:
        config = load_config(source_env)
        observer_state = load_observer_state(observer_state_path(config.root_dir))
        try:
            health_alerts = read_recent_hook_alerts(config.root_dir, now=current)
        except (OSError, RuntimeError, ValueError):
            health_alerts = ()
        snapshot = read_snapshot(
            config.queue_path,
            now=current,
            observer_state=observer_state,
            memory_home=config.root_dir,
            health_alerts=health_alerts,
        )
    except (OSError, RuntimeError, ValueError) as error:
        diagnostic = _safe_terminal_text(error)
        _write_output(
            stream,
            f"MEMORY STATUS\n\nInspection failed: {diagnostic}\n",
            encoding=encoding,
        )
        return 2

    rendered = _render_snapshot(
        snapshot,
        now=current,
        width=width,
        color=color,
        unicode_glyphs=unicode_glyphs,
    )
    _write_output(stream, rendered, encoding=encoding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
