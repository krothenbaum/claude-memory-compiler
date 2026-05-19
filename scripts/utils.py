"""Shared utilities for the personal knowledge base."""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from config import (
    CONCEPTS_DIR,
    CONNECTIONS_DIR,
    DAILY_DIR,
    INDEX_FILE,
    KNOWLEDGE_DIR,
    LOG_FILE,
    QA_DIR,
    STATE_FILE,
)


# ── Terminal notifications ────────────────────────────────────────────
#
# flush.py and compile.py run as detached subprocesses (stdout/stderr piped to
# files or DEVNULL), so the user never sees them. notify_terminal writes a
# short line to the controlling TTY device so progress is visible from the
# shell that originally launched `claude`.

_TTY_PATH_UNSET = object()
_TTY_PATH_CACHE: object = _TTY_PATH_UNSET  # str | None once resolved


def _resolve_tty_path() -> str | None:
    """Cached lookup of the controlling-terminal device path.

    Falls back to walking the process ancestry when no controlling TTY is
    attached directly. Claude Code spawns hook subprocesses without a TTY
    (`TTY=??`), but their `claude` ancestor still owns the user's real
    terminal, so we crawl PPIDs until we find one with a real TTY column.
    """
    global _TTY_PATH_CACHE
    if _TTY_PATH_CACHE is not _TTY_PATH_UNSET:
        return _TTY_PATH_CACHE  # type: ignore[return-value]

    env_path = os.environ.get("CLAUDE_MEMORY_TTY")
    if env_path and os.path.exists(env_path):
        _TTY_PATH_CACHE = env_path
        return env_path

    # Try the direct /dev/tty path first (works in foreground or when a
    # controlling terminal is attached).
    try:
        fd = os.open("/dev/tty", os.O_WRONLY)
    except OSError:
        fd = None
    if fd is not None:
        try:
            path = os.ttyname(fd)
        except OSError:
            path = None
        finally:
            os.close(fd)
        if path:
            _TTY_PATH_CACHE = path
            return path

    # No controlling TTY on this process. Walk ancestry — when launched as a
    # Claude Code hook, the `claude` CLI itself owns the user's TTY.
    path = _walk_ancestors_for_tty()
    _TTY_PATH_CACHE = path
    return path


def _walk_ancestors_for_tty(max_hops: int = 10) -> str | None:
    """Walk PPIDs upward and return the first real TTY device path found."""
    import subprocess as _sp

    pid = os.getpid()
    for _ in range(max_hops):
        try:
            result = _sp.run(
                ["ps", "-p", str(pid), "-o", "ppid=,tty="],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, _sp.TimeoutExpired, OSError):
            return None
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split()
        if len(parts) < 2:
            return None
        try:
            ppid = int(parts[0])
        except ValueError:
            return None
        tty = parts[1]
        # `??` means no controlling terminal; keep walking.
        if tty and tty != "??":
            candidate = tty if tty.startswith("/") else f"/dev/{tty}"
            if os.path.exists(candidate):
                return candidate
        if ppid in (0, 1) or ppid == pid:
            return None
        pid = ppid
    return None


def notify_terminal(msg: str) -> None:
    """Write a `[memory] msg` line to the user's terminal.

    No-op when stdout already targets a TTY (foreground run — would
    duplicate) or when no controlling terminal is reachable.
    """
    try:
        if sys.stdout.isatty():
            return
    except (AttributeError, ValueError):
        pass
    path = _resolve_tty_path()
    if not path:
        return
    try:
        with open(path, "w") as tty:
            tty.write(f"[memory] {msg}\n")
            tty.flush()
    except OSError:
        pass


# ── State management ──────────────────────────────────────────────────

def load_state() -> dict:
    """Load persistent state from state.json."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"ingested": {}, "query_count": 0, "last_lint": None, "total_cost": 0.0}


def save_state(state: dict) -> None:
    """Save state to state.json."""
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ── File hashing ──────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    """SHA-256 hash of a file (first 16 hex chars)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ── Slug / naming ─────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


# ── Wikilink helpers ──────────────────────────────────────────────────

def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[wikilinks]] from markdown content."""
    return re.findall(r"\[\[([^\]]+)\]\]", content)


def wiki_article_exists(link: str) -> bool:
    """Check if a wikilinked article exists on disk."""
    path = KNOWLEDGE_DIR / f"{link}.md"
    return path.exists()


# ── Wiki content helpers ──────────────────────────────────────────────

def read_wiki_index() -> str:
    """Read the knowledge base index file."""
    if INDEX_FILE.exists():
        return INDEX_FILE.read_text(encoding="utf-8")
    return "# Knowledge Base Index\n\n| Article | Summary | Compiled From | Updated |\n|---------|---------|---------------|---------|"


def read_all_wiki_content() -> str:
    """Read index + all wiki articles into a single string for context."""
    parts = [f"## INDEX\n\n{read_wiki_index()}"]

    for subdir in [CONCEPTS_DIR, CONNECTIONS_DIR, QA_DIR]:
        if not subdir.exists():
            continue
        for md_file in sorted(subdir.glob("*.md")):
            rel = md_file.relative_to(KNOWLEDGE_DIR)
            content = md_file.read_text(encoding="utf-8")
            parts.append(f"## {rel}\n\n{content}")

    return "\n\n---\n\n".join(parts)


def list_wiki_articles() -> list[Path]:
    """List all wiki article files."""
    articles = []
    for subdir in [CONCEPTS_DIR, CONNECTIONS_DIR, QA_DIR]:
        if subdir.exists():
            articles.extend(sorted(subdir.glob("*.md")))
    return articles


def list_raw_files() -> list[Path]:
    """List all daily log files."""
    if not DAILY_DIR.exists():
        return []
    return sorted(DAILY_DIR.glob("*.md"))


# ── Index helpers ─────────────────────────────────────────────────────

def count_inbound_links(target: str, exclude_file: Path | None = None) -> int:
    """Count how many wiki articles link to a given target."""
    count = 0
    for article in list_wiki_articles():
        if article == exclude_file:
            continue
        content = article.read_text(encoding="utf-8")
        if f"[[{target}]]" in content:
            count += 1
    return count


def get_article_word_count(path: Path) -> int:
    """Count words in an article, excluding YAML frontmatter."""
    content = path.read_text(encoding="utf-8")
    # Strip frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:]
    return len(content.split())


def build_index_entry(rel_path: str, summary: str, sources: str, updated: str) -> str:
    """Build a single index table row."""
    link = rel_path.replace(".md", "")
    return f"| [[{link}]] | {summary} | {sources} | {updated} |"
