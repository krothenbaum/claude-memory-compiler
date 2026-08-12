"""Shared utilities for the personal knowledge base."""

import hashlib
import errno
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import uuid

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows branch.
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX branch.
    msvcrt = None

if __package__:
    from .config import (
        CONCEPTS_DIR,
        CONNECTIONS_DIR,
        DAILY_DIR,
        INDEX_FILE,
        KNOWLEDGE_DIR,
        LOG_FILE,
        QA_DIR,
        STATE_FILE,
    )
else:
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


@dataclass(frozen=True)
class FileBaseline:
    """Identity of one regular file for compare-and-swap writes."""

    exists: bool
    size: int
    sha256: str | None


def _baseline_for_bytes(data: bytes) -> FileBaseline:
    return FileBaseline(True, len(data), hashlib.sha256(data).hexdigest())


def _read_file_with_baseline(path: Path | str) -> tuple[bytes | None, FileBaseline]:
    target = Path(path)
    if not target.exists() and not target.is_symlink():
        return None, FileBaseline(False, 0, None)
    if target.is_symlink():
        raise ValueError(f"baseline path must not be a symlink: {target}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"baseline path must be a regular file: {target}")
        if info.st_nlink != 1:
            raise ValueError(f"baseline path must not be hard-linked: {target}")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise ValueError(f"baseline path has an unsafe owner: {target}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read()
    finally:
        os.close(descriptor)
    return data, _baseline_for_bytes(data)


def capture_file_baseline(path: Path | str) -> FileBaseline:
    """Capture a file identity without following unsafe file types."""
    _data, baseline = _read_file_with_baseline(path)
    return baseline


def read_text_with_baseline(path: Path | str) -> tuple[str, FileBaseline]:
    """Read one UTF-8 file and return a baseline for those exact bytes."""
    data, baseline = _read_file_with_baseline(path)
    if data is None:
        raise FileNotFoundError(path)
    return data.decode("utf-8"), baseline


def _fsync_directory(path: Path | str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(Path(path), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class ExclusiveFileLock:
    """Owner-only cross-platform advisory file lock.

    The lock file is diagnostic; ownership comes from the operating-system
    lock held for the descriptor's lifetime.  ``blocking=False`` is useful for
    singleton workers, while durable writers use the blocking default.
    """

    def __init__(self, path: Path | str, *, blocking: bool = True) -> None:
        self.path = Path(os.path.abspath(Path(path).expanduser()))
        self.blocking = blocking
        self._token = f"{os.getpid()}:{uuid.uuid4()}"
        self._descriptor: int | None = None

    def _try_os_lock(self, descriptor: int) -> bool:
        try:
            if fcntl is not None:
                operation = fcntl.LOCK_EX
                if not self.blocking:
                    operation |= fcntl.LOCK_NB
                fcntl.flock(descriptor, operation)
            elif msvcrt is not None:  # pragma: no cover - Windows branch.
                os.lseek(descriptor, 0, os.SEEK_SET)
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                    os.lseek(descriptor, 0, os.SEEK_SET)
                mode = msvcrt.LK_LOCK if self.blocking else msvcrt.LK_NBLCK
                msvcrt.locking(descriptor, mode, 1)
            else:  # pragma: no cover - unsupported Python platform.
                raise RuntimeError("no supported OS file-lock implementation")
        except OSError as exc:
            if not self.blocking and exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        return True

    @staticmethod
    def _unlock(descriptor: int) -> None:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        elif msvcrt is not None:  # pragma: no cover - Windows branch.
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)

    def acquire(self) -> bool:
        if self._descriptor is not None:
            raise RuntimeError("file lock is already acquired by this instance")
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not parent_existed:
            self.path.parent.chmod(0o700)
        if self.path.parent.is_symlink() or self.path.is_symlink():
            raise ValueError("lock path must not be a symlink")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            os.close(descriptor)
            raise ValueError("lock path must be a regular file")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            os.close(descriptor)
            raise ValueError("lock path has an unsafe owner")
        if info.st_nlink != 1:
            os.close(descriptor)
            raise ValueError("lock path must not be hard-linked")
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        if not self._try_os_lock(descriptor):
            os.close(descriptor)
            return False
        try:
            payload = self._token.encode()
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, payload)
            os.fsync(descriptor)
        except BaseException:
            self._unlock(descriptor)
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return True

    def release(self) -> None:
        if self._descriptor is None:
            return
        try:
            self._unlock(self._descriptor)
        finally:
            os.close(self._descriptor)
            self._descriptor = None

    def __enter__(self) -> "ExclusiveFileLock":
        if not self.acquire():
            raise RuntimeError("file lock is already owned")
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def append_daily_entry(
    memory_home: Path | str,
    content: str,
    *,
    section: str = "Session",
    project_key: str = "unknown",
    cwd: str = "",
    agent: str = "claude",
    capture_identity: str | None = None,
    now: datetime | None = None,
) -> Path:
    """Append one provenance-tagged daily entry under the writer lock."""
    root = Path(memory_home).expanduser().resolve()
    timestamp = now or datetime.now(timezone.utc).astimezone()
    daily_dir = root / "daily"
    log_path = daily_dir / f"{timestamp.strftime('%Y-%m-%d')}.md"
    display_agent = {"claude": "Claude Code", "codex": "Codex"}.get(agent)
    if display_agent is None:
        raise ValueError("agent must be 'claude' or 'codex'")
    if capture_identity is not None and not re.fullmatch(r"[0-9a-f]{64}", capture_identity):
        raise ValueError("capture identity must be a lowercase SHA-256 digest")

    metadata_lines = [f"**Agent:** {display_agent}", f"**Project:** {project_key}"]
    if cwd:
        metadata_lines.append(f"**CWD:** {cwd}")
    identity_line = (
        f"<!-- @capture-id:{capture_identity} -->\n" if capture_identity is not None else ""
    )
    entry = (
        f"### {section} [{project_key}] ({timestamp.strftime('%H:%M')})\n\n"
        f"{identity_line}{'\n'.join(metadata_lines)}\n\n{content}\n\n"
    )

    with ExclusiveFileLock(root / "scripts" / "memory-writer.lock"):
        if __package__:
            from .staging import (
                _commit_replacements_unlocked,
                recover_incomplete_apply_unlocked,
            )
        else:
            from staging import _commit_replacements_unlocked, recover_incomplete_apply_unlocked

        recover_incomplete_apply_unlocked(root)
        if daily_dir.is_symlink():
            raise ValueError("daily directory must not be a symlink")
        if daily_dir.exists() and not daily_dir.is_dir():
            raise ValueError("daily path must be a directory")
        if capture_identity is not None and daily_dir.exists():
            marker = f"<!-- @capture-id:{capture_identity} -->".encode()
            for candidate in sorted(daily_dir.glob("*.md")):
                data, baseline = _read_file_with_baseline(candidate)
                if not baseline.exists or data is None:
                    continue
                if marker in data:
                    return candidate
        daily_dir_created = not daily_dir.exists()
        created = not log_path.exists() and not log_path.is_symlink()
        if created:
            daily_dir.mkdir(parents=True, exist_ok=True)
            original = b""
        else:
            data, baseline = _read_file_with_baseline(log_path)
            if not baseline.exists or data is None:
                raise ValueError("daily log disappeared before append")
            original = data
        header = (
            f"# Daily Log: {timestamp.strftime('%Y-%m-%d')}\n\n"
            "## Sessions\n\n## Memory Maintenance\n\n"
        ).encode() if created else b""
        relative = log_path.relative_to(root).as_posix()
        _commit_replacements_unlocked(root, {relative: original + header + entry.encode()})
        if created:
            _fsync_directory(daily_dir)
            if daily_dir_created:
                _fsync_directory(root)
    return log_path


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
    state, _baseline = load_state_with_baseline()
    return state


def load_state_with_baseline() -> tuple[dict, FileBaseline]:
    """Read state bytes once and return parsed state plus the same-byte baseline."""
    data, baseline = _read_file_with_baseline(STATE_FILE)
    if baseline.exists:
        assert data is not None
        state = json.loads(data.decode("utf-8"))
        if not isinstance(state, dict):
            raise ValueError("state.json must contain an object")
        return state, baseline
    return (
        {"ingested": {}, "query_count": 0, "last_lint": None, "total_cost": 0.0},
        FileBaseline(False, 0, None),
    )


def save_state(state: dict) -> None:
    """Atomically save state under the shared writer lock."""
    target = Path(os.path.abspath(STATE_FILE.expanduser()))
    if target.is_symlink():
        raise ValueError("state path must not be a symlink")
    root = target.parents[1]
    with ExclusiveFileLock(root / "scripts" / "memory-writer.lock"):
        save_state_unlocked(state, target)


def save_state_unlocked(state: dict, path: Path | str | None = None) -> None:
    """Atomically save state when the caller already owns the writer lock."""
    target = Path(path) if path is not None else STATE_FILE
    parent = target.parent
    if parent.exists() or parent.is_symlink():
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("state directory must be a real directory")
    else:
        parent.mkdir(parents=True, mode=0o700)
    if target.is_symlink():
        raise ValueError("state path must not be a symlink")
    if target.exists():
        info = target.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("state path must be a private regular file")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise ValueError("state path has an unsafe owner")
    serialized = json.dumps(state, indent=2).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        target.chmod(0o600)
        _fsync_directory(parent)
    finally:
        temporary.unlink(missing_ok=True)


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
