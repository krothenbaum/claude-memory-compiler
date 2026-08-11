"""Private staging, validation, and crash-safe knowledge workspace writes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Callable, Mapping, Sequence
import uuid

try:
    from .config import ROOT_DIR
    from .utils import ExclusiveFileLock
except ImportError:  # Direct execution with scripts/ on sys.path.
    from config import ROOT_DIR
    from utils import ExclusiveFileLock


_MANIFEST_NAME = ".stage-manifest.json"
_AFTER_MANIFEST_NAME = ".stage-manifest-after.json"
_ARTICLE_PREFIXES = (
    "knowledge/concepts/",
    "knowledge/connections/",
    "knowledge/qa/",
)
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


class StageValidationError(ValueError):
    """Raised when a stage cannot safely be accepted."""


class RetryableApplyError(RuntimeError):
    """Raised after an apply conflict or a transaction rollback."""


@dataclass(frozen=True)
class ManifestEntry:
    type: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Stage:
    root: Path
    memory_home: Path
    baseline: Mapping[str, ManifestEntry]
    baseline_bytes: Mapping[str, bytes]
    daily_source: str | None = None


@dataclass(frozen=True)
class ValidatedStage:
    stage: Stage
    before: Mapping[str, ManifestEntry]
    after: Mapping[str, ManifestEntry]
    changed_paths: tuple[str, ...]
    task: str
    allowed_paths: tuple[str, ...]

    @property
    def root(self) -> Path:
        return self.stage.root

    @property
    def memory_home(self) -> Path:
        return self.stage.memory_home


@dataclass(frozen=True)
class ApplyBookkeeping:
    """Host-owned updates committed after validated model replacements."""

    compiled_marker_path: str | None = None
    compiled_at: str | None = None
    state: Mapping[str, object] | None = None
    state_path: str = "scripts/state.json"
    extra_updates: Mapping[str, bytes | str] = field(default_factory=dict)
    failure_injector: Callable[[int, str], None] | None = None


@dataclass(frozen=True)
class ApplyResult:
    changed_paths: tuple[str, ...]
    recovered_journal: bool = False


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(data: bytes) -> ManifestEntry:
    return ManifestEntry(type="file", size=len(data), sha256=_sha256(data))


def _relative_path(value: str | Path) -> str:
    raw = str(value)
    if not raw or "\\" in raw:
        raise StageValidationError(f"path must be a non-empty POSIX relative path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise StageValidationError(f"path must be relative and cannot escape its root: {raw!r}")
    return path.as_posix()


def _safe_destination(root: Path, relative: str, *, allow_missing: bool = True) -> Path:
    relative = _relative_path(relative)
    root = root.resolve()
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for component in PurePosixPath(relative).parts:
        current = current / component
        if current.exists() or current.is_symlink():
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise StageValidationError(f"symlink is not allowed: {relative}")
            if current == candidate and stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                raise StageValidationError(f"hard-linked file is not allowed: {relative}")
        elif not allow_missing:
            raise StageValidationError(f"required path is missing: {relative}")
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise StageValidationError(f"path escapes staging root: {relative}") from exc
    return candidate


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise StageValidationError(f"private directory must not be a symlink: {path}")
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode):
        raise StageValidationError(f"private path must be a directory: {path}")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise StageValidationError(f"private directory has an unsafe owner: {path}")
    path.chmod(0o700)


def _copy_private(source: Path, destination: Path) -> bytes:
    info = source.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise StageValidationError(f"source must not be a symlink: {source}")
    if not stat.S_ISREG(info.st_mode):
        raise StageValidationError(f"source must be a regular file: {source}")
    if info.st_nlink != 1:
        raise StageValidationError(f"source must not be hard-linked: {source}")
    data = source.read_bytes()
    _private_directory(destination.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    destination.chmod(0o600)
    return data


def _safe_identifier(value: object) -> str:
    pieces = [
        _SAFE_COMPONENT.sub("-", piece).strip("-.")
        for piece in str(value).replace("\\", "/").split("/")
    ]
    return "-".join(piece for piece in pieces if piece) or "unknown"


def _manifest_payload(manifest: Mapping[str, ManifestEntry]) -> bytes:
    return (
        json.dumps(
            {
                path: {"type": item.type, "size": item.size, "sha256": item.sha256}
                for path, item in sorted(manifest.items())
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _write_private_file(path: Path, data: bytes) -> None:
    _private_directory(path.parent)
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise StageValidationError(f"private file must not be a symlink: {path}")
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise StageValidationError(f"private file has an unsafe identity: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def create_stage(
    memory_home: Path | str,
    job_id: object,
    attempt_id: object,
    *,
    daily_source: str | Path | None = None,
    relevant_articles: Sequence[str | Path] = (),
    include_state: bool = True,
) -> Stage:
    """Create a fresh owner-only stage containing the minimum requested files."""
    home = Path(memory_home).expanduser().resolve()
    stage_parent = home / "scripts" / "staging"
    _private_directory(stage_parent)
    stage_root = stage_parent / f"{_safe_identifier(job_id)}-{_safe_identifier(attempt_id)}"
    if stage_root.exists() or stage_root.is_symlink():
        raise StageValidationError(f"stage must be fresh: {stage_root.name}")
    stage_root.mkdir(mode=0o700)

    selected: list[str] = ["AGENTS.md", "knowledge/index.md", "knowledge/log.md"]
    normalized_daily = _relative_path(daily_source) if daily_source is not None else None
    if normalized_daily is not None:
        if not normalized_daily.startswith("daily/"):
            raise StageValidationError("daily source must be below daily/")
        selected.append(normalized_daily)
    selected.extend(_relative_path(path) for path in relevant_articles)
    if include_state and (home / "scripts/state.json").exists():
        selected.append("scripts/state.json")

    baseline_bytes: dict[str, bytes] = {}
    try:
        for relative in dict.fromkeys(selected):
            source = _safe_destination(home, relative, allow_missing=False)
            destination = _safe_destination(stage_root, relative)
            baseline_bytes[relative] = _copy_private(source, destination)
        baseline = {path: _entry(data) for path, data in baseline_bytes.items()}
        _write_private_file(stage_root / _MANIFEST_NAME, _manifest_payload(baseline))
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    return Stage(
        root=stage_root,
        memory_home=home,
        baseline=baseline,
        baseline_bytes=baseline_bytes,
        daily_source=normalized_daily,
    )


def snapshot_manifest(root: Path | str) -> dict[str, ManifestEntry]:
    """Return a strict relative manifest without following links."""
    supplied_root = Path(root)
    if supplied_root.is_symlink():
        raise StageValidationError("stage root must not be a symlink")
    root_path = supplied_root.resolve()
    if not root_path.is_dir():
        raise StageValidationError("stage root must be a directory")
    manifest: dict[str, ManifestEntry] = {}
    for candidate in sorted(root_path.rglob("*")):
        relative = candidate.relative_to(root_path).as_posix()
        if relative in {_MANIFEST_NAME, _AFTER_MANIFEST_NAME}:
            continue
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise StageValidationError(f"symlink is not allowed in stage: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise StageValidationError(f"special file is not allowed in stage: {relative}")
        if info.st_nlink != 1:
            raise StageValidationError(f"hard-linked file is not allowed in stage: {relative}")
        data = candidate.read_bytes()
        manifest[_relative_path(relative)] = _entry(data)
    return manifest


def _matches(path: str, patterns: Sequence[str]) -> bool:
    subject = PurePosixPath(path)
    for raw_pattern in patterns:
        pattern = _relative_path(raw_pattern)
        if path == pattern or subject.match(pattern):
            return True
    return False


def _validate_utf8(stage_root: Path, paths: Sequence[str]) -> None:
    for relative in paths:
        try:
            (_safe_destination(stage_root, relative, allow_missing=False)).read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError as exc:
            raise StageValidationError(f"malformed UTF-8 in {relative}") from exc


def _validate_frontmatter(relative: str, content: str) -> None:
    if not content.startswith("---\n"):
        raise StageValidationError(f"article has malformed frontmatter: {relative}")
    boundary = content.find("\n---\n", 4)
    if boundary < 0:
        raise StageValidationError(f"article has malformed frontmatter: {relative}")
    frontmatter = content[4:boundary]
    required = {"title"}
    if relative.startswith(("knowledge/concepts/", "knowledge/connections/")):
        required.add("project")
    missing = [name for name in required if not re.search(rf"(?m)^{name}:\s*\S", frontmatter)]
    if missing:
        raise StageValidationError(
            f"article frontmatter missing {', '.join(missing)}: {relative}"
        )


def _article_slug(relative: str) -> str:
    return relative.removeprefix("knowledge/").removesuffix(".md")


def _added_lines(before: str, after: str) -> list[str]:
    remaining = Counter(before.splitlines())
    added: list[str] = []
    for line in after.splitlines():
        if remaining[line]:
            remaining[line] -= 1
        else:
            added.append(line)
    return added


def _validate_task_contract(
    stage: Stage,
    after: Mapping[str, ManifestEntry],
    changed: tuple[str, ...],
    task: str,
) -> None:
    article_changes = tuple(
        path for path in changed if path.endswith(".md") and path.startswith(_ARTICLE_PREFIXES)
    )
    for relative in article_changes:
        content = (stage.root / relative).read_text(encoding="utf-8")
        _validate_frontmatter(relative, content)

    if not article_changes:
        if task in {"file_answer", "connections"}:
            raise StageValidationError(f"{task} requires an article")
        return

    if "knowledge/index.md" not in changed:
        raise StageValidationError(f"{task} article changes require an index update")
    if "knowledge/log.md" not in changed:
        raise StageValidationError(f"{task} article changes require a build log update")

    index = (stage.root / "knowledge/index.md").read_text(encoding="utf-8")
    build_log = (stage.root / "knowledge/log.md").read_text(encoding="utf-8")
    if "|" not in index:
        raise StageValidationError("knowledge index is malformed")
    baseline_index = stage.baseline_bytes.get("knowledge/index.md", b"").decode("utf-8")
    baseline_log = stage.baseline_bytes.get("knowledge/log.md", b"").decode("utf-8")
    added_index_lines = _added_lines(baseline_index, index)
    added_log_lines = _added_lines(baseline_log, build_log)
    has_new_log_heading = any(
        re.match(r"^##\s+\[[^\]]+\]\s+\S+", line) for line in added_log_lines
    )
    for article in article_changes:
        slug = _article_slug(article)
        reference = f"[[{slug}]]"
        matching_rows = [line for line in added_index_lines if reference in line]
        valid_row = False
        for row in matching_rows:
            if not row.startswith("|") or not row.endswith("|"):
                continue
            columns = [column.strip() for column in row.strip("|").split("|")]
            if len(columns) == 5 and columns[0] == reference and all(columns):
                valid_row = True
                break
        if not valid_row:
            raise StageValidationError(f"index row is missing or malformed for {reference}")
        if not has_new_log_heading or not any(
            reference in line for line in added_log_lines
        ):
            raise StageValidationError(f"build log entry is missing or malformed for {reference}")

    if task == "file_answer" and not any(
        path.startswith("knowledge/qa/") for path in article_changes
    ):
        raise StageValidationError("file_answer requires a Q&A article")
    if task == "connections" and not any(
        path.startswith("knowledge/connections/") for path in article_changes
    ):
        raise StageValidationError("connections requires a connection article")


def validate_stage(
    stage: Stage,
    *,
    allowed_paths: Sequence[str],
    task: object,
) -> ValidatedStage:
    """Validate an attempted model edit and return its immutable change set."""
    expected_parent = stage.memory_home / "scripts" / "staging"
    if stage.root.is_symlink() or stage.root.resolve().parent != expected_parent.resolve():
        raise StageValidationError("stage root is outside the private staging directory")
    manifest_file = stage.root / _MANIFEST_NAME
    if (
        manifest_file.is_symlink()
        or not manifest_file.is_file()
        or manifest_file.read_bytes() != _manifest_payload(stage.baseline)
    ):
        raise StageValidationError("stage baseline manifest was modified")
    after_manifest_file = stage.root / _AFTER_MANIFEST_NAME
    if after_manifest_file.exists() or after_manifest_file.is_symlink():
        raise StageValidationError("stage after-manifest existed before validation")
    after = snapshot_manifest(stage.root)
    before = dict(stage.baseline)
    deleted = sorted(set(before) - set(after))
    if deleted:
        raise StageValidationError(f"stage deleted files: {', '.join(deleted)}")
    changed = tuple(
        sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    )
    if stage.daily_source is not None and stage.daily_source in changed:
        raise StageValidationError("daily source content cannot be model-edited")
    forbidden = tuple(path for path in changed if not _matches(path, allowed_paths))
    if forbidden:
        raise StageValidationError(f"changed paths outside allowlist: {', '.join(forbidden)}")
    _validate_utf8(stage.root, tuple(after))
    if "scripts/state.json" in after:
        try:
            staged_state = json.loads(
                (stage.root / "scripts/state.json").read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise StageValidationError("staged state.json is malformed") from exc
        if not isinstance(staged_state, dict):
            raise StageValidationError("staged state.json must contain an object")
    task_name = getattr(task, "value", str(task))
    _validate_task_contract(stage, after, changed, task_name)
    validated = ValidatedStage(
        stage=stage,
        before=before,
        after=after,
        changed_paths=changed,
        task=task_name,
        allowed_paths=tuple(allowed_paths),
    )
    _write_private_file(after_manifest_file, _manifest_payload(after))
    return validated


def discard_stage(stage: Stage | ValidatedStage) -> None:
    """Remove one exact stage after success or classified failure."""
    target = stage.stage.root if isinstance(stage, ValidatedStage) else stage.root
    memory_home = (
        stage.stage.memory_home
        if isinstance(stage, ValidatedStage)
        else stage.memory_home
    )
    if target.is_symlink():
        raise StageValidationError("refusing to discard a symlinked stage")
    expected_parent = (memory_home / "scripts" / "staging").resolve()
    if target.resolve().parent != expected_parent:
        raise StageValidationError("refusing to discard a path outside the staging directory")
    shutil.rmtree(target)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, data: bytes) -> None:
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise StageValidationError(f"destination parent must not be a symlink: {path.parent}")
    if not parent_existed:
        path.parent.chmod(0o700)
    if path.is_symlink():
        raise StageValidationError(f"destination must not be a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _journal_path(home: Path) -> Path:
    directory = home / "scripts" / "memory-apply-journal"
    _private_directory(directory)
    return directory / f"apply-{uuid.uuid4().hex}.json"


def _journal_bytes(
    home: Path,
    state: str,
    operations: Sequence[tuple[str, bytes | None, bytes]],
) -> bytes:
    return (
        json.dumps(
            {
                "version": 1,
                "state": state,
                "root": str(home),
                "entries": [
                    {
                        "path": path,
                        "original": original.hex() if original is not None else None,
                        "replacement": replacement.hex(),
                    }
                    for path, original, replacement in operations
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _remove_journal(path: Path) -> None:
    path.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _restore_operations(home: Path, entries: Sequence[Mapping[str, object]]) -> None:
    for entry in reversed(entries):
        relative = _relative_path(str(entry["path"]))
        destination = _safe_destination(home, relative)
        original = entry.get("original")
        if original is None:
            if destination.is_symlink():
                raise StageValidationError(f"cannot recover through symlink: {relative}")
            destination.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
        else:
            _atomic_replace(destination, bytes.fromhex(str(original)))


def _recover_unlocked(home: Path) -> bool:
    directory = home / "scripts" / "memory-apply-journal"
    if not directory.exists():
        return False
    if directory.is_symlink():
        raise StageValidationError("journal directory must not be a symlink")
    recovered = False
    for journal in sorted(directory.glob("*.json")):
        if journal.is_symlink() or not journal.is_file():
            raise StageValidationError("journal must be a regular file")
        info = journal.stat()
        if info.st_nlink != 1:
            raise StageValidationError("journal must not be hard-linked")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise StageValidationError("journal has an unsafe owner")
        journal.chmod(0o600)
        payload = json.loads(journal.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or Path(payload.get("root", "")).resolve() != home:
            raise StageValidationError("journal root or version is invalid")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise StageValidationError("journal entries are invalid")
        if payload.get("state") != "complete":
            _restore_operations(home, entries)
            recovered = True
        _remove_journal(journal)
    return recovered


def recover_incomplete_apply(memory_home: Path | str | None = None) -> bool:
    """Restore any prepared/applying transaction left by an interrupted writer."""
    home = Path(memory_home or ROOT_DIR).expanduser().resolve()
    with ExclusiveFileLock(home / "scripts" / "memory-writer.lock"):
        return _recover_unlocked(home)


def _bookkeeping_updates(
    home: Path, bookkeeping: ApplyBookkeeping
) -> dict[str, bytes]:
    updates: dict[str, bytes] = {}
    if bookkeeping.compiled_marker_path is not None:
        relative = _relative_path(bookkeeping.compiled_marker_path)
        if not bookkeeping.compiled_at:
            raise StageValidationError("compiled_at is required with a compiled marker")
        source = _safe_destination(home, relative, allow_missing=False)
        updates[relative] = source.read_bytes() + (
            f"\n<!-- @compiled-through:{bookkeeping.compiled_at} -->\n"
        ).encode()
    if bookkeeping.state is not None:
        relative = _relative_path(bookkeeping.state_path)
        if relative in updates:
            raise StageValidationError(f"duplicate bookkeeping destination: {relative}")
        updates[relative] = (json.dumps(bookkeeping.state, indent=2) + "\n").encode()
    for raw_path, value in bookkeeping.extra_updates.items():
        relative = _relative_path(raw_path)
        if relative in updates:
            raise StageValidationError(f"duplicate bookkeeping destination: {relative}")
        updates[relative] = value.encode() if isinstance(value, str) else bytes(value)
    return updates


def _commit_replacements_unlocked(
    home: Path,
    replacements: Mapping[str, bytes],
    failure_injector: Callable[[int, str], None] | None = None,
) -> tuple[str, ...]:
    operations: list[tuple[str, bytes | None, bytes]] = []
    for relative, replacement in sorted(replacements.items()):
        destination = _safe_destination(home, relative)
        original = destination.read_bytes() if destination.exists() else None
        operations.append((relative, original, replacement))

    journal = _journal_path(home)
    _write_private_file(journal, _journal_bytes(home, "prepared", operations))
    _fsync_directory(journal.parent)
    try:
        _write_private_file(journal, _journal_bytes(home, "applying", operations))
        for step, (relative, _original, replacement) in enumerate(operations, 1):
            _atomic_replace(_safe_destination(home, relative), replacement)
            if failure_injector is not None:
                failure_injector(step, relative)
        _write_private_file(journal, _journal_bytes(home, "complete", operations))
        _remove_journal(journal)
    except Exception as exc:
        try:
            _restore_operations(
                home,
                [
                    {
                        "path": relative,
                        "original": original.hex() if original is not None else None,
                    }
                    for relative, original, _replacement in operations
                ],
            )
            _remove_journal(journal)
        except Exception as rollback_exc:
            raise RetryableApplyError(
                "apply failed and rollback requires journal recovery"
            ) from rollback_exc
        raise RetryableApplyError("apply failed and was rolled back") from exc
    return tuple(sorted(replacements))


def apply_host_bookkeeping(
    memory_home: Path | str,
    bookkeeping: ApplyBookkeeping,
) -> ApplyResult:
    """Commit marker/state/usage updates with the same journal as stage writes."""
    home = Path(memory_home).expanduser().resolve()
    with ExclusiveFileLock(home / "scripts" / "memory-writer.lock"):
        recovered = _recover_unlocked(home)
        replacements = _bookkeeping_updates(home, bookkeeping)
        changed = _commit_replacements_unlocked(
            home, replacements, bookkeeping.failure_injector
        )
    return ApplyResult(changed, recovered_journal=recovered)


def apply_validated_stage(
    stage: ValidatedStage,
    baseline: Mapping[str, ManifestEntry],
    bookkeeping: ApplyBookkeeping,
) -> ApplyResult:
    """Atomically apply a validated stage and host bookkeeping under one lock."""
    if not isinstance(stage, ValidatedStage):
        raise StageValidationError("apply requires a validated stage")
    if dict(baseline) != dict(stage.before):
        raise StageValidationError("baseline does not match the validated stage")
    after_manifest = stage.root / _AFTER_MANIFEST_NAME
    if (
        after_manifest.is_symlink()
        or not after_manifest.is_file()
        or after_manifest.read_bytes() != _manifest_payload(stage.after)
    ):
        raise StageValidationError("stage after-manifest was modified")
    current_after = snapshot_manifest(stage.root)
    current_changed = tuple(
        sorted(
            path
            for path in set(stage.before) | set(current_after)
            if stage.before.get(path) != current_after.get(path)
        )
    )
    if current_after != dict(stage.after) or current_changed != stage.changed_paths:
        raise StageValidationError("stage changed after validation")

    home = stage.memory_home
    recovered = False
    with ExclusiveFileLock(home / "scripts" / "memory-writer.lock"):
        recovered = _recover_unlocked(home)
        for relative, expected in baseline.items():
            real = _safe_destination(home, relative)
            if not real.exists() or _entry(real.read_bytes()) != expected:
                raise RetryableApplyError(f"real baseline changed before apply: {relative}")
        for relative in stage.changed_paths:
            if relative not in baseline:
                real = _safe_destination(home, relative)
                if real.exists() or real.is_symlink():
                    raise RetryableApplyError(f"real baseline changed before apply: {relative}")

        replacements = {
            relative: _safe_destination(stage.root, relative, allow_missing=False).read_bytes()
            for relative in stage.changed_paths
        }
        replacements.update(_bookkeeping_updates(home, bookkeeping))
        changed = _commit_replacements_unlocked(
            home, replacements, bookkeeping.failure_injector
        )

    discard_stage(stage)
    return ApplyResult(changed, recovered_journal=recovered)
