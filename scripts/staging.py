"""Private staging, validation, and crash-safe knowledge workspace writes."""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
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
    from .utils import ExclusiveFileLock, FileBaseline, capture_file_baseline
except ImportError:  # Direct execution with scripts/ on sys.path.
    from config import ROOT_DIR
    from utils import ExclusiveFileLock, FileBaseline, capture_file_baseline


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
    job_id: str = "unknown"
    attempt_id: str = "unknown"
    relevant_articles: tuple[str, ...] = ()
    include_state: bool = False


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
    compiled_marker_baseline: FileBaseline | None = None
    compiled_at: str | None = None
    state: Mapping[str, object] | None = None
    state_path: str = "scripts/state.json"
    state_baseline: FileBaseline | None = None
    input_baselines: Mapping[str, FileBaseline] = field(default_factory=dict)
    extra_updates: Mapping[str, bytes | str] = field(default_factory=dict)
    failure_injector: Callable[[int, str], None] | None = None
    journal_transition_injector: Callable[[str], None] | None = None


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


def _validate_directory_identity(path: Path, *, mode: int | None = None) -> None:
    if path.is_symlink():
        raise StageValidationError(f"directory must not be a symlink: {path}")
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise StageValidationError(f"path must be a directory: {path}")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise StageValidationError(f"directory has an unsafe owner: {path}")
    if mode is not None and stat.S_IMODE(info.st_mode) != mode:
        raise StageValidationError(f"directory must have mode {mode:o}: {path}")


def _prepare_stage_parent(home: Path) -> Path:
    if not home.exists() or home.is_symlink():
        raise StageValidationError("memory home must be an existing non-symlink directory")
    _validate_directory_identity(home)
    scripts = home / "scripts"
    if scripts.exists() or scripts.is_symlink():
        _validate_directory_identity(scripts)
    else:
        scripts.mkdir(mode=0o700)
    stage_parent = scripts / "staging"
    if stage_parent.exists() or stage_parent.is_symlink():
        _validate_directory_identity(stage_parent)
        stage_parent.chmod(0o700)
    else:
        stage_parent.mkdir(mode=0o700)
    return stage_parent


def _validate_stage_location(stage: Stage) -> None:
    home = stage.memory_home
    _validate_directory_identity(home)
    scripts = home / "scripts"
    stage_parent = scripts / "staging"
    _validate_directory_identity(scripts)
    _validate_directory_identity(stage_parent)
    if stage.root.parent.absolute() != stage_parent.absolute():
        raise StageValidationError("stage root is outside the private staging directory")
    _validate_directory_identity(stage.root, mode=0o700)


def _read_safe_regular(root: Path, relative: str) -> bytes:
    """Read a root-relative regular file once through a no-follow descriptor."""
    target = _safe_destination(root, relative, allow_missing=False)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise StageValidationError(f"stage file could not be opened safely: {relative}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise StageValidationError(f"stage file must be regular: {relative}")
        if info.st_nlink != 1:
            raise StageValidationError(f"stage file must not be hard-linked: {relative}")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise StageValidationError(f"stage file has an unsafe owner: {relative}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


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
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("private file write made no progress")
            view = view[written:]
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
    include_state: bool = False,
) -> Stage:
    """Create a fresh owner-only stage containing the minimum requested files."""
    home = Path(os.path.abspath(Path(memory_home).expanduser()))
    normalized_daily = _relative_path(daily_source) if daily_source is not None else None
    if normalized_daily is not None and not normalized_daily.startswith("daily/"):
        raise StageValidationError("daily source must be below daily/")
    normalized_articles = tuple(_relative_path(path) for path in relevant_articles)
    for relative in normalized_articles:
        if not relative.endswith(".md") or not relative.startswith(_ARTICLE_PREFIXES):
            raise StageValidationError(
                "relevant article must be Markdown below knowledge/concepts, "
                "knowledge/connections, or knowledge/qa"
            )

    stage_parent = _prepare_stage_parent(home)
    stage_root = stage_parent / f"{_safe_identifier(job_id)}-{_safe_identifier(attempt_id)}"
    if stage_root.exists() or stage_root.is_symlink():
        raise StageValidationError(f"stage must be fresh: {stage_root.name}")
    stage_root.mkdir(mode=0o700)

    selected: list[str] = ["AGENTS.md", "knowledge/index.md", "knowledge/log.md"]
    if normalized_daily is not None:
        selected.append(normalized_daily)
    selected.extend(normalized_articles)
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
        job_id=str(job_id),
        attempt_id=str(attempt_id),
        relevant_articles=normalized_articles,
        include_state=include_state,
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


def _glob_parts_match(path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    if not pattern_parts:
        return not path_parts
    head, *tail = pattern_parts
    remaining = tuple(tail)
    if head == "**":
        return _glob_parts_match(path_parts, remaining) or bool(path_parts) and _glob_parts_match(
            path_parts[1:], pattern_parts
        )
    return bool(path_parts) and fnmatchcase(path_parts[0], head) and _glob_parts_match(
        path_parts[1:], remaining
    )


def _matches(path: str, patterns: Sequence[str]) -> bool:
    path_parts = PurePosixPath(_relative_path(path)).parts
    for raw_pattern in patterns:
        pattern = _relative_path(raw_pattern)
        if _glob_parts_match(path_parts, PurePosixPath(pattern).parts):
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


def _parse_yaml_scalar(raw: str, key: str, relative: str) -> str:
    value = raw.strip()
    if not value or value.lower() in {"null", "~"}:
        raise StageValidationError(f"article frontmatter {key} has an invalid scalar: {relative}")
    if value.startswith(("&", "*", "!", "<<")):
        raise StageValidationError(
            f"article frontmatter {key} uses forbidden YAML syntax: {relative}"
        )
    if value[0] == '"':
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise StageValidationError(
                f"article frontmatter {key} has malformed quotes: {relative}"
            ) from exc
        if not isinstance(decoded, str) or not decoded:
            raise StageValidationError(
                f"article frontmatter {key} must be a non-empty scalar: {relative}"
            )
        return decoded
    if value[0] == "'":
        if len(value) < 2 or value[-1] != "'":
            raise StageValidationError(
                f"article frontmatter {key} has malformed quotes: {relative}"
            )
        inner = value[1:-1]
        if "'" in inner.replace("''", "") or not inner:
            raise StageValidationError(
                f"article frontmatter {key} has malformed quotes: {relative}"
            )
        return inner.replace("''", "'")
    forbidden_prefixes = ("- ", "? ", "|", ">", "@", "`", "%", "---", "...")
    if (
        value.startswith(forbidden_prefixes)
        or any(character in value for character in "\"'{}[]")
        or ":" in value
        or " #" in value
        or re.search(r"(?:^|\s)[&*!]", value)
    ):
        raise StageValidationError(
            f"article frontmatter {key} uses unsupported YAML syntax: {relative}"
        )
    return value


def _parse_inline_yaml_list(raw: str, key: str, relative: str) -> list[str]:
    if not raw.endswith("]"):
        raise StageValidationError(f"article has malformed inline list for {key}: {relative}")
    inner = raw[1:-1]
    if not inner.strip():
        return []
    pieces: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(inner):
        character = inner[index]
        if quote == '"':
            current.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif quote == "'":
            current.append(character)
            if character == "'":
                if index + 1 < len(inner) and inner[index + 1] == "'":
                    current.append(inner[index + 1])
                    index += 1
                else:
                    quote = None
        elif character in {'"', "'"}:
            quote = character
            current.append(character)
        elif character == ",":
            piece = "".join(current).strip()
            if not piece:
                raise StageValidationError(
                    f"article has malformed inline list for {key}: {relative}"
                )
            pieces.append(piece)
            current = []
        else:
            current.append(character)
        index += 1
    piece = "".join(current).strip()
    if quote is not None or escaped or not piece:
        raise StageValidationError(f"article has malformed inline list for {key}: {relative}")
    pieces.append(piece)
    return [_parse_yaml_scalar(piece, key, relative) for piece in pieces]


def _validate_frontmatter(relative: str, content: str) -> None:
    if not content.startswith("---\n"):
        raise StageValidationError(f"article has malformed frontmatter: {relative}")
    boundary = content.find("\n---\n", 4)
    if boundary < 0:
        raise StageValidationError(f"article has malformed frontmatter: {relative}")
    frontmatter = content[4:boundary]
    lines = frontmatter.splitlines()
    values: dict[str, str | list[str]] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            index += 1
            continue
        match = re.match(r"^([A-Za-z][A-Za-z0-9_-]*):(?:\s*(.*))$", line)
        if match is None:
            raise StageValidationError(f"article has malformed frontmatter: {relative}")
        key, inline = match.groups()
        if key in values:
            raise StageValidationError(f"article has duplicate frontmatter key {key}: {relative}")
        if inline:
            normalized = inline.strip()
            if normalized.startswith("["):
                values[key] = _parse_inline_yaml_list(normalized, key, relative)
            else:
                values[key] = _parse_yaml_scalar(normalized, key, relative)
            index += 1
            continue
        items: list[str] = []
        index += 1
        while index < len(lines) and lines[index].startswith((" ", "\t")):
            nested = lines[index]
            if "\t" in nested or re.match(r"^  -\s+\S", nested) is None:
                raise StageValidationError(f"article has malformed list for {key}: {relative}")
            items.append(_parse_yaml_scalar(nested.split("-", 1)[1], key, relative))
            index += 1
        values[key] = items

    required = {"title", "sources", "created", "updated"}
    if relative.startswith("knowledge/concepts/"):
        required.add("project")
    elif relative.startswith("knowledge/connections/"):
        required.update({"project", "connects"})
    elif relative.startswith("knowledge/qa/"):
        required = {"title", "question", "consulted", "filed"}
    missing = [name for name in sorted(required) if not values.get(name)]
    if missing:
        raise StageValidationError(
            f"article frontmatter missing {', '.join(missing)}: {relative}"
        )
    list_minimums = {"sources": 1, "connects": 2, "consulted": 1}
    for key, minimum in list_minimums.items():
        if key not in required:
            continue
        value = values.get(key)
        if not isinstance(value, list) or len(value) < minimum:
            raise StageValidationError(
                f"article frontmatter {key} must be a non-empty YAML list: {relative}"
            )
    scalar_fields = required & {"title", "question", "created", "updated", "filed"}
    for key in sorted(scalar_fields):
        value = values.get(key)
        if not isinstance(value, str) or not value:
            raise StageValidationError(
                f"article frontmatter {key} must be a non-empty YAML scalar: {relative}"
            )
    if "project" in required:
        project = values.get("project")
        if not (
            isinstance(project, str)
            and bool(project)
            or isinstance(project, list)
            and bool(project)
        ):
            raise StageValidationError(
                f"article frontmatter project must be a non-empty YAML scalar or list: {relative}"
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


def _validate_index_structure(content: str) -> None:
    rows = [line for line in content.splitlines() if line.startswith("|")]
    if len(rows) < 2:
        raise StageValidationError("knowledge index is malformed")
    parsed = [[column.strip() for column in row.strip("|").split("|")] for row in rows]
    if any(len(columns) != 5 or not all(columns) for columns in parsed):
        raise StageValidationError("knowledge index is malformed")
    expected_header = ["article", "project", "summary", "compiled from", "updated"]
    if [column.lower() for column in parsed[0]] != expected_header:
        raise StageValidationError("knowledge index header is malformed")
    if any(re.fullmatch(r":?-{3,}:?", column) is None for column in parsed[1]):
        raise StageValidationError("knowledge index separator is malformed")
    if any(re.fullmatch(r"\[\[[^\]]+\]\]", columns[0]) is None for columns in parsed[2:]):
        raise StageValidationError("knowledge index article row is malformed")


def _validate_log_append(baseline: str, content: str, task: str) -> list[str]:
    if not content.startswith(baseline):
        raise StageValidationError("knowledge log must remain an exact append-only prefix")
    appended = content[len(baseline):]
    added_lines = appended.splitlines()
    nonblank = [line for line in added_lines if line.strip()]
    task_suffix = r"\s+\|\s+\S+" if task == "compile" else r"(?:\s+\|\s+\S+)?"
    heading = rf"##\s+\[[^\]]+\]\s+{re.escape(task)}{task_suffix}"
    if not appended or not nonblank or re.fullmatch(heading, nonblank[0]) is None:
        raise StageValidationError("knowledge build log appended entry is malformed")
    return added_lines


def _validate_task_contract(
    stage: Stage,
    after: Mapping[str, ManifestEntry],
    changed: tuple[str, ...],
    task: str,
) -> None:
    all_articles = tuple(
        path for path in after if path.endswith(".md") and path.startswith(_ARTICLE_PREFIXES)
    )
    article_changes = tuple(
        path for path in changed if path.endswith(".md") and path.startswith(_ARTICLE_PREFIXES)
    )
    for relative in all_articles:
        content = (stage.root / relative).read_text(encoding="utf-8")
        _validate_frontmatter(relative, content)

    index = (stage.root / "knowledge/index.md").read_text(encoding="utf-8")
    build_log = (stage.root / "knowledge/log.md").read_text(encoding="utf-8")
    baseline_index = stage.baseline_bytes.get("knowledge/index.md", b"").decode("utf-8")
    baseline_log = stage.baseline_bytes.get("knowledge/log.md", b"").decode("utf-8")
    if "knowledge/index.md" in changed:
        _validate_index_structure(index)
    added_log_lines: list[str] = []
    if "knowledge/log.md" in changed:
        added_log_lines = _validate_log_append(baseline_log, build_log, task)

    if not article_changes:
        if task in {"file_answer", "connections"}:
            raise StageValidationError(f"{task} requires an article")
        if task == "compile" and "knowledge/log.md" not in changed:
            raise StageValidationError("compile requires a valid build log update")
        return

    if "knowledge/index.md" not in changed:
        raise StageValidationError(f"{task} article changes require an index update")
    if "knowledge/log.md" not in changed:
        raise StageValidationError(f"{task} article changes require a build log update")

    added_index_lines = _added_lines(baseline_index, index)
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
    _validate_stage_location(stage)
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
    if "AGENTS.md" in changed:
        raise StageValidationError("AGENTS.md schema must remain unchanged")
    if "scripts/state.json" in changed:
        raise StageValidationError("scripts/state.json is host-owned and cannot be model-edited")
    if stage.daily_source is not None and stage.daily_source in changed:
        raise StageValidationError("daily source content cannot be model-edited")
    forbidden = tuple(path for path in changed if not _matches(path, allowed_paths))
    if forbidden:
        raise StageValidationError(f"changed paths outside allowlist: {', '.join(forbidden)}")
    unexpected_knowledge = tuple(
        path
        for path in changed
        if path.startswith("knowledge/")
        and path not in {"knowledge/index.md", "knowledge/log.md"}
        and not path.startswith(_ARTICLE_PREFIXES)
    )
    if unexpected_knowledge:
        raise StageValidationError(
            f"unexpected knowledge paths: {', '.join(unexpected_knowledge)}"
        )
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


def create_fallback_stage(failed_stage: Stage, *, attempt_id: object) -> Stage:
    """Discard a contaminated attempt and rebuild a fallback from host files."""
    memory_home = failed_stage.memory_home
    job_id = failed_stage.job_id
    daily_source = failed_stage.daily_source
    relevant_articles = failed_stage.relevant_articles
    include_state = failed_stage.include_state
    next_name = f"{_safe_identifier(job_id)}-{_safe_identifier(attempt_id)}"
    if next_name == failed_stage.root.name:
        raise StageValidationError("fallback attempt must use a fresh stage identity")
    discard_stage(failed_stage)
    return create_stage(
        memory_home,
        job_id,
        attempt_id,
        daily_source=daily_source,
        relevant_articles=relevant_articles,
        include_state=include_state,
    )


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


def _atomic_replace(path: Path, data: bytes, *, mode: int | None = None) -> None:
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise StageValidationError(f"destination parent must not be a symlink: {path.parent}")
    if not parent_existed:
        path.parent.chmod(0o700)
    if path.is_symlink():
        raise StageValidationError(f"destination must not be a symlink: {path}")
    replacement_mode = 0o600 if mode is None else mode
    if path.exists():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise StageValidationError(f"destination has an unsafe identity: {path}")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise StageValidationError(f"destination has an unsafe owner: {path}")
        if mode is None:
            replacement_mode = stat.S_IMODE(info.st_mode)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, replacement_mode)
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


def _atomic_journal_transition(path: Path, data: bytes) -> None:
    """Publish one complete journal state without truncating the prior state."""
    _private_directory(path.parent)
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise StageValidationError("journal has an unsafe identity")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise StageValidationError("journal has an unsafe owner")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("journal write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _journal_bytes(
    home: Path,
    state: str,
    operations: Sequence[tuple[str, bytes | None, bytes, int | None]],
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
                        "original_mode": original_mode,
                        "replacement": replacement.hex(),
                    }
                    for path, original, replacement, original_mode in operations
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
            stored_mode = entry.get("original_mode")
            _atomic_replace(
                destination,
                bytes.fromhex(str(original)),
                mode=int(stored_mode) if stored_mode is not None else None,
            )


def recover_incomplete_apply_unlocked(memory_home: Path | str) -> bool:
    """Recover a journal while the caller already owns the writer lock."""
    home = Path(memory_home).expanduser().resolve()
    directory = home / "scripts" / "memory-apply-journal"
    if not directory.exists():
        return False
    _validate_directory_identity(directory)
    directory.chmod(0o700)
    for orphan in sorted(directory.glob(".*.tmp")):
        orphan.unlink(missing_ok=True)
    _fsync_directory(directory)
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


def has_incomplete_apply(memory_home: Path | str) -> bool:
    """Read-only journal presence check for diagnostic/dry-run callers."""
    home = Path(memory_home).expanduser().resolve()
    directory = home / "scripts" / "memory-apply-journal"
    if not directory.exists():
        return False
    if directory.is_symlink() or not directory.is_dir():
        raise StageValidationError("journal directory has an unsafe identity")
    return any(directory.glob("*.json")) or any(directory.glob(".*.tmp"))


def recover_incomplete_apply(memory_home: Path | str | None = None) -> bool:
    """Restore any prepared/applying transaction left by an interrupted writer."""
    home = Path(memory_home or ROOT_DIR).expanduser().resolve()
    with ExclusiveFileLock(home / "scripts" / "memory-writer.lock"):
        return recover_incomplete_apply_unlocked(home)


def _verify_state_baseline(home: Path, bookkeeping: ApplyBookkeeping) -> None:
    if bookkeeping.state is None:
        if bookkeeping.state_baseline is not None:
            raise StageValidationError("state baseline requires a state update")
        return
    if bookkeeping.state_baseline is None:
        raise StageValidationError("state update requires an exact state baseline")
    state_path = _safe_destination(home, _relative_path(bookkeeping.state_path))
    current = capture_file_baseline(state_path)
    expected = bookkeeping.state_baseline
    if not _same_baseline(current, expected):
        raise RetryableApplyError("state baseline changed before apply")


def _same_baseline(current: FileBaseline, expected: FileBaseline) -> bool:
    return (current.exists, current.size, current.sha256) == (
        expected.exists,
        expected.size,
        expected.sha256,
    )


def _verify_input_baselines(home: Path, bookkeeping: ApplyBookkeeping) -> None:
    normalized_inputs = {_relative_path(path) for path in bookkeeping.input_baselines}
    for raw_path in bookkeeping.extra_updates:
        relative = _relative_path(raw_path)
        if relative not in normalized_inputs:
            raise StageValidationError(
                f"extra update requires an exact destination baseline: {relative}"
            )
    for raw_path, expected in bookkeeping.input_baselines.items():
        relative = _relative_path(raw_path)
        current = capture_file_baseline(_safe_destination(home, relative))
        if not _same_baseline(current, expected):
            raise RetryableApplyError(f"input baseline changed before apply: {relative}")

    if bookkeeping.compiled_marker_path is None:
        if bookkeeping.compiled_marker_baseline is not None:
            raise StageValidationError("compiled marker baseline requires a compiled marker")
        return
    if bookkeeping.compiled_marker_baseline is None:
        raise StageValidationError("compiled marker requires an exact file baseline")
    relative = _relative_path(bookkeeping.compiled_marker_path)
    current = capture_file_baseline(_safe_destination(home, relative))
    if not _same_baseline(current, bookkeeping.compiled_marker_baseline):
        raise RetryableApplyError("compiled marker baseline changed before apply")


def _bookkeeping_updates(
    home: Path, bookkeeping: ApplyBookkeeping
) -> dict[str, bytes]:
    updates: dict[str, bytes] = {}
    marker_relative: str | None = None
    if bookkeeping.compiled_marker_path is not None:
        relative = _relative_path(bookkeeping.compiled_marker_path)
        marker_relative = relative
        if not bookkeeping.compiled_at:
            raise StageValidationError("compiled_at is required with a compiled marker")
        updates[relative] = _read_safe_regular(home, relative) + (
            f"\n<!-- @compiled-through:{bookkeeping.compiled_at} -->\n"
        ).encode()
    if bookkeeping.state is not None:
        relative = _relative_path(bookkeeping.state_path)
        if relative in updates:
            raise StageValidationError(f"duplicate bookkeeping destination: {relative}")
        next_state = copy.deepcopy(bookkeeping.state)
        if marker_relative is not None:
            ingested = next_state.get("ingested")
            entry = (
                ingested.get(PurePosixPath(marker_relative).name)
                if isinstance(ingested, dict)
                else None
            )
            if not isinstance(entry, dict):
                raise StageValidationError(
                    "compiled marker state requires an ingested entry for the daily file"
                )
            entry["hash"] = _sha256(updates[marker_relative])[:16]
        updates[relative] = (json.dumps(next_state, indent=2) + "\n").encode()
    for raw_path, value in bookkeeping.extra_updates.items():
        relative = _relative_path(raw_path)
        if relative == _relative_path(bookkeeping.state_path):
            raise StageValidationError(
                "state updates must use state with an exact state baseline"
            )
        if relative in updates:
            raise StageValidationError(f"duplicate bookkeeping destination: {relative}")
        updates[relative] = value.encode() if isinstance(value, str) else bytes(value)
    return updates


def _commit_replacements_unlocked(
    home: Path,
    replacements: Mapping[str, bytes],
    failure_injector: Callable[[int, str], None] | None = None,
    journal_transition_injector: Callable[[str], None] | None = None,
) -> tuple[str, ...]:
    operations: list[tuple[str, bytes | None, bytes, int | None]] = []
    for relative, replacement in sorted(replacements.items()):
        destination = _safe_destination(home, relative)
        if destination.exists():
            original = _read_safe_regular(home, relative)
            original_mode = stat.S_IMODE(destination.lstat().st_mode)
        else:
            original = None
            original_mode = None
        operations.append((relative, original, replacement, original_mode))

    journal = _journal_path(home)
    _atomic_journal_transition(journal, _journal_bytes(home, "prepared", operations))
    if journal_transition_injector is not None:
        journal_transition_injector("prepared")
    try:
        _atomic_journal_transition(journal, _journal_bytes(home, "applying", operations))
        if journal_transition_injector is not None:
            journal_transition_injector("applying")
        for step, (relative, _original, replacement, _mode) in enumerate(operations, 1):
            _atomic_replace(_safe_destination(home, relative), replacement)
            if failure_injector is not None:
                failure_injector(step, relative)
        _atomic_journal_transition(journal, _journal_bytes(home, "complete", operations))
        if journal_transition_injector is not None:
            journal_transition_injector("complete")
        _remove_journal(journal)
    except Exception as exc:
        try:
            _restore_operations(
                home,
                [
                    {
                        "path": relative,
                        "original": original.hex() if original is not None else None,
                        "original_mode": original_mode,
                    }
                    for relative, original, _replacement, original_mode in operations
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
        recovered = recover_incomplete_apply_unlocked(home)
        _verify_input_baselines(home, bookkeeping)
        _verify_state_baseline(home, bookkeeping)
        replacements = _bookkeeping_updates(home, bookkeeping)
        changed = _commit_replacements_unlocked(
            home,
            replacements,
            bookkeeping.failure_injector,
            bookkeeping.journal_transition_injector,
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
    _validate_stage_location(stage.stage)
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
    stage_replacements: dict[str, bytes] = {}
    for relative in stage.changed_paths:
        data = _read_safe_regular(stage.root, relative)
        expected = stage.after.get(relative)
        if expected is None or _entry(data) != expected:
            raise StageValidationError(f"stage file changed during apply: {relative}")
        stage_replacements[relative] = data
    _validate_stage_location(stage.stage)

    home = stage.memory_home
    recovered = False
    with ExclusiveFileLock(home / "scripts" / "memory-writer.lock"):
        recovered = recover_incomplete_apply_unlocked(home)
        for relative, expected in baseline.items():
            real = _safe_destination(home, relative)
            if not real.exists() or _entry(real.read_bytes()) != expected:
                raise RetryableApplyError(f"real baseline changed before apply: {relative}")
        for relative in stage.changed_paths:
            if relative not in baseline:
                real = _safe_destination(home, relative)
                if real.exists() or real.is_symlink():
                    raise RetryableApplyError(f"real baseline changed before apply: {relative}")

        _verify_input_baselines(home, bookkeeping)
        _verify_state_baseline(home, bookkeeping)

        replacements = dict(stage_replacements)
        host_updates = _bookkeeping_updates(home, bookkeeping)
        collisions = sorted(set(replacements) & set(host_updates))
        if collisions:
            raise StageValidationError(
                f"stage and bookkeeping destination collision: {', '.join(collisions)}"
            )
        replacements.update(host_updates)
        changed = _commit_replacements_unlocked(
            home,
            replacements,
            bookkeeping.failure_injector,
            bookkeeping.journal_transition_injector,
        )

    discard_stage(stage)
    return ApplyResult(changed, recovered_journal=recovered)
