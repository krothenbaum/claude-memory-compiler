"""Shared utilities for the personal knowledge base."""

from contextlib import contextmanager
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
from typing import Callable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows branch.
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX branch.
    msvcrt = None

_PRIVATE_STATE_DIR_FD_SUPPORTED = bool(
    os.name != "nt"
    and getattr(os, "O_NOFOLLOW", 0)
    and getattr(os, "O_DIRECTORY", 0)
    and hasattr(os, "fchmod")
    and os.open in getattr(os, "supports_dir_fd", set())
    and os.stat in getattr(os, "supports_dir_fd", set())
    and os.unlink in getattr(os, "supports_dir_fd", set())
)

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


def _validate_private_regular_file(info: os.stat_result, path: Path) -> None:
    """Validate an owner-only state file without following links."""
    if _link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"state path must be a private regular file: {path}")
    if info.st_nlink != 1:
        raise ValueError(f"state path must be a private regular file: {path}")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise ValueError(f"state path must be a private regular file: {path}")
    if os.name != "nt" and stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError(f"state path must be a private regular file: {path}")


def read_private_bounded_file(path: Path | str, *, max_bytes: int) -> bytes | None:
    """Read a small owner-only regular file, retaining its verified descriptor."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    target = Path(os.path.abspath(Path(path).expanduser()))
    if _PRIVATE_STATE_DIR_FD_SUPPORTED:
        return _read_private_bounded_file_posix(target, max_bytes=max_bytes)
    return _read_private_bounded_file_fallback(target, max_bytes=max_bytes)


def _read_private_bounded_file_posix(
    target: Path,
    *,
    max_bytes: int,
) -> bytes | None:
    parent = target.parent
    try:
        parent_before = _inspect_private_state_parent(parent)
    except FileNotFoundError:
        return None
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_descriptor = os.open(parent, directory_flags)
    descriptor = -1
    try:
        parent_opened = os.fstat(parent_descriptor)
        parent_after = parent.lstat()
        _validate_private_state_directory(parent_opened, parent)
        _validate_private_state_directory(parent_after, parent)
        if not _same_file_identity(parent_before, parent_opened) or (
            parent_opened.st_dev,
            parent_opened.st_ino,
        ) != (parent_after.st_dev, parent_after.st_ino):
            raise ValueError(f"state directory identity changed: {parent}")
        before = _relative_stat(parent_descriptor, target.name)
        if before is None:
            return None
        _validate_private_regular_file(before, target)
        if before.st_size > max_bytes:
            raise ValueError(f"state path exceeds {max_bytes} bytes: {target}")
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(target.name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        _validate_private_regular_file(opened, target)
        if not _same_file_identity(before, opened):
            raise ValueError(f"state path identity changed while opening: {target}")
        if opened.st_size > max_bytes:
            raise ValueError(f"state path exceeds {max_bytes} bytes: {target}")
        visible = _relative_stat(parent_descriptor, target.name)
        if visible is None:
            raise ValueError(f"state path identity changed while reading: {target}")
        _validate_private_regular_file(visible, target)
        if not _same_file_identity(opened, visible):
            raise ValueError(f"state path identity changed while reading: {target}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"state path exceeds {max_bytes} bytes: {target}")
        return data
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _read_private_bounded_file_fallback(
    target: Path,
    *,
    max_bytes: int,
) -> bytes | None:
    try:
        _inspect_private_state_parent(target.parent)
        before = target.lstat()
    except FileNotFoundError:
        return None
    _validate_private_regular_file(before, target)
    if before.st_size > max_bytes:
        raise ValueError(f"state path exceeds {max_bytes} bytes: {target}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(target, flags)
    except OSError as error:
        raise ValueError(f"state path must be a private regular file: {target}") from error
    try:
        opened = os.fstat(descriptor)
        _validate_private_regular_file(opened, target)
        if not _same_file_identity(before, opened):
            raise ValueError(f"state path identity changed while opening: {target}")
        if _windows_acl_required():
            _validate_windows_owner_only_file_descriptor(descriptor, target)
        visible = target.lstat()
        _validate_private_regular_file(visible, target)
        if not _same_file_identity(opened, visible):
            raise ValueError(f"state path identity changed while reading: {target}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"state path exceeds {max_bytes} bytes: {target}")
        return data
    finally:
        os.close(descriptor)


def _validate_private_state_directory(info: os.stat_result, path: Path) -> None:
    if _link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"state directory must be a real directory: {path}")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise ValueError(f"state directory has an unsafe owner: {path}")
    if os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o022:
        raise ValueError(f"state directory has unsafe permissions: {path}")


def _validate_no_linked_ancestors(path: Path) -> None:
    current = path
    while True:
        try:
            info = current.lstat()
        except FileNotFoundError:
            pass
        else:
            if _link_or_reparse(info):
                raise ValueError(f"state path has a linked ancestor: {current}")
        if current.parent == current:
            return
        current = current.parent


def _prepare_private_state_parent(parent: Path) -> os.stat_result:
    _validate_no_linked_ancestors(parent)
    created = False
    try:
        info = parent.lstat()
    except FileNotFoundError:
        parent.mkdir(parents=True, mode=0o700)
        created = True
        _validate_no_linked_ancestors(parent)
        info = parent.lstat()
    _validate_private_state_directory(info, parent)
    if _windows_acl_required():
        if created:
            _secure_windows_runtime_directory(parent, owner_only=True)
        _validate_windows_owner_only_directory(parent)
    return info


def _inspect_private_state_parent(parent: Path) -> os.stat_result:
    _validate_no_linked_ancestors(parent)
    info = parent.lstat()
    _validate_private_state_directory(info, parent)
    if _windows_acl_required():
        _validate_windows_owner_only_directory(parent)
    return info


def _relative_stat(directory_descriptor: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _write_private_temp_posix(
    parent_descriptor: int,
    target: Path,
    data: bytes,
) -> tuple[str, int, os.stat_result]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    temporary_descriptor = -1
    temporary_name = ""
    try:
        for _attempt in range(100):
            temporary_name = f".{target.name}.{uuid.uuid4().hex}.tmp"
            try:
                temporary_descriptor = os.open(
                    temporary_name,
                    flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
                break
            except FileExistsError:
                continue
        else:  # pragma: no cover - collision probability is negligible.
            raise FileExistsError("could not allocate private state temporary file")
        os.fchmod(temporary_descriptor, 0o600)
        with os.fdopen(temporary_descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_opened = os.fstat(temporary_descriptor)
        temporary_visible = _relative_stat(parent_descriptor, temporary_name)
        if temporary_visible is None:
            raise ValueError("state temporary identity changed before replacement")
        _validate_private_regular_file(temporary_opened, target.parent / temporary_name)
        _validate_private_regular_file(temporary_visible, target.parent / temporary_name)
        if not _same_file_identity(temporary_opened, temporary_visible):
            raise ValueError("state temporary identity changed before replacement")
        return temporary_name, temporary_descriptor, temporary_opened
    except BaseException:
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        raise


def _restore_private_state_posix(
    parent_descriptor: int,
    target: Path,
    prior_bytes: bytes | None,
) -> None:
    if prior_bytes is None:
        try:
            os.unlink(target.name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.fsync(parent_descriptor)
        return
    restore_name = ""
    restore_descriptor = -1
    try:
        restore_name, restore_descriptor, restore_info = _write_private_temp_posix(
            parent_descriptor,
            target,
            prior_bytes,
        )
        os.replace(
            restore_name,
            target.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        restore_name = ""
        restored = _relative_stat(parent_descriptor, target.name)
        if restored is None or (
            restored.st_dev,
            restored.st_ino,
        ) != (restore_info.st_dev, restore_info.st_ino):
            raise ValueError("state restoration identity changed after replacement")
        os.fsync(parent_descriptor)
    finally:
        if restore_name:
            try:
                os.unlink(restore_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        if restore_descriptor >= 0:
            os.close(restore_descriptor)


def _atomic_write_private_file_posix(
    target: Path,
    data: bytes,
    *,
    max_bytes: int,
) -> None:
    parent = target.parent
    parent_before = _prepare_private_state_parent(parent)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_descriptor = os.open(parent, directory_flags)
    temporary_descriptor = -1
    temporary_name = ""
    prior_descriptor = -1
    prior_bytes: bytes | None = None
    try:
        parent_opened = os.fstat(parent_descriptor)
        parent_after = parent.lstat()
        _validate_private_state_directory(parent_opened, parent)
        _validate_private_state_directory(parent_after, parent)
        if not _same_file_identity(parent_before, parent_opened) or (
            parent_opened.st_dev,
            parent_opened.st_ino,
        ) != (parent_after.st_dev, parent_after.st_ino):
            raise ValueError(f"state directory identity changed: {parent}")

        target_before = _relative_stat(parent_descriptor, target.name)
        if target_before is not None:
            _validate_private_regular_file(target_before, target)
            flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
            prior_descriptor = os.open(
                target.name,
                flags,
                dir_fd=parent_descriptor,
            )
            prior_opened = os.fstat(prior_descriptor)
            _validate_private_regular_file(prior_opened, target)
            if not _same_file_identity(target_before, prior_opened):
                raise ValueError("state destination identity changed while opening")
            with os.fdopen(prior_descriptor, "rb", closefd=False) as stream:
                prior_bytes = stream.read(max_bytes + 1)
            if len(prior_bytes) > max_bytes:
                raise ValueError(f"state path exceeds {max_bytes} bytes: {target}")

        temporary_name, temporary_descriptor, temporary_opened = (
            _write_private_temp_posix(parent_descriptor, target, data)
        )

        target_current = _relative_stat(parent_descriptor, target.name)
        if target_before is None:
            if target_current is not None:
                raise ValueError("state destination identity changed before replacement")
        elif target_current is None or not _same_file_identity(
            target_before, target_current
        ):
            raise ValueError("state destination identity changed before replacement")

        os.replace(
            temporary_name,
            target.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        destination = _relative_stat(parent_descriptor, target.name)
        if destination is None or not _same_file_identity(
            temporary_opened, destination
        ):
            _restore_private_state_posix(parent_descriptor, target, prior_bytes)
            raise ValueError("state destination identity changed after replacement")
        temporary_name = ""
        os.fsync(parent_descriptor)
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if prior_descriptor >= 0:
            os.close(prior_descriptor)
        os.close(parent_descriptor)


def _restore_private_state_fallback(target: Path, prior_bytes: bytes | None) -> None:
    if prior_bytes is None:
        target.unlink(missing_ok=True)
        _fsync_directory(target.parent)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".restore.tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        if _windows_acl_required():
            _secure_windows_runtime_file(descriptor, temporary)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(prior_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        opened = os.fstat(descriptor)
        visible = temporary.lstat()
        _validate_private_regular_file(opened, temporary)
        _validate_private_regular_file(visible, temporary)
        if not _same_file_identity(opened, visible):
            raise ValueError("state restoration identity changed before replacement")
        os.replace(temporary, target)
        restored = target.lstat()
        if (restored.st_dev, restored.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("state restoration identity changed after replacement")
        if _windows_acl_required():
            _validate_windows_owner_only_file_descriptor(descriptor, target)
        _fsync_directory(target.parent)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_write_private_file_fallback(
    target: Path,
    data: bytes,
    *,
    max_bytes: int,
) -> None:
    parent = target.parent
    parent_before = _prepare_private_state_parent(parent)
    target_before = target.lstat() if target.exists() or target.is_symlink() else None
    prior_bytes: bytes | None = None
    if target_before is not None:
        _validate_private_regular_file(target_before, target)
        prior_bytes = _read_private_bounded_file_fallback(
            target,
            max_bytes=max_bytes,
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        if _windows_acl_required():
            _secure_windows_runtime_file(descriptor, temporary)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_opened = os.fstat(descriptor)
        temporary_visible = temporary.lstat()
        _validate_private_regular_file(temporary_opened, temporary)
        _validate_private_regular_file(temporary_visible, temporary)
        parent_after = parent.lstat()
        _validate_private_state_directory(parent_after, parent)
        if not _same_file_identity(parent_before, parent_after):
            raise ValueError(f"state directory identity changed: {parent}")
        if not _same_file_identity(temporary_opened, temporary_visible):
            raise ValueError("state temporary identity changed before replacement")
        target_current = target.lstat() if target.exists() or target.is_symlink() else None
        if target_before is None:
            if target_current is not None:
                raise ValueError("state destination identity changed before replacement")
        elif target_current is None or not _same_file_identity(
            target_before, target_current
        ):
            raise ValueError("state destination identity changed before replacement")
        os.replace(temporary, target)
        destination = target.lstat()
        try:
            if not _same_file_identity(temporary_opened, destination):
                raise ValueError("state destination identity changed after replacement")
            if _windows_acl_required():
                _validate_windows_owner_only_file_descriptor(descriptor, target)
        except BaseException:
            _restore_private_state_fallback(target, prior_bytes)
            raise
        _fsync_directory(parent)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def atomic_write_private_file(
    path: Path | str,
    data: bytes,
    *,
    max_bytes: int,
) -> None:
    """Fsync and atomically replace one bounded owner-only state file."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if len(data) > max_bytes:
        raise ValueError(f"state payload exceeds {max_bytes} bytes")
    target = Path(os.path.abspath(Path(path).expanduser()))
    if _PRIVATE_STATE_DIR_FD_SUPPORTED:
        _atomic_write_private_file_posix(target, data, max_bytes=max_bytes)
    else:  # pragma: no cover - exercised on descriptor-limited platforms.
        _atomic_write_private_file_fallback(target, data, max_bytes=max_bytes)


def _validate_log_directory(info: os.stat_result, path: Path) -> None:
    if _link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(
            f"log directory must be a non-symlink, non-reparse directory: {path}"
        )
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise ValueError(f"log directory has an unsafe owner: {path}")


def _validate_log_file(info: os.stat_result, path: Path) -> None:
    if _link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
        raise ValueError(
            f"log path must be a regular non-symlink, non-reparse file: {path}"
        )
    if info.st_nlink != 1:
        raise ValueError(f"log path must not be hard-linked: {path}")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise ValueError(f"log path has an unsafe owner: {path}")


def _same_file_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _link_or_reparse(info: os.stat_result) -> bool:
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse and attributes & reparse)


def prepare_secure_log_directory(memory_root: Path | str) -> Path:
    """Create and validate the runtime ``scripts/logs`` directory.

    Existing project directories keep their modes. Newly created runtime
    directories are private, and every component inside the configured memory
    boundary must be owner-controlled and must not be a link/reparse point.
    """
    root = Path(os.path.abspath(Path(memory_root).expanduser()))
    if not root.exists() and not root.is_symlink():
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            pass
    _validate_log_directory(root.lstat(), root)

    current = root
    for name in ("scripts", "logs"):
        candidate = current / name
        if not candidate.exists() and not candidate.is_symlink():
            try:
                candidate.mkdir(mode=0o700)
            except FileExistsError:
                pass
        _validate_log_directory(candidate.lstat(), candidate)
        current = candidate
    try:
        current.chmod(0o700)
    except OSError:
        pass
    return current


def _bootstrap_secure_runtime_directory_posix(memory_root: Path | str) -> Path:
    """Bootstrap paths only; descriptor validation happens during file open."""
    root = Path(os.path.abspath(Path(memory_root).expanduser()))
    if not root.exists() and not root.is_symlink():
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            pass
    _validate_log_directory(root.lstat(), root)

    current = root
    for name in ("scripts", "runtime"):
        candidate = current / name
        if not candidate.exists() and not candidate.is_symlink():
            try:
                candidate.mkdir(mode=0o700)
            except FileExistsError:
                pass
        _validate_log_directory(candidate.lstat(), candidate)
        current = candidate
    try:
        current.chmod(0o700)
    except OSError:
        pass
    return current


def _validate_runtime_directory(
    info: os.stat_result, path: Path, *, private: bool = False
) -> None:
    _validate_log_directory(info, path)
    if os.name != "nt":
        mode = stat.S_IMODE(info.st_mode)
        if mode & 0o022:
            raise ValueError(f"runtime directory has unsafe permissions: {path}")
        if private and mode != 0o700:
            raise ValueError(f"runtime directory must have mode 0700: {path}")


def _validate_runtime_file(info: os.stat_result, path: Path) -> None:
    if _link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"runtime path must be a regular non-reparse file: {path}")
    if info.st_nlink != 1:
        raise ValueError(f"runtime path must not be hard-linked: {path}")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise ValueError(f"runtime path has an unsafe owner: {path}")
    if os.name != "nt" and stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError(f"runtime path must have mode 0600: {path}")


def validate_secure_runtime_file(path: Path | str, descriptor: int) -> None:
    """Prove that a preview path still names the retained private descriptor."""
    target = Path(path)
    opened = os.fstat(descriptor)
    visible = target.lstat()
    _validate_runtime_file(opened, target)
    _validate_runtime_file(visible, target)
    if not _same_file_identity(opened, visible):
        raise ValueError(f"runtime path identity changed: {target}")


def _validate_runtime_parent_identities(
    root: Path,
    scripts: Path,
    runtime: Path,
    descriptors: tuple[int, int, int],
) -> None:
    for path, descriptor, private in zip(
        (root, scripts, runtime), descriptors, (False, False, True), strict=True
    ):
        opened = os.fstat(descriptor)
        visible = path.lstat()
        _validate_runtime_directory(opened, path, private=private)
        _validate_runtime_directory(visible, path, private=private)
        if not _same_file_identity(opened, visible):
            raise ValueError(f"runtime directory identity changed: {path}")


def _runtime_dir_fd_supported() -> bool:
    return bool(
        getattr(os, "O_NOFOLLOW", 0)
        and getattr(os, "O_DIRECTORY", 0)
        and hasattr(os, "fchmod")
        and os.open in getattr(os, "supports_dir_fd", set())
        and os.unlink in getattr(os, "supports_dir_fd", set())
    )


def _open_runtime_directory_nofollow(path: Path) -> int | None:
    """Pin a directory by full path when relative ``dir_fd`` calls are absent."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags)
    except OSError:
        if os.name == "nt":  # Windows cannot open directory handles with os.open.
            return None
        raise


def _windows_acl_required() -> bool:
    return os.name == "nt"


def _validate_windows_owner_only_directory(path: Path) -> None:
    try:
        from scripts.windows_acl import _active_api
    except ModuleNotFoundError:  # Standalone imports from inside scripts/.
        from windows_acl import _active_api

    api = _active_api(None)
    try:
        handle = api.open_directory(path)
    except Exception as error:
        raise PermissionError(f"could not open Windows state directory: {path}") from error
    try:
        if api.is_reparse(handle):
            raise PermissionError(f"Windows state directory is a reparse point: {path}")
        identity = api.identity(handle)
        try:
            state = api.inspect(handle)
        except Exception as error:
            raise PermissionError(
                f"could not inspect Windows state directory ACL: {path}"
            ) from error
        if not state.is_owner_only:
            raise PermissionError(
                f"Windows state directory ACL is not owner-only: {path}"
            )
        try:
            observed = api.open_directory(path)
        except Exception as error:
            raise PermissionError(
                f"could not reopen Windows state directory: {path}"
            ) from error
        try:
            if api.is_reparse(observed) or api.identity(observed) != identity:
                raise PermissionError(
                    f"Windows state directory identity changed: {path}"
                )
            if not api.inspect(observed).is_owner_only:
                raise PermissionError(
                    f"Windows state directory ACL is not owner-only: {path}"
                )
        finally:
            api.close(observed)
    finally:
        api.close(handle)


def _validate_windows_owner_only_file_descriptor(descriptor: int, path: Path) -> None:
    try:
        from scripts.windows_acl import _FILE_SECURITY_ACCESS, _active_api
    except ModuleNotFoundError:  # Standalone imports from inside scripts/.
        from windows_acl import _FILE_SECURITY_ACCESS, _active_api

    if msvcrt is None:
        raise PermissionError("Windows file-handle API is unavailable")
    api = _active_api(None)
    try:
        borrowed = msvcrt.get_osfhandle(descriptor)
        identity = api.identity(borrowed)
    except Exception as error:
        raise PermissionError(
            f"could not inspect retained Windows state file: {path}"
        ) from error
    try:
        handle = api.open_file(path, access=_FILE_SECURITY_ACCESS)
    except Exception as error:
        raise PermissionError(f"could not open Windows state file: {path}") from error
    try:
        if api.is_reparse(handle) or api.identity(handle) != identity:
            raise PermissionError(f"Windows state file identity changed: {path}")
        try:
            state = api.inspect(handle)
        except Exception as error:
            raise PermissionError(
                f"could not inspect Windows state file ACL: {path}"
            ) from error
        if not state.is_owner_only:
            raise PermissionError(f"Windows state file ACL is not owner-only: {path}")
        try:
            observed = api.open_file(path, access=_FILE_SECURITY_ACCESS)
        except Exception as error:
            raise PermissionError(f"could not reopen Windows state file: {path}") from error
        try:
            if api.is_reparse(observed) or api.identity(observed) != identity:
                raise PermissionError(f"Windows state file identity changed: {path}")
            if not api.inspect(observed).is_owner_only:
                raise PermissionError(
                    f"Windows state file ACL is not owner-only: {path}"
                )
        finally:
            api.close(observed)
    finally:
        api.close(handle)


def _secure_windows_runtime_directory(path: Path, *, owner_only: bool) -> None:
    try:
        from scripts.windows_acl import secure_windows_directory
    except ModuleNotFoundError:  # Standalone imports from inside scripts/.
        from windows_acl import secure_windows_directory

    secure_windows_directory(path, owner_only=owner_only)


def _secure_windows_runtime_file(descriptor: int, path: Path) -> None:
    try:
        from scripts.windows_acl import secure_windows_file_descriptor
    except ModuleNotFoundError:  # Standalone imports from inside scripts/.
        from windows_acl import secure_windows_file_descriptor

    secure_windows_file_descriptor(descriptor, path)


def _ensure_fallback_runtime_component(path: Path, *, private: bool) -> None:
    """Create one directory while proving its parent and final identity."""
    parent = path.parent
    parent_before = parent.lstat()
    _validate_runtime_directory(parent_before, parent)
    parent_descriptor = _open_runtime_directory_nofollow(parent)
    created = False
    directory_descriptor: int | None = None
    try:
        try:
            before = path.lstat()
        except FileNotFoundError:
            os.mkdir(path, 0o700)
            created = True
            before = path.lstat()
        _validate_runtime_directory(before, path)
        directory_descriptor = _open_runtime_directory_nofollow(path)
        opened = (
            os.fstat(directory_descriptor)
            if directory_descriptor is not None
            else before
        )
        _validate_runtime_directory(opened, path)
        if not _same_file_identity(before, opened):
            raise ValueError(f"runtime directory identity changed: {path}")

        parent_after = parent.lstat()
        parent_opened = (
            os.fstat(parent_descriptor)
            if parent_descriptor is not None
            else parent_before
        )
        if not (
            _same_file_identity(parent_before, parent_after)
            and _same_file_identity(parent_before, parent_opened)
        ):
            if created:
                try:
                    visible = path.lstat()
                    current = (
                        os.fstat(directory_descriptor)
                        if directory_descriptor is not None
                        else before
                    )
                    if _same_file_identity(visible, current):
                        path.rmdir()
                except OSError:
                    pass
            raise ValueError(f"runtime parent identity changed: {parent}")
        _validate_runtime_directory(parent_after, parent)
        _validate_runtime_directory(parent_opened, parent)

        if _windows_acl_required():
            _secure_windows_runtime_directory(path, owner_only=private)

        if private or created:
            if directory_descriptor is not None and hasattr(os, "fchmod"):
                os.fchmod(directory_descriptor, 0o700)
            # A descriptorless platform cannot safely chmod this pathname: the
            # component could be replaced between validation and chmod.  Fresh
            # components were requested as 0700 and are checked again below.

        after = path.lstat()
        opened_after = (
            os.fstat(directory_descriptor)
            if directory_descriptor is not None
            else opened
        )
        _validate_runtime_directory(
            after, path, private=(private or created)
        )
        _validate_runtime_directory(
            opened_after, path, private=(private or created)
        )
        if not (
            _same_file_identity(before, after)
            and _same_file_identity(before, opened_after)
        ):
            raise ValueError(f"runtime directory identity changed: {path}")
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _bootstrap_secure_runtime_directory_fallback(memory_root: Path | str) -> Path:
    """Bootstrap paths for the identity-checking no-``dir_fd`` opener."""
    root = Path(os.path.abspath(Path(memory_root).expanduser()))
    if root.parent == root:
        raise ValueError("memory root must not be a filesystem root")
    _ensure_fallback_runtime_component(root, private=False)
    _ensure_fallback_runtime_component(root / "scripts", private=False)
    runtime = root / "scripts" / "runtime"
    _ensure_fallback_runtime_component(runtime, private=True)
    return runtime


@contextmanager
def _open_secure_runtime_file_posix(
    root: Path,
    runtime: Path,
    *,
    prefix: str,
    suffix: str,
) -> Iterator[tuple[Path, int]]:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_descriptor = os.open(root, directory_flags)
    scripts_descriptor = -1
    runtime_descriptor = -1
    file_descriptor = -1
    name = ""
    try:
        scripts_descriptor = os.open(
            "scripts", directory_flags, dir_fd=root_descriptor
        )
        runtime_descriptor = os.open(
            "runtime", directory_flags, dir_fd=scripts_descriptor
        )
        os.fchmod(runtime_descriptor, 0o700)
        scripts = root / "scripts"
        _validate_runtime_parent_identities(
            root,
            scripts,
            runtime,
            (root_descriptor, scripts_descriptor, runtime_descriptor),
        )
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        for _attempt in range(100):
            name = f"{prefix}{uuid.uuid4().hex}{suffix}"
            try:
                file_descriptor = os.open(
                    name,
                    flags,
                    0o600,
                    dir_fd=runtime_descriptor,
                )
                break
            except FileExistsError:
                continue
        else:  # pragma: no cover - collision probability is negligible.
            raise FileExistsError("could not allocate a unique runtime file")
        os.fchmod(file_descriptor, 0o600)
        path = runtime / name
        _validate_runtime_parent_identities(
            root,
            scripts,
            runtime,
            (root_descriptor, scripts_descriptor, runtime_descriptor),
        )
        validate_secure_runtime_file(path, file_descriptor)
        yield path, file_descriptor
    finally:
        if name and runtime_descriptor >= 0:
            try:
                os.unlink(name, dir_fd=runtime_descriptor)
            except FileNotFoundError:
                pass
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if runtime_descriptor >= 0:
            os.close(runtime_descriptor)
        if scripts_descriptor >= 0:
            os.close(scripts_descriptor)
        os.close(root_descriptor)


@contextmanager
def _open_secure_runtime_file_fallback(
    root: Path,
    runtime: Path,
    *,
    prefix: str,
    suffix: str,
) -> Iterator[tuple[Path, int]]:
    """Use identity checks when directory-relative no-follow APIs are absent."""
    scripts = root / "scripts"
    before = (root.lstat(), scripts.lstat(), runtime.lstat())
    for path, info, private in zip(
        (root, scripts, runtime), before, (False, False, True), strict=True
    ):
        _validate_runtime_directory(info, path, private=private)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=prefix,
        suffix=suffix,
        dir=runtime,
    )
    path = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        after = (root.lstat(), scripts.lstat(), runtime.lstat())
        for path_part, first, second, private in zip(
            (root, scripts, runtime), before, after, (False, False, True), strict=True
        ):
            _validate_runtime_directory(second, path_part, private=private)
            if not _same_file_identity(first, second):
                raise ValueError(f"runtime directory identity changed: {path_part}")
        validate_secure_runtime_file(path, descriptor)
        if _windows_acl_required():
            _secure_windows_runtime_file(descriptor, path)
            validate_secure_runtime_file(path, descriptor)
            final_parents = (root.lstat(), scripts.lstat(), runtime.lstat())
            for path_part, first, final, private in zip(
                (root, scripts, runtime),
                before,
                final_parents,
                (False, False, True),
                strict=True,
            ):
                _validate_runtime_directory(final, path_part, private=private)
                if not _same_file_identity(first, final):
                    raise ValueError(
                        f"runtime directory identity changed: {path_part}"
                    )
        yield path, descriptor
    finally:
        try:
            visible = path.lstat()
        except FileNotFoundError:
            pass
        else:
            if _same_file_identity(visible, os.fstat(descriptor)):
                path.unlink(missing_ok=True)
        os.close(descriptor)


@contextmanager
def open_secure_runtime_file(
    memory_root: Path | str,
    *,
    runtime_directory: Path | str | None = None,
    prefix: str = "ai-memory-live-",
    suffix: str = ".jsonl",
) -> Iterator[tuple[Path, int]]:
    """Create one private scratch file and retain its descriptor until cleanup."""
    root = Path(os.path.abspath(Path(memory_root).expanduser()))
    expected = root / "scripts" / "runtime"
    runtime = (
        (
            _bootstrap_secure_runtime_directory_posix(root)
            if _runtime_dir_fd_supported()
            else _bootstrap_secure_runtime_directory_fallback(root)
        )
        if runtime_directory is None
        else Path(os.path.abspath(Path(runtime_directory).expanduser()))
    )
    if runtime != expected:
        raise ValueError("runtime directory must remain inside the memory root")
    factory = (
        _open_secure_runtime_file_posix
        if _runtime_dir_fd_supported()
        else _open_secure_runtime_file_fallback
    )
    with factory(root, runtime, prefix=prefix, suffix=suffix) as opened:
        yield opened


def _open_secure_log_fallback(path: Path):
    """Open an existing log when directory-relative no-follow opens are unavailable."""
    root = path.parent.parent
    scripts = path.parent
    root_before = root.lstat()
    scripts_before = scripts.lstat()
    _validate_log_directory(root_before, root)
    _validate_log_directory(scripts_before, scripts)
    try:
        file_before = path.lstat()
    except FileNotFoundError:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        file_before = os.fstat(descriptor)
    else:
        _validate_log_file(file_before, path)
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        opened = os.fstat(descriptor)
        _validate_log_file(opened, path)
        if not _same_file_identity(file_before, opened):
            raise ValueError(f"log path identity changed while opening: {path}")
        root_after = root.lstat()
        scripts_after = scripts.lstat()
        _validate_log_directory(root_after, root)
        _validate_log_directory(scripts_after, scripts)
        if not _same_file_identity(root_before, root_after) or not _same_file_identity(
            scripts_before, scripts_after
        ):
            raise ValueError(f"log directory identity changed while opening: {scripts}")
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        elif stat.S_IMODE(opened.st_mode) & ~0o600:
            raise ValueError(f"log path has unsafe permissions: {path}")
        stream = os.fdopen(descriptor, "a", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise
    return stream


def open_secure_log_stream(path: Path | str):
    """Open an owner-controlled append log without path-following races."""
    target = Path(os.path.abspath(Path(path).expanduser()))
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    supports_dir_fd = os.open in getattr(os, "supports_dir_fd", set())
    if not (nofollow and directory and supports_dir_fd):
        return _open_secure_log_fallback(target)

    root = target.parent.parent
    root_descriptor = os.open(root, os.O_RDONLY | directory | nofollow)
    scripts_descriptor: int | None = None
    log_descriptor: int | None = None
    try:
        _validate_log_directory(os.fstat(root_descriptor), root)
        try:
            scripts_descriptor = os.open(
                target.parent.name,
                os.O_RDONLY | directory | nofollow,
                dir_fd=root_descriptor,
            )
        except OSError as exc:
            raise ValueError(
                f"log directory must be a non-symlink directory: {target.parent}"
            ) from exc
        _validate_log_directory(os.fstat(scripts_descriptor), target.parent)
        try:
            log_descriptor = os.open(
                target.name,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | nofollow,
                0o600,
                dir_fd=scripts_descriptor,
            )
        except OSError as exc:
            raise ValueError(
                f"log path must be a regular non-symlink file: {target}"
            ) from exc
        info = os.fstat(log_descriptor)
        _validate_log_file(info, target)
        if hasattr(os, "fchmod"):
            os.fchmod(log_descriptor, 0o600)
        elif stat.S_IMODE(info.st_mode) & ~0o600:
            raise ValueError(f"log path has unsafe permissions: {target}")
        stream = os.fdopen(log_descriptor, "a", encoding="utf-8")
        log_descriptor = None
        return stream
    finally:
        if log_descriptor is not None:
            os.close(log_descriptor)
        if scripts_descriptor is not None:
            os.close(scripts_descriptor)
        os.close(root_descriptor)


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


def update_state(
    mutate: Callable[[dict], object],
    *,
    max_attempts: int = 3,
) -> dict:
    """Apply a locked same-byte state mutation without losing concurrent fields."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    target = Path(os.path.abspath(STATE_FILE.expanduser()))
    if target.is_symlink():
        raise ValueError("state path must not be a symlink")
    root = target.parents[1]
    for _attempt in range(max_attempts):
        with ExclusiveFileLock(root / "scripts/memory-writer.lock"):
            data, baseline = _read_file_with_baseline(target)
            if data is None:
                state = {
                    "ingested": {}, "query_count": 0,
                    "last_lint": None, "total_cost": 0.0,
                }
            else:
                state = json.loads(data.decode("utf-8"))
                if not isinstance(state, dict):
                    raise ValueError("state.json must contain an object")
            mutate(state)
            if capture_file_baseline(target) != baseline:
                continue
            save_state_unlocked(state, target)
            return state
    raise RuntimeError("state update conflicted after bounded retries")


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
