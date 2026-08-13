"""Fail-closed owner-only Windows directory ACL support."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Protocol


_ACL_REVISION = 2
_ACCESS_ALLOWED_ACE_TYPE = 0
_OBJECT_INHERIT_ACE = 0x01
_CONTAINER_INHERIT_ACE = 0x02
_INHERITED_ACE = 0x10
_FILE_ALL_ACCESS = 0x001F01FF
_SE_FILE_OBJECT = 1
_OWNER_SECURITY_INFORMATION = 0x00000001
_DACL_SECURITY_INFORMATION = 0x00000004
_PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
_SE_DACL_PROTECTED = 0x1000
_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1
_ACL_SIZE_INFORMATION_CLASS = 2
_READ_CONTROL = 0x00020000
_WRITE_DAC = 0x00040000
_FILE_READ_ATTRIBUTES = 0x00000080
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
_OPEN_EXISTING = 3
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_SECURITY_ACCESS = _READ_CONTROL | _WRITE_DAC | _FILE_READ_ATTRIBUTES


@dataclass(frozen=True)
class AclState:
    """Security facts required for an owner-only directory boundary."""

    owner_matches: bool
    protected: bool
    ace_count: int
    inherited_aces: int
    owner_full_control_only: bool

    @property
    def is_owner_only(self) -> bool:
        return (
            self.owner_matches
            and self.protected
            and self.ace_count == 1
            and self.inherited_aces == 0
            and self.owner_full_control_only
        )


class _AclApi(Protocol):
    def open_directory(self, path: Path) -> object: ...

    def open_file(self, path: Path, *, access: int) -> object: ...

    def close(self, handle: object) -> None: ...

    def identity(self, handle: object) -> tuple[int, int]: ...

    def is_reparse(self, handle: object) -> bool: ...

    def inspect(self, handle: object) -> AclState: ...

    def protect_owner_only(self, handle: object, *, inherit: bool) -> None: ...


class _SidAndAttributes(ctypes.Structure):
    _fields_ = [("sid", ctypes.c_void_p), ("attributes", wintypes.DWORD)]


class _TokenUser(ctypes.Structure):
    _fields_ = [("user", _SidAndAttributes)]


class _AclSizeInformation(ctypes.Structure):
    _fields_ = [
        ("ace_count", wintypes.DWORD),
        ("acl_bytes_in_use", wintypes.DWORD),
        ("acl_bytes_free", wintypes.DWORD),
    ]


class _AceHeader(ctypes.Structure):
    _fields_ = [
        ("ace_type", ctypes.c_ubyte),
        ("ace_flags", ctypes.c_ubyte),
        ("ace_size", wintypes.WORD),
    ]


class _ByHandleFileInformation(ctypes.Structure):
    _fields_ = [
        ("file_attributes", wintypes.DWORD),
        ("creation_time", wintypes.FILETIME),
        ("last_access_time", wintypes.FILETIME),
        ("last_write_time", wintypes.FILETIME),
        ("volume_serial_number", wintypes.DWORD),
        ("file_size_high", wintypes.DWORD),
        ("file_size_low", wintypes.DWORD),
        ("number_of_links", wintypes.DWORD),
        ("file_index_high", wintypes.DWORD),
        ("file_index_low", wintypes.DWORD),
    ]


class _WindowsSecurityApi:
    """Small ctypes boundary over the Win32 security APIs."""

    def __init__(self) -> None:
        if os.name != "nt" or not hasattr(ctypes, "WinDLL"):
            raise OSError("Windows ACL API is unavailable")
        self.advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self.kernel.CreateFileW.restype = wintypes.HANDLE
        self.kernel.GetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ByHandleFileInformation),
        ]
        self.kernel.GetFileInformationByHandle.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel.LocalFree.argtypes = [wintypes.HLOCAL]
        self.kernel.LocalFree.restype = wintypes.HLOCAL

        self.advapi.OpenProcessToken.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        self.advapi.OpenProcessToken.restype = wintypes.BOOL
        self.advapi.GetTokenInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.advapi.GetTokenInformation.restype = wintypes.BOOL
        self.advapi.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.advapi.EqualSid.restype = wintypes.BOOL
        self.advapi.GetLengthSid.argtypes = [ctypes.c_void_p]
        self.advapi.GetLengthSid.restype = wintypes.DWORD
        self.advapi.InitializeAcl.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        self.advapi.InitializeAcl.restype = wintypes.BOOL
        self.advapi.AddAccessAllowedAceEx.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        self.advapi.AddAccessAllowedAceEx.restype = wintypes.BOOL
        pointer = ctypes.POINTER(ctypes.c_void_p)
        self.advapi.GetSecurityInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.DWORD,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
        ]
        self.advapi.GetSecurityInfo.restype = wintypes.DWORD
        self.advapi.SetSecurityInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.advapi.SetSecurityInfo.restype = wintypes.DWORD
        self.advapi.GetSecurityDescriptorControl.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.WORD),
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.advapi.GetSecurityDescriptorControl.restype = wintypes.BOOL
        self.advapi.GetAclInformation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_int,
        ]
        self.advapi.GetAclInformation.restype = wintypes.BOOL
        self.advapi.GetAce.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
            pointer,
        ]
        self.advapi.GetAce.restype = wintypes.BOOL

    @staticmethod
    def _error(label: str, code: int | None = None) -> OSError:
        number = ctypes.get_last_error() if code is None else code
        return OSError(number, f"{label} failed")

    def _open_path(self, path: Path, *, access: int, flags: int) -> wintypes.HANDLE:
        handle = self.kernel.CreateFileW(
            str(path),
            access,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
            None,
            _OPEN_EXISTING,
            flags,
            None,
        )
        if handle == wintypes.HANDLE(-1).value:
            raise self._error("CreateFileW")
        return handle

    def open_directory(self, path: Path) -> wintypes.HANDLE:
        return self._open_path(
            path,
            access=_FILE_SECURITY_ACCESS,
            flags=_FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        )

    def open_file(self, path: Path, *, access: int) -> wintypes.HANDLE:
        return self._open_path(
            path,
            access=access,
            flags=_FILE_FLAG_OPEN_REPARSE_POINT,
        )

    def close(self, handle: object) -> None:
        if not self.kernel.CloseHandle(handle):
            raise self._error("CloseHandle")

    def _file_information(self, handle: object) -> _ByHandleFileInformation:
        information = _ByHandleFileInformation()
        if not self.kernel.GetFileInformationByHandle(
            handle, ctypes.byref(information)
        ):
            raise self._error("GetFileInformationByHandle")
        return information

    def identity(self, handle: object) -> tuple[int, int]:
        information = self._file_information(handle)
        index = (information.file_index_high << 32) | information.file_index_low
        return information.volume_serial_number, index

    def is_reparse(self, handle: object) -> bool:
        information = self._file_information(handle)
        return bool(information.file_attributes & _FILE_ATTRIBUTE_REPARSE_POINT)

    def _current_user_sid(self) -> tuple[ctypes.Array[ctypes.c_char], ctypes.c_void_p]:
        token = wintypes.HANDLE()
        if not self.advapi.OpenProcessToken(
            self.kernel.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)
        ):
            raise self._error("OpenProcessToken")
        try:
            required = wintypes.DWORD()
            self.advapi.GetTokenInformation(
                token, _TOKEN_USER, None, 0, ctypes.byref(required)
            )
            if required.value == 0:
                raise self._error("GetTokenInformation")
            buffer = ctypes.create_string_buffer(required.value)
            if not self.advapi.GetTokenInformation(
                token,
                _TOKEN_USER,
                buffer,
                required,
                ctypes.byref(required),
            ):
                raise self._error("GetTokenInformation")
            sid = ctypes.cast(buffer, ctypes.POINTER(_TokenUser)).contents.user.sid
            if not sid:
                raise OSError("current Windows token has no user SID")
            return buffer, ctypes.c_void_p(sid)
        finally:
            self.kernel.CloseHandle(token)

    def inspect(self, handle: object) -> AclState:
        owner = ctypes.c_void_p()
        dacl = ctypes.c_void_p()
        descriptor = ctypes.c_void_p()
        code = self.advapi.GetSecurityInfo(
            handle,
            _SE_FILE_OBJECT,
            _OWNER_SECURITY_INFORMATION | _DACL_SECURITY_INFORMATION,
            ctypes.byref(owner),
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(descriptor),
        )
        if code:
            raise self._error("GetSecurityInfo", code)
        try:
            sid_buffer, current_sid = self._current_user_sid()
            owner_matches = bool(
                owner.value
                and self.advapi.EqualSid(owner, current_sid)
            )
            control = wintypes.WORD()
            revision = wintypes.DWORD()
            if not self.advapi.GetSecurityDescriptorControl(
                descriptor, ctypes.byref(control), ctypes.byref(revision)
            ):
                raise self._error("GetSecurityDescriptorControl")
            if not dacl.value:
                return AclState(
                    owner_matches,
                    bool(control.value & _SE_DACL_PROTECTED),
                    0,
                    0,
                    False,
                )
            size = _AclSizeInformation()
            if not self.advapi.GetAclInformation(
                dacl,
                ctypes.byref(size),
                ctypes.sizeof(size),
                _ACL_SIZE_INFORMATION_CLASS,
            ):
                raise self._error("GetAclInformation")
            inherited = 0
            owner_only = size.ace_count == 1
            for index in range(size.ace_count):
                ace = ctypes.c_void_p()
                if not self.advapi.GetAce(dacl, index, ctypes.byref(ace)):
                    raise self._error("GetAce")
                header = ctypes.cast(ace, ctypes.POINTER(_AceHeader)).contents
                inherited += int(bool(header.ace_flags & _INHERITED_ACE))
                mask = ctypes.cast(
                    ace.value + ctypes.sizeof(_AceHeader),
                    ctypes.POINTER(wintypes.DWORD),
                ).contents.value
                ace_sid = ctypes.c_void_p(
                    ace.value + ctypes.sizeof(_AceHeader) + ctypes.sizeof(wintypes.DWORD)
                )
                owner_only = bool(
                    owner_only
                    and header.ace_type == _ACCESS_ALLOWED_ACE_TYPE
                    and mask & _FILE_ALL_ACCESS == _FILE_ALL_ACCESS
                    and self.advapi.EqualSid(ace_sid, current_sid)
                )
            return AclState(
                owner_matches=owner_matches,
                protected=bool(control.value & _SE_DACL_PROTECTED),
                ace_count=size.ace_count,
                inherited_aces=inherited,
                owner_full_control_only=owner_only,
            )
        finally:
            if descriptor.value:
                self.kernel.LocalFree(descriptor)

    def protect_owner_only(self, handle: object, *, inherit: bool) -> None:
        sid_buffer, sid = self._current_user_sid()
        sid_length = self.advapi.GetLengthSid(sid)
        if not sid_length:
            raise self._error("GetLengthSid")
        acl_size = 8 + 8 + sid_length
        acl = ctypes.create_string_buffer(acl_size)
        if not self.advapi.InitializeAcl(acl, acl_size, _ACL_REVISION):
            raise self._error("InitializeAcl")
        if not self.advapi.AddAccessAllowedAceEx(
            acl,
            _ACL_REVISION,
            (_OBJECT_INHERIT_ACE | _CONTAINER_INHERIT_ACE) if inherit else 0,
            _FILE_ALL_ACCESS,
            sid,
        ):
            raise self._error("AddAccessAllowedAceEx")
        code = self.advapi.SetSecurityInfo(
            handle,
            _SE_FILE_OBJECT,
            _DACL_SECURITY_INFORMATION | _PROTECTED_DACL_SECURITY_INFORMATION,
            None,
            None,
            acl,
            None,
        )
        if code:
            raise self._error("SetSecurityInfo", code)


def _active_api(api: _AclApi | None) -> _AclApi:
    try:
        return _WindowsSecurityApi() if api is None else api
    except (AttributeError, OSError) as exc:
        raise PermissionError("Windows ACL API is unavailable") from exc


def _verify_owner_only_handle(
    api: _AclApi,
    handle: object,
    target: Path,
    *,
    inherit: bool,
    initial: AclState | None = None,
) -> None:
    if initial is None:
        try:
            initial = api.inspect(handle)
        except Exception as exc:
            raise PermissionError(f"could not inspect owner ACL: {target}") from exc
    if not initial.owner_matches:
        raise PermissionError(f"directory has an unsafe Windows owner: {target}")
    try:
        api.protect_owner_only(handle, inherit=inherit)
        final = api.inspect(handle)
    except Exception as exc:
        raise PermissionError(f"could not establish owner-only ACL: {target}") from exc
    if not final.is_owner_only:
        raise PermissionError(f"directory does not have an owner-only ACL: {target}")


def _open_matching_directory(
    api: _AclApi,
    target: Path,
    expected_identity: tuple[int, int],
) -> None:
    try:
        observed = api.open_directory(target)
    except Exception as exc:
        raise PermissionError(f"could not reopen Windows directory: {target}") from exc
    try:
        if api.is_reparse(observed):
            raise PermissionError(f"Windows directory is a reparse point: {target}")
        if api.identity(observed) != expected_identity:
            raise PermissionError(f"Windows directory identity changed: {target}")
    finally:
        api.close(observed)


def secure_windows_directory(
    path: Path | str,
    *,
    owner_only: bool,
    api: _AclApi | None = None,
) -> None:
    """Verify one pinned directory and optionally install its private DACL."""
    target = Path(path)
    active_api = _active_api(api)
    try:
        handle = active_api.open_directory(target)
    except Exception as exc:
        raise PermissionError(f"could not open Windows directory: {target}") from exc
    try:
        if active_api.is_reparse(handle):
            raise PermissionError(f"Windows directory is a reparse point: {target}")
        identity = active_api.identity(handle)
        _open_matching_directory(active_api, target, identity)
        try:
            initial = active_api.inspect(handle)
        except Exception as exc:
            raise PermissionError(f"could not inspect owner ACL: {target}") from exc
        if not initial.owner_matches:
            raise PermissionError(f"directory has an unsafe Windows owner: {target}")
        if owner_only:
            _verify_owner_only_handle(
                active_api,
                handle,
                target,
                inherit=True,
                initial=initial,
            )
        _open_matching_directory(active_api, target, identity)
    finally:
        active_api.close(handle)


def secure_windows_file_descriptor(
    descriptor: int,
    path: Path | str,
    *,
    api: _AclApi | None = None,
    handle_from_descriptor: Callable[[int], object] | None = None,
) -> None:
    """Install a private DACL through a second identity-matched security handle."""
    active_api = _active_api(api)
    if handle_from_descriptor is None:
        try:
            import msvcrt

            borrowed_handle = msvcrt.get_osfhandle(descriptor)
        except (ImportError, OSError) as exc:
            raise PermissionError("Windows file-handle API is unavailable") from exc
    else:
        borrowed_handle = handle_from_descriptor(descriptor)
    target = Path(path)
    try:
        borrowed_identity = active_api.identity(borrowed_handle)
    except Exception as exc:
        raise PermissionError(
            f"could not inspect retained Windows file handle: {target}"
        ) from exc
    try:
        security_handle = active_api.open_file(
            target,
            access=_FILE_SECURITY_ACCESS,
        )
    except Exception as exc:
        raise PermissionError(
            f"could not open Windows file security handle: {target}"
        ) from exc
    try:
        if active_api.is_reparse(security_handle):
            raise PermissionError(f"Windows file is a reparse point: {target}")
        if active_api.identity(security_handle) != borrowed_identity:
            raise PermissionError(f"Windows file identity changed: {target}")
        _verify_owner_only_handle(
            active_api,
            security_handle,
            target,
            inherit=False,
        )
        try:
            observed = active_api.open_file(
                target,
                access=_FILE_SECURITY_ACCESS,
            )
        except Exception as exc:
            raise PermissionError(
                f"could not reopen Windows file security handle: {target}"
            ) from exc
        try:
            if active_api.is_reparse(observed):
                raise PermissionError(f"Windows file is a reparse point: {target}")
            if active_api.identity(observed) != borrowed_identity:
                raise PermissionError(f"Windows file identity changed: {target}")
        finally:
            active_api.close(observed)
    finally:
        active_api.close(security_handle)
