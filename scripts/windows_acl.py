"""Fail-closed owner-only Windows directory ACL support."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol


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
    def inspect(self, path: Path) -> AclState: ...

    def protect_owner_only(self, path: Path) -> None: ...


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
        self.advapi.GetNamedSecurityInfoW.argtypes = [
            wintypes.LPWSTR,
            ctypes.c_int,
            wintypes.DWORD,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
        ]
        self.advapi.GetNamedSecurityInfoW.restype = wintypes.DWORD
        self.advapi.SetNamedSecurityInfoW.argtypes = [
            wintypes.LPWSTR,
            ctypes.c_int,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.advapi.SetNamedSecurityInfoW.restype = wintypes.DWORD
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

    def inspect(self, path: Path) -> AclState:
        owner = ctypes.c_void_p()
        dacl = ctypes.c_void_p()
        descriptor = ctypes.c_void_p()
        code = self.advapi.GetNamedSecurityInfoW(
            str(path),
            _SE_FILE_OBJECT,
            _OWNER_SECURITY_INFORMATION | _DACL_SECURITY_INFORMATION,
            ctypes.byref(owner),
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(descriptor),
        )
        if code:
            raise self._error("GetNamedSecurityInfoW", code)
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

    def protect_owner_only(self, path: Path) -> None:
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
            _OBJECT_INHERIT_ACE | _CONTAINER_INHERIT_ACE,
            _FILE_ALL_ACCESS,
            sid,
        ):
            raise self._error("AddAccessAllowedAceEx")
        code = self.advapi.SetNamedSecurityInfoW(
            str(path),
            _SE_FILE_OBJECT,
            _DACL_SECURITY_INFORMATION | _PROTECTED_DACL_SECURITY_INFORMATION,
            None,
            None,
            acl,
            None,
        )
        if code:
            raise self._error("SetNamedSecurityInfoW", code)


def secure_owner_only_directory(
    path: Path | str,
    *,
    correct: bool,
    api: _AclApi | None = None,
) -> None:
    """Establish or verify one protected, current-owner-only directory DACL."""
    target = Path(path)
    try:
        active_api = _WindowsSecurityApi() if api is None else api
    except (AttributeError, OSError) as exc:
        raise PermissionError("Windows ACL API is unavailable") from exc
    try:
        initial = active_api.inspect(target)
    except Exception as exc:
        raise PermissionError(f"could not inspect owner-only ACL: {target}") from exc
    if not initial.owner_matches:
        raise PermissionError(f"directory does not have an owner-only ACL: {target}")
    if correct:
        try:
            active_api.protect_owner_only(target)
        except Exception as exc:
            raise PermissionError(f"could not establish owner-only ACL: {target}") from exc
        try:
            final = active_api.inspect(target)
        except Exception as exc:
            raise PermissionError(f"could not inspect owner-only ACL: {target}") from exc
    else:
        final = initial
    if not final.is_owner_only:
        raise PermissionError(f"directory does not have an owner-only ACL: {target}")
