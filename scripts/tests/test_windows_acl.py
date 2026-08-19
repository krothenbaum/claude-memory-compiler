"""Windows ACL policy tests that run without a Windows host."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _state(*, owner=True, protected=False, ace_count=3, inherited=2, only=False):
    from scripts.windows_acl import AclState

    return AclState(
        owner_matches=owner,
        protected=protected,
        ace_count=ace_count,
        inherited_aces=inherited,
        owner_full_control_only=only,
    )


class FakeHandleApi:
    def __init__(self, states, *, identities=None, reparse_handles=()):
        self.states = {key: list(value) for key, value in states.items()}
        self.identities = identities or {"target": (1, 10)}
        self.reparse_handles = set(reparse_handles)
        self.opened = []
        self.closed = []
        self.protected = []
        self.open_results = ["target", "target", "target"]
        self.file_open_results = ["security", "security"]
        self.file_accesses = []
        self.file_open_error = None
        self.protect_error = None

    def open_directory(self, path):
        self.opened.append(Path(path))
        return self.open_results.pop(0)

    def open_file(self, path, *, access):
        self.opened.append(Path(path))
        self.file_accesses.append(access)
        if self.file_open_error is not None:
            raise self.file_open_error
        if not access & 0x00040000:
            raise PermissionError("WRITE_DAC was not requested")
        return self.file_open_results.pop(0)

    def close(self, handle):
        self.closed.append(handle)

    def identity(self, handle):
        return self.identities[handle]

    def is_reparse(self, handle):
        return handle in self.reparse_handles

    def inspect(self, handle):
        values = self.states[handle]
        return values[0] if len(values) == 1 else values.pop(0)

    def protect_owner_only(self, handle, *, inherit):
        if self.protect_error is not None:
            raise self.protect_error
        self.protected.append((handle, inherit))


def test_private_acl_is_set_and_verified_through_pinned_handle(tmp_path):
    from scripts.windows_acl import secure_windows_directory

    api = FakeHandleApi(
        {
            "target": [
                _state(),
                _state(protected=True, ace_count=1, inherited=0, only=True),
            ]
        }
    )

    secure_windows_directory(tmp_path, owner_only=True, api=api)

    assert api.protected == [("target", True)]
    assert api.closed == ["target", "target", "target"]


def test_private_slice_acl_uses_second_write_dac_security_handle(tmp_path):
    from scripts.windows_acl import secure_windows_file_descriptor

    api = FakeHandleApi(
        {
            "security": [
                _state(),
                _state(protected=True, ace_count=1, inherited=0, only=True),
            ]
        },
        identities={"borrowed": (1, 10), "security": (1, 10)},
    )

    secure_windows_file_descriptor(
        7,
        tmp_path / "slice.jsonl",
        api=api,
        handle_from_descriptor=lambda descriptor: (
            "borrowed" if descriptor == 7 else "unexpected"
        ),
    )

    required_access = 0x00020000 | 0x00040000 | 0x00000080
    assert all(
        access & required_access == required_access
        for access in api.file_accesses
    )
    assert api.protected == [("security", False)]
    assert api.closed == ["security", "security"]


def test_queue_acl_helper_secures_through_existing_windows_api(
    tmp_path, monkeypatch
):
    from scripts import queue as queue_module
    from scripts import windows_acl

    path = tmp_path / "jobs.sqlite3"
    path.write_bytes(b"queue")
    path.chmod(0o600)
    api = FakeHandleApi(
        {
            "security": [
                _state(),
                _state(protected=True, ace_count=1, inherited=0, only=True),
            ]
        },
        identities={"borrowed": (1, 10), "security": (1, 10)},
    )
    monkeypatch.setattr(windows_acl, "_active_api", lambda _api: api)
    monkeypatch.setitem(
        sys.modules,
        "msvcrt",
        SimpleNamespace(get_osfhandle=lambda _descriptor: "borrowed"),
    )

    queue_module._secure_windows_queue_file(path)

    assert api.protected == [("security", False)]
    assert api.closed == ["security", "security"]


def test_private_slice_rejects_security_handle_identity_mismatch(tmp_path):
    from scripts.windows_acl import secure_windows_file_descriptor

    api = FakeHandleApi(
        {"security": [_state()]},
        identities={"borrowed": (1, 10), "security": (1, 99)},
    )
    api.file_open_results = ["security"]

    with pytest.raises(PermissionError, match="identity changed"):
        secure_windows_file_descriptor(
            7,
            tmp_path / "slice.jsonl",
            api=api,
            handle_from_descriptor=lambda _descriptor: "borrowed",
        )

    assert api.protected == []
    assert api.closed == ["security"]


def test_private_slice_rejects_reparse_security_handle(tmp_path):
    from scripts.windows_acl import secure_windows_file_descriptor

    api = FakeHandleApi(
        {"security": [_state()]},
        identities={"borrowed": (1, 10), "security": (1, 10)},
        reparse_handles={"security"},
    )
    api.file_open_results = ["security"]

    with pytest.raises(PermissionError, match="reparse"):
        secure_windows_file_descriptor(
            7,
            tmp_path / "slice.jsonl",
            api=api,
            handle_from_descriptor=lambda _descriptor: "borrowed",
        )

    assert api.protected == []
    assert api.closed == ["security"]


def test_private_slice_security_handle_open_failure_is_closed(tmp_path):
    from scripts.windows_acl import secure_windows_file_descriptor

    api = FakeHandleApi(
        {}, identities={"borrowed": (1, 10)}
    )
    api.file_open_error = OSError("CreateFileW denied")

    with pytest.raises(PermissionError, match="could not open Windows file"):
        secure_windows_file_descriptor(
            7,
            tmp_path / "slice.jsonl",
            api=api,
            handle_from_descriptor=lambda _descriptor: "borrowed",
        )

    assert api.protected == []
    assert api.closed == []


def test_private_slice_set_failure_closes_security_handle(tmp_path):
    from scripts.windows_acl import secure_windows_file_descriptor

    api = FakeHandleApi(
        {"security": [_state()]},
        identities={"borrowed": (1, 10), "security": (1, 10)},
    )
    api.file_open_results = ["security"]
    api.protect_error = OSError("SetSecurityInfo denied")

    with pytest.raises(PermissionError, match="could not establish owner-only ACL"):
        secure_windows_file_descriptor(
            7,
            tmp_path / "slice.jsonl",
            api=api,
            handle_from_descriptor=lambda _descriptor: "borrowed",
        )

    assert api.closed == ["security"]


def test_private_slice_path_swap_does_not_mutate_outside_handle(tmp_path):
    from scripts.windows_acl import secure_windows_file_descriptor

    api = FakeHandleApi(
        {
            "security": [
                _state(),
                _state(protected=True, ace_count=1, inherited=0, only=True),
            ],
            "outside": [_state(owner=False)],
        },
        identities={
            "borrowed": (1, 10),
            "security": (1, 10),
            "outside": (1, 99),
        },
    )
    api.file_open_results = ["security", "outside"]

    with pytest.raises(PermissionError, match="identity changed"):
        secure_windows_file_descriptor(
            7,
            tmp_path / "slice.jsonl",
            api=api,
            handle_from_descriptor=lambda _descriptor: "borrowed",
        )

    assert api.protected == [("security", False)]
    assert all(handle != "outside" for handle, _inherit in api.protected)
    assert api.closed == ["outside", "security"]


def test_existing_ancestry_accepts_stable_inherited_system_admin_acl(tmp_path):
    from scripts.windows_acl import secure_windows_directory

    api = FakeHandleApi({"target": [_state()]})

    secure_windows_directory(tmp_path, owner_only=False, api=api)

    assert api.protected == []


def test_private_acl_path_swap_never_mutates_replacement_target(tmp_path):
    from scripts.windows_acl import secure_windows_directory

    api = FakeHandleApi(
        {
            "target": [
                _state(),
                _state(protected=True, ace_count=1, inherited=0, only=True),
            ],
            "outside": [_state(owner=False)],
        },
        identities={"target": (1, 10), "outside": (1, 99)},
    )
    api.open_results = ["target", "target", "outside"]

    with pytest.raises(PermissionError, match="identity changed"):
        secure_windows_directory(tmp_path, owner_only=True, api=api)

    assert api.protected == [("target", True)]
    assert all(handle != "outside" for handle, _inherit in api.protected)
    assert api.closed == ["target", "outside", "target"]


def test_private_acl_rejects_swap_before_correction_without_mutation(tmp_path):
    from scripts.windows_acl import secure_windows_directory

    api = FakeHandleApi(
        {"target": [_state()], "outside": [_state(owner=False)]},
        identities={"target": (1, 10), "outside": (1, 99)},
    )
    api.open_results = ["target", "outside"]

    with pytest.raises(PermissionError, match="identity changed"):
        secure_windows_directory(tmp_path, owner_only=True, api=api)

    assert api.protected == []
    assert api.closed == ["outside", "target"]


def test_windows_directory_reparse_is_rejected_before_acl_mutation(tmp_path):
    from scripts.windows_acl import secure_windows_directory

    api = FakeHandleApi({"target": [_state()]}, reparse_handles={"target"})
    api.open_results = ["target"]

    with pytest.raises(PermissionError, match="reparse"):
        secure_windows_directory(tmp_path, owner_only=True, api=api)

    assert api.protected == []
    assert api.closed == ["target"]


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(_state(owner=False), id="wrong-owner"),
        pytest.param(
            _state(protected=False, ace_count=1, inherited=0, only=True),
            id="unprotected",
        ),
        pytest.param(
            _state(protected=True, ace_count=1, inherited=1, only=True),
            id="inherited-ace",
        ),
        pytest.param(
            _state(protected=True, ace_count=2, inherited=0, only=False),
            id="extra-principal",
        ),
    ],
)
def test_private_acl_rejects_unverifiable_or_broader_final_access(tmp_path, state):
    from scripts.windows_acl import secure_windows_directory

    api = FakeHandleApi({"target": [_state(), state]})

    with pytest.raises(PermissionError, match="owner-only ACL"):
        secure_windows_directory(tmp_path, owner_only=True, api=api)


def test_windows_acl_fails_closed_when_platform_api_is_unavailable(tmp_path):
    from scripts.windows_acl import secure_windows_directory

    class UnavailableApi:
        def open_directory(self, _path):
            raise OSError("security API unavailable")

    with pytest.raises(PermissionError, match="could not open"):
        secure_windows_directory(tmp_path, owner_only=True, api=UnavailableApi())


def test_windows_acl_module_has_no_non_windows_side_effects():
    import scripts.windows_acl as windows_acl

    assert callable(windows_acl.secure_windows_directory)
    assert Path(windows_acl.__file__).name == "windows_acl.py"
