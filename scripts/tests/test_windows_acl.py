"""Windows ACL policy tests that run without a Windows host."""

from pathlib import Path

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

    def open_directory(self, path):
        self.opened.append(Path(path))
        return self.open_results.pop(0)

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


def test_private_slice_acl_uses_borrowed_file_descriptor_handle(tmp_path):
    from scripts.windows_acl import secure_windows_file_descriptor

    api = FakeHandleApi(
        {
            "target": [
                _state(),
                _state(protected=True, ace_count=1, inherited=0, only=True),
            ]
        }
    )

    secure_windows_file_descriptor(
        7,
        tmp_path / "slice.jsonl",
        api=api,
        handle_from_descriptor=lambda descriptor: (
            "target" if descriptor == 7 else "unexpected"
        ),
    )

    assert api.protected == [("target", False)]
    assert api.closed == []


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
