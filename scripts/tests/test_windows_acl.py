"""Windows ACL policy tests that run without a Windows host."""

from pathlib import Path

import pytest


def test_owner_only_acl_is_established_then_verified(tmp_path):
    from scripts.windows_acl import AclState, secure_owner_only_directory

    states = iter(
        [
            AclState(
                owner_matches=True,
                protected=False,
                ace_count=2,
                inherited_aces=1,
                owner_full_control_only=False,
            ),
            AclState(
                owner_matches=True,
                protected=True,
                ace_count=1,
                inherited_aces=0,
                owner_full_control_only=True,
            ),
        ]
    )

    class FakeApi:
        def __init__(self):
            self.protected = []

        def inspect(self, path):
            assert path == tmp_path
            return next(states)

        def protect_owner_only(self, path):
            self.protected.append(path)

    api = FakeApi()

    secure_owner_only_directory(tmp_path, correct=True, api=api)

    assert api.protected == [tmp_path]


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(
            dict(
                owner_matches=False,
                protected=True,
                ace_count=1,
                inherited_aces=0,
                owner_full_control_only=True,
            ),
            id="wrong-owner",
        ),
        pytest.param(
            dict(
                owner_matches=True,
                protected=False,
                ace_count=1,
                inherited_aces=0,
                owner_full_control_only=True,
            ),
            id="unprotected",
        ),
        pytest.param(
            dict(
                owner_matches=True,
                protected=True,
                ace_count=1,
                inherited_aces=1,
                owner_full_control_only=True,
            ),
            id="inherited-ace",
        ),
        pytest.param(
            dict(
                owner_matches=True,
                protected=True,
                ace_count=2,
                inherited_aces=0,
                owner_full_control_only=False,
            ),
            id="extra-principal",
        ),
    ],
)
def test_owner_only_acl_rejects_unverifiable_or_broader_access(tmp_path, state):
    from scripts.windows_acl import AclState, secure_owner_only_directory

    class FakeApi:
        def inspect(self, _path):
            return AclState(**state)

        def protect_owner_only(self, _path):
            raise AssertionError("verification-only calls must not rewrite ancestry")

    with pytest.raises(PermissionError, match="owner-only ACL"):
        secure_owner_only_directory(tmp_path, correct=False, api=FakeApi())


def test_owner_only_acl_fails_closed_when_platform_api_is_unavailable(tmp_path):
    from scripts.windows_acl import secure_owner_only_directory

    class UnavailableApi:
        def inspect(self, _path):
            raise OSError("security API unavailable")

        def protect_owner_only(self, _path):
            raise AssertionError("must inspect ownership before correction")

    with pytest.raises(PermissionError, match="could not inspect"):
        secure_owner_only_directory(tmp_path, correct=True, api=UnavailableApi())


def test_windows_acl_module_has_no_non_windows_side_effects():
    import scripts.windows_acl as windows_acl

    assert callable(windows_acl.secure_owner_only_directory)
    assert Path(windows_acl.__file__).name == "windows_acl.py"
