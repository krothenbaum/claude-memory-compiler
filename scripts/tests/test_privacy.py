from __future__ import annotations

import pytest

from scripts import privacy


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("safe\x1b[31mred\x1b[0mtext", "saferedtext"),
        ("safe\x9b31mred\x9b0mtext", "saferedtext"),
        ("title\x1b]0;owned\x07after", "titleafter"),
        (
            "open\x1b]8;;https://unsafe.example\x07link\x1b]8;;\x1b\\close",
            "openlinkclose",
        ),
        ("left\rright\bback\x85c1", "left right back c1"),
    ],
)
def test_strip_terminal_controls_removes_csi_osc_and_remaining_controls(
    value, expected
):
    assert privacy.strip_terminal_controls(value) == expected


def test_persistence_normalization_sanitizes_before_redaction_and_bounding():
    secret = "credential-never-display"
    disguised = "credential\x1b[31m-never-display"

    result = privacy.normalize_persistence_reason(
        f"failed {disguised}\r" + ("x" * 2_000),
        {"OPENAI_API_KEY": secret},
    )

    assert secret not in result
    assert "\x1b" not in result
    assert "\r" not in result
    assert "[REDACTED]" in result
    assert len(result) == 1_000
