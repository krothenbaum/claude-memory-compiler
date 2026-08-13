"""Detached coordinator for one reserved end-of-day compile."""

from __future__ import annotations

from pathlib import Path
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from flush import run_auto_compile_coordinator  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 3:
        return 2
    root, token, fingerprint = arguments
    return 0 if run_auto_compile_coordinator(root, token, fingerprint) else 1


if __name__ == "__main__":
    raise SystemExit(main())
