"""One-shot: reconcile state.json with what's actually in knowledge/log.md.

The partial-completion guard in compile.py previously checked only a trailing
slice of log.md and missed entries the agent inserted mid-file. Successful
compiles were rejected as "partial," state.json never advanced, and the same
files got retried on every subsequent compile run.

This script scans knowledge/log.md for `## [timestamp] compile | NAME.md`
entries, and for each NAME.md that:
  - exists in daily/
  - has at least one entry in log.md
  - is NOT in state.json["ingested"] (or has stale hash)
records it in state.json and seeds an `@compiled-through` marker at EOF so
the next compile run treats it as fully processed.

Usage:
    uv run python scripts/reconcile-state.py            # apply changes
    uv run python scripts/reconcile-state.py --dry-run  # show what would change
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from compile import COMPILED_MARKER_RE, commit_compiled_bookkeeping  # noqa: E402
from config import DAILY_DIR, KNOWLEDGE_DIR, now_iso  # noqa: E402
from staging import ApplyBookkeeping, apply_host_bookkeeping, recover_incomplete_apply  # noqa: E402
from utils import file_hash, load_state  # noqa: E402

LOG_MD_PATH = KNOWLEDGE_DIR / "log.md"
COMPILE_ENTRY_RE = re.compile(r"^##\s+\[[^\]]+\]\s+compile\s+\|\s+(\S+\.md)", re.MULTILINE)


def files_mentioned_in_log_md() -> set[str]:
    if not LOG_MD_PATH.exists():
        return set()
    content = LOG_MD_PATH.read_text(encoding="utf-8")
    return set(COMPILE_ENTRY_RE.findall(content))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    state = load_state()
    ingested = state.setdefault("ingested", {})

    mentioned = files_mentioned_in_log_md()
    print(f"Found {len(mentioned)} file(s) mentioned in knowledge/log.md compile entries.")

    to_reconcile: list[Path] = []
    for name in sorted(mentioned):
        log_path = DAILY_DIR / name
        if not log_path.exists():
            print(f"  SKIP {name}: file missing from daily/")
            continue
        if name in ingested:
            current_hash = file_hash(log_path)
            recorded_hash = ingested[name].get("hash")
            content = log_path.read_text(encoding="utf-8")
            has_marker = bool(COMPILED_MARKER_RE.search(content))
            if recorded_hash == current_hash and has_marker:
                continue  # already healthy
            print(f"  STALE {name}: in state but {'no marker' if not has_marker else 'hash mismatch'}")
            to_reconcile.append(log_path)
        else:
            print(f"  MISSING {name}: in log.md but not in state.json")
            to_reconcile.append(log_path)

    if not to_reconcile:
        print("Nothing to reconcile — state.json is in sync.")
        return

    print(f"\nWould reconcile {len(to_reconcile)} file(s):")
    for p in to_reconcile:
        print(f"  - {p.name}")

    if args.dry_run:
        print("\n(dry-run; no changes written)")
        return

    print("\nApplying...")
    recover_incomplete_apply(DAILY_DIR.parent)
    now = now_iso()
    for log_path in to_reconcile:
        content = log_path.read_text(encoding="utf-8")
        needs_marker = not COMPILED_MARKER_RE.search(content)
        prior = ingested.get(log_path.name, {})
        ingested[log_path.name] = {
            "hash": "pending-transaction" if needs_marker else file_hash(log_path),
            "compiled_at": prior.get("compiled_at", now),
            "cost_usd": prior.get("cost_usd", 0.0),
            "reconciled_at": now,
        }
        if needs_marker:
            commit_compiled_bookkeeping(log_path, state, now)
        else:
            apply_host_bookkeeping(
                DAILY_DIR.parent,
                ApplyBookkeeping(state=state),
            )
        ingested = state.setdefault("ingested", {})
    print(f"Done. state.json now tracks {len(ingested)} ingested file(s).")


if __name__ == "__main__":
    main()
