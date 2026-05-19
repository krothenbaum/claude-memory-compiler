"""
Compile daily conversation logs into structured knowledge articles.

This is the "LLM compiler" - it reads daily logs (source code) and produces
organized knowledge articles (the executable).

Usage:
    uv run python compile.py                    # compile new/changed logs only
    uv run python compile.py --all              # force recompile everything
    uv run python compile.py --file daily/2026-04-01.md  # compile a specific log
    uv run python compile.py --dry-run          # show what would be compiled
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import traceback
from pathlib import Path

from config import AGENTS_FILE, CONCEPTS_DIR, CONNECTIONS_DIR, DAILY_DIR, KNOWLEDGE_DIR, now_iso
from utils import (
    file_hash,
    list_raw_files,
    list_wiki_articles,
    load_state,
    notify_terminal,
    read_wiki_index,
    save_state,
)

# ── Paths for the LLM to use ──────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = Path(__file__).resolve().parent / "compile.log"

# ── Incremental-compile marker ────────────────────────────────────────
#
# Each successful compile appends `<!-- @compiled-through:ISO8601 -->` to the
# end of the daily log. The next compile slices everything after the LAST
# such marker and processes only that, so re-runs after off-hour sessions
# pick up exactly the new content — no missed sessions, no redundant work.

COMPILED_MARKER_RE = re.compile(r"<!--\s*@compiled-through:([^\s>]+)\s*-->")


def find_last_compiled_offset(content: str) -> int:
    """Return the byte offset just past the last @compiled-through marker.

    Returns 0 if no marker is present (caller treats whole file as new).
    """
    last_end = 0
    for m in COMPILED_MARKER_RE.finditer(content):
        last_end = m.end()
    return last_end


def append_compiled_marker(log_path: Path, when: str) -> None:
    """Append a `@compiled-through` marker line to the daily log."""
    marker = f"\n<!-- @compiled-through:{when} -->\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(marker)

logger = logging.getLogger("compile")
logger.setLevel(logging.DEBUG)
_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_file_handler)


async def compile_daily_log(log_path: Path, state: dict) -> float:
    """Compile a single daily log into knowledge articles.

    Returns the API cost of the compilation. Returns 0.0 on failure or partial
    completion; in that case state.json is NOT updated so the next run retries.
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    full_content = log_path.read_text(encoding="utf-8")
    offset = find_last_compiled_offset(full_content)
    new_content = full_content[offset:].strip()
    ingested = state.get("ingested", {})

    # Backfill: legacy log that was compiled before markers existed AND
    # hasn't changed since. Only seed when the recorded hash still matches
    # the file — if the file grew, late sessions are unprocessed and must
    # go through a real compile, not get silently marked as done.
    if offset == 0 and log_path.name in ingested:
        prior_hash = ingested[log_path.name].get("hash")
        current_hash = file_hash(log_path)
        if prior_hash == current_hash:
            logger.info("Backfilling marker for %s (unchanged since last compile)", log_path.name)
            notify_terminal(f"compile backfill — {log_path.name} (seeding marker, no new content)")
            append_compiled_marker(log_path, now_iso())
            ingested[log_path.name] = {
                "hash": file_hash(log_path),
                "compiled_at": now_iso(),
                "cost_usd": ingested[log_path.name].get("cost_usd", 0.0),
            }
            state["ingested"] = ingested
            save_state(state)
            return 0.0
        else:
            logger.info(
                "Log %s has grown since last compile (hash %s -> %s); compiling whole file",
                log_path.name, prior_hash, current_hash,
            )

    # Nothing new past the last marker — refresh marker + state.hash and skip.
    if not new_content:
        logger.info("No new content past marker in %s; skipping", log_path.name)
        notify_terminal(f"compile skipped — {log_path.name} (no new sessions)")
        append_compiled_marker(log_path, now_iso())
        ingested[log_path.name] = {
            "hash": file_hash(log_path),
            "compiled_at": now_iso(),
            "cost_usd": ingested.get(log_path.name, {}).get("cost_usd", 0.0),
        }
        state["ingested"] = ingested
        save_state(state)
        return 0.0

    schema = AGENTS_FILE.read_text(encoding="utf-8")
    wiki_index = read_wiki_index()

    timestamp = now_iso()
    is_incremental = offset > 0
    is_recompile = offset == 0 and log_path.name in ingested
    if is_incremental:
        mode_description = "incremental — earlier sessions already compiled; only the slice below is new"
    elif is_recompile:
        mode_description = (
            "recompile — the file has grown since the previous compile but the marker is missing. "
            "knowledge/log.md already has a prior entry for this file. Extract new knowledge and "
            "UPDATE existing articles where applicable; do not duplicate existing ones."
        )
    else:
        mode_description = "full — the daily log is being compiled for the first time"

    prompt = f"""You are a knowledge compiler. Your job is to read sessions from a
daily conversation log and extract knowledge into structured wiki articles.

## Schema (AGENTS.md)

{schema}

## Current Wiki Index

The full index of existing articles is below. Each row has the article slug and
a one-line summary. Use the `Read` tool to fetch any article whose content you
need before updating it. DO NOT read articles unrelated to the new sessions —
that wastes turns and tokens.

{wiki_index}

## New Sessions to Compile

**File:** {log_path.name}
**Mode:** {mode_description}

{new_content}

## Your Task

Compile the sessions above into wiki articles following the schema exactly.
Earlier sessions in this daily log (above the slice you see) have ALREADY been
compiled in prior runs; the wiki index already reflects them. Focus only on
extracting new knowledge from the sessions provided here.

### Workflow (efficient)

1. Skim the daily log and identify 1-7 concepts worth extracting.
2. For each concept, decide create-new vs update-existing by consulting the index above.
3. ONLY for concepts you're updating: `Read` that specific article before editing.
4. Write/Edit articles, then update `knowledge/index.md` and append to `knowledge/log.md`.
5. Steps 4 are MANDATORY even for small extractions — index.md and log.md MUST be updated
   before you stop. If only one concept is worth extracting, that is still a complete run
   when index.md and log.md reflect it.

### Project metadata (IMPORTANT)

Each session entry in the daily log begins with metadata lines like:

```
### Session [<project-key>] (HH:MM)

**Project:** <project-key>
**CWD:** /full/path/to/repo
```

This is the canonical scope for everything extracted from that session. When you
create or update articles, you MUST:

1. Set `project:` in the article frontmatter to the project-key from that session.
   - For concepts that legitimately span multiple projects, use a YAML list:
     `project: [main, ask-orchestrator]`
   - For project-agnostic / general knowledge, use: `project: global`
2. When updating an existing article with content from a new project, ADD that
   project to the existing list rather than overwriting.
3. In `knowledge/index.md`, every row must include the Project column.

### Rules:

1. **Extract key concepts** - Identify 1-7 distinct concepts worth their own article
2. **Create concept articles** in `knowledge/concepts/` - One .md file per concept
   - Use the exact article format from AGENTS.md (YAML frontmatter + sections)
   - Include `sources:` in frontmatter pointing to the daily log file
   - Include `project:` in frontmatter (see "Project metadata" above)
   - Use `[[concepts/slug]]` wikilinks to link to related concepts
   - Write in encyclopedia style - neutral, comprehensive
3. **Create connection articles** in `knowledge/connections/` if this log reveals non-obvious
   relationships between 2+ existing concepts
   - Connection articles also require `project:` in frontmatter (use a list if the
     connection spans projects, which is common for connections)
4. **Update existing articles** if this log adds new information to concepts already in the wiki
   - Read the existing article, add the new information, add the source to frontmatter
   - Merge `project:` values (add new project to the list if not already present)
5. **Update knowledge/index.md** - Add new entries to the table (REQUIRED before stopping)
   - Format: `| [[path/slug]] | <project> | One-line summary | source-file | {timestamp[:10]} |`
   - Columns: Article | Project | Summary | Compiled From | Updated
6. **Append to knowledge/log.md** - Add a timestamped entry (REQUIRED before stopping)
   ```
   ## [{timestamp}] compile | {log_path.name}
   - Source: daily/{log_path.name}
   - Projects touched: <comma-separated project keys seen in this log>
   - Articles created: [[concepts/x]], [[concepts/y]]
   - Articles updated: [[concepts/z]] (if any)
   ```

If the daily log has nothing worth extracting (e.g., only FLUSH_OK memory-flush
entries), STILL append a log.md entry noting that, so partial-completion detection
sees the file was processed. Example:
   ```
   ## [{timestamp}] compile | {log_path.name}
   - Source: daily/{log_path.name}
   - Projects touched: <projects seen>
   - Articles created: (none)
   - Articles updated: (none)
   - Note: No extractable knowledge (memory flushes only / etc.)
   ```

### File paths:
- Write concept articles to: {CONCEPTS_DIR}
- Write connection articles to: {CONNECTIONS_DIR}
- Update index at: {KNOWLEDGE_DIR / 'index.md'}
- Append log at: {KNOWLEDGE_DIR / 'log.md'}

### Quality standards:
- Every article must have complete YAML frontmatter
- Every article must link to at least 2 other articles via [[wikilinks]]
- Key Points section should have 3-5 bullet points
- Details section should have 2+ paragraphs
- Related Concepts section should have 2+ entries
- Sources section should cite the daily log with specific claims extracted
"""

    cost = 0.0
    log_md_path = KNOWLEDGE_DIR / "log.md"
    log_md_before = log_md_path.read_text(encoding="utf-8") if log_md_path.exists() else ""

    if is_incremental:
        mode_tag = "incremental"
    elif is_recompile:
        mode_tag = "recompile"
    else:
        mode_tag = "full"
    logger.info(
        "Begin %s compile of %s (%d new chars / %d total)",
        mode_tag, log_path.name, len(new_content), len(full_content),
    )
    notify_terminal(
        f"compile started — {log_path.name} ({mode_tag}, {len(new_content)} new chars)"
    )

    def _on_stderr(line: str) -> None:
        logger.debug("[cli stderr] %s", line.rstrip())

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=str(ROOT_DIR),
                system_prompt={"type": "preset", "preset": "claude_code"},
                allowed_tools=["Read", "Write", "Edit", "Glob", "Grep"],
                permission_mode="bypassPermissions",
                max_turns=60,
                stderr=_on_stderr,
                extra_args={"debug-to-stderr": None},
                setting_sources=[],
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        pass  # compilation output - LLM writes files directly
            elif isinstance(message, ResultMessage):
                cost = message.total_cost_usd or 0.0
                print(f"  Cost: ${cost:.4f}")
                logger.info("ResultMessage for %s: cost=$%.4f", log_path.name, cost)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  Error compiling {log_path.name}: {e}")
        print(f"  See {LOG_FILE} for full traceback and CLI stderr.")
        logger.error("Exception compiling %s: %s\n%s", log_path.name, e, tb)
        notify_terminal(f"compile failed — {log_path.name}: {e}")
        return 0.0

    # Partial-completion guard: agent must have added a log.md entry mentioning this file.
    # Count delta is robust to the agent inserting entries mid-file rather than appending,
    # which the prior "trailing slice" check missed and caused endless retry loops.
    log_md_after = log_md_path.read_text(encoding="utf-8") if log_md_path.exists() else ""
    grew = len(log_md_after) > len(log_md_before)
    mentions_before = log_md_before.count(log_path.name)
    mentions_after = log_md_after.count(log_path.name)
    gained_mention = mentions_after > mentions_before
    if not (grew and gained_mention):
        print(
            f"  Partial completion: knowledge/log.md was not updated with an entry for "
            f"{log_path.name}. State.json will NOT be advanced — re-run compile to retry."
        )
        logger.error(
            "Partial completion for %s: log.md not updated (grew=%s, mentions %d -> %d)",
            log_path.name, grew, mentions_before, mentions_after,
        )
        notify_terminal(f"compile partial — {log_path.name} (log.md not updated; will retry)")
        return 0.0

    # Drop a marker so the next compile run knows where to slice. Hash is
    # recorded AFTER the marker is appended so a no-op refresh next run
    # sees a stable hash.
    append_compiled_marker(log_path, now_iso())

    logger.info("Compile complete for %s", log_path.name)
    notify_terminal(f"compile complete — {log_path.name} (${cost:.4f})")

    # Update state
    rel_path = log_path.name
    state.setdefault("ingested", {})[rel_path] = {
        "hash": file_hash(log_path),
        "compiled_at": now_iso(),
        "cost_usd": cost,
    }
    state["total_cost"] = state.get("total_cost", 0.0) + cost
    save_state(state)

    return cost


def main():
    parser = argparse.ArgumentParser(description="Compile daily logs into knowledge articles")
    parser.add_argument("--all", action="store_true", help="Force recompile all logs")
    parser.add_argument("--file", type=str, help="Compile a specific daily log file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be compiled")
    args = parser.parse_args()

    state = load_state()

    # Determine which files to compile
    if args.file:
        target = Path(args.file)
        if not target.is_absolute():
            target = DAILY_DIR / target.name
        if not target.exists():
            # Try resolving relative to project root
            target = ROOT_DIR / args.file
        if not target.exists():
            print(f"Error: {args.file} not found")
            sys.exit(1)
        to_compile = [target]
    else:
        all_logs = list_raw_files()
        if args.all:
            to_compile = all_logs
        else:
            to_compile = []
            for log_path in all_logs:
                rel = log_path.name
                prev = state.get("ingested", {}).get(rel, {})
                if not prev or prev.get("hash") != file_hash(log_path):
                    to_compile.append(log_path)

    if not to_compile:
        print("Nothing to compile - all daily logs are up to date.")
        return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Files to compile ({len(to_compile)}):")
    for f in to_compile:
        print(f"  - {f.name}")

    if args.dry_run:
        return

    # Compile each file sequentially
    total_cost = 0.0
    for i, log_path in enumerate(to_compile, 1):
        print(f"\n[{i}/{len(to_compile)}] Compiling {log_path.name}...")
        cost = asyncio.run(compile_daily_log(log_path, state))
        total_cost += cost
        print(f"  Done.")

    articles = list_wiki_articles()
    print(f"\nCompilation complete. Total cost: ${total_cost:.2f}")
    print(f"Knowledge base: {len(articles)} articles")


if __name__ == "__main__":
    main()
