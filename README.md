# LLM Personal Knowledge Base

**Your AI conversations compile themselves into a searchable knowledge base.**

Adapted from [Karpathy's LLM Knowledge Base](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) architecture, but instead of clipping web articles, the raw data is your own conversations with Claude Code. When a session ends (or auto-compacts mid-session), Claude Code hooks capture the conversation transcript and spawn a background process that uses the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) to extract the important stuff - decisions, lessons learned, patterns, gotchas - and appends it to a daily log. You then compile those daily logs into structured, cross-referenced knowledge articles organized by concept. Retrieval uses a simple index file instead of RAG - no vector database, no embeddings, just markdown.

Anthropic has clarified that personal use of the Claude Agent SDK is covered under your existing Claude subscription (Max, Team, or Enterprise) - no separate API credits needed. Unlike OpenClaw, which requires API billing for its memory flush, this runs on your subscription.

## Quick Start

Tell your AI coding agent:

> "Clone https://github.com/coleam00/claude-memory-compiler into this project. Set up the Claude Code hooks so my conversations automatically get captured into daily logs, compiled into a knowledge base, and injected back into future sessions. Read the AGENTS.md for the full technical reference on how everything works."

The agent will:
1. Clone the repo and run `uv sync` to install dependencies
2. Copy `.claude/settings.json` into your project (or merge the hooks into your existing settings)
3. The hooks activate automatically next time you open Claude Code

From there, your conversations start accumulating. After 4 PM local time, the next session flush automatically triggers compilation of that day's logs into knowledge articles — but only when no other Claude Code instances are still open, so it never competes with an active session. You can also run `uv run python scripts/compile.py` manually at any time.

## Global Setup (Capture Every Session, Anywhere)

The default Quick Start only captures sessions started inside this repo. To capture every Claude Code session no matter the working directory:

1. **Clone anywhere you like.** No fixed path required — the hooks resolve via the `CLAUDE_MEMORY_HOME` environment variable.
2. **Install dependencies:** `cd <your-clone-path> && uv sync`.
3. **Run the setup helper for personalized instructions:**
   ```bash
   bin/setup-global.sh
   ```
   It auto-detects your clone path and prints the exact two manual steps:
   - Add `export CLAUDE_MEMORY_HOME="<your-clone-path>"` to your shell's startup file (e.g. `~/.zshrc`).
   - Merge the `hooks` object from this repo's `.claude/settings.json` into your `~/.claude/settings.json` (create it if missing).
4. **Done.** Every Claude Code session, in any directory, fires the hooks. All output lands in `$CLAUDE_MEMORY_HOME/daily/` and `$CLAUDE_MEMORY_HOME/knowledge/` — one central knowledge base, not one per project.

Each session is automatically tagged with the project key (basename of the session's working directory) so retrieval can be scoped per-project later.

**Why an env var?** Team members clone to different locations. With `$CLAUDE_MEMORY_HOME`, the same `settings.json` works on everyone's machine — they just set the env var once for their clone.

**Caveat:** Claude Code hook commands inherit env vars from the shell that launched `claude`. If you run Claude Code from a fresh terminal where `CLAUDE_MEMORY_HOME` is exported, the hooks will see it. If you launch Claude Code from a macOS GUI launcher that doesn't read your shell profile, set the var via `launchctl setenv CLAUDE_MEMORY_HOME <path>` or in `~/.zshenv` instead of `~/.zshrc`.

**Double-fire prevention:** When hooks live in both project-local (`.claude/settings.json`) and user-global (`~/.claude/settings.json`) scopes, Claude Code fires both. The hooks include a 10-second per-session dedup guard (`scripts/last-hook-fire.json`) so only the first invocation does work. You don't need to choose between project-local and global — both can coexist.

## How It Works

```
Conversation -> SessionEnd/PreCompact hooks -> flush.py extracts knowledge
    -> daily/YYYY-MM-DD.md -> compile.py -> knowledge/concepts/, connections/, qa/
        -> SessionStart hook injects index into next session -> cycle repeats
```

- **Hooks** capture conversations automatically (session end + pre-compaction safety net)
- **flush.py** calls the Claude Agent SDK to decide what's worth saving, and after 4 PM triggers end-of-day compilation automatically — gated on no other Claude Code instances being open
- **compile.py** turns daily logs into organized concept articles with cross-references (triggered automatically or run manually)
- **query.py** answers questions using index-guided retrieval (no RAG needed at personal scale)
- **lint.py** runs 7 health checks (broken links, orphans, contradictions, staleness)

## Key Commands

```bash
uv run python scripts/compile.py                    # compile new daily logs
uv run python scripts/query.py "question"            # ask the knowledge base
uv run python scripts/query.py "question" --file-back # ask + save answer back
uv run python scripts/lint.py                        # run health checks
uv run python scripts/lint.py --structural-only      # free structural checks only
uv run python scripts/batch-flush.py --dry-run       # seed KB from past transcripts
```

## Seeding the Knowledge Base from Past Conversations

If you've already been using Claude Code on a project for a while, `batch-flush.py` extracts knowledge from your existing JSONL transcripts (under `~/.claude/projects/<project>/`) so the KB starts with real context instead of empty. It parses every transcript, chunks large sessions at user-message boundaries (not just the last 30 turns like the live hook), runs LLM extraction on each chunk, and writes everything into dated daily logs ready for `compile.py`.

```bash
uv run python scripts/batch-flush.py --dry-run            # preview — shows sessions, chunks, est. cost
uv run python scripts/batch-flush.py                       # run full extraction
uv run python scripts/batch-flush.py --max-cost 5.00       # stop after $5 spent
uv run python scripts/batch-flush.py --dates 2026-04-11    # only specific dates
uv run python scripts/batch-flush.py --resume              # skip sessions already processed
uv run python scripts/batch-flush.py --compile             # extract + trigger compile
uv run python scripts/batch-flush.py --all-projects --dry-run  # preview every project on this machine
```

**Single project (default):** auto-discovers the transcripts directory from `cwd`; override with `--transcripts-dir`. Daily-log entries are tagged with the project key (defaults to `Path.cwd().name`, matching the live hook); override with `--project-key` and `--project-cwd` when seeding from a directory other than the project itself.

**All projects (`--all-projects`):** walks `~/.claude/projects/*` and seeds every project in one pass. Each project's daily-log entries are tagged with that project's basename (decoded from Claude Code's `/`→`-` path encoding via filesystem-existence checks, so dashed names like `claude-memory-compiler` round-trip correctly). Honors `--max-cost` as a global budget across all projects and stops cleanly when the cap is hit.

State (`state.json`) is shared across modes — `--resume` skips sessions already processed by any prior invocation, so single-project runs and `--all-projects` runs are interchangeable.

## Why No RAG?

Karpathy's insight: at personal scale (50-500 articles), the LLM reading a structured `index.md` outperforms vector similarity. The LLM understands what you're really asking; cosine similarity just finds similar words. RAG becomes necessary at ~2,000+ articles when the index exceeds the context window.

## Technical Reference

See **[AGENTS.md](AGENTS.md)** for the complete technical reference: article formats, hook architecture, script internals, cross-platform details, costs, and customization options. AGENTS.md is designed to give an AI agent everything it needs to understand, modify, or rebuild the system.
