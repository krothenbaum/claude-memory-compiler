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

From there, your conversations start accumulating. After 6 PM local time, the next session flush automatically triggers compilation of that day's logs into knowledge articles. You can also run `uv run python scripts/compile.py` manually at any time.

## Global Setup (Capture Every Session, Anywhere)

The default Quick Start only captures sessions started inside this repo. To capture every Claude Code session no matter the working directory:

1. **Clone to the standard location:** `git clone <your-fork-url> $HOME/Development/claude-memory-compiler` (the bundled `.claude/settings.json` uses absolute paths under `$HOME/Development/claude-memory-compiler/`). If you clone elsewhere, edit those paths in your copy of `settings.json`.
2. **Install dependencies:** `cd $HOME/Development/claude-memory-compiler && uv sync`.
3. **Copy hooks into your user-global Claude Code settings:** merge the contents of this repo's `.claude/settings.json` into `~/.claude/settings.json` (create the file if it doesn't exist; merge the `hooks` keys if you already have other settings).
4. **Done.** Every session, in any directory, fires the hooks. All output still lands in this repo's `daily/` and `knowledge/` directories — there is one central knowledge base, not one per project.

Each session is automatically tagged with the project key (basename of the session's working directory) so retrieval can be scoped per-project later.

**Double-fire prevention:** When you have hooks in both project-local (`.claude/settings.json`) and user-global (`~/.claude/settings.json`) scopes, Claude Code fires both. The hooks include a 10-second per-session dedup guard (`scripts/last-hook-fire.json`) so only the first invocation does work. You don't need to choose between project-local and global — both can coexist.

## How It Works

```
Conversation -> SessionEnd/PreCompact hooks -> flush.py extracts knowledge
    -> daily/YYYY-MM-DD.md -> compile.py -> knowledge/concepts/, connections/, qa/
        -> SessionStart hook injects index into next session -> cycle repeats
```

- **Hooks** capture conversations automatically (session end + pre-compaction safety net)
- **flush.py** calls the Claude Agent SDK to decide what's worth saving, and after 6 PM triggers end-of-day compilation automatically
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
```

## Why No RAG?

Karpathy's insight: at personal scale (50-500 articles), the LLM reading a structured `index.md` outperforms vector similarity. The LLM understands what you're really asking; cosine similarity just finds similar words. RAG becomes necessary at ~2,000+ articles when the index exceeds the context window.

## Technical Reference

See **[AGENTS.md](AGENTS.md)** for the complete technical reference: article formats, hook architecture, script internals, cross-platform details, costs, and customization options. AGENTS.md is designed to give an AI agent everything it needs to understand, modify, or rebuild the system.
