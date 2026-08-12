# LLM Personal Knowledge Base

Claude Code and Codex conversations compile into one searchable, local knowledge base.

The project adapts [Karpathy's LLM Knowledge Base](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) architecture to personal AI sessions. Claude Code and Codex hooks capture durable conversation content, a background queue extracts useful decisions and lessons, and later compilation turns append-only daily logs into cross-referenced Markdown articles. Retrieval reads a structured index instead of using a vector database.

## Data Flow

```text
Claude Code SessionEnd / PreCompact --\
                                      +--> local parser --> SQLite queue --> worker
Codex SessionEnd --------------------/                           |
                                                                 v
                                           Codex (ChatGPT login) --> Claude fallback
                                                                 |
                                                                 v
                                        daily/ --> staged compile --> knowledge/

Claude Code / Codex SessionStart --> local index and recent-log context
```

Hooks perform local parsing and enqueue work; they never call a model. The worker prefers Codex, uses `gpt-5.6-luna` for extraction and semantic lint, and uses `gpt-5.6-terra` for synthesis and staged edits. If Codex authentication, capacity, timeout, command execution, output validation, or staged validation fails, the same job falls back to the subscription-backed Claude Agent SDK. Provider attempts and fallback reasons remain available in the queue and usage log.

All model-driven edits happen in disposable staging directories. Host Python validates each stage and applies approved files under one writer lock with a recovery journal. Models never write directly to the real knowledge base.

## Requirements and Authentication

- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/)
- Claude Code with subscription credentials for fallback
- Codex CLI 0.146.1 or newer, logged in through ChatGPT

Verify Codex before enabling it:

```bash
codex --version
codex login status
```

The second command must exit successfully and print `Logged in using ChatGPT`. The memory provider rejects API-key login, unknown login states, missing credentials, and nonzero status results. It never logs in automatically. It removes `OPENAI_*`, `AZURE_OPENAI_*`, `ANTHROPIC_API_KEY`, and `CLAUDE_API_KEY` variables from provider children, so neither Codex nor the Claude fallback can silently switch to API-key or OpenAI Platform API billing.

Codex usage limits can vary by ChatGPT plan and time window. On a capacity or usage-limit response, the job records the reason and tries Claude. If both subscription providers fail, the queue retries with backoff and eventually moves the job to the dead-letter state; the source snapshot remains local for recovery.

## Quick Start

```bash
git clone https://github.com/coleam00/claude-memory-compiler
cd claude-memory-compiler
uv sync
export AI_MEMORY_HOME="$PWD"
bin/setup-global.sh
```

`AI_MEMORY_HOME` is the canonical absolute path to the central knowledge base. `CLAUDE_MEMORY_HOME` remains a deprecated compatibility alias. If both variables are set, they must resolve to the same path. Provider children receive `AI_MEMORY_INTERNAL_JOB=1`; hooks check that guard before touching the queue or spool, which prevents recursive capture.

The setup helper only prints instructions. Follow them to merge the project hooks into both user configurations:

- Claude Code: `~/.claude/settings.json`
- Codex: `~/.codex/hooks.json`, using `.codex/hooks.json.example`

Preserve existing settings and hooks; do not replace either file. Start each agent from a new terminal that exports `AI_MEMORY_HOME`. To capture only sessions started inside this repository, keep the project-local `.claude/settings.json` and skip the global merge.

After merging `~/.codex/hooks.json`, launch Codex interactively. In the hook trust review, compare the new or changed hook commands and hashes with `.codex/hooks.json.example` and the checked-out hook scripts. Approve only the vetted repository hooks. Before relying on live capture, verify that both repository hooks appear as enabled and trusted. Repeat this review whenever a hook command or hash changes.

## What Gets Captured

The local normalizers retain ordinary user and assistant messages, explicit user decisions, and selected completed subagent findings. They exclude developer instructions, hidden reasoning, token accounting, routine tool calls and output, and asynchronous launch acknowledgements before content reaches an extraction provider.

The resulting source entry keeps the existing daily-log schema and adds provenance:

```markdown
### Session [project-key] (14:20) - Brief title

**Agent:** Codex
**Project:** project-key
**CWD:** /full/path/to/project
```

Transcripts, queue rows, stages, spools, daily logs, and knowledge files remain stored locally, but model-backed operations transmit task inputs to the selected ChatGPT-authenticated Codex or Claude subscription provider. Extraction may send normalized transcript content. A text query sends its prompt containing the full index and every concept, connection, and Q&A article. Semantic lint sends its prompt containing the full index and all articles. A compile sends its prompt plus a staged workspace containing the schema, selected daily log, index, build log, every article, and compatible state when present; its output allowlist covers all concept and connection articles, the index, and the build log. A filed answer also receives the full knowledge base in its prompt and a stage containing every article, with writes allowed to Q&A articles, the index, and the build log. A connection pass receives its prompt, schema, index, build log, and staged candidate and bridge concept articles, with writes allowed to connection articles, the index, and the build log. The exclusions above still apply before extraction, and local structural lint sends nothing to either provider. Structured logs omit transcript bodies and credentials. Codex conversations that exist only in cloud history remain unavailable: live capture and historical import require a local transcript or hook event.

## Commands

```bash
uv run python scripts/compile.py                       # compile new or changed daily logs
uv run python scripts/connections.py --dry-run        # preview connection candidates locally
uv run python scripts/connections.py --top 40         # confirm and write connections
uv run python scripts/query.py "question"             # query without writing
uv run python scripts/query.py "question" --file-back # query and stage a filed answer
uv run python scripts/lint.py                         # structural and semantic checks
uv run python scripts/lint.py --structural-only       # local checks; no provider call
uv run python scripts/worker.py --drain               # recover leases and drain ready jobs
```

At personal scale, the model can select relevant articles from `knowledge/index.md` more accurately than cosine similarity. Consider hybrid retrieval only when the index grows beyond the available context window.

## Historical Import

Claude history is discovered under `~/.claude/projects/`; Codex history is discovered recursively under `~/.codex/sessions/`. Preview first:

```bash
uv run python scripts/batch-flush.py --source codex --dry-run
uv run python scripts/batch-flush.py --source codex --dates 2026-04-11 --dry-run
uv run python scripts/batch-flush.py --source codex --from-date 2026-04-01 --to-date 2026-04-30 --dry-run
```

A dry run parses, filters, chunks, checks deduplication, and estimates tokens and model tasks. It makes no model calls and writes no queue, state, daily-log, or knowledge-base data. Estimates describe possible ChatGPT subscription usage, not dollar charges.

After reviewing the preview, import with bounded concurrency and resumability:

```bash
uv run python scripts/batch-flush.py --source codex --resume --concurrency 2
uv run python scripts/batch-flush.py --source all --resume --concurrency 2
```

`--resume` uses the same agent/session/normalized-hash identity as live capture. Repeating an import does not duplicate a completed entry, even when the first attempt used Claude fallback. `--max-cost` remains a legacy Claude-only option and is rejected when `--source` includes Codex.

## Operations and Recovery

Runtime data lives below `scripts/` and is gitignored. Inspect the queue without changing it:

```bash
QUEUE_PATH="$(uv run python -c 'import os; from scripts.config import load_config; print(load_config(os.environ).queue_path)')"
uv run python - "$QUEUE_PATH" <<'PY'
import sqlite3
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
for row in db.execute(
    "SELECT status, count(*) FROM jobs GROUP BY status ORDER BY status"
):
    print(row)
for row in db.execute(
    "SELECT id,kind,source_agent,attempt_count,status,last_error "
    "FROM jobs WHERE status IN ('failed','dead') ORDER BY id"
):
    print(row)
for row in db.execute(
    "SELECT job_id,provider,model,outcome,reason "
    "FROM provider_attempts ORDER BY id DESC LIMIT 20"
):
    print(row)
db.close()
PY
```

The displayed wrapper uses Bash/Zsh command substitution and a heredoc, so run it from a POSIX Bash or Zsh shell. The configuration lookup honors `AI_MEMORY_QUEUE_PATH`, `AI_MEMORY_HOME`, and the legacy root alias, and the Python/SQLite inspection logic itself is cross-platform. On Windows, set the same environment variables in PowerShell and run the Python body with the absolute path returned by `load_config(os.environ).queue_path`, or use the displayed wrapper from a Bash environment. No separate `sqlite3` command-line tool is required.

For pending, failed, or expired leased jobs, fix the underlying authentication, capacity, path, or filesystem problem and run `uv run python scripts/worker.py --drain`. One singleton process owns the live drain, recovers expired leases, and applies bounded retry backoff. Within that process, `AI_MEMORY_WORKER_CONCURRENCY` bounds concurrent provider work (default `2`); provider-attempt persistence, job completion/retry transitions, and daily writes serialize, and daily appends also use the cross-process writer lock. Historical import's explicit `--concurrency N` option separately bounds parallel transcript parsing and provider work. All durable knowledge-base mutations, including daily appends, validated staged applies, markers, state, and usage bookkeeping, remain serialized by the writer lock. Queue/WAL, spool, temporary-stage, and operational-log writes use their own safety boundaries.

A dead job has exhausted its attempts. This release has no supported command to reset or requeue a dead row. Preserve `scripts/jobs.sqlite3*` and the matching private file in `scripts/spool/`, inspect `last_error` and provider attempts with the read-only command above, correct the cause, and obtain an operator-reviewed recovery rather than mutating SQLite directly. Never delete an active spool snapshot or edit a queue payload by hand. Remove a spool file only after its job succeeds and no queue row references it.

An interrupted staged apply leaves `scripts/memory-apply-journal/` in place. Stop new writes and recover it before other maintenance:

```bash
uv run python scripts/reconcile-state.py
```

The command acquires the writer lock, restores an incomplete transaction, and reconciles legacy marker/state drift. Do not delete a persistent journal manually. Investigate malformed or unexpected journals before retrying. Provider outcome records appear in `scripts/logs/usage.jsonl`; Codex records contain tokens when available but never fabricate `cost_usd`.

Legacy `state.json` fields, including per-entry `cost_usd` and top-level `total_cost`, remain readable and survive round trips. They represent historical Claude-reported costs only. New subscription activity uses provider, model, outcome, fallback reason, token, and elapsed-time records instead of invented dollar totals.

## Technical Reference

See [AGENTS.md](AGENTS.md) for the article schema, parser contracts, provider boundaries, queue and staging architecture, and detailed recovery behavior.
