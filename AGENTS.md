# AGENTS.md - Personal Knowledge Base Schema

> Adapted from [Andrej Karpathy's LLM Knowledge Base](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) architecture.
> Instead of ingesting external articles, this system compiles knowledge from your own AI conversations.

## The Compiler Analogy

```
daily/          = source code    (your conversations - the raw material)
LLM             = compiler       (extracts and organizes knowledge)
knowledge/      = executable     (structured, queryable knowledge base)
lint            = test suite     (health checks for consistency)
queries         = runtime        (using the knowledge)
```

You don't manually organize your knowledge. You have conversations, and the LLM handles the synthesis, cross-referencing, and maintenance.

---

## Architecture

### Layer 1: `daily/` - Conversation Logs (Immutable Source)

Daily logs capture what happened in your AI coding sessions. These are the "raw sources" - append-only, never edited after the fact.

```
daily/
├── 2026-04-01.md
├── 2026-04-02.md
├── ...
```

Each file follows this format:

```markdown
# Daily Log: YYYY-MM-DD

## Sessions

### Session [project-key] (HH:MM) - Brief Title

**Agent:** Claude Code
**Project:** project-key
**CWD:** /full/path/to/working/directory

**Context:** What the user was working on.

**Key Exchanges:**
- User asked about X, assistant explained Y
- Decided to use Z approach because...
- Discovered that W doesn't work when...

**Decisions Made:**
- Chose library X over Y because...
- Architecture: went with pattern Z

**Lessons Learned:**
- Always do X before Y to avoid...
- The gotcha with Z is that...

**Action Items:**
- [ ] Follow up on X
- [ ] Refactor Y when time permits
```

The `project-key` is the basename of the session's working directory (e.g. `main`,
`claude-memory-compiler`, `Po`). It is captured automatically by the SessionEnd /
PreCompact hooks from the `cwd` field of the hook input. Use `unknown` if the
hook could not determine the cwd, and `global` for project-agnostic content.

### Layer 2: `knowledge/` - Compiled Knowledge (LLM-Owned)

The LLM owns this directory entirely. Humans read it but rarely edit it directly.

```
knowledge/
├── index.md              # Master catalog - every article with one-line summary
├── log.md                # Append-only chronological build log
├── concepts/             # Atomic knowledge articles
├── connections/          # Cross-cutting insights linking 2+ concepts
└── qa/                   # Filed query answers (compounding knowledge)
```

### Layer 3: This File (AGENTS.md)

The schema that tells the LLM how to compile and maintain the knowledge base. This is the "compiler specification."

---

## Structural Files

### `knowledge/index.md` - Master Catalog

A table listing every knowledge article. This is the primary retrieval mechanism - the LLM reads this FIRST when answering any query, then selects relevant articles to read in full.

Format:

```markdown
# Knowledge Base Index

| Article | Project | Summary | Compiled From | Updated |
|---------|---------|---------|---------------|---------|
| [[concepts/supabase-auth]] | my-app | Row-level security patterns and JWT gotchas | daily/2026-04-02.md | 2026-04-02 |
| [[connections/auth-and-webhooks]] | my-app, payments-svc | Token verification patterns shared across Supabase auth and Stripe webhooks | daily/2026-04-02.md, daily/2026-04-04.md | 2026-04-04 |
| [[concepts/python-uv-basics]] | global | Using uv for Python dependency management | daily/2026-04-03.md | 2026-04-03 |
```

The Project column lists the project-keys from the originating sessions. Use a
comma-separated list when an article spans multiple projects, and `global` for
project-agnostic content. Articles compiled before this column existed will
have Project = `unknown` until they are recompiled or backfilled.

### `knowledge/log.md` - Build Log

Append-only chronological record of every compile, query, and lint operation.

Format:

```markdown
# Build Log

## [2026-04-01T14:30:00] compile | Daily Log 2026-04-01
- Source: daily/2026-04-01.md
- Articles created: [[concepts/nextjs-project-structure]], [[concepts/tailwind-setup]]
- Articles updated: (none)

## [2026-04-02T09:00:00] query | "How do I handle auth redirects?"
- Consulted: [[concepts/supabase-auth]], [[concepts/nextjs-middleware]]
- Filed to: [[qa/auth-redirect-handling]]
```

---

## Article Formats

### Concept Articles (`knowledge/concepts/`)

One article per atomic piece of knowledge. These are facts, patterns, decisions, preferences, and lessons extracted from your conversations.

```markdown
---
title: "Concept Name"
aliases: [alternate-name, abbreviation]
tags: [domain, topic]
project: my-app          # single project-key, or list e.g. [my-app, payments-svc], or `global`
sources:
  - "daily/2026-04-01.md"
  - "daily/2026-04-03.md"
created: 2026-04-01
updated: 2026-04-03
---

# Concept Name

[2-4 sentence core explanation]

## Key Points

- [Bullet points, each self-contained]

## Details

[Deeper explanation, encyclopedia-style paragraphs]

## Related Concepts

- [[concepts/related-concept]] - How it connects

## Sources

- [[daily/2026-04-01.md]] - Initial discovery during project setup
- [[daily/2026-04-03.md]] - Updated after debugging session
```

### Connection Articles (`knowledge/connections/`)

Cross-cutting synthesis linking 2+ concepts. Created when a conversation reveals a non-obvious relationship.

```markdown
---
title: "Connection: X and Y"
connects:
  - "concepts/concept-x"
  - "concepts/concept-y"
project: [my-app, payments-svc]   # list when the connection spans projects (the common case)
sources:
  - "daily/2026-04-04.md"
created: 2026-04-04
updated: 2026-04-04
---

# Connection: X and Y

## The Connection

[What links these concepts]

## Key Insight

[The non-obvious relationship discovered]

## Evidence

[Specific examples from conversations]

## Related Concepts

- [[concepts/concept-x]]
- [[concepts/concept-y]]
```

### Q&A Articles (`knowledge/qa/`)

Filed answers from queries. Every complex question answered by the system can be permanently stored, making future queries smarter.

```markdown
---
title: "Q: Original Question"
question: "The exact question asked"
consulted:
  - "concepts/article-1"
  - "concepts/article-2"
filed: 2026-04-05
---

# Q: Original Question

## Answer

[The synthesized answer with [[wikilinks]] to sources]

## Sources Consulted

- [[concepts/article-1]] - Relevant because...
- [[concepts/article-2]] - Provided context on...

## Follow-Up Questions

- What about edge case X?
- How does this change if Y?
```

---

## Core Operations

### 1. Compile (daily/ -> knowledge/)

When processing a daily log:

1. Read the daily log file
2. Read `knowledge/index.md` to understand current knowledge state
3. Read existing articles that may need updating
4. For each piece of knowledge found in the log:
   - If an existing concept article covers this topic: UPDATE it with new information, add the daily log as a source
   - If it's a new topic: CREATE a new `concepts/` article
5. If the log reveals a non-obvious connection between 2+ existing concepts: CREATE a `connections/` article
6. UPDATE `knowledge/index.md` with new/modified entries
7. APPEND to `knowledge/log.md`

**Important guidelines:**
- A single daily log may touch 3-10 knowledge articles
- Prefer updating existing articles over creating near-duplicates
- Use Obsidian-style `[[wikilinks]]` with full relative paths from knowledge/
- Write in encyclopedia style - factual, concise, self-contained
- Every article must have YAML frontmatter
- Every article must link back to its source daily logs

### 2. Query (Ask the Knowledge Base)

1. Read `knowledge/index.md` (the master catalog)
2. Based on the question, identify 3-10 relevant articles from the index
3. Read those articles in full
4. Synthesize an answer with `[[wikilink]]` citations
5. If `--file-back` is specified: create a `knowledge/qa/` article and update index.md and log.md

**Why this works without RAG:** At personal knowledge base scale (50-500 articles), the LLM reading a structured index outperforms cosine similarity. The LLM understands what the question is really asking and selects pages accordingly. Embeddings find similar words; the LLM finds relevant concepts.

### 3. Lint (Health Checks)

Seven checks, run periodically:

1. **Broken links** - `[[wikilinks]]` pointing to non-existent articles
2. **Orphan pages** - Articles with zero inbound links from other articles
3. **Orphan sources** - Daily logs that haven't been compiled yet
4. **Stale articles** - Source daily log changed since article was last compiled
5. **Contradictions** - Conflicting claims across articles (requires LLM judgment)
6. **Missing backlinks** - A links to B but B doesn't link back to A
7. **Sparse articles** - Below 200 words, likely incomplete

Output: a markdown report with severity levels (error, warning, suggestion).

---

## Conventions

- **Wikilinks:** Use Obsidian-style `[[path/to/article]]` without `.md` extension
- **Writing style:** Encyclopedia-style, factual, third-person where appropriate
- **Dates:** ISO 8601 (YYYY-MM-DD for dates, full ISO for timestamps in log.md)
- **File naming:** lowercase, hyphens for spaces (e.g., `supabase-row-level-security.md`)
- **Frontmatter:** Every article must have YAML frontmatter with at minimum: title, sources, created, updated
- **Sources:** Always link back to the daily log(s) that contributed to an article

---

## Full Project Structure

```
llm-personal-kb/
|-- .claude/
|   |-- settings.json                # Claude Code hook commands
|-- .codex/
|   |-- hooks.json.example           # Opt-in Codex hook commands
|-- .gitignore                       # Excludes runtime state, temp files, caches
|-- AGENTS.md                        # This file - schema + full technical reference
|-- README.md                        # Concise overview + quick start
|-- pyproject.toml                   # Dependencies (at root so hooks can find it)
|-- daily/                           # "Source code" - conversation logs (immutable)
|-- knowledge/                       # "Executable" - compiled knowledge (LLM-owned)
|   |-- index.md                     #   Master catalog - THE retrieval mechanism
|   |-- log.md                       #   Append-only build log
|   |-- concepts/                    #   Atomic knowledge articles
|   |-- connections/                 #   Cross-cutting insights linking 2+ concepts
|   |-- qa/                          #   Filed query answers (compounding knowledge)
|-- scripts/                         # CLI tools
|   |-- batch-flush.py               # Historical Claude/Codex import
|   |-- capture.py                   # Fast snapshot and enqueue boundary
|   |-- auto-compile.py              # Reserved end-of-day compile coordinator
|   |-- compile.py                   #   Compile daily logs -> knowledge articles
|   |-- connections.py               #   Cross-graph connection discovery
|   |-- providers.py                 # Subscription providers and fallback router
|   |-- queue.py                     # SQLite jobs, leases, attempts, and retry
|   |-- query.py                     #   Ask questions (index-guided, no RAG)
|   |-- lint.py                      #   7 health checks
|   |-- flush.py                     # Extraction prompt and daily-log writer
|   |-- staging.py                   # Staged validation and atomic apply journal
|   |-- transcripts.py               # Claude/Codex normalizers
|   |-- usage.py                     # Provider-neutral usage JSONL
|   |-- worker.py                    # Detached queue drain
|   |-- config.py                    #   Path constants
|   |-- utils.py                     #   Shared helpers
|-- hooks/                           # Claude Code and Codex adapters
|   |-- codex-session-start.py       #   Injects shared local context into Codex
|   |-- codex-session-end.py         #   Enqueues a Codex transcript slice
|   |-- session-start.py             #   Injects knowledge into every session
|   |-- session-end.py               #   Enqueues a Claude transcript slice
|   |-- pre-compact.py               #   Enqueues context before compaction
|-- reports/                         # Lint reports (gitignored)
```

---

## Runtime Architecture

```text
Claude hook --\
               +--> transcript normalizer --> private spool --> SQLite queue
Codex hook ----/                                           |
                                                            v
                                             singleton detached worker
                                                            |
                      Codex (ChatGPT auth) --> Claude subscription fallback
                                                            |
                                      text result or disposable staged workspace
                                                            |
                                      validate --> writer lock --> recovery journal
                                                            |
                                      daily/ + knowledge/ + log + marker + state
```

Capture, model execution, and durable writes are separate boundaries. Hooks perform bounded local work and return within the host timeout. Live capture uses one singleton worker process with bounded concurrent provider work; provider-attempt persistence, job completion/retry transitions, and daily writes serialize within that process. Historical import's explicit `--concurrency N` option separately bounds parallel transcript parsing and provider work. All durable knowledge-base mutations—daily appends, validated staged applies, markers, state, and usage bookkeeping—remain serialized by the writer lock. Queue/WAL, spool, temporary-stage, and operational-log writes use their own safety boundaries. Models never write directly to the real knowledge root.

## Configuration and Subscription Authentication

`AI_MEMORY_HOME` is the canonical knowledge-base root. `CLAUDE_MEMORY_HOME` remains a deprecated compatibility alias. The loader rejects empty roots and rejects different resolved values when both variables are set.

| Variable | Default | Purpose |
|---|---|---|
| `AI_MEMORY_HOME` | repository root | Canonical memory root |
| `CLAUDE_MEMORY_HOME` | unset | Compatibility alias |
| `AI_MEMORY_PROVIDER_ORDER` | `codex,claude` | Fixed provider order |
| `AI_MEMORY_CODEX_LUNA_MODEL` | `gpt-5.6-luna` | Extraction and semantic lint |
| `AI_MEMORY_CODEX_TERRA_MODEL` | `gpt-5.6-terra` | Synthesis and staged edits |
| `AI_MEMORY_CLAUDE_MODEL` | `claude-sonnet-5` | Claude subscription fallback |
| `AI_MEMORY_JOB_TIMEOUT_SECONDS` | `900` | Provider attempt timeout |
| `AI_MEMORY_QUEUE_PATH` | `$AI_MEMORY_HOME/scripts/jobs.sqlite3` | Absolute queue path |
| `AI_MEMORY_WORKER_CONCURRENCY` | `2` | Concurrent live-worker provider jobs; durable writes serialize |
| `AI_MEMORY_INTERNAL_JOB` | unset | Recursion guard for provider children |
| `AI_MEMORY_USAGE_ESTIMATE_ONLY` | `0` | Advisory usage estimation mode |

Each Codex attempt first runs the bounded `codex --version` preflight and requires version 0.146.1 or newer, then runs `codex login status`. It proceeds only when the commands exit zero and the login output contains the exact `Logged in using ChatGPT` status without a competing login mode. An old or malformed version, API-key login, unknown status, missing CLI, timeout, truncated output, or nonzero exit rejects Codex and records a fallback reason. The provider never logs in automatically.

Both providers receive a minimal child environment with `OPENAI_*`, `AZURE_OPENAI_*`, `ANTHROPIC_API_KEY`, and `CLAUDE_API_KEY` removed. Codex text jobs run ephemeral, read-only, and noninteractive. Workspace jobs run ephemeral with write access confined to a disposable stage. The implementation never uses OpenAI Platform API billing or a sandbox-bypass flag.

## Hooks and Shared Retrieval

`.claude/settings.json` defines Claude Code `SessionStart`, `PreCompact`, and `SessionEnd`. `.codex/hooks.json.example` defines Codex `SessionStart` and `SessionEnd` for codex-cli 0.146.1 or newer. Global installation is opt-in: `bin/setup-global.sh` prints non-destructive merge instructions for `~/.claude/settings.json` and `~/.codex/hooks.json`.

Run `uv sync` in the repository before installing or updating Codex hooks. The
Codex hook commands use `uv run --no-sync` so dependency resolution stays out
of the three-second SessionEnd timeout.

After merging the Codex configuration, launch Codex interactively. In the hook trust review, compare the new or changed hook commands and hashes with `.codex/hooks.json.example` and the checked-out hook scripts. Approve only the vetted repository hooks. Before relying on live capture, verify that both repository hooks appear as enabled and trusted. Repeat this review whenever a hook command or hash changes.

Gate-only exception: disposable Gate 2 automation may use `--dangerously-bypass-hook-trust` for one already-vetted invocation when an interactive review cannot be persisted in the disposable profile. Codex labels this flag **DANGEROUS**. Never persist it in configuration, aliases, scripts, or normal launch commands.

Both SessionStart adapters call the same local context builder. It reads `knowledge/index.md` and recent project/global daily sections, then emits the host-specific JSON envelope. It makes no model call.

The end and pre-compaction adapters:

1. Stop before file or queue work when `AI_MEMORY_INTERNAL_JOB=1` or the legacy `CLAUDE_INVOKED_BY` guard exists.
2. Validate the hook payload and transcript path.
3. Create a bounded normalized slice and a private queue-owned spool snapshot.
4. Insert or find the job by `(kind, source_agent, session_id, source_hash)`.
5. Launch `scripts/worker.py --drain` as a detached process.
6. Return without waiting for a provider.

Claude keeps both PreCompact and SessionEnd because compaction can discard intermediate context. Canonical normalized hashing deduplicates equivalent slices. Codex uses SessionEnd; its adapter fills missing hook metadata from `session_meta`.

## Transcript Normalization and Privacy

`scripts/transcripts.py` produces immutable `NormalizedSession` values containing the agent, session ID, project, CWD, timestamp, trigger, normalized turns, source path, and canonical source hash.

Claude JSONL records store user and assistant content under `message`. The parser also recognizes `AskUserQuestion` decisions and completed `Agent` or `Task` findings. Codex JSONL records use `type == "response_item"`; user `input_text` and assistant `output_text` come from `payload.type == "message"`. The parser uses `session_meta` for identity and location when necessary.

Before provider execution, both parsers exclude developer instructions, hidden reasoning, duplicate `event_msg.agent_message` records, routine tool calls and output, token counts, session instructions, and asynchronous launch acknowledgements. Unknown tool records remain excluded. Selected explicit choices and completed collaboration findings use a small allowlist and bounded text.

Live-hook operational records append as compact JSON Lines in `scripts/logs/hooks.log`. Every record has `timestamp` (UTC ISO 8601), `level`, `component`, `event`, `logger`, and `message` fields. Message quotes, control characters, and newlines remain JSON-encoded inside one physical line; implicit exception tracebacks are excluded.

Transcripts, stages, queue rows, usage logs, daily logs, and knowledge files remain stored locally, but model-backed operations transmit task inputs to the selected ChatGPT-authenticated Codex or Claude subscription provider. Extraction may send normalized transcript content. A text query sends a prompt containing the full index and every concept, connection, and Q&A article. Semantic lint sends a prompt containing the full index and all articles. Compile sends a prompt plus a staged copy of the schema, selected daily log, index, build log, every article, and compatible state when present; its broad current output allowlist covers all concept and connection articles, the index, and the build log. A filed answer receives the full knowledge base in its prompt and a stage containing every article; its output allowlist covers Q&A articles, the index, and the build log. A connection pass receives its prompt, schema, index, build log, and staged candidate and bridge concept articles; its output allowlist covers connection articles, the index, and the build log. The parser exclusions above still apply before extraction. Local structural lint sends no content to either provider. Logs retain job metadata and bounded errors, not transcript bodies or credentials. Codex cloud-only history cannot be imported unless Codex exposes a local transcript or hook event.

## Provider Layer and Model Routing

All LLM-backed operations use `GenerationProvider` and `ProviderRouter` from `scripts/providers.py`:

| Operation | Task kind | Codex model | Request type |
|---|---|---|---|
| Live or historical extraction | `EXTRACT` | `gpt-5.6-luna` | text |
| Contradiction lint | `SEMANTIC_LINT` | `gpt-5.6-luna` | text |
| Compile | `COMPILE` | `gpt-5.6-terra` | staged workspace |
| Query | `QUERY` | `gpt-5.6-terra` | text |
| Filed answer | `FILE_ANSWER` | `gpt-5.6-terra` | staged workspace |
| Connection confirmation | `CONNECTIONS` | `gpt-5.6-terra` | staged workspace |

The router tries Codex once. Authentication, capacity, timeout, command, invalid-output, and staged-validation failures cause a fresh Claude attempt. A successful Codex result never calls Claude. If both fail, the logical job remains retryable. Structural lint is entirely local.

Capacity and usage limits are subscription constraints, not dollar balances. The queue and `scripts/logs/usage.jsonl` record provider, model, task, outcome, fallback reason, tokens when available, elapsed time, and timestamp. Codex usage never receives a fabricated dollar cost.

## Queue, Leases, and Worker

`scripts/jobs.sqlite3` uses SQLite WAL mode and contains `jobs` plus `provider_attempts`. The job identity includes source agent, so equal Claude and Codex session IDs remain distinct; provider fallback stays an attempt on one job. Job states are `pending`, `leased`, `succeeded`, `failed`, and `dead`.

Claims use short immediate transactions. A worker renews each lease while its provider or writer runs, recovers expired leases after a crash, retries transient failure with bounded exponential backoff and jitter, and marks a job dead after the attempt limit. Multiple hooks may start workers, but a singleton drain lock lets only one live worker process own the queue drain; a later worker exits successfully when another healthy worker owns it. Within the owner process, `AI_MEMORY_WORKER_CONCURRENCY` bounds concurrent provider jobs (default `2`), while attempt persistence, terminal queue transitions, and daily writes serialize. Historical import uses its separate `--concurrency N` bound for parallel parsing and provider work. Durable knowledge-base mutations still serialize through the writer lock. Queue/WAL, spool, temporary-stage, and operational-log writes retain their separate transaction, permission, atomic-file, and lock boundaries.

The singleton live worker restores end-of-day compilation after a successful daily append. When its queue drain becomes idle at or after 16:00 local time, it fingerprints content after today's last `@compiled-through` marker. One immediate SQLite transaction assigns the request either the owner role or the single live watcher role. Every later successful idle drain replaces the pending fingerprint with its latest observation and installs a watcher only when no live watcher exists. Owner and watcher spawn failures clear only their matching roles. The watcher heartbeats while the owner lease is live, polls the automatic-child lock and marker, exits when the request is covered or released, and atomically takes owner identity after expiry. A later drain can replace an expired watcher.

The detached owner rechecks the fingerprint, reservation, and absence of pending, failed, or leased jobs. It holds an immediate queue transaction across the `compile.py` launch, so an enqueue that commits before launch cancels compilation and one that commits afterward belongs to a later drain. When the child exits, the coordinator reads the post-compile fingerprint inside an immediate transaction. It releases if the marker covers all observed content, cancels if queue work arrived, or promotes the current fingerprint and serially runs one coalesced follow-up generation. Exit code 75 is a deferred overlap signal: the owner retains and renews the request, then polls `scripts/memory-auto-compile.lock` and the marker until the orphan child covers the content or the lock clears for retry. Other unchanged failures release instead of looping; changed content remains retryable.

The scheduler requires zero interactive Claude Code and Codex CLI sessions. It excludes the Claude Agent SDK's bundled provider child and only the memory provider's complete fixed Codex `exec` signature, including its sandbox, workspace/output relationship, memory-specific output filename, optional schema, and final stdin marker. Partial or prompt-mimicked signatures remain interactive. Process inspection fails closed when unavailable or ambiguous. A failed extraction, a drain with no successful daily write, pending or retrying work, an active interactive session, an unreadable daily log, or fully compiled content does not trigger compilation. The coordinator and compile child receive `AI_MEMORY_INTERNAL_JOB=1` to prevent recursive capture.

Queue inspection is read-only:

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

The displayed wrapper uses Bash/Zsh command substitution and a heredoc, so run it from a POSIX Bash or Zsh shell. The configuration lookup honors `AI_MEMORY_QUEUE_PATH`, `AI_MEMORY_HOME`, and `CLAUDE_MEMORY_HOME`, and the Python/SQLite inspection logic itself is cross-platform. On Windows, set the same environment variables in PowerShell and run the Python body with the absolute queue path returned by `load_config(os.environ).queue_path`, or use the displayed wrapper from a Bash environment. After correcting authentication, capacity, or filesystem failures, run `uv run python scripts/worker.py --drain`. This recovers expired leases and drains ready retry jobs.

A dead job has exhausted its attempts. The current CLI exposes no supported reset or requeue command. Preserve the queue database and retained spool input, inspect `last_error` and attempts read-only, correct the root cause, and obtain an operator-reviewed recovery. Do not mutate SQLite directly or delete active queue payloads.

## Staging and Single-Writer Transactions

Compile, connection, and filed-answer providers edit fresh owner-only stages. Compile and filed-answer currently copy every article; compile also copies the selected daily input and compatible state. Connections copy only their candidate and bridge concept articles. Every stage contains the schema, index, and build log. Output allowlists remain deliberately broader where the operation may create or update articles. Validation rejects unexpected paths outside those allowlists, escaping links, special files, structural deletion, malformed UTF-8 or frontmatter, source-daily edits, and incomplete article/index/log change sets. A rejected Codex stage is discarded before Claude receives a fresh stage.

Approved changes pass through `scripts/memory-writer.lock`. The host rechecks real-file baselines, writes an fsynced journal of original and replacement bytes, applies same-directory atomic replacements, and commits markers, state, and usage bookkeeping. Any failure restores original bytes and leaves the job retryable. Daily-log appends use the same lock.

An interrupted apply leaves `scripts/memory-apply-journal/`. Before other maintenance, run:

```bash
uv run python scripts/reconcile-state.py
```

The command acquires the writer lock, restores an incomplete transaction, and reconciles legacy compile markers and state. Do not delete a persistent journal manually; investigate invalid journal identity or content before retrying.

Queue-owned inputs remain under owner-only `scripts/spool/` so failed jobs can retry. Preserve the spool for pending, failed, leased, or dead jobs. Remove a snapshot only after its job succeeds and no queue row references its path.

## Commands

```bash
uv run python scripts/compile.py
uv run python scripts/compile.py --all
uv run python scripts/compile.py --file daily/2026-04-01.md
uv run python scripts/compile.py --dry-run
uv run python scripts/connections.py --dry-run
uv run python scripts/connections.py --top 40
uv run python scripts/query.py "What auth patterns do I use?"
uv run python scripts/query.py "What's my error handling strategy?" --file-back
uv run python scripts/lint.py
uv run python scripts/lint.py --structural-only
uv run python scripts/worker.py --drain
```

Historical discovery reads Claude sessions from `~/.claude/projects/` and Codex sessions from `~/.codex/sessions/**/*.jsonl`:

```bash
uv run python scripts/batch-flush.py --source codex --dry-run
uv run python scripts/batch-flush.py --source codex --dates 2026-04-11 --dry-run
uv run python scripts/batch-flush.py --source codex --from-date 2026-04-01 --to-date 2026-04-30 --dry-run
uv run python scripts/batch-flush.py --source codex --resume --concurrency 2
uv run python scripts/batch-flush.py --source all --resume --concurrency 2
```

Dry run parses, filters, chunks, checks deduplication, and estimates tokens/tasks without a model call or any queue, state, daily-log, or knowledge write. `--resume` shares the live job identity and skips completed sessions regardless of which provider succeeded. `--max-cost` is legacy Claude-only accounting and is rejected when `--source` includes Codex.

## State and Usage Compatibility

`scripts/state.json` still stores `ingested`, `query_count`, `last_lint`, and legacy `total_cost`. Existing per-entry `cost_usd` values and top-level `total_cost` round-trip unchanged. These fields record historical Claude-reported costs only; they do not represent total subscription usage.

New queued operations use SQLite `provider_attempts` as their source of truth. `scripts/logs/usage.jsonl` is a recoverable, bounded projection for operations outside or inside the queue. It records Codex tokens when the CLI provides them and uses an unavailable value otherwise. It never invents `cost_usd` for Codex. Historical previews report advisory token and task estimates because ChatGPT plan limits vary.

Runtime queue, lock, journal, spool, stage, log, state, daily, knowledge, and report files created during tests or operations must never enter implementation commits.

## Dependencies

Python 3.12+ is managed with `uv`. `claude-agent-sdk` supplies the subscription-backed fallback, `python-dotenv` handles environment files, and `tzdata` supplies cross-platform timezone data. Codex is an external CLI and must be version 0.146.1 or newer. Neither provider requires an API key in this design.

---

## Customization

### Additional Article Types

Add directories like `people/`, `projects/`, `tools/` to `knowledge/`. Define the article format in this file (AGENTS.md) and update `utils.py`'s `list_wiki_articles()` to include them.

### Obsidian Integration

The knowledge base is pure markdown with `[[wikilinks]]` - works natively in Obsidian. Point a vault at `knowledge/` for graph view, backlinks, and search.

### Scaling Beyond Index-Guided Retrieval

At ~2,000+ articles / ~2M+ tokens, the index becomes too large for the context window. At that point, add hybrid RAG (keyword + semantic search) as a retrieval layer before the LLM. See Karpathy's recommendation of `qmd` by Tobi Lutke for search at scale.
