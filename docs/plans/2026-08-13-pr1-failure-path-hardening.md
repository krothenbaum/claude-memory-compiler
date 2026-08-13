# PR #1 Failure-Path Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Repair every validated PR #1 review issue except malformed-journal recovery, which remains deliberately fail-closed.

**Architecture:** Preserve the existing queue, staged-write, and fallback architecture. Narrow cleanup to true timeouts, make descriptor ownership explicit, keep live capture independent of usage recovery, bound provider probes separately, and make nonessential bookkeeping unable to erase successful results.

**Tech Stack:** Python 3.12, asyncio, sqlite3/WAL, pytest, uv, Codex CLI hooks, cross-platform filesystem primitives.

---

### Task 1: Preserve failed capture snapshots and descriptor ownership

**Files:**
- Modify: `hooks/session-end.py`
- Modify: `scripts/capture.py`
- Test: `scripts/tests/test_hooks.py`
- Test: `scripts/tests/test_capture.py`

**Step 1: Write failing tests**

Add tests that prove:

- a timeout invokes `on_timeout` and removes only an uncommitted token-owned capture;
- a child exit 1 does not invoke `on_timeout` and preserves content-addressed `failed-<agent>-<digest>.jsonl`;
- identical ordinary failures from different capture tokens deduplicate safely, distinct content remains distinct, and true-timeout cleanup remains token-scoped to uncommitted capture paths;
- both `_private_spool_copy` and `bounded_transcript_slice` leave an unrelated descriptor open when the original descriptor number is reused after ownership transfer;
- the failed snapshot is owner-only and remains absent from the queue.

**Step 2: Run RED tests**

Run:

```bash
uv run pytest scripts/tests/test_hooks.py scripts/tests/test_capture.py -q -k 'nonzero or failed_snapshot or reused_descriptor or deadline_cleanup'
```

Expected: failures showing nonzero cleanup deletes the snapshot and both reuse probes close the unrelated descriptor.

**Step 3: Implement the minimum fix**

- Call `on_timeout` only before process start when no budget remains and after `subprocess.TimeoutExpired`.
- Remove the nonzero-exit cleanup call.
- Set each descriptor to `-1` immediately after `os.close` or `os.fdopen` takes ownership.
- Close a descriptor in exception cleanup only when it is nonnegative.

**Step 4: Run GREEN tests and focused suites**

```bash
uv run pytest scripts/tests/test_hooks.py scripts/tests/test_capture.py -q
```

Expected: all pass.

**Step 5: Commit**

```bash
git add hooks/session-end.py scripts/capture.py scripts/tests/test_hooks.py scripts/tests/test_capture.py
git commit -m "fix: preserve failed live captures"
```

### Task 2: Keep live transcript slices inside the memory root

**Files:**
- Modify: `hooks/session-end.py`
- Modify: `hooks/pre-compact.py` if helper inputs change
- Modify: `hooks/codex-session-end.py` if helper inputs change
- Test: `scripts/tests/test_hooks.py`

**Step 1: Write failing tests**

Assert that Claude SessionEnd, PreCompact, and Codex SessionEnd place every live slice below an owner-only memory-root runtime directory, reject linked or unsafe parents, use mode `0600`, and remove the slice on success, failure, and cancellation.

**Step 2: Run RED tests**

```bash
uv run pytest scripts/tests/test_hooks.py -q -k 'slice and memory_root'
```

Expected: the observed path is the system temporary directory.

**Step 3: Implement the minimum fix**

Create or reuse a no-follow, owner-only runtime directory under `AI_MEMORY_HOME/scripts/`. Retain a descriptor for every slice write and validate that the preview path still names that descriptor before and after preview. Use pinned directory-relative creation when available. Without secure `dir_fd`, safely create a fresh root/scripts/runtime one component at a time with no-link/reparse, owner, permission, and pre/post identity checks; fail closed on unsafe ancestry, swaps, or permission-establishment failure. On Windows, establish and verify protected current-owner-only DACLs for fresh components, verify existing ancestry without accepting inherited or additional ACEs, and fail closed if the security API is unavailable. Preserve standalone hook imports and the internal deadline.

**Step 4: Run GREEN tests**

```bash
uv run pytest scripts/tests/test_hooks.py -q
```

**Step 5: Commit**

```bash
git add hooks/session-end.py hooks/pre-compact.py hooks/codex-session-end.py scripts/tests/test_hooks.py
git commit -m "fix: confine live transcript slices"
```

### Task 3: Bound Codex preflight probes

**Files:**
- Modify: `scripts/providers.py`
- Test: `scripts/tests/test_providers.py`

**Step 1: Write failing tests**

Cover version and login probes independently. Assert the complete Codex attempt uses one monotonic request deadline: each probe receives `min(5 seconds, remaining)`, generation receives the remaining budget, an exhausted deadline starts no later command, timeout kills the process group, the recorded reason identifies the correct probe, and fallback occurs once.

**Step 2: Run RED tests**

```bash
uv run pytest scripts/tests/test_providers.py -q -k 'preflight and timeout'
```

Expected: probes receive the full request timeout.

**Step 3: Implement the minimum fix**

Add one named preflight timeout constant and one monotonic deadline per attempt. Extend `_run_command` with an optional timeout override, pass `min(preflight timeout, remaining)` to version and login, and pass the remaining deadline to generation.

**Step 4: Run GREEN tests**

```bash
uv run pytest scripts/tests/test_providers.py -q
```

**Step 5: Commit**

```bash
git add scripts/providers.py scripts/tests/test_providers.py
git commit -m "fix: bound Codex preflight probes"
```

### Task 4: Remove usage recovery from the capture deadline

**Files:**
- Modify: `scripts/queue.py`
- Modify: `scripts/capture.py`
- Modify: `scripts/worker.py`
- Modify: `scripts/usage.py` only if a narrow public recovery entry point is needed
- Test: `scripts/tests/test_capture.py`
- Test: `scripts/tests/test_queue.py`
- Test: `scripts/tests/test_usage.py`

**Step 1: Write failing tests**

Add tests that:

- hold `scripts/memory-writer.lock` while a live capture opens the queue and prove enqueue does not wait on usage recovery;
- install several valid archives and prove capture queue construction does not read or hash them;
- prove only the singleton-winning worker performs recovery/projection, exactly once and before stale recovery or processing;
- prove a losing worker reads/hashes no archives and does not wait for the writer lock;
- prove corrupted active logs and tampered archives still recover or quarantine when the worker opens;
- prove authoritative provider attempts project exactly once after capture skipped projection.

**Step 2: Run RED tests**

```bash
uv run pytest scripts/tests/test_capture.py scripts/tests/test_queue.py scripts/tests/test_usage.py -q -k 'capture and usage'
```

Expected: capture blocks or reads archives during `QueueRepository` construction.

**Step 3: Implement the minimum fix**

Add an explicit repository option such as `sync_usage=True` and a narrow explicit synchronization method. Live capture and detached-worker construction pass `False`. After acquiring singleton ownership, the winning worker invokes synchronization once before stale recovery or claiming jobs. Do not change archive formats, integrity checks, or corrupt-log quarantine rules.

**Step 4: Run GREEN tests**

```bash
uv run pytest scripts/tests/test_capture.py scripts/tests/test_queue.py scripts/tests/test_usage.py scripts/tests/test_staging.py -q
```

**Step 5: Commit**

```bash
git add scripts/queue.py scripts/capture.py scripts/worker.py scripts/usage.py scripts/tests/test_capture.py scripts/tests/test_queue.py scripts/tests/test_usage.py
git commit -m "fix: keep capture queue opens deadline-safe"
```

### Task 5: Preserve successful query answers

**Files:**
- Modify: `scripts/query.py`
- Test: `scripts/tests/test_operation_routing.py`

**Step 1: Change the existing exhaustion test first**

Require `run_query` to return the successful provider answer when all query-count CAS attempts conflict. Assert state bytes remain unchanged and a bounded operational warning is recorded without prompt or answer content.

**Step 2: Run RED test**

```bash
uv run pytest scripts/tests/test_operation_routing.py -q -k 'query_state_cas_exhaustion'
```

Expected: the function returns an error envelope instead of the answer.

**Step 3: Implement the minimum fix**

Catch the exhausted bookkeeping conflict, log a bounded warning, and return the existing answer. Do not weaken the CAS or overwrite state.

**Step 4: Run GREEN tests**

```bash
uv run pytest scripts/tests/test_operation_routing.py scripts/tests/test_usage.py -q
```

**Step 5: Commit**

```bash
git add scripts/query.py scripts/tests/test_operation_routing.py
git commit -m "fix: preserve answers across state contention"
```

### Task 6: Correct configuration, root, and fallback-model seams

**Files:**
- Modify: `scripts/flush.py`
- Modify: `scripts/compile.py`
- Modify: `scripts/providers.py`
- Test: `scripts/tests/test_operation_routing.py`
- Test: `scripts/tests/test_providers.py`

**Step 1: Write failing tests**

Assert that:

- an injected flush router observes `AI_MEMORY_JOB_TIMEOUT_SECONDS` from the environment;
- explicit `memory_home` controls early-return compile marker and state bookkeeping even when the log lives elsewhere;
- an injected Claude provider with `_model = None` records model `"unknown"` on invalid fallback workspace output.

**Step 2: Run RED tests**

```bash
uv run pytest scripts/tests/test_operation_routing.py scripts/tests/test_providers.py -q -k 'timeout or explicit_memory_home or unknown_model'
```

**Step 3: Implement the minimum fixes**

- Build injected-router config from `dict(os.environ)` with canonical root override and compatibility alias removal.
- Pass the resolved `home` into compile bookkeeping helpers and use it for relative paths.
- Normalize the fallback model with `getattr(..., None) or "unknown"`.

**Step 4: Run GREEN tests**

```bash
uv run pytest scripts/tests/test_operation_routing.py scripts/tests/test_providers.py scripts/tests/test_staging.py -q
```

**Step 5: Commit**

```bash
git add scripts/flush.py scripts/compile.py scripts/providers.py scripts/tests/test_operation_routing.py scripts/tests/test_providers.py
git commit -m "fix: preserve routed operation configuration"
```

### Task 7: Protect the Codex hook startup margin

**Files:**
- Modify: `.codex/hooks.json.example`
- Modify: `bin/setup-global.sh`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Test: `scripts/tests/test_hooks.py`

**Step 1: Write failing tests**

Require generated and example Codex hook commands to include `uv run --no-sync`. Add a bounded subprocess test with a fresh private uv cache and the committed environment that exercises SessionEnd without provider calls and completes below three seconds. Retain existing large-transcript and deadline-cleanup tests.

**Step 2: Run RED tests**

```bash
uv run pytest scripts/tests/test_hooks.py -q -k 'no_sync or cold_cache'
```

**Step 3: Implement the minimum fix**

Add `--no-sync` to Codex hook commands and document that setup must run `uv sync` before hooks. Do not increase the supported three-second SessionEnd timeout or reduce capture semantics.

**Step 4: Run GREEN tests**

```bash
uv run pytest scripts/tests/test_hooks.py -q
bash -n bin/setup-global.sh
```

**Step 5: Commit**

```bash
git add .codex/hooks.json.example bin/setup-global.sh README.md AGENTS.md scripts/tests/test_hooks.py
git commit -m "fix: protect Codex hook startup margin"
```

### Task 8: Full verification and review

**Files:**
- No new production scope

**Step 1: Run focused combined verification**

```bash
uv run pytest scripts/tests/test_hooks.py scripts/tests/test_capture.py \
  scripts/tests/test_providers.py scripts/tests/test_queue.py \
  scripts/tests/test_usage.py scripts/tests/test_operation_routing.py \
  scripts/tests/test_staging.py -q
```

**Step 2: Run the complete suite and static checks**

```bash
uv run pytest -q
uv run python -m py_compile hooks/*.py scripts/*.py
uv lock --check
git diff --check
git status --short
```

Expected: all tests pass, static checks exit zero, and the worktree is clean after commits.

**Step 3: Spec review**

Dispatch a fresh reviewer for findings 1–4 and 6–11. Resolve every Critical, Important, and Minor issue, then obtain explicit approval.

**Step 4: Quality review**

Dispatch a fresh security/concurrency reviewer. Resolve every finding and obtain explicit approval.

**Step 5: Update PR #1**

Push `feature/codex-memory-integration`, verify the PR head, and add a concise review-resolution summary. Preserve the worktree.
