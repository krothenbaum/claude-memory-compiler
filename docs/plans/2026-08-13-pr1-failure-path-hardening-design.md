# PR #1 Failure-Path Hardening Design

## Scope

Fix review findings 1–4 and 6–11. Leave malformed apply journals fail-closed. A malformed published journal may describe a partially applied transaction, so the worker must not quarantine it and continue automatically.

## Capture safety

The hook parent cleans token-owned uncommitted `capture-<token>-*.jsonl` snapshots only when its child exceeds the deadline. A normal nonzero child exit preserves an owner-only, content-addressed `failed-<agent>-<digest>.jsonl` recovery snapshot. Identical ordinary failures deduplicate atomically; distinct content remains distinct; timeout cleanup never targets ordinary failed snapshots.

Temporary live transcript slices live under an owner-only directory inside the configured memory root, not the system temporary directory. On platforms with secure directory-relative operations, creation and cleanup use pinned no-follow directory descriptors. The slice descriptor remains open for every write, and the preview path must match its device/inode, type, ownership, link count, and mode before and after the callback. On platforms without secure `dir_fd` support, the implementation creates `memory_root/scripts/runtime` one component at a time with no-link/reparse, owner, permission, and pre/post identity checks, then applies the same pre/post parent and file checks around slice creation. Windows traversal uses pinned no-reparse directory handles and verifies that each path still names the same handle identity. Existing current-owner root/scripts ancestry may retain normal inherited SYSTEM and Administrators ACEs. The private runtime and live slice instead receive protected current-owner-only DACLs through their retained handles with `SetSecurityInfo`, then are reverified through those handles and against their paths. A path swap is rejected without applying an ACL to the replacement target; unavailable or unverifiable security APIs fail closed. Unsafe ancestry, swaps, or permission-establishment failures fail closed without leaving an outside artifact.

Codex hook commands use `uv run --no-sync`. The three-second host timeout and 2.25-second internal budget remain unchanged. Tests cover cold private-cache startup and ensure the command remains below the host limit without dependency resolution.

## Provider and foreground correctness

Each Codex attempt has one monotonic deadline equal to `AI_MEMORY_JOB_TIMEOUT_SECONDS`. Version and login probes each receive `min(5 seconds, remaining)`, and generation receives only the remaining attempt budget. No later command starts after exhaustion. Probe timeout, truncation, failure classification, process-group cleanup, fallback behavior, and safe elapsed-time accounting remain unchanged.

A successful query answer survives query-count contention. Query-count bookkeeping remains bounded and atomic, but exhaustion becomes a logged best-effort bookkeeping failure instead of replacing the provider payload.

Injected flush routers load configuration from the current environment with only the selected memory root overridden. Compile bookkeeping receives the already-resolved memory root instead of deriving a second root from the daily-log path. Invalid fallback-workspace results normalize a falsey injected Claude model to `"unknown"`.

## Usage recovery

Deadline-bound live capture queue opens do not run usage-log recovery or projection. SQLite `provider_attempts` remains authoritative. The detached worker also opens the queue without recovery, acquires the singleton drain lock, and only the winning worker performs recovery and projection once before stale-lease recovery or job claims. A losing worker reads and hashes no usage archives and never waits on the writer lock. Other foreground and maintenance queue opens retain recovery unless they explicitly choose the capture-safe mode.

This design removes the global writer-lock wait and archive scan from the live hook critical path without adding a new persistent index or weakening archive verification.

## Error handling and verification

Malformed apply journals continue to stop workers and require operator review. New tests reproduce every confirmed bug before its fix, including descriptor-number reuse, recovery-snapshot retention, state-CAS exhaustion, capture queue opening while the writer lock is held, explicit-root bookkeeping, falsey model injection, and cold-cache hook startup.

Each coherent change is committed separately. After focused tests pass, run the full suite, spec review, quality review, and update PR #1 only after both reviews approve.
