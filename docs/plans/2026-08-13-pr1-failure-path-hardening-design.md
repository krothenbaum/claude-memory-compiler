# PR #1 Failure-Path Hardening Design

## Scope

Fix review findings 1–4 and 6–11. Leave malformed apply journals fail-closed. A malformed published journal may describe a partially applied transaction, so the worker must not quarantine it and continue automatically.

## Capture safety

The hook parent cleans token-owned uncommitted snapshots only when its child exceeds the deadline. A normal nonzero child exit preserves the child's owner-only `failed-<agent>-<token>-<digest>.jsonl` recovery snapshot.

Temporary live transcript slices live under an owner-only directory inside the configured memory root, not the system temporary directory. File-descriptor ownership is explicit: after `os.fdopen` or `os.close` takes or releases ownership, code sets the integer descriptor to `-1`; exception cleanup closes only a still-owned descriptor.

Codex hook commands use `uv run --no-sync`. The three-second host timeout and 2.25-second internal budget remain unchanged. Tests cover cold private-cache startup and ensure the command remains below the host limit without dependency resolution.

## Provider and foreground correctness

Codex version and login probes each receive a small, dedicated timeout rather than the generation timeout. The generation command retains `AI_MEMORY_JOB_TIMEOUT_SECONDS`. Probe timeout, truncation, failure classification, process-group cleanup, and fallback behavior remain unchanged.

A successful query answer survives query-count contention. Query-count bookkeeping remains bounded and atomic, but exhaustion becomes a logged best-effort bookkeeping failure instead of replacing the provider payload.

Injected flush routers load configuration from the current environment with only the selected memory root overridden. Compile bookkeeping receives the already-resolved memory root instead of deriving a second root from the daily-log path. Invalid fallback-workspace results normalize a falsey injected Claude model to `"unknown"`.

## Usage recovery

Deadline-bound live capture queue opens do not run usage-log recovery or projection. SQLite `provider_attempts` remains authoritative. The detached worker performs recovery and projection outside the hook deadline before it processes jobs. Other foreground and maintenance queue opens retain recovery unless they explicitly choose the capture-safe mode.

This design removes the global writer-lock wait and archive scan from the live hook critical path without adding a new persistent index or weakening archive verification.

## Error handling and verification

Malformed apply journals continue to stop workers and require operator review. New tests reproduce every confirmed bug before its fix, including descriptor-number reuse, recovery-snapshot retention, state-CAS exhaustion, capture queue opening while the writer lock is held, explicit-root bookkeeping, falsey model injection, and cold-cache hook startup.

Each coherent change is committed separately. After focused tests pass, run the full suite, spec review, quality review, and update PR #1 only after both reviews approve.
