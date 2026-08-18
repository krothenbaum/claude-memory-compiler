"""Compile daily conversation logs through validated provider workspaces."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from pathlib import Path

from config import DAILY_DIR, load_config, now_iso
from providers import (
    ClaudeProvider,
    CodexProvider,
    ProviderRouter,
    TaskKind,
    WorkspaceRequest,
)
from staging import (
    ApplyBookkeeping,
    RetryableApplyError,
    StageValidationError,
    apply_host_bookkeeping,
    apply_validated_stage,
    create_fallback_stage,
    create_stage,
    discard_stage,
    recover_incomplete_apply,
    validate_stage,
)
from utils import (
    ExclusiveFileLock,
    FileBaseline,
    capture_file_baseline,
    file_hash,
    list_raw_files,
    list_wiki_articles,
    load_state_with_baseline,
    read_text_with_baseline,
)
from usage import record_routed_usage, routed_invalid_output

try:
    from scripts.queue import QueueRepository
except ImportError:  # Direct execution with scripts/ on sys.path.
    from queue import QueueRepository  # type: ignore[attr-defined]

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = Path(__file__).resolve().parent / "compile.log"
COMPILED_MARKER_RE = re.compile(r"<!--\s*@compiled-through:([^\s>]+)\s*-->")

logger = logging.getLogger("compile")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


def _default_workspace_router(config, fallback_workspace_factory):
    return ProviderRouter(
        CodexProvider(task_models=config.task_models),
        ClaudeProvider(model=config.claude_model),
        fallback_workspace_factory=fallback_workspace_factory,
    )


def find_last_compiled_offset(content: str) -> int:
    last_end = 0
    for match in COMPILED_MARKER_RE.finditer(content):
        last_end = match.end()
    return last_end


def _home_for(log_path: Path, memory_home: Path | str | None) -> Path:
    if memory_home is not None:
        return Path(memory_home).expanduser().resolve()
    try:
        return log_path.resolve().parents[1]
    except IndexError:
        return ROOT_DIR


def _config(home: Path):
    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(home)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    return load_config(environment)


def _automatic_status_run_id(
    environment: dict[str, str] | None = None,
) -> int | None:
    env = os.environ if environment is None else environment
    if env.get("AI_MEMORY_AUTO_COMPILE") != "1":
        return None
    raw = env.get("AI_MEMORY_STATUS_RUN_ID")
    try:
        run_id = int(raw) if raw is not None else 0
    except ValueError:
        return None
    return run_id if run_id > 0 and str(run_id) == raw else None


def _record_automatic_phase(
    home: Path,
    run_id: int | None,
    phase: str,
    *,
    details: dict[str, int] | None = None,
) -> bool:
    if run_id is None:
        return False
    try:
        config = _config(home)
        with QueueRepository(
            config.queue_path, memory_home=home, sync_usage=False
        ) as repository:
            return repository.record_active_auto_compile_phase(
                run_id, phase, details=details
            )
    except Exception:  # noqa: BLE001 - observability must never block compilation.
        try:
            logger.warning("automatic compile status phase unavailable: %s", phase)
        except Exception:  # noqa: BLE001,S110 - diagnostics are best-effort.
            pass
        return False


def _article_paths(home: Path) -> tuple[str, ...]:
    return tuple(
        path.relative_to(home).as_posix()
        for directory in ("concepts", "connections", "qa")
        for path in sorted((home / "knowledge" / directory).glob("*.md"))
    )


def build_compile_prompt(
    *, schema: str, wiki_index: str, source_name: str, new_content: str,
    mode_description: str, timestamp: str
) -> str:
    """Build a stage-relative compile prompt without host filesystem paths."""
    return f"""You are a knowledge compiler. Extract durable knowledge from the daily
conversation-log slice into structured wiki articles.

## Schema (AGENTS.md)

{schema}

## Current Wiki Index

{wiki_index}

## New Sessions to Compile

**File:** {source_name}
**Mode:** {mode_description}

{new_content}

## Required workflow

1. Identify 1-7 durable concepts and prefer updating an existing article over a duplicate.
2. Preserve each session's **Agent:** and **Project:** provenance in content and source links;
   do not change the article or index schema.
3. Create or edit concept articles below knowledge/concepts/ and genuine non-obvious
   connections below knowledge/connections/.
4. Update knowledge/index.md for every changed article.
5. Append an exact build-log entry headed:
   `## [{timestamp}] compile | {source_name}`
   The entry must cite every changed article. If nothing is extractable, still append
   the heading with `Articles created: (none)` and `Articles updated: (none)`.
6. Never edit AGENTS.md, daily/{source_name}, or scripts/state.json.

All paths are relative to the staged workspace. Do not access files outside it."""


def append_compiled_marker(home: Path, log_path: Path, when: str) -> None:
    relative = log_path.resolve().relative_to(home).as_posix()
    _content, baseline = read_text_with_baseline(log_path)
    apply_host_bookkeeping(
        home,
        ApplyBookkeeping(
            compiled_marker_path=relative,
            compiled_marker_baseline=baseline,
            compiled_at=when,
        ),
    )


def commit_compiled_bookkeeping(
    home: Path,
    log_path: Path,
    state: dict,
    when: str,
    state_baseline: FileBaseline,
    log_baseline: FileBaseline,
) -> None:
    apply_host_bookkeeping(
        home,
        ApplyBookkeeping(
            compiled_marker_path=log_path.resolve().relative_to(home).as_posix(),
            compiled_marker_baseline=log_baseline,
            compiled_at=when,
            state=state,
            state_baseline=state_baseline,
        ),
    )


async def compile_daily_log(
    log_path: Path,
    state: dict,
    state_baseline: FileBaseline,
    *,
    router: object | None = None,
    router_factory: object | None = None,
    memory_home: Path | str | None = None,
) -> float:
    """Compile one daily log and atomically apply the first valid provider stage."""
    home = _home_for(log_path, memory_home)
    config = _config(home)
    full_content, log_baseline = read_text_with_baseline(log_path)
    offset = find_last_compiled_offset(full_content)
    new_content = full_content[offset:].strip()
    ingested = state.setdefault("ingested", {})

    if offset == 0 and log_path.name in ingested:
        prior_hash = ingested[log_path.name].get("hash")
        current_hash = log_baseline.sha256[:16] if log_baseline.sha256 else ""
        if prior_hash == current_hash:
            compiled_at = now_iso()
            ingested[log_path.name] = {
                **ingested[log_path.name],
                "hash": "pending-transaction",
                "compiled_at": compiled_at,
            }
            commit_compiled_bookkeeping(
                home, log_path, state, compiled_at, state_baseline, log_baseline
            )
            return 0.0
    if not new_content:
        compiled_at = now_iso()
        prior = ingested.get(log_path.name, {})
        ingested[log_path.name] = {
            **prior,
            "hash": "pending-transaction",
            "compiled_at": compiled_at,
        }
        commit_compiled_bookkeeping(
            home, log_path, state, compiled_at, state_baseline, log_baseline
        )
        return 0.0

    timestamp = now_iso()
    is_incremental = offset > 0
    is_recompile = offset == 0 and log_path.name in ingested
    mode = (
        "incremental — only this slice is new"
        if is_incremental
        else "recompile — update existing articles without duplication"
        if is_recompile
        else "full — first compilation"
    )
    prompt = build_compile_prompt(
        schema=(home / "AGENTS.md").read_text(encoding="utf-8"),
        wiki_index=(home / "knowledge/index.md").read_text(encoding="utf-8"),
        source_name=log_path.name,
        new_content=new_content,
        mode_description=mode,
        timestamp=timestamp,
    )
    status_run_id = _automatic_status_run_id()
    _record_automatic_phase(home, status_run_id, "staging_started")
    stage = create_stage(
        home,
        f"compile-{log_path.stem}",
        "codex",
        daily_source=log_path.resolve().relative_to(home).as_posix(),
        relevant_articles=_article_paths(home),
        include_state=True,
    )
    allowed = (
        "knowledge/concepts/*.md",
        "knowledge/connections/*.md",
        "knowledge/index.md",
        "knowledge/log.md",
    )
    fallback_holder = []

    def fallback_factory(request: WorkspaceRequest) -> WorkspaceRequest:
        fallback = create_fallback_stage(stage, attempt_id="claude")
        fallback_holder.append(fallback)
        return WorkspaceRequest(
            task=request.task,
            prompt=request.prompt,
            cwd=fallback.root,
            timeout_seconds=request.timeout_seconds,
            output_schema=request.output_schema,
            allowed_paths=request.allowed_paths,
        )

    if router is not None and router_factory is not None:
        raise ValueError("provide router or router_factory, not both")
    if router_factory is not None:
        provider_router = router_factory(fallback_factory)
    elif router is not None:
        provider_router = router
    else:
        provider_router = _default_workspace_router(config, fallback_factory)
    request = WorkspaceRequest(
        TaskKind.COMPILE,
        prompt,
        stage.root,
        config.job_timeout_seconds,
        allowed_paths=allowed,
    )
    usage_recorded = False

    def record_usage(result) -> None:
        nonlocal usage_recorded
        if usage_recorded or result is None:
            return
        try:
            record_routed_usage(home, result, source_agent="system")
        except (OSError, ValueError) as exc:
            logger.warning("could not append compile usage: %s", exc)
        usage_recorded = True

    try:
        _record_automatic_phase(home, status_run_id, "provider_started")
        result = await provider_router.edit_workspace(request)
        if result.outcome != "success":
            record_usage(result)
            for candidate in [*fallback_holder, stage]:
                if candidate.root.exists():
                    discard_stage(candidate)
            return 0.0
        selected = fallback_holder[-1] if result.provider == "claude" and fallback_holder else stage
        try:
            _record_automatic_phase(home, status_run_id, "validation_started")
            validated = validate_stage(selected, allowed_paths=allowed, task=TaskKind.COMPILE)
        except StageValidationError as validation_error:
            if result.provider != "codex" or (router is not None and router_factory is None):
                result = routed_invalid_output(result, validation_error)
                record_usage(result)
                discard_stage(selected)
                return 0.0
            failed = routed_invalid_output(result, validation_error).attempts[-1]
            result = await provider_router.edit_workspace(request, codex_attempt=failed)
            if result.outcome != "success" or not fallback_holder:
                record_usage(result)
                for candidate in [*fallback_holder, stage]:
                    if candidate.root.exists():
                        discard_stage(candidate)
                return 0.0
            selected = fallback_holder[-1]
            try:
                _record_automatic_phase(home, status_run_id, "validation_started")
                validated = validate_stage(
                    selected, allowed_paths=allowed, task=TaskKind.COMPILE
                )
            except StageValidationError as fallback_validation_error:
                result = routed_invalid_output(result, fallback_validation_error)
                raise
        record_usage(result)
    except Exception:
        record_usage(locals().get("result"))
        logger.exception("compile provider failed for %s", log_path.name)
        for candidate in [*fallback_holder, stage]:
            if candidate.root.exists():
                discard_stage(candidate)
        return 0.0

    prior = state.setdefault("ingested", {}).get(log_path.name, {})
    state["ingested"][log_path.name] = {
        **prior,
        "hash": "pending-transaction",
        "compiled_at": timestamp,
    }
    bookkeeping = ApplyBookkeeping(
        compiled_marker_path=log_path.resolve().relative_to(home).as_posix(),
        compiled_marker_baseline=log_baseline,
        compiled_at=timestamp,
        state=state,
        state_baseline=state_baseline,
    )
    try:
        _record_automatic_phase(
            home,
            status_run_id,
            "apply_started",
            details={"changed_files": len(validated.changed_paths)},
        )
        apply_validated_stage(validated, validated.before, bookkeeping)
    except RetryableApplyError as exc:
        logger.warning("compile apply deferred for %s: %s", log_path.name, exc)
        return 0.0
    finally:
        for candidate in [*fallback_holder, stage]:
            if candidate.root.exists():
                discard_stage(candidate)
    return 0.0


def _run_compile(args: argparse.Namespace) -> int:
    if not args.dry_run:
        recover_incomplete_apply(ROOT_DIR)
    state, _ = load_state_with_baseline()
    if args.file:
        candidate = Path(args.file)
        target = candidate if candidate.is_absolute() else DAILY_DIR / candidate.name
        if not target.exists():
            print(f"Error: {args.file} not found")
            return 1
        logs = [target]
    else:
        logs = list_raw_files()
        if not args.all:
            logs = [
                path for path in logs
                if state.get("ingested", {}).get(path.name, {}).get("hash") != file_hash(path)
            ]
    if args.dry_run:
        for path in logs:
            print(path.name)
        return 0
    for path in logs:
        current_state, baseline = load_state_with_baseline()
        asyncio.run(compile_daily_log(path, current_state, baseline))
    print(f"Knowledge base: {len(list_wiki_articles())} articles")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile daily logs into knowledge articles")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    auto_lock = None
    if os.environ.get("AI_MEMORY_AUTO_COMPILE") == "1":
        config = load_config(os.environ)
        auto_lock = ExclusiveFileLock(
            config.root_dir / "scripts" / "memory-auto-compile.lock",
            blocking=False,
        )
        if not auto_lock.acquire():
            logger.info("Skipping overlapping automatic compile")
            return 75
    try:
        return _run_compile(args)
    finally:
        if auto_lock is not None:
            auto_lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
