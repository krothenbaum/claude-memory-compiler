"""Index-guided provider-neutral knowledge-base queries."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from config import QA_DIR, load_config, now_iso
from providers import (
    ClaudeProvider,
    CodexProvider,
    ProviderResult,
    ProviderRouter,
    RoutedResult,
    TaskKind,
    TextRequest,
    WorkspaceRequest,
)
from staging import (
    ApplyBookkeeping,
    RetryableApplyError,
    StageValidationError,
    apply_validated_stage,
    apply_host_bookkeeping,
    create_fallback_stage,
    create_stage,
    discard_stage,
    validate_stage,
)
from utils import FileBaseline, read_text_with_baseline
from usage import record_routed_usage

ROOT_DIR = Path(__file__).resolve().parent.parent
_STATE_UPDATE_ATTEMPTS = 3


def _state_with_baseline(home: Path) -> tuple[dict, FileBaseline]:
    """Parse state from the same bytes represented by its CAS baseline."""
    path = home / "scripts/state.json"
    try:
        content, baseline = read_text_with_baseline(path)
    except FileNotFoundError:
        return (
            {"ingested": {}, "query_count": 0, "last_lint": None, "total_cost": 0.0},
            FileBaseline(False, 0, None),
        )
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("state.json must contain an object")
    return value, baseline


def _query_state_bookkeeping(home: Path) -> ApplyBookkeeping:
    state, baseline = _state_with_baseline(home)
    state["query_count"] = state.get("query_count", 0) + 1
    return ApplyBookkeeping(state=state, state_baseline=baseline)


def _apply_query_state(home: Path) -> None:
    last_error: RetryableApplyError | None = None
    for _attempt in range(_STATE_UPDATE_ATTEMPTS):
        bookkeeping = _query_state_bookkeeping(home)
        try:
            apply_host_bookkeeping(home, bookkeeping)
            return
        except RetryableApplyError as exc:
            last_error = exc
    raise RetryableApplyError("query state update conflicted after bounded retries") from last_error


def _wiki_content(home: Path) -> str:
    parts: list[str] = []
    index = home / "knowledge/index.md"
    if index.exists():
        parts.append(f"# INDEX\n\n{index.read_text(encoding='utf-8')}")
    for directory in ("concepts", "connections", "qa"):
        for article in sorted((home / "knowledge" / directory).glob("*.md")):
            relative = article.relative_to(home / "knowledge")
            parts.append(f"# ARTICLE: {relative}\n\n{article.read_text(encoding='utf-8')}")
    return "\n\n---\n\n".join(parts)


def build_query_prompt(question: str, wiki_content: str, *, file_back: bool) -> str:
    file_back_instructions = ""
    if file_back:
        timestamp = now_iso()
        file_back_instructions = f"""

## File Back Instructions

After answering, do the following in the staged workspace:
1. Create a Q&A article below knowledge/qa/ with a slugified filename.
2. Use the Q&A article schema with title, question, consulted articles, and filed date.
3. Update knowledge/index.md with a row for the Q&A article.
4. Append a build-log entry headed exactly:
   ## [{timestamp}] query (filed) | question summary
"""
    return f"""You are a knowledge base query engine. Answer the user's question by
consulting the knowledge base below.

## How to Answer

1. Read the INDEX section first.
2. Identify 3-10 relevant articles.
3. Synthesize a clear answer with [[wikilink]] citations.
4. Say honestly when the knowledge base lacks relevant information.

## Knowledge Base

{wiki_content}

## Question

{question}
{file_back_instructions}"""


def _config(home: Path):
    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(home)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    return load_config(environment)


def _text_router(config):
    return ProviderRouter(
        CodexProvider(task_models=config.task_models),
        ClaudeProvider(model=config.claude_model),
    )


def _workspace_router(config, fallback_workspace_factory):
    return ProviderRouter(
        CodexProvider(task_models=config.task_models),
        ClaudeProvider(model=config.claude_model),
        fallback_workspace_factory=fallback_workspace_factory,
    )


def _article_paths(home: Path) -> tuple[str, ...]:
    paths: list[str] = []
    for directory in ("concepts", "connections", "qa"):
        paths.extend(
            path.relative_to(home).as_posix()
            for path in sorted((home / "knowledge" / directory).glob("*.md"))
        )
    return tuple(paths)


async def _file_answer(
    question: str,
    prompt: str,
    home: Path,
    config,
    router: object | None,
    router_factory: object | None,
) -> str:
    stage = create_stage(
        home,
        f"file-answer-{abs(hash(question))}",
        "codex",
        relevant_articles=_article_paths(home),
    )
    allowed = (
        "knowledge/qa/*.md",
        "knowledge/index.md",
        "knowledge/log.md",
    )
    fallback_holder: list[object] = []

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
        provider_router = _workspace_router(config, fallback_factory)
    request = WorkspaceRequest(
        task=TaskKind.FILE_ANSWER,
        prompt=prompt,
        cwd=stage.root,
        timeout_seconds=config.job_timeout_seconds,
        allowed_paths=allowed,
    )
    result = None
    authoritative_result = None
    try:
        result = await provider_router.edit_workspace(request)
        authoritative_result = result
        if result.outcome != "success":
            for candidate in [*fallback_holder, stage]:
                if candidate.root.exists():
                    discard_stage(candidate)
            return f"Error querying knowledge base: {result.reason or result.outcome}"
        selected = fallback_holder[-1] if result.provider == "claude" and fallback_holder else stage
        validated = validate_stage(selected, allowed_paths=allowed, task=TaskKind.FILE_ANSWER)
        _apply_file_answer_with_state(validated, home)
        return result.text
    except StageValidationError as exc:
        # A successful Codex command with an invalid manifest is a failed Codex
        # attempt; ask the router to perform its Claude fallback in a fresh stage.
        if getattr(result, "provider", None) != "codex" or (
            router is not None and router_factory is None
        ):
            if stage.root.exists():
                discard_stage(stage)
            return f"Error querying knowledge base: {exc}"
        failed = ProviderResult(
            provider="codex",
            model=result.model,
            task=TaskKind.FILE_ANSWER,
            outcome="invalid_output",
            reason=str(exc),
        )
        retry = await provider_router.edit_workspace(request, codex_attempt=failed)
        authoritative_result = retry
        if retry.outcome != "success" or not fallback_holder:
            if stage.root.exists():
                discard_stage(stage)
            return f"Error querying knowledge base: {retry.reason or retry.outcome}"
        fallback = fallback_holder[-1]
        try:
            validated = validate_stage(
                fallback, allowed_paths=allowed, task=TaskKind.FILE_ANSWER
            )
        except StageValidationError as fallback_error:
            failed_claude = ProviderResult(
                provider="claude",
                model=retry.model,
                task=TaskKind.FILE_ANSWER,
                outcome="invalid_output",
                input_tokens=retry.input_tokens,
                output_tokens=retry.output_tokens,
                elapsed_ms=retry.elapsed_ms,
                reason=str(fallback_error),
            )
            authoritative_result = RoutedResult.from_result(
                failed_claude,
                (*retry.attempts[:-1], failed_claude),
                retry.fallback_reason,
            )
            return f"Error querying knowledge base: {fallback_error}"
        try:
            _apply_file_answer_with_state(validated, home)
            return retry.text
        except RetryableApplyError as fallback_error:
            return f"Error querying knowledge base: {fallback_error}"
    except Exception as exc:
        for candidate in [*fallback_holder, stage]:
            if candidate.root.exists():
                discard_stage(candidate)
        return f"Error querying knowledge base: {exc}"
    finally:
        if authoritative_result is not None:
            try:
                record_routed_usage(home, authoritative_result, source_agent="system")
            except (OSError, ValueError):
                pass
        for candidate in [*fallback_holder, stage]:
            if candidate.root.exists():
                discard_stage(candidate)


def _apply_file_answer_with_state(validated, home: Path) -> None:
    last_error: RetryableApplyError | None = None
    for _attempt in range(_STATE_UPDATE_ATTEMPTS):
        bookkeeping = _query_state_bookkeeping(home)
        try:
            apply_validated_stage(validated, validated.before, bookkeeping)
            return
        except RetryableApplyError as exc:
            last_error = exc
    raise RetryableApplyError("file-answer state update conflicted after bounded retries") from last_error


async def run_query(
    question: str,
    file_back: bool = False,
    *,
    router: object | None = None,
    router_factory: object | None = None,
    memory_home: Path | str | None = None,
) -> str:
    """Query the knowledge base, optionally applying a validated staged answer."""
    home = Path(memory_home).expanduser().resolve() if memory_home is not None else ROOT_DIR
    config = _config(home)
    prompt = build_query_prompt(question, _wiki_content(home), file_back=file_back)
    if file_back:
        answer = await _file_answer(
            question, prompt, home, config, router, router_factory
        )
    else:
        request = TextRequest(
            task=TaskKind.QUERY,
            prompt=prompt,
            cwd=home,
            timeout_seconds=config.job_timeout_seconds,
        )
        try:
            result = await (router or _text_router(config)).generate_text(request)
            try:
                record_routed_usage(home, result, source_agent="system")
            except (OSError, ValueError):
                pass
            answer = (
                result.text
                if result.outcome == "success"
                else f"Error querying knowledge base: {result.reason or result.outcome}"
            )
        except Exception as exc:
            answer = f"Error querying knowledge base: {exc}"

    if not file_back:
        try:
            _apply_query_state(home)
        except RetryableApplyError as exc:
            return f"Error querying knowledge base: {exc}"
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query the personal knowledge base")
    parser.add_argument("question")
    parser.add_argument("--file-back", action="store_true")
    args = parser.parse_args(argv)
    print(f"Question: {args.question}")
    print(f"File back: {'yes' if args.file_back else 'no'}")
    print("-" * 60)
    print(asyncio.run(run_query(args.question, file_back=args.file_back)))
    if args.file_back:
        print("\n" + "-" * 60)
        qa_count = len(list(QA_DIR.glob("*.md"))) if QA_DIR.exists() else 0
        print(f"Answer filed to knowledge/qa/ ({qa_count} Q&A articles total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
