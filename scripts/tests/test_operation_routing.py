from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import compile as compile_module
import connections
import flush
import lint
import query
from providers import ProviderResult, RoutedResult, TaskKind, TextRequest, WorkspaceRequest
from staging import RetryableApplyError
from utils import capture_file_baseline
from worker import MemoryWorker


@pytest.fixture
def batch_flush_module():
    path = Path(__file__).resolve().parents[1] / "batch-flush.py"
    spec = importlib.util.spec_from_file_location("task9_batch_flush", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _routed(task: TaskKind, *, provider: str = "codex", text: str = "saved") -> RoutedResult:
    model = "gpt-5.6-luna" if task in {TaskKind.EXTRACT, TaskKind.SEMANTIC_LINT} else "gpt-5.6-terra"
    attempt = ProviderResult(provider=provider, model=model, task=task, outcome="success", text=text)
    return RoutedResult.from_result(attempt, (attempt,), None)


class RecordingTextRouter:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[TextRequest] = []

    async def generate_text(self, request: TextRequest) -> RoutedResult:
        self.requests.append(request)
        return _routed(request.task, text=self.response)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def memory_home(tmp_path: Path) -> Path:
    home = tmp_path / "memory"
    _write(home / "AGENTS.md", "# Schema\n")
    _write(home / "daily/2026-08-11.md", "# Daily\n\n**Agent:** Codex\n\nA durable decision.\n")
    _write(
        home / "knowledge/index.md",
        "# Knowledge Base Index\n\n"
        "| Article | Project | Summary | Compiled From | Updated |\n"
        "|---|---|---|---|---|\n",
    )
    _write(home / "knowledge/log.md", "# Build Log\n")
    _write(home / "scripts/state.json", '{"ingested": {}, "query_count": 0, "total_cost": 0.0}\n')
    return home


def test_live_extraction_uses_extract_text_request_and_pure_prompt(memory_home: Path):
    router = RecordingTextRouter("**Context:** durable")

    result = asyncio.run(
        flush.run_flush(
            "User: Keep this decision",
            "memory",
            "/tmp/memory",
            router=router,
            memory_home=memory_home,
        )
    )

    assert result == "**Context:** durable"
    assert len(router.requests) == 1
    request = router.requests[0]
    assert type(request) is TextRequest
    assert request.task is TaskKind.EXTRACT
    assert request.cwd == memory_home.resolve()
    assert request.prompt == flush.build_flush_prompt(
        "User: Keep this decision", "memory", "/tmp/memory"
    )


def test_batch_extraction_uses_extract_text_request_and_pure_prompt(
    memory_home: Path, batch_flush_module
):
    router = RecordingTextRouter("**Context:** historical")

    result = asyncio.run(
        batch_flush_module.flush_chunk(
            "User: Historical decision",
            "Session date: 2026-08-11",
            "memory",
            "/tmp/memory",
            router=router,
            memory_home=memory_home,
        )
    )

    assert result == "**Context:** historical"
    request = router.requests[0]
    assert type(request) is TextRequest
    assert request.task is TaskKind.EXTRACT
    assert request.prompt == batch_flush_module.build_chunk_prompt(
        "User: Historical decision",
        "Session date: 2026-08-11",
        "memory",
        "/tmp/memory",
    )


def test_worker_live_capture_uses_flush_prompt_not_raw_transcript(tmp_path: Path):
    router = RecordingTextRouter("**Context:** worker")
    job = SimpleNamespace(
        id=7,
        kind="capture",
        source_agent="codex",
        session_id="session-7",
        source_hash="hash-7",
        project="memory",
        cwd="/tmp/memory",
        payload={"rendered_context": "User: queue this"},
        attempt_count=1,
    )

    class Queue:
        def renew(self, *_args):
            return True

        def record_attempt(self, *_args):
            return None

        def complete(self, *_args):
            return None

        def get_job(self, _job_id):
            return SimpleNamespace(status="succeeded", lease_owner="worker")

    worker = MemoryWorker(
        Queue(),
        router,
        daily_writer=lambda *_args: None,
        owner="worker",
        lock_path=tmp_path / "worker.lock",
        heartbeat_sleeper=lambda _delay: asyncio.sleep(3600),
    )
    asyncio.run(worker.process(job))

    request = router.requests[0]
    assert request.task is TaskKind.EXTRACT
    assert request.prompt == flush.build_flush_prompt(
        "User: queue this", "memory", "/tmp/memory"
    )


def test_read_only_query_uses_query_text_request(memory_home: Path):
    router = RecordingTextRouter("Use [[concepts/example]].")

    answer = asyncio.run(
        query.run_query(
            "What changed?", router=router, memory_home=memory_home
        )
    )

    assert answer == "Use [[concepts/example]]."
    request = router.requests[0]
    assert type(request) is TextRequest
    assert request.task is TaskKind.QUERY
    assert str(memory_home) not in request.prompt


def test_query_state_cas_retries_without_losing_concurrent_update(memory_home: Path, monkeypatch):
    router = RecordingTextRouter("answer")
    real_apply = query.apply_host_bookkeeping
    calls = 0

    def racing_apply(home, bookkeeping):
        nonlocal calls
        calls += 1
        if calls == 1:
            state_path = home / "scripts/state.json"
            state_path.write_text(
                '{"ingested": {}, "query_count": 7, "concurrent": "kept", "total_cost": 4.0}\n',
                encoding="utf-8",
            )
        return real_apply(home, bookkeeping)

    monkeypatch.setattr(query, "apply_host_bookkeeping", racing_apply)
    assert asyncio.run(query.run_query("race?", router=router, memory_home=memory_home)) == "answer"
    state = __import__("json").loads((memory_home / "scripts/state.json").read_text())
    assert state["query_count"] == 8
    assert state["concurrent"] == "kept"
    assert state["total_cost"] == 4.0
    assert calls == 2


def test_query_state_cas_exhaustion_preserves_root(
    memory_home: Path, monkeypatch, caplog
):
    answer = "PRIVATE_PROVIDER_ANSWER"
    question = "PRIVATE_USER_QUESTION"
    router = RecordingTextRouter(answer)
    before = (memory_home / "scripts/state.json").read_bytes()
    monkeypatch.setattr(
        query,
        "apply_host_bookkeeping",
        lambda *_args: (_ for _ in ()).throw(RetryableApplyError("race")),
    )
    with caplog.at_level(logging.WARNING, logger=query.__name__):
        result = asyncio.run(
            query.run_query(question, router=router, memory_home=memory_home)
        )

    assert result == answer
    assert (memory_home / "scripts/state.json").read_bytes() == before
    assert len(caplog.records) == 1
    warning = caplog.records[0].getMessage()
    assert warning == "query count bookkeeping conflicted after 3 attempts"
    assert question not in warning
    assert answer not in warning


@pytest.mark.parametrize("file_back", [False, True])
def test_query_cli_preserves_legacy_output_envelope(
    monkeypatch, capsys, tmp_path: Path, file_back: bool
):
    monkeypatch.setattr(query, "QA_DIR", tmp_path / "knowledge/qa")
    if file_back:
        _write(query.QA_DIR / "answer.md", "answer")

    async def fake_run(question, *, file_back=False):
        assert question == "What changed?"
        return "Provider answer"

    monkeypatch.setattr(query, "run_query", fake_run)
    argv = ["What changed?"] + (["--file-back"] if file_back else [])

    assert query.main(argv) == 0
    output = capsys.readouterr().out
    assert output.startswith(
        f"Question: What changed?\nFile back: {'yes' if file_back else 'no'}\n"
        + "-" * 60
        + "\nProvider answer\n"
    )
    if file_back:
        assert output.endswith(
            "\n" + "-" * 60 + "\nAnswer filed to knowledge/qa/ (1 Q&A articles total)\n"
        )


def test_default_router_wiring_uses_configured_luna_and_terra(
    memory_home: Path, batch_flush_module, monkeypatch
):
    expected_luna = "configured-luna"
    expected_terra = "configured-terra"
    environment = {
        "AI_MEMORY_HOME": str(memory_home),
        "AI_MEMORY_CODEX_LUNA_MODEL": expected_luna,
        "AI_MEMORY_CODEX_TERRA_MODEL": expected_terra,
    }
    config = flush.load_config(environment)
    assert config.task_models == {
        TaskKind.EXTRACT: expected_luna,
        TaskKind.SEMANTIC_LINT: expected_luna,
        TaskKind.COMPILE: expected_terra,
        TaskKind.QUERY: expected_terra,
        TaskKind.CONNECTIONS: expected_terra,
        TaskKind.FILE_ANSWER: expected_terra,
    }

    captured = []

    class CapturingCodex:
        def __init__(self, *, task_models):
            captured.append(dict(task_models))

    class Claude:
        def __init__(self, **_kwargs):
            pass

    class Router:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setenv("AI_MEMORY_CODEX_LUNA_MODEL", expected_luna)
    monkeypatch.setenv("AI_MEMORY_CODEX_TERRA_MODEL", expected_terra)
    monkeypatch.setenv("AI_MEMORY_HOME", str(memory_home))
    monkeypatch.delenv("CLAUDE_MEMORY_HOME", raising=False)
    for module, factory in (
        (flush, lambda: flush._default_router(memory_home)),
        (batch_flush_module, lambda: batch_flush_module._default_router(config)),
        (query, lambda: query._text_router(config)),
        (query, lambda: query._workspace_router(config, Mock())),
        (compile_module, lambda: compile_module._default_workspace_router(config, Mock())),
        (connections, lambda: connections._default_workspace_router(config, Mock())),
        (lint, lambda: lint._default_router(config)),
    ):
        monkeypatch.setattr(module, "CodexProvider", CapturingCodex)
        monkeypatch.setattr(module, "ClaudeProvider", Claude)
        monkeypatch.setattr(module, "ProviderRouter", Router)
        factory()
    assert captured == [dict(config.task_models)] * 7


def test_semantic_lint_uses_luna_text_request_and_parses_lines(memory_home: Path):
    router = RecordingTextRouter(
        "CONTRADICTION: [concepts/a] vs [concepts/b] - incompatible claims"
    )

    issues = asyncio.run(lint.check_contradictions(router=router, memory_home=memory_home))

    assert issues[0]["check"] == "contradiction"
    assert router.requests[0].task is TaskKind.SEMANTIC_LINT
    assert type(router.requests[0]) is TextRequest


def test_structural_lint_never_constructs_or_calls_provider(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        lint,
        "check_broken_links",
        lambda: [],
    )
    monkeypatch.setattr(lint, "check_orphan_pages", lambda: [])
    monkeypatch.setattr(lint, "check_orphan_sources", lambda: [])
    monkeypatch.setattr(lint, "check_stale_articles", lambda: [])
    monkeypatch.setattr(lint, "check_missing_backlinks", lambda: [])
    monkeypatch.setattr(lint, "check_sparse_articles", lambda: [])
    monkeypatch.setattr(
        lint,
        "check_contradictions",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("provider called")),
    )
    monkeypatch.setattr(lint, "load_state", lambda: {})
    monkeypatch.setattr(lint, "update_state", lambda mutate: mutate({}))
    monkeypatch.setattr(lint, "REPORTS_DIR", tmp_path / "reports")

    assert lint.main(["--structural-only"]) == 0


class EditingRouter:
    def __init__(self) -> None:
        self.requests: list[WorkspaceRequest] = []

    async def edit_workspace(self, request: WorkspaceRequest, **_kwargs) -> RoutedResult:
        self.requests.append(request)
        root = request.cwd
        if request.task is TaskKind.COMPILE:
            with (root / "knowledge/log.md").open("a", encoding="utf-8") as stream:
                stream.write(
                    "\n## [2026-08-11T12:00:00+00:00] compile | 2026-08-11.md\n"
                    "- Articles created: (none)\n"
                )
        elif request.task is TaskKind.CONNECTIONS:
            self._write_connection(root)
        elif request.task is TaskKind.FILE_ANSWER:
            self._write_file_answer(root)
        return _routed(request.task, text="filed answer")

    @staticmethod
    def _write_connection(root: Path) -> None:
        article = root / "knowledge/connections/a-and-b.md"
        _write(
            article,
                "---\ntitle: Connection A and B\nconnects:\n"
                "  - concepts/a\n  - concepts/b\nproject: memory\n"
                "sources:\n  - daily/2026-08-11.md\n"
                "created: 2026-08-11\nupdated: 2026-08-11\n---\n"
                "# Connection A and B\n",
        )
        with (root / "knowledge/index.md").open("a", encoding="utf-8") as stream:
            stream.write(
                "| [[connections/a-and-b]] | memory | Linked | daily/2026-08-11.md | 2026-08-11 |\n"
            )
        with (root / "knowledge/log.md").open("a", encoding="utf-8") as stream:
            stream.write(
                "\n## [2026-08-11T12:00:00+00:00] connections | swanson-pass\n"
                "- Connections created: [[connections/a-and-b]]\n"
            )

    @staticmethod
    def _write_file_answer(root: Path) -> None:
        article = root / "knowledge/qa/what-changed.md"
        _write(
            article,
                "---\ntitle: Q What changed\nquestion: What changed?\n"
                "consulted:\n  - concepts/example\nfiled: 2026-08-11\n---\n"
                "# Q What changed\n\n## Answer\nNothing.\n",
        )
        with (root / "knowledge/index.md").open("a", encoding="utf-8") as stream:
            stream.write(
                "| [[qa/what-changed]] | memory | Answer | daily/2026-08-11.md | 2026-08-11 |\n"
            )
        with (root / "knowledge/log.md").open("a", encoding="utf-8") as stream:
            stream.write(
                "\n## [2026-08-11T12:00:00+00:00] query (filed) | what changed\n"
                "- Filed to: [[qa/what-changed]]\n"
            )


def test_compile_uses_terra_staged_workspace_and_applies_valid_stage(memory_home: Path):
    router = EditingRouter()
    log_path = memory_home / "daily/2026-08-11.md"
    state_path = memory_home / "scripts/state.json"
    state = {"ingested": {}, "query_count": 0, "total_cost": 0.0}

    asyncio.run(
        compile_module.compile_daily_log(
            log_path,
            state,
            capture_file_baseline(state_path),
            router=router,
            memory_home=memory_home,
        )
    )

    request = router.requests[0]
    assert request.task is TaskKind.COMPILE
    assert request.cwd != memory_home
    assert request.cwd.parent == memory_home / "scripts/staging"
    assert request.allowed_paths == (
        "knowledge/concepts/*.md",
        "knowledge/connections/*.md",
        "knowledge/index.md",
        "knowledge/log.md",
    )
    assert "compile | 2026-08-11.md" in (memory_home / "knowledge/log.md").read_text()
    assert "@compiled-through:" in log_path.read_text()


def test_invalid_codex_compile_uses_fresh_clean_claude_stage(
    memory_home: Path, monkeypatch
):
    calls: list[tuple[str, Path, str]] = []

    class InvalidCodex:
        def __init__(self, **_kwargs):
            pass

        async def edit_workspace(self, request):
            calls.append(("codex", request.cwd, (request.cwd / "knowledge/log.md").read_text()))
            _write(request.cwd / "outside.txt", "invalid")
            return ProviderResult(
                "codex", "gpt-5.6-terra", request.task, "success", text="edited"
            )

    class ValidClaude:
        def __init__(self, **_kwargs):
            self._model = "claude-sonnet-5"

        async def edit_workspace(self, request):
            calls.append(("claude", request.cwd, (request.cwd / "knowledge/log.md").read_text()))
            assert not (request.cwd / "outside.txt").exists()
            with (request.cwd / "knowledge/log.md").open("a", encoding="utf-8") as stream:
                stream.write(
                    "\n## [2026-08-11T12:00:00+00:00] compile | 2026-08-11.md\n"
                    "- Articles created: (none)\n"
                )
            return ProviderResult(
                "claude", "claude-sonnet-5", request.task, "success", text="edited"
            )

    monkeypatch.setattr(compile_module, "CodexProvider", InvalidCodex)
    monkeypatch.setattr(compile_module, "ClaudeProvider", ValidClaude)
    log_path = memory_home / "daily/2026-08-11.md"
    state_path = memory_home / "scripts/state.json"

    asyncio.run(
        compile_module.compile_daily_log(
            log_path,
            {"ingested": {}, "query_count": 0, "total_cost": 0.0},
            capture_file_baseline(state_path),
            memory_home=memory_home,
        )
    )

    assert [provider for provider, _path, _baseline in calls] == ["codex", "claude"]
    assert calls[0][1] != calls[1][1]
    assert calls[0][2] == calls[1][2] == "# Build Log\n"
    assert list((memory_home / "scripts/staging").iterdir()) == []
    assert not (memory_home / "outside.txt").exists()
    assert "compile | 2026-08-11.md" in (memory_home / "knowledge/log.md").read_text()


def test_injected_compile_router_factory_handles_invalid_stage_fallback(memory_home: Path):
    calls: list[tuple[str, Path]] = []
    attempts: list[ProviderResult] = []

    class Codex:
        async def edit_workspace(self, request):
            calls.append(("codex", request.cwd))
            _write(request.cwd / "unexpected.txt", "bad")
            return ProviderResult("codex", "gpt-5.6-terra", request.task, "success")

    class Claude:
        async def edit_workspace(self, request):
            calls.append(("claude", request.cwd))
            assert not (request.cwd / "unexpected.txt").exists()
            with (request.cwd / "knowledge/log.md").open("a", encoding="utf-8") as stream:
                stream.write(
                    "\n## [2026-08-11T12:00:00+00:00] compile | 2026-08-11.md\n"
                    "- Articles created: (none)\n"
                )
            return ProviderResult("claude", "claude-sonnet-5", request.task, "success")

    def router_factory(fallback_workspace_factory):
        from providers import ProviderRouter

        return ProviderRouter(
            Codex(), Claude(), attempt_callback=attempts.append,
            fallback_workspace_factory=fallback_workspace_factory,
        )

    log_path = memory_home / "daily/2026-08-11.md"
    asyncio.run(
        compile_module.compile_daily_log(
            log_path,
            {"ingested": {}, "query_count": 0, "total_cost": 0.0},
            capture_file_baseline(memory_home / "scripts/state.json"),
            router_factory=router_factory,
            memory_home=memory_home,
        )
    )
    assert [name for name, _path in calls] == ["codex", "claude"]
    assert calls[0][1] != calls[1][1]
    assert [attempt.outcome for attempt in attempts] == [
        "success",
        "invalid_output",
        "success",
    ]


def test_connections_uses_terra_staged_workspace(memory_home: Path, monkeypatch):
    router = EditingRouter()
    _write(memory_home / "knowledge/concepts/a.md", "---\ntitle: A\nproject: memory\nsources:\n  - daily/2026-08-11.md\ncreated: 2026-08-11\nupdated: 2026-08-11\n---\n# A\n")
    _write(memory_home / "knowledge/concepts/b.md", "---\ntitle: B\nproject: memory\nsources:\n  - daily/2026-08-11.md\ncreated: 2026-08-11\nupdated: 2026-08-11\n---\n# B\n")

    asyncio.run(
        connections.synthesize_connections(
            [connections.Candidate("a", "b", ["bridge"], 1.0)],
            router=router,
            memory_home=memory_home,
        )
    )

    request = router.requests[0]
    assert request.task is TaskKind.CONNECTIONS
    assert request.cwd != memory_home
    assert request.allowed_paths == (
        "knowledge/connections/*.md",
        "knowledge/index.md",
        "knowledge/log.md",
    )
    assert (memory_home / "knowledge/connections/a-and-b.md").exists()


def test_file_back_query_uses_terra_staged_workspace(memory_home: Path):
    router = EditingRouter()

    answer = asyncio.run(
        query.run_query(
            "What changed?",
            file_back=True,
            router=router,
            memory_home=memory_home,
        )
    )

    request = router.requests[0]
    assert answer == "filed answer"
    assert request.task is TaskKind.FILE_ANSWER
    assert request.cwd != memory_home
    assert request.allowed_paths == (
        "knowledge/qa/*.md",
        "knowledge/index.md",
        "knowledge/log.md",
    )
    assert (memory_home / "knowledge/qa/what-changed.md").exists()


@pytest.mark.parametrize("operation", ["connections", "file_answer"])
def test_invalid_codex_specialized_write_uses_fresh_claude_stage(
    memory_home: Path, operation: str
):
    calls: list[tuple[str, Path, str]] = []
    attempts: list[ProviderResult] = []

    class InvalidCodex:
        def __init__(self, **_kwargs):
            pass

        async def edit_workspace(self, request):
            calls.append(("codex", request.cwd, (request.cwd / "knowledge/log.md").read_text()))
            _write(request.cwd / "unexpected.txt", "contaminated")
            return ProviderResult("codex", "gpt-5.6-terra", request.task, "success", text="codex")

    class ValidClaude:
        def __init__(self, **_kwargs):
            self._model = "claude-sonnet-5"

        async def edit_workspace(self, request):
            calls.append(("claude", request.cwd, (request.cwd / "knowledge/log.md").read_text()))
            assert not (request.cwd / "unexpected.txt").exists()
            if request.task is TaskKind.CONNECTIONS:
                EditingRouter._write_connection(request.cwd)
            else:
                EditingRouter._write_file_answer(request.cwd)
            return ProviderResult("claude", "claude-sonnet-5", request.task, "success", text="claude")

    def router_factory(fallback_workspace_factory):
        from providers import ProviderRouter

        return ProviderRouter(
            InvalidCodex(),
            ValidClaude(),
            attempt_callback=attempts.append,
            fallback_workspace_factory=fallback_workspace_factory,
        )
    before = {
        path.relative_to(memory_home): path.read_bytes()
        for path in memory_home.rglob("*")
        if path.is_file()
    }
    if operation == "connections":
        _write(memory_home / "knowledge/concepts/a.md", "---\ntitle: A\nproject: memory\nsources:\n  - daily/2026-08-11.md\ncreated: 2026-08-11\nupdated: 2026-08-11\n---\n# A\n")
        _write(memory_home / "knowledge/concepts/b.md", "---\ntitle: B\nproject: memory\nsources:\n  - daily/2026-08-11.md\ncreated: 2026-08-11\nupdated: 2026-08-11\n---\n# B\n")
        asyncio.run(
            connections.synthesize_connections(
                [connections.Candidate("a", "b", [], 1.0)],
                router_factory=router_factory,
                memory_home=memory_home,
            )
        )
    else:
        asyncio.run(
            query.run_query(
                "What changed?",
                file_back=True,
                router_factory=router_factory,
                memory_home=memory_home,
            )
        )

    assert [item[0] for item in calls] == ["codex", "claude"]
    assert calls[0][1] != calls[1][1]
    assert calls[0][2] == calls[1][2] == "# Build Log\n"
    assert [attempt.outcome for attempt in attempts] == [
        "success",
        "invalid_output",
        "success",
    ]
    assert list((memory_home / "scripts/staging").iterdir()) == []
    assert not (memory_home / "unexpected.txt").exists()
    for relative, contents in before.items():
        if relative in {Path("knowledge/index.md"), Path("knowledge/log.md"), Path("scripts/state.json")}:
            continue
        assert (memory_home / relative).read_bytes() == contents


@pytest.mark.parametrize("operation", ["compile", "connections", "file_answer"])
def test_invalid_claude_fallback_cleans_every_stage_and_can_rerun(
    memory_home: Path, operation: str
):
    class Codex:
        async def edit_workspace(self, request):
            _write(request.cwd / "unexpected.txt", "codex invalid")
            return ProviderResult("codex", "gpt-5.6-terra", request.task, "success")

    class Claude:
        async def edit_workspace(self, request):
            return ProviderResult("claude", "claude-sonnet-5", request.task, "success")

    def router_factory(fallback_workspace_factory):
        from providers import ProviderRouter
        return ProviderRouter(Codex(), Claude(), fallback_workspace_factory=fallback_workspace_factory)

    if operation == "connections":
        for slug in ("a", "b"):
            _write(memory_home / f"knowledge/concepts/{slug}.md", f"---\ntitle: {slug}\nproject: memory\nsources:\n  - daily/2026-08-11.md\ncreated: 2026-08-11\nupdated: 2026-08-11\n---\n# {slug}\n")

    def run_once():
        if operation == "compile":
            return asyncio.run(compile_module.compile_daily_log(
                memory_home / "daily/2026-08-11.md",
                {"ingested": {}, "query_count": 0, "total_cost": 0.0},
                capture_file_baseline(memory_home / "scripts/state.json"),
                router_factory=router_factory,
                memory_home=memory_home,
            ))
        if operation == "connections":
            return asyncio.run(connections.synthesize_connections(
                [connections.Candidate("a", "b", [], 1.0)],
                router_factory=router_factory,
                memory_home=memory_home,
            ))
        return asyncio.run(query.run_query(
            "What changed?", file_back=True, router_factory=router_factory,
            memory_home=memory_home,
        ))

    first = run_once()
    assert list((memory_home / "scripts/staging").iterdir()) == []
    second = run_once()
    assert list((memory_home / "scripts/staging").iterdir()) == []
    if operation == "file_answer":
        assert str(first).startswith("Error querying knowledge base:")
        assert str(second).startswith("Error querying knowledge base:")


def test_compile_fallback_error_discards_actual_claude_stage_and_can_rerun(memory_home: Path):
    calls: list[Path] = []

    class Codex:
        async def edit_workspace(self, request):
            return ProviderResult("codex", "gpt-5.6-terra", request.task, "capacity")

    class Claude:
        async def edit_workspace(self, request):
            calls.append(request.cwd)
            return ProviderResult(
                "claude", "claude-sonnet-5", request.task, "error", reason="failed"
            )

    def router_factory(fallback_workspace_factory):
        from providers import ProviderRouter
        return ProviderRouter(Codex(), Claude(), fallback_workspace_factory=fallback_workspace_factory)

    def run_once():
        return asyncio.run(compile_module.compile_daily_log(
            memory_home / "daily/2026-08-11.md", {"ingested": {}},
            capture_file_baseline(memory_home / "scripts/state.json"),
            router_factory=router_factory, memory_home=memory_home,
        ))

    assert run_once() == 0.0
    assert list((memory_home / "scripts/staging").iterdir()) == []
    assert run_once() == 0.0
    assert list((memory_home / "scripts/staging").iterdir()) == []
    assert len(calls) == 2


def test_compile_apply_conflict_cleans_stage_and_next_run_succeeds(memory_home: Path, monkeypatch):
    router = EditingRouter()
    real_apply = compile_module.apply_validated_stage
    calls = 0

    def conflict_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableApplyError("concurrent baseline")
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(compile_module, "apply_validated_stage", conflict_once)
    log_path = memory_home / "daily/2026-08-11.md"
    before = {p.relative_to(memory_home): p.read_bytes() for p in memory_home.rglob("*") if p.is_file()}
    assert asyncio.run(compile_module.compile_daily_log(
        log_path, {"ingested": {}}, capture_file_baseline(memory_home / "scripts/state.json"),
        router=router, memory_home=memory_home,
    )) == 0.0
    assert list((memory_home / "scripts/staging").iterdir()) == []
    for relative, data in before.items():
        assert (memory_home / relative).read_bytes() == data
    router.requests.clear()
    assert asyncio.run(compile_module.compile_daily_log(
        log_path, {"ingested": {}}, capture_file_baseline(memory_home / "scripts/state.json"),
        router=router, memory_home=memory_home,
    )) == 0.0
    assert "compile | 2026-08-11.md" in (memory_home / "knowledge/log.md").read_text()


@pytest.mark.parametrize("malformed", [False, True])
def test_connections_all_rejected_applies_only_canonical_audit(
    memory_home: Path, malformed: bool
):
    for slug in ("a", "b"):
        _write(memory_home / f"knowledge/concepts/{slug}.md", f"---\ntitle: {slug}\nproject: memory\nsources:\n  - daily/2026-08-11.md\ncreated: 2026-08-11\nupdated: 2026-08-11\n---\n# {slug}\n")

    class RejectingRouter:
        async def edit_workspace(self, request, **_kwargs):
            with (request.cwd / "knowledge/log.md").open("a", encoding="utf-8") as stream:
                stream.write(
                    "\n## malformed audit\n" if malformed else
                    "\n## [2026-08-11T12:00:00+00:00] connections | swanson-pass\n"
                    "- Candidates evaluated: 1\n- Connections created: none\n"
                    "- Rejected (co-occurrence / too weak): concepts/a <-> concepts/b - weak\n"
                )
            return _routed(TaskKind.CONNECTIONS)

    before = (memory_home / "knowledge/log.md").read_bytes()
    asyncio.run(connections.synthesize_connections(
        [connections.Candidate("a", "b", [], 1.0)], router=RejectingRouter(),
        memory_home=memory_home,
    ))
    after = (memory_home / "knowledge/log.md").read_bytes()
    assert (after == before) is malformed
    assert list((memory_home / "scripts/staging").iterdir()) == []
