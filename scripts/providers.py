"""Provider-neutral request and result contracts for memory generation."""

import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Literal, Protocol


if __name__ == "providers":
    provider_module = sys.modules[__name__]
    scripts_package = sys.modules.get("scripts")
    if scripts_package is None:
        scripts_package = ModuleType("scripts")
        scripts_package.__path__ = [str(Path(__file__).resolve().parent)]
        scripts_package.__package__ = "scripts"
        sys.modules["scripts"] = scripts_package
    scripts_package.providers = provider_module
    sys.modules.setdefault("scripts.providers", provider_module)
elif __name__ == "scripts.providers":
    sys.modules.setdefault("providers", sys.modules[__name__])


class TaskKind(StrEnum):
    """Supported memory-generation task categories."""

    EXTRACT = "extract"
    COMPILE = "compile"
    QUERY = "query"
    CONNECTIONS = "connections"
    FILE_ANSWER = "file_answer"
    SEMANTIC_LINT = "semantic_lint"


@dataclass(frozen=True)
class TextRequest:
    """A provider-neutral text generation request."""

    task: TaskKind
    prompt: str
    cwd: Path
    timeout_seconds: int
    output_schema: Path | None = None


@dataclass(frozen=True)
class WorkspaceRequest(TextRequest):
    """A generation request that may edit an allowlist of relative paths."""

    allowed_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderResult:
    """Normalized result returned by any generation provider."""

    provider: Literal["codex", "claude"]
    model: str
    task: TaskKind
    outcome: Literal[
        "success", "auth_failed", "capacity", "timeout", "invalid_output", "error"
    ]
    text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_ms: int = 0
    reason: str | None = None


class GenerationProvider(Protocol):
    """Interface implemented by provider-specific generation adapters."""

    async def generate_text(self, request: TextRequest) -> ProviderResult: ...

    async def edit_workspace(self, request: WorkspaceRequest) -> ProviderResult: ...
