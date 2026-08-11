"""Tests for canonical memory configuration parsing."""

import importlib
import os
import sys
import warnings
from pathlib import Path

import pytest

import config as config_module
from config import load_config
from providers import TaskKind


_CONFIG_VARIABLES = (
    "AI_MEMORY_HOME",
    "CLAUDE_MEMORY_HOME",
    "AI_MEMORY_PROVIDER_ORDER",
    "AI_MEMORY_CODEX_LUNA_MODEL",
    "AI_MEMORY_CODEX_TERRA_MODEL",
    "AI_MEMORY_CLAUDE_MODEL",
    "AI_MEMORY_JOB_TIMEOUT_SECONDS",
    "AI_MEMORY_INTERNAL_JOB",
    "AI_MEMORY_QUEUE_PATH",
    "AI_MEMORY_WORKER_CONCURRENCY",
    "AI_MEMORY_USAGE_ESTIMATE_ONLY",
)


def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CONFIG_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_ai_memory_home_wins(monkeypatch, tmp_path):
    _clean_environment(monkeypatch)
    home = tmp_path / "canonical"
    monkeypatch.setenv("AI_MEMORY_HOME", str(home))

    config = load_config(os.environ)

    assert config.root_dir == home.resolve()


def test_claude_memory_home_is_compatibility_alias(monkeypatch, tmp_path):
    _clean_environment(monkeypatch)
    home = tmp_path / "compatibility"
    monkeypatch.setenv("CLAUDE_MEMORY_HOME", str(home))
    monkeypatch.setattr(
        sys, config_module._COMPATIBILITY_WARNING_STATE, False, raising=False
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = load_config(os.environ)
        load_config(os.environ)

    assert config.root_dir == home.resolve()
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)


def test_compatibility_warning_is_shared_across_import_styles(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    package_config = importlib.import_module("scripts.config")
    monkeypatch.setattr(
        sys, config_module._COMPATIBILITY_WARNING_STATE, False, raising=False
    )
    env = {"CLAUDE_MEMORY_HOME": str(tmp_path)}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config_module.load_config(env)
        package_config.load_config(env)

    assert len(caught) == 1


def test_conflicting_home_variables_fail(monkeypatch, tmp_path):
    _clean_environment(monkeypatch)
    monkeypatch.setenv("AI_MEMORY_HOME", str(tmp_path / "canonical"))
    monkeypatch.setenv("CLAUDE_MEMORY_HOME", str(tmp_path / "compatibility"))

    with pytest.raises(ValueError, match="AI_MEMORY_HOME.*CLAUDE_MEMORY_HOME"):
        load_config(os.environ)


def test_task_models_use_luna_and_terra():
    assert load_config({}).task_models == {
        TaskKind.EXTRACT: "gpt-5.6-luna",
        TaskKind.SEMANTIC_LINT: "gpt-5.6-luna",
        TaskKind.COMPILE: "gpt-5.6-terra",
        TaskKind.QUERY: "gpt-5.6-terra",
        TaskKind.CONNECTIONS: "gpt-5.6-terra",
        TaskKind.FILE_ANSWER: "gpt-5.6-terra",
    }


def test_task_models_follow_model_configuration(monkeypatch):
    _clean_environment(monkeypatch)
    monkeypatch.setenv("AI_MEMORY_CODEX_LUNA_MODEL", "luna-override")
    monkeypatch.setenv("AI_MEMORY_CODEX_TERRA_MODEL", "terra-override")

    config = load_config(os.environ)

    assert config.task_models[TaskKind.EXTRACT] == "luna-override"
    assert config.task_models[TaskKind.SEMANTIC_LINT] == "luna-override"
    assert config.task_models[TaskKind.COMPILE] == "terra-override"


def test_invalid_provider_order_fails(monkeypatch):
    _clean_environment(monkeypatch)
    monkeypatch.setenv("AI_MEMORY_PROVIDER_ORDER", "claude,codex")

    with pytest.raises(ValueError, match="AI_MEMORY_PROVIDER_ORDER"):
        load_config(os.environ)


def test_queue_path_must_be_absolute(monkeypatch):
    _clean_environment(monkeypatch)
    monkeypatch.setenv("AI_MEMORY_QUEUE_PATH", "scripts/jobs.sqlite3")

    with pytest.raises(ValueError, match="AI_MEMORY_QUEUE_PATH.*absolute"):
        load_config(os.environ)


def test_default_paths_and_settings(monkeypatch):
    _clean_environment(monkeypatch)

    config = load_config(os.environ)

    expected_root = Path(__file__).resolve().parents[2]
    assert config.root_dir == expected_root
    assert config.queue_path == expected_root / "scripts" / "jobs.sqlite3"
    assert config.provider_order == ("codex", "claude")
    assert config.codex_luna_model == "gpt-5.6-luna"
    assert config.codex_terra_model == "gpt-5.6-terra"
    assert config.claude_model == "claude-sonnet-5"
    assert config.job_timeout_seconds == 900
    assert config.internal_job is False
    assert config.worker_concurrency == 2
    assert config.usage_estimate_only is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("AI_MEMORY_JOB_TIMEOUT_SECONDS", "not-an-int"),
        ("AI_MEMORY_WORKER_CONCURRENCY", "0"),
        ("AI_MEMORY_USAGE_ESTIMATE_ONLY", "yes"),
    ],
)
def test_invalid_scalar_settings_fail(monkeypatch, name, value):
    _clean_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        load_config(os.environ)


def test_empty_home_fails(monkeypatch):
    _clean_environment(monkeypatch)
    monkeypatch.setenv("AI_MEMORY_HOME", "   ")

    with pytest.raises(ValueError, match="AI_MEMORY_HOME"):
        load_config(os.environ)
