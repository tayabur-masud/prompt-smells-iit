from __future__ import annotations

from pathlib import Path

import pytest

from app.config import load_config

ENV_VARS = ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS",
            "LLM_TIMEOUT", "LLM_ORGANIZATION", "LLM_PROJECT", "LLM_JSON_MODE"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_env_file_overrides_yaml_and_yaml_api_key_is_ignored(tmp_path: Path) -> None:
    yaml_file = tmp_path / "c.yaml"
    yaml_file.write_text(
        "llm:\n  api_key: ${LLM_API_KEY}\n  model: yaml-model\nprocessing:\n  concurrency: 9\n",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API_KEY=from-env\nLLM_MODEL=env-model\nLLM_BASE_URL=https://x.test/v1\n"
        "LLM_MAX_TOKENS=321\nLLM_ORGANIZATION=\n",
        encoding="utf-8",
    )
    config = load_config(str(yaml_file), str(env_file))
    assert config.llm.api_key.get_secret_value() == "from-env"
    assert config.llm.model == "env-model" and config.llm.base_url == "https://x.test/v1"
    assert config.llm.max_tokens == 321 and config.llm.organization is None
    assert config.processing.concurrency == 9


def test_missing_key_detected(tmp_path: Path) -> None:
    config = load_config(None, str(tmp_path / "missing.env"))
    assert not config.llm.has_api_key
    assert config.processing.max_prompts is None and config.dataset.max_records is None
