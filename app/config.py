"""Configuration: defaults <- config.yaml <- environment/.env <- CLI overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator

DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/allenai/WildChat/resolve/main/data/"
    "train-00003-of-00006.parquet"
)


class DatasetConfig(BaseModel):
    url: str = DEFAULT_DATASET_URL
    local_path: Optional[str] = None
    prompts_file: Optional[str] = None  # pre-filtered prompts Parquet; bypasses conversation parsing
    cache_dir: str = "data"
    max_records: Optional[int] = Field(default=None, ge=1)
    sample_rate: Optional[float] = Field(default=None, gt=0, le=1)
    read_batch_size: int = Field(default=2000, ge=1)


class LLMConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = Field(default=1000, ge=1)
    timeout: float = Field(default=60.0, gt=0)
    organization: Optional[str] = None
    project: Optional[str] = None
    json_mode: bool = True

    @field_validator("organization", "project", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key.get_secret_value().strip())


class ProcessingConfig(BaseModel):
    max_prompts: Optional[int] = Field(default=None, ge=1)
    concurrency: int = Field(default=5, ge=1)
    retry_attempts: int = Field(default=3, ge=1)
    requests_per_minute: Optional[int] = Field(default=None, ge=1)
    max_prompt_chars: int = Field(default=12000, ge=100)
    output_file: str = "output/prompt_smells.json"
    checkpoint_file: str = "output/checkpoint.jsonl"
    failures_file: str = "output/failures.jsonl"
    resume: bool = False


class Config(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)


_ENV_TO_LLM_FIELD = {
    "LLM_API_KEY": "api_key",
    "LLM_BASE_URL": "base_url",
    "LLM_MODEL": "model",
    "LLM_TEMPERATURE": "temperature",
    "LLM_MAX_TOKENS": "max_tokens",
    "LLM_TIMEOUT": "timeout",
    "LLM_ORGANIZATION": "organization",
    "LLM_PROJECT": "project",
    "LLM_JSON_MODE": "json_mode",
}


def load_config(
    yaml_path: Optional[str] = "config.yaml",
    env_file: Optional[str] = ".env",
) -> Config:
    if env_file and Path(env_file).exists():
        load_dotenv(env_file, override=False)

    data: dict[str, Any] = {}
    if yaml_path and Path(yaml_path).exists():
        with open(yaml_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

    llm_data: dict[str, Any] = dict(data.get("llm") or {})
    llm_data.pop("api_key", None)  # secrets come from the environment only
    for env_name, field_name in _ENV_TO_LLM_FIELD.items():
        value = os.getenv(env_name)
        if value is not None and (value.strip() or field_name in ("organization", "project")):
            llm_data[field_name] = value.strip()

    return Config(
        dataset=DatasetConfig(**(data.get("dataset") or {})),
        llm=LLMConfig(**llm_data),
        processing=ProcessingConfig(**(data.get("processing") or {})),
    )
