"""Data models shared across the pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Smell(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("type", "reason", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    smells: list[Smell]


class OutputRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    model: str
    timestamp: str
    content: str
    smell_type: Optional[str]
    smell_reason: Optional[str]


@dataclass(frozen=True)
class ExtractedPrompt:
    conversation_id: str
    model: str
    timestamp: str
    content: str

    @property
    def prompt_id(self) -> str:
        return compute_prompt_id(self.conversation_id, self.content)


def compute_prompt_id(conversation_id: str, content: str) -> str:
    return hashlib.sha256(f"{conversation_id}\x1f{content}".encode("utf-8")).hexdigest()


@dataclass
class ExtractionStats:
    records_read: int = 0
    valid_conversations: int = 0
    malformed_records: int = 0
    sampled_out: int = 0
    user_messages: int = 0
    english_user_messages: int = 0


@dataclass
class AnalysisStats:
    already_processed: int = 0
    sent_to_llm: int = 0
    analyzed: int = 0
    failed: int = 0
    with_smells: int = 0
    without_smells: int = 0
    failed_ids: list[str] = field(default_factory=list)
