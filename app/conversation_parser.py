"""Turn raw dataset rows into English user prompts."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Iterable, Iterator, Optional

from app.models import ExtractedPrompt, ExtractionStats

logger = logging.getLogger(__name__)

ENGLISH_LABELS = frozenset({"english"})


def normalize_messages(conversation: Any) -> Optional[list[dict[str, Any]]]:
    """Return the conversation as a list of message dicts, or None if malformed."""
    if conversation is None:
        return None
    if isinstance(conversation, str):
        try:
            conversation = json.loads(conversation)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(conversation, dict):
        conversation = conversation.get("messages")
    # pyarrow/pandas deliver list<struct> columns as list or numpy.ndarray
    if conversation is None or isinstance(conversation, (str, bytes)):
        return None
    if not hasattr(conversation, "__iter__"):
        return None
    try:
        items = list(conversation)
    except TypeError:
        return None
    return [item for item in items if isinstance(item, dict)]


def format_timestamp(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def is_in_sample(conversation_id: str, sample_rate: Optional[float]) -> bool:
    """Deterministic, conversation-level sampling so resumed runs see the same sample."""
    if sample_rate is None or sample_rate >= 1:
        return True
    bucket = int(hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:8], 16)
    return bucket / 0xFFFFFFFF < sample_rate


class ConversationParser:
    def __init__(self, sample_rate: Optional[float] = None) -> None:
        self.sample_rate = sample_rate
        self.stats = ExtractionStats()

    def extract_from_record(self, record: dict[str, Any]) -> list[ExtractedPrompt]:
        self.stats.records_read += 1
        try:
            conversation_id = record.get("conversation_id")
            if not isinstance(conversation_id, str) or not conversation_id.strip():
                self.stats.malformed_records += 1
                return []
            messages = normalize_messages(record.get("conversation"))
            if not messages:
                self.stats.malformed_records += 1
                return []
            if not is_in_sample(conversation_id, self.sample_rate):
                self.stats.sampled_out += 1
                return []

            self.stats.valid_conversations += 1
            model = str(record.get("model") or "")
            timestamp = format_timestamp(record.get("timestamp"))

            prompts: list[ExtractedPrompt] = []
            for message in messages:
                if str(message.get("role") or "").strip().lower() != "user":
                    continue
                self.stats.user_messages += 1
                language = str(message.get("language") or "").strip().lower()
                content = message.get("content")
                if language not in ENGLISH_LABELS or not isinstance(content, str):
                    continue
                if not content.strip():
                    continue
                self.stats.english_user_messages += 1
                prompts.append(ExtractedPrompt(conversation_id, model, timestamp, content))
            return prompts
        except Exception:  # one bad row must never stop the job
            self.stats.malformed_records += 1
            logger.warning("Skipping malformed record #%d", self.stats.records_read, exc_info=True)
            return []

    def extract(
        self, records: Iterable[dict[str, Any]], max_prompts: Optional[int] = None
    ) -> Iterator[ExtractedPrompt]:
        emitted = 0
        for record in records:
            for prompt in self.extract_from_record(record):
                if max_prompts is not None and emitted >= max_prompts:
                    return
                emitted += 1
                yield prompt
            if max_prompts is not None and emitted >= max_prompts:
                return
