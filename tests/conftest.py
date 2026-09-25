from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MESSAGE_TYPE = pa.struct(
    [
        ("content", pa.string()),
        ("language", pa.string()),
        ("redacted", pa.bool_()),
        ("role", pa.string()),
        ("toxic", pa.bool_()),
    ]
)


def msg(role: str, content: str, language: str = "English") -> dict[str, Any]:
    return {"content": content, "language": language, "redacted": False, "role": role, "toxic": False}


@pytest.fixture
def make_parquet(tmp_path: Path) -> Callable[[list[dict[str, Any]]], Path]:
    """Write rows with the real WildChat schema (plus an unused extra column)."""

    def _make(rows: list[dict[str, Any]]) -> Path:
        table = pa.table(
            {
                "conversation_id": pa.array([r["conversation_id"] for r in rows], pa.string()),
                "model": pa.array([r["model"] for r in rows], pa.string()),
                "timestamp": pa.array(
                    [r["timestamp"] for r in rows], pa.timestamp("ms", tz="UTC")
                ),
                "conversation": pa.array(
                    [r["conversation"] for r in rows], pa.list_(MESSAGE_TYPE)
                ),
                "turn": pa.array([1] * len(rows), pa.int64()),
            }
        )
        path = tmp_path / "sample.parquet"
        pq.write_table(table, path)
        return path

    return _make


@pytest.fixture
def sample_rows() -> list[dict[str, Any]]:
    ts = datetime(2023, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    return [
        {
            "conversation_id": "c1",
            "model": "gpt-3.5-turbo",
            "timestamp": ts,
            "conversation": [msg("user", "Help me fix this."), msg("assistant", "Sure, what?")],
        },
        {
            "conversation_id": "c2",
            "model": "gpt-4",
            "timestamp": ts,
            "conversation": [msg("user", "Bonjour", "French"), msg("assistant", "Salut", "French")],
        },
        {
            "conversation_id": "c3",
            "model": "gpt-4",
            "timestamp": ts,
            "conversation": [
                msg("user", "Write a haiku about rain."),
                msg("assistant", "Rain..."),
                msg("user", "Now translate it to Spanish — ¿sí? 日本"),
            ],
        },
        {"conversation_id": "c4", "model": "gpt-4", "timestamp": ts, "conversation": None},
    ]


class FakeLLMClient:
    """Stands in for LLMClient; maps prompt substrings to canned responses."""

    def __init__(self, responder: Callable[[str], str]) -> None:
        self.responder = responder
        self.calls: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(user_prompt)
        return self.responder(user_prompt)
