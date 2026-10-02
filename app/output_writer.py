"""Checkpointing, failure logging, and final JSON output."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from app.models import AnalysisResult, ExtractedPrompt, OutputRecord

logger = logging.getLogger(__name__)


@dataclass
class OutputTotals:
    records: int = 0
    prompts: int = 0
    prompts_with_smells: int = 0


def build_output_records(prompt: ExtractedPrompt, result: AnalysisResult) -> list[OutputRecord]:
    base = {
        "conversation_id": prompt.conversation_id,
        "model": prompt.model,
        "timestamp": prompt.timestamp,
        "content": prompt.content,
    }
    if not result.smells:
        return [OutputRecord(**base, smell_type=None, smell_reason=None)]
    return [
        OutputRecord(**base, smell_type=smell.type, smell_reason=smell.reason)
        for smell in result.smells
    ]


class CheckpointStore:
    """Append-only JSONL file: one line per successfully analyzed prompt."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def reset(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def ensure_trailing_newline(self) -> None:
        """Keep a line torn by a hard kill from merging with the next appended entry."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with open(self.path, "rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")

    def iter_entries(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # a torn final line from an interrupted write is expected; skip it
                    logger.warning("Ignoring corrupt checkpoint line %d", line_number)
                    continue
                if isinstance(entry, dict) and "prompt_id" in entry and "records" in entry:
                    yield entry

    def load_processed_ids(self) -> set[str]:
        return {entry["prompt_id"] for entry in self.iter_entries()}

    def append(self, prompt_id: str, records: list[OutputRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"prompt_id": prompt_id, "records": [r.model_dump() for r in records]},
            ensure_ascii=False,
        )
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


class FailureLog:
    """Records failed prompt ids and error types; never the prompt text."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def reset(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def append(self, prompt: ExtractedPrompt, error: BaseException) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "prompt_id": prompt.prompt_id,
            "conversation_id": prompt.conversation_id,
            "error_type": type(error).__name__,
            "error": str(error)[:300],
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


class OutputWriter:
    def __init__(self, output_file: str) -> None:
        self.output_file = Path(output_file)

    def write_from_checkpoint(self, checkpoint: CheckpointStore) -> OutputTotals:
        """Stream every checkpointed record into a JSON array, written atomically.

        Streaming produces the same bytes as json.dump(records, indent=2, ensure_ascii=False)
        without holding the full dataset's results in memory.
        """
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.output_file.with_suffix(self.output_file.suffix + ".tmp")
        totals = OutputTotals()
        seen_ids: set[str] = set()
        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write("[")
            for entry in checkpoint.iter_entries():
                if entry["prompt_id"] in seen_ids:
                    continue
                seen_ids.add(entry["prompt_id"])
                totals.prompts += 1
                if any(record.get("smell_type") for record in entry["records"]):
                    totals.prompts_with_smells += 1
                for record in entry["records"]:
                    handle.write(",\n" if totals.records else "\n")
                    body = json.dumps(record, ensure_ascii=False, indent=2)
                    handle.write("  " + body.replace("\n", "\n  "))
                    totals.records += 1
            handle.write("\n]" if totals.records else "]")
        os.replace(temp_path, self.output_file)
        return totals


def validate_output_file(path: str) -> tuple[bool, int, Optional[str]]:
    """Returns (is_valid, record_count, error_message)."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return False, 0, f"Output file not found: {path}"
    except json.JSONDecodeError as exc:
        return False, 0, f"Output is not valid JSON: {exc}"
    if not isinstance(data, list):
        return False, 0, "Output JSON is not an array"
    for index, record in enumerate(data):
        try:
            OutputRecord.model_validate(record)
        except ValidationError as exc:
            return False, len(data), f"Record {index} failed schema validation: {exc.error_count()} error(s)"
    return True, len(data), None


def write_prompts_parquet(prompts: list[ExtractedPrompt], path: str) -> int:
    """Export the selected input prompts (no LLM results) to a Parquet file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "prompt_id": pa.array([p.prompt_id for p in prompts], pa.string()),
            "conversation_id": pa.array([p.conversation_id for p in prompts], pa.string()),
            "model": pa.array([p.model for p in prompts], pa.string()),
            "timestamp": pa.array([p.timestamp for p in prompts], pa.string()),
            "content": pa.array([p.content for p in prompts], pa.string()),
        }
    )
    pq.write_table(table, target)
    return table.num_rows
