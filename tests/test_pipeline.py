from __future__ import annotations

import json
from pathlib import Path

from app.config import Config
from app.models import AnalysisResult, ExtractedPrompt, Smell
from app.output_writer import (
    CheckpointStore,
    OutputWriter,
    build_output_records,
    validate_output_file,
)
from app.pipeline import Pipeline
from app.prompt_analyzer import PromptAnalyzer
from tests.conftest import FakeLLMClient

TWO_SMELLS = json.dumps(
    {
        "smells": [
            {"type": "Vague / Missing Context", "reason": "Nothing says what to fix."},
            {"type": "Ambiguous References", "reason": "'this' refers to nothing."},
        ]
    }
)


def responder(user_prompt: str) -> str:
    if "fix this" in user_prompt:
        return TWO_SMELLS
    if "FAIL" in user_prompt:
        raise RuntimeError("simulated API outage")
    return '{"smells": []}'


def make_config(tmp_path: Path, parquet: Path, **processing) -> Config:
    config = Config()
    config.dataset.local_path = str(parquet)
    config.processing.output_file = str(tmp_path / "out" / "prompt_smells.json")
    config.processing.checkpoint_file = str(tmp_path / "out" / "checkpoint.jsonl")
    config.processing.failures_file = str(tmp_path / "out" / "failures.jsonl")
    config.processing.concurrency = 2
    for key, value in processing.items():
        setattr(config.processing, key, value)
    return config


def run(config: Config, fake: FakeLLMClient):
    return Pipeline(config, PromptAnalyzer(fake, invalid_response_attempts=1)).run()  # type: ignore[arg-type]


def load_output(config: Config) -> list[dict]:
    return json.loads(Path(config.processing.output_file).read_text(encoding="utf-8"))


def test_end_to_end_output(tmp_path, make_parquet, sample_rows) -> None:
    config = make_config(tmp_path, make_parquet(sample_rows))
    fake = FakeLLMClient(responder)
    summary = run(config, fake)

    assert summary.extraction.records_read == 4
    assert summary.extraction.english_user_messages == 3
    assert summary.analysis.sent_to_llm == 3 and summary.analysis.analyzed == 3
    assert summary.analysis.with_smells == 1 and summary.analysis.without_smells == 2
    assert summary.output_valid and summary.output.records == 4
    assert all("Bonjour" not in call and "Sure, what?" not in call for call in fake.calls)

    records = load_output(config)
    smelly = [r for r in records if r["conversation_id"] == "c1"]
    assert [r["smell_type"] for r in smelly] == ["Vague / Missing Context", "Ambiguous References"]
    assert all(r["content"] == "Help me fix this." for r in smelly)
    clean = [r for r in records if r["conversation_id"] == "c3"]
    assert len(clean) == 2 and all(r["smell_type"] is None and r["smell_reason"] is None for r in clean)
    assert set(records[0]) == {"conversation_id", "model", "timestamp", "content", "smell_type", "smell_reason"}
    assert records[0]["timestamp"] == "2023-05-01T12:00:00+00:00"

    raw = Path(config.processing.output_file).read_text(encoding="utf-8")
    assert "¿sí? 日本" in raw  # ensure_ascii=False


def test_failures_are_isolated_and_retried_on_resume(tmp_path, make_parquet, sample_rows) -> None:
    sample_rows[2]["conversation"][0]["content"] = "FAIL please"
    config = make_config(tmp_path, make_parquet(sample_rows))
    summary = run(config, FakeLLMClient(responder))

    assert summary.analysis.failed == 1 and summary.analysis.analyzed == 2
    assert summary.output_valid
    assert all(r["content"] != "FAIL please" for r in load_output(config))
    failure = json.loads(Path(config.processing.failures_file).read_text(encoding="utf-8").strip())
    assert "FAIL please" not in json.dumps(failure)  # prompt text never written to failure log

    config.processing.resume = True
    fake = FakeLLMClient(lambda _: '{"smells": []}')
    resumed = run(config, fake)
    assert resumed.analysis.already_processed == 2
    assert resumed.analysis.sent_to_llm == 1 and len(fake.calls) == 1
    assert resumed.output.prompts == 3


def test_fatal_provider_error_aborts_run(tmp_path, make_parquet) -> None:
    from datetime import datetime, timezone

    from app.llm_client import LLMError
    from tests.conftest import msg

    ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"conversation_id": f"c{i}", "model": "m", "timestamp": ts, "conversation": [msg("user", f"q{i}")]}
        for i in range(50)
    ]
    config = make_config(tmp_path, make_parquet(rows), concurrency=1)

    def no_credits(_: str) -> str:
        raise LLMError("RateLimitError: no credits", fatal=True)

    fake = FakeLLMClient(no_credits)
    summary = run(config, fake)
    assert summary.aborted_reason and "no credits" in summary.aborted_reason
    assert len(fake.calls) < 10  # stopped early instead of failing all 50
    assert summary.analysis.analyzed == 0


def test_resume_does_not_resend_processed_prompts(tmp_path, make_parquet, sample_rows) -> None:
    config = make_config(tmp_path, make_parquet(sample_rows), max_prompts=1)
    run(config, FakeLLMClient(responder))
    assert len(load_output(config)) == 2

    config.processing.max_prompts = None
    config.processing.resume = True
    fake = FakeLLMClient(responder)
    summary = run(config, fake)
    assert summary.analysis.already_processed == 1 and len(fake.calls) == 2
    assert len(load_output(config)) == 4


def test_fresh_run_resets_previous_checkpoint(tmp_path, make_parquet, sample_rows) -> None:
    config = make_config(tmp_path, make_parquet(sample_rows))
    run(config, FakeLLMClient(responder))
    fake = FakeLLMClient(responder)
    run(config, fake)
    assert len(fake.calls) == 3 and len(load_output(config)) == 4


def test_max_prompts_limits_api_calls(tmp_path, make_parquet, sample_rows) -> None:
    config = make_config(tmp_path, make_parquet(sample_rows), max_prompts=2)
    fake = FakeLLMClient(responder)
    summary = run(config, fake)
    assert summary.prompts_selected == 2 and len(fake.calls) == 2


def test_dry_run_makes_no_calls(tmp_path, make_parquet, sample_rows) -> None:
    config = make_config(tmp_path, make_parquet(sample_rows))
    fake = FakeLLMClient(responder)
    summary = Pipeline(config, PromptAnalyzer(fake)).run(dry_run=True)  # type: ignore[arg-type]
    assert summary.prompts_selected == 3 and fake.calls == []
    assert not Path(config.processing.output_file).exists()


def test_checkpoint_tolerates_torn_line(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path / "cp.jsonl"))
    prompt = ExtractedPrompt("c1", "m", "t", "hi")
    store.append(prompt.prompt_id, build_output_records(prompt, AnalysisResult(smells=[])))
    with open(store.path, "a", encoding="utf-8") as handle:
        handle.write('{"prompt_id": "torn", "rec')
    store.ensure_trailing_newline()
    other = ExtractedPrompt("c2", "m", "t", "yo")
    store.append(other.prompt_id, build_output_records(other, AnalysisResult(smells=[])))
    assert store.load_processed_ids() == {prompt.prompt_id, other.prompt_id}


def test_output_writer_one_record_per_smell_and_valid_json(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path / "cp.jsonl"))
    prompt = ExtractedPrompt("c1", "m", "t", "Help me fix this.")
    result = AnalysisResult(smells=[Smell(type="A", reason="r1"), Smell(type="B", reason="r2")])
    store.append(prompt.prompt_id, build_output_records(prompt, result))
    out = tmp_path / "o.json"
    totals = OutputWriter(str(out)).write_from_checkpoint(store)
    assert (totals.records, totals.prompts, totals.prompts_with_smells) == (2, 1, 1)
    assert validate_output_file(str(out)) == (True, 2, None)
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(
        json.dumps([r.model_dump() for r in build_output_records(prompt, result)])
    )


def test_empty_output_is_valid_json(tmp_path) -> None:
    out = tmp_path / "o.json"
    OutputWriter(str(out)).write_from_checkpoint(CheckpointStore(str(tmp_path / "none.jsonl")))
    assert validate_output_file(str(out)) == (True, 0, None)


def test_validation_detects_bad_output(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[{\"conversation_id\": 1}", encoding="utf-8")
    assert validate_output_file(str(bad))[0] is False
    bad.write_text('[{"conversation_id": "x"}]', encoding="utf-8")
    assert validate_output_file(str(bad))[0] is False


def test_export_prompts_parquet_matches_selection(tmp_path, make_parquet, sample_rows) -> None:
    import pyarrow.parquet as pq

    config = make_config(tmp_path, make_parquet(sample_rows), max_prompts=2)
    export = tmp_path / "out" / "prompts.parquet"
    summary = Pipeline(config, PromptAnalyzer(FakeLLMClient(responder))).run(  # type: ignore[arg-type]
        dry_run=True, export_prompts=str(export)
    )
    table = pq.read_table(export).to_pylist()
    assert len(table) == summary.prompts_selected == 2
    assert set(table[0]) == {"prompt_id", "conversation_id", "model", "timestamp", "content"}
    assert [r["content"] for r in table] == ["Help me fix this.", "Write a haiku about rain."]


def test_prompt_file_round_trip_and_resume_across_modes(tmp_path, make_parquet, sample_rows) -> None:
    config = make_config(tmp_path, make_parquet(sample_rows), max_prompts=1)
    export = tmp_path / "data" / "prompts.parquet"
    run(config, FakeLLMClient(responder))  # dataset mode analyzes the first prompt only
    Pipeline(config, PromptAnalyzer(FakeLLMClient(responder))).run(  # type: ignore[arg-type]
        dry_run=True, export_prompts=str(export)
    )
    config.processing.max_prompts = None
    Pipeline(config, PromptAnalyzer(FakeLLMClient(responder))).run(  # type: ignore[arg-type]
        dry_run=True, export_prompts=str(export)
    )

    config.dataset.prompts_file = str(export)
    config.dataset.local_path = str(tmp_path / "raw-dataset-not-needed.parquet")
    config.processing.resume = True
    fake = FakeLLMClient(responder)
    summary = run(config, fake)
    assert summary.prompts_file == str(export)
    assert summary.extraction.records_read == 3 and summary.prompts_selected == 3
    assert summary.analysis.already_processed == 1 and len(fake.calls) == 2
    assert summary.output.prompts == 3 and summary.output_valid


def test_prompt_file_skips_invalid_rows_and_respects_max_prompts(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from app.pipeline import load_prompt_file

    path = tmp_path / "p.parquet"
    pq.write_table(
        pa.table(
            {
                "conversation_id": ["a", "b", None, "c", "a", "d"],
                "model": ["m"] * 6,
                "timestamp": ["t"] * 6,
                "content": ["one", "  ", "x", "three", "one", "four"],
            }
        ),
        path,
    )
    prompts, stats, duplicates = load_prompt_file(str(path), max_prompts=None, sample_rate=None)
    assert [p.content for p in prompts] == ["one", "three", "four"]
    assert stats.malformed_records == 2 and duplicates == 1
    limited, _, _ = load_prompt_file(str(path), max_prompts=2, sample_rate=None)
    assert [p.content for p in limited] == ["one", "three"]


def test_prompt_file_missing_columns_raises(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest

    from app.dataset_loader import iter_prompt_file

    path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"content": ["hi"]}), path)
    with pytest.raises(ValueError, match="missing required columns"):
        list(iter_prompt_file(str(path)))
