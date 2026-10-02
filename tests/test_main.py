from __future__ import annotations

from pathlib import Path

from app.config import Config
from main import resolve_export_path


def test_export_path_defaults_to_dataset_folder() -> None:
    config = Config()
    config.processing.max_prompts = 1000
    assert resolve_export_path(config, None) is None
    assert resolve_export_path(config, "x/y.parquet") == "x/y.parquet"
    assert Path(resolve_export_path(config, "")) == Path("data") / "prompts_1000.parquet"
    config.processing.max_prompts = None
    assert Path(resolve_export_path(config, "")) == Path("data") / "prompts_all.parquet"


def test_prompts_and_from_dataset_flags() -> None:
    from main import apply_overrides, parse_args

    config = Config()
    config.dataset.prompts_file = "data/prompts_1000.parquet"
    assert apply_overrides(config.model_copy(deep=True), parse_args([])).dataset.prompts_file == (
        "data/prompts_1000.parquet"
    )
    assert apply_overrides(config.model_copy(deep=True), parse_args(["--prompts", "x.parquet"])).dataset.prompts_file == (
        "x.parquet"
    )
    assert apply_overrides(config.model_copy(deep=True), parse_args(["--from-dataset"])).dataset.prompts_file is None


def test_output_dir_moves_all_run_files() -> None:
    from main import apply_overrides, parse_args

    processing = apply_overrides(Config(), parse_args(["--output-dir", "output/run2"])).processing
    assert Path(processing.output_file) == Path("output/run2/prompt_smells.json")
    assert Path(processing.checkpoint_file) == Path("output/run2/checkpoint.jsonl")
    assert Path(processing.failures_file) == Path("output/run2/failures.jsonl")
