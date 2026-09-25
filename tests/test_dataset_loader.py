from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.dataset_loader import REQUIRED_COLUMNS, DatasetLoader, to_download_url


def test_blob_url_is_converted_to_resolve_url() -> None:
    blob = "https://huggingface.co/datasets/allenai/WildChat/blob/main/data/train-00003-of-00006.parquet"
    assert to_download_url(blob) == blob.replace("/blob/", "/resolve/")


def test_non_hf_url_unchanged() -> None:
    assert to_download_url("https://example.com/blob/x.parquet") == "https://example.com/blob/x.parquet"


def test_loads_only_required_columns(make_parquet, sample_rows) -> None:
    loader = DatasetLoader(url="", local_path=str(make_parquet(sample_rows)))
    records = list(loader.iter_records())
    assert len(records) == 4
    assert set(records[0]) == set(REQUIRED_COLUMNS)
    assert isinstance(records[0]["conversation"], list)


def test_max_records_limits_rows(make_parquet, sample_rows) -> None:
    loader = DatasetLoader(url="", local_path=str(make_parquet(sample_rows)), read_batch_size=1)
    assert len(list(loader.iter_records(max_records=2))) == 2
    assert loader.count_records() == 4


def test_missing_required_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"conversation_id": ["a"], "model": ["m"]}), path)
    with pytest.raises(ValueError, match="missing required columns"):
        list(DatasetLoader(url="", local_path=str(path)).iter_records())


def test_missing_local_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DatasetLoader(url="", local_path=str(tmp_path / "nope.parquet")).resolve_path()


def test_cached_download_is_reused(tmp_path: Path, make_parquet, sample_rows) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    source = make_parquet(sample_rows)
    (cache / "train.parquet").write_bytes(source.read_bytes())
    loader = DatasetLoader(url="https://example.com/data/train.parquet", cache_dir=str(cache))
    assert loader.resolve_path() == cache / "train.parquet"
