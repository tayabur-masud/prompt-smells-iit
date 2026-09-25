"""Download (with caching) and stream the WildChat Parquet file."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import pyarrow.parquet as pq
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("conversation_id", "model", "timestamp", "conversation")


def to_download_url(url: str) -> str:
    """Hugging Face '/blob/' URLs point at an HTML page; '/resolve/' serves the raw file."""
    if "huggingface.co" in url and "/blob/" in url:
        return url.replace("/blob/", "/resolve/", 1)
    return url


class DatasetLoader:
    def __init__(
        self,
        url: str,
        local_path: Optional[str] = None,
        cache_dir: str = "data",
        read_batch_size: int = 2000,
        download_timeout: float = 120.0,
    ) -> None:
        self.url = to_download_url(url)
        self.local_path = local_path
        self.cache_dir = Path(cache_dir)
        self.read_batch_size = read_batch_size
        self.download_timeout = download_timeout
        self._resolved: Optional[Path] = None

    def resolve_path(self) -> Path:
        if self._resolved is None:
            self._resolved = self._resolve_path()
        return self._resolved

    def _resolve_path(self) -> Path:
        if self.local_path:
            path = Path(self.local_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            return path

        if not urlparse(self.url).scheme.startswith("http"):
            path = Path(self.url)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            return path

        target = self.cache_dir / Path(urlparse(self.url).path).name
        if target.exists() and target.stat().st_size > 0:
            logger.info("Using cached dataset file: %s", target)
            return target
        self._download(target)
        return target

    def _download(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        logger.info("Downloading dataset from %s", self.url)
        with requests.get(self.url, stream=True, timeout=self.download_timeout) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0)) or None
            with open(partial, "wb") as handle, tqdm(
                total=total, unit="B", unit_scale=True, desc="Downloading"
            ) as bar:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
                    bar.update(len(chunk))
        os.replace(partial, target)
        logger.info("Dataset downloaded to %s", target)

    def count_records(self) -> int:
        return pq.ParquetFile(self.resolve_path()).metadata.num_rows

    def iter_records(self, max_records: Optional[int] = None) -> Iterator[dict[str, Any]]:
        """Yield one dict per dataset row, reading only the needed columns."""
        parquet = pq.ParquetFile(self.resolve_path())
        available = set(parquet.schema_arrow.names)
        missing = [c for c in REQUIRED_COLUMNS if c not in available]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        yielded = 0
        for batch in parquet.iter_batches(
            batch_size=self.read_batch_size, columns=list(REQUIRED_COLUMNS)
        ):
            for record in batch.to_pylist():
                if max_records is not None and yielded >= max_records:
                    return
                yielded += 1
                yield record
