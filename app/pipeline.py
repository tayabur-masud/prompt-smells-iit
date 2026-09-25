"""End-to-end orchestration: dataset -> prompts -> LLM analysis -> JSON output."""

from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Iterable, Optional

from tqdm import tqdm

from app.config import Config
from app.conversation_parser import ConversationParser
from app.dataset_loader import DatasetLoader
from app.llm_client import LLMError
from app.models import AnalysisResult, AnalysisStats, ExtractedPrompt, ExtractionStats
from app.output_writer import (
    CheckpointStore,
    FailureLog,
    OutputTotals,
    OutputWriter,
    build_output_records,
    validate_output_file,
)
from app.prompt_analyzer import PromptAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class RunSummary:
    dataset_rows_total: int = 0
    extraction: ExtractionStats = field(default_factory=ExtractionStats)
    prompts_selected: int = 0
    duplicate_prompts: int = 0
    analysis: AnalysisStats = field(default_factory=AnalysisStats)
    output: OutputTotals = field(default_factory=OutputTotals)
    output_file: str = ""
    output_valid: bool = False
    output_error: Optional[str] = None
    interrupted: bool = False
    aborted_reason: Optional[str] = None
    dry_run: bool = False
    limited: bool = False


def extract_prompts(
    loader: DatasetLoader,
    parser: ConversationParser,
    max_records: Optional[int],
    max_prompts: Optional[int],
    total_rows: int,
) -> tuple[list[ExtractedPrompt], int]:
    """Returns unique prompts (in dataset order) and the number of duplicates dropped."""
    records = loader.iter_records(max_records=max_records)
    bar_total = min(total_rows, max_records) if max_records else total_rows
    progress = tqdm(records, total=bar_total, desc="Reading dataset", unit="rec")
    prompts: list[ExtractedPrompt] = []
    seen: set[str] = set()
    duplicates = 0
    try:
        for prompt in parser.extract(progress, max_prompts=None):
            if prompt.prompt_id in seen:
                duplicates += 1
                continue
            seen.add(prompt.prompt_id)
            prompts.append(prompt)
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    finally:
        progress.close()
    return prompts, duplicates


class Pipeline:
    def __init__(
        self,
        config: Config,
        analyzer: PromptAnalyzer,
        loader: Optional[DatasetLoader] = None,
    ) -> None:
        self.config = config
        self.analyzer = analyzer
        self.loader = loader or DatasetLoader(
            url=config.dataset.url,
            local_path=config.dataset.local_path,
            cache_dir=config.dataset.cache_dir,
            read_batch_size=config.dataset.read_batch_size,
        )
        processing = config.processing
        self.checkpoint = CheckpointStore(processing.checkpoint_file)
        self.failures = FailureLog(processing.failures_file)
        self.writer = OutputWriter(processing.output_file)
        self._fatal_error: Optional[str] = None

    def run(self, dry_run: bool = False) -> RunSummary:
        processing = self.config.processing
        summary = RunSummary(
            output_file=processing.output_file,
            dry_run=dry_run,
            limited=bool(
                processing.max_prompts
                or self.config.dataset.max_records
                or self.config.dataset.sample_rate
            ),
        )

        logger.info("Loading dataset...")
        summary.dataset_rows_total = self.loader.count_records()
        logger.info("Dataset loaded: %d records available", summary.dataset_rows_total)

        logger.info("Extracting English user prompts...")
        parser = ConversationParser(sample_rate=self.config.dataset.sample_rate)
        prompts, summary.duplicate_prompts = extract_prompts(
            self.loader,
            parser,
            self.config.dataset.max_records,
            processing.max_prompts,
            summary.dataset_rows_total,
        )
        summary.extraction = parser.stats
        summary.prompts_selected = len(prompts)
        logger.info("Extracted English user prompts: %d", len(prompts))

        if dry_run:
            return summary

        if processing.resume:
            self.checkpoint.ensure_trailing_newline()
            done = self.checkpoint.load_processed_ids()
            logger.info("Resume: %d prompts already processed in checkpoint", len(done))
        else:
            self.checkpoint.reset()
            self.failures.reset()
            done = set()

        pending = [p for p in prompts if p.prompt_id not in done]
        summary.analysis.already_processed = len(prompts) - len(pending)

        logger.info("Processing prompts... (%d to analyze)", len(pending))
        try:
            self._analyze_all(pending, summary.analysis)
        except KeyboardInterrupt:
            summary.interrupted = True
            logger.warning("Interrupted. Completed work is checkpointed; rerun with --resume.")
        summary.aborted_reason = self._fatal_error

        summary.output = self.writer.write_from_checkpoint(self.checkpoint)
        summary.output_valid, _, summary.output_error = validate_output_file(processing.output_file)
        logger.info("Results written to %s", processing.output_file)
        return summary

    def _analyze_one(self, prompt: ExtractedPrompt) -> AnalysisResult:
        return self.analyzer.analyze(prompt.content, prompt_id=prompt.prompt_id)

    def _analyze_all(self, prompts: Iterable[ExtractedPrompt], stats: AnalysisStats) -> None:
        prompt_list = list(prompts)
        concurrency = self.config.processing.concurrency
        max_in_flight = concurrency * 4
        iterator = iter(prompt_list)
        in_flight: dict[Future[AnalysisResult], ExtractedPrompt] = {}
        progress = tqdm(total=len(prompt_list), desc="Analyzing prompts", unit="prompt")

        executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="llm")
        try:
            while True:
                while len(in_flight) < max_in_flight and self._fatal_error is None:
                    prompt = next(iterator, None)
                    if prompt is None:
                        break
                    in_flight[executor.submit(self._analyze_one, prompt)] = prompt
                    stats.sent_to_llm += 1
                if not in_flight:
                    break
                finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in finished:
                    prompt = in_flight.pop(future)
                    self._handle_result(prompt, future, stats)
                    progress.update(1)
                    progress.set_postfix(failed=stats.failed, refresh=False)
                if self._fatal_error is not None:
                    for future in list(in_flight):
                        if future.cancel():
                            in_flight.pop(future)
                            stats.sent_to_llm -= 1
        except KeyboardInterrupt:
            for future in in_flight:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            for future, prompt in in_flight.items():
                if future.done() and not future.cancelled():
                    self._handle_result(prompt, future, stats)
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            progress.close()

    def _handle_result(
        self, prompt: ExtractedPrompt, future: Future[AnalysisResult], stats: AnalysisStats
    ) -> None:
        try:
            result = future.result()
        except Exception as exc:  # isolate per-prompt failures
            stats.failed += 1
            stats.failed_ids.append(prompt.prompt_id)
            self.failures.append(prompt, exc)
            logger.error(
                "Failed to analyze prompt: %s (%s)", prompt.prompt_id[:12], type(exc).__name__
            )
            if isinstance(exc, LLMError) and exc.fatal and self._fatal_error is None:
                self._fatal_error = str(exc)[:300]
                logger.error("Stopping: the LLM provider rejected requests permanently: %s", self._fatal_error)
            return
        self.checkpoint.append(prompt.prompt_id, build_output_records(prompt, result))
        stats.analyzed += 1
        if result.smells:
            stats.with_smells += 1
        else:
            stats.without_smells += 1
