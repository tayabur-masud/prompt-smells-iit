"""CLI entry point for the WildChat prompt smell detector."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from tqdm.contrib.logging import logging_redirect_tqdm

from app.config import Config, load_config
from app.llm_client import LLMClient
from app.pipeline import Pipeline, RunSummary
from app.prompt_analyzer import PromptAnalyzer

logger = logging.getLogger("prompt_smells")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect prompt smells in English user prompts from the WildChat dataset.",
    )
    parser.add_argument("--config", default="config.yaml", help="YAML config file (default: config.yaml)")
    parser.add_argument("--env-file", default=".env", help="dotenv file with LLM_* settings (default: .env)")
    parser.add_argument("--dataset", help="Dataset URL (Hugging Face blob/resolve URL) or local .parquet path")
    parser.add_argument("--max-prompts", type=int, help="Analyze only the first N English user prompts (default: all)")
    parser.add_argument("--max-records", type=int, help="Read only the first N dataset rows (default: all)")
    parser.add_argument("--sample-rate", type=float, help="Deterministic fraction of conversations to keep, e.g. 0.1")
    parser.add_argument("--output", help="Output JSON file (default: output/prompt_smells.json)")
    parser.add_argument("--concurrency", type=int, help="Number of concurrent LLM requests")
    parser.add_argument("--requests-per-minute", type=int, help="Client-side cap on LLM requests per minute")
    parser.add_argument("--retry-attempts", type=int, help="Attempts per prompt for transient/invalid responses")
    parser.add_argument("--resume", action="store_true", help="Skip prompts already recorded in the checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Parse and count prompts without calling the LLM")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    dataset, processing = config.dataset, config.processing
    if args.dataset:
        if args.dataset.lower().startswith(("http://", "https://")):
            dataset.url, dataset.local_path = args.dataset, None
        else:
            dataset.local_path = args.dataset
    if args.max_prompts is not None:
        processing.max_prompts = args.max_prompts
    if args.max_records is not None:
        dataset.max_records = args.max_records
    if args.sample_rate is not None:
        dataset.sample_rate = args.sample_rate
    if args.output:
        processing.output_file = args.output
    if args.concurrency is not None:
        processing.concurrency = args.concurrency
    if args.requests_per_minute is not None:
        processing.requests_per_minute = args.requests_per_minute
    if args.retry_attempts is not None:
        processing.retry_attempts = args.retry_attempts
    if args.resume:
        processing.resume = True
    # re-validate so CLI values get the same bounds checks as config values
    return Config.model_validate(config.model_dump())


def setup_logging(verbose: bool) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def print_summary(summary: RunSummary) -> None:
    ext, ana = summary.extraction, summary.analysis
    if summary.dry_run:
        run_kind = "Dry run (no LLM calls)"
    elif summary.limited:
        run_kind = "Test/sample run"
    else:
        run_kind = "Complete dataset run"
    if summary.interrupted:
        run_kind += " - INTERRUPTED"

    line = "=" * 60
    rows = [
        ("Run type", run_kind),
        ("Dataset records available", f"{summary.dataset_rows_total:,}"),
        ("Dataset records read", f"{ext.records_read:,}"),
        ("Valid conversations", f"{ext.valid_conversations:,}"),
        ("Malformed/empty records", f"{ext.malformed_records:,}"),
        ("User messages found", f"{ext.user_messages:,}"),
        ("English user messages found", f"{ext.english_user_messages:,}"),
        ("Duplicate prompts skipped", f"{summary.duplicate_prompts:,}"),
        ("Prompts selected", f"{summary.prompts_selected:,}"),
    ]
    if ext.sampled_out:
        rows.append(("Conversations sampled out", f"{ext.sampled_out:,}"))
    if not summary.dry_run:
        rows += [
            ("Skipped (already processed)", f"{ana.already_processed:,}"),
            ("Prompts sent to the LLM", f"{ana.sent_to_llm:,}"),
            ("Successfully analyzed", f"{ana.analyzed:,}"),
            ("API/analysis failures", f"{ana.failed:,}"),
            ("  with smells (this run)", f"{ana.with_smells:,}"),
            ("  without smells (this run)", f"{ana.without_smells:,}"),
            ("Prompts in output file", f"{summary.output.prompts:,}"),
            ("  with smells", f"{summary.output.prompts_with_smells:,}"),
            ("  without smells", f"{summary.output.prompts - summary.output.prompts_with_smells:,}"),
            ("Output records", f"{summary.output.records:,}"),
        ]
    width = max(len(label) for label, _ in rows)
    if run_failed(summary):
        title = "Prompt Smell Detection FAILED"
    elif summary.interrupted:
        title = "Prompt Smell Detection Stopped"
    else:
        title = "Prompt Smell Detection Completed"
    print("\n" + line)
    print(title)
    print(line)
    for label, value in rows:
        print(f"{label:<{width}} : {value}")
    if not summary.dry_run:
        print(f"\nOutput file: {summary.output_file}")
        if summary.output_valid:
            print("Output validation: SUCCESS")
            print(f"Output records: {summary.output.records:,}")
        else:
            print(f"Output validation: FAILED - {summary.output_error}")
        if summary.aborted_reason:
            print("\nRun aborted: the LLM provider rejected requests permanently.")
            print(f"Provider error: {summary.aborted_reason}")
            print("Check LLM_API_KEY, account credits/billing, and LLM_MODEL / LLM_BASE_URL in .env,")
            print("then rerun with --resume.")
        elif ana.failed:
            print("Failed prompt ids are listed in the failures file; rerun with --resume to retry them.")
    print(line)


def run_failed(summary: RunSummary) -> bool:
    if summary.dry_run:
        return False
    no_progress = summary.analysis.sent_to_llm > 0 and summary.analysis.analyzed == 0
    return bool(summary.aborted_reason) or no_progress or not summary.output_valid


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    try:
        config = apply_overrides(load_config(args.config, args.env_file), args)
    except Exception as exc:
        logger.error("Invalid configuration: %s", exc)
        return 2

    if not args.dry_run and not config.llm.has_api_key:
        logger.error(
            "LLM analysis cannot run: LLM_API_KEY is not set. Add it to %s (see .env.example) "
            "or export it, and set LLM_BASE_URL / LLM_MODEL for your provider. "
            "Use --dry-run to parse the dataset without an API key.",
            args.env_file,
        )
        return 2

    logger.info(
        "LLM provider: %s | model: %s | concurrency: %d",
        config.llm.base_url,
        config.llm.model,
        config.processing.concurrency,
    )
    llm_client = LLMClient(
        config.llm,
        retry_attempts=config.processing.retry_attempts,
        requests_per_minute=config.processing.requests_per_minute,
    ) if not args.dry_run else None
    analyzer = PromptAnalyzer(
        llm_client,  # type: ignore[arg-type]
        invalid_response_attempts=config.processing.retry_attempts,
        max_prompt_chars=config.processing.max_prompt_chars,
    )

    try:
        with logging_redirect_tqdm():
            summary = Pipeline(config, analyzer).run(dry_run=args.dry_run)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=args.verbose)
        return 1

    print_summary(summary)
    if run_failed(summary):
        return 1
    return 130 if summary.interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
