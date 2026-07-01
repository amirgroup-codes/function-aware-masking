"""CLI entry point: download and preprocess OAS paired antibody data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.download_paired import download_paired_subset
from data.preprocessing_paired import preprocess_paired_sequences
from training.config import load_config
from utils.io import save_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess OAS paired data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multispecific_medium.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="Max number of OAS paired data-units to download",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=500_000,
        help="Max total paired sequences to collect",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = Path("data/raw/paired")

    logger.info(
        "Downloading OAS paired data (max %d files, %d sequences)...",
        args.max_files, args.max_sequences,
    )
    raw_records = download_paired_subset(
        output_dir=raw_dir,
        species="human",
        max_files=args.max_files,
        max_sequences=args.max_sequences,
    )
    logger.info("Downloaded %d raw paired records", len(raw_records))

    logger.info("Preprocessing paired sequences...")
    cleaned = preprocess_paired_sequences(
        raw_records,
        valid_amino_acids=config.data.valid_amino_acids,
    )
    logger.info("After preprocessing: %d paired sequences", len(cleaned))

    output_path = Path(config.data.processed_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(cleaned, output_path)
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
