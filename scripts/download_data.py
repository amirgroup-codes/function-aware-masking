"""CLI entry point: download and preprocess OAS antibody data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.download import download_oas_subset
from data.preprocessing import preprocess_sequences
from training.config import load_config
from utils.io import save_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess OAS data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Max number of OAS data-units to download",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=50_000,
        help="Max total sequences to collect",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = Path("data/raw")

    logger.info("Downloading OAS data (max %d files, %d sequences)...", args.max_files, args.max_sequences)
    raw_records = download_oas_subset(
        output_dir=raw_dir,
        species="human",
        chain="Heavy",
        max_files=args.max_files,
        max_sequences=args.max_sequences,
    )
    logger.info("Downloaded %d raw records", len(raw_records))

    logger.info("Preprocessing sequences...")
    cleaned = preprocess_sequences(
        raw_records,
        min_length=config.data.min_length,
        max_length=config.data.max_length,
        valid_amino_acids=config.data.valid_amino_acids,
    )
    logger.info("After preprocessing: %d sequences", len(cleaned))

    output_path = Path(config.data.processed_path)
    save_jsonl(cleaned, output_path)
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
