"""CLI entry point: annotate antibody sequences with CDR positions."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.annotate_cdrs import annotate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate sequences with CDR positions")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/oas_vh_tiny.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/oas_vh_tiny_annotated.jsonl",
        help="Output annotated JSONL file",
    )
    parser.add_argument(
        "--anarci-fallback",
        action="store_true",
        help="Use ANARCI for sequences without OAS CDR fields",
    )
    args = parser.parse_args()

    annotate_dataset(args.input, args.output, use_anarci_fallback=args.anarci_fallback)


if __name__ == "__main__":
    main()
