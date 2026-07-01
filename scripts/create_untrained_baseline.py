"""Instantiate a randomly-initialized medium RoFormer and save it as a checkpoint.

This is the "no pretraining" floor for the masking-strategy comparison: same
architecture and tokenizer as every other experiment, but the weights are
never trained. Probes and fine-tunes on top of this checkpoint isolate how
much downstream signal comes from MLM pretraining versus the architecture and
tokenization alone.

Usage:
    python scripts/create_untrained_baseline.py --config configs/untrained_medium.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.model import build_model
from training.config import load_config
from utils.seed import set_seed
from utils.tokenizer import load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a random-init medium RoFormer checkpoint for the untrained baseline",
    )
    parser.add_argument(
        "--config", type=str, default="configs/untrained_medium.yaml",
        help="Path to the experiment config (defaults to configs/untrained_medium.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if config.model.from_pretrained:
        raise ValueError(
            "config.model.from_pretrained must be false for the untrained baseline"
        )

    set_seed(config.seed)

    logger.info("Building %s RoFormer (random init, seed=%d)",
                config.model.model_size, config.seed)
    model = build_model(
        model_name=config.model.model_name,
        from_pretrained=False,
        model_size=config.model.model_size,
    )

    tokenizer = load_tokenizer(config.model.model_name)

    out_dir = Path(config.training.output_dir) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving checkpoint to %s", out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Done. %d parameters, vocab_size=%d, hidden_size=%d",
                n_params, model.config.vocab_size, model.config.hidden_size)


if __name__ == "__main__":
    main()
