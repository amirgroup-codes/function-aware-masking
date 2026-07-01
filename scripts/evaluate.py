#!/usr/bin/env python
"""Zero-shot evaluation for a trained antibody MLM checkpoint.

Computes the two lightweight, dependency-free zero-shot metrics shipped in
this release:

  * MLM accuracy / perplexity under a *uniform reference mask* -- the fair
    comparison setting, where every model is scored on the same masking
    distribution regardless of how it was trained -- with a per-region
    (framework / CDR / CDR3) breakdown when CDR labels are available.
  * Pseudo-log-likelihood (PLL), the standard zero-shot protein-LM fitness
    score, averaged over a sample of sequences.

Example:
    python scripts/evaluate.py \\
        --checkpoint models/checkpoints/demo/final \\
        --data data/sample/oas_vh_demo.jsonl \\
        --device cpu --max-seqs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import RoFormerForMaskedLM

# Make the repo importable when run as `python scripts/evaluate.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import AntibodyDataset
from evaluation.mlm_accuracy import MLMAccuracyEvaluator
from evaluation.pseudo_loglikelihood import compute_pll_batch
from masking import get_strategy
from utils.io import load_jsonl
from utils.tokenizer import load_tokenizer_for_checkpoint

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a trained checkpoint directory (e.g. models/checkpoints/demo/final).",
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to a JSONL eval set (sequences + optional CDR fields).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument(
        "--max-seqs", type=int, default=200,
        help="Cap on #sequences used for PLL (each costs L forward passes).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write the metrics JSON.",
    )
    args = parser.parse_args()

    logger.info("Loading tokenizer + model from %s", args.checkpoint)
    tokenizer = load_tokenizer_for_checkpoint(args.checkpoint)
    model = RoFormerForMaskedLM.from_pretrained(args.checkpoint).to(args.device)
    model.eval()

    logger.info("Loading eval data: %s", args.data)
    dataset = AntibodyDataset(args.data, tokenizer, max_length=args.max_length)

    # Fair zero-shot setting: score every model under a UNIFORM reference mask,
    # decoupled from whatever masking strategy it was trained with.
    uniform = get_strategy("uniform", tokenizer=tokenizer, mask_prob=0.15)
    mlm_eval = MLMAccuracyEvaluator(model, tokenizer, uniform, device=args.device)
    mlm_metrics = mlm_eval.evaluate(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Pseudo-log-likelihood over a sample of raw sequences.
    records = load_jsonl(args.data)[: args.max_seqs]
    sequences = [r["sequence"] for r in records]
    logger.info("Computing PLL over %d sequences", len(sequences))
    pll_results = compute_pll_batch(
        model, tokenizer, sequences, device=args.device,
        batch_size=args.batch_size, show_progress=True,
    )
    pll_vals = [r["pll_normalized"] for r in pll_results]
    pll_mean = sum(pll_vals) / len(pll_vals) if pll_vals else 0.0

    metrics = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "num_eval_sequences_pll": len(sequences),
        **mlm_metrics,
        "pll_normalized_mean": pll_mean,
    }

    print(json.dumps(metrics, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Wrote metrics to %s", out)


if __name__ == "__main__":
    main()
