"""Predict spatial neighborhoods via IgFold real Cα coordinates.

Replaces the ESM-2 contact-prediction proxy used to train the structure
masking strategy with real predicted 3D coordinates from IgFold, an
antibody-specific structure predictor. Output format is byte-compatible
with predict_structures.py so the training pipeline does not need to
change.

Output .pt format:
    List[dict | None] of length N (same as JSONL record count).
    Each entry is:
        {"knn_indices": Int16Tensor(L, k)} — per-residue k nearest
        neighbors in 3D, computed from real Euclidean Cα-Cα distance.

Usage (single GPU):
    python scripts/predict_structures_igfold.py \\
        --input data/processed/oas_vh_500k.jsonl \\
        --output data/structures/oas_vh_500k_igfold.pt \\
        --k_neighbors 32 --num_models 1

Usage (multi-GPU, see scripts/predict_structures_igfold_launch.sh):
    --start_idx and --end_idx slice the input; spawn one process per GPU.

IgFold is only imported here — training code never depends on it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.io import load_jsonl

logger = logging.getLogger(__name__)


def _patch_torch_load_for_igfold() -> None:
    """IgFold checkpoints predate torch 2.6's weights_only=True default."""
    _orig = torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _patched


def coords_to_knn(ca_coords: torch.Tensor, k: int) -> torch.Tensor:
    """Convert Cα coordinates to per-residue kNN indices.

    Args:
        ca_coords: [L, 3] Cα positions in Å.
        k: number of neighbors per residue.

    Returns:
        [L, k] int16 tensor of neighbor indices (excluding self).
    """
    seq_len = ca_coords.size(0)
    dists = torch.cdist(ca_coords.unsqueeze(0), ca_coords.unsqueeze(0))[0]
    dists.fill_diagonal_(float("inf"))
    actual_k = min(k, seq_len - 1)
    _, indices = dists.topk(actual_k, dim=1, largest=False)
    indices = indices.to(torch.int16)
    if actual_k < k:
        pad = torch.zeros(seq_len, k - actual_k, dtype=torch.int16)
        indices = torch.cat([indices, pad.to(indices.device)], dim=1)
    return indices.cpu()


def predict_one(
    sequence: str,
    igfold,
    k: int,
) -> dict[str, torch.Tensor] | None:
    """Run IgFold on a single VH sequence and return its kNN dict."""
    try:
        out = igfold.fold(
            "/dev/null",
            sequences={"H": sequence},
            do_refine=False,
            use_openmm=False,
            do_renum=False,
            skip_pdb=True,
        )
        ca = out.coords[0, :, 1, :]
        knn = coords_to_knn(ca, k)
        return {"knn_indices": knn}
    except Exception as e:
        logger.warning("IgFold failed (len=%d): %s", len(sequence), e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict spatial neighborhoods via IgFold Cα coordinates"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--k_neighbors", type=int, default=32)
    parser.add_argument(
        "--num_models", type=int, default=1,
        help="IgFold ensemble size 1-4. 1 is ~4x faster; 4 is most accurate.",
    )
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="Start index (inclusive) into the JSONL. Used for sharding.",
    )
    parser.add_argument(
        "--end_idx", type=int, default=-1,
        help="End index (exclusive). -1 means end of file.",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=2000,
        help="Save partial results every N sequences.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip indices that already have a non-None entry in --output.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Loading sequences from %s", args.input)
    records = load_jsonl(args.input)
    n_total = len(records)

    start = max(0, args.start_idx)
    end = n_total if args.end_idx < 0 else min(n_total, args.end_idx)
    logger.info("Shard range: [%d, %d) of %d", start, end, n_total)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shard_len = end - start
    results: list[dict[str, torch.Tensor] | None] = [None] * shard_len

    if args.resume and output_path.exists():
        logger.info("Resuming from %s", output_path)
        existing = torch.load(output_path, weights_only=False)
        if len(existing) == shard_len:
            results = existing
            done = sum(1 for e in results if e is not None)
            logger.info("Loaded %d existing predictions", done)
        else:
            logger.warning(
                "Existing file length %d != shard length %d, starting fresh",
                len(existing), shard_len,
            )

    _patch_torch_load_for_igfold()

    if not torch.cuda.is_available():
        logger.error("CUDA not available; IgFold will be unusably slow on CPU.")
        sys.exit(1)

    logger.info("Loading IgFold (num_models=%d)...", args.num_models)
    from igfold import IgFoldRunner
    igfold = IgFoldRunner(num_models=args.num_models)
    logger.info("IgFold loaded on cuda:%d", torch.cuda.current_device())

    todo = [i for i in range(shard_len) if results[i] is None]
    logger.info(
        "To predict: %d (%d already done)", len(todo), shard_len - len(todo),
    )

    import time
    t0 = time.time()
    processed = 0
    for local_i in todo:
        global_i = start + local_i
        seq = records[global_i]["sequence"]
        results[local_i] = predict_one(seq, igfold, args.k_neighbors)
        processed += 1

        if processed % 50 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed
            remaining = (len(todo) - processed) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d (%.2f s/seq, ETA %.1f min)",
                processed, len(todo), 1.0 / rate, remaining / 60,
            )

        if processed % args.checkpoint_every == 0:
            torch.save(results, str(output_path))
            logger.info("Checkpoint saved to %s", output_path)

    torch.save(results, str(output_path))
    succeeded = sum(1 for e in results if e is not None)
    logger.info(
        "Done. Saved %d entries to %s (coverage %d/%d = %.1f%%)",
        len(results), output_path, succeeded, shard_len,
        succeeded / shard_len * 100 if shard_len else 0.0,
    )


if __name__ == "__main__":
    main()
