"""Compute paratope labels for all training sequences.

Builds an empirical P(paratope | amino_acid, region) frequency table
from TDC SAbDab_Liberis paratope data annotated with ANARCI IMGT CDR
regions, then predicts soft paratope probabilities for every training
sequence.

Output .pt format:
    List[dict | None] of length N (same as JSONL record count).
    Each entry: {"paratope_labels": FloatTensor(L,)} in [0, 1]

Usage:
    python scripts/compute_paratope_labels.py \
        --input data/processed/oas_vh_500k.jsonl \
        --output data/structures/oas_vh_500k_paratope.pt
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from utils.io import load_jsonl

logger = logging.getLogger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


# ---------------------------------------------------------------------------
# CDR annotation
# ---------------------------------------------------------------------------

def build_cdr_region_labels(sequence: str, cdr_fields: dict) -> list[int]:
    """Assign CDR region labels (0=FR, 1=CDR1, 2=CDR2, 3=CDR3)."""
    labels = [0] * len(sequence)
    search_start = 0
    for field, region_id in [("cdr1_aa", 1), ("cdr2_aa", 2), ("cdr3_aa", 3)]:
        cdr_seq = cdr_fields.get(field, "")
        if not cdr_seq:
            continue
        idx = sequence.find(cdr_seq, search_start)
        if idx == -1:
            continue
        for i in range(idx, idx + len(cdr_seq)):
            labels[i] = region_id
        search_start = idx + len(cdr_seq)
    return labels


def annotate_with_anarci(sequence: str) -> list[int]:
    """Assign IMGT CDR region labels using ANARCI numbering.

    Returns per-residue labels: 0=FR, 1=CDR1, 2=CDR2, 3=CDR3.
    Falls back to all-framework if ANARCI fails.
    """
    from anarci import anarci as run_anarci

    # IMGT CDR boundary definitions (inclusive)
    cdr_ranges = {1: (27, 38), 2: (56, 65), 3: (105, 117)}

    try:
        numbering, _, _ = run_anarci([("query", sequence)], scheme="imgt")
    except Exception:
        return [0] * len(sequence)

    if not numbering or not numbering[0]:
        return [0] * len(sequence)

    positions_list, _, _ = numbering[0][0]
    labels = []
    for (pos_num, _insertion), aa in positions_list:
        if aa == "-":
            continue
        region = 0
        for region_id, (start, end) in cdr_ranges.items():
            if start <= pos_num <= end:
                region = region_id
                break
        labels.append(region)

    # If ANARCI didn't cover the full sequence (rare), pad with FR
    if len(labels) < len(sequence):
        labels.extend([0] * (len(sequence) - len(labels)))

    return labels[:len(sequence)]


# ---------------------------------------------------------------------------
# Paratope frequency table
# ---------------------------------------------------------------------------

class ParatopeFrequencyTable:
    """Empirical P(paratope | amino_acid, region) lookup table.

    Built from TDC paratope data annotated with ANARCI IMGT CDR regions.
    Learns per-(amino_acid, region) paratope rates directly from labeled
    data — no hardcoded priors needed.
    """

    def __init__(self) -> None:
        # P(paratope | amino_acid, region) — key is (aa, region_id)
        self._rates: dict[tuple[str, int], float] = {}
        # Fallback rates per region
        self._region_rates: dict[int, float] = {}
        self._global_rate: float = 0.0

    def fit(self, tdc_data: pd.DataFrame) -> None:
        """Compute per-(amino_acid, region) paratope rates from labeled data.

        Uses ANARCI to annotate TDC sequences with IMGT CDR regions,
        then counts paratope frequency for each (aa, region) pair.
        """
        # Counts indexed by (aa, region)
        counts: dict[tuple[str, int], int] = defaultdict(int)
        paratope_counts: dict[tuple[str, int], int] = defaultdict(int)
        # Region-level counts
        region_total: dict[int, int] = defaultdict(int)
        region_paratope: dict[int, int] = defaultdict(int)

        logger.info("Annotating %d TDC sequences with ANARCI (IMGT)...", len(tdc_data))
        for idx, (_, row) in enumerate(tdc_data.iterrows()):
            seq = row["Antibody"] if "Antibody" in tdc_data.columns else row["X"]
            indices = row["Y"]
            if isinstance(indices, str):
                indices = ast.literal_eval(indices)
            paratope_set = set(int(i) for i in indices)

            region_labels = annotate_with_anarci(seq)

            for i, aa in enumerate(seq):
                if aa not in AA_TO_IDX:
                    continue
                region = region_labels[i] if i < len(region_labels) else 0
                key = (aa, region)
                counts[key] += 1
                region_total[region] += 1
                if i in paratope_set:
                    paratope_counts[key] += 1
                    region_paratope[region] += 1

            if (idx + 1) % 200 == 0:
                logger.info("  Annotated %d / %d TDC sequences", idx + 1, len(tdc_data))

        # Compute rates
        total_res = sum(counts.values())
        total_para = sum(paratope_counts.values())
        self._global_rate = total_para / max(total_res, 1)

        for key, count in counts.items():
            para = paratope_counts.get(key, 0)
            if count >= 5:
                self._rates[key] = para / count
            else:
                self._rates[key] = self._global_rate

        for region in range(4):
            total = region_total.get(region, 0)
            para = region_paratope.get(region, 0)
            self._region_rates[region] = para / max(total, 1)

        logger.info(
            "Frequency table built from %d residues (%.1f%% paratope)",
            total_res, self._global_rate * 100,
        )
        region_names = {0: "FR", 1: "CDR1", 2: "CDR2", 3: "CDR3"}
        for region in range(4):
            logger.info(
                "  %s: %.1f%% paratope (%d residues)",
                region_names[region],
                self._region_rates[region] * 100,
                region_total.get(region, 0),
            )

    def predict(self, sequence: str, cdr_fields: dict) -> torch.Tensor:
        """Predict per-residue paratope probability.

        Uses CDR fields from OAS (fast substring matching) when available,
        falls back to ANARCI if CDR fields are missing.
        """
        has_cdr_fields = any(cdr_fields.get(f) for f in ("cdr1_aa", "cdr2_aa", "cdr3_aa"))

        if has_cdr_fields:
            region_labels = build_cdr_region_labels(sequence, cdr_fields)
        else:
            region_labels = annotate_with_anarci(sequence)

        probs = torch.zeros(len(sequence), dtype=torch.float)

        for i, aa in enumerate(sequence):
            region = region_labels[i] if i < len(region_labels) else 0
            key = (aa, region)
            if key in self._rates:
                probs[i] = self._rates[key]
            elif region in self._region_rates:
                probs[i] = self._region_rates[region]
            else:
                probs[i] = self._global_rate

        return probs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paratope labels for antibody MLM training. "
            "Builds a frequency table from TDC paratope data with ANARCI "
            "CDR annotation, then predicts soft labels for all sequences."
        ),
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to processed JSONL training file",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output .pt sidecar file",
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to existing .pt file to resume from",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ---- Load training sequences ----
    logger.info("Loading training sequences from %s", args.input)
    records = load_jsonl(args.input)
    logger.info("Loaded %d sequences", len(records))

    # ---- Resume if available ----
    results: list[dict[str, torch.Tensor] | None] = [None] * len(records)
    if args.resume and Path(args.resume).exists():
        logger.info("Resuming from %s", args.resume)
        existing = torch.load(args.resume, weights_only=False)
        for i, entry in enumerate(existing):
            if i < len(results):
                results[i] = entry
        done = sum(1 for e in results if e is not None)
        logger.info("Loaded %d existing entries", done)

    # ---- Build paratope frequency table from TDC data ----
    logger.info("Loading TDC SAbDab_Liberis paratope data...")
    from tdc.single_pred import Paratope
    tdc_data = Paratope(name="SAbDab_Liberis")
    tdc_split = tdc_data.get_split()
    tdc_all = pd.concat(
        [tdc_split["train"], tdc_split["valid"], tdc_split["test"]],
        ignore_index=True,
    )
    logger.info("TDC paratope data: %d sequences", len(tdc_all))

    freq_table = ParatopeFrequencyTable()
    freq_table.fit(tdc_all)

    # ---- Predict soft labels for all sequences ----
    remaining = sum(1 for r in results if r is None)
    logger.info("Predicting soft labels for %d sequences...", remaining)

    for i, record in enumerate(records):
        if results[i] is not None:
            continue

        sequence = record["sequence"]
        cdr_fields = {
            "cdr1_aa": record.get("cdr1_aa", ""),
            "cdr2_aa": record.get("cdr2_aa", ""),
            "cdr3_aa": record.get("cdr3_aa", ""),
        }

        probs = freq_table.predict(sequence, cdr_fields)
        results[i] = {"paratope_labels": probs}

        if (i + 1) % 50000 == 0:
            logger.info("Predicted %d / %d sequences", i + 1, len(records))
            # Checkpoint
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(results, str(output_path))
            logger.info("Checkpoint saved to %s", output_path)

    # ---- Save final output ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, str(output_path))

    logger.info("Done. Saved %d entries to %s", len(results), output_path)
    logger.info("  Total coverage: 100%%")


if __name__ == "__main__":
    main()
