"""Compute all annotations for multispecific-aware masking of paired VH+VL sequences.

Produces three sidecar .pt files in a single run:
  1. Paratope labels — soft P(paratope | AA, region) per chain
  2. VH-VL interface labels — canonical IMGT interface positions per chain
  3. Germline mutation labels — germline-match vs mutated per chain

Each output is a List[dict] aligned 1:1 with the input JSONL records.

Usage:
    python scripts/compute_multispecific_labels.py \
        --input data/processed/oas_paired_500k.jsonl \
        --output-dir data/structures \
        --prefix oas_paired_500k

    This produces:
        data/structures/oas_paired_500k_paratope.pt
        data/structures/oas_paired_500k_interface.pt
        data/structures/oas_paired_500k_germline.pt

    To run only a subset of annotations:
        --annotations paratope,interface   (skip germline)
"""

from __future__ import annotations

import argparse
import ast
import logging
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.io import load_jsonl

logger = logging.getLogger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Canonical VH-VL interface positions in IMGT numbering
# Based on Chothia & Lesk (1985) and Abhinandan & Martin (2010)
VH_INTERFACE_IMGT = {37, 39, 45, 47, 91, 93, 100, 103}
VL_INTERFACE_IMGT = {36, 38, 44, 46, 87, 89, 98, 100}
INTERFACE_LABEL = 1.0
ADJACENT_LABEL = 0.3


# ============================================================================
# Shared: CDR annotation helpers
# ============================================================================

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
    """Assign IMGT CDR region labels using ANARCI numbering."""
    from anarci import anarci as run_anarci

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

    if len(labels) < len(sequence):
        labels.extend([0] * (len(sequence) - len(labels)))
    return labels[:len(sequence)]


def _extract_single_chain_records(records: list[dict], chain: str) -> list[dict]:
    """Extract single-chain records from paired data."""
    single_records = []
    for record in records:
        seq = record.get(f"sequence_{chain}", "")
        if not seq:
            continue
        single_records.append({
            "sequence": seq,
            "v_call": record.get(f"v_call_{chain}", ""),
            "j_call": record.get(f"j_call_{chain}", ""),
            "cdr1_aa": record.get(f"cdr1_aa_{chain}", ""),
            "cdr2_aa": record.get(f"cdr2_aa_{chain}", ""),
            "cdr3_aa": record.get(f"cdr3_aa_{chain}", ""),
        })
    return single_records


# ============================================================================
# Paratope labels
# ============================================================================

class ParatopeFrequencyTable:
    """Empirical P(paratope | amino_acid, region) lookup table."""

    def __init__(self) -> None:
        self._rates: dict[tuple[str, int], float] = {}
        self._region_rates: dict[int, float] = {}
        self._global_rate: float = 0.0

    def fit(self, tdc_data) -> None:
        """Compute per-(amino_acid, region) paratope rates from TDC data."""
        counts: dict[tuple[str, int], int] = defaultdict(int)
        paratope_counts: dict[tuple[str, int], int] = defaultdict(int)
        region_total: dict[int, int] = defaultdict(int)
        region_paratope: dict[int, int] = defaultdict(int)

        logger.info("  Annotating %d TDC sequences with ANARCI (IMGT)...", len(tdc_data))
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
                logger.info("    Annotated %d / %d TDC sequences", idx + 1, len(tdc_data))

        total_res = sum(counts.values())
        total_para = sum(paratope_counts.values())
        self._global_rate = total_para / max(total_res, 1)

        for key, count in counts.items():
            para = paratope_counts.get(key, 0)
            self._rates[key] = para / count if count >= 5 else self._global_rate

        for region in range(4):
            total = region_total.get(region, 0)
            para = region_paratope.get(region, 0)
            self._region_rates[region] = para / max(total, 1)

        logger.info(
            "  Frequency table: %d residues (%.1f%% paratope)",
            total_res, self._global_rate * 100,
        )

    def predict(self, sequence: str, cdr_fields: dict) -> torch.Tensor:
        """Predict per-residue paratope probability."""
        has_cdr_fields = any(cdr_fields.get(f) for f in ("cdr1_aa", "cdr2_aa", "cdr3_aa"))
        region_labels = (build_cdr_region_labels(sequence, cdr_fields)
                         if has_cdr_fields else annotate_with_anarci(sequence))

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


def compute_paratope_labels(records: list[dict]) -> list[dict | None]:
    """Compute paratope labels for all paired records."""
    import pandas as pd
    from tdc.single_pred import Paratope

    logger.info("[Paratope] Loading TDC SAbDab_Liberis data...")
    tdc_data = Paratope(name="SAbDab_Liberis")
    tdc_split = tdc_data.get_split()
    tdc_all = pd.concat(
        [tdc_split["train"], tdc_split["valid"], tdc_split["test"]],
        ignore_index=True,
    )
    logger.info("[Paratope] TDC data: %d sequences", len(tdc_all))

    freq_table = ParatopeFrequencyTable()
    freq_table.fit(tdc_all)

    results: list[dict | None] = [None] * len(records)
    for i, record in enumerate(records):
        vh_probs = freq_table.predict(record["sequence_heavy"], {
            "cdr1_aa": record.get("cdr1_aa_heavy", ""),
            "cdr2_aa": record.get("cdr2_aa_heavy", ""),
            "cdr3_aa": record.get("cdr3_aa_heavy", ""),
        })
        vl_probs = freq_table.predict(record["sequence_light"], {
            "cdr1_aa": record.get("cdr1_aa_light", ""),
            "cdr2_aa": record.get("cdr2_aa_light", ""),
            "cdr3_aa": record.get("cdr3_aa_light", ""),
        })
        results[i] = {
            "paratope_labels_heavy": vh_probs,
            "paratope_labels_light": vl_probs,
        }
        if (i + 1) % 50000 == 0:
            logger.info("[Paratope] %d / %d", i + 1, len(records))

    logger.info("[Paratope] Done: %d records", len(records))
    return results


# ============================================================================
# Interface labels
# ============================================================================

def number_sequence_imgt(sequence: str) -> list[tuple[int, str]] | None:
    """Number a sequence using ANARCI IMGT scheme."""
    from anarci import anarci as run_anarci

    try:
        numbering, _, _ = run_anarci([("query", sequence)], scheme="imgt")
    except Exception:
        return None

    if not numbering or not numbering[0]:
        return None

    positions_list, _, _ = numbering[0][0]
    result = []
    for (pos_num, _insertion), aa in positions_list:
        if aa != "-":
            result.append((pos_num, aa))
    return result


def _compute_interface_labels_single(
    sequence: str,
    interface_imgt_positions: set[int],
    adjacent_label: float = ADJACENT_LABEL,
) -> torch.Tensor | None:
    """Compute per-residue interface labels for a single chain."""
    numbered = number_sequence_imgt(sequence)
    if numbered is None:
        return None

    seq_len = len(sequence)
    labels = torch.zeros(seq_len, dtype=torch.float)

    interface_seq_indices: set[int] = set()
    for seq_idx, (imgt_pos, _aa) in enumerate(numbered):
        if seq_idx >= seq_len:
            break
        if imgt_pos in interface_imgt_positions:
            interface_seq_indices.add(seq_idx)
            labels[seq_idx] = INTERFACE_LABEL

    for idx in interface_seq_indices:
        for adj in [idx - 1, idx + 1]:
            if 0 <= adj < seq_len and adj not in interface_seq_indices:
                labels[adj] = max(labels[adj].item(), adjacent_label)

    return labels


def _interface_worker(record: dict, adjacent_label: float = ADJACENT_LABEL) -> tuple:
    """Worker function for parallel interface label computation.

    Returns plain lists (not tensors) to avoid torch shared memory exhaustion
    when passing results back through multiprocessing IPC.
    """
    vh = record["sequence_heavy"]
    vl = record["sequence_light"]

    labels_h = _compute_interface_labels_single(vh, VH_INTERFACE_IMGT, adjacent_label)
    labels_l = _compute_interface_labels_single(vl, VL_INTERFACE_IMGT, adjacent_label)

    if labels_h is not None and labels_l is not None:
        return (labels_h.tolist(), labels_l.tolist(), True)
    else:
        return ([0.0] * len(vh), [0.0] * len(vl), False)


def compute_interface_labels(
    records: list[dict],
    adjacent_label: float = ADJACENT_LABEL,
    num_workers: int = 0,
) -> list[dict | None]:
    """Compute VH-VL interface labels for all paired records.

    Uses multiprocessing when num_workers > 0 to parallelize ANARCI calls.
    """
    if num_workers <= 0:
        num_workers = min(mp.cpu_count(), 48)

    worker_fn = partial(_interface_worker, adjacent_label=adjacent_label)

    logger.info("[Interface] Using %d workers for %d records", num_workers, len(records))

    results: list[dict | None] = []
    n_success = 0
    n_fallback = 0

    with mp.Pool(num_workers) as pool:
        for i, (lh, ll, success) in enumerate(pool.imap(worker_fn, records, chunksize=512)):
            if success:
                n_success += 1
            else:
                n_fallback += 1
            results.append({
                "interface_labels_heavy": torch.tensor(lh, dtype=torch.float),
                "interface_labels_light": torch.tensor(ll, dtype=torch.float),
            })

            if (i + 1) % 10000 == 0:
                logger.info(
                    "[Interface] %d / %d (success: %d, fallback: %d)",
                    i + 1, len(records), n_success, n_fallback,
                )

    logger.info(
        "[Interface] Done: %d records (success: %d, ANARCI fallback: %d)",
        len(records), n_success, n_fallback,
    )
    return results


# ============================================================================
# Germline labels
# ============================================================================

def compute_germline_labels(
    records: list[dict],
    min_seqs: int = 20,
    cdr3_label: float = 0.5,
) -> list[dict | None]:
    """Compute germline mutation labels for all paired records."""
    from scripts.compute_germline_labels import (
        build_germline_reference,
        label_sequence,
    )

    logger.info("[Germline] Building heavy chain reference...")
    heavy_records = _extract_single_chain_records(records, "heavy")
    ref_heavy = build_germline_reference(heavy_records, min_seqs=min_seqs)
    logger.info(
        "[Germline] Heavy: %d V genes, %d J genes",
        len(ref_heavy.v_genes), len(ref_heavy.j_genes),
    )

    logger.info("[Germline] Building light chain reference...")
    light_records = _extract_single_chain_records(records, "light")
    ref_light = build_germline_reference(light_records, min_seqs=min_seqs)
    logger.info(
        "[Germline] Light: %d V genes, %d J genes",
        len(ref_light.v_genes), len(ref_light.j_genes),
    )

    results: list[dict | None] = [None] * len(records)
    for i, record in enumerate(records):
        heavy_record = {
            "sequence": record["sequence_heavy"],
            "v_call": record.get("v_call_heavy", ""),
            "j_call": record.get("j_call_heavy", ""),
            "cdr1_aa": record.get("cdr1_aa_heavy", ""),
            "cdr2_aa": record.get("cdr2_aa_heavy", ""),
            "cdr3_aa": record.get("cdr3_aa_heavy", ""),
        }
        light_record = {
            "sequence": record["sequence_light"],
            "v_call": record.get("v_call_light", ""),
            "j_call": record.get("j_call_light", ""),
            "cdr1_aa": record.get("cdr1_aa_light", ""),
            "cdr2_aa": record.get("cdr2_aa_light", ""),
            "cdr3_aa": record.get("cdr3_aa_light", ""),
        }

        gl_heavy = label_sequence(heavy_record, ref_heavy, cdr3_label=cdr3_label)
        gl_light = label_sequence(light_record, ref_light, cdr3_label=cdr3_label)

        results[i] = {
            "germline_labels_heavy": gl_heavy,
            "germline_labels_light": gl_light,
        }

        if (i + 1) % 50000 == 0:
            logger.info("[Germline] %d / %d", i + 1, len(records))

    logger.info("[Germline] Done: %d records", len(records))
    return results


# ============================================================================
# Saving helper
# ============================================================================

def _save(results: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, str(path))
    logger.info("Saved %d entries to %s", len(results), path)


# ============================================================================
# Main
# ============================================================================

ALL_ANNOTATIONS = {"paratope", "interface", "germline"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute all multispecific-aware annotations (paratope, interface, "
            "germline) for paired VH+VL antibody sequences."
        ),
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to processed paired JSONL training file",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for output .pt sidecar files",
    )
    parser.add_argument(
        "--prefix", type=str, required=True,
        help="Filename prefix (e.g. 'oas_paired_500k')",
    )
    parser.add_argument(
        "--annotations", type=str, default="paratope,interface,germline",
        help="Comma-separated list of annotations to compute (default: all)",
    )
    parser.add_argument(
        "--adjacent-label", type=float, default=ADJACENT_LABEL,
        help=f"Interface adjacent-position label (default: {ADJACENT_LABEL})",
    )
    parser.add_argument(
        "--min-seqs-for-consensus", type=int, default=20,
        help="Min sequences per gene for germline consensus (default: 20)",
    )
    parser.add_argument(
        "--cdr3-label", type=float, default=0.5,
        help="Germline label for CDR3 junction positions (default: 0.5)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    requested = set(a.strip() for a in args.annotations.split(","))
    unknown = requested - ALL_ANNOTATIONS
    if unknown:
        parser.error(f"Unknown annotations: {unknown}. Choose from {ALL_ANNOTATIONS}")

    output_dir = Path(args.output_dir)

    # ---- Load training sequences ----
    logger.info("Loading paired sequences from %s", args.input)
    records = load_jsonl(args.input)
    logger.info("Loaded %d paired records", len(records))

    # ---- Paratope ----
    if "paratope" in requested:
        logger.info("=" * 60)
        logger.info("Computing PARATOPE labels...")
        paratope_results = compute_paratope_labels(records)
        _save(paratope_results, output_dir / f"{args.prefix}_paratope.pt")

    # ---- Interface ----
    if "interface" in requested:
        logger.info("=" * 60)
        logger.info("Computing INTERFACE labels...")
        interface_results = compute_interface_labels(
            records, adjacent_label=args.adjacent_label,
        )
        _save(interface_results, output_dir / f"{args.prefix}_interface.pt")

    # ---- Germline ----
    if "germline" in requested:
        logger.info("=" * 60)
        logger.info("Computing GERMLINE labels...")
        germline_results = compute_germline_labels(
            records,
            min_seqs=args.min_seqs_for_consensus,
            cdr3_label=args.cdr3_label,
        )
        _save(germline_results, output_dir / f"{args.prefix}_germline.pt")

    logger.info("=" * 60)
    logger.info("All requested annotations complete.")


if __name__ == "__main__":
    main()
