"""CDR annotation for antibody sequences.

Two annotation paths:
  1. From OAS pre-annotations: maps CDR amino acid subsequences back to
     positions in the full sequence (fast, no external dependencies).
  2. From ANARCI: runs IMGT numbering on raw sequences (slower, requires
     ANARCI + HMMER installed). Used as fallback for unannotated sequences.

The output format is a per-residue binary mask where 1 = CDR position.
"""

from __future__ import annotations

import logging
from typing import Any

from utils.io import load_jsonl, save_jsonl

logger = logging.getLogger(__name__)

IMGT_CDR_RANGES = {
    "cdr1": (27, 38),
    "cdr2": (56, 65),
    "cdr3": (105, 117),
}


def _find_subsequence(sequence: str, subseq: str) -> tuple[int, int] | None:
    """Find start and end indices of a subsequence in a sequence.

    Returns (start, end) where end is exclusive, or None if not found.
    """
    if not subseq:
        return None
    idx = sequence.find(subseq)
    if idx == -1:
        return None
    return (idx, idx + len(subseq))


def annotate_from_oas_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Build CDR position mask from OAS pre-annotated CDR fields.

    Expects record to have 'sequence' and optionally 'cdr1_aa', 'cdr2_aa', 'cdr3_aa'.

    Returns a new record dict with added fields:
      - cdr_mask: list[int] of length len(sequence), 1 at CDR positions
      - cdr_regions: dict mapping region name to (start, end) tuple
    """
    sequence = record["sequence"]
    seq_len = len(sequence)
    cdr_mask = [0] * seq_len
    cdr_regions: dict[str, tuple[int, int]] = {}

    for cdr_name in ("cdr1_aa", "cdr2_aa", "cdr3_aa"):
        cdr_seq = record.get(cdr_name, "")
        if not cdr_seq:
            continue
        span = _find_subsequence(sequence, cdr_seq)
        if span is None:
            logger.debug(
                "CDR %s subsequence '%s' not found in sequence (len=%d)",
                cdr_name, cdr_seq, seq_len,
            )
            continue
        start, end = span
        for i in range(start, end):
            cdr_mask[i] = 1
        region_key = cdr_name.replace("_aa", "")
        cdr_regions[region_key] = (start, end)

    out = dict(record)
    out["cdr_mask"] = cdr_mask
    out["cdr_regions"] = cdr_regions
    return out


def annotate_with_anarci(sequence: str) -> dict[str, Any] | None:
    """Annotate a single sequence using ANARCI with IMGT numbering.

    Returns dict with cdr_mask and cdr_regions, or None if ANARCI fails.
    Requires ANARCI to be installed.
    """
    try:
        from anarci import anarci as run_anarci
    except ImportError:
        logger.error("ANARCI not installed. pip install anarci")
        return None

    results = run_anarci([("query", sequence)], scheme="imgt", output=False)
    if results is None or results[1] is None:
        return None

    numbering_results, _, _ = results
    if not numbering_results or not numbering_results[0]:
        return None

    numbering = numbering_results[0][0][0]
    seq_len = len(sequence)
    cdr_mask = [0] * seq_len
    cdr_regions: dict[str, tuple[int, int]] = {}

    pos_to_seq_idx: dict[int, int] = {}
    seq_idx = 0
    for (imgt_pos, insertion), aa in numbering:
        if aa != "-":
            pos_to_seq_idx[imgt_pos] = seq_idx
            seq_idx += 1

    for cdr_name, (start_imgt, end_imgt) in IMGT_CDR_RANGES.items():
        region_positions = []
        for imgt_pos in range(start_imgt, end_imgt + 1):
            if imgt_pos in pos_to_seq_idx:
                seq_i = pos_to_seq_idx[imgt_pos]
                cdr_mask[seq_i] = 1
                region_positions.append(seq_i)
        if region_positions:
            cdr_regions[cdr_name] = (min(region_positions), max(region_positions) + 1)

    return {"cdr_mask": cdr_mask, "cdr_regions": cdr_regions}


def annotate_dataset(
    input_path: str,
    output_path: str,
    use_anarci_fallback: bool = False,
) -> None:
    """Annotate an entire dataset with CDR positions.

    Reads JSONL, adds cdr_mask and cdr_regions to each record, writes JSONL.

    Args:
        input_path: Path to input JSONL (must have 'sequence' field).
        output_path: Path to write annotated JSONL.
        use_anarci_fallback: If True, use ANARCI for records without OAS CDR fields.
    """
    records = load_jsonl(input_path)
    annotated = []
    success_count = 0
    fail_count = 0

    for record in records:
        has_oas_cdrs = any(record.get(f"cdr{i}_aa") for i in (1, 2, 3))

        if has_oas_cdrs:
            result = annotate_from_oas_fields(record)
            if sum(result["cdr_mask"]) > 0:
                annotated.append(result)
                success_count += 1
                continue

        if use_anarci_fallback:
            anarci_result = annotate_with_anarci(record["sequence"])
            if anarci_result:
                out = dict(record)
                out.update(anarci_result)
                annotated.append(out)
                success_count += 1
                continue

        out = dict(record)
        out["cdr_mask"] = [0] * len(record["sequence"])
        out["cdr_regions"] = {}
        annotated.append(out)
        fail_count += 1

    save_jsonl(annotated, output_path)
    logger.info(
        "Annotated %d/%d sequences with CDR positions (%d without CDRs)",
        success_count, len(records), fail_count,
    )
