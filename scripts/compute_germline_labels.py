"""Compute per-residue germline mutation labels for all training sequences.

Builds consensus germline V/J gene sequences from the training data itself
(grouped by v_call / j_call), then labels each residue as germline-matching
(0.0), mutated (1.0), or CDR3-junction (configurable, default 0.5).

Output .pt format:
    List[dict] of length N (same as JSONL record count).
    Each entry: {"germline_labels": FloatTensor(L,)} in [0, 1]

Usage:
    python scripts/compute_germline_labels.py \
        --input data/processed/oas_vh_500k.jsonl \
        --output data/structures/oas_vh_500k_germline.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.io import load_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Germline reference building
# ---------------------------------------------------------------------------

@dataclass
class GeneReference:
    """Consensus germline sequence for a single V or J gene."""
    consensus: str
    count: int
    cdr1_start: int = -1
    cdr1_end: int = -1
    cdr2_start: int = -1
    cdr2_end: int = -1


@dataclass
class GermlineReference:
    """Lookup tables for consensus V-region and J-region (FR4) sequences."""
    v_genes: dict[str, GeneReference] = field(default_factory=dict)
    j_genes: dict[str, GeneReference] = field(default_factory=dict)


def _consensus_sequence(sequences: list[str]) -> str:
    """Compute per-position consensus (most frequent AA) from equal-length sequences."""
    if not sequences:
        return ""
    length = len(sequences[0])
    consensus = []
    for pos in range(length):
        counts = Counter(seq[pos] for seq in sequences if pos < len(seq))
        consensus.append(counts.most_common(1)[0][0])
    return "".join(consensus)


def _find_cdr_boundaries(sequence: str, cdr_seq: str, search_start: int = 0) -> tuple[int, int]:
    """Find start and end index of a CDR substring in a sequence."""
    if not cdr_seq:
        return -1, -1
    idx = sequence.find(cdr_seq, search_start)
    if idx == -1:
        return -1, -1
    return idx, idx + len(cdr_seq)


def build_germline_reference(
    records: list[dict],
    min_seqs: int = 20,
) -> GermlineReference:
    """Build consensus germline V/J gene sequences from OAS records.

    Groups sequences by v_call/j_call, extracts V-regions (before CDR3)
    and FR4 (after CDR3), and computes per-position consensus.
    """
    ref = GermlineReference()

    # ---- Group V-regions by v_call ----
    v_regions: dict[str, list[str]] = defaultdict(list)
    v_cdr_fields: dict[str, list[dict]] = defaultdict(list)
    j_regions: dict[str, list[str]] = defaultdict(list)

    for record in records:
        v_call = record.get("v_call", "")
        j_call = record.get("j_call", "")
        sequence = record.get("sequence", "")
        cdr3 = record.get("cdr3_aa", "")

        if not sequence or not cdr3:
            continue

        cdr3_idx = sequence.find(cdr3)
        if cdr3_idx == -1:
            continue

        # V-region: everything before CDR3
        v_region = sequence[:cdr3_idx]
        if v_call and v_region:
            v_regions[v_call].append(v_region)
            v_cdr_fields[v_call].append({
                "cdr1_aa": record.get("cdr1_aa", ""),
                "cdr2_aa": record.get("cdr2_aa", ""),
            })

        # FR4: everything after CDR3
        fr4 = sequence[cdr3_idx + len(cdr3):]
        if j_call and fr4:
            j_regions[j_call].append(fr4)

    # ---- Build V-gene consensus ----
    for v_call, regions in v_regions.items():
        if len(regions) < min_seqs:
            logger.warning(
                "V gene %s has only %d sequences (< %d), skipping consensus",
                v_call, len(regions), min_seqs,
            )
            continue

        # Group by length, take most common
        length_counts = Counter(len(r) for r in regions)
        canonical_len = length_counts.most_common(1)[0][0]
        canonical_seqs = [r for r in regions if len(r) == canonical_len]

        consensus = _consensus_sequence(canonical_seqs)
        gene_ref = GeneReference(consensus=consensus, count=len(regions))

        # Find CDR1/CDR2 boundaries by most common start position
        # (substring search in consensus can fail when per-position consensus
        # differs from the most-common CDR substring)
        cdr_fields = v_cdr_fields[v_call]
        canonical_cdr_fields = [
            f for f, r in zip(cdr_fields, regions) if len(r) == canonical_len
        ]
        for cdr_field, attr_start, attr_end in [
            ("cdr1_aa", "cdr1_start", "cdr1_end"),
            ("cdr2_aa", "cdr2_start", "cdr2_end"),
        ]:
            positions: list[tuple[int, int]] = []
            search_offset = getattr(gene_ref, "cdr1_end", 0) if cdr_field == "cdr2_aa" else 0
            for f, region in zip(canonical_cdr_fields, canonical_seqs):
                cdr_seq = f.get(cdr_field, "")
                if not cdr_seq:
                    continue
                idx = region.find(cdr_seq, search_offset)
                if idx >= 0:
                    positions.append((idx, idx + len(cdr_seq)))
            if positions:
                most_common_pos = Counter(positions).most_common(1)[0][0]
                setattr(gene_ref, attr_start, most_common_pos[0])
                setattr(gene_ref, attr_end, most_common_pos[1])

        ref.v_genes[v_call] = gene_ref
        logger.info(
            "  V gene %s: consensus len=%d from %d seqs (%d canonical-length)",
            v_call, len(consensus), len(regions), len(canonical_seqs),
        )

    # ---- Build J-gene (FR4) consensus ----
    for j_call, regions in j_regions.items():
        if len(regions) < min_seqs:
            logger.warning(
                "J gene %s has only %d sequences (< %d), skipping consensus",
                j_call, len(regions), min_seqs,
            )
            continue

        length_counts = Counter(len(r) for r in regions)
        canonical_len = length_counts.most_common(1)[0][0]
        canonical_seqs = [r for r in regions if len(r) == canonical_len]

        consensus = _consensus_sequence(canonical_seqs)
        ref.j_genes[j_call] = GeneReference(consensus=consensus, count=len(regions))
        logger.info(
            "  J gene %s: consensus len=%d from %d seqs",
            j_call, len(consensus), len(regions),
        )

    return ref


# ---------------------------------------------------------------------------
# Per-sequence labeling
# ---------------------------------------------------------------------------

def _compare_segments(mature: str, germline: str) -> list[float]:
    """Compare two aligned segments character-by-character.

    Returns per-position labels: 0.0 = match, 1.0 = mismatch.
    If lengths differ, uses CDR-anchored alignment or marks as 0.5.
    """
    if len(mature) == len(germline):
        return [0.0 if m == g else 1.0 for m, g in zip(mature, germline)]

    # Length mismatch: try to align from the start, mark overflow as 0.5
    min_len = min(len(mature), len(germline))
    labels = [0.0 if mature[i] == germline[i] else 1.0 for i in range(min_len)]
    # Extra positions in mature get 0.5 (uncertain)
    labels.extend([0.5] * (len(mature) - min_len))
    return labels


def _segment_with_cdrs(
    sequence: str,
    cdr1: str,
    cdr2: str,
) -> list[tuple[str, str]]:
    """Segment a V-region into (segment_string, segment_type) pairs.

    Returns list of (substring, type) where type is 'fr1', 'cdr1', 'fr2', 'cdr2', 'fr3'.
    """
    segments = []
    search_start = 0

    cdr1_idx = sequence.find(cdr1, search_start) if cdr1 else -1
    if cdr1_idx >= 0:
        segments.append((sequence[:cdr1_idx], "fr1"))
        segments.append((cdr1, "cdr1"))
        search_start = cdr1_idx + len(cdr1)
    else:
        # No CDR1 found — can't segment further
        return [("", "unsegmented")]

    cdr2_idx = sequence.find(cdr2, search_start) if cdr2 else -1
    if cdr2_idx >= 0:
        segments.append((sequence[search_start:cdr2_idx], "fr2"))
        segments.append((cdr2, "cdr2"))
        segments.append((sequence[cdr2_idx + len(cdr2):], "fr3"))
    else:
        segments.append((sequence[search_start:], "fr2_plus"))

    return segments


def label_sequence(
    record: dict,
    ref: GermlineReference,
    cdr3_label: float = 0.5,
) -> torch.Tensor:
    """Compute per-residue germline mutation labels for a single sequence.

    Returns FloatTensor of length L with values:
        0.0 = germline match
        1.0 = mutated from germline
        cdr3_label = CDR3 junction (default 0.5)
        0.5 = uncertain / could not align
    """
    sequence = record.get("sequence", "")
    v_call = record.get("v_call", "")
    j_call = record.get("j_call", "")
    cdr1_aa = record.get("cdr1_aa", "")
    cdr2_aa = record.get("cdr2_aa", "")
    cdr3_aa = record.get("cdr3_aa", "")

    seq_len = len(sequence)
    labels = [0.5] * seq_len  # default: uncertain

    if not cdr3_aa or not sequence:
        return torch.tensor(labels, dtype=torch.float)

    # Find CDR3 boundaries in mature sequence
    cdr3_idx = sequence.find(cdr3_aa)
    if cdr3_idx == -1:
        return torch.tensor(labels, dtype=torch.float)

    v_region = sequence[:cdr3_idx]
    fr4 = sequence[cdr3_idx + len(cdr3_aa):]

    # ---- Label V-region (FR1-CDR1-FR2-CDR2-FR3) ----
    v_ref = ref.v_genes.get(v_call)
    if v_ref is not None:
        consensus_v = v_ref.consensus

        if len(v_region) == len(consensus_v):
            # Same length: direct character comparison
            v_labels = _compare_segments(v_region, consensus_v)
        else:
            # Different length: try CDR-anchored segmented comparison
            # Get CDR boundaries in consensus
            consensus_cdr1 = consensus_v[v_ref.cdr1_start:v_ref.cdr1_end] if v_ref.cdr1_start >= 0 else ""
            consensus_cdr2 = consensus_v[v_ref.cdr2_start:v_ref.cdr2_end] if v_ref.cdr2_start >= 0 else ""

            mature_segs = _segment_with_cdrs(v_region, cdr1_aa, cdr2_aa)
            consensus_segs = _segment_with_cdrs(consensus_v, consensus_cdr1, consensus_cdr2)

            if (len(mature_segs) == len(consensus_segs)
                    and mature_segs[0][1] != "unsegmented"):
                v_labels = []
                for (m_seg, _), (c_seg, _) in zip(mature_segs, consensus_segs):
                    v_labels.extend(_compare_segments(m_seg, c_seg))
            else:
                # Fallback: align from start
                v_labels = _compare_segments(v_region, consensus_v)

        for i, label in enumerate(v_labels):
            if i < cdr3_idx:
                labels[i] = label
    else:
        # Unknown V gene: leave as 0.5
        pass

    # ---- Label CDR3 junction ----
    for i in range(cdr3_idx, min(cdr3_idx + len(cdr3_aa), seq_len)):
        labels[i] = cdr3_label

    # ---- Label FR4 (J gene region) ----
    j_ref = ref.j_genes.get(j_call)
    if j_ref is not None and fr4:
        fr4_start = cdr3_idx + len(cdr3_aa)
        fr4_labels = _compare_segments(fr4, j_ref.consensus)
        for i, label in enumerate(fr4_labels):
            pos = fr4_start + i
            if pos < seq_len:
                labels[pos] = label

    return torch.tensor(labels, dtype=torch.float)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute germline mutation labels for antibody MLM training. "
            "Builds consensus germline V/J sequences from the data, then "
            "labels each residue as germline-matching or mutated."
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
    parser.add_argument(
        "--min-seqs-for-consensus", type=int, default=20,
        help="Minimum sequences per gene to build consensus (default: 20)",
    )
    parser.add_argument(
        "--cdr3-label", type=float, default=0.5,
        help="Label for CDR3 junction positions (default: 0.5)",
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

    # ---- Pass 1: Build consensus germline reference ----
    logger.info("Building consensus germline reference...")
    ref = build_germline_reference(records, min_seqs=args.min_seqs_for_consensus)
    logger.info(
        "Built reference: %d V genes, %d J genes",
        len(ref.v_genes), len(ref.j_genes),
    )

    # ---- Pass 2: Label each sequence ----
    remaining = sum(1 for r in results if r is None)
    logger.info("Labeling %d sequences (cdr3_label=%.2f)...", remaining, args.cdr3_label)

    # Stats tracking
    n_labeled = 0
    n_fallback = 0
    region_mutation_counts: dict[str, list[float]] = defaultdict(list)

    for i, record in enumerate(records):
        if results[i] is not None:
            continue

        germline_labels = label_sequence(record, ref, cdr3_label=args.cdr3_label)
        results[i] = {"germline_labels": germline_labels}
        n_labeled += 1

        # Track stats
        if germline_labels.mean().item() == 0.5:
            n_fallback += 1
        else:
            # Compute per-region mutation rates
            sequence = record.get("sequence", "")
            cdr3_aa = record.get("cdr3_aa", "")
            cdr3_idx = sequence.find(cdr3_aa) if cdr3_aa else -1
            if cdr3_idx >= 0:
                v_labels = germline_labels[:cdr3_idx]
                fr4_labels = germline_labels[cdr3_idx + len(cdr3_aa):]
                if len(v_labels) > 0:
                    region_mutation_counts["v_region"].append(
                        (v_labels == 1.0).float().mean().item()
                    )
                if len(fr4_labels) > 0:
                    region_mutation_counts["fr4"].append(
                        (fr4_labels == 1.0).float().mean().item()
                    )

        if (i + 1) % 50000 == 0:
            logger.info("Labeled %d / %d sequences", i + 1, len(records))
            # Checkpoint
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(results, str(output_path))
            logger.info("Checkpoint saved to %s", output_path)

    # ---- Save final output ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, str(output_path))

    # ---- Summary statistics ----
    total_coverage = sum(1 for r in results if r is not None)
    logger.info("Done. Saved %d entries to %s", len(results), output_path)
    logger.info("  Coverage: %d / %d (%.1f%%)",
                total_coverage, len(results),
                total_coverage / len(results) * 100)
    logger.info("  Fallback (all 0.5): %d / %d (%.1f%%)",
                n_fallback, n_labeled,
                n_fallback / max(n_labeled, 1) * 100)

    # Per-region stats
    for region, rates in sorted(region_mutation_counts.items()):
        if rates:
            avg = sum(rates) / len(rates)
            logger.info("  %s avg mutation rate: %.1f%%", region, avg * 100)

    # Overall mutation rate (excluding CDR3)
    all_mutation_rates = []
    for r in results:
        if r is not None:
            gl = r["germline_labels"]
            mutated = (gl == 1.0).float().mean().item()
            all_mutation_rates.append(mutated)
    if all_mutation_rates:
        avg = sum(all_mutation_rates) / len(all_mutation_rates)
        logger.info("  Overall avg mutation rate (incl. CDR3 as %.1f): %.1f%%",
                     args.cdr3_label, avg * 100)


if __name__ == "__main__":
    main()
