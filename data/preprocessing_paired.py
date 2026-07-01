"""Paired antibody sequence preprocessing: filtering, deduplication, length constraints."""

from __future__ import annotations

import re
from typing import Any

DEFAULT_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(sequence: str) -> str:
    """Normalize an amino acid sequence: uppercase, strip whitespace."""
    return re.sub(r"\s+", "", sequence.upper().strip())


def is_valid_paired_sequence(
    record: dict[str, Any],
    min_heavy_length: int = 80,
    max_heavy_length: int = 160,
    min_light_length: int = 80,
    max_light_length: int = 140,
    valid_aa: set[str] = DEFAULT_VALID_AA,
) -> bool:
    """Check whether a paired VH+VL record passes quality filters."""
    seq_h = record.get("sequence_heavy", "")
    seq_l = record.get("sequence_light", "")
    if not seq_h or not seq_l:
        return False
    if not min_heavy_length <= len(seq_h) <= max_heavy_length:
        return False
    if not min_light_length <= len(seq_l) <= max_light_length:
        return False
    if not set(seq_h.upper()).issubset(valid_aa):
        return False
    if not set(seq_l.upper()).issubset(valid_aa):
        return False
    return True


def preprocess_paired_sequences(
    records: list[dict[str, Any]],
    min_heavy_length: int = 80,
    max_heavy_length: int = 160,
    min_light_length: int = 80,
    max_light_length: int = 140,
    valid_amino_acids: str = "ACDEFGHIKLMNPQRSTVWY",
) -> list[dict[str, Any]]:
    """Filter and deduplicate paired antibody sequence records.

    Deduplication key: (sequence_heavy, sequence_light) tuple.

    Args:
        records: Raw paired records with sequence_heavy and sequence_light.
        min_heavy_length: Minimum VH sequence length.
        max_heavy_length: Maximum VH sequence length.
        min_light_length: Minimum VL sequence length.
        max_light_length: Maximum VL sequence length.
        valid_amino_acids: String of allowed amino acid characters.

    Returns:
        Deduplicated, filtered records with cleaned sequences.
    """
    valid_aa = set(valid_amino_acids)
    seen: set[tuple[str, str]] = set()
    cleaned: list[dict[str, Any]] = []

    for record in records:
        seq_h = clean_sequence(record.get("sequence_heavy", ""))
        seq_l = clean_sequence(record.get("sequence_light", ""))
        if not seq_h or not seq_l:
            continue

        out = dict(record)
        out["sequence_heavy"] = seq_h
        out["sequence_light"] = seq_l

        if not is_valid_paired_sequence(
            out, min_heavy_length, max_heavy_length,
            min_light_length, max_light_length, valid_aa,
        ):
            continue

        key = (seq_h, seq_l)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(out)

    return cleaned
