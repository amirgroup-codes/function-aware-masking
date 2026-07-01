"""Antibody sequence preprocessing: filtering, deduplication, length constraints."""

from __future__ import annotations

import re
from typing import Any

DEFAULT_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def is_valid_sequence(
    sequence: str,
    min_length: int = 80,
    max_length: int = 200,
    valid_aa: set[str] = DEFAULT_VALID_AA,
) -> bool:
    """Check whether an amino acid sequence passes quality filters."""
    if not sequence:
        return False
    if not min_length <= len(sequence) <= max_length:
        return False
    if not set(sequence.upper()).issubset(valid_aa):
        return False
    return True


def clean_sequence(sequence: str) -> str:
    """Normalize an amino acid sequence: uppercase, strip whitespace."""
    return re.sub(r"\s+", "", sequence.upper().strip())


def preprocess_sequences(
    records: list[dict[str, Any]],
    sequence_key: str = "sequence",
    min_length: int = 80,
    max_length: int = 200,
    valid_amino_acids: str = "ACDEFGHIKLMNPQRSTVWY",
) -> list[dict[str, Any]]:
    """Filter and deduplicate antibody sequence records.

    Args:
        records: Raw records, each containing at least `sequence_key`.
        sequence_key: Key in each record holding the amino acid sequence.
        min_length: Minimum acceptable sequence length.
        max_length: Maximum acceptable sequence length.
        valid_amino_acids: String of allowed amino acid characters.

    Returns:
        Deduplicated, filtered records with cleaned sequences.
    """
    valid_aa = set(valid_amino_acids)
    seen: set[str] = set()
    cleaned: list[dict[str, Any]] = []

    for record in records:
        seq = clean_sequence(record.get(sequence_key, ""))
        if not seq:
            continue
        if not is_valid_sequence(seq, min_length, max_length, valid_aa):
            continue
        if seq in seen:
            continue
        seen.add(seq)
        out = dict(record)
        out[sequence_key] = seq
        cleaned.append(out)

    return cleaned
