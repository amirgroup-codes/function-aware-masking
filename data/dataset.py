"""PyTorch Dataset for antibody sequences."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils.io import load_jsonl

logger = logging.getLogger(__name__)

CDR_FIELDS = {
    "cdr1_aa": 1,
    "cdr2_aa": 2,
    "cdr3_aa": 3,
}


def _build_cdr_region_labels(record: dict, seq_len: int) -> list[int]:
    """Build per-residue CDR region labels from OAS CDR amino acid fields.

    Returns a list of length seq_len where 0=framework, 1=CDR1, 2=CDR2, 3=CDR3.
    CDR1 and CDR2 are located via sequential forward search (left-to-right).
    CDR3 is located via rfind (rightmost match) because it is always anchored
    at the C-terminus of the V region; a forward search from an inherited
    search_start can match a false positive earlier in the framework when
    CDR1/CDR2 lookups fail.
    """
    sequence = record["sequence"]
    labels = [0] * seq_len
    search_start = 0

    for field, region_id in [("cdr1_aa", 1), ("cdr2_aa", 2)]:
        cdr_seq = record.get(field, "")
        if not cdr_seq:
            continue
        idx = sequence.find(cdr_seq, search_start)
        if idx == -1:
            logger.debug(
                "CDR%d substring %r not found after position %d",
                region_id, cdr_seq, search_start,
            )
            continue
        for i in range(idx, min(idx + len(cdr_seq), seq_len)):
            labels[i] = region_id
        search_start = idx + len(cdr_seq)

    cdr3_seq = record.get("cdr3_aa", "")
    if cdr3_seq:
        idx = sequence.rfind(cdr3_seq)
        if idx == -1:
            logger.debug("CDR3 substring %r not found in sequence", cdr3_seq)
        else:
            for i in range(idx, min(idx + len(cdr3_seq), seq_len)):
                labels[i] = 3

    return labels


class AntibodyDataset(Dataset):
    """Dataset that tokenizes antibody sequences on-the-fly.

    Each item returns a dict with 'input_ids', 'attention_mask',
    'special_tokens_mask', and optionally 'cdr_mask' (per-token CDR
    region labels) ready to be passed to an MLM data collator.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 160,
        coords_path: str | None = None,
        paratope_path: str | None = None,
        germline_path: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = load_jsonl(data_path)
        self.coords: list | None = None
        if coords_path and Path(coords_path).exists():
            self.coords = torch.load(coords_path, weights_only=False)
        self.paratope_labels: list | None = None
        if paratope_path and Path(paratope_path).exists():
            self.paratope_labels = torch.load(paratope_path, weights_only=False)
        self.germline_labels: list | None = None
        if germline_path and Path(germline_path).exists():
            self.germline_labels = torch.load(germline_path, weights_only=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        sequence = record["sequence"]
        spaced = " ".join(list(sequence))
        encoding = self.tokenizer(
            spaced,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_special_tokens_mask=True,
        )

        result = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "special_tokens_mask": encoding["special_tokens_mask"],
        }

        has_cdrs = any(record.get(f) for f in CDR_FIELDS)
        if has_cdrs:
            aa_labels = _build_cdr_region_labels(record, len(sequence))
            num_tokens = len(encoding["input_ids"])
            token_labels = [0] * num_tokens
            for i, label in enumerate(aa_labels):
                token_pos = i + 1  # offset for [CLS]
                if token_pos < num_tokens - 1:  # don't overwrite [SEP]
                    token_labels[token_pos] = label
            result["cdr_mask"] = token_labels

        if self.coords is not None and idx < len(self.coords) and self.coords[idx] is not None:
            entry = self.coords[idx]
            num_tokens = len(encoding["input_ids"])

            if "knn_indices" in entry:
                # ESM2 contact-map based kNN neighbors (per amino acid)
                aa_knn = entry["knn_indices"]  # [L_aa, k]
                k = aa_knn.size(1)
                # Build token-level kNN: offset by 1 for [CLS], pad specials to 0
                token_knn = torch.zeros(num_tokens, k, dtype=torch.long)
                n_aa = min(len(aa_knn), num_tokens - 2)
                # Offset neighbor indices by 1 to account for [CLS] token
                token_knn[1 : 1 + n_aa] = aa_knn[:n_aa] + 1
                result["knn_indices"] = token_knn.tolist()
            elif "coords_ca" in entry:
                # Legacy: IgFold Calpha coordinates
                aa_coords = entry["coords_ca"]
                token_coords = torch.zeros(num_tokens, 3)
                n_aa = min(len(aa_coords), num_tokens - 2)
                token_coords[1 : 1 + n_aa] = aa_coords[:n_aa].float()
                result["coords_ca"] = token_coords.tolist()

        if (self.paratope_labels is not None
                and idx < len(self.paratope_labels)
                and self.paratope_labels[idx] is not None):
            entry = self.paratope_labels[idx]
            aa_labels = entry["paratope_labels"]
            num_tokens = len(encoding["input_ids"])
            token_labels = torch.zeros(num_tokens, dtype=torch.float)
            n_aa = min(len(aa_labels), num_tokens - 2)
            token_labels[1:1 + n_aa] = aa_labels[:n_aa].float()
            result["paratope_labels"] = token_labels.tolist()

        if (self.germline_labels is not None
                and idx < len(self.germline_labels)
                and self.germline_labels[idx] is not None):
            entry = self.germline_labels[idx]
            aa_labels = entry["germline_labels"]
            num_tokens = len(encoding["input_ids"])
            token_labels = torch.zeros(num_tokens, dtype=torch.float)
            n_aa = min(len(aa_labels), num_tokens - 2)
            token_labels[1:1 + n_aa] = aa_labels[:n_aa].float()
            result["germline_labels"] = token_labels.tolist()

        return result
