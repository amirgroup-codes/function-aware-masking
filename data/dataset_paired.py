"""PyTorch Dataset for paired VH+VL antibody sequences in multi-module format."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils.io import load_jsonl
from utils.tokenizer import encode_multispecific


PAIRED_CDR_FIELDS_HEAVY = {
    "cdr1_aa_heavy": 1,
    "cdr2_aa_heavy": 2,
    "cdr3_aa_heavy": 3,
}

PAIRED_CDR_FIELDS_LIGHT = {
    "cdr1_aa_light": 1,
    "cdr2_aa_light": 2,
    "cdr3_aa_light": 3,
}


def _build_cdr_region_labels_chain(
    sequence: str, cdr_fields: dict[str, int], record: dict,
) -> list[int]:
    """Build per-residue CDR region labels for a single chain.

    Same logic as dataset.py:_build_cdr_region_labels but parameterized
    for chain-specific CDR field names.

    Returns a list of length len(sequence) where 0=framework,
    1=CDR1, 2=CDR2, 3=CDR3.
    """
    labels = [0] * len(sequence)
    search_start = 0
    for field, region_id in cdr_fields.items():
        cdr_seq = record.get(field, "")
        if not cdr_seq:
            continue
        idx = sequence.find(cdr_seq, search_start)
        if idx == -1:
            continue
        for i in range(idx, idx + len(cdr_seq)):
            labels[i] = region_id
        search_start = idx + len(cdr_seq)
    return labels


class PairedAntibodyDataset(Dataset):
    """Dataset for paired VH+VL antibody sequences in multi-module format.

    Each item returns a dict with:
      - input_ids: tokenized multi-module sequence
      - attention_mask
      - special_tokens_mask
      - module_ids: per-token module assignment (0=global, 1=mod1, 2=mod2)
      - chain_type_ids: per-token chain type (0=special, 1=heavy, 2=light)

    Optional metadata (loaded from sidecar .pt files):
      - paratope_labels: per-token paratope probability
      - interface_labels: per-token VH-VL interface indicator
      - germline_labels: per-token germline mutation labels
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 384,
        paratope_path: str | None = None,
        interface_path: str | None = None,
        germline_path: str | None = None,
        bispecific: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bispecific = bispecific
        self.records = load_jsonl(data_path)

        self.paratope_labels: list | None = None
        if paratope_path and Path(paratope_path).exists():
            self.paratope_labels = torch.load(paratope_path, weights_only=False)
        self.interface_labels: list | None = None
        if interface_path and Path(interface_path).exists():
            self.interface_labels = torch.load(interface_path, weights_only=False)
        self.germline_labels: list | None = None
        if germline_path and Path(germline_path).exists():
            self.germline_labels = torch.load(germline_path, weights_only=False)

    def __len__(self) -> int:
        return len(self.records)

    def _map_chain_labels_to_tokens(
        self,
        aa_labels_heavy: torch.Tensor,
        aa_labels_light: torch.Tensor,
        aa_to_token_map: dict[tuple[int, str, int], int],
        num_tokens: int,
        module_idx: int = 1,
    ) -> torch.Tensor:
        """Map per-amino-acid labels from both chains to token-level tensor.

        Uses aa_to_token_map from encode_multispecific to place amino acid
        labels at the correct token positions. Unmapped positions stay 0.
        """
        token_labels = torch.zeros(num_tokens, dtype=torch.float)

        for aa_pos in range(len(aa_labels_heavy)):
            key = (module_idx, "heavy", aa_pos)
            if key in aa_to_token_map:
                token_labels[aa_to_token_map[key]] = aa_labels_heavy[aa_pos].float()

        for aa_pos in range(len(aa_labels_light)):
            key = (module_idx, "light", aa_pos)
            if key in aa_to_token_map:
                token_labels[aa_to_token_map[key]] = aa_labels_light[aa_pos].float()

        return token_labels

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        vh_2 = record.get("sequence_heavy_2") if self.bispecific else None
        vl_2 = record.get("sequence_light_2") if self.bispecific else None

        encoding = encode_multispecific(
            vh_1=record["sequence_heavy"],
            vl_1=record["sequence_light"],
            vh_2=vh_2,
            vl_2=vl_2,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        aa_to_token_map = encoding.pop("aa_to_token_map")
        num_tokens = len(encoding["input_ids"])

        result = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "special_tokens_mask": encoding["special_tokens_mask"],
            "module_ids": encoding["module_ids"],
            "chain_type_ids": encoding["chain_type_ids"],
        }

        # Build CDR region mask for both chains
        has_cdrs = any(
            record.get(f)
            for f in list(PAIRED_CDR_FIELDS_HEAVY) + list(PAIRED_CDR_FIELDS_LIGHT)
        )
        if has_cdrs:
            heavy_cdr_labels = _build_cdr_region_labels_chain(
                record["sequence_heavy"], PAIRED_CDR_FIELDS_HEAVY, record,
            )
            light_cdr_labels = _build_cdr_region_labels_chain(
                record["sequence_light"], PAIRED_CDR_FIELDS_LIGHT, record,
            )
            token_cdr_mask = self._map_chain_labels_to_tokens(
                torch.tensor(heavy_cdr_labels, dtype=torch.float),
                torch.tensor(light_cdr_labels, dtype=torch.float),
                aa_to_token_map,
                num_tokens,
            )
            result["cdr_mask"] = token_cdr_mask.long().tolist()

        # Attach paratope labels
        if (self.paratope_labels is not None
                and idx < len(self.paratope_labels)
                and self.paratope_labels[idx] is not None):
            entry = self.paratope_labels[idx]
            token_labels = self._map_chain_labels_to_tokens(
                entry["paratope_labels_heavy"],
                entry["paratope_labels_light"],
                aa_to_token_map,
                num_tokens,
            )
            result["paratope_labels"] = token_labels.tolist()

        # Attach interface labels
        if (self.interface_labels is not None
                and idx < len(self.interface_labels)
                and self.interface_labels[idx] is not None):
            entry = self.interface_labels[idx]
            token_labels = self._map_chain_labels_to_tokens(
                entry["interface_labels_heavy"],
                entry["interface_labels_light"],
                aa_to_token_map,
                num_tokens,
            )
            result["interface_labels"] = token_labels.tolist()

        # Attach germline labels
        if (self.germline_labels is not None
                and idx < len(self.germline_labels)
                and self.germline_labels[idx] is not None):
            entry = self.germline_labels[idx]
            token_labels = self._map_chain_labels_to_tokens(
                entry["germline_labels_heavy"],
                entry["germline_labels_light"],
                aa_to_token_map,
                num_tokens,
            )
            result["germline_labels"] = token_labels.tolist()

        return result
