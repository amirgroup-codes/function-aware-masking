"""Germline and SHM mutation masking using germline alignment.

Biases the MLM mask budget toward positions that have mutated away from
germline (somatic hypermutation sites) using per-residue labels in [0, 1].
Labels are 0.0 for germline-matching positions, 1.0 for mutated positions,
and an intermediate value (default 0.5) for CDR3 junction positions where
junctional diversity and SHM cannot be distinguished.

Addresses the "germline bias" problem identified by AbLang-2: ~85% of VH
residues match germline, so uniform masking overwhelmingly trains on
germline residues rather than functionally important SHM mutations.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from masking.base import BaseMaskingStrategy, register_strategy

logger = logging.getLogger(__name__)


@register_strategy("germline")
class GermlineMasking(BaseMaskingStrategy):
    """Mutation-biased masking with weighted Bernoulli sampling.

    Allocates the mask budget (default 15%) disproportionately toward
    positions that differ from germline using per-residue weights derived
    from germline mutation labels.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_prob: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
        mutated_weight: float = 6.0,
        germline_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            mask_prob=mask_prob,
            mask_token_ratio=mask_token_ratio,
            random_token_ratio=random_token_ratio,
            **kwargs,
        )
        self.mutated_weight = mutated_weight
        self.germline_weight = germline_weight
        self._total_calls = 0
        self._fallback_calls = 0
        self._log_interval = 1000

    def select_mask_positions(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        germline_labels = metadata.get("germline_labels") if metadata else None
        self._total_calls += 1

        if germline_labels is None:
            self._fallback_calls += 1
            if self._fallback_calls == 1:
                logger.warning(
                    "GermlineMasking: falling back to uniform (no germline_labels). "
                    "Check that the germline .pt sidecar file was generated and "
                    "germline_path is set in the config."
                )
            if self._total_calls % self._log_interval == 0:
                pct = self._fallback_calls / self._total_calls * 100
                logger.warning(
                    "GermlineMasking fallback rate: %d / %d (%.1f%%)",
                    self._fallback_calls, self._total_calls, pct,
                )
            probability_matrix = torch.full(input_ids.shape, self.mask_prob)
            probability_matrix[special_tokens_mask.bool()] = 0.0
            return torch.bernoulli(probability_matrix).bool()

        if self._total_calls % self._log_interval == 0 and self._fallback_calls > 0:
            pct = self._fallback_calls / self._total_calls * 100
            logger.info(
                "GermlineMasking fallback rate: %d / %d (%.1f%%)",
                self._fallback_calls, self._total_calls, pct,
            )

        weights = self.germline_weight + (
            self.mutated_weight - self.germline_weight
        ) * germline_labels.float()

        non_special = ~special_tokens_mask.bool()
        weights[~non_special] = 0.0

        mean_weight = weights[non_special].mean()
        if mean_weight > 0:
            probability_matrix = (self.mask_prob * weights / mean_weight).clamp(max=1.0)
        else:
            probability_matrix = torch.full(input_ids.shape, self.mask_prob)

        probability_matrix[~non_special] = 0.0
        return torch.bernoulli(probability_matrix).bool()
