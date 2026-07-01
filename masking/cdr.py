"""CDR-focused masking strategy.

Biases the MLM mask budget toward CDR regions using per-region weights.
Falls back to uniform masking when CDR annotations are unavailable.

Region labels in cdr_mask: 0=framework, 1=CDR1, 2=CDR2, 3=CDR3.
Default weights (framework=1, CDR1=3, CDR2=3, CDR3=6) concentrate
~60-70% of masked positions in CDRs despite CDRs being only ~25% of
the sequence, matching the biological prior that functional diversity
is concentrated in hypervariable loops.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from masking.base import BaseMaskingStrategy, register_strategy


@register_strategy("cdr")
class CDRMasking(BaseMaskingStrategy):
    """CDR-focused masking with per-region weight control.

    Allocates the mask budget (default 15%) disproportionately toward CDR
    positions using weighted Bernoulli sampling. Each position's masking
    probability is proportional to its region weight, normalized so that
    the expected total masking rate matches `mask_prob`.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_prob: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
        framework_weight: float = 1.0,
        cdr1_weight: float = 3.0,
        cdr2_weight: float = 3.0,
        cdr3_weight: float = 6.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            mask_prob=mask_prob,
            mask_token_ratio=mask_token_ratio,
            random_token_ratio=random_token_ratio,
            **kwargs,
        )
        self._region_weights = torch.tensor(
            [framework_weight, cdr1_weight, cdr2_weight, cdr3_weight],
            dtype=torch.float,
        )

    def select_mask_positions(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        cdr_mask = metadata.get("cdr_mask") if metadata else None

        if cdr_mask is None:
            probability_matrix = torch.full(input_ids.shape, self.mask_prob)
            probability_matrix[special_tokens_mask.bool()] = 0.0
            return torch.bernoulli(probability_matrix).bool()

        weights = self._region_weights[cdr_mask.long()]

        non_special = ~special_tokens_mask.bool()
        weights[~non_special] = 0.0

        mean_weight = weights[non_special].mean()
        if mean_weight > 0:
            probability_matrix = (self.mask_prob * weights / mean_weight).clamp(max=1.0)
        else:
            probability_matrix = torch.full(input_ids.shape, self.mask_prob)

        probability_matrix[~non_special] = 0.0
        return torch.bernoulli(probability_matrix).bool()
