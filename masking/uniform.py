"""Uniform random masking strategy (BERT-style baseline).

Selects positions uniformly at random, excluding special tokens.
This is the control experiment against which all other strategies
are compared.
"""

from __future__ import annotations

import torch

from masking.base import BaseMaskingStrategy, register_strategy


@register_strategy("uniform")
class UniformMasking(BaseMaskingStrategy):
    """Standard BERT-style uniform random masking.

    Randomly selects `mask_prob` fraction of non-special tokens.
    Of those selected:
      - mask_token_ratio are replaced with [MASK]
      - random_token_ratio are replaced with a random token
      - the rest are kept unchanged
    """

    def select_mask_positions(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        probability_matrix[special_tokens_mask.bool()] = 0.0
        return torch.bernoulli(probability_matrix).bool()
