"""SpanBERT-style contiguous span masking strategy.

Masks contiguous spans of residues rather than independent positions.
Span lengths are drawn from a truncated geometric distribution. Spans
never cross special tokens (chain separators), preserving chain
boundaries for future paired VH+VL sequences.

Default parameters (geometric_p=0.2, max_span_length=10) produce a
mean span length of ~5 tokens and are taken from the SpanBERT defaults,
which are reasonable for antibody variable domains of ~120 residues.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from masking.base import BaseMaskingStrategy, register_strategy


@register_strategy("span")
class SpanMasking(BaseMaskingStrategy):
    """Contiguous span masking with geometric span-length sampling.

    Iteratively samples spans until the mask budget (default 15% of
    non-special tokens) is filled. Each span starts at a randomly
    chosen unmasked position and extends rightward, stopping at
    special tokens or the sequence boundary.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_prob: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
        geometric_p: float = 0.2,
        max_span_length: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            mask_prob=mask_prob,
            mask_token_ratio=mask_token_ratio,
            random_token_ratio=random_token_ratio,
            **kwargs,
        )
        self.geometric_p = geometric_p
        self.max_span_length = max_span_length

    def _sample_span_length(self) -> int:
        """Sample from a geometric distribution truncated to [1, max_span_length]."""
        raw = int(torch.distributions.Geometric(probs=self.geometric_p).sample().item()) + 1
        return min(raw, self.max_span_length)

    def select_mask_positions(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        seq_len = input_ids.size(0)
        is_special = special_tokens_mask.bool()
        num_maskable = int((~is_special).sum().item())

        if num_maskable == 0:
            return torch.zeros(seq_len, dtype=torch.bool)

        budget = max(1, round(self.mask_prob * num_maskable))
        mask = torch.zeros(seq_len, dtype=torch.bool)
        masked_count = 0

        max_attempts = budget * 10
        attempts = 0

        while masked_count < budget and attempts < max_attempts:
            attempts += 1

            available = (~is_special) & (~mask)
            available_indices = available.nonzero(as_tuple=True)[0]
            if available_indices.numel() == 0:
                break

            start = available_indices[torch.randint(available_indices.numel(), (1,)).item()].item()
            span_len = self._sample_span_length()

            for offset in range(span_len):
                pos = start + offset
                if pos >= seq_len or is_special[pos]:
                    break
                if not mask[pos]:
                    mask[pos] = True
                    masked_count += 1
                    if masked_count >= budget:
                        break

        return mask
