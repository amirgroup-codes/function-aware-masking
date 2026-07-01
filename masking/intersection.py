"""Intersection masking: stack two biological priors multiplicatively.

A single biological prior identifies "important" positions (paratope,
mutated, CDR, etc.). The intersection of two priors identifies positions
that are important *under both* — e.g., paratope ∩ germline yields
"mutated paratope residues," which are the positions where somatic
hypermutation has occurred at the antigen interface. These are the
highest-signal residues for binding-affinity learning and a strict
super-prior over either component alone.

Generalizes naturally to 2+ priors by elementwise product of weight
vectors. Each prior contributes a per-residue weight in [a, b], and the
final per-residue weight is the product, so a position is heavily masked
only if *every* prior agrees it's interesting.

Falls back to uniform when any required prior is missing.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from masking.base import BaseMaskingStrategy, register_strategy

logger = logging.getLogger(__name__)


# Map prior name -> (metadata key, default high_weight, default low_weight)
PRIOR_SPECS: dict[str, tuple[str, float, float]] = {
    "paratope": ("paratope_labels", 6.0, 1.0),
    "germline": ("germline_labels", 6.0, 1.0),
    # cdr is encoded via the cdr_mask metadata, where 1=CDR, 0=framework
    "cdr": ("cdr_mask", 3.0, 1.0),
}


@register_strategy("intersection")
class IntersectionMasking(BaseMaskingStrategy):
    """Multiplicatively combine two-or-more per-residue biological priors.

    Each prior provides a per-residue weight; the final weight is the
    elementwise product, so a position lands in the mask budget only if
    *all* priors weight it heavily.

    Args:
        priors: List of prior names. Each must be in PRIOR_SPECS.
                Default is ["paratope", "germline"] = mutated paratope.
        prior_weights: Optional per-prior (high, low) weight overrides.
                       Maps prior_name -> [high, low]. Falls back to PRIOR_SPECS
                       defaults when not provided.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_prob: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
        priors: list[str] | None = None,
        prior_weights: dict[str, list[float]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            mask_prob=mask_prob,
            mask_token_ratio=mask_token_ratio,
            random_token_ratio=random_token_ratio,
            **kwargs,
        )
        if priors is None:
            priors = ["paratope", "germline"]
        for p in priors:
            if p not in PRIOR_SPECS:
                raise ValueError(f"Unknown prior {p!r}; valid: {list(PRIOR_SPECS)}")
        self.priors = list(priors)

        # Resolve (high, low) weight tuples per prior
        prior_weights = prior_weights or {}
        self._weight_specs: dict[str, tuple[float, float]] = {}
        for p in self.priors:
            if p in prior_weights:
                hi, lo = prior_weights[p]
            else:
                _, hi, lo = PRIOR_SPECS[p]
            self._weight_specs[p] = (float(hi), float(lo))

        self._total_calls = 0
        self._fallback_calls = 0
        self._log_interval = 1000

    def _per_residue_weights(
        self,
        prior: str,
        metadata: dict[str, torch.Tensor],
        seq_len: int,
    ) -> torch.Tensor | None:
        """Return [seq_len] weight vector for one prior, or None if missing."""
        meta_key, _, _ = PRIOR_SPECS[prior]
        labels = metadata.get(meta_key)
        if labels is None:
            return None
        hi, lo = self._weight_specs[prior]
        labels = labels.float()
        # All priors use the same convention: 1.0 = "interesting" -> hi weight,
        # 0.0 = "not" -> lo weight, with smooth interpolation in between.
        return lo + (hi - lo) * labels[:seq_len]

    def select_mask_positions(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self._total_calls += 1
        seq_len = input_ids.size(0)

        if metadata is None:
            return self._uniform_fallback(input_ids, special_tokens_mask)

        weight_vecs: list[torch.Tensor] = []
        for p in self.priors:
            w = self._per_residue_weights(p, metadata, seq_len)
            if w is None:
                self._fallback_calls += 1
                if self._fallback_calls == 1 or self._fallback_calls % 1000 == 0:
                    logger.warning(
                        "IntersectionMasking falling back to uniform "
                        "(missing prior=%s, %d/%d so far)",
                        p, self._fallback_calls, self._total_calls,
                    )
                return self._uniform_fallback(input_ids, special_tokens_mask)
            weight_vecs.append(w)

        # Elementwise product across priors -> intersection weight
        weights = weight_vecs[0]
        for w in weight_vecs[1:]:
            weights = weights * w

        non_special = ~special_tokens_mask.bool()
        # Pad-or-truncate weight vector to match input length
        if weights.numel() < seq_len:
            pad = torch.zeros(
                seq_len - weights.numel(),
                device=weights.device,
                dtype=weights.dtype,
            )
            weights = torch.cat([weights, pad])
        elif weights.numel() > seq_len:
            weights = weights[:seq_len]

        weights = weights.clone()
        weights[~non_special] = 0.0

        mean_weight = weights[non_special].mean()
        if mean_weight > 0:
            probability_matrix = (
                self.mask_prob * weights / mean_weight
            ).clamp(max=1.0)
        else:
            probability_matrix = torch.full(
                input_ids.shape, self.mask_prob, dtype=torch.float,
            )

        probability_matrix[~non_special] = 0.0
        return torch.bernoulli(probability_matrix).bool()

    def _uniform_fallback(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        probability_matrix[special_tokens_mask.bool()] = 0.0
        return torch.bernoulli(probability_matrix).bool()
