"""Multispecific-aware masking for multi-module antibody sequences.

Implements three masking policies for paired/multispecific antibodies:
  Policy A: Module-isolated paratope masking — sample a module, mask
            primarily within its paratope residues
  Policy B: Cross-module consistency masking — aggressively mask the
            light chain (shared-chain proxy) for conditional infilling
  Policy C: VH-VL interface masking — bias toward cross-chain packing
            residues that carry pairing information

Each training step samples one policy from a categorical distribution
(configurable via policy_weights), then applies that policy's mask
selection. Falls back to uniform masking when required metadata
(module_ids, chain_type_ids) is missing.

References:
  - PairedAbNGS (>14M paired sequences) for VH-VL pairing data
  - LICHEN for shared light chain conditioning motivation
  - ImmunoMatch for VH-VL interface sensitivity
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from masking.base import BaseMaskingStrategy, register_strategy

logger = logging.getLogger(__name__)


@register_strategy("multispecific")
class MultispecificMasking(BaseMaskingStrategy):
    """Multi-policy masking for paired/multispecific antibody sequences.

    Requires metadata keys:
      - module_ids: LongTensor(seq_len,) with values 0 (global), 1 (mod1), 2 (mod2)
      - chain_type_ids: LongTensor(seq_len,) with values 0 (special), 1 (heavy), 2 (light)

    Optional metadata keys (enable specific policies):
      - paratope_labels: FloatTensor(seq_len,) in [0,1] — enables Policy A paratope bias
      - interface_labels: FloatTensor(seq_len,) in [0,1] — enables Policy C interface bias
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_prob: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
        # Policy A parameters
        paratope_weight: float = 6.0,
        non_paratope_weight: float = 1.0,
        # Policy B parameters
        shared_chain_boost: float = 3.0,
        # Policy C parameters
        interface_weight: float = 6.0,
        non_interface_weight: float = 1.0,
        # Policy mixing
        policy_weights: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            mask_prob=mask_prob,
            mask_token_ratio=mask_token_ratio,
            random_token_ratio=random_token_ratio,
            **kwargs,
        )
        self.paratope_weight = paratope_weight
        self.non_paratope_weight = non_paratope_weight
        self.shared_chain_boost = shared_chain_boost
        self.interface_weight = interface_weight
        self.non_interface_weight = non_interface_weight

        # Default: equal weight to all three policies
        pw = policy_weights or [1.0, 1.0, 1.0]
        self._policy_probs = torch.tensor(pw, dtype=torch.float)
        self._policy_probs = self._policy_probs / self._policy_probs.sum()

        self._total_calls = 0
        self._fallback_calls = 0
        self._policy_counts = [0, 0, 0]
        self._log_interval = 1000

    def select_mask_positions(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self._total_calls += 1

        module_ids = metadata.get("module_ids") if metadata else None
        chain_type_ids = metadata.get("chain_type_ids") if metadata else None

        if module_ids is None or chain_type_ids is None:
            return self._uniform_fallback(input_ids, special_tokens_mask)

        # Sample a policy
        policy_idx = torch.multinomial(self._policy_probs, 1).item()
        self._policy_counts[policy_idx] += 1

        if self._total_calls % self._log_interval == 0:
            logger.info(
                "MultispecificMasking policy counts: A=%d, B=%d, C=%d "
                "(fallback=%d / %d)",
                *self._policy_counts,
                self._fallback_calls, self._total_calls,
            )

        if policy_idx == 0:
            return self._policy_a_module_paratope(
                input_ids, special_tokens_mask, metadata,
                module_ids, chain_type_ids,
            )
        elif policy_idx == 1:
            return self._policy_b_shared_chain(
                input_ids, special_tokens_mask, metadata,
                module_ids, chain_type_ids,
            )
        else:
            return self._policy_c_vh_vl_interface(
                input_ids, special_tokens_mask, metadata,
                module_ids, chain_type_ids,
            )

    def _uniform_fallback(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Standard uniform masking when metadata is unavailable."""
        self._fallback_calls += 1
        if self._fallback_calls == 1:
            logger.warning(
                "MultispecificMasking: falling back to uniform "
                "(no module_ids/chain_type_ids). Check that paired data "
                "paths are configured correctly."
            )
        if self._total_calls % self._log_interval == 0:
            pct = self._fallback_calls / self._total_calls * 100
            logger.warning(
                "MultispecificMasking fallback rate: %d / %d (%.1f%%)",
                self._fallback_calls, self._total_calls, pct,
            )
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        probability_matrix[special_tokens_mask.bool()] = 0.0
        return torch.bernoulli(probability_matrix).bool()

    def _weighted_bernoulli(
        self,
        weights: torch.Tensor,
        non_special: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Shared weighted Bernoulli sampling with budget normalization.

        Same formula as interface.py and germline.py: normalize weights
        so that the expected number of masked positions equals mask_prob
        times the number of maskable positions.
        """
        weights[~non_special] = 0.0
        mean_weight = weights[non_special].mean()
        if mean_weight > 0:
            probability_matrix = (self.mask_prob * weights / mean_weight).clamp(max=1.0)
        else:
            probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        probability_matrix[~non_special] = 0.0
        return torch.bernoulli(probability_matrix).bool()

    def _policy_a_module_paratope(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor],
        module_ids: torch.Tensor,
        chain_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Policy A: Sample a module, mask paratope-biased within that module.

        1. Sample module k uniformly from available modules
        2. For positions in module k: weight by paratope_labels
        3. For positions in other modules: apply very low weight (10% leak)
        """
        paratope_labels = metadata.get("paratope_labels")
        non_special = ~special_tokens_mask.bool()

        # Find available modules (those with at least 1 non-special token)
        available_modules = []
        for mod_id in [1, 2]:
            if (non_special & (module_ids == mod_id)).any():
                available_modules.append(mod_id)

        if not available_modules:
            return self._uniform_fallback(input_ids, special_tokens_mask)

        # Sample a target module uniformly
        target_module = available_modules[
            torch.randint(len(available_modules), (1,)).item()
        ]

        seq_len = input_ids.size(0)
        weights = torch.ones(seq_len, dtype=torch.float)

        in_target = (module_ids == target_module) & non_special
        not_in_target = (~(module_ids == target_module)) & non_special

        if paratope_labels is not None:
            # Within target module: paratope-biased weights
            weights[in_target] = self.non_paratope_weight + (
                self.paratope_weight - self.non_paratope_weight
            ) * paratope_labels[in_target].float()
        else:
            # No paratope data: uniform weight within target module
            weights[in_target] = self.paratope_weight

        # Other modules: very low weight (10% leak prevents representation collapse)
        weights[not_in_target] = self.non_paratope_weight * 0.1

        return self._weighted_bernoulli(weights, non_special, input_ids)

    def _policy_b_shared_chain(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor],
        module_ids: torch.Tensor,
        chain_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Policy B: Aggressively mask the light chain (shared chain proxy).

        For bispecific antibodies, the light chain is often shared.
        This policy masks light chain positions more heavily to train
        the model for shared-chain conditional infilling.

        For monospecific paired data, this trains VL prediction
        conditioned on VH context.
        """
        non_special = ~special_tokens_mask.bool()
        is_light = (chain_type_ids == 2) & non_special
        is_heavy = (chain_type_ids == 1) & non_special

        if not is_light.any():
            return self._uniform_fallback(input_ids, special_tokens_mask)

        seq_len = input_ids.size(0)
        weights = torch.ones(seq_len, dtype=torch.float)

        # Light chain gets boosted weight
        weights[is_light] = self.shared_chain_boost
        # Heavy chain gets baseline weight
        weights[is_heavy] = 1.0

        return self._weighted_bernoulli(weights, non_special, input_ids)

    def _policy_c_vh_vl_interface(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor],
        module_ids: torch.Tensor,
        chain_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Policy C: Mask VH-VL interface residues.

        Uses precomputed interface_labels from cross-chain contact analysis
        or known interface positions from IMGT numbering.
        Falls back to uniform within the module if no interface labels.
        """
        interface_labels = metadata.get("interface_labels")

        if interface_labels is None:
            return self._uniform_fallback(input_ids, special_tokens_mask)

        non_special = ~special_tokens_mask.bool()

        weights = self.non_interface_weight + (
            self.interface_weight - self.non_interface_weight
        ) * interface_labels.float()

        return self._weighted_bernoulli(weights, non_special, input_ids)
