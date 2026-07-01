"""Structure-aware masking with contact preservation.

Selects mask positions so that masked residues are spatially dispersed —
their k nearest 3D neighbors remain visible. This teaches the model to
USE spatial context (contact partners) when predicting a masked residue,
instead of forcing it to predict with contacts hidden.

Contrast with seed-and-grow clustering, which masks a residue and its
contacts together and produces representations that are less sensitive
to contact information (worse downstream contact probing).

Requires per-token Calpha coordinates in ``metadata["coords_ca"]`` (float
shape L x 3) or precomputed kNN indices in ``metadata["knn_indices"]``
(int shape L x k). Positions with zero coordinates or empty kNN rows are
excluded. Falls back to uniform masking when neither is available.

Algorithm:
  1. Build the set of maskable positions (non-special, has kNN data).
  2. Shuffle them randomly.
  3. Walk the shuffled list. For each position:
     a. Skip if already masked or marked as a contact partner of an
        already-masked residue.
     b. Otherwise mask it and mark its k nearest 3D neighbors as
        protected so they cannot be masked subsequently.
  4. Stop when the mask budget is reached.
  5. If protection over-constrains the problem and budget is not yet
     met, fill remaining slots uniformly from any unprotected maskable
     positions, then from any maskable positions as a last resort.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from masking.base import BaseMaskingStrategy, register_strategy

logger = logging.getLogger(__name__)


@register_strategy("structure")
class StructureMasking(BaseMaskingStrategy):
    """3D neighborhood masking via contact-preserving selection.

    For each masked residue, its k nearest 3D neighbors are kept
    visible. This lets the model learn to use spatial contacts as
    context for predicting masked tokens.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_prob: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
        k_neighbors: int = 5,
        min_seq_separation: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            mask_prob=mask_prob,
            mask_token_ratio=mask_token_ratio,
            random_token_ratio=random_token_ratio,
            **kwargs,
        )
        self.k_neighbors = k_neighbors
        # Drop neighbors with |i - j| <= min_seq_separation from the kNN list
        # before masking. With real Cα coordinates, the top-k by Euclidean
        # distance is dominated by sequence-adjacent residues (i±1, i±2, ...)
        # because the backbone is rigid; setting min_seq_separation > 0
        # restores the long-range-coupling signal the strategy was originally
        # meant to capture. Setting to 0 (default) preserves legacy behavior.
        self.min_seq_separation = min_seq_separation
        self._fallback_count = 0
        self._total_count = 0

    def _uniform_fallback(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._fallback_count += 1
        if self._fallback_count <= 10 or self._fallback_count % 1000 == 0:
            logger.warning(
                "Structure masking fallback to uniform (%d/%d sequences so far)",
                self._fallback_count, self._total_count,
            )
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        probability_matrix[special_tokens_mask.bool()] = 0.0
        return torch.bernoulli(probability_matrix).bool()

    def select_mask_positions(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self._total_count += 1

        if metadata is None:
            return self._uniform_fallback(input_ids, special_tokens_mask)

        knn_indices = metadata.get("knn_indices")
        coords = metadata.get("coords_ca")

        if knn_indices is not None:
            return self._mask_from_knn(input_ids, special_tokens_mask, knn_indices)
        elif coords is not None:
            return self._mask_from_coords(input_ids, special_tokens_mask, coords)
        else:
            return self._uniform_fallback(input_ids, special_tokens_mask)

    def _contact_preserving_select(
        self,
        seq_len: int,
        maskable_idx: torch.Tensor,
        knn_global: torch.Tensor,
        budget: int,
    ) -> torch.Tensor:
        """Pick mask positions that do not share 3D neighborhoods.

        Args:
            seq_len: full token sequence length.
            maskable_idx: [num_maskable] global indices of maskable positions.
            knn_global: [num_maskable, k] neighbor indices in global (token)
                space, one row per maskable position (same ordering).
            budget: target number of positions to mask.

        Returns:
            [seq_len] boolean mask.
        """
        mask = torch.zeros(seq_len, dtype=torch.bool)
        protected = torch.zeros(seq_len, dtype=torch.bool)

        num_maskable = maskable_idx.numel()
        if num_maskable == 0 or budget == 0:
            return mask

        perm = torch.randperm(num_maskable)
        shuffled_pos = maskable_idx[perm]
        shuffled_knn = knn_global[perm]

        k = min(self.k_neighbors, shuffled_knn.size(1))
        masked_count = 0

        for i in range(num_maskable):
            if masked_count >= budget:
                break
            pos = shuffled_pos[i].item()
            if mask[pos] or protected[pos]:
                continue
            mask[pos] = True
            masked_count += 1
            for j in range(k):
                neighbor = shuffled_knn[i, j].item()
                if 0 <= neighbor < seq_len:
                    protected[neighbor] = True
            # Don't let protection mask out the residue itself
            protected[pos] = True

        if masked_count >= budget:
            return mask

        # Over-protection prevented us from hitting budget. Relax:
        # fill from unprotected maskable positions first, then from any
        # remaining maskable positions.
        maskable_bool = torch.zeros(seq_len, dtype=torch.bool)
        maskable_bool[maskable_idx] = True

        unprotected = maskable_bool & (~mask) & (~protected)
        masked_count = self._fill_from(
            mask, unprotected, budget - masked_count, masked_count,
        )
        if masked_count < budget:
            remaining = maskable_bool & (~mask)
            self._fill_from(mask, remaining, budget - masked_count, masked_count)

        return mask

    @staticmethod
    def _fill_from(
        mask: torch.Tensor,
        candidates: torch.Tensor,
        n_more: int,
        current: int,
    ) -> int:
        """Mask up to n_more positions sampled uniformly from candidates.

        Mutates ``mask`` in place. Returns the new masked count.
        """
        if n_more <= 0:
            return current
        avail_idx = candidates.nonzero(as_tuple=True)[0]
        if avail_idx.numel() == 0:
            return current
        take = min(n_more, avail_idx.numel())
        perm = torch.randperm(avail_idx.numel())[:take]
        mask[avail_idx[perm]] = True
        return current + take

    def _mask_from_knn(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        knn_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Mask using precomputed kNN neighbor lists (from ESM2 contacts or IgFold)."""
        seq_len = input_ids.size(0)
        is_special = special_tokens_mask.bool()

        has_neighbors = knn_indices.sum(dim=-1) > 0
        maskable = has_neighbors & (~is_special)
        maskable_idx = maskable.nonzero(as_tuple=True)[0]
        num_maskable = maskable_idx.numel()

        if num_maskable == 0:
            return self._uniform_fallback(input_ids, special_tokens_mask)

        budget = max(1, round(self.mask_prob * num_maskable))

        knn_global = knn_indices[maskable_idx].long().clamp(0, seq_len - 1)

        # Filter out near-sequence neighbors. With real backbone coords the
        # top-k by Euclidean distance includes mostly i±1, i±2, ..., which is
        # the wrong inductive bias if we want the strategy to teach
        # long-range structural coupling. Setting the filtered slot to its
        # own row index makes the contact-preserving select treat that slot
        # as a self-protect (no-op) — see _contact_preserving_select.
        if self.min_seq_separation > 0:
            row_idx = maskable_idx.unsqueeze(1).expand_as(knn_global)
            sep = (knn_global - row_idx).abs()
            too_close = sep <= self.min_seq_separation
            # Replace too-close neighbors with the row's own index; the
            # downstream loop already protects the seed itself, so this
            # entries become harmless self-references.
            knn_global = torch.where(too_close, row_idx, knn_global)

        return self._contact_preserving_select(
            seq_len, maskable_idx, knn_global, budget,
        )

    def _mask_from_coords(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Mask using Calpha coordinates (legacy IgFold path)."""
        seq_len = input_ids.size(0)
        is_special = special_tokens_mask.bool()

        has_coords = (coords.abs().sum(dim=-1) > 1e-6) & (~is_special)
        maskable_idx = has_coords.nonzero(as_tuple=True)[0]
        num_maskable = maskable_idx.numel()

        if num_maskable == 0:
            return self._uniform_fallback(input_ids, special_tokens_mask)

        budget = max(1, round(self.mask_prob * num_maskable))

        if num_maskable == 1:
            mask = torch.zeros(seq_len, dtype=torch.bool)
            mask[maskable_idx[0]] = True
            return mask

        maskable_coords = coords[maskable_idx]
        dists = torch.cdist(
            maskable_coords.unsqueeze(0),
            maskable_coords.unsqueeze(0),
        ).squeeze(0)

        # Apply long-range filter at distance computation. We add +inf to
        # near-sequence pairs so they're never selected by topk. This is
        # equivalent to filtering the kNN result but cleaner.
        if self.min_seq_separation > 0:
            local_idx = torch.arange(num_maskable, device=dists.device)
            global_idx = maskable_idx
            ii = global_idx.unsqueeze(1)
            jj = global_idx.unsqueeze(0)
            sep = (ii - jj).abs()
            too_close = sep <= self.min_seq_separation
            dists = dists.masked_fill(too_close, float("inf"))

        k = min(self.k_neighbors, num_maskable - 1)
        _, knn_local = dists.topk(k + 1, dim=1, largest=False)
        knn_local = knn_local[:, 1:]
        knn_global = maskable_idx[knn_local]

        return self._contact_preserving_select(
            seq_len, maskable_idx, knn_global, budget,
        )
