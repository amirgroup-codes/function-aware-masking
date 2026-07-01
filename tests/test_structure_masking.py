"""Tests for structure-aware 3D neighborhood masking and collator fixes.

Uses synthetic Calpha coordinates (no IgFold dependency) to validate
budget correctness, spatial clustering, fallback behavior, collator
dtype/padding handling, and edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from masking.structure import StructureMasking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tokenizer(vocab_size: int = 30, mask_token_id: int = 4) -> MagicMock:
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.mask_token_id = mask_token_id
    tok.pad_token_id = 0
    tok.unk_token_id = 1
    _aa_to_id = {aa: 5 + i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    tok.convert_tokens_to_ids.side_effect = lambda c: _aa_to_id.get(c, 1)
    return tok


def _make_linear_coords(seq_len: int, special_positions: list[int] | None = None) -> torch.Tensor:
    """Create coords where residues are spaced 3.8 A apart along the x-axis.

    Special positions get [0,0,0] coords (mimicking CLS/SEP/PAD).
    """
    if special_positions is None:
        special_positions = [0, seq_len - 1]
    coords = torch.zeros(seq_len, 3)
    aa_idx = 0
    for i in range(seq_len):
        if i in special_positions:
            continue
        coords[i, 0] = aa_idx * 3.8
        aa_idx += 1
    return coords


def _make_clustered_coords(seq_len: int, special_positions: list[int] | None = None) -> torch.Tensor:
    """Create coords with two distinct 3D clusters far apart.

    First half of non-special positions near origin, second half ~100 A away.
    """
    if special_positions is None:
        special_positions = [0, seq_len - 1]
    coords = torch.zeros(seq_len, 3)
    non_special = [i for i in range(seq_len) if i not in special_positions]
    mid = len(non_special) // 2
    for j, pos in enumerate(non_special):
        if j < mid:
            coords[pos] = torch.tensor([j * 1.0, 0.0, 0.0])
        else:
            coords[pos] = torch.tensor([100.0 + (j - mid) * 1.0, 0.0, 0.0])
    return coords


def _make_inputs(
    seq_len: int,
    special_positions: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.arange(5, 5 + seq_len, dtype=torch.long)
    special_tokens_mask = torch.zeros(seq_len, dtype=torch.long)
    if special_positions is None:
        special_positions = [0, seq_len - 1]
    for pos in special_positions:
        special_tokens_mask[pos] = 1
    return input_ids, special_tokens_mask


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestBudgetCorrectness:
    def test_budget_approximate(self) -> None:
        torch.manual_seed(0)
        strategy = StructureMasking(_make_tokenizer(), mask_prob=0.15, k_neighbors=32)
        input_ids, special_mask = _make_inputs(120)
        coords = _make_linear_coords(120)
        num_maskable = int((~special_mask.bool()).sum().item())
        expected = round(0.15 * num_maskable)

        counts: list[int] = []
        for seed in range(30):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask, metadata={"coords_ca": coords}
            )
            counts.append(int(mask.sum().item()))

        mean_count = sum(counts) / len(counts)
        assert abs(mean_count - expected) / expected < 0.15, (
            f"Mean masked {mean_count:.1f} deviates >15% from expected {expected}"
        )


class TestSpatialDispersion:
    """Contact-preserving masking should spread masks in 3D space.

    Masked residues' k nearest neighbors are protected from being
    masked, so masked positions are on average farther apart than a
    uniform random sample of the same size.
    """

    def test_masked_farther_than_random(self) -> None:
        torch.manual_seed(42)
        strategy = StructureMasking(_make_tokenizer(), mask_prob=0.15, k_neighbors=5)
        input_ids, special_mask = _make_inputs(100)
        coords = _make_linear_coords(100)

        # Average across multiple seeds to reduce variance
        mean_masked_over_seeds = []
        mean_rand_over_seeds = []
        for seed in range(20):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask, metadata={"coords_ca": coords}
            )
            masked_coords = coords[mask]
            if masked_coords.size(0) < 2:
                continue
            masked_dists = torch.cdist(
                masked_coords.unsqueeze(0), masked_coords.unsqueeze(0)
            ).squeeze(0)
            mean_masked_over_seeds.append(
                masked_dists.sum().item()
                / (masked_dists.numel() - masked_dists.size(0))
            )

            non_special = ~special_mask.bool()
            all_coords = coords[non_special]
            n_sample = int(mask.sum().item())
            torch.manual_seed(1000 + seed)
            rand_idx = torch.randperm(all_coords.size(0))[:n_sample]
            rand_coords = all_coords[rand_idx]
            rand_dists = torch.cdist(
                rand_coords.unsqueeze(0), rand_coords.unsqueeze(0)
            ).squeeze(0)
            mean_rand_over_seeds.append(
                rand_dists.sum().item()
                / (rand_dists.numel() - rand_dists.size(0))
            )

        mean_masked = sum(mean_masked_over_seeds) / len(mean_masked_over_seeds)
        mean_rand = sum(mean_rand_over_seeds) / len(mean_rand_over_seeds)

        assert mean_masked > mean_rand, (
            f"Masked mean dist {mean_masked:.1f} should be > random {mean_rand:.1f}"
        )

    def test_two_clusters_spans_both(self) -> None:
        """With two far-apart clusters, masks should spread across both.

        The inverted (contact-preserving) strategy picks seeds
        uniformly at random but protects 3D neighbors, so after
        masking a residue in cluster A it is still possible to mask
        residues in cluster B (they are not spatial neighbors).
        """
        seq_len = 60
        strategy = StructureMasking(_make_tokenizer(), mask_prob=0.3, k_neighbors=5)
        input_ids, special_mask = _make_inputs(seq_len)
        coords = _make_clustered_coords(seq_len)

        spans_both = 0
        trials = 20
        for seed in range(trials):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask, metadata={"coords_ca": coords}
            )
            masked_coords = coords[mask]
            if masked_coords.size(0) < 2:
                continue
            x_values = masked_coords[:, 0]
            has_near = (x_values < 50.0).any().item()
            has_far = (x_values > 50.0).any().item()
            if has_near and has_far:
                spans_both += 1

        # With 30% budget and contact preservation, most trials should
        # naturally span both clusters.
        assert spans_both >= trials // 2, (
            f"Only {spans_both}/{trials} trials spanned both clusters"
        )


class TestSpecialTokensNeverMasked:
    def test_special_positions_untouched(self) -> None:
        torch.manual_seed(0)
        strategy = StructureMasking(_make_tokenizer(), mask_prob=0.15, k_neighbors=16)
        input_ids, special_mask = _make_inputs(80)
        coords = _make_linear_coords(80)

        for seed in range(20):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask, metadata={"coords_ca": coords}
            )
            assert not mask[special_mask.bool()].any(), f"Special token masked (seed={seed})"


class TestFallbackToUniform:
    def test_no_coords_in_metadata(self) -> None:
        torch.manual_seed(42)
        strategy = StructureMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(80)

        mask = strategy.select_mask_positions(input_ids, special_mask, metadata={})
        assert mask.sum().item() > 0, "Should fall back to uniform and mask something"
        assert not mask[special_mask.bool()].any()

    def test_none_metadata(self) -> None:
        torch.manual_seed(42)
        strategy = StructureMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(80)

        mask = strategy.select_mask_positions(input_ids, special_mask, metadata=None)
        assert mask.sum().item() > 0

    def test_all_zero_coords_falls_back(self) -> None:
        torch.manual_seed(42)
        strategy = StructureMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(80)
        coords = torch.zeros(80, 3)

        mask = strategy.select_mask_positions(
            input_ids, special_mask, metadata={"coords_ca": coords}
        )
        assert mask.sum().item() > 0, "All-zero coords should trigger uniform fallback"


class TestZeroCoordsExcluded:
    def test_zero_coord_positions_not_masked(self) -> None:
        torch.manual_seed(0)
        seq_len = 20
        input_ids, special_mask = _make_inputs(seq_len, special_positions=[0, seq_len - 1])
        coords = _make_linear_coords(seq_len)
        coords[5] = torch.zeros(3)
        coords[10] = torch.zeros(3)

        strategy = StructureMasking(_make_tokenizer(), mask_prob=0.5, k_neighbors=8)
        mask = strategy.select_mask_positions(
            input_ids, special_mask, metadata={"coords_ca": coords}
        )
        assert not mask[5].item(), "Position 5 (zero coords) should not be masked"
        assert not mask[10].item(), "Position 10 (zero coords) should not be masked"


class TestEdgeCases:
    def test_all_special_tokens(self) -> None:
        strategy = StructureMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids = torch.arange(5, 15, dtype=torch.long)
        special_mask = torch.ones(10, dtype=torch.long)
        coords = torch.zeros(10, 3)

        mask = strategy.select_mask_positions(
            input_ids, special_mask, metadata={"coords_ca": coords}
        )
        assert mask.sum().item() == 0

    def test_single_maskable_token(self) -> None:
        strategy = StructureMasking(_make_tokenizer(), mask_prob=1.0, k_neighbors=8)
        input_ids = torch.tensor([1, 10, 2], dtype=torch.long)
        special_mask = torch.tensor([1, 0, 1], dtype=torch.long)
        coords = torch.tensor([[0.0, 0.0, 0.0], [5.0, 3.0, 1.0], [0.0, 0.0, 0.0]])

        mask = strategy.select_mask_positions(
            input_ids, special_mask, metadata={"coords_ca": coords}
        )
        assert mask.sum().item() == 1
        assert mask[1].item() is True

    def test_two_maskable_tokens(self) -> None:
        strategy = StructureMasking(_make_tokenizer(), mask_prob=0.5, k_neighbors=8)
        input_ids = torch.tensor([1, 10, 11, 2], dtype=torch.long)
        special_mask = torch.tensor([1, 0, 0, 1], dtype=torch.long)
        coords = torch.tensor([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        ])

        mask = strategy.select_mask_positions(
            input_ids, special_mask, metadata={"coords_ca": coords}
        )
        assert 0 < mask.sum().item() <= 2
        assert not mask[0].item()
        assert not mask[3].item()

    def test_k_larger_than_maskable(self) -> None:
        torch.manual_seed(0)
        strategy = StructureMasking(_make_tokenizer(), mask_prob=0.5, k_neighbors=100)
        input_ids, special_mask = _make_inputs(10)
        coords = _make_linear_coords(10)

        mask = strategy.select_mask_positions(
            input_ids, special_mask, metadata={"coords_ca": coords}
        )
        assert mask.sum().item() > 0
        assert not mask[special_mask.bool()].any()


class TestDeterminism:
    def test_same_seed_same_mask(self) -> None:
        strategy = StructureMasking(_make_tokenizer(), mask_prob=0.15, k_neighbors=16)
        input_ids, special_mask = _make_inputs(100)
        coords = _make_linear_coords(100)

        torch.manual_seed(99)
        mask_a = strategy.select_mask_positions(
            input_ids, special_mask, metadata={"coords_ca": coords}
        )
        torch.manual_seed(99)
        mask_b = strategy.select_mask_positions(
            input_ids, special_mask, metadata={"coords_ca": coords}
        )
        assert torch.equal(mask_a, mask_b)


class TestRegistration:
    def test_strategy_registered(self) -> None:
        from masking.base import _STRATEGY_REGISTRY
        assert "structure" in _STRATEGY_REGISTRY

    def test_get_strategy_returns_structure(self) -> None:
        from masking import get_strategy
        strategy = get_strategy("structure", tokenizer=_make_tokenizer(), k_neighbors=16)
        assert isinstance(strategy, StructureMasking)
        assert strategy.k_neighbors == 16


# ---------------------------------------------------------------------------
# Collator tests for float dtype and 2D padding
# ---------------------------------------------------------------------------

class TestCollatorFloatAndMultiDim:
    """Verify the collator handles float 2D metadata correctly."""

    def test_float_dtype_preserved(self) -> None:
        from masking.collator import _to_tensor
        int_val = [1, 2, 3]
        float_val = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        int_t = _to_tensor(int_val)
        float_t = _to_tensor(float_val)

        assert int_t.dtype == torch.long
        assert float_t.dtype == torch.float32
        assert int_t.shape == (3,)
        assert float_t.shape == (2, 3)

    def test_collator_pads_2d_metadata(self) -> None:
        from masking.collator import MLMDataCollator
        from masking.uniform import UniformMasking

        tok = _make_tokenizer()
        strategy = UniformMasking(tok, mask_prob=0.15)
        collator = MLMDataCollator(
            tokenizer=tok, strategy=strategy,
            pad_to_multiple_of=None, return_metadata=True,
        )

        ex1 = {
            "input_ids": [1, 10, 11, 2],
            "attention_mask": [1, 1, 1, 1],
            "special_tokens_mask": [1, 0, 0, 1],
            "coords_ca": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]],
        }
        ex2 = {
            "input_ids": [1, 10, 11, 12, 13, 2],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "special_tokens_mask": [1, 0, 0, 0, 0, 1],
            "coords_ca": [
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            ],
        }

        batch = collator([ex1, ex2])
        assert "coords_ca" in batch
        assert batch["coords_ca"].shape == (2, 6, 3)
        assert batch["coords_ca"].dtype == torch.float32
        assert (batch["coords_ca"][0, 4:, :] == 0.0).all(), "Padding should be zeros"

    def test_collator_preserves_1d_long_metadata(self) -> None:
        """Existing cdr_mask (1D long) still works after the collator changes."""
        from masking.collator import MLMDataCollator
        from masking.uniform import UniformMasking

        tok = _make_tokenizer()
        strategy = UniformMasking(tok, mask_prob=0.15)
        collator = MLMDataCollator(
            tokenizer=tok, strategy=strategy,
            pad_to_multiple_of=None, return_metadata=True,
        )

        ex1 = {
            "input_ids": [1, 10, 11, 2],
            "attention_mask": [1, 1, 1, 1],
            "special_tokens_mask": [1, 0, 0, 1],
            "cdr_mask": [0, 1, 2, 0],
        }
        ex2 = {
            "input_ids": [1, 10, 11, 12, 2],
            "attention_mask": [1, 1, 1, 1, 1],
            "special_tokens_mask": [1, 0, 0, 0, 1],
            "cdr_mask": [0, 1, 2, 3, 0],
        }

        batch = collator([ex1, ex2])
        assert "cdr_mask" in batch
        assert batch["cdr_mask"].dtype == torch.long
        assert batch["cdr_mask"].shape == (2, 5)
