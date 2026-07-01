"""Tests for interface and paratope masking strategy.

Uses synthetic paratope labels (no PDB dependency) to validate budget
correctness, paratope bias, soft label support, fallback behavior,
edge cases, registration, and determinism.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from masking.interface import InterfaceMasking


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


def _make_paratope_labels(
    seq_len: int,
    paratope_positions: list[int],
    special_positions: list[int] | None = None,
) -> torch.Tensor:
    """Create hard paratope labels (0/1) with 0.0 at special positions."""
    if special_positions is None:
        special_positions = [0, seq_len - 1]
    labels = torch.zeros(seq_len, dtype=torch.float)
    for pos in paratope_positions:
        if pos not in special_positions:
            labels[pos] = 1.0
    return labels


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestBudgetCorrectness:
    def test_budget_approximate(self) -> None:
        torch.manual_seed(0)
        strategy = InterfaceMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(120)
        paratope_labels = _make_paratope_labels(120, list(range(20, 40)))
        num_maskable = int((~special_mask.bool()).sum().item())
        expected = round(0.15 * num_maskable)

        counts: list[int] = []
        for seed in range(30):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask,
                metadata={"paratope_labels": paratope_labels},
            )
            counts.append(int(mask.sum().item()))

        mean_count = sum(counts) / len(counts)
        assert abs(mean_count - expected) / expected < 0.15, (
            f"Mean masked {mean_count:.1f} deviates >15% from expected {expected}"
        )


class TestParatopeBias:
    """Paratope positions should be masked significantly more often."""

    def test_paratope_masked_more_than_non_paratope(self) -> None:
        seq_len = 100
        paratope_positions = list(range(10, 25))
        strategy = InterfaceMasking(
            _make_tokenizer(), mask_prob=0.15,
            paratope_weight=6.0, non_paratope_weight=1.0,
        )
        input_ids, special_mask = _make_inputs(seq_len)
        paratope_labels = _make_paratope_labels(seq_len, paratope_positions)

        paratope_mask_counts = torch.zeros(seq_len)
        n_trials = 100
        for seed in range(n_trials):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask,
                metadata={"paratope_labels": paratope_labels},
            )
            paratope_mask_counts += mask.float()

        non_special = ~special_mask.bool()
        is_paratope = paratope_labels.bool() & non_special
        is_non_paratope = (~paratope_labels.bool()) & non_special

        paratope_rate = paratope_mask_counts[is_paratope].mean().item() / n_trials
        non_paratope_rate = paratope_mask_counts[is_non_paratope].mean().item() / n_trials

        assert paratope_rate > non_paratope_rate * 2, (
            f"Paratope rate {paratope_rate:.3f} should be >2x non-paratope "
            f"rate {non_paratope_rate:.3f}"
        )


class TestSoftLabels:
    """Positions with higher confidence should be masked more often."""

    def test_higher_confidence_masked_more(self) -> None:
        seq_len = 80
        strategy = InterfaceMasking(
            _make_tokenizer(), mask_prob=0.15,
            paratope_weight=6.0, non_paratope_weight=1.0,
        )
        input_ids, special_mask = _make_inputs(seq_len)

        # Create soft labels: positions 10-19 get c=0.2, positions 30-39 get c=0.8
        soft_labels = torch.zeros(seq_len, dtype=torch.float)
        low_conf_positions = list(range(10, 20))
        high_conf_positions = list(range(30, 40))
        for pos in low_conf_positions:
            soft_labels[pos] = 0.2
        for pos in high_conf_positions:
            soft_labels[pos] = 0.8

        low_counts = torch.zeros(seq_len)
        high_counts = torch.zeros(seq_len)
        n_trials = 100
        for seed in range(n_trials):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask,
                metadata={"paratope_labels": soft_labels},
            )
            low_counts += mask.float()
            high_counts += mask.float()

        low_rate = low_counts[low_conf_positions].mean().item() / n_trials
        high_rate = high_counts[high_conf_positions].mean().item() / n_trials

        assert high_rate > low_rate, (
            f"High confidence rate {high_rate:.3f} should exceed "
            f"low confidence rate {low_rate:.3f}"
        )


class TestSpecialTokensNeverMasked:
    def test_special_positions_untouched(self) -> None:
        torch.manual_seed(0)
        strategy = InterfaceMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(80)
        paratope_labels = _make_paratope_labels(80, list(range(10, 30)))

        for seed in range(20):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask,
                metadata={"paratope_labels": paratope_labels},
            )
            assert not mask[special_mask.bool()].any(), (
                f"Special token masked (seed={seed})"
            )


class TestFallbackToUniform:
    def test_no_paratope_in_metadata(self) -> None:
        torch.manual_seed(42)
        strategy = InterfaceMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(80)

        mask = strategy.select_mask_positions(input_ids, special_mask, metadata={})
        assert mask.sum().item() > 0, "Should fall back to uniform and mask something"
        assert not mask[special_mask.bool()].any()

    def test_none_metadata(self) -> None:
        torch.manual_seed(42)
        strategy = InterfaceMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(80)

        mask = strategy.select_mask_positions(input_ids, special_mask, metadata=None)
        assert mask.sum().item() > 0

    def test_all_zero_labels_degrades_to_uniform(self) -> None:
        """All-zero paratope labels should produce uniform-like masking."""
        torch.manual_seed(42)
        strategy = InterfaceMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(80)
        paratope_labels = torch.zeros(80, dtype=torch.float)

        mask = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"paratope_labels": paratope_labels},
        )
        assert mask.sum().item() > 0
        assert not mask[special_mask.bool()].any()


class TestEdgeCases:
    def test_all_special_tokens(self) -> None:
        strategy = InterfaceMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids = torch.arange(5, 15, dtype=torch.long)
        special_mask = torch.ones(10, dtype=torch.long)
        paratope_labels = torch.zeros(10, dtype=torch.float)

        mask = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"paratope_labels": paratope_labels},
        )
        assert mask.sum().item() == 0

    def test_single_maskable_token(self) -> None:
        strategy = InterfaceMasking(_make_tokenizer(), mask_prob=1.0)
        input_ids = torch.tensor([1, 10, 2], dtype=torch.long)
        special_mask = torch.tensor([1, 0, 1], dtype=torch.long)
        paratope_labels = torch.tensor([0.0, 1.0, 0.0])

        mask = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"paratope_labels": paratope_labels},
        )
        assert mask.sum().item() >= 0
        assert not mask[0].item()
        assert not mask[2].item()

    def test_all_paratope(self) -> None:
        """All non-special positions are paratope — should work like uniform."""
        torch.manual_seed(42)
        seq_len = 50
        strategy = InterfaceMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(seq_len)
        paratope_labels = torch.ones(seq_len, dtype=torch.float)
        paratope_labels[0] = 0.0
        paratope_labels[seq_len - 1] = 0.0

        mask = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"paratope_labels": paratope_labels},
        )
        assert mask.sum().item() > 0
        assert not mask[special_mask.bool()].any()

    def test_single_paratope_residue(self) -> None:
        """Only one position has label 1.0, rest are 0.0."""
        torch.manual_seed(0)
        seq_len = 50
        strategy = InterfaceMasking(
            _make_tokenizer(), mask_prob=0.15,
            paratope_weight=6.0, non_paratope_weight=1.0,
        )
        input_ids, special_mask = _make_inputs(seq_len)
        paratope_labels = torch.zeros(seq_len, dtype=torch.float)
        paratope_labels[25] = 1.0

        mask = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"paratope_labels": paratope_labels},
        )
        assert mask.sum().item() > 0
        assert not mask[special_mask.bool()].any()


class TestRegistration:
    def test_strategy_registered(self) -> None:
        from masking.base import _STRATEGY_REGISTRY
        assert "interface" in _STRATEGY_REGISTRY

    def test_get_strategy_returns_interface(self) -> None:
        from masking import get_strategy
        strategy = get_strategy(
            "interface", tokenizer=_make_tokenizer(),
            paratope_weight=8.0, non_paratope_weight=0.5,
        )
        assert isinstance(strategy, InterfaceMasking)
        assert strategy.paratope_weight == 8.0
        assert strategy.non_paratope_weight == 0.5


class TestDeterminism:
    def test_same_seed_same_mask(self) -> None:
        strategy = InterfaceMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(100)
        paratope_labels = _make_paratope_labels(100, list(range(20, 40)))

        torch.manual_seed(99)
        mask_a = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"paratope_labels": paratope_labels},
        )
        torch.manual_seed(99)
        mask_b = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"paratope_labels": paratope_labels},
        )
        assert torch.equal(mask_a, mask_b)
