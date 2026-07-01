"""Tests for germline and SHM mutation masking strategy.

Uses synthetic germline labels (no external data dependency) to validate
budget correctness, mutation bias, CDR3 intermediate handling, soft labels,
fallback behavior, edge cases, registration, and determinism.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from masking.germline import GermlineMasking


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


def _make_germline_labels(
    seq_len: int,
    mutated_positions: list[int],
    cdr3_positions: list[int] | None = None,
    special_positions: list[int] | None = None,
    cdr3_label: float = 0.5,
) -> torch.Tensor:
    """Create germline labels: 0.0=germline, 1.0=mutated, cdr3_label for CDR3."""
    if special_positions is None:
        special_positions = [0, seq_len - 1]
    labels = torch.zeros(seq_len, dtype=torch.float)
    for pos in mutated_positions:
        if pos not in special_positions:
            labels[pos] = 1.0
    if cdr3_positions:
        for pos in cdr3_positions:
            if pos not in special_positions:
                labels[pos] = cdr3_label
    return labels


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestBudgetCorrectness:
    def test_budget_approximate(self) -> None:
        torch.manual_seed(0)
        strategy = GermlineMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(120)
        germline_labels = _make_germline_labels(
            120, mutated_positions=list(range(20, 35)),
            cdr3_positions=list(range(80, 95)),
        )
        num_maskable = int((~special_mask.bool()).sum().item())
        expected = round(0.15 * num_maskable)

        counts: list[int] = []
        for seed in range(30):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask,
                metadata={"germline_labels": germline_labels},
            )
            counts.append(int(mask.sum().item()))

        mean_count = sum(counts) / len(counts)
        assert abs(mean_count - expected) / expected < 0.15, (
            f"Mean masked {mean_count:.1f} deviates >15% from expected {expected}"
        )


class TestMutationBias:
    """Mutated positions should be masked significantly more often."""

    def test_mutated_masked_more_than_germline(self) -> None:
        seq_len = 100
        mutated_positions = list(range(10, 25))
        strategy = GermlineMasking(
            _make_tokenizer(), mask_prob=0.15,
            mutated_weight=6.0, germline_weight=1.0,
        )
        input_ids, special_mask = _make_inputs(seq_len)
        germline_labels = _make_germline_labels(seq_len, mutated_positions)

        mask_counts = torch.zeros(seq_len)
        n_trials = 100
        for seed in range(n_trials):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask,
                metadata={"germline_labels": germline_labels},
            )
            mask_counts += mask.float()

        non_special = ~special_mask.bool()
        is_mutated = (germline_labels == 1.0) & non_special
        is_germline = (germline_labels == 0.0) & non_special

        mutated_rate = mask_counts[is_mutated].mean().item() / n_trials
        germline_rate = mask_counts[is_germline].mean().item() / n_trials

        assert mutated_rate > germline_rate * 2, (
            f"Mutated rate {mutated_rate:.3f} should be >2x germline "
            f"rate {germline_rate:.3f}"
        )


class TestCDR3IntermediateBias:
    """CDR3 positions (label=0.5) should be masked at an intermediate rate."""

    def test_cdr3_between_germline_and_mutated(self) -> None:
        seq_len = 120
        mutated_positions = list(range(10, 25))
        cdr3_positions = list(range(60, 80))
        strategy = GermlineMasking(
            _make_tokenizer(), mask_prob=0.15,
            mutated_weight=6.0, germline_weight=1.0,
        )
        input_ids, special_mask = _make_inputs(seq_len)
        germline_labels = _make_germline_labels(
            seq_len, mutated_positions, cdr3_positions,
        )

        mask_counts = torch.zeros(seq_len)
        n_trials = 200
        for seed in range(n_trials):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask,
                metadata={"germline_labels": germline_labels},
            )
            mask_counts += mask.float()

        non_special = ~special_mask.bool()
        is_mutated = (germline_labels == 1.0) & non_special
        is_cdr3 = (germline_labels == 0.5) & non_special
        is_germline = (germline_labels == 0.0) & non_special

        mutated_rate = mask_counts[is_mutated].mean().item() / n_trials
        cdr3_rate = mask_counts[is_cdr3].mean().item() / n_trials
        germline_rate = mask_counts[is_germline].mean().item() / n_trials

        assert germline_rate < cdr3_rate < mutated_rate, (
            f"Expected germline ({germline_rate:.3f}) < CDR3 ({cdr3_rate:.3f}) "
            f"< mutated ({mutated_rate:.3f})"
        )


class TestSoftLabelsGradient:
    """A gradient of labels should produce monotonically increasing mask rates."""

    def test_gradient_labels_monotonic(self) -> None:
        seq_len = 60
        strategy = GermlineMasking(
            _make_tokenizer(), mask_prob=0.15,
            mutated_weight=6.0, germline_weight=1.0,
        )
        input_ids, special_mask = _make_inputs(seq_len)

        # Create gradient: positions 5-14 get 0.0, 15-24 get 0.25,
        # 25-34 get 0.5, 35-44 get 0.75, 45-54 get 1.0
        labels = torch.zeros(seq_len, dtype=torch.float)
        groups = [(5, 15, 0.0), (15, 25, 0.25), (25, 35, 0.5),
                  (35, 45, 0.75), (45, 55, 1.0)]
        for start, end, val in groups:
            labels[start:end] = val

        mask_counts = torch.zeros(seq_len)
        n_trials = 200
        for seed in range(n_trials):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask,
                metadata={"germline_labels": labels},
            )
            mask_counts += mask.float()

        rates = []
        for start, end, _ in groups:
            rate = mask_counts[start:end].mean().item() / n_trials
            rates.append(rate)

        for i in range(len(rates) - 1):
            assert rates[i] < rates[i + 1], (
                f"Rates should be monotonically increasing: {rates}"
            )


class TestSpecialTokensNeverMasked:
    def test_special_positions_untouched(self) -> None:
        torch.manual_seed(0)
        strategy = GermlineMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(80)
        germline_labels = _make_germline_labels(80, list(range(10, 30)))

        for seed in range(20):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask,
                metadata={"germline_labels": germline_labels},
            )
            assert not mask[special_mask.bool()].any(), (
                f"Special token masked (seed={seed})"
            )


class TestFallbackToUniform:
    def test_no_germline_in_metadata(self) -> None:
        torch.manual_seed(42)
        strategy = GermlineMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(80)

        mask = strategy.select_mask_positions(input_ids, special_mask, metadata={})
        assert mask.sum().item() > 0, "Should fall back to uniform and mask something"
        assert not mask[special_mask.bool()].any()

    def test_none_metadata(self) -> None:
        torch.manual_seed(42)
        strategy = GermlineMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(80)

        mask = strategy.select_mask_positions(input_ids, special_mask, metadata=None)
        assert mask.sum().item() > 0

    def test_all_zero_labels_degrades_to_uniform(self) -> None:
        """All-zero germline labels should produce uniform-like masking."""
        torch.manual_seed(42)
        strategy = GermlineMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(80)
        germline_labels = torch.zeros(80, dtype=torch.float)

        mask = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"germline_labels": germline_labels},
        )
        assert mask.sum().item() > 0
        assert not mask[special_mask.bool()].any()


class TestEdgeCases:
    def test_all_special_tokens(self) -> None:
        strategy = GermlineMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids = torch.arange(5, 15, dtype=torch.long)
        special_mask = torch.ones(10, dtype=torch.long)
        germline_labels = torch.zeros(10, dtype=torch.float)

        mask = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"germline_labels": germline_labels},
        )
        assert mask.sum().item() == 0

    def test_single_maskable_token(self) -> None:
        strategy = GermlineMasking(_make_tokenizer(), mask_prob=1.0)
        input_ids = torch.tensor([1, 10, 2], dtype=torch.long)
        special_mask = torch.tensor([1, 0, 1], dtype=torch.long)
        germline_labels = torch.tensor([0.0, 1.0, 0.0])

        mask = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"germline_labels": germline_labels},
        )
        assert mask.sum().item() >= 0
        assert not mask[0].item()
        assert not mask[2].item()

    def test_all_mutated(self) -> None:
        """All non-special positions mutated — should work like uniform."""
        torch.manual_seed(42)
        seq_len = 50
        strategy = GermlineMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(seq_len)
        germline_labels = torch.ones(seq_len, dtype=torch.float)
        germline_labels[0] = 0.0
        germline_labels[seq_len - 1] = 0.0

        mask = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"germline_labels": germline_labels},
        )
        assert mask.sum().item() > 0
        assert not mask[special_mask.bool()].any()

    def test_all_germline(self) -> None:
        """All non-special positions germline — should work like uniform."""
        torch.manual_seed(42)
        seq_len = 50
        strategy = GermlineMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(seq_len)
        germline_labels = torch.zeros(seq_len, dtype=torch.float)

        mask = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"germline_labels": germline_labels},
        )
        assert mask.sum().item() > 0
        assert not mask[special_mask.bool()].any()

    def test_single_mutation(self) -> None:
        """Only one position has label 1.0, rest are 0.0."""
        torch.manual_seed(0)
        seq_len = 50
        strategy = GermlineMasking(
            _make_tokenizer(), mask_prob=0.15,
            mutated_weight=6.0, germline_weight=1.0,
        )
        input_ids, special_mask = _make_inputs(seq_len)
        germline_labels = torch.zeros(seq_len, dtype=torch.float)
        germline_labels[25] = 1.0

        mask = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"germline_labels": germline_labels},
        )
        assert mask.sum().item() > 0
        assert not mask[special_mask.bool()].any()


class TestRegistration:
    def test_strategy_registered(self) -> None:
        from masking.base import _STRATEGY_REGISTRY
        assert "germline" in _STRATEGY_REGISTRY

    def test_get_strategy_returns_germline(self) -> None:
        from masking import get_strategy
        strategy = get_strategy(
            "germline", tokenizer=_make_tokenizer(),
            mutated_weight=8.0, germline_weight=0.5,
        )
        assert isinstance(strategy, GermlineMasking)
        assert strategy.mutated_weight == 8.0
        assert strategy.germline_weight == 0.5


class TestDeterminism:
    def test_same_seed_same_mask(self) -> None:
        strategy = GermlineMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(100)
        germline_labels = _make_germline_labels(100, list(range(20, 40)))

        torch.manual_seed(99)
        mask_a = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"germline_labels": germline_labels},
        )
        torch.manual_seed(99)
        mask_b = strategy.select_mask_positions(
            input_ids, special_mask,
            metadata={"germline_labels": germline_labels},
        )
        assert torch.equal(mask_a, mask_b)
