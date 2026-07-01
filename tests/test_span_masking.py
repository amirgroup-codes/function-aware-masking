"""Tests for the span masking strategy.

Covers budget correctness, span contiguity, chain-boundary respect,
geometric distribution bounds, determinism, and edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from masking.span import SpanMasking


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
    """Create dummy input_ids and special_tokens_mask.

    special_positions defaults to [0, seq_len-1] (CLS + SEP).
    """
    input_ids = torch.arange(5, 5 + seq_len, dtype=torch.long)
    special_tokens_mask = torch.zeros(seq_len, dtype=torch.long)
    if special_positions is None:
        special_positions = [0, seq_len - 1]
    for pos in special_positions:
        special_tokens_mask[pos] = 1
    return input_ids, special_tokens_mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSpanMaskingBudget:
    """Masked count should be close to mask_prob * num_maskable_tokens."""

    def test_budget_approximate(self) -> None:
        torch.manual_seed(0)
        strategy = SpanMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(120)
        num_maskable = int((~special_mask.bool()).sum().item())
        expected = round(0.15 * num_maskable)

        counts: list[int] = []
        for seed in range(50):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(input_ids, special_mask)
            counts.append(int(mask.sum().item()))

        mean_count = sum(counts) / len(counts)
        assert abs(mean_count - expected) / expected < 0.15, (
            f"Mean masked count {mean_count:.1f} deviates >15% from expected {expected}"
        )

    def test_budget_exact_small_sequence(self) -> None:
        """With a tiny sequence the budget should still be met (or all maskable used)."""
        torch.manual_seed(42)
        strategy = SpanMasking(_make_tokenizer(), mask_prob=0.5)
        input_ids, special_mask = _make_inputs(6)
        num_maskable = int((~special_mask.bool()).sum().item())
        expected = round(0.5 * num_maskable)

        mask = strategy.select_mask_positions(input_ids, special_mask)
        assert mask.sum().item() == expected


class TestSpecialTokensNeverMasked:
    def test_special_positions_untouched(self) -> None:
        torch.manual_seed(7)
        strategy = SpanMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(100)

        for seed in range(20):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(input_ids, special_mask)
            assert not mask[special_mask.bool()].any(), (
                f"Special token was masked (seed={seed})"
            )


class TestSpanContiguity:
    """Masked regions should form contiguous runs (spans)."""

    def test_no_isolated_unmasked_within_span(self) -> None:
        torch.manual_seed(3)
        strategy = SpanMasking(
            _make_tokenizer(), mask_prob=0.15, geometric_p=0.1, max_span_length=15,
        )
        input_ids, special_mask = _make_inputs(120)
        mask = strategy.select_mask_positions(input_ids, special_mask)

        maskable = ~special_mask.bool()
        spans = _extract_spans(mask, maskable)
        for start, end in spans:
            assert (end - start) >= 1


class TestChainBoundaryRespect:
    """Spans must not cross special tokens (chain separators)."""

    def test_sep_in_middle_blocks_span(self) -> None:
        torch.manual_seed(0)
        seq_len = 60
        sep_pos = 30
        special_positions = [0, sep_pos, seq_len - 1]
        input_ids, special_mask = _make_inputs(seq_len, special_positions)

        strategy = SpanMasking(
            _make_tokenizer(), mask_prob=0.15, geometric_p=0.05, max_span_length=20,
        )

        for seed in range(30):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(input_ids, special_mask)
            assert not mask[sep_pos], "SEP token itself was masked"

            maskable = ~special_mask.bool()
            spans = _extract_spans(mask, maskable)
            for start, end in spans:
                assert not (start < sep_pos and end > sep_pos), (
                    f"Span [{start}, {end}) crosses SEP at position {sep_pos}"
                )


class TestGeometricDistributionBounds:
    def test_span_length_within_bounds(self) -> None:
        strategy = SpanMasking(_make_tokenizer(), max_span_length=8)
        lengths: list[int] = []
        for _ in range(500):
            lengths.append(strategy._sample_span_length())
        assert all(1 <= l <= 8 for l in lengths)

    def test_different_p_changes_mean(self) -> None:
        strat_low_p = SpanMasking(_make_tokenizer(), geometric_p=0.1, max_span_length=30)
        strat_high_p = SpanMasking(_make_tokenizer(), geometric_p=0.5, max_span_length=30)

        torch.manual_seed(0)
        low_p_lengths = [strat_low_p._sample_span_length() for _ in range(500)]
        torch.manual_seed(0)
        high_p_lengths = [strat_high_p._sample_span_length() for _ in range(500)]

        mean_low = sum(low_p_lengths) / len(low_p_lengths)
        mean_high = sum(high_p_lengths) / len(high_p_lengths)
        assert mean_low > mean_high, (
            f"Lower p should give longer spans: mean_low={mean_low:.1f}, mean_high={mean_high:.1f}"
        )


class TestDeterminism:
    def test_same_seed_same_mask(self) -> None:
        strategy = SpanMasking(_make_tokenizer(), mask_prob=0.15)
        input_ids, special_mask = _make_inputs(100)

        torch.manual_seed(99)
        mask_a = strategy.select_mask_positions(input_ids, special_mask)

        torch.manual_seed(99)
        mask_b = strategy.select_mask_positions(input_ids, special_mask)

        assert torch.equal(mask_a, mask_b)


class TestEdgeCases:
    def test_all_special_tokens(self) -> None:
        strategy = SpanMasking(_make_tokenizer(), mask_prob=0.15)
        seq_len = 10
        input_ids = torch.arange(5, 5 + seq_len, dtype=torch.long)
        special_mask = torch.ones(seq_len, dtype=torch.long)

        mask = strategy.select_mask_positions(input_ids, special_mask)
        assert mask.sum().item() == 0

    def test_single_maskable_token(self) -> None:
        strategy = SpanMasking(_make_tokenizer(), mask_prob=1.0)
        input_ids = torch.tensor([1, 10, 2], dtype=torch.long)
        special_mask = torch.tensor([1, 0, 1], dtype=torch.long)

        mask = strategy.select_mask_positions(input_ids, special_mask)
        assert mask.sum().item() == 1
        assert mask[1].item() is True

    def test_very_short_sequence(self) -> None:
        torch.manual_seed(0)
        strategy = SpanMasking(_make_tokenizer(), mask_prob=0.15, max_span_length=10)
        input_ids, special_mask = _make_inputs(5)
        mask = strategy.select_mask_positions(input_ids, special_mask)
        assert not mask[special_mask.bool()].any()


class TestRegistration:
    def test_strategy_registered(self) -> None:
        from masking.base import _STRATEGY_REGISTRY
        assert "span" in _STRATEGY_REGISTRY

    def test_get_strategy_returns_span(self) -> None:
        from masking import get_strategy
        strategy = get_strategy("span", tokenizer=_make_tokenizer(), geometric_p=0.3)
        assert isinstance(strategy, SpanMasking)
        assert strategy.geometric_p == 0.3


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_spans(
    mask: torch.Tensor, maskable: torch.Tensor,
) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) for each contiguous run of True in mask.

    Only considers maskable positions.
    """
    spans: list[tuple[int, int]] = []
    in_span = False
    start = 0
    for i in range(mask.size(0)):
        if mask[i] and maskable[i]:
            if not in_span:
                start = i
                in_span = True
        else:
            if in_span:
                spans.append((start, i))
                in_span = False
    if in_span:
        spans.append((start, mask.size(0)))
    return spans
