"""Tests for hybrid masking meta-strategy.

Uses synthetic inputs (no external data dependency) to validate static mixture,
curriculum scheduling, per-sample availability filtering, budget correctness,
special token protection, determinism, registration, param routing, paired
compatibility, and edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from masking.hybrid import HybridMasking, _strategy_available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tokenizer(vocab_size: int = 30, mask_token_id: int = 4) -> MagicMock:
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.mask_token_id = mask_token_id
    tok.pad_token_id = 0
    tok.unk_token_id = 1
    # Map canonical AAs to deterministic integer ids so BaseMaskingStrategy
    # can build its precomputed _aa_ids tensor for random-token replacement.
    _aa_to_id = {aa: 5 + i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    tok.convert_tokens_to_ids.side_effect = lambda c: _aa_to_id.get(c, 1)
    return tok


def _make_inputs(
    seq_len: int = 100,
    special_positions: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.arange(5, 5 + seq_len, dtype=torch.long)
    special_tokens_mask = torch.zeros(seq_len, dtype=torch.long)
    if special_positions is None:
        special_positions = [0, seq_len - 1]
    for pos in special_positions:
        special_tokens_mask[pos] = 1
    return input_ids, special_tokens_mask


def _make_paired_inputs(
    vh_len: int = 120,
    vl_len: int = 110,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Create synthetic multi-module paired input."""
    seq_len = vh_len + vl_len + 6
    input_ids = torch.arange(5, 5 + seq_len, dtype=torch.long)

    special_tokens_mask = torch.zeros(seq_len, dtype=torch.long)
    special_positions = [0, 1, 2, 3 + vh_len, 4 + vh_len, 5 + vh_len + vl_len]
    for pos in special_positions:
        if pos < seq_len:
            special_tokens_mask[pos] = 1

    module_ids = torch.zeros(seq_len, dtype=torch.long)
    module_ids[1:seq_len] = 1

    chain_type_ids = torch.zeros(seq_len, dtype=torch.long)
    chain_type_ids[3:3 + vh_len] = 1
    chain_type_ids[5 + vh_len:5 + vh_len + vl_len] = 2

    metadata = {
        "module_ids": module_ids,
        "chain_type_ids": chain_type_ids,
    }
    return input_ids, special_tokens_mask, metadata


def _add_cdr_mask(metadata: dict, seq_len: int, cdr_positions: list[int]) -> dict:
    """Add cdr_mask to metadata."""
    cdr_mask = torch.zeros(seq_len, dtype=torch.long)
    for pos in cdr_positions:
        cdr_mask[pos] = 3  # CDR3
    metadata["cdr_mask"] = cdr_mask
    return metadata


def _add_paratope_labels(
    metadata: dict, seq_len: int, paratope_positions: list[int],
) -> dict:
    labels = torch.zeros(seq_len, dtype=torch.float)
    for pos in paratope_positions:
        labels[pos] = 1.0
    metadata["paratope_labels"] = labels
    return metadata


def _add_germline_labels(
    metadata: dict, seq_len: int, mutated_positions: list[int],
) -> dict:
    labels = torch.zeros(seq_len, dtype=torch.float)
    for pos in mutated_positions:
        labels[pos] = 1.0
    metadata["germline_labels"] = labels
    return metadata


# ---------------------------------------------------------------------------
# TestStaticMixture
# ---------------------------------------------------------------------------

class TestStaticMixture:
    """Verify static mixture routing."""

    def test_all_strategies_used_over_many_samples(self):
        tok = _make_tokenizer()
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform", "span"],
            policy_weights=[1.0, 1.0],
        )

        input_ids, special_mask = _make_inputs(100)
        counts = {"uniform": 0, "span": 0}

        for seed in range(200):
            torch.manual_seed(seed)
            strategy.select_mask_positions(input_ids, special_mask, metadata={})

        # Both strategies should have been used
        for name in ["uniform", "span"]:
            assert strategy._strategy_counts[name] > 0, (
                f"Strategy {name} was never selected"
            )

    def test_single_strategy_weight_routes_all_calls(self):
        tok = _make_tokenizer()
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform", "span"],
            policy_weights=[1.0, 0.0],
        )

        input_ids, special_mask = _make_inputs(100)
        for seed in range(50):
            torch.manual_seed(seed)
            strategy.select_mask_positions(input_ids, special_mask, metadata={})

        assert strategy._strategy_counts["uniform"] == 50
        assert strategy._strategy_counts["span"] == 0

    def test_weights_proportional_to_config(self):
        tok = _make_tokenizer()
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform", "span"],
            policy_weights=[3.0, 1.0],  # 75% / 25%
        )

        input_ids, special_mask = _make_inputs(100)
        n_trials = 1000
        for seed in range(n_trials):
            torch.manual_seed(seed)
            strategy.select_mask_positions(input_ids, special_mask, metadata={})

        uniform_frac = strategy._strategy_counts["uniform"] / n_trials
        # Should be ~0.75, allow wide margin
        assert 0.60 < uniform_frac < 0.90, (
            f"Expected ~75% uniform, got {uniform_frac:.2%}"
        )


# ---------------------------------------------------------------------------
# TestCurriculumSchedule
# ---------------------------------------------------------------------------

class TestCurriculumSchedule:
    """Verify curriculum weight interpolation."""

    def _make_strategy(self):
        tok = _make_tokenizer()
        return HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform", "span"],
            policy_weights=[1.0, 0.0],
            curriculum=[
                {"step": 0, "weights": [1.0, 0.0]},
                {"step": 100, "weights": [0.0, 1.0]},
            ],
        )

    def test_weights_at_first_breakpoint(self):
        s = self._make_strategy()
        s.set_step(0)
        assert torch.allclose(s._current_weights, torch.tensor([1.0, 0.0]))

    def test_weights_at_last_breakpoint(self):
        s = self._make_strategy()
        s.set_step(100)
        assert torch.allclose(s._current_weights, torch.tensor([0.0, 1.0]))

    def test_linear_interpolation_midpoint(self):
        s = self._make_strategy()
        s.set_step(50)
        assert torch.allclose(s._current_weights, torch.tensor([0.5, 0.5]), atol=0.01)

    def test_before_first_breakpoint_uses_first(self):
        s = self._make_strategy()
        s.set_step(-10)
        assert torch.allclose(s._current_weights, torch.tensor([1.0, 0.0]))

    def test_after_last_breakpoint_uses_last(self):
        s = self._make_strategy()
        s.set_step(200)
        assert torch.allclose(s._current_weights, torch.tensor([0.0, 1.0]))

    def test_set_step_noop_without_curriculum(self):
        tok = _make_tokenizer()
        s = HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform", "span"],
            policy_weights=[0.6, 0.4],
        )
        original = s._current_weights.clone()
        s.set_step(9999)
        assert torch.equal(s._current_weights, original)

    def test_three_breakpoints(self):
        tok = _make_tokenizer()
        s = HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform", "span"],
            policy_weights=[1.0, 0.0],
            curriculum=[
                {"step": 0, "weights": [1.0, 0.0]},
                {"step": 100, "weights": [0.5, 0.5]},
                {"step": 200, "weights": [0.0, 1.0]},
            ],
        )
        s.set_step(50)
        assert torch.allclose(s._current_weights, torch.tensor([0.75, 0.25]), atol=0.01)
        s.set_step(150)
        assert torch.allclose(s._current_weights, torch.tensor([0.25, 0.75]), atol=0.01)


# ---------------------------------------------------------------------------
# TestPerSampleAvailability
# ---------------------------------------------------------------------------

class TestPerSampleAvailability:
    """Verify per-sample strategy filtering based on metadata availability."""

    def test_unavailable_strategy_excluded(self):
        """CDR excluded when no cdr_mask; only uniform fires."""
        tok = _make_tokenizer()
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform", "cdr"],
            policy_weights=[0.5, 0.5],
        )

        input_ids, special_mask = _make_inputs(100)
        # No cdr_mask in metadata → CDR unavailable
        for seed in range(50):
            torch.manual_seed(seed)
            strategy.select_mask_positions(input_ids, special_mask, metadata={})

        assert strategy._strategy_counts["uniform"] == 50
        assert strategy._strategy_counts["cdr"] == 0

    def test_cdr_available_with_cdr_mask(self):
        tok = _make_tokenizer()
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform", "cdr"],
            policy_weights=[0.5, 0.5],
        )

        input_ids, special_mask = _make_inputs(100)
        metadata = {}
        _add_cdr_mask(metadata, 100, list(range(30, 50)))

        for seed in range(100):
            torch.manual_seed(seed)
            strategy.select_mask_positions(input_ids, special_mask, metadata=metadata)

        assert strategy._strategy_counts["cdr"] > 0

    def test_all_strategies_unavailable_falls_back(self):
        """If all strategies need metadata and none is present, uniform fallback."""
        tok = _make_tokenizer()
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["cdr", "interface"],
            policy_weights=[0.5, 0.5],
        )

        input_ids, special_mask = _make_inputs(100)
        torch.manual_seed(0)
        mask = strategy.select_mask_positions(input_ids, special_mask, metadata={})

        assert mask.sum().item() > 0
        assert strategy._fallback_count == 1

    def test_structure_available_with_knn_indices(self):
        metadata = {"knn_indices": torch.zeros(100, 32, dtype=torch.long)}
        assert _strategy_available("structure", metadata) is True

    def test_structure_available_with_coords_ca(self):
        metadata = {"coords_ca": torch.zeros(100, 3, dtype=torch.float)}
        assert _strategy_available("structure", metadata) is True

    def test_structure_unavailable_without_either(self):
        assert _strategy_available("structure", {}) is False

    def test_multispecific_needs_both_keys(self):
        assert _strategy_available("multispecific", {"module_ids": torch.zeros(10)}) is False
        assert _strategy_available("multispecific", {
            "module_ids": torch.zeros(10),
            "chain_type_ids": torch.zeros(10),
        }) is True

    def test_uniform_always_available(self):
        assert _strategy_available("uniform", None) is True
        assert _strategy_available("uniform", {}) is True

    def test_span_always_available(self):
        assert _strategy_available("span", None) is True


# ---------------------------------------------------------------------------
# TestBudgetCorrectness
# ---------------------------------------------------------------------------

class TestBudgetCorrectness:
    """Mean masked count should approximate mask_prob * maskable."""

    def test_budget_approximate(self):
        tok = _make_tokenizer()
        mask_prob = 0.15
        strategy = HybridMasking(
            tokenizer=tok,
            mask_prob=mask_prob,
            sub_strategies=["uniform", "span"],
            policy_weights=[1.0, 1.0],
        )

        seq_len = 100
        input_ids, special_mask = _make_inputs(seq_len)
        num_maskable = seq_len - special_mask.sum().item()
        expected = mask_prob * num_maskable

        counts = []
        for seed in range(50):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(input_ids, special_mask, metadata={})
            counts.append(mask.sum().item())

        mean_count = sum(counts) / len(counts)
        assert abs(mean_count - expected) / expected < 0.20, (
            f"Mean masked {mean_count:.1f} vs expected {expected:.1f}"
        )


# ---------------------------------------------------------------------------
# TestSpecialTokensNeverMasked
# ---------------------------------------------------------------------------

class TestSpecialTokensNeverMasked:
    """Special token positions must never be masked."""

    def test_special_positions_untouched(self):
        tok = _make_tokenizer()
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform", "span"],
            policy_weights=[1.0, 1.0],
        )

        seq_len = 100
        input_ids, special_mask = _make_inputs(seq_len)

        for seed in range(30):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(input_ids, special_mask, metadata={})
            assert not mask[special_mask.bool()].any(), (
                f"Special tokens masked at seed {seed}"
            )


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same seed must produce identical masks."""

    def test_same_seed_same_mask(self):
        tok = _make_tokenizer()
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform", "span"],
            policy_weights=[1.0, 1.0],
        )

        input_ids, special_mask = _make_inputs(100)

        torch.manual_seed(99)
        mask_a = strategy.select_mask_positions(input_ids, special_mask, metadata={})

        torch.manual_seed(99)
        mask_b = strategy.select_mask_positions(input_ids, special_mask, metadata={})

        assert torch.equal(mask_a, mask_b)


# ---------------------------------------------------------------------------
# TestRegistration
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_strategy_registered(self):
        from masking.base import _STRATEGY_REGISTRY
        assert "hybrid" in _STRATEGY_REGISTRY

    def test_get_strategy_returns_hybrid(self):
        from masking import get_strategy
        tok = _make_tokenizer()
        strategy = get_strategy(
            "hybrid",
            tokenizer=tok,
            sub_strategies=["uniform"],
            policy_weights=[1.0],
        )
        assert isinstance(strategy, HybridMasking)


# ---------------------------------------------------------------------------
# TestSubStrategyParamRouting
# ---------------------------------------------------------------------------

class TestSubStrategyParamRouting:
    """Verify sub-strategy-specific params are forwarded."""

    def test_cdr_params_forwarded(self):
        tok = _make_tokenizer()
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["cdr"],
            policy_weights=[1.0],
            sub_strategy_params={"cdr": {"cdr3_weight": 10.0}},
        )
        cdr_strategy = strategy._strategies["cdr"]
        # cdr strategy stores region weights as tensor [fw, c1, c2, c3]
        assert cdr_strategy._region_weights[3].item() == 10.0

    def test_span_params_forwarded(self):
        tok = _make_tokenizer()
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["span"],
            policy_weights=[1.0],
            sub_strategy_params={"span": {"geometric_p": 0.5}},
        )
        span_strategy = strategy._strategies["span"]
        assert span_strategy.geometric_p == 0.5

    def test_default_params_when_not_specified(self):
        tok = _make_tokenizer()
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform"],
            policy_weights=[1.0],
        )
        uniform_strategy = strategy._strategies["uniform"]
        assert uniform_strategy.mask_prob == 0.15


# ---------------------------------------------------------------------------
# TestPairedCompatibility
# ---------------------------------------------------------------------------

class TestPairedCompatibility:
    """Verify hybrid works with multispecific as a sub-strategy."""

    def test_hybrid_with_multispecific(self):
        tok = _make_tokenizer(vocab_size=34)
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform", "multispecific"],
            policy_weights=[0.3, 0.7],
            sub_strategy_params={
                "multispecific": {
                    "paratope_weight": 6.0,
                    "shared_chain_boost": 3.0,
                    "interface_weight": 6.0,
                    "policy_weights": [1.0, 1.0, 1.0],
                },
            },
        )

        input_ids, special_mask, metadata = _make_paired_inputs()
        _add_paratope_labels(metadata, len(input_ids), list(range(10, 30)))

        for seed in range(50):
            torch.manual_seed(seed)
            mask = strategy.select_mask_positions(
                input_ids, special_mask, metadata=metadata,
            )
            assert not mask[special_mask.bool()].any()

        assert strategy._strategy_counts["multispecific"] > 0
        assert strategy._strategy_counts["uniform"] > 0


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_sub_strategies_raises(self):
        tok = _make_tokenizer()
        with pytest.raises(ValueError, match="at least one strategy"):
            HybridMasking(
                tokenizer=tok,
                sub_strategies=[],
                policy_weights=[],
            )

    def test_mismatched_weights_length_raises(self):
        tok = _make_tokenizer()
        with pytest.raises(ValueError, match="policy_weights length"):
            HybridMasking(
                tokenizer=tok,
                sub_strategies=["uniform", "span"],
                policy_weights=[1.0],
            )

    def test_curriculum_weights_length_mismatch_raises(self):
        tok = _make_tokenizer()
        with pytest.raises(ValueError, match="Curriculum weights at step"):
            HybridMasking(
                tokenizer=tok,
                sub_strategies=["uniform", "span"],
                policy_weights=[1.0, 1.0],
                curriculum=[{"step": 0, "weights": [1.0]}],
            )

    def test_unregistered_sub_strategy_raises(self):
        tok = _make_tokenizer()
        with pytest.raises(KeyError, match="nonexistent"):
            HybridMasking(
                tokenizer=tok,
                sub_strategies=["nonexistent"],
                policy_weights=[1.0],
            )

    def test_none_metadata(self):
        tok = _make_tokenizer()
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform"],
            policy_weights=[1.0],
        )
        input_ids, special_mask = _make_inputs(100)
        mask = strategy.select_mask_positions(input_ids, special_mask, metadata=None)
        assert mask.sum().item() > 0


# ---------------------------------------------------------------------------
# TestDiagnosticLogging
# ---------------------------------------------------------------------------

class TestDiagnosticLogging:
    """Verify strategy counts are tracked."""

    def test_strategy_counts_tracked(self):
        tok = _make_tokenizer()
        strategy = HybridMasking(
            tokenizer=tok,
            sub_strategies=["uniform", "span"],
            policy_weights=[1.0, 1.0],
        )

        input_ids, special_mask = _make_inputs(100)
        n = 100
        for seed in range(n):
            torch.manual_seed(seed)
            strategy.select_mask_positions(input_ids, special_mask, metadata={})

        total = sum(strategy._strategy_counts.values())
        assert total == n
        assert strategy._total_calls == n
