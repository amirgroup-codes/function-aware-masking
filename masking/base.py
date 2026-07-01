"""Base masking strategy ABC and strategy registry.

All masking strategies inherit from BaseMaskingStrategy and register
themselves via the @register_strategy decorator. The training code
resolves strategies by name through get_strategy(), never importing
concrete classes directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


_CANONICAL_AAS = "ACDEFGHIKLMNPQRSTVWY"


class BaseMaskingStrategy(ABC):
    """Abstract base class for all masking strategies.

    Subclasses implement `apply()` which takes tokenized input_ids and
    returns (masked_input_ids, labels) for MLM training.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_prob: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_token_ratio = mask_token_ratio
        self.random_token_ratio = random_token_ratio
        self.keep_ratio = 1.0 - mask_token_ratio - random_token_ratio
        # Precomputed canonical-AA token IDs for random-token replacement.
        # The previous implementation sampled `randint(low=5, high=vocab_size)`
        # which can pick non-standard tokens like X/B/Z, replacing
        # ~0.065% of tokens with garbage.
        aa_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in _CANONICAL_AAS]
        unk = tokenizer.unk_token_id
        self._aa_ids = torch.tensor(
            [tid for tid in aa_ids if tid is not None and tid != unk],
            dtype=torch.long,
        )

    @abstractmethod
    def select_mask_positions(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Select which positions to mask.

        Args:
            input_ids: Token IDs of shape (seq_len,).
            special_tokens_mask: Binary mask where 1 = special token (do not mask).
            metadata: Optional per-sample tensors (e.g. cdr_mask for region-aware
                strategies). Strategies that don't need metadata can ignore it.

        Returns:
            Boolean tensor of shape (seq_len,) where True = position will be masked.
        """
        ...

    def apply(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply masking to a single sequence.

        Args:
            input_ids: Token IDs of shape (seq_len,).
            special_tokens_mask: Binary mask where 1 = special token.
            metadata: Optional per-sample tensors forwarded to select_mask_positions.

        Returns:
            masked_input_ids: Token IDs with some positions replaced.
            labels: Original token IDs at masked positions, -100 elsewhere.
        """
        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()

        mask_positions = self.select_mask_positions(
            input_ids, special_tokens_mask, metadata=metadata
        )

        labels[~mask_positions] = -100

        num_masked = mask_positions.sum().item()
        if num_masked == 0:
            return masked_input_ids, labels

        rand = torch.rand(num_masked)

        replace_with_mask = rand < self.mask_token_ratio
        replace_with_random = (rand >= self.mask_token_ratio) & (
            rand < self.mask_token_ratio + self.random_token_ratio
        )

        mask_indices = mask_positions.nonzero(as_tuple=True)[0]

        masked_input_ids[mask_indices[replace_with_mask]] = self.tokenizer.mask_token_id
        n_random = int(replace_with_random.sum().item())
        if n_random > 0:
            sampled = torch.randint(
                low=0, high=self._aa_ids.numel(), size=(n_random,),
            )
            random_tokens = self._aa_ids[sampled]
            masked_input_ids[mask_indices[replace_with_random]] = random_tokens

        return masked_input_ids, labels


_STRATEGY_REGISTRY: dict[str, type[BaseMaskingStrategy]] = {}


def register_strategy(name: str):
    """Decorator to register a masking strategy class by name."""
    def decorator(cls: type[BaseMaskingStrategy]) -> type[BaseMaskingStrategy]:
        _STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def get_strategy(
    name: str,
    tokenizer: PreTrainedTokenizerBase,
    **kwargs: Any,
) -> BaseMaskingStrategy:
    """Instantiate a masking strategy by its registered name.

    Args:
        name: Registered strategy name (e.g., "uniform").
        tokenizer: Tokenizer instance to pass to the strategy.
        **kwargs: Additional arguments forwarded to the strategy constructor.

    Raises:
        KeyError: If the strategy name is not registered.
    """
    if name not in _STRATEGY_REGISTRY:
        available = ", ".join(sorted(_STRATEGY_REGISTRY.keys()))
        raise KeyError(
            f"Unknown masking strategy '{name}'. Available: {available}"
        )
    return _STRATEGY_REGISTRY[name](tokenizer=tokenizer, **kwargs)
