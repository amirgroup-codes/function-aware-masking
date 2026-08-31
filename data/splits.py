"""Deterministic train/eval partitioning of the pretraining corpus.

Both the training loop and the evaluation scripts call
:func:`make_train_eval_split` so the partition can only be defined in one
place.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset, random_split

#: Generator seed for the corpus partition. Changing this invalidates every
#: previously computed metric — all experiments would need re-evaluating.
DEFAULT_DATA_SPLIT_SEED = 42


def make_train_eval_split(
    dataset: Dataset,
    train_split: float,
    split_seed: int = DEFAULT_DATA_SPLIT_SEED,
) -> tuple[Subset, Subset]:
    """Partition ``dataset`` into ``(train, eval)`` deterministically.

    Args:
        dataset: Full pretraining corpus; must enumerate in a stable order.
        train_split: Fraction assigned to training (e.g. ``0.9``).
        split_seed: Generator seed for the partition. Deliberately *not*
            the experiment seed — see module docstring.

    Returns:
        ``(train_subset, eval_subset)``. The eval subset is the complement
        of the train subset, so no sequence appears in both.
    """
    train_size = int(len(dataset) * train_split)
    eval_size = len(dataset) - train_size
    return random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(split_seed),
    )
