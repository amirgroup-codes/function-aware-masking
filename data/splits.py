"""Deterministic train/eval partitioning of the pretraining corpus.

The held-out split must satisfy two properties:

1. **Identical across experiments**, so cross-model metric comparisons are
   well-defined (every model is scored on exactly the same sequences).
2. **Independent of the per-experiment ``seed``**, so that seed replicates
   vary only model initialization and masking RNG. If the split moved with
   ``seed``, a replicate would be evaluated on sequences it had trained on
   and its zero-shot metrics (CDR3 infilling, MLM accuracy, PLL) would be
   silently inflated rather than obviously broken.

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
