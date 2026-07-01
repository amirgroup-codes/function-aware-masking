"""HuggingFace TrainerCallbacks for masking strategy integration.

Provides HybridMaskingCallback which updates the HybridMasking strategy's
mixture weights at each training step based on the curriculum schedule.
"""

from __future__ import annotations

import logging

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class HybridMaskingCallback(TrainerCallback):
    """Callback that updates HybridMasking weights based on training step.

    At each step begin, calls strategy.set_step(global_step) to allow
    the hybrid strategy to interpolate its curriculum schedule.
    """

    def __init__(self, strategy) -> None:
        self.strategy = strategy

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self.strategy.set_step(state.global_step)
