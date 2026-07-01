"""Base evaluator ABC for antibody MLM evaluation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """Abstract base class for evaluation tasks.

    Subclasses implement evaluate() which takes a model + data
    and returns a metrics dict.
    """

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        """Run the evaluation and return metrics.

        Returns:
            Dict mapping metric names to values.
        """
        ...
