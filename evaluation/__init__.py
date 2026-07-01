"""Zero-shot evaluation utilities for antibody masked language models.

This public release ships only the lightweight, dependency-free zero-shot
metrics used to compare masking strategies under identical conditions:

  * :class:`MLMAccuracyEvaluator` -- masked-token accuracy / perplexity,
    with a per-region (framework / CDR / CDR3) breakdown when CDR labels
    are present. Always scored under a *uniform reference mask* so every
    model is measured on the same masking distribution.
  * :func:`compute_pll` / :func:`compute_pll_batch` -- exact
    pseudo-log-likelihood, the standard zero-shot protein-LM fitness score.

The full benchmark suite from the paper (CDR infilling, AB-Bind mutation
scoring, attention analysis, and the supervised downstream probes) lives in
the research repository and is intentionally omitted here to keep the public
codebase focused on the masking method and training loop.
"""

from evaluation.base import BaseEvaluator
from evaluation.mlm_accuracy import MLMAccuracyEvaluator
from evaluation.pseudo_loglikelihood import compute_pll, compute_pll_batch

__all__ = [
    "BaseEvaluator",
    "MLMAccuracyEvaluator",
    "compute_pll",
    "compute_pll_batch",
]
