from masking.base import BaseMaskingStrategy, get_strategy, register_strategy
from masking.collator import MLMDataCollator

import masking.cdr  # noqa: F401 — triggers registration
import masking.uniform  # noqa: F401 — triggers registration
import masking.span  # noqa: F401 — triggers registration
import masking.structure  # noqa: F401 — triggers registration
import masking.interface  # noqa: F401 — triggers registration
import masking.germline  # noqa: F401 — triggers registration
import masking.multispecific  # noqa: F401 — triggers registration
import masking.intersection  # noqa: F401 — triggers registration
import masking.hybrid  # noqa: F401 — triggers registration

__all__ = [
    "BaseMaskingStrategy",
    "MLMDataCollator",
    "get_strategy",
    "register_strategy",
]
