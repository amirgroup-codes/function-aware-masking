"""MLM Data Collator that delegates masking to a pluggable strategy.

This collator handles padding/batching and calls the masking strategy's
apply() method on each sequence. It is strategy-agnostic — swap the
strategy and the collator works unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from masking.base import BaseMaskingStrategy


def _to_tensor(value: Any) -> torch.Tensor:
    """Convert a value to a tensor, preserving float dtype where appropriate."""
    t = torch.as_tensor(value)
    return t.float() if t.is_floating_point() else t.long()


@dataclass
class MLMDataCollator:
    """Collator for masked language modeling with pluggable masking strategy.

    Pads sequences to equal length within a batch, then applies the
    masking strategy to produce (input_ids, attention_mask, labels).
    """

    tokenizer: PreTrainedTokenizerBase
    strategy: BaseMaskingStrategy
    pad_to_multiple_of: int | None = 8
    return_metadata: bool = False

    _CORE_KEYS = {"input_ids", "attention_mask", "special_tokens_mask"}

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Some strategies (HybridMasking with sampling_mode="per_batch") draw
        # one sub-strategy per batch instead of per sample. Notify the
        # strategy that a new batch is starting if it supports the hook.
        begin_batch = getattr(self.strategy, "begin_batch", None)
        if callable(begin_batch):
            begin_batch()

        input_ids_list = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples]
        attention_mask_list = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in examples]
        special_tokens_mask_list = [torch.tensor(ex["special_tokens_mask"], dtype=torch.long) for ex in examples]

        extra_keys = {k for ex in examples for k in ex if k not in self._CORE_KEYS}

        max_len = max(ids.size(0) for ids in input_ids_list)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_metadata: dict[str, list[torch.Tensor]] = {
            k: [] for k in extra_keys
        }

        # Reference shape AND dtype per key, used to zero-fill missing
        # metadata so the output batch tensor stacks cleanly even when some
        # examples lack the key. Storing dtype is critical: float-typed
        # metadata (paratope_labels, coords_ca, germline_labels) must be
        # filled with zeros of matching dtype, otherwise torch.stack() raises
        # a dtype mismatch error on mixed-presence batches.
        ref_info: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        for ex in examples:
            for k in extra_keys:
                if k in ex and k not in ref_info:
                    t = _to_tensor(ex[k])
                    ref_info[k] = (tuple(t.shape), t.dtype)

        for i, (ex, input_ids, attn_mask, special_mask) in enumerate(zip(
            examples, input_ids_list, attention_mask_list, special_tokens_mask_list
        )):
            pad_len = max_len - input_ids.size(0)
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
                attn_mask = torch.cat([attn_mask, torch.zeros(pad_len, dtype=torch.long)])
                special_mask = torch.cat([special_mask, torch.ones(pad_len, dtype=torch.long)])

            # Build per-example metadata directly from THIS example to avoid
            # the historical bug where filtering examples-with-key and then
            # indexing by global position misaligned metadata across batch
            # members. Each example's metadata always corresponds to that
            # example's tokens. Strategies see only keys this example has.
            metadata: dict[str, torch.Tensor] | None = None
            if extra_keys:
                metadata = {}
                for k in extra_keys:
                    if k not in ex:
                        continue
                    t = _to_tensor(ex[k])
                    if pad_len > 0 and t.dim() >= 1:
                        pad_shape = (pad_len,) + t.shape[1:]
                        t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)])
                    metadata[k] = t

            # Build the per-example output batch_metadata WITH zero fills
            # for missing keys, so downstream consumers (e.g. region-stratified
            # MLM accuracy that reads cdr_mask) don't silently drop the entire
            # key from the batch when one example happens to lack it.
            for k in extra_keys:
                if metadata is not None and k in metadata:
                    batch_metadata[k].append(metadata[k])
                elif k in ref_info:
                    # Use reference shape AND dtype from another example for
                    # the zero fill, then pad to current max_len for any 1D+
                    # tensor. Matching dtype is essential — see comment above.
                    shape, dtype = ref_info[k]
                    fill = torch.zeros(shape, dtype=dtype)
                    if fill.dim() >= 1 and fill.size(0) < max_len:
                        pad_shape = (max_len - fill.size(0),) + fill.shape[1:]
                        fill = torch.cat([fill, torch.zeros(pad_shape, dtype=dtype)])
                    batch_metadata[k].append(fill)

            masked_ids, labels = self.strategy.apply(
                input_ids, special_mask, metadata=metadata
            )

            batch_input_ids.append(masked_ids)
            batch_attention_mask.append(attn_mask)
            batch_labels.append(labels)

        result = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }

        if self.return_metadata:
            for k, tensors in batch_metadata.items():
                # Every example contributed an entry (real or zero-filled),
                # so all tensors share the same dim-0 size (max_len) and
                # stacking always succeeds. The key never silently
                # disappears from the batch.
                if tensors:
                    result[k] = torch.stack(tensors)

        return result
