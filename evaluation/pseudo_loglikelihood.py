"""Pseudo-log-likelihood (PLL) scoring for antibody sequences.

Exact PLL: for each non-special position i, mask it, run a forward pass,
and accumulate log p(true_token_i | context). This is the standard
zero-shot fitness metric used in ESM and protein LM benchmarks.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from utils.tokenizer import tokenize_single_chain

logger = logging.getLogger(__name__)


STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def sanitize_sequence(sequence: str) -> str:
    """Remove non-standard amino acid characters from a sequence."""
    return "".join(c for c in sequence if c in STANDARD_AAS)


def _get_max_seq_length(model: torch.nn.Module) -> int | None:
    """Extract max_position_embeddings from a model's config, if available."""
    config = getattr(model, "config", None)
    if config is None:
        return None
    return getattr(config, "max_position_embeddings", None)


def compute_pll(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    sequence: str | None = None,
    device: str = "cuda",
    batch_size: int = 64,
    pre_tokenized_ids: torch.Tensor | None = None,
    pre_tokenized_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute the exact pseudo-log-likelihood for a single sequence.

    Creates L copies of the tokenized sequence (one per non-special position),
    each with a different position masked, and batches the forward passes.

    Sequences longer than the model's max_position_embeddings are truncated
    to avoid out-of-bounds position embedding lookups.

    Args:
        model: Trained masked LM.
        tokenizer: Matching tokenizer.
        sequence: Raw amino acid string (e.g. "EVQLVES..."). Ignored when
            pre_tokenized_ids is provided.
        device: Torch device.
        batch_size: How many masked copies to process per forward pass.
        pre_tokenized_ids: Optional pre-tokenized input IDs (1-D tensor).
            Use this for paired sequences where the tokenization format
            must be preserved (e.g. [CLS][MOD1][H]VH...[SEP][L]VL...[SEP]).
        pre_tokenized_mask: Optional attention mask matching pre_tokenized_ids.
            Defaults to all-ones if not provided.

    Returns:
        Dict with 'pll' (sum of log probs), 'pll_normalized' (pll / L),
        and 'was_truncated' (bool).
    """
    if pre_tokenized_ids is not None:
        input_ids = pre_tokenized_ids
        attention_mask = (
            pre_tokenized_mask
            if pre_tokenized_mask is not None
            else torch.ones_like(input_ids)
        )
        was_truncated = False
        # Symmetric safety with the string path: clip if longer than the
        # model's positional window so the forward pass doesn't hit an
        # out-of-bounds position embedding lookup.
        max_model_len = _get_max_seq_length(model)
        if max_model_len is not None and input_ids.size(0) > max_model_len:
            logger.debug(
                "Truncating pre-tokenized input from %d to %d (model max_position_embeddings=%d)",
                input_ids.size(0), max_model_len, max_model_len,
            )
            input_ids = input_ids[:max_model_len]
            attention_mask = attention_mask[:max_model_len]
            was_truncated = True
    else:
        sequence = sanitize_sequence(sequence)
        if not sequence:
            return {"pll": 0.0, "pll_normalized": 0.0, "was_truncated": False}

        max_model_len = _get_max_seq_length(model) or 512

        # Reserve space for special tokens. Standard tokenizer adds
        # [CLS]+[SEP] (2). Multispecific in heavy-only mode adds
        # [CLS][MOD1][H]...[SEP] (4). Computing this here so the
        # was_truncated flag matches what tokenize_single_chain emits.
        additional = tokenizer.additional_special_tokens or []
        num_special = 4 if "[MOD1]" in additional else 2
        max_aa = max_model_len - num_special
        was_truncated = len(sequence) > max_aa
        if was_truncated:
            logger.debug(
                "Truncating sequence from %d to %d AA (model max_position_embeddings=%d)",
                len(sequence), max_aa, max_model_len,
            )
            sequence = sequence[:max_aa]

        # Use tokenize_single_chain so paired-model tokenizers add the
        # [MOD1][H] framing they were trained with. The standard tokenizer
        # path falls through to a normal tokenizer call inside the helper.
        encoding = tokenize_single_chain(tokenizer, sequence, max_length=max_model_len)
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)

    special_tokens = set(tokenizer.all_special_ids)
    mask_token_id = tokenizer.mask_token_id
    vocab_size = model.config.vocab_size

    maskable = [
        i for i in range(len(input_ids))
        if input_ids[i].item() not in special_tokens
        and input_ids[i].item() < vocab_size
    ]

    if not maskable:
        return {"pll": 0.0, "pll_normalized": 0.0, "was_truncated": was_truncated}

    model.eval()
    total_log_prob = 0.0

    for start in range(0, len(maskable), batch_size):
        chunk = maskable[start : start + batch_size]
        batch_ids = input_ids.unsqueeze(0).expand(len(chunk), -1).clone()
        true_tokens = torch.tensor(
            [input_ids[pos].item() for pos in chunk], dtype=torch.long, device=device
        )

        for batch_idx, pos in enumerate(chunk):
            batch_ids[batch_idx, pos] = mask_token_id

        batch_ids = batch_ids.to(device)
        batch_attn = attention_mask.unsqueeze(0).expand(len(chunk), -1).to(device)

        with torch.no_grad():
            logits = model(input_ids=batch_ids, attention_mask=batch_attn).logits

        log_probs = F.log_softmax(logits, dim=-1)
        for batch_idx, pos in enumerate(chunk):
            total_log_prob += log_probs[batch_idx, pos, true_tokens[batch_idx]].item()

    seq_len = len(maskable)
    return {
        "pll": total_log_prob,
        "pll_normalized": total_log_prob / seq_len,
        "was_truncated": was_truncated,
    }


def compute_pll_batch(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    sequences: list[str],
    device: str = "cuda",
    batch_size: int = 64,
    show_progress: bool = True,
) -> list[dict[str, float]]:
    """Compute exact PLL for multiple sequences.

    Each sequence is processed independently (L forward passes per sequence,
    batched internally).

    Returns:
        List of dicts, one per sequence, each with 'pll' and 'pll_normalized'.
    """
    results = []
    iterator = tqdm(sequences, desc="Computing PLL") if show_progress else sequences
    for seq in iterator:
        results.append(compute_pll(model, tokenizer, seq, device, batch_size))
    return results
