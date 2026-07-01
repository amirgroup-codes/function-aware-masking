"""Tokenizer loading and helpers for antibody sequences.

Uses AntiBERTa2's tokenizer (per-amino-acid vocabulary with 28 tokens).
Sequences must be space-separated before tokenization (e.g., "E V Q L V E S").
"""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase

ANTIBERTA2_MODEL_NAME = "alchemab/antiberta2"

# Additional special tokens for multispecific multi-module format
MULTISPECIFIC_SPECIAL_TOKENS = ["[MOD1]", "[MOD2]", "[H]", "[L]"]


def load_tokenizer(model_name: str = ANTIBERTA2_MODEL_NAME) -> PreTrainedTokenizerBase:
    """Load the AntiBERTa2 tokenizer from HuggingFace.

    Returns a tokenizer with special tokens [CLS], [SEP], [MASK], [PAD]
    and a per-amino-acid vocabulary (vocab_size=28).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def tokenize_sequence(sequence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    """Convert a raw amino acid string into space-separated format for the tokenizer."""
    return " ".join(list(sequence))


def is_paired_checkpoint(checkpoint_path: str) -> bool:
    """Return True if the given checkpoint was trained with the paired
    multispecific tokenizer (vocab_size large enough to include the
    [MOD1]/[MOD2]/[H]/[L] special tokens at IDs 28-31).

    Single-chain antiberta2 checkpoints have vocab_size 28-30 (sometimes
    padded for alignment); paired checkpoints have vocab_size >= 32.
    """
    from transformers import RoFormerConfig

    cfg = RoFormerConfig.from_pretrained(checkpoint_path)
    return cfg.vocab_size >= 32


def load_tokenizer_for_checkpoint(
    checkpoint_path: str, model_name: str = ANTIBERTA2_MODEL_NAME,
) -> PreTrainedTokenizerBase:
    """Load the tokenizer matching the checkpoint's training setup.

    Paired checkpoints get the multispecific tokenizer (with [MOD1]/[H]/[L]
    framing tokens); single-chain checkpoints get the standard tokenizer.
    """
    if is_paired_checkpoint(checkpoint_path):
        return load_tokenizer_multispecific(model_name)
    return load_tokenizer(model_name)


def tokenize_single_chain(
    tokenizer: PreTrainedTokenizerBase,
    sequence: str,
    max_length: int,
) -> dict:
    """Tokenize a single VH (or single-chain) sequence.

    If the tokenizer has the multispecific framing tokens ([MOD1]/[H]/[L]),
    produces ``[CLS] [MOD1] [H] VH... [SEP]`` via :func:`encode_multispecific`
    so that paired models see the framing tokens they were trained with.
    Otherwise produces the standard ``[CLS] VH... [SEP]`` via the plain
    tokenizer call.

    Returns a dict with ``input_ids``, ``attention_mask``, and
    ``special_tokens_mask`` (matching the HuggingFace tokenizer format).
    """
    additional = tokenizer.additional_special_tokens or []
    if "[MOD1]" in additional:
        enc = encode_multispecific(
            vh_1=sequence, vl_1=None,
            tokenizer=tokenizer, max_length=max_length,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "special_tokens_mask": enc["special_tokens_mask"],
        }
    spaced = " ".join(list(sequence))
    return tokenizer(
        spaced,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_special_tokens_mask=True,
    )


def load_tokenizer_multispecific(
    model_name: str = ANTIBERTA2_MODEL_NAME,
) -> PreTrainedTokenizerBase:
    """Load tokenizer with additional special tokens for multi-module format.

    Adds [MOD1], [MOD2], [H], [L] tokens for module and chain-type
    delimiters used in paired/multispecific antibody representation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({
        "additional_special_tokens": MULTISPECIFIC_SPECIAL_TOKENS,
    })
    return tokenizer


def encode_multispecific(
    vh_1: str,
    vl_1: str | None,
    vh_2: str | None = None,
    vl_2: str | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    max_length: int = 384,
) -> dict:
    """Encode a paired/multispecific antibody into multi-module token format.

    Monospecific paired (K=1):
      [CLS] [MOD1] [H] VH_1... [SEP] [L] VL_1... [SEP]

    Heavy-chain only (vl_1=None, used for evaluating paired models on
    single-chain benchmarks):
      [CLS] [MOD1] [H] VH_1... [SEP]

    Bispecific (K=2):
      [CLS] [MOD1] [H] VH_1... [SEP] [L] VL_1... [SEP]
             [MOD2] [H] VH_2... [SEP] [L] VL_2... [SEP]

    Returns dict with:
      input_ids: list[int]
      attention_mask: list[int]
      special_tokens_mask: list[int]
      module_ids: list[int] (0=global/CLS, 1=mod1, 2=mod2)
      chain_type_ids: list[int] (0=special, 1=heavy, 2=light)
      aa_to_token_map: dict mapping (module_idx, chain, aa_pos) -> token_pos
    """
    # Resolve special token IDs
    additional = tokenizer.additional_special_tokens
    additional_ids = tokenizer.additional_special_tokens_ids
    special_map = dict(zip(additional, additional_ids))

    mod1_id = special_map["[MOD1]"]
    mod2_id = special_map["[MOD2]"]
    h_id = special_map["[H]"]
    l_id = special_map["[L]"]
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # Tokenize each chain segment (no special tokens — we insert them manually)
    def _tokenize_chain(sequence: str) -> list[int]:
        spaced = " ".join(list(sequence))
        return tokenizer.encode(spaced, add_special_tokens=False)

    vh1_ids = _tokenize_chain(vh_1)
    vl1_ids = _tokenize_chain(vl_1) if vl_1 is not None else None

    # Build token sequence and metadata arrays
    input_ids: list[int] = []
    module_ids: list[int] = []
    chain_type_ids: list[int] = []
    special_tokens_mask: list[int] = []
    aa_to_token_map: dict[tuple[int, str, int], int] = {}

    def _append_special(token_id: int, mod: int, chain: int) -> None:
        input_ids.append(token_id)
        module_ids.append(mod)
        chain_type_ids.append(chain)
        special_tokens_mask.append(1)

    def _append_chain(token_ids: list[int], mod: int, chain: int,
                      chain_label: str) -> None:
        for aa_pos, tid in enumerate(token_ids):
            pos = len(input_ids)
            aa_to_token_map[(mod, chain_label, aa_pos)] = pos
            input_ids.append(tid)
            module_ids.append(mod)
            chain_type_ids.append(chain)
            special_tokens_mask.append(0)

    # [CLS]
    _append_special(cls_id, 0, 0)

    # Module 1: [MOD1] [H] VH_1... [SEP] [L] VL_1... [SEP]
    # (If vl_1 is None, the light-chain section is omitted entirely.)
    _append_special(mod1_id, 1, 0)
    _append_special(h_id, 1, 0)
    _append_chain(vh1_ids, 1, 1, "heavy")
    _append_special(sep_id, 1, 0)
    if vl1_ids is not None:
        _append_special(l_id, 1, 0)
        _append_chain(vl1_ids, 1, 2, "light")
        _append_special(sep_id, 1, 0)

    # Module 2 (bispecific): [MOD2] [H] VH_2... [SEP] [L] VL_2... [SEP]
    if vh_2 is not None and vl_2 is not None:
        vh2_ids = _tokenize_chain(vh_2)
        vl2_ids = _tokenize_chain(vl_2)

        _append_special(mod2_id, 2, 0)
        _append_special(h_id, 2, 0)
        _append_chain(vh2_ids, 2, 1, "heavy")
        _append_special(sep_id, 2, 0)
        _append_special(l_id, 2, 0)
        _append_chain(vl2_ids, 2, 2, "light")
        _append_special(sep_id, 2, 0)

    # Truncate to max_length if needed
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        module_ids = module_ids[:max_length]
        chain_type_ids = chain_type_ids[:max_length]
        special_tokens_mask = special_tokens_mask[:max_length]
        # Remove aa_to_token_map entries beyond max_length
        aa_to_token_map = {
            k: v for k, v in aa_to_token_map.items() if v < max_length
        }

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "special_tokens_mask": special_tokens_mask,
        "module_ids": module_ids,
        "chain_type_ids": chain_type_ids,
        "aa_to_token_map": aa_to_token_map,
    }
