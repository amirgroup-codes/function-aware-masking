"""Model factory: build antibody MLM models from config.

Uses the AntiBERTa2 architecture (RoFormer) with either:
  - Random initialization (for controlled masking experiments)
  - Pretrained weights (for reference comparisons)

Three named sizes are available for random-init training:
  - "small"  : 6L, 256h, 8 heads,  ~4.8M params (pipeline validation)
  - "medium" : 12L, 512h, 8 heads, ~50M params  (main experiments)
  - "full"   : 16L, 1024h, 16 heads, ~202M params (matches AntiBERTa2)
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import RoFormerConfig, RoFormerForMaskedLM

_SHARED_DEFAULTS = dict(
    vocab_size=28,  # AntiBERTa2 base tokenizer has exactly 28 tokens
    max_position_embeddings=256,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    type_vocab_size=1,  # token_type_ids never passed → no need for >1
    pad_token_id=0,
)

# Defaults for multispecific models with extended vocab and positions
_MULTISPECIFIC_DEFAULTS = dict(
    vocab_size=32,  # 28 base + 4 added special tokens ([MOD1], [MOD2], [H], [L])
    max_position_embeddings=512,  # up from 256 for ~384-token paired sequences
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    # chain_type_ids are returned by the dataset but never forwarded to the
    # model as token_type_ids, so a >1 type vocab is dead capacity. Chain
    # identity is encoded via the [H] / [L] framing tokens instead.
    type_vocab_size=1,
    pad_token_id=0,
)


@dataclass
class ModelSpec:
    """Architecture specification for a RoFormer model size."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(hidden_size=256, num_hidden_layers=6, num_attention_heads=8, intermediate_size=1024),
    "medium": ModelSpec(hidden_size=512, num_hidden_layers=12, num_attention_heads=8, intermediate_size=2048),
    "full": ModelSpec(hidden_size=1024, num_hidden_layers=16, num_attention_heads=16, intermediate_size=4096),
}


def build_model(
    model_name: str = "alchemab/antiberta2",
    from_pretrained: bool = False,
    model_size: str = "small",
) -> RoFormerForMaskedLM:
    """Build a RoFormer masked language model.

    Args:
        model_name: HuggingFace model identifier for loading config/weights.
        from_pretrained: If True, load pretrained weights. If False, random init.
        model_size: One of "small", "medium", "full". Only used when
                    from_pretrained is False.

    Returns:
        A RoFormerForMaskedLM instance.

    Raises:
        KeyError: If model_size is not a recognized size name.
    """
    if from_pretrained:
        return RoFormerForMaskedLM.from_pretrained(model_name)

    if model_size not in MODEL_SPECS:
        available = ", ".join(sorted(MODEL_SPECS.keys()))
        raise KeyError(f"Unknown model_size '{model_size}'. Available: {available}")

    spec = MODEL_SPECS[model_size]
    config = RoFormerConfig(
        hidden_size=spec.hidden_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        intermediate_size=spec.intermediate_size,
        **_SHARED_DEFAULTS,
    )
    return RoFormerForMaskedLM(config)


def build_multispecific_model(
    model_name: str = "alchemab/antiberta2",
    from_pretrained: bool = False,
    model_size: str = "medium",
    num_added_tokens: int = 4,
) -> RoFormerForMaskedLM:
    """Build a RoFormer model with extended vocab and positions for multispecific.

    Uses type_vocab_size=3 for chain-type embeddings (special/heavy/light).
    Module identity is encoded via position + delimiter tokens ([MOD1]/[MOD2]).

    Args:
        model_name: HuggingFace model identifier.
        from_pretrained: If True, load pretrained weights and resize embeddings.
        model_size: One of "small", "medium", "full".
        num_added_tokens: Number of new special tokens added to the tokenizer.

    Returns:
        A RoFormerForMaskedLM instance configured for multispecific sequences.
    """
    if from_pretrained:
        model = RoFormerForMaskedLM.from_pretrained(model_name)
        model.resize_token_embeddings(model.config.vocab_size + num_added_tokens)
        return model

    if model_size not in MODEL_SPECS:
        available = ", ".join(sorted(MODEL_SPECS.keys()))
        raise KeyError(f"Unknown model_size '{model_size}'. Available: {available}")

    spec = MODEL_SPECS[model_size]
    config = RoFormerConfig(
        hidden_size=spec.hidden_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        intermediate_size=spec.intermediate_size,
        **_MULTISPECIFIC_DEFAULTS,
    )
    return RoFormerForMaskedLM(config)
