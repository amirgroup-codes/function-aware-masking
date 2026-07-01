"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    processed_path: str = "data/processed/oas_vh_tiny.jsonl"
    max_length: int = 160
    min_length: int = 80
    train_split: float = 0.9
    valid_amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"
    coords_path: str = ""
    paratope_path: str = ""
    germline_path: str = ""
    interface_path: str = ""
    paired: bool = False
    bispecific: bool = False


@dataclass
class MaskingConfig:
    """Configuration for masking strategy selection and parameters."""

    strategy: str = "uniform"
    mask_prob: float = 0.15
    mask_token_ratio: float = 0.8
    random_token_ratio: float = 0.1
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    model_name: str = "alchemab/antiberta2"
    from_pretrained: bool = False
    model_size: str = "small"


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    output_dir: str = "models/checkpoints/uniform_baseline"
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 3
    # When > 0, training stops at exactly this many optimizer steps and
    # ``num_epochs`` is ignored. Use this for fair cross-strategy comparison
    # so every model sees the same amount of compute.
    max_steps: int = 0
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 250
    fp16: bool = True
    dataloader_num_workers: int = 4
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 0


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file.

    Nested dicts are mapped to the corresponding dataclass fields.
    Unspecified fields use their defaults.
    """
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    data_cfg = DataConfig(**raw.get("data", {}))
    masking_cfg = MaskingConfig(**raw.get("masking", {}))
    model_cfg = ModelConfig(**raw.get("model", {}))
    training_cfg = TrainingConfig(**raw.get("training", {}))

    return ExperimentConfig(
        seed=raw.get("seed", 42),
        data=data_cfg,
        masking=masking_cfg,
        model=model_cfg,
        training=training_cfg,
    )
