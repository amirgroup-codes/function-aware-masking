"""Training orchestration for antibody MLM experiments.

Wraps HuggingFace Trainer with our config system, masking strategy
registry, and dataset/collator wiring. The train() function is the
single entry point — it takes an ExperimentConfig and runs the full
training pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import random_split
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from data.dataset import AntibodyDataset
from masking import MLMDataCollator, get_strategy
from models.model import build_model
from training.config import ExperimentConfig
from utils.seed import set_seed
from utils.tokenizer import load_tokenizer

logger = logging.getLogger(__name__)


def compute_mlm_metrics(eval_pred) -> dict[str, float]:
    """Compute MLM accuracy from Trainer's EvalPrediction.

    With ``preprocess_logits_for_metrics`` set on the Trainer, the first
    element of ``eval_pred`` is already an argmax (int tensor), not raw
    logits — saves ~32× CPU RAM during eval.
    """
    predictions, labels = eval_pred
    mask = labels != -100
    correct = (predictions[mask] == labels[mask]).sum()
    total = mask.sum()
    accuracy = correct / total if total > 0 else 0.0
    return {"mlm_accuracy": float(accuracy)}


def _argmax_for_metrics(logits, labels):
    """preprocess_logits_for_metrics callback: drop the vocab dimension before
    HF Trainer accumulates eval-batch tensors in CPU RAM."""
    return logits.argmax(dim=-1)


def train(config: ExperimentConfig) -> Path:
    """Run a full MLM training experiment.

    Args:
        config: Typed experiment configuration.

    Returns:
        Path to the saved model checkpoint directory.
    """
    set_seed(config.seed)

    if config.data.paired:
        from data.dataset_paired import PairedAntibodyDataset
        from models.model import build_multispecific_model
        from utils.tokenizer import load_tokenizer_multispecific

        logger.info("Loading multispecific tokenizer: %s", config.model.model_name)
        tokenizer = load_tokenizer_multispecific(config.model.model_name)

        logger.info("Building multispecific model (from_pretrained=%s, model_size=%s)", config.model.from_pretrained, config.model.model_size)
        model = build_multispecific_model(
            model_name=config.model.model_name,
            from_pretrained=config.model.from_pretrained,
            model_size=config.model.model_size,
        )
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model parameters: %s", f"{total_params:,}")

        logger.info("Loading paired dataset: %s", config.data.processed_path)
        full_dataset = PairedAntibodyDataset(
            data_path=config.data.processed_path,
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            paratope_path=config.data.paratope_path or None,
            interface_path=config.data.interface_path or None,
            germline_path=config.data.germline_path or None,
            bispecific=config.data.bispecific,
        )
    else:
        logger.info("Loading tokenizer: %s", config.model.model_name)
        tokenizer = load_tokenizer(config.model.model_name)

        logger.info("Building model (from_pretrained=%s, model_size=%s)", config.model.from_pretrained, config.model.model_size)
        model = build_model(
            model_name=config.model.model_name,
            from_pretrained=config.model.from_pretrained,
            model_size=config.model.model_size,
        )
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model parameters: %s", f"{total_params:,}")

        logger.info("Loading dataset: %s", config.data.processed_path)
        full_dataset = AntibodyDataset(
            data_path=config.data.processed_path,
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            coords_path=config.data.coords_path or None,
            paratope_path=config.data.paratope_path or None,
            germline_path=config.data.germline_path or None,
        )

    train_size = int(len(full_dataset) * config.data.train_split)
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    logger.info("Train: %d, Eval: %d", len(train_dataset), len(eval_dataset))

    logger.info("Setting up masking strategy: %s", config.masking.strategy)
    strategy = get_strategy(
        config.masking.strategy,
        tokenizer=tokenizer,
        mask_prob=config.masking.mask_prob,
        mask_token_ratio=config.masking.mask_token_ratio,
        random_token_ratio=config.masking.random_token_ratio,
        **config.masking.params,
    )
    collator = MLMDataCollator(tokenizer=tokenizer, strategy=strategy)

    # Wire curriculum callback for hybrid masking
    hybrid_callback = None
    try:
        from masking.hybrid import HybridMasking
        from training.callbacks import HybridMaskingCallback
        if isinstance(strategy, HybridMasking):
            hybrid_callback = HybridMaskingCallback(strategy)
            logger.info("Hybrid masking curriculum callback registered")
    except ImportError:
        pass

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If max_steps > 0, HF Trainer ignores num_train_epochs and stops at
    # exactly max_steps optimizer updates. Use this for fair comparison.
    use_max_steps = config.training.max_steps > 0
    if use_max_steps:
        logger.info(
            "max_steps=%d set; training will stop at exactly %d optimizer steps "
            "(num_epochs=%d ignored)",
            config.training.max_steps, config.training.max_steps, config.training.num_epochs,
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.training.num_epochs,
        max_steps=config.training.max_steps if use_max_steps else -1,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        lr_scheduler_type="cosine",  # smooth decay vs HF default linear
        weight_decay=config.training.weight_decay,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        fp16=config.training.fp16,
        dataloader_num_workers=config.training.dataloader_num_workers,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        save_total_limit=2,
        # For equal-step comparison we report the FINAL checkpoint at exactly
        # max_steps for every model. load_best_model_at_end would pick a
        # different intermediate checkpoint per model based on a noisy eval
        # metric, defeating the equal-compute fairness goal.
        load_best_model_at_end=False,
        seed=config.seed,
        data_seed=42,  # decouple dataloader RNG from per-model seed (Fix 4.3/4.4)
        report_to="none",
        remove_unused_columns=False,
    )

    callbacks = []
    if config.training.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.training.early_stopping_patience,
            )
        )
        logger.info("Early stopping enabled (patience=%d evals)", config.training.early_stopping_patience)

    if hybrid_callback is not None:
        callbacks.append(hybrid_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_mlm_metrics,
        preprocess_logits_for_metrics=_argmax_for_metrics,
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Model saved to %s", final_dir)

    final_metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", final_metrics)

    eval_history = [
        entry for entry in trainer.state.log_history if "eval_loss" in entry
    ]
    train_history = [
        entry
        for entry in trainer.state.log_history
        if "loss" in entry and "eval_loss" not in entry
    ]
    summary = {
        "experiment": {
            "model_size": config.model.model_size,
            "from_pretrained": config.model.from_pretrained,
            "masking_strategy": config.masking.strategy,
            "mask_prob": config.masking.mask_prob,
            "dataset": config.data.processed_path,
            "num_epochs": config.training.num_epochs,
            "effective_batch_size": (
                config.training.batch_size
                * config.training.gradient_accumulation_steps
            ),
            "learning_rate": config.training.learning_rate,
            "seed": config.seed,
            "total_params": total_params,
        },
        "final_metrics": final_metrics,
        "eval_history": eval_history,
        "train_history": train_history,
    }
    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training summary saved to %s", summary_path)

    return final_dir
