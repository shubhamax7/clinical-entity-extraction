"""Bio_ClinicalBERT Token Classification Training Pipeline.

Designed for Google Colab GPU training with Hugging Face Trainer.
Uses exact 19 BIO labels, seqeval entity-level F1 for checkpoint selection,
early stopping, mixed precision on CUDA, and held-out test evaluation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Support running directly or as part of the package
sys.path.insert(0, str(Path(__file__).parent))

from evaluator import (
    build_compute_metrics,
    evaluate_predictions,
    save_evaluation_report,
)
from preprocessing import (
    ENTITY_TYPES,
    IGNORE_INDEX,
    MODEL_NAME,
    build_label_maps,
    encode_record,
    load_jsonl,
    load_tokenizer,
    validate_records,
)

logger = logging.getLogger("clinical_ner_train")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ClinicalNERDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed clinical NER records."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        label2id: dict[str, int],
        max_length: int = 512,
    ) -> None:
        self.examples: list[dict[str, Any]] = []
        for record in records:
            enc = encode_record(
                record=record,
                tokenizer=tokenizer,
                label2id=label2id,
                max_length=max_length,
            )
            self.examples.append({
                "input_ids": enc.input_ids,
                "attention_mask": enc.attention_mask,
                "labels": enc.label_ids,
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Configuration & CLI Arguments
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Configuration parameters for Clinical NER training."""

    train_file: str = "datasets/syn_clinner_v1/data/train.jsonl"
    val_file: str = "datasets/syn_clinner_v1/data/validation.jsonl"
    test_file: str = "datasets/syn_clinner_v1/data/test.jsonl"
    model_name: str = MODEL_NAME
    output_dir: str = "artifacts/clinicalbert_ner"
    max_length: int = 512
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    early_stopping_patience: int = 3
    fp16: bool = False
    bf16: bool = False
    save_total_limit: int = 2
    logging_steps: int = 10
    resume_from_checkpoint: str | None = None
    do_test: bool = True
    local_files_only: bool = False


def parse_args(args: list[str] | None = None) -> TrainConfig:
    """Parse command line arguments into TrainConfig."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Bio_ClinicalBERT for Clinical NER on GPU."
    )
    parser.add_argument("--train_file", type=str, default="datasets/syn_clinner_v1/data/train.jsonl")
    parser.add_argument("--val_file", type=str, default="datasets/syn_clinner_v1/data/validation.jsonl")
    parser.add_argument("--test_file", type=str, default="datasets/syn_clinner_v1/data/test.jsonl")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default="artifacts/clinicalbert_ner")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 mixed precision on CUDA")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision on compatible GPUs")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--no_test", action="store_false", dest="do_test", help="Skip final test evaluation")
    parser.add_argument("--local_files_only", action="store_true", help="Only load tokenizer from local cache (offline mode). Default False.")

    parsed = parser.parse_args(args)
    return TrainConfig(
        train_file=parsed.train_file,
        val_file=parsed.val_file,
        test_file=parsed.test_file,
        model_name=parsed.model_name,
        output_dir=parsed.output_dir,
        max_length=parsed.max_length,
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        warmup_ratio=parsed.warmup_ratio,
        seed=parsed.seed,
        early_stopping_patience=parsed.early_stopping_patience,
        fp16=parsed.fp16,
        bf16=parsed.bf16,
        save_total_limit=parsed.save_total_limit,
        logging_steps=parsed.logging_steps,
        resume_from_checkpoint=parsed.resume_from_checkpoint,
        do_test=parsed.do_test,
        local_files_only=parsed.local_files_only,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_train_records(train_file: str | Path) -> int:
    """Count non-empty lines in a JSONL training file.

    Raises
    ------
    FileNotFoundError:
        If train_file does not exist.
    ValueError:
        If train_file is empty or contains no records.
    """
    path = Path(train_file)
    if not path.is_file():
        raise FileNotFoundError(f"Training file '{path}' does not exist.")

    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1

    if count == 0:
        raise ValueError(f"Training file '{path}' is empty or contains no records.")
    return count


def calculate_warmup_steps(
    num_records: int,
    batch_size: int,
    epochs: int,
    warmup_ratio: float,
) -> int:
    """Calculate warmup steps dynamically from records, batch size, epochs, and ratio.

    Formula:
        steps_per_epoch = ceil(num_records / batch_size)
        total_training_steps = steps_per_epoch * epochs
        warmup_steps = ceil(total_training_steps * warmup_ratio)
    """
    if num_records <= 0:
        raise ValueError(f"Invalid num_records={num_records}; must have at least 1 record.")
    if batch_size <= 0:
        raise ValueError(f"Invalid batch_size={batch_size}; must be > 0.")
    if epochs <= 0:
        raise ValueError(f"Invalid epochs={epochs}; must be > 0.")
    if not (0.0 <= warmup_ratio <= 1.0):
        raise ValueError(f"Invalid warmup_ratio={warmup_ratio}; must be between 0.0 and 1.0.")

    steps_per_epoch = math.ceil(num_records / batch_size)
    total_training_steps = steps_per_epoch * epochs
    warmup_steps = math.ceil(total_training_steps * warmup_ratio)
    return warmup_steps


def build_training_args_dict(
    config: TrainConfig,
    num_train_records: int | None = None,
) -> dict[str, Any]:
    """Construct keyword arguments for TrainingArguments with dynamic warmup_steps."""
    # Detect whether 'eval_strategy' or 'evaluation_strategy' is used
    sig = inspect.signature(TrainingArguments.__init__)
    eval_arg_name = "eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"

    if num_train_records is None:
        num_train_records = count_train_records(config.train_file)

    warmup_steps = calculate_warmup_steps(
        num_records=num_train_records,
        batch_size=config.batch_size,
        epochs=config.epochs,
        warmup_ratio=config.warmup_ratio,
    )

    kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        eval_arg_name: "epoch",
        "save_strategy": "epoch",
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "num_train_epochs": config.epochs,
        "weight_decay": config.weight_decay,
        "warmup_steps": warmup_steps,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "load_best_model_at_end": True,
        "save_total_limit": config.save_total_limit,
        "logging_steps": config.logging_steps,
        "seed": config.seed,
        "optim": "adamw_torch",
        "report_to": "none",
    }

    # Auto-detect or respect mixed precision: BF16 only when CUDA device supports it
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        cuda_supports_bf16 = bool(hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
        if config.bf16:
            if cuda_supports_bf16:
                kwargs["bf16"] = True
                logger.info("BF16 mixed precision enabled on CUDA device.")
            else:
                kwargs["fp16"] = True
                logger.warning(
                    "BF16 was requested, but current CUDA device does not support BF16. "
                    "Falling back to FP16 mixed precision."
                )
        elif config.fp16:
            kwargs["fp16"] = True
            logger.info("FP16 mixed precision enabled on CUDA device.")
        else:
            # Default auto-selection on CUDA: BF16 if supported, otherwise FP16
            if cuda_supports_bf16:
                kwargs["bf16"] = True
                logger.info("Auto-enabled BF16 mixed precision on compatible CUDA device.")
            else:
                kwargs["fp16"] = True
                logger.info("Auto-enabled FP16 mixed precision on CUDA device.")

    return kwargs


def create_training_arguments(
    config: TrainConfig,
    num_train_records: int | None = None,
) -> TrainingArguments:
    """Create TrainingArguments with backward/forward compatibility and dynamic warmup_steps."""
    kwargs = build_training_args_dict(config, num_train_records=num_train_records)
    return TrainingArguments(**kwargs)


def resolve_resume_checkpoint(
    resume_from_checkpoint: str | bool | None,
    output_dir: str | Path,
) -> str | None:
    """Resolve resume_from_checkpoint argument to a verified checkpoint path.

    Parameters
    ----------
    resume_from_checkpoint:
        - None or False or "" -> returns None (fresh training).
        - True or string in {"true", "1", "latest"} -> resolves latest checkpoint in output_dir.
        - Path to an explicit checkpoint dir containing trainer_state.json -> returns path.
        - Path to an output dir containing checkpoint-* subdirs -> resolves latest checkpoint.
    output_dir:
        Output directory containing checkpoints when resuming with True / latest.

    Returns
    -------
    str | None:
        The verified checkpoint directory path, or None to start from scratch.

    Raises
    ------
    FileNotFoundError:
        If an explicit checkpoint path does not exist, or does not contain a valid checkpoint.
    """
    if resume_from_checkpoint is None or resume_from_checkpoint is False or resume_from_checkpoint == "":
        return None

    from transformers.trainer import TRAINER_STATE_NAME
    from transformers.trainer_utils import get_last_checkpoint

    # Boolean True or string representation of True / latest / 1
    if isinstance(resume_from_checkpoint, bool) or str(resume_from_checkpoint).strip().lower() in ("true", "1", "latest"):
        out_path = Path(output_dir)
        if not out_path.exists():
            logger.warning("Output directory %s does not exist. Starting training from scratch.", output_dir)
            return None
        last_cp = get_last_checkpoint(str(out_path))
        if last_cp is None:
            logger.warning("No checkpoint found in output directory %s. Starting training from scratch.", output_dir)
            return None
        logger.info("Resuming training from latest checkpoint in %s: %s", output_dir, last_cp)
        return last_cp

    # Explicit path provided as string
    candidate = Path(str(resume_from_checkpoint).strip())
    if not candidate.exists():
        raise FileNotFoundError(
            f"Checkpoint path '{candidate}' does not exist. Cannot resume training."
        )

    # Path directly contains trainer_state.json
    if (candidate / TRAINER_STATE_NAME).exists():
        logger.info("Resuming training from explicit checkpoint: %s", candidate)
        return str(candidate)

    # Path is a directory containing checkpoint-* subdirectories (e.g. parent output_dir)
    if candidate.is_dir():
        last_cp = get_last_checkpoint(str(candidate))
        if last_cp is not None and (Path(last_cp) / TRAINER_STATE_NAME).exists():
            logger.info("Resolved parent directory '%s' to latest checkpoint: %s", candidate, last_cp)
            return str(last_cp)

    raise FileNotFoundError(
        f"Specified resume_from_checkpoint='{resume_from_checkpoint}' is not a valid checkpoint directory "
        f"(missing '{TRAINER_STATE_NAME}') and contains no valid checkpoint subdirectories."
    )


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------


def train_model(config: TrainConfig) -> dict[str, Any]:
    """Execute the fine-tuning workflow on GPU."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Starting Bio_ClinicalBERT NER training pipeline")
    logger.info("Configuration: %s", asdict(config))

    # 1. Set seed for reproducibility
    set_seed(config.seed)

    # 2. Check execution device
    device_name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Target execution device: %s", device_name)
    if torch.cuda.is_available():
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))
        logger.info("CUDA device count: %d", torch.cuda.device_count())

    # 3. Label mappings (19 BIO labels)
    labels, label2id, id2label = build_label_maps()
    logger.info("Deterministic label vocabulary initialized with %d labels", len(labels))

    # 4. Load & validate dataset splits
    logger.info("Loading dataset splits...")
    train_records = load_jsonl(config.train_file)
    val_records = load_jsonl(config.val_file)
    test_records = load_jsonl(config.test_file)

    validate_records(train_records)
    validate_records(val_records)
    validate_records(test_records)
    logger.info(
        "Records loaded and validated: train=%d, val=%d, test=%d",
        len(train_records),
        len(val_records),
        len(test_records),
    )

    # 5. Tokenizer
    logger.info("Loading tokenizer: %s (local_files_only=%s)", config.model_name, config.local_files_only)
    tokenizer = (
        load_tokenizer(local_files_only=config.local_files_only)
        if config.model_name == MODEL_NAME
        else AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=True,
            local_files_only=config.local_files_only,
        )
    )

    # 6. Build Datasets
    logger.info("Encoding records into token datasets (max_length=%d)...", config.max_length)
    train_dataset = ClinicalNERDataset(train_records, tokenizer, label2id, config.max_length)
    val_dataset = ClinicalNERDataset(val_records, tokenizer, label2id, config.max_length)
    test_dataset = ClinicalNERDataset(test_records, tokenizer, label2id, config.max_length)

    # 7. Model
    logger.info("Initializing AutoModelForTokenClassification with %d labels", len(labels))
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # 8. Training arguments & callbacks
    training_args = create_training_arguments(config, num_train_records=len(train_records))
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX)
    compute_metrics_fn = build_compute_metrics(id2label)

    callbacks = []
    if config.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))
        logger.info("Early stopping enabled with patience=%d", config.early_stopping_patience)

    # 9. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )

    # 10. Train
    resume_checkpoint = resolve_resume_checkpoint(
        resume_from_checkpoint=config.resume_from_checkpoint,
        output_dir=config.output_dir,
    )
    if resume_checkpoint:
        logger.info("Executing training loop (resuming from: %s)...", resume_checkpoint)
    else:
        logger.info("Executing training loop (from scratch)...")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 11. Save best model and tokenizer together
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving best model and tokenizer to: %s", out_dir)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # 12. Evaluate best model on Validation Set
    logger.info("Running evaluation on validation set with best model...")
    val_eval = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    val_preds = trainer.predict(val_dataset)
    val_res = evaluate_predictions(val_preds.predictions, val_preds.label_ids, id2label)
    save_evaluation_report(val_res, out_dir, prefix="validation")

    # 13. Evaluate best model on Held-Out Test Set (NEVER used for model selection)
    test_eval: dict[str, Any] = {}
    if config.do_test:
        logger.info("Running evaluation on held-out test set with best model...")
        test_eval = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        test_preds = trainer.predict(test_dataset)
        test_res = evaluate_predictions(test_preds.predictions, test_preds.label_ids, id2label)
        save_evaluation_report(test_res, out_dir, prefix="test")

    # 14. Save metadata
    metadata = {
        "model_name": config.model_name,
        "num_labels": len(labels),
        "labels": labels,
        "id2label": id2label,
        "config": asdict(config),
        "device": device_name,
        "train_metrics": train_result.metrics,
        "val_metrics": val_eval,
        "test_metrics": test_eval,
    }
    meta_path = out_dir / "training_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    logger.info("Training metadata saved to: %s", meta_path)

    logger.info("Training completed successfully!")
    return metadata


if __name__ == "__main__":
    cfg = parse_args()
    train_model(cfg)

