#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_PARQUET = "../dataset/v1_0_release/v1_0_stable.parquet"

MODEL_KEY = "mdeberta"

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "mdeberta": {
        "model_name": "microsoft/mdeberta-v3-base",
        "output_dir": "outputs_silver_grade_only_mdeberta_v3_base",
        # Lower LR for stability. Previous 1e-5 caused NaN grad_norm.
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
    },
}

QUESTION_COL = "question"
ANSWER_COL = "answer"
GRADE_COL = "grade"
SPLIT_COL = "split"
LABEL_TYPE_COL = "label_type"

TRAIN_SPLIT_VALUE = "train"
TEST_SPLIT_VALUE = "test"

TRAIN_LABEL_TYPE = "silver"
TEST_LABEL_TYPE = "gold"

MAX_SEQ_LENGTH = 512
RANDOM_SEED = 42

NUM_TRAIN_EPOCHS = 2
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

LOGGING_STEPS = 10
SAVE_STEPS = 250
SAVE_TOTAL_LIMIT = 3

ALLOWED_GRADES = {0.0, 0.25, 0.5, 0.75, 1.0}


# =========================================================
# PATHS
# =========================================================

if MODEL_KEY not in MODEL_CONFIGS:
    raise ValueError(f"Unknown MODEL_KEY: {MODEL_KEY}. Available: {list(MODEL_CONFIGS)}")

MODEL_NAME = MODEL_CONFIGS[MODEL_KEY]["model_name"]
OUTPUT_DIR = (BASE_DIR / MODEL_CONFIGS[MODEL_KEY]["output_dir"]).resolve()
LOG_FILE = OUTPUT_DIR / "training.log"


# =========================================================
# HELPERS
# =========================================================

def open_parquet_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Expected parquet file, got: {path}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    return pd.read_parquet(path)


def normalize_grade(value) -> float | None:
    if pd.isna(value):
        return None

    try:
        value = float(value)
    except Exception:
        return None

    if value not in ALLOWED_GRADES:
        return None

    return value


def build_encoder_input(question: str, answer: str) -> str:
    question = "" if pd.isna(question) else str(question).strip()
    answer = "" if pd.isna(answer) else str(answer).strip()

    return (
        f"QUESTION: {question} "
        f"[SEP] STUDENT_ANSWER: {answer}"
    )


def prepare_train_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [QUESTION_COL, ANSWER_COL, GRADE_COL, SPLIT_COL, LABEL_TYPE_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out[GRADE_COL] = out[GRADE_COL].apply(normalize_grade)

    train_df = out[
        (out[SPLIT_COL] == TRAIN_SPLIT_VALUE) &
        (out[LABEL_TYPE_COL] == TRAIN_LABEL_TYPE) &
        (out[GRADE_COL].notna())
    ].copy()

    if train_df.empty:
        raise ValueError("No usable silver train rows found.")

    train_df["text"] = train_df.apply(
        lambda row: build_encoder_input(
            question=row[QUESTION_COL],
            answer=row[ANSWER_COL],
        ),
        axis=1,
    )

    # Important: regression labels as float32.
    train_df["labels"] = train_df[GRADE_COL].astype(np.float32)

    return train_df


def prepare_reserved_test_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [QUESTION_COL, ANSWER_COL, GRADE_COL, SPLIT_COL, LABEL_TYPE_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out[GRADE_COL] = out[GRADE_COL].apply(normalize_grade)

    test_df = out[
        (out[SPLIT_COL] == TEST_SPLIT_VALUE) &
        (out[LABEL_TYPE_COL] == TEST_LABEL_TYPE) &
        (out[GRADE_COL].notna())
    ].copy()

    if test_df.empty:
        raise ValueError("No usable gold test rows found.")

    return test_df


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        tokenized["labels"] = [np.float32(x).item() for x in batch["labels"]]
        return tokenized

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )


def get_latest_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None

    checkpoints = []
    for path in output_dir.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint-"):
            step_str = path.name.replace("checkpoint-", "")
            if step_str.isdigit():
                checkpoints.append((int(step_str), path))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return str(checkpoints[-1][1])


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = logits.squeeze()
    labels = labels.squeeze()

    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    predictions_clipped = np.clip(predictions, 0.0, 1.0)

    mse = float(np.mean((predictions_clipped - labels) ** 2))
    mae = float(np.mean(np.abs(predictions_clipped - labels)))

    return {
        "mse": mse,
        "mae": mae,
    }


# =========================================================
# CALLBACKS
# =========================================================

class FileLoggerCallback(TrainerCallback):
    def __init__(self, log_file: Path):
        self.log_file = Path(log_file)

    def _write(self, message: str) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self._write("Training started.")
        self._write(f"Model: {MODEL_NAME}")
        self._write(f"Output dir: {args.output_dir}")
        self._write(f"Max steps: {state.max_steps}")
        self._write(f"Num train epochs: {getattr(args, 'num_train_epochs', 'unknown')}")
        self._write(
            f"Save strategy: {args.save_strategy} | "
            f"save_steps: {getattr(args, 'save_steps', 'n/a')} | "
            f"save_total_limit: {args.save_total_limit}"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero:
            return

        logs = dict(logs or {})
        logs.pop("total_flos", None)

        parts = [
            f"step={state.global_step}/{state.max_steps}",
            f"epoch={logs.get('epoch', state.epoch)}",
        ]

        if "loss" in logs:
            parts.append(f"loss={logs['loss']}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']}")
        if "grad_norm" in logs:
            parts.append(f"grad_norm={logs['grad_norm']}")
        if "train_loss" in logs:
            parts.append(f"train_loss={logs['train_loss']}")

        self._write(" | ".join(parts))

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self._write(
                f"Epoch ended: epoch={state.epoch}, "
                f"step={state.global_step}/{state.max_steps}. "
                "Forcing checkpoint save."
            )
        control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self._write(f"Checkpoint saved at global_step={state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        self._write("Training finished.")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"NEW RUN STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    df = open_parquet_file(INPUT_PARQUET)

    train_df = prepare_train_dataframe(df)
    test_df = prepare_reserved_test_dataframe(df)

    print("Dataset summary:")
    print(f"Model key:         {MODEL_KEY}")
    print(f"Model name:        {MODEL_NAME}")
    print(f"Silver train rows: {len(train_df)}")
    print(f"Gold test rows:    {len(test_df)}")
    print("Gold test is reserved and NOT used during training.")

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"MODEL_KEY: {MODEL_KEY}\n")
        f.write(f"MODEL_NAME: {MODEL_NAME}\n")
        f.write(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}\n")
        f.write(f"NUM_TRAIN_EPOCHS: {NUM_TRAIN_EPOCHS}\n")
        f.write(f"LEARNING_RATE: {MODEL_CONFIGS[MODEL_KEY]['learning_rate']}\n")
        f.write(f"PER_DEVICE_BATCH_SIZE: {MODEL_CONFIGS[MODEL_KEY]['per_device_train_batch_size']}\n")
        f.write(f"GRADIENT_ACCUMULATION_STEPS: {MODEL_CONFIGS[MODEL_KEY]['gradient_accumulation_steps']}\n")
        f.write(
            f"EFFECTIVE_BATCH_SIZE: "
            f"{MODEL_CONFIGS[MODEL_KEY]['per_device_train_batch_size'] * MODEL_CONFIGS[MODEL_KEY]['gradient_accumulation_steps']}\n"
        )
        f.write("MIXED_PRECISION: disabled fp16=False bf16=False\n")
        f.write(f"Silver train rows: {len(train_df)}\n")
        f.write(f"Gold test rows: {len(test_df)}\n")
        f.write("Gold test is reserved and NOT used during training.\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression",
        torch_dtype=torch.float32,
    )

    model.config.problem_type = "regression"
    model = model.float()

    first_param = next(model.parameters())
    print(f"Model dtype before training: {first_param.dtype}")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"Model dtype before training: {first_param.dtype}\n")

    if first_param.dtype != torch.float32:
        raise RuntimeError(
            f"Expected model dtype torch.float32, but got {first_param.dtype}. "
            "Do not continue because this can cause Float/Half loss errors."
        )

    raw_train_dataset = Dataset.from_pandas(
        train_df[["text", "labels"]],
        preserve_index=False,
    )

    train_dataset = tokenize_dataset(raw_train_dataset, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=MODEL_CONFIGS[MODEL_KEY]["learning_rate"],
        per_device_train_batch_size=MODEL_CONFIGS[MODEL_KEY]["per_device_train_batch_size"],
        gradient_accumulation_steps=MODEL_CONFIGS[MODEL_KEY]["gradient_accumulation_steps"],
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        # Important for mDeBERTa stability and avoiding Float/Half mismatch.
        fp16=False,
        bf16=False,

        report_to="none",
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        max_grad_norm=1.0,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[FileLoggerCallback(LOG_FILE)],
    )

    resume_checkpoint = get_latest_checkpoint(OUTPUT_DIR)

    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"Resuming from checkpoint: {resume_checkpoint}\n")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        print("No checkpoint found. Starting training from scratch.")
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write("No checkpoint found. Starting training from scratch.\n")
        trainer.train()

    trainer.save_model(str(OUTPUT_DIR / "final_model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final_model"))

    print("\nTraining finished.")
    print(f"Model/checkpoints saved in: {OUTPUT_DIR}")
    print(f"Final model saved in:       {OUTPUT_DIR / 'final_model'}")
    print(f"Training log saved in:      {LOG_FILE}")
    print(f"Gold test rows kept untouched for final evaluation: {len(test_df)}")

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write("Training finished.\n")
        f.write(f"Model/checkpoints saved in: {OUTPUT_DIR}\n")
        f.write(f"Final model saved in: {OUTPUT_DIR / 'final_model'}\n")
        f.write(f"Training log saved in: {LOG_FILE}\n")
        f.write(f"Gold test rows kept untouched for final evaluation: {len(test_df)}\n")


if __name__ == "__main__":
    main()