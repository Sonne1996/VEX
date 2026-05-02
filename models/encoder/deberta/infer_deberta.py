#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

INPUT_PARQUET = PROJECT_ROOT / "dataset" / "vex" / "v1_0_release" / "v1_0_stable.parquet"

MODEL_KEY = "mdeberta"

# Choose one:
# "base" = pretrained encoder with randomly initialized regression head.
# "ft"   = fine-tuned model from your training script.
INFER_MODE = "base"

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "bert": {
        "base_model_name": "bert-base-german-cased",
        "ft_model_dir": BASE_DIR / "outputs_silver_grade_only_bert_base_german_cased" / "final_model",
        "output_base": "bert_base_gold_predictions.parquet",
        "output_ft": "bert_ft_gold_predictions.parquet",
        "batch_size": 32,
    },
    "mdeberta": {
        "base_model_name": "microsoft/mdeberta-v3-base",
        "ft_model_dir": BASE_DIR / "outputs_silver_grade_only_mdeberta_v3_base" / "final_model",
        "output_base": "mdeberta_base_gold_predictions.parquet",
        "output_ft": "mdeberta_ft_gold_predictions.parquet",
        "batch_size": 16,
    },
}

QUESTION_COL = "question"
ANSWER_COL = "answer"
GRADE_COL = "grade"
SPLIT_COL = "split"
LABEL_TYPE_COL = "label_type"

TEST_SPLIT_VALUE = "test"
TEST_LABEL_TYPE = "gold"

MAX_SEQ_LENGTH = 512
RANDOM_SEED = 42

ALLOWED_GRADES_ARRAY = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)


# =========================================================
# PATHS
# =========================================================

if MODEL_KEY not in MODEL_CONFIGS:
    raise ValueError(f"Unknown MODEL_KEY: {MODEL_KEY}. Available: {list(MODEL_CONFIGS)}")

if INFER_MODE not in {"base", "ft"}:
    raise ValueError("INFER_MODE must be either 'base' or 'ft'.")

MODEL_INFO = MODEL_CONFIGS[MODEL_KEY]

if INFER_MODE == "base":
    MODEL_PATH = MODEL_INFO["base_model_name"]
    OUTPUT_PARQUET = BASE_DIR / MODEL_INFO["output_base"]
else:
    MODEL_PATH = MODEL_INFO["ft_model_dir"]
    OUTPUT_PARQUET = BASE_DIR / MODEL_INFO["output_ft"]

LOG_FILE = OUTPUT_PARQUET.with_suffix(".log")


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

    if value not in set(ALLOWED_GRADES_ARRAY.tolist()):
        return None

    return value


def build_encoder_input(question: str, answer: str) -> str:
    question = "" if pd.isna(question) else str(question).strip()
    answer = "" if pd.isna(answer) else str(answer).strip()

    return (
        f"QUESTION: {question} "
        f"[SEP] STUDENT_ANSWER: {answer}"
    )


def round_to_allowed_grade(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    distances = np.abs(values[:, None] - ALLOWED_GRADES_ARRAY[None, :])
    nearest_indices = np.argmin(distances, axis=1)
    return ALLOWED_GRADES_ARRAY[nearest_indices]


def prepare_gold_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [QUESTION_COL, ANSWER_COL, GRADE_COL, SPLIT_COL, LABEL_TYPE_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out[GRADE_COL] = out[GRADE_COL].apply(normalize_grade)

    gold_df = out[
        (out[SPLIT_COL] == TEST_SPLIT_VALUE) &
        (out[LABEL_TYPE_COL] == TEST_LABEL_TYPE) &
        (out[GRADE_COL].notna())
    ].copy()

    if gold_df.empty:
        raise ValueError("No usable gold test rows found.")

    gold_df["text"] = gold_df.apply(
        lambda row: build_encoder_input(
            question=row[QUESTION_COL],
            answer=row[ANSWER_COL],
        ),
        axis=1,
    )

    return gold_df


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    keep_cols = dataset.column_names

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
    )

    return tokenized


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(mse))

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }


def write_log(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    write_log("=" * 100)
    write_log(f"NEW INFERENCE RUN STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(f"MODEL_KEY: {MODEL_KEY}")
    write_log(f"INFER_MODE: {INFER_MODE}")
    write_log(f"MODEL_PATH: {MODEL_PATH}")
    write_log(f"OUTPUT_PARQUET: {OUTPUT_PARQUET}")

    df = open_parquet_file(INPUT_PARQUET)
    gold_df = prepare_gold_dataframe(df)

    write_log(f"Gold test rows: {len(gold_df)}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    if INFER_MODE == "base":
        # Important:
        # This adds a randomly initialized regression head.
        # It is useful only as a sanity/random-head baseline.
        model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_PATH),
            num_labels=1,
            problem_type="regression",
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_PATH),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    raw_dataset = Dataset.from_pandas(
        gold_df[["text"]],
        preserve_index=False,
    )

    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=MODEL_INFO["batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )

    predictions_raw: list[float] = []

    start_time = time.time()

    with torch.no_grad():
        for step, batch in enumerate(dataloader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits.detach().float().cpu().numpy().reshape(-1)

            predictions_raw.extend(logits.tolist())

            if step % 25 == 0:
                write_log(f"Inference batch {step}/{len(dataloader)}")

    elapsed = time.time() - start_time

    pred_raw = np.asarray(predictions_raw, dtype=np.float32)
    pred_clipped = np.clip(pred_raw, 0.0, 1.0)
    pred_rounded = round_to_allowed_grade(pred_clipped)

    result_df = gold_df.copy()

    result_df[f"grade_{MODEL_KEY}_{INFER_MODE}_raw"] = pred_raw
    result_df[f"grade_{MODEL_KEY}_{INFER_MODE}_clipped"] = pred_clipped
    result_df[f"grade_{MODEL_KEY}_{INFER_MODE}"] = pred_rounded

    y_true = result_df[GRADE_COL].astype(float).to_numpy(dtype=np.float32)

    metrics_clipped = compute_regression_metrics(y_true, pred_clipped)
    metrics_rounded = compute_regression_metrics(y_true, pred_rounded)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(OUTPUT_PARQUET, index=False)

    write_log(f"Inference finished in {elapsed:.2f}s")
    write_log(f"Rows written: {len(result_df)}")
    write_log(f"Saved predictions to: {OUTPUT_PARQUET}")

    write_log("Metrics using clipped continuous predictions:")
    write_log(json.dumps(metrics_clipped, indent=2, ensure_ascii=False))

    write_log("Metrics using rounded 5-point predictions:")
    write_log(json.dumps(metrics_rounded, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
