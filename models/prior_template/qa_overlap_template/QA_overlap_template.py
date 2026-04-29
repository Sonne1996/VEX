#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

INPUT_PARQUET = PROJECT_ROOT / "dataset" / "vex" / "v1_0_release" / "v1_0_stable.parquet"
OUTPUT_PARQUET = BASE_DIR / "qa_overlap_template.parquet"

ID_COL = "grading_id"
QUESTION_COL = "question"
ANSWER_COL = "answer"
SPLIT_COL = "split"

PRED_COL = "grade_prior_template_overlap"

ALLOWED_GRADES = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)


# =========================================================
# HELPERS
# =========================================================

def open_parquet_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Input file must be a parquet file, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_parquet(path)


def save_parquet(path: Path, df: pd.DataFrame) -> None:
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved parquet to: {path.resolve()}")


def tokenize_text(text: str) -> list[str]:
    """
    Very simple tokenizer:
    - lowercase
    - keeps letters/numbers/umlauts
    - splits on non-word characters
    """
    if pd.isna(text):
        return []

    text = str(text).lower().strip()

    tokens = re.findall(r"[a-zA-Z0-9äöüÄÖÜ]+", text)
    return tokens


def round_to_allowed_grade(value: float) -> float:
    return float(ALLOWED_GRADES[np.argmin(np.abs(ALLOWED_GRADES - value))])


# =========================================================
# CORE LOGIC
# =========================================================

def predict_overlap_template(question: str, answer: str) -> float:
    """
    Simple rule-based template baseline using:
    - empty answer
    - answer length
    - token overlap between question and answer

    The logic is intentionally simple and inspectable.
    """

    answer_str = "" if pd.isna(answer) else str(answer).strip()
    question_tokens = tokenize_text(question)
    answer_tokens = tokenize_text(answer)

    # -----------------------------------------------------
    # Rule 1: Empty answer => 0.0
    # -----------------------------------------------------
    if answer_str == "":
        return 0.0

    # -----------------------------------------------------
    # Basic stats
    # -----------------------------------------------------
    answer_len_chars = len(answer_str)

    question_token_set = set(question_tokens)
    answer_token_set = set(answer_tokens)

    if len(question_token_set) == 0:
        overlap_ratio = 0.0
    else:
        shared_tokens = question_token_set.intersection(answer_token_set)
        overlap_ratio = len(shared_tokens) / len(question_token_set)

    # -----------------------------------------------------
    # Rule 2: Very short answer
    # -----------------------------------------------------
    if answer_len_chars < 8:
        return 0.25

    # -----------------------------------------------------
    # Rule 3: Very low overlap + short answer
    # -----------------------------------------------------
    if overlap_ratio < 0.05 and answer_len_chars < 25:
        return 0.25

    # -----------------------------------------------------
    # Rule 4: Low overlap + medium/short answer
    # -----------------------------------------------------
    if overlap_ratio < 0.10:
        if answer_len_chars < 50:
            return 0.25
        return 0.5

    # -----------------------------------------------------
    # Rule 5: Moderate overlap
    # -----------------------------------------------------
    if overlap_ratio < 0.25:
        if answer_len_chars < 40:
            return 0.5
        return 0.75

    # -----------------------------------------------------
    # Rule 6: Strong overlap
    # -----------------------------------------------------
    if overlap_ratio < 0.45:
        if answer_len_chars < 30:
            return 0.5
        return 0.75

    # -----------------------------------------------------
    # Rule 7: Very strong overlap
    # -----------------------------------------------------
    if answer_len_chars < 20:
        return 0.75

    return 1.0


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    df = open_parquet_file(INPUT_PARQUET)

    required_cols = [ID_COL, QUESTION_COL, ANSWER_COL, SPLIT_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    test_mask = df[SPLIT_COL] == "test"
    if not test_mask.any():
        raise ValueError("No test rows found.")

    df[PRED_COL] = np.nan

    df.loc[test_mask, PRED_COL] = df.loc[test_mask].apply(
        lambda row: predict_overlap_template(
            question=row[QUESTION_COL],
            answer=row[ANSWER_COL]
        ),
        axis=1
    )

    save_parquet(OUTPUT_PARQUET, df)

    print("\nSummary:")
    print(df[[SPLIT_COL, PRED_COL]].groupby(SPLIT_COL).count())
    print("\nPredicted grade distribution on test:")
    print(df.loc[test_mask, PRED_COL].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
