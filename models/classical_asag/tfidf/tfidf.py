#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


# =========================================================
# CONFIG
# =========================================================

INPUT_PARQUET = Path("../dataset/v1_0_release/v1_0_stable.parquet")
OUTPUT_PARQUET = Path("tfidf_results.parquet")

ID_COL = "grading_id"
QUESTION_COL = "question"
ANSWER_COL = "answer"
SPLIT_COL = "split"

# Silver labels for training
TRAIN_TARGET_COL = "grade"

ALLOWED_GRADES = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)

PRED_COLS = {
    "v1_answer_word_unigram": "pred_tfidf_v1_answer_word_unigram",
    "v2_answer_word_uni_bigram": "pred_tfidf_v2_answer_word_uni_bigram",
    "v3_qa_concat_word_uni_bigram": "pred_tfidf_v3_qa_concat_word_uni_bigram",
    "v4_question_and_answer_separate": "pred_tfidf_v4_question_and_answer_separate",
    "v5_answer_char_3_5": "pred_tfidf_v5_answer_char_3_5",
    "v6_mixed_word_char_qa": "pred_tfidf_v6_mixed_word_char_qa",
}


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


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def concat_question_answer(question: object, answer: object) -> str:
    q = normalize_text(question)
    a = normalize_text(answer)
    return f"question: {q} answer: {a}"


def round_to_allowed_grade(value: float) -> float:
    return float(ALLOWED_GRADES[np.argmin(np.abs(ALLOWED_GRADES - value))])


def round_array_to_allowed_grades(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.array([round_to_allowed_grade(v) for v in values], dtype=float)


# =========================================================
# FEATURE BUILDERS
# =========================================================

def build_features_v1(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[csr_matrix, csr_matrix]:
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        lowercase=True,
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )
    X_train = vec.fit_transform(train_df[ANSWER_COL].map(normalize_text))
    X_test = vec.transform(test_df[ANSWER_COL].map(normalize_text))
    return X_train, X_test


def build_features_v2(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[csr_matrix, csr_matrix]:
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )
    X_train = vec.fit_transform(train_df[ANSWER_COL].map(normalize_text))
    X_test = vec.transform(test_df[ANSWER_COL].map(normalize_text))
    return X_train, X_test


def build_features_v3(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[csr_matrix, csr_matrix]:
    train_text = [
        concat_question_answer(q, a)
        for q, a in zip(train_df[QUESTION_COL], train_df[ANSWER_COL], strict=False)
    ]
    test_text = [
        concat_question_answer(q, a)
        for q, a in zip(test_df[QUESTION_COL], test_df[ANSWER_COL], strict=False)
    ]

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
    )
    X_train = vec.fit_transform(train_text)
    X_test = vec.transform(test_text)
    return X_train, X_test


def build_features_v4(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[csr_matrix, csr_matrix]:
    q_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=1,
        max_features=20000,
        sublinear_tf=True,
    )
    a_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        max_features=40000,
        sublinear_tf=True,
    )

    Xq_train = q_vec.fit_transform(train_df[QUESTION_COL].map(normalize_text))
    Xq_test = q_vec.transform(test_df[QUESTION_COL].map(normalize_text))

    Xa_train = a_vec.fit_transform(train_df[ANSWER_COL].map(normalize_text))
    Xa_test = a_vec.transform(test_df[ANSWER_COL].map(normalize_text))

    X_train = hstack([Xq_train, Xa_train]).tocsr()
    X_test = hstack([Xq_test, Xa_test]).tocsr()
    return X_train, X_test


def build_features_v5(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[csr_matrix, csr_matrix]:
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
        max_features=80000,
        sublinear_tf=True,
    )
    X_train = vec.fit_transform(train_df[ANSWER_COL].map(normalize_text))
    X_test = vec.transform(test_df[ANSWER_COL].map(normalize_text))
    return X_train, X_test


def build_features_v6(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[csr_matrix, csr_matrix]:
    q_word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=1,
        max_features=15000,
        sublinear_tf=True,
    )
    a_word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )
    a_char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
        max_features=40000,
        sublinear_tf=True,
    )

    q_train = q_word_vec.fit_transform(train_df[QUESTION_COL].map(normalize_text))
    q_test = q_word_vec.transform(test_df[QUESTION_COL].map(normalize_text))

    a_word_train = a_word_vec.fit_transform(train_df[ANSWER_COL].map(normalize_text))
    a_word_test = a_word_vec.transform(test_df[ANSWER_COL].map(normalize_text))

    a_char_train = a_char_vec.fit_transform(train_df[ANSWER_COL].map(normalize_text))
    a_char_test = a_char_vec.transform(test_df[ANSWER_COL].map(normalize_text))

    X_train = hstack([q_train, a_word_train, a_char_train]).tocsr()
    X_test = hstack([q_test, a_word_test, a_char_test]).tocsr()
    return X_train, X_test


# =========================================================
# MODEL
# =========================================================

def fit_predict_regression(X_train: csr_matrix, y_train: np.ndarray, X_test: csr_matrix) -> np.ndarray:
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = np.clip(pred, ALLOWED_GRADES.min(), ALLOWED_GRADES.max())
    pred = round_array_to_allowed_grades(pred)
    return pred


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    df = open_parquet_file(INPUT_PARQUET)

    required_cols = [ID_COL, QUESTION_COL, ANSWER_COL, SPLIT_COL, TRAIN_TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    train_mask = df[SPLIT_COL].astype(str).str.lower() == "train"
    test_mask = df[SPLIT_COL].astype(str).str.lower() == "test"

    if not train_mask.any():
        raise ValueError("No train rows found (split == 'train').")
    if not test_mask.any():
        raise ValueError("No test rows found (split == 'test').")

    df["_train_target_num"] = pd.to_numeric(df[TRAIN_TARGET_COL], errors="coerce")
    usable_train_mask = train_mask & df["_train_target_num"].notna()

    if not usable_train_mask.any():
        raise ValueError("No usable train rows with silver labels found.")

    train_df = df.loc[usable_train_mask].copy()
    test_df = df.loc[test_mask].copy()
    y_train = df.loc[usable_train_mask, "_train_target_num"].astype(float).to_numpy()

    print("=========================================================")
    print("TF-IDF PREDICTION GENERATION")
    print("=========================================================")
    print(f"Input file: {INPUT_PARQUET.resolve()}")
    print(f"Output file: {OUTPUT_PARQUET.resolve()}")
    print(f"Rows total: {len(df)}")
    print(f"Train rows total: {int(train_mask.sum())}")
    print(f"Test rows total:  {int(test_mask.sum())}")
    print(f"Usable train rows: {int(usable_train_mask.sum())}")
    print(f"Prediction target rows: {len(test_df)}")
    print("=========================================================\n")

    for pred_col in PRED_COLS.values():
        df[pred_col] = np.nan

    variants = [
        ("v1_answer_word_unigram", build_features_v1),
        ("v2_answer_word_uni_bigram", build_features_v2),
        ("v3_qa_concat_word_uni_bigram", build_features_v3),
        ("v4_question_and_answer_separate", build_features_v4),
        ("v5_answer_char_3_5", build_features_v5),
        ("v6_mixed_word_char_qa", build_features_v6),
    ]

    for variant_name, feature_builder in variants:
        pred_col = PRED_COLS[variant_name]

        print(f"Running {variant_name} ...")

        X_train, X_test = feature_builder(train_df, test_df)
        preds_test = fit_predict_regression(X_train, y_train, X_test)

        df.loc[test_df.index, pred_col] = preds_test

        print(f"  wrote predictions to column: {pred_col}")
        print(f"  predicted rows: {len(preds_test)}\n")

    df = df.drop(columns=["_train_target_num"])

    save_parquet(OUTPUT_PARQUET, df)

    print("\nPrediction distributions on test:")
    for variant_name, pred_col in PRED_COLS.items():
        print(f"\n{variant_name} ({pred_col})")
        print(df.loc[test_mask, pred_col].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()