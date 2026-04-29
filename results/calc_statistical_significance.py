#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
VEX_METRIC_DIR = PROJECT_ROOT / "vex_metric"

sys.path.insert(0, str(VEX_METRIC_DIR))

import vex_config as cfg


# =========================================================
# CONFIG
# =========================================================

TEST_SIZE = 10

N_PERMUTATIONS = 10_000
RANDOM_SEED = 4242

OUTPUT_DIR = SCRIPT_PATH.parent / "statistical_significance_q10"

# Compare best model against all others.
# Ranking metric for choosing the best model.
BEST_MODEL_METRIC = "el_acc"

# Columns in dataframe_env.parquet
TEST_ID_COL = "test_id"
TEST_SIZE_COL = "test_size"
STUDENT_ID_COL = "member_id"
QUESTION_ID_COL = "question_id"
ANSWER_ID_COL = "answer_id"
HUMAN_GRADE_COL = "human_grade"

# Exclude prior/template and transformer baselines.
EXCLUDED_MODEL_PREFIXES = (
    "grade_prior_",
    "grade_bert_",
    "grade_mdeberta_",
)

# If True, also exclude TF-IDF baselines.
EXCLUDE_TFIDF = False


# =========================================================
# MODEL SELECTION
# =========================================================

def get_eval_model_columns() -> list[str]:
    models: list[str] = []

    for col in cfg.MODEL_COLUMNS:
        if col.startswith(EXCLUDED_MODEL_PREFIXES):
            continue

        if EXCLUDE_TFIDF and col.startswith("pred_tfidf_"):
            continue

        models.append(col)

    if not models:
        raise ValueError("No model columns left after filtering.")

    return models


def short_model_name(model_col: str) -> str:
    name = model_col

    prefixes = [
        "new_grade_",
        "grade_",
        "pred_",
    ]

    for prefix in prefixes:
        if name.startswith(prefix):
            name = name.removeprefix(prefix)

    return name


# =========================================================
# BASIC METRIC HELPERS
# =========================================================

def round_and_clip_linear_grades(grades: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(grades, dtype=float)

    if cfg.LINEAR_ROUNDING_STEP and cfg.LINEAR_ROUNDING_STEP > 0:
        arr = np.round(arr / cfg.LINEAR_ROUNDING_STEP) * cfg.LINEAR_ROUNDING_STEP

    arr = np.clip(arr, cfg.LINEAR_MIN_GRADE, cfg.LINEAR_MAX_GRADE)
    return arr


def normalized_to_linear_grade_absolute(scores: pd.Series | np.ndarray) -> np.ndarray:
    scores_arr = np.asarray(scores, dtype=float)

    grades = cfg.LINEAR_MIN_GRADE + (
        (cfg.LINEAR_MAX_GRADE - cfg.LINEAR_MIN_GRADE) * scores_arr
    )

    return round_and_clip_linear_grades(grades)


def normalized_to_pass_fail(scores: pd.Series | np.ndarray) -> np.ndarray:
    scores_arr = np.asarray(scores, dtype=float)
    return (scores_arr >= cfg.LINEAR_PASS_THRESHOLD_NORM).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")

    return float(np.mean(y_true == y_pred))


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = sorted(pd.unique(np.concatenate([y_true, y_pred])))

    if len(labels) <= 1:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0

    mapping = {label: idx for idx, label in enumerate(labels)}

    y_true_enc = np.array([mapping[x] for x in y_true], dtype=int)
    y_pred_enc = np.array([mapping[x] for x in y_pred], dtype=int)

    return float(cohen_kappa_score(y_true_enc, y_pred_enc, weights="quadratic"))


# =========================================================
# DATA LOADING / VALIDATION
# =========================================================

def input_env_path() -> Path:
    """
    Uses the dataframe environment path exactly as defined in vex_config.py.

    In your pipeline, cfg.OUTPUT_PARQUET should point to:
        vex_test_env/4_dataframe/dataframe_env.parquet

    This script does not construct its own input path.
    """
    return Path(cfg.OUTPUT_PARQUET)


def validate_env_df(df: pd.DataFrame, model_cols: list[str]) -> None:
    required = [
        TEST_ID_COL,
        TEST_SIZE_COL,
        STUDENT_ID_COL,
        QUESTION_ID_COL,
        ANSWER_ID_COL,
        HUMAN_GRADE_COL,
    ]

    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns in dataframe_env.parquet: {missing_required}"
        )

    missing_models = [col for col in model_cols if col not in df.columns]
    if missing_models:
        raise ValueError(
            f"Missing model columns in dataframe_env.parquet: {missing_models}"
        )


def assert_no_duplicate_exam_student_question_pairs(df: pd.DataFrame) -> None:
    duplicate_mask = df.duplicated(
        subset=[TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL, QUESTION_ID_COL],
        keep=False,
    )

    if duplicate_mask.any():
        preview = (
            df.loc[
                duplicate_mask,
                [
                    TEST_ID_COL,
                    TEST_SIZE_COL,
                    STUDENT_ID_COL,
                    QUESTION_ID_COL,
                    ANSWER_ID_COL,
                ],
            ]
            .sort_values([TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL, QUESTION_ID_COL])
            .head(100)
        )

        raise ValueError(
            "Duplicate (test_id, test_size, member_id, question_id) pairs found. "
            "This would make VEX exam-level totals invalid.\n\n"
            f"{preview.to_string(index=False)}"
        )


# =========================================================
# BUILD Q10 EXAM-LEVEL LABELS
# =========================================================

def build_q10_exam_level_labels(
    df_env: pd.DataFrame,
    model_cols: list[str],
) -> pd.DataFrame:
    df = df_env[df_env[TEST_SIZE_COL].astype(int) == TEST_SIZE].copy()

    if df.empty:
        raise ValueError(f"No rows found for test_size={TEST_SIZE}.")

    keep_cols = [
        TEST_ID_COL,
        TEST_SIZE_COL,
        STUDENT_ID_COL,
        QUESTION_ID_COL,
        ANSWER_ID_COL,
        HUMAN_GRADE_COL,
    ] + model_cols

    df = df[keep_cols].copy()

    df[HUMAN_GRADE_COL] = pd.to_numeric(df[HUMAN_GRADE_COL], errors="coerce")

    for model_col in model_cols:
        df[model_col] = pd.to_numeric(df[model_col], errors="coerce")

    rows: list[pd.DataFrame] = []

    group_cols = [TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL]

    for model_col in model_cols:
        subset = df[
            [
                TEST_ID_COL,
                TEST_SIZE_COL,
                STUDENT_ID_COL,
                QUESTION_ID_COL,
                HUMAN_GRADE_COL,
                model_col,
            ]
        ].copy()

        grouped = (
            subset.groupby(group_cols)
            .agg(
                n_rows=(QUESTION_ID_COL, "size"),
                n_questions=(QUESTION_ID_COL, "nunique"),
                human_valid=(HUMAN_GRADE_COL, lambda s: int(s.notna().sum())),
                pred_valid=(model_col, lambda s: int(s.notna().sum())),
                gold_total=(HUMAN_GRADE_COL, "sum"),
                pred_total=(model_col, "sum"),
            )
            .reset_index()
        )

        complete_mask = (
            (grouped["n_rows"] == TEST_SIZE)
            & (grouped["n_questions"] == TEST_SIZE)
            & (grouped["human_valid"] == TEST_SIZE)
            & (grouped["pred_valid"] == TEST_SIZE)
        )

        grouped = grouped[complete_mask].copy()

        if grouped.empty:
            print(f"WARNING: No complete q10 student-exam rows for {model_col}")
            continue

        grouped["gold_norm"] = grouped["gold_total"] / float(TEST_SIZE)
        grouped["pred_norm"] = grouped["pred_total"] / float(TEST_SIZE)

        grouped["gold_linear_grade"] = normalized_to_linear_grade_absolute(
            grouped["gold_norm"].to_numpy()
        )
        grouped["pred_linear_grade"] = normalized_to_linear_grade_absolute(
            grouped["pred_norm"].to_numpy()
        )

        grouped["gold_pass"] = normalized_to_pass_fail(
            grouped["gold_norm"].to_numpy()
        )
        grouped["pred_pass"] = normalized_to_pass_fail(
            grouped["pred_norm"].to_numpy()
        )

        grouped["model_col"] = model_col
        grouped["model"] = short_model_name(model_col)

        grouped["correct_linear_grade"] = (
            grouped["gold_linear_grade"].to_numpy()
            == grouped["pred_linear_grade"].to_numpy()
        ).astype(int)

        grouped["correct_pass_fail"] = (
            grouped["gold_pass"].to_numpy()
            == grouped["pred_pass"].to_numpy()
        ).astype(int)

        rows.append(grouped)

    if not rows:
        raise ValueError("No exam-level labels could be built.")

    labels = pd.concat(rows, ignore_index=True)

    return labels


# =========================================================
# RANKING
# =========================================================

def compute_model_ranking(labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_col, df_model in labels.groupby("model_col", sort=False):
        y_true_grade = df_model["gold_linear_grade"].to_numpy()
        y_pred_grade = df_model["pred_linear_grade"].to_numpy()

        y_true_pass = df_model["gold_pass"].to_numpy()
        y_pred_pass = df_model["pred_pass"].to_numpy()

        rows.append(
            {
                "model_col": model_col,
                "model": short_model_name(model_col),
                "n_exam_student_rows": int(len(df_model)),
                "el_acc": accuracy(y_true_grade, y_pred_grade),
                "el_qwk": qwk(y_true_grade, y_pred_grade),
                "el_pass_acc": accuracy(y_true_pass, y_pred_pass),
                "model_pass_rate": float(np.mean(y_pred_pass)),
                "human_pass_rate": float(np.mean(y_true_pass)),
            }
        )

    ranking = pd.DataFrame(rows)

    ranking = ranking.sort_values(
        by=[BEST_MODEL_METRIC, "el_qwk", "el_pass_acc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return ranking


# =========================================================
# MCNEMAR FOR EL-ACC
# =========================================================

def exact_mcnemar_p_value(b: int, c: int) -> float:
    n = int(b + c)

    if n == 0:
        return 1.0

    result = binomtest(
        k=min(b, c),
        n=n,
        p=0.5,
        alternative="two-sided",
    )

    return float(result.pvalue)


def mcnemar_compare_el_acc(
    merged: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> dict[str, Any]:
    a_correct = merged["correct_a"].astype(bool).to_numpy()
    b_correct = merged["correct_b"].astype(bool).to_numpy()

    both_correct = int(np.sum(a_correct & b_correct))
    a_correct_b_wrong = int(np.sum(a_correct & ~b_correct))
    a_wrong_b_correct = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))

    p_value = exact_mcnemar_p_value(
        b=a_correct_b_wrong,
        c=a_wrong_b_correct,
    )

    return {
        "test": "mcnemar_exact_el_acc",
        "model_a": short_model_name(model_a),
        "model_b": short_model_name(model_b),
        "model_a_col": model_a,
        "model_b_col": model_b,
        "n_paired": int(len(merged)),
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "p_value": p_value,
        "significant_0_05": bool(p_value < 0.05),
    }


# =========================================================
# PERMUTATION TEST FOR EL-QWK
# =========================================================

def permutation_test_qwk(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    qwk_a = qwk(y_true, pred_a)
    qwk_b = qwk(y_true, pred_b)

    observed_delta = qwk_a - qwk_b
    observed_abs_delta = abs(observed_delta)

    perm_deltas = np.empty(n_permutations, dtype=float)

    for i in range(n_permutations):
        swap_mask = rng.random(len(y_true)) < 0.5

        perm_a = pred_a.copy()
        perm_b = pred_b.copy()

        perm_a[swap_mask] = pred_b[swap_mask]
        perm_b[swap_mask] = pred_a[swap_mask]

        perm_qwk_a = qwk(y_true, perm_a)
        perm_qwk_b = qwk(y_true, perm_b)

        perm_deltas[i] = perm_qwk_a - perm_qwk_b

    p_value = float(np.mean(np.abs(perm_deltas) >= observed_abs_delta))

    return {
        "qwk_a": float(qwk_a),
        "qwk_b": float(qwk_b),
        "delta_qwk_a_minus_b": float(observed_delta),
        "abs_delta_qwk": float(observed_abs_delta),
        "n_permutations": int(n_permutations),
        "p_value": p_value,
        "significant_0_05": bool(p_value < 0.05),
    }


def permutation_compare_el_qwk(
    merged: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> dict[str, Any]:
    result = permutation_test_qwk(
        y_true=merged["gold_linear_grade"].to_numpy(),
        pred_a=merged["pred_linear_grade_a"].to_numpy(),
        pred_b=merged["pred_linear_grade_b"].to_numpy(),
        n_permutations=N_PERMUTATIONS,
        seed=RANDOM_SEED,
    )

    return {
        "test": "paired_permutation_el_qwk",
        "model_a": short_model_name(model_a),
        "model_b": short_model_name(model_b),
        "model_a_col": model_a,
        "model_b_col": model_b,
        "n_paired": int(len(merged)),
        **result,
    }


# =========================================================
# PAIRING
# =========================================================

def paired_model_frame(
    labels: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    key_cols = [
        TEST_ID_COL,
        TEST_SIZE_COL,
        STUDENT_ID_COL,
    ]

    df_a = labels[labels["model_col"] == model_a].copy()
    df_b = labels[labels["model_col"] == model_b].copy()

    a_keep = key_cols + [
        "gold_linear_grade",
        "pred_linear_grade",
        "correct_linear_grade",
    ]

    b_keep = key_cols + [
        "gold_linear_grade",
        "pred_linear_grade",
        "correct_linear_grade",
    ]

    df_a = df_a[a_keep].rename(
        columns={
            "pred_linear_grade": "pred_linear_grade_a",
            "correct_linear_grade": "correct_a",
        }
    )

    df_b = df_b[b_keep].rename(
        columns={
            "gold_linear_grade": "gold_linear_grade_b",
            "pred_linear_grade": "pred_linear_grade_b",
            "correct_linear_grade": "correct_b",
        }
    )

    merged = df_a.merge(
        df_b,
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            f"No paired rows for {model_a} vs {model_b}."
        )

    mismatch = (
        merged["gold_linear_grade"].to_numpy()
        != merged["gold_linear_grade_b"].to_numpy()
    )

    if np.any(mismatch):
        raise ValueError(
            f"Gold labels mismatch after pairing {model_a} vs {model_b}. "
            "This should never happen."
        )

    merged = merged.drop(columns=["gold_linear_grade_b"])

    return merged


# =========================================================
# REPORTING
# =========================================================

def format_p_value(p: float) -> str:
    if pd.isna(p):
        return "nan"

    if p < 0.0001:
        return f"{p:.2e}"

    return f"{p:.6f}"


def write_text_report(
    ranking: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines: list[str] = []

    lines.append("=" * 100)
    lines.append("VEX STATISTICAL SIGNIFICANCE - Q10 ONLY")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"input_path={input_env_path().resolve()}")
    lines.append(f"test_size={TEST_SIZE}")
    lines.append(f"best_model_metric={BEST_MODEL_METRIC}")
    lines.append(f"n_permutations={N_PERMUTATIONS}")
    lines.append(f"random_seed={RANDOM_SEED}")
    lines.append("")
    lines.append("Excluded model prefixes:")
    for prefix in EXCLUDED_MODEL_PREFIXES:
        lines.append(f"- {prefix}")
    lines.append(f"exclude_tfidf={EXCLUDE_TFIDF}")
    lines.append("")

    lines.append("=" * 100)
    lines.append("MODEL RANKING")
    lines.append("=" * 100)
    lines.append(
        ranking[
            [
                "model",
                "n_exam_student_rows",
                "el_acc",
                "el_qwk",
                "el_pass_acc",
                "model_pass_rate",
                "human_pass_rate",
            ]
        ].to_string(index=False)
    )
    lines.append("")

    best_model = ranking.iloc[0]["model"]
    lines.append(f"Best model by {BEST_MODEL_METRIC}: {best_model}")
    lines.append("")

    lines.append("=" * 100)
    lines.append("MCNEMAR TEST FOR EL-ACC")
    lines.append("=" * 100)
    lines.append(
        "Interpretation: tests whether model A and model B differ in exact "
        "exam-level grade correctness on the same q10 student-exam rows."
    )
    lines.append("")
    lines.append(
        mcnemar_df[
            [
                "model_a",
                "model_b",
                "n_paired",
                "both_correct",
                "a_correct_b_wrong",
                "a_wrong_b_correct",
                "both_wrong",
                "p_value",
                "significant_0_05",
            ]
        ].to_string(
            index=False,
            formatters={"p_value": format_p_value},
        )
    )
    lines.append("")

    lines.append("=" * 100)
    lines.append("PAIRED PERMUTATION TEST FOR EL-QWK")
    lines.append("=" * 100)
    lines.append(
        "Interpretation: randomly swaps paired model predictions and tests "
        "whether the observed EL-QWK difference is larger than expected under "
        "exchangeability."
    )
    lines.append("")
    lines.append(
        permutation_df[
            [
                "model_a",
                "model_b",
                "n_paired",
                "qwk_a",
                "qwk_b",
                "delta_qwk_a_minus_b",
                "p_value",
                "significant_0_05",
            ]
        ].to_string(
            index=False,
            formatters={"p_value": format_p_value},
        )
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def main() -> int:
    env_path = input_env_path()

    if not env_path.exists():
        raise FileNotFoundError(
            f"dataframe_env.parquet not found: {env_path.resolve()}"
        )

    model_cols = get_eval_model_columns()

    print("=" * 100)
    print("VEX STATISTICAL SIGNIFICANCE - Q10 ONLY")
    print("=" * 100)
    print(f"Input from vex_config.OUTPUT_PARQUET: {env_path.resolve()}")
    print(f"Output dir:                       {OUTPUT_DIR.resolve()}")
    print(f"Test size:                        {TEST_SIZE}")
    print("")
    print("Models used:")
    for col in model_cols:
        print(f"  - {col}")
    print("")

    df_env = pd.read_parquet(env_path)
    validate_env_df(df_env, model_cols)
    assert_no_duplicate_exam_student_question_pairs(df_env)

    labels = build_q10_exam_level_labels(
        df_env=df_env,
        model_cols=model_cols,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = OUTPUT_DIR / "q10_exam_level_labels.parquet"
    labels.to_parquet(labels_path, index=False)

    ranking = compute_model_ranking(labels)

    ranking_path = OUTPUT_DIR / "q10_model_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    print("Model ranking:")
    print(
        ranking[
            [
                "model",
                "el_acc",
                "el_qwk",
                "el_pass_acc",
                "model_pass_rate",
                "human_pass_rate",
            ]
        ].to_string(index=False)
    )
    print("")

    best_model_col = str(ranking.iloc[0]["model_col"])

    mcnemar_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []

    for other_model_col in ranking["model_col"].tolist()[1:]:
        other_model_col = str(other_model_col)

        print(
            f"Comparing best model {short_model_name(best_model_col)} "
            f"vs {short_model_name(other_model_col)}"
        )

        merged = paired_model_frame(
            labels=labels,
            model_a=best_model_col,
            model_b=other_model_col,
        )

        mcnemar_rows.append(
            mcnemar_compare_el_acc(
                merged=merged,
                model_a=best_model_col,
                model_b=other_model_col,
            )
        )

        permutation_rows.append(
            permutation_compare_el_qwk(
                merged=merged,
                model_a=best_model_col,
                model_b=other_model_col,
            )
        )

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    permutation_df = pd.DataFrame(permutation_rows)

    mcnemar_path = OUTPUT_DIR / "q10_mcnemar_el_acc.csv"
    permutation_path = OUTPUT_DIR / "q10_permutation_el_qwk.csv"
    report_path = OUTPUT_DIR / "q10_statistical_significance_report.txt"

    mcnemar_df.to_csv(mcnemar_path, index=False)
    permutation_df.to_csv(permutation_path, index=False)

    write_text_report(
        ranking=ranking,
        mcnemar_df=mcnemar_df,
        permutation_df=permutation_df,
        output_path=report_path,
    )

    print("")
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Exam-level labels:     {labels_path.resolve()}")
    print(f"Model ranking:         {ranking_path.resolve()}")
    print(f"McNemar results:       {mcnemar_path.resolve()}")
    print(f"Permutation results:   {permutation_path.resolve()}")
    print(f"Text report:           {report_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
