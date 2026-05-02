#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create Figure 4: exam-level metric sensitivity to grade-scale granularity.

Run from:
    VEX/results/

Example:
    python create_plot_4_a_b_qwk_linear_qwk_distro.py

Input:
    ../vex_metric/vex_test_env/4_dataframe/dataframe_env.parquet

Scope:
    One independent plot set per available test_size.

Grade-scale construction:
    n_classes = 2..10

    absolute_threshold:
        class 0:
            fail, normalized score < PASS_THRESHOLD

        classes 1..n_classes-1:
            passing categories, equal-width bins on [PASS_THRESHOLD, 1.0]

        Therefore:
            n_classes = 2 means binary fail/pass.
            n_classes = 10 means fail + 9 absolute passing categories.

    bologna_distribution:
        class 0:
            fail, normalized score < PASS_THRESHOLD

        classes 1..n_classes-1:
            passing categories assigned by rank among passing students.

        The passing-class distribution is interpolated from the default
        Bologna distribution:

            [0.10, 0.25, 0.30, 0.25, 0.10]

        Therefore:
            n_classes = 2 means binary fail/pass.
            n_classes = 6 means F + A/B/C/D/E-style Bologna.
            n_classes = 10 means fail + 9 finer Bologna-shaped passing categories.

Metrics:
    - EL-Acc
    - EL-QWK
    - EL-tau_b

Outputs:
    figures_plot_4/
        figure_4_q5_...png/pdf
        figure_4_q10_absolute_el_acc_granularity.png/pdf
        figure_4_q10_absolute_el_qwk_granularity.png/pdf
        figure_4_q10_absolute_el_tau_granularity.png/pdf
        figure_4_q10_bologna_el_acc_granularity.png/pdf
        figure_4_q10_bologna_el_qwk_granularity.png/pdf
        figure_4_q10_bologna_el_tau_granularity.png/pdf
        figure_4_q10_granularity_data.csv
        figure_4_q10_granularity_per_exam.csv
        figure_4_q10_sanity_check.txt
        figure_4_q15_...png/pdf
        figure_4_q20_...png/pdf
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# RELATIVE PATH CONFIG
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent

VEX_METRIC_DIR = (SCRIPT_DIR / ".." / "vex_metric").resolve()
PROJECT_ROOT = (SCRIPT_DIR / "..").resolve()

sys.path.insert(0, str(VEX_METRIC_DIR))

try:
    import vex_config as cfg
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import vex_config.\n"
        "Expected this script to be located in:\n"
        "  VEX/results/\n"
        "Expected vex_config.py at:\n"
        f"  {VEX_METRIC_DIR / 'vex_config.py'}\n"
        "Run it from VEX/results/ with:\n"
        "  python create_plot_4_q10_granularity.py"
    ) from exc


INPUT_PARQUET = (
    SCRIPT_DIR
    / ".."
    / "vex_metric"
    / "vex_test_env"
    / "4_dataframe"
    / "dataframe_env.parquet"
)
OUTPUT_DIR = SCRIPT_DIR / "figures_plot_4"


# =========================================================
# COLUMN CONFIG
# =========================================================

MODEL_COL = "model_col"

TEST_ID_COL = "test_id"
TEST_SIZE_COL = "test_size"
QUESTION_ID_COL = "question_id"
STUDENT_ID_COL = "member_id"
ANSWER_ID_COL = "answer_id"
HUMAN_GRADE_COL = "human_grade"

MODEL_COLUMNS = list(cfg.MODEL_COLUMNS)


# =========================================================
# GRANULARITY CONFIG
# =========================================================

# None means: use all test sizes available in dataframe_env.parquet.
# Set to e.g. [10] for a q10-only run.
TARGET_TEST_SIZES: list[int] | None = None

MIN_CLASSES = 2
MAX_CLASSES = 10
CLASS_COUNTS = list(range(MIN_CLASSES, MAX_CLASSES + 1))

PASS_THRESHOLD = float(cfg.LINEAR_PASS_THRESHOLD_NORM)

RECOMPUTE_PER_EXAM_METRICS = True

SCALE_ABSOLUTE = "absolute_threshold"
SCALE_BOLOGNA = "bologna_distribution"
SCALE_TYPES = [SCALE_ABSOLUTE, SCALE_BOLOGNA]


COMBINED_SUMMARY_CSV = OUTPUT_DIR / "figure_4_granularity_data.csv"
COMBINED_PER_EXAM_CSV = OUTPUT_DIR / "figure_4_granularity_per_exam.csv"


def figure_4_paths(test_size: int) -> dict[str, Path]:
    prefix = f"figure_4_q{test_size}"

    return {
        "abs_acc_pdf": OUTPUT_DIR / f"{prefix}_absolute_el_acc_granularity.pdf",
        "abs_acc_png": OUTPUT_DIR / f"{prefix}_absolute_el_acc_granularity.png",
        "abs_qwk_pdf": OUTPUT_DIR / f"{prefix}_absolute_el_qwk_granularity.pdf",
        "abs_qwk_png": OUTPUT_DIR / f"{prefix}_absolute_el_qwk_granularity.png",
        "abs_tau_pdf": OUTPUT_DIR / f"{prefix}_absolute_el_tau_granularity.pdf",
        "abs_tau_png": OUTPUT_DIR / f"{prefix}_absolute_el_tau_granularity.png",
        "bol_acc_pdf": OUTPUT_DIR / f"{prefix}_bologna_el_acc_granularity.pdf",
        "bol_acc_png": OUTPUT_DIR / f"{prefix}_bologna_el_acc_granularity.png",
        "bol_qwk_pdf": OUTPUT_DIR / f"{prefix}_bologna_el_qwk_granularity.pdf",
        "bol_qwk_png": OUTPUT_DIR / f"{prefix}_bologna_el_qwk_granularity.png",
        "bol_tau_pdf": OUTPUT_DIR / f"{prefix}_bologna_el_tau_granularity.pdf",
        "bol_tau_png": OUTPUT_DIR / f"{prefix}_bologna_el_tau_granularity.png",
        "summary_csv": OUTPUT_DIR / f"{prefix}_granularity_data.csv",
        "per_exam_csv": OUTPUT_DIR / f"{prefix}_granularity_per_exam.csv",
        "sanity_txt": OUTPUT_DIR / f"{prefix}_sanity_check.txt",
    }


# =========================================================
# PLOT CONFIG
# =========================================================

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 7,
    }
)

FAMILY_COLORS = {
    "LLM": "#1f77b4",
    "Transformer": "#2ca02c",
    "Prior": "#9467bd",
    "TF-IDF": "#ff7f0e",
    "Other": "#666666",
}

FAMILY_LINESTYLES = {
    "LLM": "-",
    "Transformer": "-.",
    "Prior": ":",
    "TF-IDF": "--",
    "Other": "-",
}


# =========================================================
# MODEL LABELS
# =========================================================

DISPLAY_NAMES = {
    "new_grade_deepseek/deepseek-v3.2-thinking": "DeepSeek Thinking",
    "new_grade_deepseek/deepseek-v3.2": "DeepSeek",
    "new_grade_google/gemini-2.5-pro": "Gemini",
    "new_grade_anthropic/claude-sonnet-4.6": "Claude",
    "new_grade_openai/gpt-5.4": "GPT",
    "new_grade_llama32_3b_base": "Llama Base",
    "new_grade_gemma_e4_base": "Gemma Base",
    "new_grade_llama32_3b_ft": "Llama FT",
    "new_grade_gemma_e4_ft": "Gemma FT",
    "grade_bert_base": "BERT Base",
    "grade_bert_ft": "BERT FT",
    "grade_mdeberta_base": "mDeBERTa Base",
    "grade_mdeberta_ft": "mDeBERTa FT",
    "grade_prior_global": "Global Prior",
    "grade_prior_template_overlap": "Template",
    "pred_tfidf_v5_answer_char_3_5": "TF-IDF Char",
    "pred_tfidf_v1_answer_word_unigram": "TF-IDF Word",
    "pred_tfidf_v4_question_and_answer_separate": "TF-IDF QA",
}


def short_model_name(model_col: str) -> str:
    name = str(model_col)

    for prefix in ("new_grade_", "grade_", "pred_"):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)

    return name.replace("/", "_")


def display_name(model_col: str) -> str:
    return DISPLAY_NAMES.get(str(model_col), short_model_name(str(model_col)))


def model_family(model_col: str) -> str:
    model_col = str(model_col)

    if model_col.startswith("new_grade_"):
        return "LLM"

    if model_col.startswith("grade_bert") or model_col.startswith("grade_mdeberta"):
        return "Transformer"

    if model_col.startswith("grade_prior"):
        return "Prior"

    if model_col.startswith("pred_tfidf"):
        return "TF-IDF"

    return "Other"


# =========================================================
# IO AND VALIDATION
# =========================================================

def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input parquet not found: {path}")

    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Pandas cannot read parquet. Install pyarrow, e.g.:\n"
            "  pip install pyarrow"
        ) from exc


def validate_input(df: pd.DataFrame) -> None:
    required = [
        TEST_ID_COL,
        TEST_SIZE_COL,
        QUESTION_ID_COL,
        STUDENT_ID_COL,
        HUMAN_GRADE_COL,
        *MODEL_COLUMNS,
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns in dataframe_env.parquet:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def assert_no_duplicate_exam_student_question_pairs(df_env: pd.DataFrame) -> None:
    key_cols = [
        TEST_ID_COL,
        TEST_SIZE_COL,
        STUDENT_ID_COL,
        QUESTION_ID_COL,
    ]

    duplicate_mask = df_env.duplicated(subset=key_cols, keep=False)

    if duplicate_mask.any():
        preview_cols = key_cols.copy()

        if ANSWER_ID_COL in df_env.columns:
            preview_cols.append(ANSWER_ID_COL)

        duplicates = (
            df_env.loc[duplicate_mask, preview_cols]
            .sort_values(key_cols)
            .head(100)
        )

        raise ValueError(
            "dataframe_env.parquet contains duplicate "
            "(test_id, test_size, member_id, question_id) rows. "
            "Exam-level totals would be invalid.\n"
            f"Examples:\n{duplicates.to_string(index=False)}"
        )


# =========================================================
# METRIC HELPERS
# =========================================================

def accuracy_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not valid_mask.any():
        return np.nan

    return float(np.mean(y_true[valid_mask] == y_pred[valid_mask]))


def qwk_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> float:
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[valid_mask].astype(int)
    y_pred = y_pred[valid_mask].astype(int)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return np.nan

    if len(np.unique(y_true)) == 1 and len(np.unique(y_pred)) == 1:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0

    observed = np.zeros((n_classes, n_classes), dtype=float)

    for true_value, pred_value in zip(y_true, y_pred, strict=False):
        if 0 <= true_value < n_classes and 0 <= pred_value < n_classes:
            observed[true_value, pred_value] += 1.0

    total = observed.sum()

    if total == 0:
        return np.nan

    hist_true = observed.sum(axis=1)
    hist_pred = observed.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / total

    denom = float((n_classes - 1) ** 2)

    if denom == 0:
        return np.nan

    weights = np.fromfunction(
        lambda i, j: ((i - j) ** 2) / denom,
        (n_classes, n_classes),
        dtype=float,
    )

    observed_weighted = float((weights * observed).sum())
    expected_weighted = float((weights * expected).sum())

    if expected_weighted == 0:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0

    return 1.0 - (observed_weighted / expected_weighted)


def tau_b_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[valid_mask].astype(int)
    y_pred = y_pred[valid_mask].astype(int)

    if len(y_true) < 2 or len(y_true) != len(y_pred):
        return np.nan

    if len(np.unique(y_true)) <= 1 or len(np.unique(y_pred)) <= 1:
        return np.nan

    n_classes = int(max(y_true.max(), y_pred.max()) + 1)

    contingency = np.zeros((n_classes, n_classes), dtype=np.int64)

    for true_value, pred_value in zip(y_true, y_pred, strict=False):
        if 0 <= true_value < n_classes and 0 <= pred_value < n_classes:
            contingency[true_value, pred_value] += 1

    concordant = 0
    discordant = 0

    for true_idx in range(n_classes):
        for pred_idx in range(n_classes):
            count = int(contingency[true_idx, pred_idx])

            if count == 0:
                continue

            concordant += count * int(
                contingency[:true_idx, :pred_idx].sum()
                + contingency[true_idx + 1 :, pred_idx + 1 :].sum()
            )
            discordant += count * int(
                contingency[:true_idx, pred_idx + 1 :].sum()
                + contingency[true_idx + 1 :, :pred_idx].sum()
            )

    concordant //= 2
    discordant //= 2

    def pair_count(values: np.ndarray) -> int:
        values = values.astype(np.int64)
        return int(np.sum(values * (values - 1) // 2))

    tied_true = pair_count(contingency.sum(axis=1)) - pair_count(contingency.ravel())
    tied_pred = pair_count(contingency.sum(axis=0)) - pair_count(contingency.ravel())

    denominator = np.sqrt(
        float(concordant + discordant + tied_true)
        * float(concordant + discordant + tied_pred)
    )

    if denominator == 0:
        return np.nan

    return float((concordant - discordant) / denominator)


# =========================================================
# ABSOLUTE THRESHOLD SCALE
# =========================================================

def labels_from_normalized_scores_absolute_threshold(
    scores: pd.Series | np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Converts normalized exam scores to absolute threshold-based final classes.

    class 0:
        fail, score < PASS_THRESHOLD

    classes 1..n_classes-1:
        passing categories, equal-width bins on [PASS_THRESHOLD, 1.0]

    Higher class value means better final grade.
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2.")

    scores_arr = np.asarray(scores, dtype=float)
    labels = np.full(len(scores_arr), np.nan, dtype=float)

    valid_mask = np.isfinite(scores_arr)
    valid_scores = scores_arr[valid_mask]

    valid_labels = np.zeros(len(valid_scores), dtype=float)

    pass_mask = valid_scores >= PASS_THRESHOLD

    if pass_mask.any():
        passing_scores = valid_scores[pass_mask]

        denominator = 1.0 - PASS_THRESHOLD

        if denominator <= 0:
            raise ValueError(
                f"Invalid PASS_THRESHOLD={PASS_THRESHOLD}. Must be < 1.0."
            )

        passing_relative = (passing_scores - PASS_THRESHOLD) / denominator
        passing_bins = np.floor(passing_relative * float(n_classes - 1)) + 1.0
        passing_bins = np.clip(passing_bins, 1.0, float(n_classes - 1))

        valid_labels[pass_mask] = passing_bins

    labels[valid_mask] = valid_labels

    return labels


# =========================================================
# BOLOGNA DISTRIBUTION SCALE
# =========================================================

def interpolated_bologna_passing_distribution(
    n_passing_classes: int,
) -> list[float]:
    """
    Builds a Bologna-shaped passing distribution for an arbitrary number
    of passing classes.

    Anchor:
        5 passing classes -> cfg.BOLOGNA_PASSING_DISTRIBUTION
        normally [0.10, 0.25, 0.30, 0.25, 0.10]

    Examples:
        n_passing_classes = 1 -> [1.0]
        n_passing_classes = 5 -> default Bologna A/B/C/D/E
        n_passing_classes = 9 -> finer Bologna-shaped distribution
    """
    if n_passing_classes < 1:
        raise ValueError("n_passing_classes must be >= 1.")

    if n_passing_classes == 1:
        return [1.0]

    base_distribution = np.asarray(
        cfg.BOLOGNA_PASSING_DISTRIBUTION,
        dtype=float,
    )

    if base_distribution.ndim != 1 or len(base_distribution) == 0:
        raise ValueError("cfg.BOLOGNA_PASSING_DISTRIBUTION must be a non-empty list.")

    if not np.isclose(base_distribution.sum(), 1.0):
        base_distribution = base_distribution / base_distribution.sum()

    base_cdf = np.concatenate([[0.0], np.cumsum(base_distribution)])
    base_x = np.linspace(0.0, 1.0, len(base_cdf))

    target_x = np.linspace(0.0, 1.0, n_passing_classes + 1)
    target_cdf = np.interp(target_x, base_x, base_cdf)

    distribution = np.diff(target_cdf)
    distribution = np.maximum(distribution, 0.0)

    if distribution.sum() <= 0:
        return [1.0 / n_passing_classes] * n_passing_classes

    distribution = distribution / distribution.sum()

    return distribution.tolist()


def labels_from_normalized_scores_bologna_distribution(
    scores: pd.Series | np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Converts normalized exam scores to distribution-based Bologna-style classes.

    class 0:
        fail, score < PASS_THRESHOLD

    classes 1..n_classes-1:
        passing students are ranked by score and assigned to passing classes
        according to an interpolated Bologna-shaped distribution.

    Higher class value means better final grade.

    Tie handling:
        Identical score totals always receive the same category.
        If a tied group crosses a boundary, the full group receives the
        better category touched by the group's first rank.
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2.")

    scores_arr = np.asarray(scores, dtype=float)
    labels = np.full(len(scores_arr), np.nan, dtype=float)

    valid_mask = np.isfinite(scores_arr)

    if not valid_mask.any():
        return labels

    valid_indices = np.where(valid_mask)[0]
    valid_scores = scores_arr[valid_indices]

    labels[valid_indices] = 0.0

    passed_mask = valid_scores >= PASS_THRESHOLD

    if not passed_mask.any():
        return labels

    passed_indices = valid_indices[passed_mask]
    passed_scores = scores_arr[passed_indices]

    n_passed = len(passed_indices)
    n_passing_classes = n_classes - 1

    passing_distribution = interpolated_bologna_passing_distribution(
        n_passing_classes=n_passing_classes,
    )

    cumulative = np.cumsum(passing_distribution)
    cutoffs = [int(np.ceil(n_passed * x)) for x in cumulative]
    cutoffs[-1] = n_passed

    passed_df = (
        pd.DataFrame(
            {
                "idx": passed_indices,
                "score": passed_scores,
            }
        )
        .sort_values(["score", "idx"], ascending=[False, True])
        .reset_index(drop=True)
    )

    current_rank_start = 1

    for _, group in passed_df.groupby("score", sort=False):
        group_size = len(group)
        rank_start = current_rank_start
        rank_end = current_rank_start + group_size - 1

        pass_class_index = n_passing_classes - 1

        for candidate_index, cutoff in enumerate(cutoffs):
            if rank_start <= cutoff:
                pass_class_index = candidate_index
                break

        ordinal_label = n_classes - 1 - pass_class_index

        labels[group["idx"].to_numpy()] = float(ordinal_label)

        current_rank_start = rank_end + 1

    return labels


def labels_from_normalized_scores(
    scores: pd.Series | np.ndarray,
    n_classes: int,
    scale_type: str,
) -> np.ndarray:
    if scale_type == SCALE_ABSOLUTE:
        return labels_from_normalized_scores_absolute_threshold(
            scores=scores,
            n_classes=n_classes,
        )

    if scale_type == SCALE_BOLOGNA:
        return labels_from_normalized_scores_bologna_distribution(
            scores=scores,
            n_classes=n_classes,
        )

    raise ValueError(f"Unknown scale_type: {scale_type}")


# =========================================================
# STUDENT TOTALS
# =========================================================

def build_base_student_totals(df_env: pd.DataFrame) -> pd.DataFrame:
    base = df_env[
        [
            TEST_ID_COL,
            TEST_SIZE_COL,
            STUDENT_ID_COL,
            QUESTION_ID_COL,
            HUMAN_GRADE_COL,
        ]
    ].copy()

    base[HUMAN_GRADE_COL] = pd.to_numeric(
        base[HUMAN_GRADE_COL],
        errors="coerce",
    )

    grouped = (
        base.groupby([TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL], sort=False)
        .agg(
            n_rows=(QUESTION_ID_COL, "size"),
            n_questions=(QUESTION_ID_COL, "nunique"),
            human_valid=(HUMAN_GRADE_COL, lambda s: int(s.notna().sum())),
            gold_total=(HUMAN_GRADE_COL, "sum"),
        )
        .reset_index()
    )

    grouped[TEST_SIZE_COL] = pd.to_numeric(
        grouped[TEST_SIZE_COL],
        errors="coerce",
    ).astype("Int64")

    grouped["gold_norm"] = grouped["gold_total"] / grouped[TEST_SIZE_COL].astype(float)

    return grouped


def build_model_student_totals(
    df_env: pd.DataFrame,
    base_totals: pd.DataFrame,
    model_col: str,
) -> pd.DataFrame:
    pred_df = df_env[
        [
            TEST_ID_COL,
            TEST_SIZE_COL,
            STUDENT_ID_COL,
            model_col,
        ]
    ].copy()

    pred_df[model_col] = pd.to_numeric(pred_df[model_col], errors="coerce")

    pred_totals = (
        pred_df.groupby([TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL], sort=False)
        .agg(
            pred_valid=(model_col, lambda s: int(s.notna().sum())),
            pred_total=(model_col, "sum"),
        )
        .reset_index()
    )

    totals = base_totals.merge(
        pred_totals,
        on=[TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL],
        how="left",
        validate="one_to_one",
    )

    test_size_numeric = totals[TEST_SIZE_COL].astype(float)

    complete_mask = (
        (totals["n_rows"] == totals[TEST_SIZE_COL])
        & (totals["n_questions"] == totals[TEST_SIZE_COL])
        & (totals["human_valid"] == totals[TEST_SIZE_COL])
        & (totals["pred_valid"] == totals[TEST_SIZE_COL])
    )

    totals = totals[complete_mask].copy()
    totals["pred_norm"] = totals["pred_total"] / test_size_numeric[complete_mask]

    return totals


# =========================================================
# GRANULARITY METRICS
# =========================================================

def compute_granularity_metrics_for_model(
    df_env: pd.DataFrame,
    base_totals: pd.DataFrame,
    model_col: str,
) -> list[dict[str, Any]]:
    totals = build_model_student_totals(
        df_env=df_env,
        base_totals=base_totals,
        model_col=model_col,
    )

    rows: list[dict[str, Any]] = []

    grouped = totals.groupby([TEST_ID_COL, TEST_SIZE_COL], sort=True)

    for (test_id, test_size), exam_df in grouped:
        test_size_int = int(test_size)

        gold_norm = exam_df["gold_norm"].to_numpy(dtype=float)
        pred_norm = exam_df["pred_norm"].to_numpy(dtype=float)

        for scale_type in SCALE_TYPES:
            for n_classes in CLASS_COUNTS:
                gold_labels = labels_from_normalized_scores(
                    gold_norm,
                    n_classes=n_classes,
                    scale_type=scale_type,
                )
                pred_labels = labels_from_normalized_scores(
                    pred_norm,
                    n_classes=n_classes,
                    scale_type=scale_type,
                )

                distribution = ""
                if scale_type == SCALE_BOLOGNA:
                    distribution = ",".join(
                        f"{x:.6f}"
                        for x in interpolated_bologna_passing_distribution(
                            n_passing_classes=n_classes - 1
                        )
                    )

                rows.append(
                    {
                        MODEL_COL: model_col,
                        "model": display_name(model_col),
                        "family": model_family(model_col),
                        TEST_ID_COL: test_id,
                        TEST_SIZE_COL: test_size_int,
                        "scale_type": scale_type,
                        "n_classes": int(n_classes),
                        "n_passing_classes": int(n_classes - 1),
                        "scale_label": str(n_classes),
                        "scale_name": f"{scale_type}_{n_classes}_classes",
                        "pass_threshold": PASS_THRESHOLD,
                        "bologna_passing_distribution": distribution,
                        "n_students": int(len(exam_df)),
                        "el_acc": accuracy_safe(gold_labels, pred_labels),
                        "el_qwk": qwk_safe(
                            gold_labels,
                            pred_labels,
                            n_classes=n_classes,
                        ),
                        "el_tau": tau_b_safe(gold_labels, pred_labels),
                    }
                )

    return rows


def compute_granularity_metrics(
    df_env: pd.DataFrame,
    test_size: int,
) -> pd.DataFrame:
    base_totals = build_base_student_totals(df_env)
    all_rows: list[dict[str, Any]] = []

    for model_col in MODEL_COLUMNS:
        print(
            f"  computing q{test_size} granularity metrics for "
            f"{display_name(model_col)}"
        )
        all_rows.extend(
            compute_granularity_metrics_for_model(
                df_env=df_env,
                base_totals=base_totals,
                model_col=model_col,
            )
        )

    result = pd.DataFrame(all_rows)

    if result.empty:
        return result

    result[TEST_SIZE_COL] = pd.to_numeric(
        result[TEST_SIZE_COL],
        errors="coerce",
    ).astype("Int64")

    result["n_classes"] = pd.to_numeric(
        result["n_classes"],
        errors="coerce",
    ).astype("Int64")

    result["n_passing_classes"] = pd.to_numeric(
        result["n_passing_classes"],
        errors="coerce",
    ).astype("Int64")

    return result


def summarize_metrics(per_exam_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        per_exam_df.groupby(
            [
                MODEL_COL,
                "model",
                "family",
                TEST_SIZE_COL,
                "scale_type",
                "n_classes",
                "n_passing_classes",
                "scale_label",
                "scale_name",
                "pass_threshold",
                "bologna_passing_distribution",
            ],
            sort=False,
        )
        .agg(
            exam_instances=(TEST_ID_COL, "size"),
            n_runs=(TEST_ID_COL, "nunique"),
            n_students_mean=("n_students", "mean"),
            el_acc_mean=("el_acc", "mean"),
            el_acc_std=("el_acc", "std"),
            el_acc_missing=("el_acc", lambda s: int(s.isna().sum())),
            el_qwk_mean=("el_qwk", "mean"),
            el_qwk_std=("el_qwk", "std"),
            el_qwk_missing=("el_qwk", lambda s: int(s.isna().sum())),
            el_tau_mean=("el_tau", "mean"),
            el_tau_std=("el_tau", "std"),
            el_tau_missing=("el_tau", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )

    order_rows = []

    for scale_type in SCALE_TYPES:
        rank_source = summary[
            (summary["scale_type"] == scale_type)
            & (summary["n_classes"] == MAX_CLASSES)
        ].copy()

        rank_source = rank_source.sort_values(
            ["el_qwk_mean", "el_tau_mean", "el_acc_mean", "model"],
            ascending=[False, False, False, True],
        )

        for rank, model_col in enumerate(rank_source[MODEL_COL], start=1):
            order_rows.append(
                {
                    "scale_type": scale_type,
                    MODEL_COL: model_col,
                    "model_order": rank,
                }
            )

    order_df = pd.DataFrame(order_rows)

    summary = summary.merge(
        order_df,
        on=["scale_type", MODEL_COL],
        how="left",
        validate="many_to_one",
    )

    summary = summary.sort_values(
        [TEST_SIZE_COL, "scale_type", "model_order", "n_classes"]
    ).reset_index(drop=True)

    return summary


# =========================================================
# SANITY CHECK
# =========================================================

def write_sanity_check(
    raw_df: pd.DataFrame,
    q_df: pd.DataFrame,
    per_exam_df: pd.DataFrame,
    summary: pd.DataFrame,
    test_size: int,
    output_path: Path,
) -> None:
    lines: list[str] = []

    lines.append("=" * 100)
    lines.append(f"FIGURE 4 Q{test_size} SANITY CHECK")
    lines.append("=" * 100)
    lines.append(f"Script dir:            {SCRIPT_DIR}")
    lines.append(f"VEX metric dir:        {VEX_METRIC_DIR}")
    lines.append(f"Input parquet:         {INPUT_PARQUET}")
    lines.append(f"Output dir:            {OUTPUT_DIR}")
    lines.append(f"Raw env rows:          {len(raw_df)}")
    lines.append(f"Q{test_size} env rows: {len(q_df)}")
    lines.append(f"Target test_size:      {test_size}")
    lines.append(f"Pass threshold:        {PASS_THRESHOLD}")
    lines.append(f"Models:                {len(MODEL_COLUMNS)}")
    lines.append(f"Class counts:          {CLASS_COUNTS}")
    lines.append(f"Scale types:           {SCALE_TYPES}")
    lines.append("")
    lines.append("Absolute threshold scale:")
    lines.append("  class 0 = fail, normalized score < pass threshold")
    lines.append("  classes 1..n-1 = equal-width passing categories on [threshold, 1]")
    lines.append("  n=2 therefore corresponds to fail/pass.")
    lines.append("")
    lines.append("Bologna distribution scale:")
    lines.append("  class 0 = fail, normalized score < pass threshold")
    lines.append("  classes 1..n-1 = ranked passing categories")
    lines.append("  passing-category distribution is interpolated from:")
    lines.append(f"  {cfg.BOLOGNA_PASSING_DISTRIBUTION}")
    lines.append("  n=6 therefore corresponds to F + A/B/C/D/E-style Bologna.")
    lines.append("")
    lines.append("Interpolated Bologna passing distributions:")
    for n_classes in CLASS_COUNTS:
        dist = interpolated_bologna_passing_distribution(n_classes - 1)
        lines.append(
            f"  n_classes={n_classes}, n_passing={n_classes - 1}: "
            + ", ".join(f"{x:.6f}" for x in dist)
        )
    lines.append("")
    lines.append("Raw counts by test_size:")

    raw_counts = (
        raw_df.groupby(TEST_SIZE_COL)
        .agg(
            rows=(TEST_ID_COL, "size"),
            virtual_exams=(TEST_ID_COL, "nunique"),
            students=(STUDENT_ID_COL, "nunique"),
            questions=(QUESTION_ID_COL, "nunique"),
        )
        .reset_index()
    )

    lines.append(raw_counts.to_string(index=False))
    lines.append("")
    lines.append(f"Q{test_size} counts:")

    q_counts = (
        q_df.groupby(TEST_SIZE_COL)
        .agg(
            rows=(TEST_ID_COL, "size"),
            virtual_exams=(TEST_ID_COL, "nunique"),
            students=(STUDENT_ID_COL, "nunique"),
            questions=(QUESTION_ID_COL, "nunique"),
        )
        .reset_index()
    )

    lines.append(q_counts.to_string(index=False))
    lines.append("")
    lines.append(f"Per-exam metric rows: {len(per_exam_df)}")
    lines.append("")
    lines.append("Expected per-exam metric rows:")
    expected_rows = (
        len(MODEL_COLUMNS)
        * q_df[[TEST_ID_COL, TEST_SIZE_COL]].drop_duplicates().shape[0]
        * len(CLASS_COUNTS)
        * len(SCALE_TYPES)
    )
    lines.append(f"  {expected_rows}")
    lines.append("")
    lines.append("Summary:")
    lines.append(summary.to_string(index=False))

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# PLOTTING
# =========================================================

def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", color="#d9d9d9", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)


def get_top_models(
    summary: pd.DataFrame,
    metric_mean_col: str,
    scale_type: str,
    n_top: int = 8,
) -> set[str]:
    last_points = (
        summary[
            (summary["scale_type"] == scale_type)
            & (summary["n_classes"] == MAX_CLASSES)
        ]
        .sort_values([metric_mean_col, "model"], ascending=[False, True])
        .copy()
    )

    return set(last_points.head(n_top)[MODEL_COL].tolist())


def save_single_metric_plot(
    summary: pd.DataFrame,
    scale_type: str,
    metric_mean_col: str,
    y_label: str,
    title: str,
    output_pdf: Path,
    output_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.8))

    plot_df = summary[summary["scale_type"] == scale_type].copy()

    top_models = get_top_models(
        summary=summary,
        metric_mean_col=metric_mean_col,
        scale_type=scale_type,
        n_top=8,
    )

    for model_col, model_df in plot_df.groupby(MODEL_COL, sort=False):
        model_df = model_df.sort_values("n_classes")

        family = str(model_df["family"].iloc[0])
        color = FAMILY_COLORS.get(family, "#666666")
        linestyle = FAMILY_LINESTYLES.get(family, "-")
        model_name = str(model_df["model"].iloc[0])
        is_top = model_col in top_models

        ax.plot(
            model_df["n_classes"],
            model_df[metric_mean_col],
            marker="o",
            markersize=4 if is_top else 3,
            linewidth=1.8 if is_top else 0.9,
            alpha=0.95 if is_top else 0.35,
            color=color,
            linestyle=linestyle,
            label=model_name,
        )

    ax.set_xlabel("Grade-scale granularity (number of final classes)")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_xticks(CLASS_COUNTS)
    ax.set_xticklabels([str(x) for x in CLASS_COUNTS])

    y_values = pd.to_numeric(plot_df[metric_mean_col], errors="coerce").dropna()

    if not y_values.empty:
        y_min = max(0.0, float(y_values.min()) - 0.05)
        y_max = min(1.0, float(y_values.max()) + 0.04)
        ax.set_ylim(y_min, y_max)

    style_axes(ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=6,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_plots(summary: pd.DataFrame, test_size: int) -> list[Path]:
    paths = figure_4_paths(test_size)

    save_single_metric_plot(
        summary=summary,
        scale_type=SCALE_ABSOLUTE,
        metric_mean_col="el_acc_mean",
        y_label="Mean exam-level accuracy",
        title=f"Q{test_size} Absolute EL-Acc vs. Grade-Scale Granularity",
        output_pdf=paths["abs_acc_pdf"],
        output_png=paths["abs_acc_png"],
    )

    save_single_metric_plot(
        summary=summary,
        scale_type=SCALE_ABSOLUTE,
        metric_mean_col="el_qwk_mean",
        y_label="Mean exam-level QWK",
        title=f"Q{test_size} Absolute EL-QWK vs. Grade-Scale Granularity",
        output_pdf=paths["abs_qwk_pdf"],
        output_png=paths["abs_qwk_png"],
    )

    save_single_metric_plot(
        summary=summary,
        scale_type=SCALE_ABSOLUTE,
        metric_mean_col="el_tau_mean",
        y_label=r"Mean exam-level $\tau_b$",
        title=rf"Q{test_size} Absolute EL-$\tau_b$ vs. Grade-Scale Granularity",
        output_pdf=paths["abs_tau_pdf"],
        output_png=paths["abs_tau_png"],
    )

    save_single_metric_plot(
        summary=summary,
        scale_type=SCALE_BOLOGNA,
        metric_mean_col="el_acc_mean",
        y_label="Mean exam-level accuracy",
        title=f"Q{test_size} Bologna EL-Acc vs. Grade-Scale Granularity",
        output_pdf=paths["bol_acc_pdf"],
        output_png=paths["bol_acc_png"],
    )

    save_single_metric_plot(
        summary=summary,
        scale_type=SCALE_BOLOGNA,
        metric_mean_col="el_qwk_mean",
        y_label="Mean exam-level QWK",
        title=f"Q{test_size} Bologna EL-QWK vs. Grade-Scale Granularity",
        output_pdf=paths["bol_qwk_pdf"],
        output_png=paths["bol_qwk_png"],
    )

    save_single_metric_plot(
        summary=summary,
        scale_type=SCALE_BOLOGNA,
        metric_mean_col="el_tau_mean",
        y_label=r"Mean exam-level $\tau_b$",
        title=rf"Q{test_size} Bologna EL-$\tau_b$ vs. Grade-Scale Granularity",
        output_pdf=paths["bol_tau_pdf"],
        output_png=paths["bol_tau_png"],
    )

    return [
        paths["abs_acc_pdf"],
        paths["abs_acc_png"],
        paths["abs_qwk_pdf"],
        paths["abs_qwk_png"],
        paths["abs_tau_pdf"],
        paths["abs_tau_png"],
        paths["bol_acc_pdf"],
        paths["bol_acc_png"],
        paths["bol_qwk_pdf"],
        paths["bol_qwk_png"],
        paths["bol_tau_pdf"],
        paths["bol_tau_png"],
    ]


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CREATE GRANULARITY PLOTS")
    print("=" * 100)
    print(f"Script dir:            {SCRIPT_DIR}")
    print(f"VEX metric dir:        {VEX_METRIC_DIR}")
    print(f"Input parquet:         {INPUT_PARQUET}")
    print(f"Output folder:         {OUTPUT_DIR}")
    print(f"Pass threshold:        {PASS_THRESHOLD}")
    print(f"Class counts:          {CLASS_COUNTS}")
    print(f"Scale types:           {SCALE_TYPES}")
    print(f"Models:                {len(MODEL_COLUMNS)}")
    print("")

    raw_df = read_parquet(INPUT_PARQUET)
    validate_input(raw_df)
    assert_no_duplicate_exam_student_question_pairs(raw_df)

    test_size_numeric = pd.to_numeric(raw_df[TEST_SIZE_COL], errors="coerce")
    available_test_sizes = sorted(
        int(value) for value in test_size_numeric.dropna().unique()
    )

    if TARGET_TEST_SIZES is None:
        target_test_sizes = available_test_sizes
    else:
        requested_test_sizes = [int(value) for value in TARGET_TEST_SIZES]
        target_test_sizes = [
            value for value in requested_test_sizes if value in available_test_sizes
        ]

        missing_test_sizes = sorted(
            set(requested_test_sizes).difference(available_test_sizes)
        )
        if missing_test_sizes:
            print(
                "Warning: requested test sizes not present in input: "
                f"{missing_test_sizes}"
            )

    if not target_test_sizes:
        raise RuntimeError(
            f"No target test sizes found in {INPUT_PARQUET}. "
            f"Available values: {available_test_sizes}"
        )

    print(f"Raw rows: {len(raw_df)}")
    print(f"Available test sizes: {available_test_sizes}")
    print(f"Target test sizes:    {target_test_sizes}")
    print("")

    all_per_exam: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    written_paths: list[Path] = []

    for test_size in target_test_sizes:
        paths = figure_4_paths(test_size)

        q_df = raw_df[test_size_numeric == test_size].copy()

        if q_df.empty:
            print(f"Skipping q{test_size}: no rows found.")
            continue

        print("-" * 100)
        print(f"Q{test_size}")
        print("-" * 100)
        print(f"Rows: {len(q_df)}")

        if paths["per_exam_csv"].exists() and not RECOMPUTE_PER_EXAM_METRICS:
            print(f"Loading cached per-exam metrics: {paths['per_exam_csv']}")
            per_exam_df = pd.read_csv(paths["per_exam_csv"])
        else:
            print("Computing per-exam granularity metrics...")
            per_exam_df = compute_granularity_metrics(q_df, test_size)
            per_exam_df.to_csv(
                paths["per_exam_csv"],
                index=False,
                encoding="utf-8",
            )

        if per_exam_df.empty:
            raise RuntimeError(f"No per-exam metric rows were computed for q{test_size}.")

        summary = summarize_metrics(per_exam_df)

        summary.to_csv(paths["summary_csv"], index=False, encoding="utf-8")

        write_sanity_check(
            raw_df=raw_df,
            q_df=q_df,
            per_exam_df=per_exam_df,
            summary=summary,
            test_size=test_size,
            output_path=paths["sanity_txt"],
        )

        written_paths.extend(save_plots(summary=summary, test_size=test_size))
        written_paths.extend(
            [
                paths["summary_csv"],
                paths["per_exam_csv"],
                paths["sanity_txt"],
            ]
        )

        all_per_exam.append(per_exam_df)
        all_summary.append(summary)

    if not all_per_exam or not all_summary:
        raise RuntimeError("No plot data was generated.")

    combined_per_exam = pd.concat(all_per_exam, ignore_index=True)
    combined_summary = pd.concat(all_summary, ignore_index=True)

    combined_per_exam.to_csv(COMBINED_PER_EXAM_CSV, index=False, encoding="utf-8")
    combined_summary.to_csv(COMBINED_SUMMARY_CSV, index=False, encoding="utf-8")

    written_paths.extend([COMBINED_SUMMARY_CSV, COMBINED_PER_EXAM_CSV])

    print("")
    print("Saved files:")
    for path in written_paths:
        print(f"  {path}")

    print("")
    print("DONE")


if __name__ == "__main__":
    main()
