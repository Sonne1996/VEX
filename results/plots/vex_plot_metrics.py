#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared plotting utilities for VEX result figures.

The helpers compute exam-level metrics directly from dataframe_env.parquet.
Results are cached because the raw environment contains several million rows.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
VEX_METRIC_DIR = PROJECT_ROOT / "vex_metric"
DATAFRAME_DIR = VEX_METRIC_DIR / "vex_test_env" / "4_dataframe"
OUTPUT_CACHE_DIR = SCRIPT_DIR / "_metric_cache"
EXAM_METRICS_CACHE = OUTPUT_CACHE_DIR / "exam_level_metrics_cache.csv"
FIGURE_4_DIR = SCRIPT_DIR / "figures_plot_4"

sys.path.insert(0, str(VEX_METRIC_DIR))

import vex_config as cfg


MODEL_COLUMNS = list(cfg.MODEL_COLUMNS)

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
    "grade_bert_ft": "BERT FT",
    "grade_mdeberta_ft": "mDeBERTa FT",
    "grade_prior_global": "Global Prior",
    "grade_prior_template_overlap": "Template",
    "pred_tfidf_v5_answer_char_3_5": "TF-IDF Char",
    "pred_tfidf_v1_answer_word_unigram": "TF-IDF Word",
    "pred_tfidf_v4_question_and_answer_separate": "TF-IDF QA",
}

FAMILY_COLORS = {
    "LLM": "#1f77b4",
    "Transformer": "#2ca02c",
    "Prior": "#9467bd",
    "TF-IDF": "#ff7f0e",
    "Other": "#666666",
}

FAMILY_MARKERS = {
    "LLM": "o",
    "Transformer": "s",
    "Prior": "D",
    "TF-IDF": "^",
    "Other": "o",
}

BLOOM_ORDER = ["remember", "understand", "apply", "analyse"]
BLOOM_LABELS = {
    "remember": "Remember",
    "understand": "Understand",
    "apply": "Apply",
    "analyse": "Analyse",
}


def display_name(model_col: str) -> str:
    return DISPLAY_NAMES.get(model_col, short_model_name(model_col))


def short_model_name(model_col: str) -> str:
    name = str(model_col)
    for prefix in ("new_grade_", "grade_", "pred_"):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
    return name.replace("/", "_")


def model_family(model_col: str) -> str:
    if model_col.startswith("new_grade_"):
        return "LLM"
    if model_col.startswith("grade_bert") or model_col.startswith("grade_mdeberta"):
        return "Transformer"
    if model_col.startswith("grade_prior"):
        return "Prior"
    if model_col.startswith("pred_tfidf"):
        return "TF-IDF"
    return "Other"


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def ordinal_encode_pair(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    mapping = {value: idx for idx, value in enumerate(values)}
    true_codes = np.array([mapping[value] for value in y_true], dtype=int)
    pred_codes = np.array([mapping[value] for value in y_pred], dtype=int)
    return true_codes, pred_codes


def qwk_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return np.nan

    true_codes, pred_codes = ordinal_encode_pair(y_true, y_pred)

    if len(np.unique(true_codes)) == 1 and len(np.unique(pred_codes)) == 1:
        return 1.0 if np.array_equal(true_codes, pred_codes) else 0.0

    n_classes = int(max(true_codes.max(), pred_codes.max()) + 1)
    observed = np.zeros((n_classes, n_classes), dtype=float)

    for true_value, pred_value in zip(true_codes, pred_codes, strict=False):
        observed[int(true_value), int(pred_value)] += 1.0

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
        return 1.0 if np.array_equal(true_codes, pred_codes) else 0.0

    return 1.0 - (observed_weighted / expected_weighted)


def tau_b_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(y_true) != len(y_pred):
        return np.nan

    true_codes, pred_codes = ordinal_encode_pair(y_true, y_pred)

    if len(np.unique(true_codes)) <= 1 or len(np.unique(pred_codes)) <= 1:
        return np.nan

    n_classes = int(max(true_codes.max(), pred_codes.max()) + 1)
    contingency = np.zeros((n_classes, n_classes), dtype=np.int64)

    for true_value, pred_value in zip(true_codes, pred_codes, strict=False):
        contingency[int(true_value), int(pred_value)] += 1

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


def accuracy_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return np.nan
    return float(np.mean(y_true == y_pred))


def round_and_clip_linear_grades(grades: pd.Series) -> pd.Series:
    grades = pd.to_numeric(grades, errors="coerce")
    step = float(cfg.LINEAR_ROUNDING_STEP)
    if step > 0:
        grades = (grades / step).round() * step
    return grades.clip(lower=float(cfg.LINEAR_MIN_GRADE), upper=float(cfg.LINEAR_MAX_GRADE))


def normalized_to_linear_grade(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    grades = float(cfg.LINEAR_MIN_GRADE) + (
        (float(cfg.LINEAR_MAX_GRADE) - float(cfg.LINEAR_MIN_GRADE)) * numeric
    )
    return round_and_clip_linear_grades(grades)


def label_for_rank_position(position_1_based: int, cutoffs: list[int]) -> str:
    for label, cutoff in zip(cfg.DISTRIBUTION_PASSING_LABELS, cutoffs, strict=False):
        if position_1_based <= cutoff:
            return str(label)
    return str(cfg.DISTRIBUTION_PASSING_LABELS[-1])


def assign_distrobution_labels_from_normalized(
    normalized_scores: pd.Series,
    test_size: int,
) -> pd.Series:
    scores = pd.to_numeric(normalized_scores, errors="coerce")
    absolute_points = scores * float(test_size)
    pass_threshold_abs = float(test_size) * float(cfg.DISTRIBUTION_PASS_THRESHOLD_NORM)

    result = pd.Series(cfg.DISTRIBUTION_FAIL_LABEL, index=scores.index, dtype="object")
    passed = absolute_points[absolute_points >= pass_threshold_abs].dropna()
    if passed.empty:
        return result

    cumulative = np.cumsum(cfg.DISTRIBUTION_PASSING_DISTRIBUTION)
    cutoffs = [int(np.ceil(len(passed) * value)) for value in cumulative]
    cutoffs[-1] = len(passed)

    ranked = (
        pd.DataFrame({"idx": passed.index, "points": passed.values})
        .sort_values(["points", "idx"], ascending=[False, True])
        .reset_index(drop=True)
    )

    rank_start = 1
    for _points_value, tied_group in ranked.groupby("points", sort=False):
        label = label_for_rank_position(rank_start, cutoffs)
        result.loc[tied_group["idx"].tolist()] = label
        rank_start += len(tied_group)

    return result


def distrobution_labels_to_ordinals(labels: pd.Series | list[str]) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(cfg.DISTRIBUTION_ORDERED_LABELS)}
    return np.array([mapping[x] for x in labels], dtype=int)


def q_env_paths() -> list[Path]:
    q_paths = sorted(DATAFRAME_DIR.glob("df_env_q*.parquet"))
    if q_paths:
        return q_paths

    fallback = DATAFRAME_DIR / "dataframe_env.parquet"
    if fallback.exists():
        return [fallback]

    return []


def test_size_from_path(path: Path) -> int | None:
    stem = path.stem
    if "_q" not in stem:
        return None
    try:
        return int(stem.rsplit("_q", 1)[1])
    except ValueError:
        return None


def compute_exam_metrics_for_q(df: pd.DataFrame, test_size_hint: int | None) -> pd.DataFrame:
    required = ["test_id", "test_size", "question_id", "member_id", "human_grade"]
    require_columns(df, [*required, *MODEL_COLUMNS])

    if test_size_hint is not None:
        df = df[pd.to_numeric(df["test_size"], errors="coerce") == int(test_size_hint)].copy()

    df["human_grade"] = pd.to_numeric(df["human_grade"], errors="coerce")
    df["test_size"] = pd.to_numeric(df["test_size"], errors="coerce").astype(int)

    base = (
        df.groupby(["test_id", "test_size", "member_id"], sort=False)
        .agg(
            n_rows=("question_id", "size"),
            n_questions=("question_id", "nunique"),
            human_valid=("human_grade", lambda s: int(s.notna().sum())),
            gold_total=("human_grade", "sum"),
        )
        .reset_index()
    )

    rows: list[dict[str, Any]] = []

    for model_col in MODEL_COLUMNS:
        print(f"  computing exam metrics for {display_name(model_col)}")
        pred = df[["test_id", "test_size", "member_id", model_col]].copy()
        pred[model_col] = pd.to_numeric(pred[model_col], errors="coerce")

        pred_grouped = (
            pred.groupby(["test_id", "test_size", "member_id"], sort=False)
            .agg(
                pred_valid=(model_col, lambda s: int(s.notna().sum())),
                pred_total=(model_col, "sum"),
            )
            .reset_index()
        )

        totals = base.merge(
            pred_grouped,
            on=["test_id", "test_size", "member_id"],
            how="left",
        )
        totals["pred_valid"] = totals["pred_valid"].fillna(0).astype(int)
        totals["pred_total"] = pd.to_numeric(totals["pred_total"], errors="coerce")

        for (test_id, test_size), exam_df in totals.groupby(["test_id", "test_size"], sort=False):
            test_size_int = int(test_size)
            complete_mask = (
                (exam_df["n_rows"] == test_size_int)
                & (exam_df["n_questions"] == test_size_int)
                & (exam_df["human_valid"] == test_size_int)
                & (exam_df["pred_valid"] == test_size_int)
            )
            complete = exam_df[complete_mask].copy()

            row: dict[str, Any] = {
                "model_col": model_col,
                "model": display_name(model_col),
                "family": model_family(model_col),
                "test_id": test_id,
                "test_size": test_size_int,
                "students_raw": int(len(exam_df)),
                "n_students": int(len(complete)),
                "students_dropped_incomplete": int((~complete_mask).sum()),
                "students_missing_human": int((exam_df["human_valid"] < test_size_int).sum()),
                "students_missing_prediction": int((exam_df["pred_valid"] < test_size_int).sum()),
            }

            if complete.empty:
                row.update(
                    {
                        "el_tau_b": np.nan,
                        "el_acc_linear_abs": np.nan,
                        "el_qwk_linear_abs": np.nan,
                        "el_acc_distrobution": np.nan,
                        "el_qwk_distrobution": np.nan,
                    }
                )
                rows.append(row)
                continue

            complete["gold_norm"] = complete["gold_total"] / float(test_size_int)
            complete["pred_norm"] = complete["pred_total"] / float(test_size_int)

            complete["gold_linear_abs"] = normalized_to_linear_grade(complete["gold_norm"])
            complete["pred_linear_abs"] = normalized_to_linear_grade(complete["pred_norm"])
            complete["gold_distrobution"] = assign_distrobution_labels_from_normalized(
                complete["gold_norm"],
                test_size_int,
            )
            complete["pred_distrobution"] = assign_distrobution_labels_from_normalized(
                complete["pred_norm"],
                test_size_int,
            )

            row.update(
                {
                    "el_tau_b": tau_b_safe(
                        complete["gold_total"].to_numpy(),
                        complete["pred_total"].to_numpy(),
                    ),
                    "el_acc_linear_abs": accuracy_safe(
                        complete["gold_linear_abs"].to_numpy(),
                        complete["pred_linear_abs"].to_numpy(),
                    ),
                    "el_qwk_linear_abs": qwk_safe(
                        complete["gold_linear_abs"].to_numpy(),
                        complete["pred_linear_abs"].to_numpy(),
                    ),
                    "el_acc_distrobution": accuracy_safe(
                        complete["gold_distrobution"].to_numpy(),
                        complete["pred_distrobution"].to_numpy(),
                    ),
                    "el_qwk_distrobution": qwk_safe(
                        distrobution_labels_to_ordinals(complete["gold_distrobution"]),
                        distrobution_labels_to_ordinals(complete["pred_distrobution"]),
                    ),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def load_or_compute_exam_metrics(force: bool = False) -> pd.DataFrame:
    if EXAM_METRICS_CACHE.exists() and not force:
        return pd.read_csv(EXAM_METRICS_CACHE)

    OUTPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []
    env_paths = q_env_paths()

    if not env_paths:
        result = load_exam_metrics_from_granularity_exports()
        result.to_csv(EXAM_METRICS_CACHE, index=False, encoding="utf-8")
        return result

    for path in env_paths:
        test_size_hint = test_size_from_path(path)
        columns = [
            "test_id",
            "test_size",
            "question_id",
            "member_id",
            "human_grade",
            *MODEL_COLUMNS,
        ]
        print(f"Reading {path}")
        df = pd.read_parquet(path, columns=columns)
        all_frames.append(compute_exam_metrics_for_q(df, test_size_hint))

    result = pd.concat(all_frames, ignore_index=True)
    result["metric_source"] = "dataframe_env"
    result.to_csv(EXAM_METRICS_CACHE, index=False, encoding="utf-8")
    return result


def load_exam_metrics_from_granularity_exports() -> pd.DataFrame:
    files = sorted(FIGURE_4_DIR.glob("figure_4_q*_granularity_per_exam.csv"))
    if not files:
        raise FileNotFoundError(
            "No dataframe_env parquet and no Figure-4 per-exam metric exports found."
        )

    frames: list[pd.DataFrame] = []
    usecols = [
        "model_col",
        "model",
        "family",
        "test_id",
        "test_size",
        "scale_type",
        "n_classes",
        "n_students",
        "el_acc",
        "el_qwk",
        "el_tau",
    ]

    for path in files:
        print(f"Reading fallback metric export {path}")
        df = pd.read_csv(path, usecols=usecols)
        df = df[pd.to_numeric(df["n_classes"], errors="coerce") == 6].copy()
        if df.empty:
            continue

        key_cols = ["model_col", "model", "family", "test_id", "test_size"]
        abs_df = df[df["scale_type"].eq("absolute_threshold")].copy()
        dist_df = df[~df["scale_type"].eq("absolute_threshold")].copy()

        abs_df = abs_df[
            [*key_cols, "n_students", "el_acc", "el_qwk", "el_tau"]
        ].rename(
            columns={
                "el_acc": "el_acc_linear_abs",
                "el_qwk": "el_qwk_linear_abs",
                "el_tau": "el_tau_b",
            }
        )
        dist_df = dist_df[[*key_cols, "el_acc", "el_qwk"]].rename(
            columns={
                "el_acc": "el_acc_distrobution",
                "el_qwk": "el_qwk_distrobution",
            }
        )

        merged = abs_df.merge(dist_df, on=key_cols, how="inner")
        merged["students_raw"] = merged["n_students"]
        merged["students_dropped_incomplete"] = 0
        merged["students_missing_human"] = 0
        merged["students_missing_prediction"] = 0
        merged["metric_source"] = "figure_4_granularity_exports_n_classes_6"
        frames.append(merged)

    if not frames:
        raise RuntimeError(f"No n_classes=6 rows found in {FIGURE_4_DIR}")

    return pd.concat(frames, ignore_index=True)


def summarize_exam_metrics(exam_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "el_tau_b",
        "el_acc_linear_abs",
        "el_qwk_linear_abs",
        "el_acc_distrobution",
        "el_qwk_distrobution",
    ]
    summary = (
        exam_df.groupby(["model_col", "model", "family", "test_size"], sort=True)
        .agg(
            exam_instances=("test_id", "nunique"),
            n_students_mean=("n_students", "mean"),
            n_students_total=("n_students", "sum"),
            **{f"{col}_mean": (col, "mean") for col in metric_cols},
            **{f"{col}_std": (col, "std") for col in metric_cols},
        )
        .reset_index()
    )
    return summary


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
