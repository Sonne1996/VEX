#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from vex_config import (
    INPUT_PARQUET,
    TEST_ENV_FOLDER,
    TESTS_ROOT_FOLDER,
    TEST_RUN_FOLDER,
    QUESTIONS_FOLDER,
    STUDENTS_FOLDER,
    QUESTION_FILE,
    STUDENT_FILE,
    TEST_SIZES,
    N_RUNS,
    OUTPUT_PARQUET,
    TESTS_DATAFRAME,
    MODEL_COLUMNS,
)

# =========================================================
# INPUT COLUMN CONFIG
# =========================================================

QUESTION_ID_COL = "question_id"
STUDENT_ID_COL = "member_id"
ANSWER_ID_COL = "answer_id"
QUESTION_COL = "question"
ANSWER_COL = "answer"
BLOOM_COL = "bloom_level"
TOPIC_COL = "question_topic"
HUMAN_GRADE_COL = "grade"

HUMAN_ONE_GOLD = "human_expert_one_gold"
HUMAN_TWO_MODEL = "human_expert_two"

# Optional
NAME_COL = "name"


# =========================================================
# PATH HELPERS
# =========================================================

def _test_env_root() -> Path:
    return Path(TEST_ENV_FOLDER)


def _tests_root_dir() -> Path:
    return _test_env_root() / TESTS_ROOT_FOLDER


def _run_dir(run_idx: int) -> Path:
    return _tests_root_dir() / TEST_RUN_FOLDER.format(test_number=run_idx)


def _questions_dir(run_idx: int) -> Path:
    return _run_dir(run_idx) / QUESTIONS_FOLDER


def _students_dir(run_idx: int) -> Path:
    return _run_dir(run_idx) / STUDENTS_FOLDER


def _question_file(run_idx: int, test_size: int) -> Path:
    return _questions_dir(run_idx) / QUESTION_FILE.format(
        questions_number=test_size
    )


def _student_file(run_idx: int, test_size: int) -> Path:
    return _students_dir(run_idx) / STUDENT_FILE.format(
        questions_number=test_size
    )


def _dataframe_dir() -> Path:
    return _test_env_root() / TESTS_DATAFRAME


def _df_test_questions_path() -> Path:
    return _dataframe_dir() / "df_test_questions.parquet"


def _df_test_students_path() -> Path:
    return _dataframe_dir() / "df_test_students.parquet"


def _df_answers_master_path() -> Path:
    return _dataframe_dir() / "df_answers_master.parquet"


def _df_grades_path() -> Path:
    return _dataframe_dir() / "df_grades.parquet"


def _df_env_by_test_size_path(test_size: int) -> Path:
    return _dataframe_dir() / f"df_env_q{test_size}.parquet"


# =========================================================
# TXT HELPERS
# =========================================================

def read_txt_file(path: str | Path) -> str:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path.resolve()}")

    return file_path.read_text(encoding="utf-8")


def read_txt_lines(path: str | Path, drop_empty: bool = True) -> list[str]:
    text = read_txt_file(path)
    lines = [line.strip() for line in text.splitlines()]

    if drop_empty:
        lines = [line for line in lines if line != ""]

    return lines


def read_id_txt_file(path: str | Path) -> list[str]:
    return read_txt_lines(path, drop_empty=True)


# =========================================================
# VALIDATION / NORMALIZATION
# =========================================================

def _normalize_string_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _validate_input_columns(df: pd.DataFrame) -> None:
    required = [
        QUESTION_ID_COL,
        STUDENT_ID_COL,
        ANSWER_ID_COL,
        QUESTION_COL,
        ANSWER_COL,
        BLOOM_COL,
        TOPIC_COL,
        HUMAN_GRADE_COL,
        HUMAN_ONE_GOLD,
        HUMAN_TWO_MODEL,
    ]

    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        raise ValueError(
            f"Pflichtspalten fehlen im Input-Parquet: {missing_required}"
        )

    missing_models = [col for col in MODEL_COLUMNS if col not in df.columns]
    if missing_models:
        raise ValueError(
            f"Modellspalten fehlen im Input-Parquet: {missing_models}"
        )


# =========================================================
# BUILDERS: ENV TABLES
# =========================================================

def build_test_questions_df() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for run_idx in range(1, N_RUNS + 1):
        test_id = f"test_{run_idx}"

        for test_size in TEST_SIZES:
            file_path = _question_file(run_idx, test_size)

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Question-Datei nicht gefunden: {file_path.resolve()}"
                )

            question_ids = read_id_txt_file(file_path)

            for question_order, question_id in enumerate(question_ids, start=1):
                rows.append(
                    {
                        "test_id": test_id,
                        "test_size": test_size,
                        "question_id": str(question_id),
                        "question_order": question_order,
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("df_test_questions ist leer.")

    return df


def build_test_students_df() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for run_idx in range(1, N_RUNS + 1):
        test_id = f"test_{run_idx}"

        for test_size in TEST_SIZES:
            file_path = _student_file(run_idx, test_size)

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Student-Datei nicht gefunden: {file_path.resolve()}"
                )

            student_ids = read_id_txt_file(file_path)

            for member_id in student_ids:
                rows.append(
                    {
                        "test_id": test_id,
                        "test_size": test_size,
                        "member_id": str(member_id),
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("df_test_students ist leer.")

    return df


# =========================================================
# BUILDERS: INPUT TABLES
# =========================================================

def build_answers_master_df(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    _validate_input_columns(df)

    keep_cols = [
        ANSWER_ID_COL,
        QUESTION_ID_COL,
        STUDENT_ID_COL,
        QUESTION_COL,
        ANSWER_COL,
        BLOOM_COL,
        TOPIC_COL,
    ]

    if NAME_COL in df.columns:
        keep_cols.append(NAME_COL)

    df_answers = df[keep_cols].copy()

    df_answers[ANSWER_ID_COL] = _normalize_string_series(df_answers[ANSWER_ID_COL])
    df_answers[QUESTION_ID_COL] = _normalize_string_series(df_answers[QUESTION_ID_COL])
    df_answers[STUDENT_ID_COL] = _normalize_string_series(df_answers[STUDENT_ID_COL])

    df_answers[QUESTION_COL] = df_answers[QUESTION_COL].astype(str)
    df_answers[ANSWER_COL] = df_answers[ANSWER_COL].astype(str)
    df_answers[BLOOM_COL] = df_answers[BLOOM_COL].astype(str)
    df_answers[TOPIC_COL] = df_answers[TOPIC_COL].astype(str)

    df_answers = df_answers.drop_duplicates(subset=[ANSWER_ID_COL]).copy()

    if df_answers[ANSWER_ID_COL].duplicated().any():
        raise ValueError("df_answers_master enthält doppelte answer_id.")

    return df_answers


def build_grades_df(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    _validate_input_columns(df)

    keep_cols = [
        ANSWER_ID_COL,
        HUMAN_GRADE_COL,
        HUMAN_ONE_GOLD,
        HUMAN_TWO_MODEL,
    ] + MODEL_COLUMNS

    df_grades = df[keep_cols].copy()

    df_grades[ANSWER_ID_COL] = _normalize_string_series(df_grades[ANSWER_ID_COL])

    rename_map = {
        HUMAN_GRADE_COL: "human_grade",
    }

    df_grades = df_grades.rename(columns=rename_map)

    df_grades = df_grades.drop_duplicates(subset=[ANSWER_ID_COL]).copy()

    if df_grades[ANSWER_ID_COL].duplicated().any():
        raise ValueError("df_grades enthält doppelte answer_id.")

    return df_grades


# =========================================================
# FINAL JOIN
# =========================================================

def build_env_dataframe(
    df_test_questions: pd.DataFrame,
    df_test_students: pd.DataFrame,
    df_answers_master: pd.DataFrame,
    df_grades: pd.DataFrame,
) -> pd.DataFrame:
    df_test_structure = df_test_questions.merge(
        df_test_students,
        on=["test_id", "test_size"],
        how="inner",
        validate="many_to_many",
    )

    df_env = df_test_structure.merge(
        df_answers_master,
        on=["question_id", "member_id"],
        how="left",
        validate="many_to_many",
    )

    df_env = df_env.merge(
        df_grades,
        on="answer_id",
        how="left",
        validate="many_to_one",
    )

    column_order = [
        "test_id",
        "test_size",
        "question_order",
        "question_id",
        "member_id",
        "answer_id",
        "question",
        "answer",
        "bloom",
        "topic",
        "human_grade",
        HUMAN_ONE_GOLD,
        HUMAN_TWO_MODEL,
    ]

    column_order.extend(MODEL_COLUMNS)

    if NAME_COL in df_env.columns:
        column_order.append(NAME_COL)

    existing_columns = [col for col in column_order if col in df_env.columns]
    other_columns = [col for col in df_env.columns if col not in existing_columns]

    df_env = df_env[existing_columns + other_columns].copy()

    return df_env


# =========================================================
# SAVE
# =========================================================

def save_dataframes(
    df_test_questions: pd.DataFrame,
    df_test_students: pd.DataFrame,
    df_answers_master: pd.DataFrame,
    df_grades: pd.DataFrame,
    df_env: pd.DataFrame,
) -> None:
    _dataframe_dir().mkdir(parents=True, exist_ok=True)

    df_test_questions.to_parquet(_df_test_questions_path(), index=False)
    df_test_students.to_parquet(_df_test_students_path(), index=False)
    df_answers_master.to_parquet(_df_answers_master_path(), index=False)
    df_grades.to_parquet(_df_grades_path(), index=False)
    df_env.to_parquet(OUTPUT_PARQUET, index=False)

    for test_size in sorted(df_env["test_size"].dropna().astype(int).unique()):
        df_env[df_env["test_size"] == test_size].to_parquet(
            _df_env_by_test_size_path(test_size),
            index=False,
        )


# =========================================================
# MAIN
# =========================================================

def create_dataframe() -> None:
    input_path = Path(INPUT_PARQUET)

    if not input_path.exists():
        raise FileNotFoundError(
            f"INPUT_PARQUET nicht gefunden: {input_path.resolve()}"
        )

    if not _tests_root_dir().exists():
        raise FileNotFoundError(
            f"Test-Environment nicht gefunden: {_tests_root_dir().resolve()}"
        )

    df_test_questions = build_test_questions_df()
    df_test_students = build_test_students_df()
    df_answers_master = build_answers_master_df(input_path)
    df_grades = build_grades_df(input_path)

    df_env = build_env_dataframe(
        df_test_questions=df_test_questions,
        df_test_students=df_test_students,
        df_answers_master=df_answers_master,
        df_grades=df_grades,
    )

    save_dataframes(
        df_test_questions=df_test_questions,
        df_test_students=df_test_students,
        df_answers_master=df_answers_master,
        df_grades=df_grades,
        df_env=df_env,
    )

    print("DataFrame-Erstellung abgeschlossen.")
    print(f"df_test_questions: {df_test_questions.shape}")
    print(f"df_test_students:  {df_test_students.shape}")
    print(f"df_answers_master: {df_answers_master.shape}")
    print(f"df_grades:         {df_grades.shape}")
    print(f"df_env:            {df_env.shape}")
    print(f"Gespeichert unter: {_dataframe_dir().resolve()}")
    print(f"df_env gespeichert unter: {Path(OUTPUT_PARQUET).resolve()}")


if __name__ == "__main__":
    create_dataframe()