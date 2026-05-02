#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from vex_config import (
    INPUT_PARQUET,
    TEST_ENV_FOLDER,
    TEST_ENV_METADATA_FOLDER,
    TEST_ENV_METRICS_FOLDER,
    TESTS_ROOT_FOLDER,
    TEST_RUN_FOLDER,
    QUESTIONS_FOLDER,
    STUDENTS_FOLDER,
    TEST_METADATA_FOLDER,
    TEST_METRICS_FOLDER,
    LINEAR_FOLDER,
    DISTRIBUTION_FOLDER,
    TEST_SIZE_FOLDER,
    QUESTION_FILE,
    STUDENT_FILE,
    TEST_METADATA_FILE,
    TEST_ENV_METADATA_FILE,
    OUTPUT_REPORT_FILE,
    TEST_SIZES,
    N_RUNS,
    RANDOM_SEED,
)

# =========================================================
# REQUIRED / OPTIONAL INPUT COLUMNS
# =========================================================

QUESTION_ID_COL = "question_id"
STUDENT_ID_COL = "member_id"
ANSWER_ID_COL = "answer_id"
ANSWER_COL = "answer"
BLOOM_COL = "bloom"
TOPIC_COL = "topic"


# =========================================================
# META TEXT
# =========================================================

def _build_global_meta_text(
    input_parquet: Path | str,
    n_runs: int,
    test_sizes: list[int],
    random_seed: int,
    question_count: int,
    row_count: int,
    unique_student_count: int,
) -> str:
    return "\n".join(
        [
            f"input_parquet={Path(input_parquet).resolve()}",
            f"n_runs={n_runs}",
            f"test_sizes={','.join(map(str, sorted(test_sizes)))}",
            f"random_seed={random_seed}",
            f"question_count={question_count}",
            f"row_count={row_count}",
            f"unique_student_count={unique_student_count}",
        ]
    )


def _read_existing_meta(meta_file: Path) -> str | None:
    if not meta_file.exists():
        return None
    return meta_file.read_text(encoding="utf-8").strip()


# =========================================================
# PATH HELPERS
# =========================================================

def _test_env_root() -> Path:
    return Path(TEST_ENV_FOLDER)


def _test_env_metadata_dir() -> Path:
    return _test_env_root() / TEST_ENV_METADATA_FOLDER


def _test_env_metrics_dir() -> Path:
    return _test_env_root() / TEST_ENV_METRICS_FOLDER


def _tests_root_dir() -> Path:
    return _test_env_root() / TESTS_ROOT_FOLDER


def _run_dir(run_idx: int) -> Path:
    return _tests_root_dir() / TEST_RUN_FOLDER.format(test_number=run_idx)


def _questions_dir(run_idx: int) -> Path:
    return _run_dir(run_idx) / QUESTIONS_FOLDER


def _students_dir(run_idx: int) -> Path:
    return _run_dir(run_idx) / STUDENTS_FOLDER


def _metadata_dir(run_idx: int) -> Path:
    return _run_dir(run_idx) / TEST_METADATA_FOLDER


def _metrics_dir(run_idx: int) -> Path:
    return _run_dir(run_idx) / TEST_METRICS_FOLDER


def _linear_size_dir(run_idx: int, test_size: int) -> Path:
    return (
        _run_dir(run_idx)
        / LINEAR_FOLDER
        / TEST_SIZE_FOLDER.format(questions_number=test_size)
    )


def _distribution_size_dir(run_idx: int, test_size: int) -> Path:
    return (
        _run_dir(run_idx)
        / DISTRIBUTION_FOLDER
        / TEST_SIZE_FOLDER.format(questions_number=test_size)
    )


def _question_file(run_idx: int, test_size: int) -> Path:
    return _questions_dir(run_idx) / QUESTION_FILE.format(
        questions_number=test_size
    )


def _student_file(run_idx: int, test_size: int) -> Path:
    return _students_dir(run_idx) / STUDENT_FILE.format(
        questions_number=test_size
    )


def _test_metadata_file(run_idx: int, test_size: int) -> Path:
    return _metadata_dir(run_idx) / TEST_METADATA_FILE.format(
        test_number=run_idx,
        questions_number=test_size,
    )


def _test_env_metadata_file() -> Path:
    return _test_env_metadata_dir() / TEST_ENV_METADATA_FILE


def _test_env_report_file() -> Path:
    return _test_env_metrics_dir() / OUTPUT_REPORT_FILE


def _duplicate_report_file() -> Path:
    return _test_env_metadata_dir() / "duplicate_student_question_combinations.tsv"


# =========================================================
# VALIDATION / DATA PREP
# =========================================================

def _validate_input(df: pd.DataFrame) -> None:
    """
    Required columns:

    question_id:
        Needed to sample questions.

    member_id:
        Needed to determine which student answered which question.

    answer_id:
        Needed to distinguish an actually existing answer record from a missing
        student-question answer. Empty answer text is allowed, but missing answer_id
        means the student-question pair is not valid for the virtual exam.
    """
    required = [QUESTION_ID_COL, STUDENT_ID_COL, ANSWER_ID_COL]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Pflichtspalten fehlen im Input-Parquet: {missing}"
        )


def _normalize_string_series(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "").astype(str).str.strip()


def _prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardisiert Schlüsselfelder und entfernt nur Zeilen, die keine gültige
    student-question-answer-Zuordnung darstellen.

    Wichtig:
    - Leere Antworttexte werden NICHT entfernt.
    - Eine leere Antwort mit vorhandener answer_id bleibt gültig.
    - Nur fehlende/leere question_id, member_id oder answer_id werden entfernt.
    """
    base = df.copy()

    original_row_count = len(base)

    base[QUESTION_ID_COL] = _normalize_string_series(base[QUESTION_ID_COL])
    base[STUDENT_ID_COL] = _normalize_string_series(base[STUDENT_ID_COL])
    base[ANSWER_ID_COL] = _normalize_string_series(base[ANSWER_ID_COL])

    missing_question_id_mask = base[QUESTION_ID_COL] == ""
    missing_student_id_mask = base[STUDENT_ID_COL] == ""
    missing_answer_id_mask = base[ANSWER_ID_COL] == ""

    number_of_missing_question_ids = int(missing_question_id_mask.sum())
    number_of_missing_student_ids = int(missing_student_id_mask.sum())
    number_of_missing_answer_ids = int(missing_answer_id_mask.sum())

    number_of_empty_answers = 0
    if ANSWER_COL in base.columns:
        answer_series = base[ANSWER_COL]
        empty_answer_mask = answer_series.isna() | (
            answer_series.astype(str).str.strip() == ""
        )
        number_of_empty_answers = int(empty_answer_mask.sum())

    valid_mask = (
        ~missing_question_id_mask
        & ~missing_student_id_mask
        & ~missing_answer_id_mask
    )

    base = base[valid_mask].copy()

    print(f"Input-Zeilen gesamt: {original_row_count}")
    print(f"Entfernte Zeilen ohne question_id: {number_of_missing_question_ids}")
    print(f"Entfernte Zeilen ohne member_id: {number_of_missing_student_ids}")
    print(f"Entfernte Zeilen ohne answer_id: {number_of_missing_answer_ids}")
    print(f"Leere Antworttexte, aber NICHT entfernt: {number_of_empty_answers}")
    print(f"Gültige Zeilen nach Key-Filterung: {len(base)}")

    return base


def _get_question_ids(df: pd.DataFrame) -> list[str]:
    question_ids = (
        df[QUESTION_ID_COL]
        .dropna()
        .pipe(_normalize_string_series)
    )

    question_ids = question_ids[question_ids != ""].drop_duplicates().tolist()
    question_ids = sorted(question_ids)

    if not question_ids:
        raise ValueError("Keine gültigen question_id im Input-Parquet gefunden.")

    for test_size in TEST_SIZES:
        if len(question_ids) < test_size:
            raise ValueError(
                f"Zu wenige eindeutige Fragen für TEST_SIZE={test_size}: "
                f"{len(question_ids)} verfügbar."
            )

    return question_ids


def _eligible_students_for_sampled_questions(
    df_base: pd.DataFrame,
    sampled_questions: list[str],
) -> tuple[list[str], pd.DataFrame]:
    """
    Ein Student ist eligible genau dann, wenn er für jede gesampelte Frage
    mindestens eine gültige Zeile mit vorhandener answer_id hat.

    Leere Antworttexte sind erlaubt.

    Returns
    -------
    eligible_students:
        Sortierte Liste gültiger student IDs.

    duplicate_counts:
        DataFrame mit mehrfach vorhandenen (student_id, question_id)-Kombinationen.
        count > 1 bedeutet: derselbe Student hat für dieselbe Frage mehrere
        gültige answer_id-Einträge.
    """
    subset = df_base[df_base[QUESTION_ID_COL].isin(sampled_questions)].copy()

    if subset.empty:
        return [], pd.DataFrame(columns=[STUDENT_ID_COL, QUESTION_ID_COL, "count"])

    duplicate_counts = (
        subset.groupby([STUDENT_ID_COL, QUESTION_ID_COL])
        .size()
        .reset_index(name="count")
    )

    duplicate_counts = duplicate_counts[duplicate_counts["count"] > 1].copy()

    coverage = (
        subset[[STUDENT_ID_COL, QUESTION_ID_COL]]
        .drop_duplicates()
        .groupby(STUDENT_ID_COL)[QUESTION_ID_COL]
        .nunique()
    )

    eligible = coverage[coverage == len(sampled_questions)].index.tolist()
    eligible = sorted(map(str, eligible))

    return eligible, duplicate_counts


def _sample_questions_for_run_and_size(
    question_ids: list[str],
    run_idx: int,
    test_size: int,
) -> list[str]:
    """
    Deterministisches Sampling pro (run_idx, test_size).

    Wichtig:
    Dadurch ist z.B. TEST_SIZE=10 unabhängig davon, ob TEST_SIZES nur [10]
    oder [5, 10, 15, 20, 21] enthält.

    Ohne diesen Fix verbrauchen kleinere Testgrössen vorher RNG-State und
    verschieben dadurch alle späteren Samples.
    """
    seed_sequence = np.random.SeedSequence(
        [
            int(RANDOM_SEED),
            int(run_idx),
            int(test_size),
        ]
    )

    rng = np.random.default_rng(seed_sequence)

    sampled_questions = rng.choice(
        question_ids,
        size=int(test_size),
        replace=False,
    )

    return sorted(map(str, sampled_questions))


# =========================================================
# TEST METADATA
# =========================================================

def _distribution_lines(series: pd.Series, header: str) -> list[str]:
    series = series.dropna().astype(str).str.strip()
    series = series[series != ""]

    if series.empty:
        return []

    counts = series.value_counts()
    total = int(counts.sum())

    lines = ["", f"{header}:"]
    for label, count in counts.items():
        ratio = count / total if total > 0 else 0.0
        lines.append(f"{label}={count} ({ratio:.4f})")

    return lines


def _build_test_metadata_text(
    test_number: int,
    questions_number: int,
    sampled_questions: list[str],
    eligible_students: list[str],
    df_source: pd.DataFrame,
    df_valid: pd.DataFrame,
) -> str:
    df_subset_all = df_source[
        df_source[QUESTION_ID_COL].astype(str).isin(sampled_questions)
    ].copy()

    df_subset_valid = df_valid[
        df_valid[QUESTION_ID_COL].isin(sampled_questions)
    ].copy()

    df_subset_eligible = df_subset_valid[
        df_subset_valid[STUDENT_ID_COL].isin(eligible_students)
    ].copy()

    total_rows_all = len(df_subset_all)
    total_rows_valid = len(df_subset_valid)
    total_rows_eligible = len(df_subset_eligible)

    unique_students_all = (
        df_subset_all[STUDENT_ID_COL].nunique()
        if STUDENT_ID_COL in df_subset_all.columns else "n/a"
    )

    unique_students_valid = (
        df_subset_valid[STUDENT_ID_COL].nunique()
        if STUDENT_ID_COL in df_subset_valid.columns else "n/a"
    )

    unique_answers_all = (
        df_subset_all[ANSWER_ID_COL].nunique()
        if ANSWER_ID_COL in df_subset_all.columns else total_rows_all
    )

    unique_answers_valid = (
        df_subset_valid[ANSWER_ID_COL].nunique()
        if ANSWER_ID_COL in df_subset_valid.columns else total_rows_valid
    )

    empty_answers_all = "n/a"
    empty_answers_valid = "n/a"
    empty_answers_eligible = "n/a"

    if ANSWER_COL in df_subset_all.columns:
        empty_answers_all = int(
            (
                df_subset_all[ANSWER_COL].isna()
                | (df_subset_all[ANSWER_COL].astype(str).str.strip() == "")
            ).sum()
        )

    if ANSWER_COL in df_subset_valid.columns:
        empty_answers_valid = int(
            (
                df_subset_valid[ANSWER_COL].isna()
                | (df_subset_valid[ANSWER_COL].astype(str).str.strip() == "")
            ).sum()
        )

    if ANSWER_COL in df_subset_eligible.columns:
        empty_answers_eligible = int(
            (
                df_subset_eligible[ANSWER_COL].isna()
                | (df_subset_eligible[ANSWER_COL].astype(str).str.strip() == "")
            ).sum()
        )

    lines = [
        f"test_number={test_number}",
        f"questions_number={questions_number}",
        f"rows_total_all={total_rows_all}",
        f"rows_total_valid={total_rows_valid}",
        f"rows_total_eligible={total_rows_eligible}",
        f"unique_students_all={unique_students_all}",
        f"unique_students_valid={unique_students_valid}",
        f"eligible_students={len(eligible_students)}",
        f"unique_answers_all={unique_answers_all}",
        f"unique_answers_valid={unique_answers_valid}",
        f"empty_answers_all={empty_answers_all}",
        f"empty_answers_valid_not_removed={empty_answers_valid}",
        f"empty_answers_eligible_not_removed={empty_answers_eligible}",
        "question_ids=" + ",".join(sampled_questions),
    ]

    if eligible_students:
        lines.append("student_ids=" + ",".join(eligible_students))

    if BLOOM_COL in df_subset_all.columns:
        lines.extend(
            _distribution_lines(
                df_subset_all[BLOOM_COL],
                "bloom_distribution",
            )
        )

    if TOPIC_COL in df_subset_all.columns:
        lines.extend(
            _distribution_lines(
                df_subset_all[TOPIC_COL],
                "topic_distribution",
            )
        )

    return "\n".join(lines)


# =========================================================
# ENV COMPLETENESS CHECK
# =========================================================

def _environment_is_complete(global_meta: str) -> bool:
    meta_file_env = _test_env_metadata_file()
    existing_meta = _read_existing_meta(meta_file_env)

    if existing_meta != global_meta:
        return False

    if not _test_env_root().exists():
        return False

    if not _test_env_metadata_dir().exists():
        return False

    if not _test_env_metrics_dir().exists():
        return False

    if not _tests_root_dir().exists():
        return False

    for run_idx in range(1, N_RUNS + 1):
        run_dir = _run_dir(run_idx)

        if not run_dir.exists() or not run_dir.is_dir():
            return False

        required_dirs = [
            _questions_dir(run_idx),
            _students_dir(run_idx),
            _metadata_dir(run_idx),
            _metrics_dir(run_idx),
        ]

        if any(not directory.exists() for directory in required_dirs):
            return False

        for test_size in TEST_SIZES:
            if not _question_file(run_idx, test_size).exists():
                return False

            if not _student_file(run_idx, test_size).exists():
                return False

            if not _test_metadata_file(run_idx, test_size).exists():
                return False

            if not _linear_size_dir(run_idx, test_size).exists():
                return False

            if not _distribution_size_dir(run_idx, test_size).exists():
                return False

    return True


# =========================================================
# MAIN
# =========================================================

def create_virtual_test_env() -> None:
    input_path = Path(INPUT_PARQUET)

    if not input_path.exists():
        raise FileNotFoundError(
            f"INPUT_PARQUET nicht gefunden: {input_path.resolve()}"
        )

    df = pd.read_parquet(input_path)
    _validate_input(df)

    df_valid = _prepare_base_dataframe(df)
    question_ids = _get_question_ids(df_valid)

    global_meta = _build_global_meta_text(
        input_parquet=input_path,
        n_runs=N_RUNS,
        test_sizes=TEST_SIZES,
        random_seed=RANDOM_SEED,
        question_count=len(question_ids),
        row_count=len(df_valid),
        unique_student_count=df_valid[STUDENT_ID_COL].nunique(),
    )

    if _environment_is_complete(global_meta):
        print(
            "Virtual test environment existiert bereits und passt zu Seed/Config. "
            "Keine Neuerstellung."
        )
        return

    if _test_env_root().exists():
        shutil.rmtree(_test_env_root())

    _test_env_metadata_dir().mkdir(parents=True, exist_ok=True)
    _test_env_metrics_dir().mkdir(parents=True, exist_ok=True)
    _tests_root_dir().mkdir(parents=True, exist_ok=True)

    all_duplicate_frames: list[pd.DataFrame] = []

    for run_idx in range(1, N_RUNS + 1):
        run_dir = _run_dir(run_idx)
        run_dir.mkdir(parents=True, exist_ok=True)

        _questions_dir(run_idx).mkdir(parents=True, exist_ok=True)
        _students_dir(run_idx).mkdir(parents=True, exist_ok=True)
        _metadata_dir(run_idx).mkdir(parents=True, exist_ok=True)
        _metrics_dir(run_idx).mkdir(parents=True, exist_ok=True)

        for test_size in sorted(TEST_SIZES):
            sampled_questions = _sample_questions_for_run_and_size(
                question_ids=question_ids,
                run_idx=run_idx,
                test_size=int(test_size),
            )

            eligible_students, duplicate_counts = _eligible_students_for_sampled_questions(
                df_base=df_valid,
                sampled_questions=sampled_questions,
            )

            if not duplicate_counts.empty:
                duplicate_counts = duplicate_counts.copy()
                duplicate_counts["test_number"] = run_idx
                duplicate_counts["questions_number"] = test_size
                all_duplicate_frames.append(duplicate_counts)

            _linear_size_dir(run_idx, int(test_size)).mkdir(parents=True, exist_ok=True)
            _distribution_size_dir(run_idx, int(test_size)).mkdir(parents=True, exist_ok=True)

            question_file = _question_file(run_idx, int(test_size))
            question_file.write_text(
                "\n".join(sampled_questions),
                encoding="utf-8",
            )

            student_file = _student_file(run_idx, int(test_size))
            student_file.write_text(
                "\n".join(eligible_students),
                encoding="utf-8",
            )

            metadata_text = _build_test_metadata_text(
                test_number=run_idx,
                questions_number=int(test_size),
                sampled_questions=sampled_questions,
                eligible_students=eligible_students,
                df_source=df,
                df_valid=df_valid,
            )

            metadata_file = _test_metadata_file(run_idx, int(test_size))
            metadata_file.write_text(metadata_text, encoding="utf-8")

    if all_duplicate_frames:
        duplicates_report_df = pd.concat(all_duplicate_frames, ignore_index=True)
    else:
        duplicates_report_df = pd.DataFrame(
            columns=[
                "test_number",
                "questions_number",
                STUDENT_ID_COL,
                QUESTION_ID_COL,
                "count",
            ]
        )

    duplicates_report_df.to_csv(
        _duplicate_report_file(),
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    _test_env_metadata_file().write_text(global_meta, encoding="utf-8")

    print(f"Virtual test environment erfolgreich erstellt unter: {_test_env_root().resolve()}")
    print(f"Env-Metadatei: {_test_env_metadata_file().resolve()}")
    print(f"Duplicate-Report: {_duplicate_report_file().resolve()}")
    print(f"Anzahl Runs: {N_RUNS}")
    print(f"Testgrössen: {TEST_SIZES}")
    print(f"Gültige Fragen gesamt: {len(question_ids)}")
    print(f"Gültige Studenten gesamt: {df_valid[STUDENT_ID_COL].nunique()}")


if __name__ == "__main__":
    create_virtual_test_env()