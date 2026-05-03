#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

DEFAULT_RELEASE_DIR = PROJECT_ROOT / "dataset" / "vex" / "v1_0_release"
DEFAULT_INPUT = DEFAULT_RELEASE_DIR / "v1_0_stable.parquet"
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent / "dataset_metrics"

STUDENT_COL = "member_id"
QUESTION_COL = "question_id"
QUESTION_TEXT_COL = "question"
ANSWER_COL = "answer"
LABEL_TYPE_COL = "label_type"
GRADE_COL = "grade"

REPORT_TXT = "vex_v1_0_core_statistics.txt"
REPORT_CSV = "vex_v1_0_core_statistics.csv"

EN_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "yourself", "yourselves"
}

DE_STOPWORDS = {
    "aber", "als", "am", "an", "auch", "auf", "aus", "bei", "beim", "bin",
    "bis", "da", "das", "dass", "daten", "de", "dem", "den", "der", "des",
    "die", "dies", "diese", "dieser", "dieses", "du", "ein", "eine", "einem",
    "einen", "einer", "er", "es", "für", "haben", "hat", "ich", "im", "in",
    "ist", "ja", "man", "mit", "nein", "nicht", "oder", "sich", "sie",
    "sind", "und", "von", "was", "weil", "wenn", "wer", "wie", "wird",
    "wo", "zu", "zum", "zur",
}

GERMAN_CHAR_RE = re.compile(r"[äöüÄÖÜß]")
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")


def resolve_input_path(input_path: Path | None) -> Path:
    if input_path is not None:
        path = input_path
    else:
        path = DEFAULT_INPUT

    if path.exists():
        return path

    if input_path is None:
        parquet_files = sorted(DEFAULT_RELEASE_DIR.glob("*.parquet"))
        if len(parquet_files) == 1:
            return parquet_files[0]

        if len(parquet_files) > 1:
            candidates = "\n".join(f"  - {p}" for p in parquet_files)
            raise FileNotFoundError(
                "Could not choose one v1.0 release parquet automatically. "
                "Pass --input explicitly.\n"
                f"Candidates:\n{candidates}"
            )

    raise FileNotFoundError(
        "Input parquet not found.\n"
        f"Expected: {path}\n"
        "Pass --input path/to/release.parquet if the file has another name."
    )


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Input dataframe is missing required columns: "
            + ", ".join(missing)
        )


def whitespace_token_count(value: Any) -> int:
    if pd.isna(value):
        return 0

    text = str(value).strip()
    if not text:
        return 0

    return len(re.findall(r"\S+", text))


def detect_language(value: Any) -> str:
    if pd.isna(value):
        return "unknown"

    text = str(value).strip().lower()
    if not text:
        return "unknown"

    tokens = WORD_RE.findall(text)
    if not tokens:
        return "unknown"

    de_score = sum(token in DE_STOPWORDS for token in tokens)
    en_score = sum(token in EN_STOPWORDS for token in tokens)

    if GERMAN_CHAR_RE.search(text):
        de_score += 2

    if de_score > en_score:
        return "de"

    if en_score > de_score:
        return "en"

    return "unknown"


def classify_response_language(question: Any) -> tuple[str, bool]:
    question_lang = detect_language(question)

    if question_lang == "de":
        return "de", False

    if question_lang == "en":
        return "en", False

    # Unknown questions are counted as English for the main split but reported
    # separately.
    return "en", True


def language_counts(df: pd.DataFrame) -> dict[str, int | float]:
    question_values = (
        df[QUESTION_TEXT_COL]
        if QUESTION_TEXT_COL in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )

    english = 0
    german = 0
    unknown_counted_as_english = 0

    for question in question_values:
        language, was_unknown = classify_response_language(question)

        if language == "de":
            german += 1
        else:
            english += 1

        if was_unknown:
            unknown_counted_as_english += 1

    total = len(df)

    return {
        "english": english,
        "german": german,
        "unknown_counted_as_english": unknown_counted_as_english,
        "english_pct": (english / total * 100.0) if total else 0.0,
        "german_pct": (german / total * 100.0) if total else 0.0,
        "unknown_pct": (
            unknown_counted_as_english / total * 100.0
        ) if total else 0.0,
    }


def normalized_label_type(df: pd.DataFrame) -> pd.Series:
    if LABEL_TYPE_COL in df.columns:
        return df[LABEL_TYPE_COL].astype("string").str.lower().str.strip()

    if GRADE_COL in df.columns:
        return pd.Series(
            ["gold" if pd.notna(value) else "" for value in df[GRADE_COL]],
            index=df.index,
            dtype="string",
        )

    raise ValueError(
        f"Need either '{LABEL_TYPE_COL}' or '{GRADE_COL}' to count label types."
    )


def count_label_type(df: pd.DataFrame, label_type: str) -> int:
    labels = normalized_label_type(df)
    return int((labels == label_type.lower()).sum())


def count_unlabeled_or_other_responses(df: pd.DataFrame) -> int:
    labels = normalized_label_type(df)
    return int((~labels.isin(["gold", "silver"])).sum())


def fmt_value(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)

    if float(value).is_integer():
        return f"{value:.2f}"

    return f"{value:.2f}"


def gold_subset(df: pd.DataFrame) -> pd.DataFrame:
    if LABEL_TYPE_COL in df.columns:
        labels = df[LABEL_TYPE_COL].astype("string").str.lower().str.strip()
        return df[labels == "gold"].copy()

    if GRADE_COL in df.columns:
        return df[df[GRADE_COL].notna()].copy()

    raise ValueError(
        f"Need either '{LABEL_TYPE_COL}' or '{GRADE_COL}' to identify gold rows."
    )


def compute_statistics(
    df: pd.DataFrame,
    student_label: str,
) -> list[tuple[str, int | float]]:
    require_columns(df, [STUDENT_COL, QUESTION_COL, ANSWER_COL])

    responses_per_question = df.groupby(QUESTION_COL, dropna=False).size()
    responses_per_student = df.groupby(STUDENT_COL, dropna=False).size()
    response_lengths = df[ANSWER_COL].map(whitespace_token_count)
    lang = language_counts(df)

    return [
        (student_label, int(df[STUDENT_COL].nunique(dropna=True))),
        ("Total responses", int(len(df))),
        ("Avg. responses per question", float(responses_per_question.mean())),
        ("Avg. responses per student", float(responses_per_student.mean())),
        ("Avg. response length (whitespace tokens)", float(response_lengths.mean())),
        ("Responses to English questions (incl. unknown)", int(lang["english"])),
        ("Responses to German questions", int(lang["german"])),
        ("Unknown-question responses counted as English", int(lang["unknown_counted_as_english"])),
        ("Gold-labeled responses", count_label_type(df, "gold")),
        ("Unique questions", int(df[QUESTION_COL].nunique(dropna=True))),
        ("Silver-labeled responses", count_label_type(df, "silver")),
        ("Median responses per question", float(responses_per_question.median())),
        ("Median responses per student", float(responses_per_student.median())),
        ("Median response length (tokens)", float(response_lengths.median())),
        ("Responses to English questions (%)", float(lang["english_pct"])),
        ("Responses to German questions (%)", float(lang["german_pct"])),
        ("Unknown questions counted as English (%)", float(lang["unknown_pct"])),
        ("Unlabeled/other responses", count_unlabeled_or_other_responses(df)),
    ]


def paired_rows(
    stats: list[tuple[str, int | float]],
) -> list[tuple[str, int | float, str, int | float]]:
    if len(stats) % 2 != 0:
        raise ValueError("Statistics list must contain an even number of rows.")

    midpoint = len(stats) // 2
    left = stats[:midpoint]
    right = stats[midpoint:]

    return [
        (left_stat, left_value, right_stat, right_value)
        for (left_stat, left_value), (right_stat, right_value) in zip(left, right)
    ]


def render_report(
    sections: list[tuple[str, list[tuple[str, int | float]]]],
    input_path: Path,
) -> str:
    section_rows = [(title, paired_rows(stats)) for title, stats in sections]

    stat_width = max(
        len("Statistic"),
        max(len(row[0]) for _, rows in section_rows for row in rows),
        max(len(row[2]) for _, rows in section_rows for row in rows),
    )
    value_width = max(
        len("Value"),
        max(len(fmt_value(row[1])) for _, rows in section_rows for row in rows),
        max(len(fmt_value(row[3])) for _, rows in section_rows for row in rows),
    )

    total_width = (stat_width * 2) + (value_width * 2) + 10

    lines: list[str] = []
    lines.append("Table 3: Core statistics of VEX after curation.")
    lines.append("=" * total_width)
    lines.append(f"Input parquet: {input_path}")
    lines.append(f"Generated on:  {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for section_title, rows in section_rows:
        lines.append(section_title)
        lines.append("-" * len(section_title))
        lines.append(
            f"{'Statistic':<{stat_width}}  "
            f"{'Value':>{value_width}}  "
            f"{'Statistic':<{stat_width}}  "
            f"{'Value':>{value_width}}"
        )
        lines.append("-" * total_width)

        for left_stat, left_value, right_stat, right_value in rows:
            lines.append(
                f"{left_stat:<{stat_width}}  "
                f"{fmt_value(left_value):>{value_width}}  "
                f"{right_stat:<{stat_width}}  "
                f"{fmt_value(right_value):>{value_width}}"
            )

        lines.append("")

    lines.append("=" * total_width)
    lines.append("")
    lines.append("Definitions:")
    lines.append(f"- Students: unique '{STUDENT_COL}'.")
    lines.append(f"- Total responses: number of rows in the release parquet.")
    lines.append(f"- Questions: unique '{QUESTION_COL}'.")
    lines.append(
        f"- Gold subset: rows where '{LABEL_TYPE_COL}' == 'gold'; falls back to "
        f"non-missing '{GRADE_COL}' if '{LABEL_TYPE_COL}' is absent."
    )
    lines.append(
        f"- Response length: whitespace-token count over '{ANSWER_COL}'."
    )
    lines.append(
        f"- Language split: only '{QUESTION_TEXT_COL}' is checked. Every response "
        "inherits the detected question language."
    )
    lines.append(
        "- Unknown question language is counted as English and reported separately."
    )

    return "\n".join(lines) + "\n"


def write_outputs(
    sections: list[tuple[str, list[tuple[str, int | float]]]],
    input_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / REPORT_TXT
    csv_path = output_dir / REPORT_CSV

    txt_path.write_text(
        render_report(sections, input_path=input_path),
        encoding="utf-8",
    )

    csv_rows: list[dict[str, int | float | str]] = []
    for section_title, stats in sections:
        for left_stat, left_value, right_stat, right_value in paired_rows(stats):
            csv_rows.append(
                {
                    "section": section_title,
                    "statistic_left": left_stat,
                    "value_left": left_value,
                    "statistic_right": right_stat,
                    "value_right": right_value,
                }
            )

    pd.DataFrame(
        csv_rows,
        columns=[
            "section",
            "statistic_left",
            "value_left",
            "statistic_right",
            "value_right",
        ],
    ).to_csv(csv_path, index=False)

    return txt_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create VEX v1.0 core dataset statistics as a TXT table."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Path to v1.0 release parquet. Defaults to "
            "dataset/vex/v1_0_release/v1_0_stable.parquet."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the statistics files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = resolve_input_path(args.input)
    df = pd.read_parquet(input_path)

    gold_df = gold_subset(df)
    sections = [
        (
            "Full Dataset (100%)",
            compute_statistics(
                df,
                student_label="Students (full dataset)",
            ),
        ),
        (
            "Gold Subset (10%)",
            compute_statistics(
                gold_df,
                student_label="Students (gold subset)",
            ),
        ),
    ]

    txt_path, csv_path = write_outputs(
        sections=sections,
        input_path=input_path,
        output_dir=args.output_dir,
    )

    print("DATASET METRICS WRITTEN")
    print(f"Input: {input_path.resolve()}")
    print(f"TXT:   {txt_path.resolve()}")
    print(f"CSV:   {csv_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
