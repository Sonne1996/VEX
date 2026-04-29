#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

# =========================================================
# INPUT
# =========================================================

# Input parquet containing at least:
# - question_id (question identifier)
# - member_id (student identifier)
# - answer_id (optional, but recommended)
# - one human score column
# - one or more model prediction columns
# - effectively: Test(Q | (S, A, ŷ, y))
INPUT_PARQUET = "../dataset/vex_metric_dataset/merged_model_predictions.parquet"

# =========================================================
# TEST ENVIRONMENT STORAGE
# =========================================================

# Base folder where the whole virtual test environment is stored.
#
# Structure example:
#
# vex_test_env/
#   1_test_env_metadata/
#       test_env_metadata.txt
#   2_metrics/
#       virtual_test_report.txt
#   3_tests/
#       test_1/
#           questions/
#               questions_10.txt
#               questions_15.txt
#           students/
#               students_10.txt
#               students_15.txt
#           metadata/
#               1_metadata_questions_10.txt
#               1_metadata_questions_15.txt
#           linear/
#               10/
#                   linear_df_human.txt
#                   linear_df_grade_google_gemini_2.5_pro.txt
#                   ...
#               15/
#                   linear_df_human.txt
#                   linear_df_grade_google_gemini_2.5_pro.txt
#                   ...
#           bologna/
#               10/
#                   bologna_df_human.txt
#                   bologna_df_grade_google_gemini_2.5_pro.txt
#                   ...
#               15/
#                   bologna_df_human.txt
#                   bologna_df_grade_google_gemini_2.5_pro.txt
#                   ...
#           metrics/
#               test_1_metrics.txt
#       test_2/
#           ...
#   4_dataframe
#
TEST_ENV_FOLDER = "vex_test_env"

# Top-level subfolders inside the test environment
TEST_ENV_METADATA_FOLDER = "1_test_env_metadata"
TEST_ENV_METRICS_FOLDER = "2_metrics"
TESTS_ROOT_FOLDER = "3_tests"
TESTS_DATAFRAME = "4_dataframe"

# Folder name pattern for one sampled test run
TEST_RUN_FOLDER = "test_{test_number}"

# Subfolders inside one test run
QUESTIONS_FOLDER = "questions"
STUDENTS_FOLDER = "students"
TEST_METADATA_FOLDER = "metadata"
TEST_METRICS_FOLDER = "metrics"

# Scale-specific dataframe folders inside one test run
LINEAR_FOLDER = "linear"
BOLOGNA_FOLDER = "bologna"

# Subfolder name inside linear/ and bologna/ based on test size
TEST_SIZE_FOLDER = "{questions_number}"

# Files inside one test run
QUESTION_FILE = "questions_{questions_number}.txt"
STUDENT_FILE = "students_{questions_number}.txt"
TEST_METADATA_FILE = "{test_number}_metadata_questions_{questions_number}.txt"
TEST_METRICS_FILE = "test_{test_number}_metrics.txt"

# Files at the overall test environment level
TEST_ENV_METADATA_FILE = "test_env_metadata.txt"
OUTPUT_REPORT_FILE = "virtual_test_report.txt"

# Stored dataframe file names
LINEAR_DATAFRAME_FILE = "linear_df_{h_or_m}.txt"
BOLOGNA_DATAFRAME_FILE = "bologna_df_{h_or_m}.txt"

OUTPUT_PARQUET = Path(TEST_ENV_FOLDER) / TESTS_DATAFRAME / "dataframe_env.parquet"


# =========================================================
# SAMPLING
# =========================================================

# Number of virtual tests to create
N_RUNS = 500

# Number of questions per sampled test
TEST_SIZES = [10, 15]

# Random seed for reproducibility
RANDOM_SEED = 4242


# =========================================================
# MODELS
# =========================================================

# Names of the model prediction columns to evaluate
MODEL_COLUMNS = [
    # joint LLM
    "new_grade_deepseek/deepseek-v3.2-thinking",
    "new_grade_deepseek/deepseek-v3.2",
    "new_grade_google/gemini-2.5-pro",
    "new_grade_anthropic/claude-sonnet-4.6",
    "new_grade_openai/gpt-5.4",
    "new_grade_llama32_3b_base",
    "new_grade_gemma_e4_base",
    "new_grade_llama32_3b_ft",
    "new_grade_gemma_e4_ft",

    # Transformer
    "grade_bert_base",
    "grade_bert_ft",
    "grade_mdeberta_base",
    "grade_mdeberta_ft",

    # Prior / Tempalte
    "grade_prior_global",
    "grade_prior_template_overlap",

    # TFIDF
    "pred_tfidf_v5_answer_char_3_5",
    "pred_tfidf_v1_answer_word_unigram",
    "pred_tfidf_v4_question_and_answer_separate",
]


# =========================================================
# ABSOLUTE / LINEAR SCALE
# =========================================================

LINEAR_GRADE_SCALE_NAME = "Absolute"

# Swiss-style grade scale boundaries
LINEAR_MIN_GRADE = 1.0
LINEAR_MAX_GRADE = 6.0

# Grade rounding step, e.g. 0.5 -> half grades
LINEAR_ROUNDING_STEP = 0.5

# Pass threshold on normalized score scale
LINEAR_PASS_THRESHOLD_NORM = 0.6

# Absolute linear mapping:
# absolute_grade = 1.0 + 5.0 * normalized_score
LINEAR_PASS_THRESHOLD_ABS = LINEAR_MIN_GRADE + (
    (LINEAR_MAX_GRADE - LINEAR_MIN_GRADE) * LINEAR_PASS_THRESHOLD_NORM
)


# =========================================================
# BOLOGNA SCALE
# =========================================================

BOLOGNA_GRADE_SCALE_NAME = "Bologna"

# Minimum achievable points
BOLOGNA_MIN_POINTS = 0

# Passing threshold
BOLOGNA_PASS_THRESHOLD_NORM = 0.6

# Number of passing Bologna categories
BOLOGNA_NUM_PASSING_CATEGORIES = 5

# Relative distribution among passing students
BOLOGNA_PASSING_DISTRIBUTION = [0.10, 0.25, 0.30, 0.25, 0.10]

# Labels for passed students
BOLOGNA_PASSING_LABELS = ["A", "B", "C", "D", "E"]

# Label for failed students
BOLOGNA_FAIL_LABEL = "F"

# Ordered labels for evaluation
BOLOGNA_ORDERED_LABELS = ["F", "E", "D", "C", "B", "A"]