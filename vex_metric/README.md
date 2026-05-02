# VEX: Virtual Exam Evaluation Pipeline

This directory contains the VEX evaluation pipeline for automatic short-answer
grading (ASAG). It evaluates grading models at two levels:

1. item level, over individual student answers;
2. exam level, after aggregating predictions over sampled virtual exams.

The exam-level layer is designed to reflect educational outcomes more closely
than isolated answer-level scoring.

## Pipeline Overview

From the repository root, run:

```bash
python vex_metric/run_vex.py
```

This executes:

```text
create_vex_test_env.py
create_dataframe.py
evaluate_dataframe.py
```

The generated environment is written to:

```text
vex_metric/vex_test_env/
```

This folder is a reproducible output and should not be committed.

## Input Data

The pipeline expects the merged prediction parquet configured in
`vex_config.py`:

```text
dataset/additional/vex_metric_dataset/merged_model_predictions.parquet
```

Required core columns:

| Column | Meaning |
|---|---|
| `question_id` | Question identifier |
| `member_id` | Student identifier |
| `answer_id` | Answer identifier |
| `question` | Question text |
| `answer` | Student answer text |
| `bloom_level` | Bloom taxonomy level |
| `question_topic` | Question topic |
| `grade` | Human reference grade |
| model prediction columns | One column per model |

Human and model grades are expected on the normalized scale:

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

## Current Configuration

The current configuration in [vex_config.py](vex_config.py) uses:

```python
N_RUNS = 500
TEST_SIZES = [5, 10, 15, 20]
RANDOM_SEED = 4242
```

The configured model columns are:

```text
new_grade_deepseek/deepseek-v3.2-thinking
new_grade_deepseek/deepseek-v3.2
new_grade_google/gemini-2.5-pro
new_grade_anthropic/claude-sonnet-4.6
new_grade_openai/gpt-5.4
new_grade_llama32_3b_base
new_grade_gemma_e4_base
new_grade_llama32_3b_ft
new_grade_gemma_e4_ft
grade_bert_ft
grade_mdeberta_ft
grade_prior_global
grade_prior_template_overlap
pred_tfidf_v5_answer_char_3_5
pred_tfidf_v1_answer_word_unigram
pred_tfidf_v4_question_and_answer_separate
```

## Generated Structure

After a successful run, the generated environment has the following structure:

```text
vex_test_env/
├── 1_test_env_metadata/
│   ├── test_env_metadata.txt
│   └── duplicate_student_question_combinations.tsv
├── 2_metrics/
│   ├── virtual_test_report.txt
│   ├── exam_level_precomputed_metrics.parquet
│   ├── exam_level_summary_by_test_size.parquet
│   └── exam_level_summary_all.parquet
├── 3_tests/
│   └── test_*/
│       ├── questions/
│       ├── students/
│       ├── metadata/
│       ├── linear/
│       ├── bologna/
│       └── metrics/
└── 4_dataframe/
    ├── dataframe_env.parquet
    ├── df_env_q5.parquet
    ├── df_env_q10.parquet
    ├── df_env_q15.parquet
    ├── df_env_q20.parquet
    └── dataframe_env_exam_metrics_wide.parquet
```

## Step 1: Create Virtual Test Environment

Script:

```bash
python vex_metric/create_vex_test_env.py
```

For each run and test size, this script samples question IDs without
replacement. A student is eligible for a sampled virtual exam only if the
student has a valid `answer_id` for every selected question.

The script writes:

- sampled question IDs,
- eligible student IDs,
- metadata with answer counts, empty-answer counts, topic distribution, and Bloom-level distribution,
- global environment metadata for reproducibility.

If the existing environment metadata matches the current configuration, the
environment is reused. If the metadata differs, the environment is regenerated.

## Step 2: Create Evaluation Dataframe

Script:

```bash
python vex_metric/create_dataframe.py
```

This script joins the sampled exam structure with answer text, human grades, and
model predictions.

The main output is:

```text
vex_metric/vex_test_env/4_dataframe/dataframe_env.parquet
```

It contains one row per:

```text
virtual test x test size x student x question
```

Important columns:

| Column | Meaning |
|---|---|
| `test_id` | Virtual test ID, e.g. `test_1` |
| `test_size` | Number of questions in the virtual exam |
| `question_order` | Position of the question in the sampled test |
| `question_id` | Question identifier |
| `member_id` | Student identifier |
| `answer_id` | Answer identifier |
| `question` | Question text |
| `answer` | Student answer text |
| `bloom_level` | Bloom taxonomy level |
| `question_topic` | Question topic |
| `human_grade` | Human reference grade |
| model columns | Model predictions |

## Step 3: Evaluate Models

Script:

```bash
python vex_metric/evaluate_dataframe.py
```

This script computes:

- item-level metrics from the original held-out input parquet,
- exam-level metrics from `dataframe_env.parquet`,
- per-exam metric rows,
- aggregated summaries by model and test size,
- sanity exports of per-student linear and Bologna grade assignments.

The main report is:

```text
vex_metric/vex_test_env/2_metrics/virtual_test_report.txt
```

## Item-Level Evaluation

Item-level metrics are computed directly on:

```text
dataset/additional/vex_metric_dataset/merged_model_predictions.parquet
```

They are not computed on `dataframe_env.parquet`, because the virtual-exam
dataframe reuses the same answers across many sampled exams.

Metrics:

| Metric | Meaning |
|---|---|
| `item_mae` | Mean absolute error |
| `item_mse` | Mean squared error |
| `item_rmse` | Root mean squared error |
| `item_qwk` | Quadratic weighted kappa |

## Exam-Level Evaluation

For each virtual exam, student totals are computed:

```text
gold_total = sum(human grades over sampled questions)
pred_total = sum(model grades over sampled questions)
```

Normalized totals are:

```text
gold_norm = gold_total / test_size
pred_norm = pred_total / test_size
```

A student is retained for a specific model and exam only if they have:

- exactly `test_size` rows,
- exactly `test_size` unique questions,
- exactly `test_size` valid human grades,
- exactly `test_size` valid model predictions.

Exam-level metrics are computed per virtual exam first and then averaged over
the sampled exam runs.

## Exam-Level Metrics

### Kendall Rank Agreement

```text
EL-tau_b
```

Kendall's tau-b between human and model student totals.

### Absolute Linear Scale

Normalized scores are mapped to Swiss-style grades:

```text
grade = 1.0 + 5.0 * normalized_score
```

Current defaults:

```python
LINEAR_MIN_GRADE = 1.0
LINEAR_MAX_GRADE = 6.0
LINEAR_ROUNDING_STEP = 0.5
LINEAR_PASS_THRESHOLD_NORM = 0.6
```

Metrics:

| Metric | Meaning |
|---|---|
| `EL-Acc Linear Abs` | Exact agreement of rounded linear grades |
| `EL-QWK Linear Abs` | QWK over rounded linear grades |
| `EL-PassAcc Linear Abs` | Pass/fail accuracy |
| `EL-PassQWK Linear Abs` | QWK over pass/fail decisions |

### Mean-Centered Linear Scale

The mean-centered scale is a sensitivity analysis. It lets each distribution
open its own grade scale around its own empirical mean.

For totals `T`, mean `mean`, and maximum total `max_total`:

```text
if T <= mean:
    z = 0.5 * T / mean

if T > mean:
    z = 0.5 + 0.5 * (T - mean) / (max_total - mean)
```

The resulting `z` is clipped to `[0, 1]`, converted to the `1..6` grade range,
and rounded. Human totals and each model's predicted totals are transformed
separately, so this variant is cohort-dependent and model-dependent. It is not
the main absolute grading scheme.

Metrics:

| Metric | Meaning |
|---|---|
| `EL-Acc Linear Mean` | Exact agreement of mean-centered rounded grades |
| `EL-QWK Linear Mean` | QWK over mean-centered rounded grades |
| `EL-PassAcc Linear Mean` | Pass/fail accuracy |
| `EL-PassQWK Linear Mean` | QWK over pass/fail decisions |

### Bologna Scale

Students below the absolute pass threshold receive `F`. Passing students are
ranked by total points and assigned to:

```text
A, B, C, D, E
```

using the configured passing distribution:

```python
BOLOGNA_PASSING_DISTRIBUTION = [0.10, 0.25, 0.30, 0.25, 0.10]
BOLOGNA_PASSING_LABELS = ["A", "B", "C", "D", "E"]
BOLOGNA_FAIL_LABEL = "F"
BOLOGNA_ORDERED_LABELS = ["F", "E", "D", "C", "B", "A"]
```

Tie handling is strict: students with identical point totals receive the same
Bologna label. If a tied group crosses a boundary, the whole group receives the
better category touched by the first rank of the tied group.

Metrics:

| Metric | Meaning |
|---|---|
| `EL-Acc Bologna` | Exact Bologna label agreement |
| `EL-QWK Bologna` | QWK over ordered Bologna labels |

## Output Metric Tables

The evaluator writes reusable metric tables:

```text
vex_metric/vex_test_env/2_metrics/exam_level_precomputed_metrics.parquet
vex_metric/vex_test_env/2_metrics/exam_level_summary_by_test_size.parquet
vex_metric/vex_test_env/2_metrics/exam_level_summary_all.parquet
vex_metric/vex_test_env/4_dataframe/dataframe_env_exam_metrics_wide.parquet
```

`exam_level_precomputed_metrics.parquet` is long format with one row per:

```text
model x test_id x test_size
```

The summary files average per-exam metrics over virtual runs. Plot scripts in
`results/` should normally consume these precomputed tables.

## Report Format

The main report contains one section per test size and one section over all test
sizes:

```text
VEX EVALUATION REPORT - TEST SIZE 5
VEX EVALUATION REPORT - TEST SIZE 10
VEX EVALUATION REPORT - TEST SIZE 15
VEX EVALUATION REPORT - TEST SIZE 20
VEX EVALUATION REPORT - ALL TEST SIZES
```

For each model, the report includes item-level metrics, student completeness
counts, and all exam-level metrics. Exam-level values are shown as:

```text
mean +/- standard deviation
```

where the mean and standard deviation are computed over sampled virtual exams.

## Validation Checks

The pipeline rejects:

- missing required columns,
- missing configured model prediction columns,
- duplicate `answer_id` values in the answer master table,
- duplicate student-question pairs in the item-level source,
- duplicate `(test_id, test_size, member_id, question_id)` pairs in the virtual exam dataframe.

This prevents inflated exam totals and invalid exam-level metrics.

## Missing Predictions

At item level, missing model predictions are counted and excluded from metric
computation.

At exam level, a student is excluded for a specific model and virtual exam if
any required prediction is missing for the sampled questions.

The report includes:

```text
Exam students raw total
Exam students valid total
Students dropped incomplete
Students missing human
Students missing predictions
```

## Relation to `results/`

The scripts in `results/` consume the VEX metric outputs for paper figures,
dataset statistics, confusion matrices, and statistical significance tests.

Important result scripts:

- [../results/create_plot_2_qwk_vs_tau.py](../results/create_plot_2_qwk_vs_tau.py)
- [../results/create_plot_3_tau_vs_acc.py](../results/create_plot_3_tau_vs_acc.py)
- [../results/create_plot_4_a_b_qwk_linear_qwk_distro.py](../results/create_plot_4_a_b_qwk_linear_qwk_distro.py)
- [../results/calc_statistical_significance.py](../results/calc_statistical_significance.py)

## Notes

- `test_10` is a virtual run ID, not q10.
- q10 means `test_size = 10`.
- Empty answer text is allowed when identifiers are valid.
- Rows without valid `question_id`, `member_id`, or `answer_id` are removed during environment construction.

## Dependencies

The pipeline requires Python 3.10+ and:

```text
numpy
pandas
pyarrow
scipy
scikit-learn
tqdm
```

Minimal install:

```bash
pip install numpy pandas pyarrow scipy scikit-learn tqdm
```
