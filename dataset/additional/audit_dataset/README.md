# VEX Metric Dataset

This folder contains the dataset used by the VEX evaluation pipeline.

## File

`merged_model_predictions.parquet`

This is the main evaluation file. It contains student answers, human reference grades, and model predictions. The VEX pipeline uses it to compute both item-level ASAG metrics and virtual-exam-level metrics.

## Columns

| Column | Description |
|---|---|
| `question_id` | Unique ID of the question. Used to sample virtual exams. |
| `member_id` | Anonymous student/member ID. Used to aggregate answers per student. |
| `answer_id` | Unique ID of the student answer. |
| `question` | The question text shown to the student. |
| `answer` | The student's answer text. |
| `bloom_level` | Bloom taxonomy level of the question, if available. |
| `question_topic` | Topic/category of the question. |
| `grade` | Human reference grade on the normalized scale `0.0, 0.25, 0.5, 0.75, 1.0`. |
| `new_grade_*` | Predictions from LLM-based grading models. |
| `grade_bert_*` | Predictions from BERT-based grading models. |
| `grade_mdeberta_*` | Predictions from mDeBERTa-based grading models. |
| `grade_prior_*` | Prior/template baseline predictions. |
| `pred_tfidf_*` | TF-IDF baseline predictions. |

## Usage

The VEX pipeline expects this file at:

```text
dataset/vex_metric_dataset/merged_model_predictions.parquet