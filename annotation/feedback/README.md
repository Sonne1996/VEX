# Feedback Annotation Dataset

This directory documents the sampling procedure used to create the feedback-analysis subset for manual review of model-generated feedback.

The current documentation covers the sampling design only. A fuller description of the feedback annotation protocol, rating dimensions, and adjudication procedure will be added in a later revision.

## Source Data

The feedback sample was derived from `merged_feedback_long.parquet`, a long-format table containing model-level feedback generations aligned to base answers.

Only rows with both a non-null `model_grade` and a non-null `model_feedback` were considered eligible for sampling.

## Sampling Procedure

The sampling process was designed to create a balanced question-level subset while avoiding repeated use of the same student answer across models within each question.

First, the merged long-format table was reduced back to unique base answers by removing the model-specific feedback fields and deduplicating on `grading_id`.

Second, the script verified that the merged file covered exactly 21 questions. This served as a dataset-integrity check for the feedback study snapshot.

Third, for each question, 14 base answers were sampled uniformly at random using a fixed seed (`42`). This produced a balanced pool of candidate answers with the same number of sampled base answers per question.

Fourth, model-specific rows were assigned from this sampled base pool using a second fixed seed (`1337`). For each question and each active model, the script selected 2 answer rows. Assignment was disjoint at the answer level within a question, meaning that the same sampled answer was not reused across different models for that question.

Finally, the selected rows were sorted by `question_id`, `model_name`, and `grading_id`, augmented with a within-group index (`question_model_row_index`), and written to `sampled_feedback.parquet`.

## Resulting Structure

Under this design, each question contributes:

- 14 sampled base answers
- 2 assigned feedback rows per model
- a fixed and reproducible allocation determined by the two random seeds above

If the number of active models is smaller than the size of the base sample permits, some sampled base answers remain unused. This is expected and was part of the design, allowing disjoint per-model selection without forcing overlap.

## Release Note

This README currently documents the data selection step only. The downstream feedback-evaluation methodology will be added once the annotation and analysis protocol is finalized for release.
