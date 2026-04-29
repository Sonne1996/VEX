# Feedback Dataset

This folder contains the feedback-oriented VEX dataset.

## File

`merged_feedback_long.parquet`

This dataset is designed for experiments where the primary object of study is not only the predicted grade, but also the written feedback produced by grading models.

The file is stored in **long format**: one row corresponds to one student answer paired with one model output.

## Structure

The dataset includes the base answer context:

- `member_id`
- `subject_id`
- `answer_id`
- `question_id`
- `grading_id`
- `student_name`
- `question`
- `bloom_level`
- `question_topic`
- `answer`
- `grade`
- `label_type`
- `gold_is_llm`
- `split`

and the feedback-model columns:

- `model_name`
- `model_grade`
- `model_feedback`

## Intended Use

This dataset is appropriate for:

- feedback quality evaluation,
- comparing feedback-generating models,
- studying the relation between predicted grade and written explanation,
- building feedback-specific metrics or human review studies.

Because the data is long-format, the same answer may appear multiple times with different `model_name` values.

## Notes

- The presence of `model_feedback` distinguishes this dataset from the benchmark-oriented VEX metric dataset.
- From the project context, this dataset is best interpreted as a paper-specific artifact for feedback studies rather than the main benchmark input.
