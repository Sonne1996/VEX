# Audit Dataset

This folder contains the audit-oriented VEX dataset.

## File

`audit_dataset.parquet`

This dataset preserves the human annotation and audit columns that are largely removed from the cleaner benchmark-facing releases. It is intended for analyses such as:

- inter-annotator disagreement,
- manual audit decisions,
- gold-label resolution studies,
- LLM-detection audit inspection.

In project terms, this dataset is closest to the annotation workflow described in [annotation/dataset/README.md](../../../annotation/dataset/README.md#L1).

## Structure

The file contains the standard answer-level identifiers and content fields:

- `member_id`
- `subject_id`
- `answer_id`
- `question_id`
- `grading_id`
- `name`
- `question`
- `bloom`
- `topic`
- `answer`
- `grade`
- `label_type`
- `gold_is_llm`
- `split`

and, importantly, the preserved audit columns:

- `rating`
- `human_grade 1`
- `is_llm 1`
- `grader_name 1`
- `human_grade 2`
- `is_llm 2`
- `grader_name 2`
- `gold_label_after_human_audit`
- `consensus_status_audit`
- `human_audit_comment`

## Intended Use

Use this dataset when the goal is to study how gold labels were produced or corrected, rather than to train or benchmark a final grading model. Compared with later stable releases, this file keeps the annotation trail visible.

## Notes

- Column names follow an earlier dataset stage and still use fields such as `name`, `bloom`, and `topic`.
- This dataset is not the main VEX metric input. For benchmark evaluation, use [../vex_metric_dataset/README.md](../vex_metric_dataset/README.md#L1).

## Provenance Note

This dataset is most closely connected to the gold-label and audit workflow documented in [../../../annotation/dataset/README.md](../../../annotation/dataset/README.md#L1). In particular, it preserves the kind of annotation and audit columns that are later removed from cleaner modeling and benchmark releases.
