# Dataset Change Log

This changelog summarizes the stable dataset milestones in the VEX release. It focuses on the public dataset lineage rather than every intermediate internal snapshot.

The main version path is:

```text
v0_1 -> v0_2 -> v0_3 -> v1_0
```

## v0_1 -> v0_2

`v0_1` is the first stable export from the institutional SQLite source. It combines members, answers, model gradings, and optional feedback into a single parquet artifact.

After `v0_1`, the dataset is split at question level into `train` and `test`. This split is preserved through the later release stages and becomes the basis for the silver-labeled training split and the gold-labeled test split.

`v0_2` turns this raw export into the first cleaned modeling dataset.

Main changes:

- extracts plain-text `answer` content from the original structured answer representation,
- extracts plain-text `question` content from the original structured question representation,
- adds question metadata such as `bloom` and `topic`,
- removes unused raw database columns and intermediate processing fields,
- removes known problematic questions, members, and answers,
- maps audited human labels to the normalized numeric grade scale,
- creates the canonical `grade`, `label_type`, and `split` columns,
- carries forward the question-level train/test split created after `v0_1`.

The result is a cleaner answer-level dataset with stable identifiers, text fields, labels, metadata, and split information.

## v0_2 -> v0_3

`v0_3` extends the cleaned dataset with silver labels for scalable model training and aligns the schema with the downstream modeling pipeline.

Main changes:

- selects a teacher model using the human-audited gold subset,
- generates silver labels for the training split,
- merges generated silver-label predictions back into the master dataset via `grading_id`,
- fills missing training grades from the silver-label source,
- clamps grades to the valid normalized grading range,
- removes transient generation metadata and annotation-heavy audit columns,
- removes remaining non-core fields such as `rating`,
- renames fields to the later stable naming convention:
  - `name` -> `student_name`,
  - `bloom` -> `bloom_level`,
  - `topic` -> `question_topic`.

The result is `v0.3_stable`, the first stable version aligned with the released model-training workflow.

After `v0.3_stable`, four additional datasets are derived from this stable artifact:

- `audit_dataset`,
- `feedback_dataset`,
- `teacher_selection_dataset`,
- `vex_metric_dataset`.

These derived datasets support human-audit analysis, feedback analysis, teacher-model selection, and VEX metric evaluation, respectively.

## v0_3 -> v1_0

`v1_0` is the frozen public release used by the model scripts as:

```text
v1_0_stable.parquet
```

This stage packages the stable dataset for public training and evaluation.

Main changes:

- freezes the stable answer-level dataset as the benchmark-facing release artifact,
- keeps the canonical columns expected by the released model code,
- removes the `gold_is_llm` column from the public snapshot,
- preserves the `split` field so the release can be exported as `train.parquet` and `test.parquet`,
- places the release artifact under `dataset/vex/v1_0_release/`,
- includes helper scripts for split-specific exports.

The v1.0 release contains:

- 30,682 total responses,
- 27,460 silver-labeled training responses,
- 3,222 human-audited gold test responses,
- 239 unique questions,
- 173 anonymized students.

## Notes

The fully raw institutional SQLite source is not redistributed. The public lineage therefore starts from the first stable export and documents the reproducible processing stages available in the repository.

The `v1_0.py` helper currently still resembles a carry-over from the `v0.3` stage. The canonical public artifact for downstream use is nevertheless `v1_0_stable.parquet`, and the release folder documents how to export split-specific `train.parquet` and `test.parquet` files from it.
