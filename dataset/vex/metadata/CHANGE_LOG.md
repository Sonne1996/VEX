# Dataset Change Log

This changelog summarizes the main dataset transitions in the VEX release. It focuses on the stable milestones rather than every intermediate internal snapshot.

## v0_1 -> v0_2

`v0_1` is the first stable export from the institutional SQLite source. It joins members, answers, model gradings, and optional feedback into one parquet artifact.

`v0_2` converts this raw export into a cleaner modeling dataset:

- extracts plain-text `answer` content from the original JSON structure,
- extracts plain-text `question` content from the original JSON structure,
- adds question metadata such as `bloom` and `topic`,
- removes unused raw columns and intermediate fields,
- removes known problematic questions, members, and answers,
- maps audited human labels to the normalized numeric grade scale,
- creates the canonical `grade`, `label_type`, and `split` columns.

## v0_2 -> v0_3

`v0_3` extends the cleaned dataset with silver-label information used for model training.

Main changes:

- selects a teacher model on gold data and generates silver labels for the training split,
- merges the generated silver-label predictions back into the master dataset via `grading_id`,
- fills missing training grades from the silver-label source,
- clamps grades to the valid range,
- removes transient generation metadata and annotation-heavy audit columns,
- removes remaining non-core columns such as `rating`,
- renames fields to the later stable naming convention, including:
  - `name` -> `student_name`
  - `bloom` -> `bloom_level`
  - `topic` -> `question_topic`

The result is `v0.3_stable`, which is the first stable dataset version aligned with the downstream modeling pipeline.

## v0_3 -> v1_0

`v1_0` is the public release version used by the model scripts in this repository as `v1_0_stable.parquet`.

From the available release context, this step represents:

- freezing the stable dataset for public model training and evaluation,
- keeping the canonical columns expected by the released model code,
- removing the `gold_is_llm` column from the public release snapshot,
- preserving the `split` field so the release can also be exported as `train.parquet` and `test.parquet`,
- packaging the dataset under the `v1_0_release/` directory as the benchmark release artifact.

Note: the current repository references `v1_0_stable.parquet` throughout the modeling code, but the public `v1_0.py` file still looks like a carry-over from the `v0.3` stage rather than a polished final conversion script. So `v1_0` can be documented confidently as the released frozen dataset, while the exact final packaging step is only partially reflected in the public scripts here.
