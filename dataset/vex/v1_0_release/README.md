# VEX v1.0 Release

This folder contains the **public release stage** of the main VEX dataset.

In the dataset lineage, `v1.0` is the frozen benchmark-facing release that downstream model scripts reference as:

```text
v1_0_stable.parquet
```

Conceptually, this stage packages the stable VEX dataset for public training and evaluation use, with only minor release-facing cleanup relative to `v0.3_stable`.

## Role in the Dataset Lineage

The main version path is:

```text
v0_1 -> v0_2 -> v0_3 -> v1_0
```

Within that sequence:

- `v0_1` is the first stable export from the institutional SQLite source,
- `v0_2` cleans and restructures the raw export,
- `v0_3` integrates silver labels and aligns the modeling schema,
- `v1_0` freezes the dataset as the benchmark release artifact used by the released code.

## Intended Artifact

The intended release artifact for this folder is:

```text
v1_0_stable.parquet
```

This file is the canonical dataset file expected by the model scripts in [../../../models/README.md](C:/Git/Bachelor/VEX/models/README.md:1).

The release dataset preserves the stable answer-level structure used across the repository, including:

- identifiers such as `member_id`, `subject_id`, `answer_id`, `question_id`, and `grading_id`,
- content fields such as `question` and `answer`,
- metadata fields such as `student_name`, `bloom_level`, and `question_topic`,
- the canonical `grade`, `label_type`, and `split` columns.

Compared with `v0.3_stable`, the `v1.0` release no longer retains the `gold_is_llm` column.

## Split Helper

This folder also includes [split_release.py](C:/Git/Bachelor/VEX/dataset/vex/v1_0_release/split_release.py:1).

That helper script reads:

```text
v1_0_stable.parquet
```

and writes two split-specific files:

- `train.parquet`
- `test.parquet`

The split is derived directly from the existing `split` column and is intended as a convenience export for downstream experimentation.

## Reproducibility Note

From the released repository contents, `v1.0` should be understood primarily as a **freezing and packaging stage** rather than a major schema change.

The current [v1_0.py](C:/Git/Bachelor/VEX/dataset/vex/v1_0_release/v1_0.py:1) file appears to be a carry-over script from the earlier `v0.3` stage and does not yet fully reflect the final public naming in this directory. In particular, it still reads from a `v0.3` path and writes `v0.3_stable` again.

For that reason, the most reliable interpretation of this folder is:

- `v1.0` is the frozen public release stage referenced by the modeling code,
- the stable schema is inherited from `v0.3_stable` with minor cleanup for release,
- `split_release.py` documents how the released file can be partitioned into `train` and `test`,
- the public repository snapshot does not contain a fully polished standalone conversion script for the final renaming step.

## Relation to Other Folders

- Dataset overview: [../README.md](C:/Git/Bachelor/VEX/dataset/vex/README.md:1)
- Dataset changelog: [../metadata/CHANGE_LOG.md](C:/Git/Bachelor/VEX/dataset/vex/metadata/CHANGE_LOG.md:1)
- Silver-label integration stage: [../v0_3_silver_labels/generation_silver_labels/README.md](C:/Git/Bachelor/VEX/dataset/vex/v0_3_silver_labels/generation_silver_labels/README.md:1)
- Model consumers of this release: [../../../models/README.md](C:/Git/Bachelor/VEX/models/README.md:1)
