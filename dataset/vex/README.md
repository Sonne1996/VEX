# VEX Dataset

This folder contains the main VEX dataset lineage from the first stable export
to the benchmark-facing release files.

It is the canonical source for:

- dataset version history,
- release metadata,
- stable parquet artifacts,
- public processing scripts from the released stages onward.

## Structure

```text
vex/
├── metadata/
├── raw/
├── v0_1_sqlite_export/
├── v0_2_cleaning/
├── v0_3_silver_labels/
└── v1_0_release/
```

## Version Overview

### `raw/`

Documents the raw institutional SQLite source. The database itself is not part
of the public release.

### `v0_1_sqlite_export/`

First stable export from the institutional SQLite source into a unified parquet
format.

After this stage, the dataset is split into train and test partitions. The split
is then carried through the later cleaning, silver-label, and release stages.

### `v0_2_cleaning/`

Cleaning and restructuring stage. This version extracts plain-text question and
answer content, merges question metadata such as Bloom level and topic,
normalizes gold labels, and creates the canonical `grade`, `label_type`, and
`split` fields.

### `v0_3_silver_labels/`

Extends the cleaned dataset with silver-label information used for scalable
training and aligns the schema with the downstream modeling pipeline. The result
is `v0.3_stable`, the first stable modeling-oriented version.

After `v0.3_stable`, the four derived datasets under `dataset/additional/` are
created for audit analysis, feedback evaluation, teacher selection, and VEX
metric evaluation.

### `v1_0_release/`

Frozen public release stage. It contains:

- `v1_0_stable.parquet`
- `train.parquet`
- `test.parquet`
- `split_release.py`
- `croissant_metadata.json`

The model scripts use `v1_0_stable.parquet` as their main released dataset
input.

## Metadata

The release metadata is stored in [metadata/](metadata/):

- [CHANGE_LOG.md](metadata/CHANGE_LOG.md)
- [DATA_LICENSE.md](metadata/DATA_LICENSE.md)
- [croissant_metadata.json](metadata/croissant_metadata.json)

The dataset license also covers the derived datasets under
`dataset/additional/`.

## Reproducibility Scope

The fully raw institutional source database is not public. Therefore:

- `v0_1` is the first stable public artifact in the lineage,
- the public release includes transformation scripts from the released processing stages onward,
- later stable versions can be inspected from the released scripts, parquet artifacts, and metadata.

## Relation to Other Folders

- Gold annotation and audit workflow: [../../annotation/dataset/README.md](../../annotation/dataset/README.md)
- Feedback sampling workflow: [../../annotation/feedback/README.md](../../annotation/feedback/README.md)
- Model consumers of `v1_0_stable.parquet`: [../../models/README.md](../../models/README.md)
- VEX metric input dataset: [../additional/vex_metric_dataset/README.md](../additional/vex_metric_dataset/README.md)
