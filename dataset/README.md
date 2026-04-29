# Dataset

This folder contains the released dataset assets for **VEX** together with several derived datasets used for different analyses in our papers.

At a high level, the dataset side of the repository is split into two parts:

- `vex/`: the main VEX dataset lineage and release metadata,
- `additional/`: task-specific derived datasets for audit analysis, feedback studies, teacher selection, and VEX metric evaluation.

## Structure

```text
dataset/
├── additional/
│   ├── audit_dataset/
│   ├── feedback_dataset/
│   ├── teacher_selection_dataset/
│   └── vex_metric_dataset/
├── vex/
│   ├── metadata/
│   ├── raw/
│   ├── v0_1_sqlite_export/
│   ├── v0_2_cleaning/
│   ├── v0_3_silver_labels/
│   └── v1_0_release/
├── download.py
└── verify_checksums.py
```

## Main Dataset Line

The main VEX dataset workflow lives in [vex/](vex/). It contains:

- the raw-to-release version lineage,
- the metadata files for release,
- the stable dataset versions used by downstream modeling and evaluation.

The key stable transitions are documented in:

- [vex/metadata/CHANGE_LOG.md](vex/metadata/CHANGE_LOG.md#L1)

In short:

- `v0_1` is the first stable export from the institutional SQLite source,
- `v0_2` cleans and restructures the answer/question content and label fields,
- `v0_3` integrates silver-label information and aligns the schema for modeling,
- `v1_0` is the public release version referenced by the released model scripts.

## Additional Derived Datasets

The [additional/](additional/) folder contains derived datasets for specific experimental purposes:

- `audit_dataset`: preserves annotation and audit columns for gold-label analysis,
- `feedback_dataset`: long-format dataset for model feedback studies,
- `teacher_selection_dataset`: gold-based dataset for teacher-model selection,
- `vex_metric_dataset`: merged predictions used by the VEX evaluation pipeline.

These datasets share the same underlying VEX data lineage but keep different fields depending on the downstream task.

## Relation to Annotation Workflows

The dataset release is closely tied to the human annotation workflows documented in:

- [../annotation/dataset/README.md](../annotation/dataset/README.md#L1)
- [../annotation/feedback/README.md](../annotation/feedback/README.md#L1)

Those folders document how the gold subset, audit process, and feedback-analysis samples were constructed.

## Release Metadata

The release metadata for the main VEX dataset is stored in:

- [vex/metadata/DATA_LICENSE.md](vex/metadata/DATA_LICENSE.md#L1)
- [vex/metadata/croissant_metadata.json](vex/metadata/croissant_metadata.json#L1)

The current dataset license is **CC BY 4.0**.

## Notes

- `dataset/download.py` and `dataset/verify_checksums.py` are repository-level helpers for dataset access and integrity checking.
- The dataset release should be read together with the model and evaluation folders, since several derived files are prepared specifically for the released benchmarking pipeline.
