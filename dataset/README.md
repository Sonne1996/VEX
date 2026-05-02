# Dataset

This folder contains the released dataset assets for VEX together with derived
datasets used for paper-specific analyses and evaluation scripts.

At a high level, the dataset side of the repository is split into:

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

The main dataset workflow lives in [vex/](vex/). It documents the public
raw-to-release lineage, stable parquet artifacts, and release metadata.

The version path is:

```text
v0_1 -> v0_2 -> v0_3 -> v1_0
```

In short:

- `v0_1` is the first stable export from the institutional SQLite source.
- After `v0_1`, the dataset is split into train and test partitions.
- `v0_2` cleans and restructures question, answer, metadata, and label fields.
- `v0_3` integrates silver labels and creates the stable modeling schema.
- After `v0_3_stable`, the four additional derived datasets are created.
- `v1_0` freezes the public benchmark release used by the released model scripts.

The main changelog is:

- [vex/metadata/CHANGE_LOG.md](vex/metadata/CHANGE_LOG.md)

## Additional Derived Datasets

The [additional/](additional/) folder contains derived datasets for specific
experimental purposes:

- `audit_dataset`: annotation and audit columns for gold-label analysis.
- `feedback_dataset`: long-format dataset for model feedback studies.
- `teacher_selection_dataset`: gold-based dataset for teacher-model selection.
- `vex_metric_dataset`: merged model predictions used by the VEX evaluation pipeline.

These datasets are derived from the same VEX lineage but keep different fields
depending on the downstream task.

## Relation to Annotation Workflows

The dataset release is closely tied to the human annotation workflows documented
in:

- [../annotation/dataset/README.md](../annotation/dataset/README.md)
- [../annotation/feedback/README.md](../annotation/feedback/README.md)

Those folders document how the gold subset, audit process, and feedback-analysis
samples were constructed.

## Release Metadata and License

Release metadata is stored in:

- [vex/metadata/DATA_LICENSE.md](vex/metadata/DATA_LICENSE.md)
- [vex/metadata/CHANGE_LOG.md](vex/metadata/CHANGE_LOG.md)
- [vex/metadata/croissant_metadata.json](vex/metadata/croissant_metadata.json)

The current dataset license is CC BY 4.0. It applies to the main dataset release
under `dataset/vex/` and to the derived datasets under `dataset/additional/`.

## Notes

- `dataset/download.py` and `dataset/verify_checksums.py` are repository-level helpers for dataset access and integrity checks.
- The dataset release should be read together with `models/`, `vex_metric/`, and `results/`, since several derived files are prepared specifically for the released benchmarking pipeline.
