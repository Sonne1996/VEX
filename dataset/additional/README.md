# Additional Datasets

This folder contains derived VEX datasets prepared after `v0.3_stable` for
specific downstream analyses beyond the main benchmark release.

The datasets here are not redundant copies of the main release. Each one keeps a
task-specific view of the same underlying VEX data lineage.

## Subfolders

### `audit_dataset/`

Audit-focused dataset with human annotation and audit metadata preserved.

Main file:

```text
audit_dataset.parquet
```

Use this dataset for disagreement analysis, audit tracing, and studying how
final gold labels were resolved.

### `feedback_dataset/`

Long-format dataset for feedback-oriented experiments.

Main file:

```text
merged_feedback_long.parquet
```

Each row contains a student answer together with one model's grade and written
feedback.

### `teacher_selection_dataset/`

Gold-based dataset used for teacher-model selection workflows.

Main file:

```text
gold_with_all_models.parquet
```

This dataset preserves gold annotation context together with teacher-candidate
outputs and evaluator-model columns used before silver-label generation.

### `vex_metric_dataset/`

Merged prediction dataset used by the VEX metric pipeline.

Main file:

```text
merged_model_predictions.parquet
```

This is the benchmark-facing dataset for item-level and virtual-exam-level
evaluation.

## Relation to the Main Dataset

These datasets are created after the stable modeling schema is available. They
share the same VEX lineage, but each keeps the columns needed for its own
analysis:

- the audit dataset keeps annotation and audit columns,
- the feedback dataset reshapes feedback-generating model outputs into long format,
- the teacher-selection dataset keeps teacher-candidate and evaluator outputs,
- the VEX metric dataset keeps final prediction columns for benchmark evaluation.

For the benchmark-ready dataset lineage itself, see
[../vex/metadata/CHANGE_LOG.md](../vex/metadata/CHANGE_LOG.md).

## Metadata and License

Each subfolder includes Croissant metadata where available. The dataset license
is documented in [../vex/metadata/DATA_LICENSE.md](../vex/metadata/DATA_LICENSE.md)
and applies to these additional datasets as well.
