# Additional Datasets

This folder contains derived VEX datasets prepared for specific downstream analyses beyond the main benchmark release. Each subfolder corresponds to a different paper-facing use case or metric layer.

The datasets here are not redundant copies of the main release. Instead, they preserve task-specific structure that is useful for:

- human audit analysis,
- teacher-model selection,
- feedback quality studies,
- VEX exam-level evaluation.

## Subfolders

### `audit_dataset/`

Audit-focused dataset with human annotation and audit metadata preserved. This version is useful for disagreement analysis, audit tracing, and studying how final gold labels were resolved.

Main file:

```text
audit_dataset.parquet
```

### `feedback_dataset/`

Long-format dataset for feedback-oriented experiments. Each row contains a student answer together with one model's grade and written feedback, making it suitable for feedback evaluation or comparison across feedback-generating models.

Main file:

```text
merged_feedback_long.parquet
```

### `teacher_selection_dataset/`

Gold-based dataset used for teacher-model selection workflows. It preserves gold annotation context together with teacher-candidate outputs and model-based evaluation columns used to compare or rank candidate teachers before silver-label generation.

Main file:

```text
gold_with_all_models.parquet
```

### `vex_metric_dataset/`

The merged prediction dataset used by the VEX evaluation pipeline. This is the most benchmark-facing derived dataset in this folder and contains the human reference grade plus the prediction columns needed for item-level and exam-level evaluation.

Main file:

```text
merged_model_predictions.parquet
```

## Relation to the Main Dataset

These datasets are derived from the same underlying VEX data lineage, but they keep different subsets of metadata depending on the target use case:

- the audit dataset keeps annotation and audit columns,
- the feedback dataset reshapes outputs into feedback-centric long format,
- the teacher-selection dataset keeps teacher-candidate and evaluator outputs,
- the VEX metric dataset keeps final model prediction columns for benchmark evaluation.

For the benchmark-ready dataset lineage itself, see [../vex/metadata/CHANGE_LOG.md](../vex/metadata/CHANGE_LOG.md#L1).
