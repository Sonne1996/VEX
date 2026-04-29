# Silver Label Generation

This folder documents how the silver labels for **VEX v0.3** were generated.

The silver-label workflow was used to scale beyond the manually annotated gold subset. In short, we first selected a teacher model on gold data, then used that teacher to grade the large training split, and finally merged those grades back into the main dataset lineage.

## Overview

The silver labels in `v0.3` were produced in three stages:

1. **Teacher selection on gold data**
2. **Large-scale silver-label generation on the training split**
3. **Merge and cleanup into `v0.3_stable`**

## 1. Teacher Selection

Before generating silver labels, we selected the teacher model using the gold-based teacher-selection dataset documented in:

- [../../../additional/teacher_selection_dataset/README.md](../../../additional/teacher_selection_dataset/README.md#L1)

In the released workflow, the selected teacher was:

```text
google/gemini-2.5-pro
```

This means the silver-label generation stage was not based on an arbitrary LLM choice. The teacher was chosen from a dedicated gold-evaluation stage before being used to scale grading over the larger training pool.

## 2. Silver-Label Generation via Vercel Gateway

After teacher selection, silver grades were generated for the training split through the Vercel AI Gateway using the OpenAI-compatible async client.

The script was configured with:

- input file: `../train_split/train_split.parquet`
- output file: `results_silver_labels.parquet`
- teacher model: `google/gemini-2.5-pro`
- concurrency: `16`
- row limit: `27464`

The row limit corresponds to the full train-split size used for silver-label generation in this workspace snapshot.

### Prompting Setup

Each request used a grading prompt tailored to university-level database systems short answers.

The prompt instructed the model to:

- grade only from the `QUESTION` and `STUDENT_ANSWER`,
- use the normalized five-point grading scale,
- focus on technical correctness rather than fluency,
- accept paraphrases when semantically correct,
- output only JSON of the form `{"grade": ...}`.

The allowed score set was:

```text
0.0, 0.25, 0.5, 0.75, 1.0
```

### Runtime Behavior

For each row, the script:

1. builds a grading prompt from `question` and `answer`,
2. sends the request asynchronously to the selected teacher model,
3. stores the raw model output,
4. extracts the grade using a strict-to-relaxed parsing strategy,
5. saves intermediate progress back to parquet so the run can resume safely.

The extraction logic was intentionally defensive:

- first try strict JSON parsing,
- then try cleaned JSON after removing code fences,
- finally fall back to a regex match over the allowed grade values.

If no valid grade could be recovered, the script returned `-1.0` as a failure marker.

### Resume and Fault Tolerance

The generation script supports interrupted long-running jobs.

If `results_silver_labels.parquet` already exists, the script reloads it, merges cached silver-label columns back onto the original input via `grading_id`, and only processes rows that are still missing generated grades.

This makes the workflow resumable without restarting the full generation run.

## 3. Generated Output

The gateway script writes a compact parquet file containing:

- `grading_id`
- `question`
- `answer`
- `split`
- `new_grade_google/gemini-2.5-pro`
- `raw_output_google/gemini-2.5-pro`
- `metadata_google/gemini-2.5-pro`

These columns preserve:

- the generated silver grade,
- the raw teacher response,
- request metadata such as token counts, latency, finish reason, model identity, and timestamps.

## 4. Merge into the Main Dataset Lineage

The generated silver labels are incorporated into the main dataset through [../v0_3.py](../v0_3.py#L1).

That script performs the following versioned steps:

### `v0.21`

`results_silver_labels.parquet` is merged back into `v0.2_stable.parquet` using:

```text
grading_id
```

Only genuinely new silver-generation columns are brought in, which prevents duplicate fields such as `question_x` / `question_y`.

### `v0.22`

Missing values in the canonical `grade` column are filled from:

```text
new_grade_google/gemini-2.5-pro
```

This is the step where the generated teacher grades become the actual silver supervision for previously unlabeled training rows.

### `v0.23`

Grades are clamped to the valid range `[0.0, 1.0]`.

### `v0.24`

Transient generation columns are removed:

- `model_response_with_metadata`
- `new_grade_google/gemini-2.5-pro`
- `raw_output_google/gemini-2.5-pro`
- `metadata_google/gemini-2.5-pro`

### `v0.25` and `v0.26`

Remaining audit-heavy and intermediate columns are removed to align the dataset with the cleaner modeling schema.

### `v0.3_stable`

Finally, the stable naming convention is applied:

- `name` -> `student_name`
- `bloom` -> `bloom_level`
- `topic` -> `question_topic`

The result is the released `v0.3_stable.parquet`, the first stable dataset version aligned with the downstream modeling pipeline.

## Workflow Summary

In release terms, the silver-label workflow is:

1. select the teacher on gold data,
2. run the selected teacher over the train split through the Vercel gateway,
3. parse and cache normalized grades,
4. merge generated grades into the cleaned dataset,
5. promote those grades into the canonical `grade` field where needed,
6. remove transient generation metadata for the stable release.

## Reproducibility Notes

This folder documents the released silver-label generation procedure, but not all runtime secrets are public.

In particular, the original script depends on a private gateway key file:

```text
../api_key/vercel_api_key.txt
```

This credential must not be committed or redistributed.

The public release therefore documents:

- the teacher-selection decision,
- the prompting and parsing logic,
- the merge path into `v0.3`,
- the stable output schema used by downstream models.

## Relation to Other Folders

- Gold annotation and audit workflow: [../../../../annotation/dataset/README.md](../../../../annotation/dataset/README.md#L1)
- Teacher selection dataset: [../../../additional/teacher_selection_dataset/README.md](../../../additional/teacher_selection_dataset/README.md#L1)
- Main VEX dataset lineage: [../../README.md](../../README.md#L1)
