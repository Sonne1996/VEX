# Gold Label and Audit Workflow

This folder contains the workflow used to create, clean, audit, and freeze the human gold labels for ASAG 2026.

The goal of this workflow is to turn independently annotated human labels from Google Sheets into a clean, auditable, and frozen gold-label dataset that can later be used for teacher selection, model evaluation, and benchmark release.

This README only describes the gold-label and audit workflow. The creation of the initial 10% gold subset is documented separately.

The resulting gold labels feed into the later dataset split, teacher-selection,
silver-label generation, and VEX metric evaluation workflows documented under
`dataset/`.

---

## Overview

The gold-label workflow follows these steps:

1. Export human annotations from Google Sheets to Parquet
2. Clean inconsistent annotator-column assignments
3. Compute automatic consensus labels from two human annotations
4. Flag large disagreements for manual audit
5. Upload the consensus/audit sheet back to Google Sheets
6. Manually audit unresolved or suspicious cases in Google Sheets
7. Download the audited sheet again
8. Freeze the final gold labels

Google Sheets was used as the main annotation and audit interface. Parquet files were used as stable local snapshots between workflow steps.

---

## Main Files

### `human_labeler_5_data.parquet`

Export of the Google Sheet worksheet:

`labels_jf`

This file contains annotations from an additional human labeler.

It is mainly used as an extra annotation source or quality-control reference.

---

### `dual_labels.parquet`

Export of the Google Sheet worksheet:

`additional_sampled_answers_dual_label`

This file contains answers that were independently labeled by two human annotators.

The important columns are:

| Column | Meaning |
|---|---|
| `human_grade` | First human grade |
| `human_grade 2` | Second human grade |
| `is_llm` | First annotator judgement whether the answer looks LLM-generated |
| `is_llm 2` | Second annotator judgement whether the answer looks LLM-generated |
| `grader_name` | Name of the first annotator |
| `grader_name 2` | Name of the second annotator |

The raw Google Sheet export keeps all columns as strings to avoid mixed-type problems caused by Google Sheets.

---

### `post_first_audit.parquet`

Export of the Google Sheet worksheet:

`first_audit_gold_label`

This file represents the gold-label sheet after the first audit round.

It is an intermediate snapshot and should not automatically be treated as final.

---

### `post_first_audit_cleaned.parquet`

Cleaned version of `post_first_audit.parquet`.

This file fixes the issue where the annotator columns changed order inside the sheet.

The intended final convention is:

| Column | Meaning |
|---|---|
| `human_grade 1` | Kevin's human grade |
| `human_grade 2` | Sercan's human grade |
| `is_llm 1` | Kevin's LLM-detection label |
| `is_llm 2` | Sercan's LLM-detection label |
| `grader_name 1` | Always `Kevin` |
| `grader_name 2` | Always `Sercan` |

This cleanup is important because some rows originally had:

- `grader_name 1 = Kevin`
- `grader_name 2 = Sercan`

while later rows had:

- `grader_name 1 = Sercan`
- `grader_name 2 = Kevin`

The cleaning script detects this switch and normalizes the columns.

---

### `dual_labels_consensus.parquet`

Output of the automatic consensus step.

This file contains the two human labels plus automatically computed consensus labels and audit flags.

Important columns:

| Column | Meaning |
|---|---|
| `gold_grade_before_audit` | Automatically computed consensus grade before manual audit |
| `consensus_status_audit` | Whether the row needs audit |
| `consensus_audit_reason` | Reason why the row was or was not flagged |
| `gold_is_llm_before_audit` | Merged LLM-detection label before audit |
| `_label_one_num` | Normalized numeric value of the first human grade |
| `_label_two_num` | Normalized numeric value of the second human grade |
| `_label_distance` | Absolute difference between both human grades |
| `_label_mean_raw` | Raw average of both human grades before snapping to the valid label scale |

The allowed numeric grade scale is:

| Numeric value | Text label |
|---:|---|
| `0.00` | `Incorrect` |
| `0.25` | `Mostly incorrect` |
| `0.50` | `Partially correct` |
| `0.75` | `Mostly correct` |
| `1.00` | `Correct` |

---

### `frozen_gold.parquet`

Export of the final audited Google Sheet.

This is the frozen gold-label dataset after manual audit.

This file should be treated as the authoritative human-grounded gold standard.

Once this file is created and checked, it should not be modified silently. Any later change must be documented as a new dataset version.

---

## Google Sheets Worksheets

The workflow uses one shared Google Spreadsheet.

The main worksheets are:

| Worksheet | Purpose |
|---|---|
| `labels_jf` | Additional human labeler export |
| `additional_sampled_answers_dual_label` | Dual-label annotation sheet |
| `first_audit_gold_label` | Main audit sheet after consensus generation |

Google Sheets was used because it allowed manual inspection, annotation, audit, and correction by humans.

---

## Label Schema

All grading labels are mapped to a five-level ordinal scale.

| Text label | Numeric value |
|---|---:|
| `Incorrect` | `0.00` |
| `Mostly incorrect` | `0.25` |
| `Partially correct` | `0.50` |
| `Mostly correct` | `0.75` |
| `Correct` | `1.00` |

The numeric scale is used for computation.

The text labels are used in Google Sheets for readability.

---

## Exporting Google Sheets to Parquet

Several scripts export worksheets from Google Sheets into local Parquet files.

The general logic is:

1. Authenticate with Google Sheets using `credentials.json`
2. Open the spreadsheet by URL
3. Select the worksheet by name
4. Load all records into a Pandas DataFrame
5. Convert all columns to strings
6. Write the result as a Snappy-compressed Parquet file

All columns are explicitly stored as `pa.string()` to avoid problems with mixed column types from Google Sheets.

Example output files:

- `human_labeler_5_data.parquet`
- `dual_labels.parquet`
- `post_first_audit.parquet`
- `frozen_gold.parquet`

This ensures that every important state of the annotation workflow exists as a reproducible local snapshot.

---

## Cleaning Switched Annotator Columns

During manual annotation, the order of annotators changed in part of the sheet.

This means that in one part of the dataset:

- `grader_name 1 = Kevin`
- `grader_name 2 = Sercan`

but in another part:

- `grader_name 1 = Sercan`
- `grader_name 2 = Kevin`

The cleanup script fixes this by detecting the first row where the switch happens.

It then creates temporary split-view columns:

- `kevin_grade_up`
- `kevin_grade_down`
- `sercan_grade_up`
- `sercan_grade_down`
- `kevin_is_llm_up`
- `kevin_is_llm_down`
- `sercan_is_llm_up`
- `sercan_is_llm_down`

Afterwards, the final columns are rebuilt so that the convention is always:

- `human_grade 1 = Kevin`
- `human_grade 2 = Sercan`
- `is_llm 1 = Kevin`
- `is_llm 2 = Sercan`
- `grader_name 1 = Kevin`
- `grader_name 2 = Sercan`

The temporary output is:

`dual_labels_temp_split_view.parquet`

The cleaned final output is:

`post_first_audit_cleaned.parquet`

Only the cleaned file should be used for later consensus or audit steps.

---

## Consensus Rule

The consensus script combines two independent human labels into a preliminary gold label.

Let:

- `label_one = first human grade`
- `label_two = second human grade`
- `distance = abs(label_one - label_two)`
- `raw_mean = (label_one + label_two) / 2`

The rule is:

```python
if distance > 0.5:
    send row to audit
else:
    average both labels and round to the nearest valid label
```

The valid labels are:

- `0.00`
- `0.25`
- `0.50`
- `0.75`
- `1.00`

If the average lies exactly between two valid labels, the value is rounded towards `0.50`.

This avoids systematically rounding disagreements upwards or downwards.

---

## Examples of Consensus Behaviour

| Label 1 | Label 2 | Distance | Result | Audit? |
|---:|---:|---:|---:|---|
| `1.00` | `1.00` | `0.00` | `1.00` | No |
| `1.00` | `0.75` | `0.25` | `1.00` | No |
| `0.00` | `0.25` | `0.25` | `0.25` | No |
| `1.00` | `0.50` | `0.50` | `0.75` | No |
| `1.00` | `0.25` | `0.75` | Empty | Yes |
| `1.00` | `0.00` | `1.00` | Empty | Yes |

Large disagreements are not resolved automatically. They are flagged for manual audit.

---

## Audit Flagging

Rows are flagged for audit when:

```python
abs(label_one - label_two) > 0.5
```

These rows receive:

- `consensus_status_audit = audit`
- `consensus_audit_reason = distance_gt_0.5`

Rows that can be resolved automatically receive:

- `consensus_status_audit = no audit`
- `consensus_audit_reason = within_threshold`

Invalid or missing grades are also flagged for audit:

- `consensus_audit_reason = invalid_grade_input`

The main purpose of this step is to keep automatic consensus conservative.

The workflow avoids silently resolving strong disagreements between human annotators.

---

## LLM-Detection Label Merge

In addition to grading correctness, annotators also marked whether an answer looked LLM-generated.

The two columns are:

- `is_llm`
- `is_llm 2`

or, after cleanup:

- `is_llm 1`
- `is_llm 2`

The merge mode used in the script is:

`AND`

This means the final pre-audit LLM label is only `Yes` if both annotators marked the answer as LLM-generated.

| Label 1 | Label 2 | Result |
|---|---|---|
| `Yes` | `Yes` | `Yes` |
| `Yes` | `No` | `No` |
| `No` | `Yes` | `No` |
| `No` | `No` | `No` |

The output column is:

`gold_is_llm_before_audit`

This conservative rule avoids over-labeling answers as LLM-generated based on only one annotator.

---

## Uploading Consensus Results Back to Google Sheets

After automatic consensus generation, the result is uploaded back to Google Sheets for manual audit.

The upload script reads:

`dual_labels_consensus.parquet`

and uploads it to:

`first_audit_gold_label`

Before upload, the numeric consensus column:

`gold_grade_before_audit`

is converted into text labels in:

`gold_label_before_audit`

The mapping is:

| Numeric value | Text label |
|---:|---|
| `0.00` | `Incorrect` |
| `0.25` | `Mostly incorrect` |
| `0.50` | `Partially correct` |
| `0.75` | `Mostly correct` |
| `1.00` | `Correct` |

This makes the sheet easier to review manually.

If the worksheet already exists, it is cleared and overwritten.

---

## Manual Audit in Google Sheets

The manual audit is performed in the worksheet:

`first_audit_gold_label`

The audit focuses especially on rows where:

`consensus_status_audit = audit`

These are cases where the two human annotators disagreed too strongly for automatic resolution.

Typical audit cases include:

- One annotator marked the answer as correct, the other as incorrect
- One annotator gave `1.00`, the other gave `0.25`
- One annotator gave `1.00`, the other gave `0.00`
- The answer is ambiguous
- The answer is partially correct but difficult to place on the scale
- The answer contains vague, contradictory, empty, or nonsensical content
- The question itself makes grading difficult

The audit should be done question by question where possible, so that similar answers are judged consistently.

The aim is to reduce unresolved disagreement and create one final human-grounded label per answer.

If a case cannot be resolved by the main annotators, it should be escalated to an additional auditor.

---

## Freezing the Gold Labels

After the manual audit is finished, the audited Google Sheet is downloaded again as:

`frozen_gold.parquet`

This file represents the final gold labels.

The frozen gold labels are used for:

- Teacher selection
- Gold-dev evaluation
- Gold-test evaluation
- Benchmark reporting
- Final model comparison
- Dataset release documentation

After freezing, the file should be treated as immutable.

If changes are required later, they should be released as a new version, not silently overwritten.

---

## Recommended Workflow Order

1. Export dual labels from Google Sheets

   `additional_sampled_answers_dual_label -> dual_labels.parquet`

2. Clean annotator-column inconsistencies if needed

   `post_first_audit.parquet -> post_first_audit_cleaned.parquet`

3. Compute consensus labels

   `dual_labels.parquet -> dual_labels_consensus.parquet`

4. Upload consensus file to Google Sheets

   `dual_labels_consensus.parquet -> first_audit_gold_label`

5. Manually audit flagged rows in Google Sheets

6. Download audited sheet

   `first_audit_gold_label -> frozen_gold.parquet`

7. Treat `frozen_gold.parquet` as the final gold-label file

---

## Important Design Decisions

### Google Sheets as annotation interface

Google Sheets was used because it is simple for human annotators and auditors.

It allows fast manual review, filtering, correction, and discussion.

The downside is that types can become inconsistent, which is why all exports are forced to string columns before saving to Parquet.

---

### Parquet as stable snapshot format

Each relevant workflow stage is stored as a Parquet file.

This makes the workflow easier to reproduce, inspect, and version.

Parquet also avoids many CSV-related problems, such as broken delimiters, formatting issues, and accidental type conversion.

---

### Conservative automatic consensus

The automatic consensus step only resolves small disagreements.

Large disagreements are flagged for manual audit.

This is important because the gold labels should be human-grounded and not mainly produced by automatic averaging.

---

### Strong disagreements require audit

Any pair of human labels with a distance greater than `0.5` is considered too unstable for automatic resolution.

These rows are not assigned a final consensus label automatically.

Instead, they are sent to manual audit.

---

### Rounding towards the middle

When the average lies exactly between two valid labels, the workflow rounds towards `0.5`.

This avoids artificially making uncertain answers too correct or too incorrect.

---

### Frozen gold is separated from silver labels

The frozen gold labels are kept separate from later silver labels.

Gold labels are human-grounded and used for evaluation.

Silver labels are model-generated and used for scalable training or additional experiments.

This separation is important to avoid methodological leakage between training and evaluation.

---

## Column Reference

### Human annotation columns

| Column | Meaning |
|---|---|
| `human_grade` | First human grade before cleanup |
| `human_grade 2` | Second human grade before cleanup |
| `human_grade 1` | First normalized human grade after cleanup |
| `human_grade 2` | Second normalized human grade after cleanup |
| `grader_name` | First annotator name before cleanup |
| `grader_name 1` | First normalized annotator name after cleanup |
| `grader_name 2` | Second normalized annotator name |
| `is_llm` | First LLM-detection label before cleanup |
| `is_llm 1` | First normalized LLM-detection label after cleanup |
| `is_llm 2` | Second normalized LLM-detection label |

---

### Consensus columns

| Column | Meaning |
|---|---|
| `gold_grade_before_audit` | Preliminary consensus grade before manual audit |
| `gold_label_before_audit` | Text version of the preliminary consensus grade |
| `consensus_status_audit` | Indicates whether the row needs manual audit |
| `consensus_audit_reason` | Reason for the consensus/audit decision |
| `gold_is_llm_before_audit` | Merged pre-audit LLM-detection label |

---

### Debug columns

| Column | Meaning |
|---|---|
| `_label_one_num` | Numeric version of first human grade |
| `_label_two_num` | Numeric version of second human grade |
| `_is_llm_one_bool` | Boolean version of first LLM-detection label |
| `_is_llm_two_bool` | Boolean version of second LLM-detection label |
| `_label_distance` | Absolute difference between both human grades |
| `_label_mean_raw` | Raw mean before snapping to valid label scale |

These columns are useful for validation and audit analysis.

They may be removed from a final public release if they are not needed.

---

## Quality Checks

Before freezing the gold labels, the following checks should be performed:

- Verify that all final gold labels are on the allowed five-point scale
- Verify that no audit rows are unresolved
- Verify that all rows have a final human-grounded label
- Verify that annotator columns are consistently assigned
- Verify that Kevin and Sercan labels were not accidentally swapped
- Check the distribution of final labels
- Check the amount of empty or nonsensical answers
- Check question-level difficulty based on average score
- Check whether some questions are too easy or too hard
- Check Bloom-level distribution
- Check answer length distribution
- Check whether the gold subset is still representative of the full dataset

---

## Role of the Frozen Gold Dataset

The frozen gold dataset is the central evaluation anchor of ASAG 2026.

It should be used to:

- Select the best teacher model on gold-dev
- Evaluate final systems on gold-test
- Compare item-level and exam-level metrics
- Study how much human gold is needed to create a useful silver-labeled extension
- Prevent training/evaluation leakage
- Support reproducible benchmark reporting

The frozen gold dataset should not be mixed with silver-labeled training data.

---

## Notes for Release

For a public GitHub or Hugging Face release, the following files should be documented clearly:

```text
frozen_gold.parquet
dual_labels_consensus.parquet
post_first_audit_cleaned.parquet
```

Recommended public release structure:

```text
gold_workflow/
  README.md
  scripts/
    export_google_sheet_to_parquet.py
    clean_switched_annotator_columns.py
    create_dual_label_consensus.py
    upload_consensus_to_google_sheet.py
  data/
    frozen_gold.parquet
    post_first_audit_cleaned.parquet
    dual_labels_consensus.parquet
```

Do not release:

```text
credentials.json
```

The Google service-account credentials are private and must never be committed to GitHub or uploaded to Hugging Face.

---

## Reproducibility Notes

The workflow is reproducible if the following are available:

- The Google Sheet or exported Parquet snapshots
- The scripts used for export, cleanup, consensus, and upload
- The exact label schema
- The exact consensus rule
- The frozen final gold file
- Documentation of all manual audit decisions

Because manual audit decisions happen in Google Sheets, the final frozen Parquet file is the authoritative snapshot for later experiments.

---

## Summary

This workflow creates a human-grounded gold-label dataset from dual human annotations.

Small disagreements are resolved automatically using a documented consensus rule.

Large disagreements are sent to manual audit.

The audited result is exported as `frozen_gold.parquet` and used as the stable gold standard for ASAG 2026.

The key principle is:

```text
Human gold is used for trustworthy evaluation.
Silver labels are used for scalable training.
The two must remain clearly separated.
```

## Initial Gold Subset Sampling

We created the initial gold-annotation subset by sampling approximately 10% of the question pool from the exported `answers.csv` table using a fixed random seed (`210`).

Before sampling, we excluded questions with fewer than 10 available answers so that each selected question contributed a meaningful answer set to the annotation workflow.

Sampling was performed at the `question_id` level rather than at the individual-answer level. After the question subset was chosen, all answers belonging to the selected questions were retained and written to `additional_sampled_answers.csv`.

Because the sampling unit was the question, not the answer, the resulting proportion of sampled answers is only approximately 10% of the eligible answer pool and can vary slightly depending on the number of answers attached to each sampled question.

This sampled subset served as the starting point for the dual-annotation, consensus, and audit workflow documented above.
