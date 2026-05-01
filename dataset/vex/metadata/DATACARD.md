# VEX Dataset Card

## Dataset Summary

VEX is an answer-level dataset for short-answer grading in a university-level database systems course. It contains student responses to exam-style questions, question metadata, normalized grades, and a train/test split intended for supervised grading experiments and evaluation.

The public release artifact is:

```text
dataset/vex/v1_0_release/v1_0_stable.parquet
```

The released dataset contains both human-audited gold labels and model-generated silver labels. The gold subset is used as the benchmark-facing evaluation subset, while the silver-labeled training split supports scalable model training.

## Intended Use

The dataset is intended for:

- training and evaluating automatic short-answer grading systems,
- comparing model behavior on answer-level and exam-level grading tasks,
- studying grading calibration, ranking stability, and ordinal agreement,
- reproducible benchmark experiments for the VEX paper and accompanying code.

The dataset is not intended for high-stakes grading decisions without human review. Model outputs trained or evaluated on VEX should be treated as research artifacts, not as final grades for real students.

## Dataset Composition

The v1.0 release contains 30,682 answer-level rows from 173 anonymized students and 239 unique questions.

| Statistic | Value |
| --- | ---: |
| Students | 173 |
| Total responses | 30,682 |
| Unique questions | 239 |
| Gold-labeled responses | 3,222 |
| Silver-labeled responses | 27,460 |
| Average responses per question | 128.38 |
| Median responses per question | 152.00 |
| Average responses per student | 177.35 |
| Median responses per student | 199.00 |
| Average response length | 29.07 whitespace tokens |
| Median response length | 15.00 whitespace tokens |

The gold subset contains 3,222 responses from 168 students and 21 questions. It corresponds to the held-out evaluation subset in the release.

## Language Distribution

Language is estimated from the question text. Each response inherits the detected language of its question. If a question cannot be classified reliably, it is counted as English and reported separately.

| Split | English-question responses | German-question responses | Unknown-question responses counted as English |
| --- | ---: | ---: | ---: |
| Full dataset | 1,654 (5.39%) | 29,028 (94.61%) | 155 (0.51%) |
| Gold subset | 148 (4.59%) | 3,074 (95.41%) | 0 (0.00%) |

Most questions and answers are German. A small English-question portion is retained in the release, and unknown question-language cases are explicitly reported for transparency.

## Schema

The v1.0 release uses a compact answer-level schema:

| Column | Description |
| --- | --- |
| `member_id` | Anonymized student identifier. |
| `subject_id` | Course or subject identifier. |
| `answer_id` | Unique answer identifier. |
| `question_id` | Unique question identifier. |
| `grading_id` | Unique grading record identifier. |
| `student_name` | An anonymized readable student alias. |
| `question` | Plain-text question prompt. |
| `bloom_level` | Question-level Bloom taxonomy category. |
| `question_topic` | Topic label for the question. |
| `answer` | Plain-text student answer. |
| `grade` | Normalized grade on the dataset's numeric scoring scale. |
| `label_type` | Label source, either `gold` or `silver`. |
| `split` | Release split, either `train` or `test`. |

## Labels and Splits

The release contains two label types:

- `gold`: human-audited labels used for evaluation.
- `silver`: teacher-model-generated labels used for training.

The split is aligned with the label source:

| Split | Rows | Label type |
| --- | ---: | --- |
| `train` | 27,460 | silver |
| `test` | 3,222 | gold |

This design keeps the benchmark evaluation subset human-audited while allowing the training split to scale beyond the manually labeled data.

## Dataset Creation

The dataset lineage is:

```text
v0_1 -> v0_2 -> v0_3 -> v1_0
```

The main processing stages are:

- `v0_1`: first stable export from the institutional SQLite source.
- `v0_2`: cleaning and restructuring, including plain-text extraction, question metadata integration, grade normalization, and split creation.
- `v0_3`: silver-label integration and alignment with the downstream modeling schema.
- `v1_0`: frozen public release used by the model and evaluation scripts.

The fully raw institutional source database is not redistributed. The public repository documents the stable public processing stages and provides the release artifact used by downstream experiments.

## Preprocessing

The released dataset applies the following preprocessing steps:

- extracts question and answer text from the original structured content,
- removes non-core raw database fields and transient processing fields,
- maps audited human annotations to the canonical numeric `grade` column,
- merges silver labels for the training split,
- clamps grades to the valid normalized grading range,
- removes the `gold_is_llm` column from the public v1.0 release,
- preserves question metadata such as Bloom level and topic.

## Recommended Evaluation

For answer-level experiments, use the `test` split with `label_type == "gold"`.

For exam-level experiments, use the VEX metric pipeline in `vex_metric/`, which constructs virtual exams and reports metrics such as:

- item-level QWK and MSE,
- exam-level Kendall's tau-b,
- exam-level accuracy,
- exam-level QWK under linear and Bologna-style grading.

When reporting results, clearly distinguish between answer-level metrics and exam-level metrics, because they measure different properties of model behavior.

## Limitations

VEX reflects one institutional course context and one subject area: database systems. Results may not transfer directly to other subjects, grading rubrics, educational levels, or languages.

Most responses are German, with a smaller English-question subset. Language identification in the metadata reports is heuristic and based on question text, not a full language-identification model.

The silver-labeled training split inherits the biases and errors of the teacher model used to generate it. For this reason, gold-label evaluation remains the primary benchmark target.

Although identifiers are anonymized, the data comes from an educational context. Users must not attempt to re-identify students or use the dataset for student-level profiling.

## Ethical Considerations

Automatic grading systems can affect educational outcomes if used without oversight. VEX should be used to study and improve grading models, not to replace accountable human assessment in high-stakes settings.

Researchers should evaluate calibration, bias, robustness, and failure cases, especially when adapting models trained on VEX to new classrooms, subjects, or languages.

## License

The dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See [DATA_LICENSE.md](DATA_LICENSE.md) for the full license summary and attribution guidance.

## Citation

For research use, cite the associated VEX paper and specify the dataset version used. If a formal citation file or DOI is added later, prefer that citation in downstream work.
