# VEX Metric Dataset

This folder contains the dataset used by the VEX evaluation pipeline.

## File

`merged_model_predictions.parquet`

This is the benchmark-facing merged prediction file. It contains the human reference grade together with the prediction columns needed for the VEX item-level and exam-level evaluation pipeline.

The VEX pipeline uses this dataset to compute:

- standard item-level ASAG metrics,
- virtual-exam-level metrics after aggregation over sampled exams,
- cross-model comparisons under the VEX evaluation protocol.

## Structure

The file contains the core answer-level context:

- `grading_id`
- `member_id`
- `subject_id`
- `answer_id`
- `question_id`
- `student_name`
- `question`
- `bloom_level`
- `question_topic`
- `answer`
- `grade`
- `label_type`
- `gold_is_llm`
- `split`

and the released prediction columns:

### Joint LLM grading models

- `new_grade_deepseek/deepseek-v3.2-thinking`
- `new_grade_deepseek/deepseek-v3.2`
- `new_grade_google/gemini-2.5-pro`
- `new_grade_anthropic/claude-sonnet-4.6`
- `new_grade_openai/gpt-5.4`
- `new_grade_llama32_3b_base`
- `new_grade_gemma_e4_base`
- `new_grade_llama32_3b_ft`
- `new_grade_gemma_e4_ft`

### Transformer models

- `grade_bert_base`
- `grade_bert_ft`
- `grade_mdeberta_base`
- `grade_mdeberta_ft`

### Prior/template baselines

- `grade_prior_global`
- `grade_prior_template_overlap`

### TF-IDF baselines

- `pred_tfidf_v5_answer_char_3_5`
- `pred_tfidf_v1_answer_word_unigram`
- `pred_tfidf_v4_question_and_answer_separate`
- `pred_tfidf_v2_answer_word_uni_bigram`
- `pred_tfidf_v3_qa_concat_word_uni_bigram`
- `pred_tfidf_v6_mixed_word_char_qa`

## Usage

The VEX metric code expects this dataset at:

```text
dataset/additional/vex_metric_dataset/merged_model_predictions.parquet
```

in the benchmark-oriented pipeline layout documented in [../../../vex_metric/README.md](../../../vex_metric/README.md#L1).

## Notes

- This dataset uses the later stable naming convention such as `student_name`, `bloom_level`, and `question_topic`.
- Unlike the audit and teacher-selection datasets, it is intentionally trimmed to the fields most relevant for evaluation rather than annotation tracing.

## Provenance Note

This dataset is the evaluation-facing derivative of the main VEX data lineage. It is meant to be used together with the VEX metric pipeline documented in [../../../vex_metric/README.md](../../../vex_metric/README.md#L1), not as the annotation or teacher-selection working dataset.
