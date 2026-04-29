# Teacher Selection Dataset

This folder contains the teacher-selection VEX dataset.

## File

`gold_with_all_models.parquet`

This dataset is built around the gold-labeled portion of the data and is intended for selecting or comparing teacher models before generating larger silver-labeled training data.

In the broader project context, the gold set is explicitly used to select the best teacher model on gold-dev before scaling to silver-label generation.

## Structure

The dataset keeps the answer-level gold context:

- `member_id`
- `subject_id`
- `answer_id`
- `question_id`
- `grading_id`
- `name`
- `question`
- `bloom`
- `topic`
- `answer`
- `grade`
- `label_type`
- `gold_is_llm`
- `split`

It also preserves annotation and audit metadata:

- `rating`
- `human_grade 1`
- `is_llm 1`
- `grader_name 1`
- `human_grade 2`
- `is_llm 2`
- `grader_name 2`
- `gold_label_after_human_audit`
- `consensus_status_audit`
- `human_audit_comment`

and adds teacher-selection model outputs:

- `model_response_with_metadata`
- `new_grade_google/gemini-2.5-pro`
- `raw_output_google/gemini-2.5-pro`
- `metadata_google/gemini-2.5-pro`

plus evaluator-model columns:

- `eval_grade_deepseek_deepseek_v3_2_thinking`
- `eval_grade_deepseek_deepseek_v3_2`
- `eval_grade_google_gemini_2_5_pro`
- `eval_grade_anthropic_claude_sonnet_4_6`
- `eval_grade_openai_gpt_5_4`

## Intended Use

This dataset is meant for analyses such as:

- selecting a teacher model on gold data,
- comparing teacher-candidate outputs against gold labels,
- studying how different evaluator models score candidate teacher behavior,
- supporting the transition from gold supervision to silver-label generation.

## Notes

- The exact teacher-selection workflow is only partially documented in the public release, but the column structure clearly reflects a gold-set teacher-evaluation stage.
- Compared with the final VEX metric dataset, this file keeps more raw generation and audit metadata and fewer finalized benchmark prediction columns.
