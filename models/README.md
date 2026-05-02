# Models

This directory contains the modeling code released with VEX. The code is
organized by model family and reflects the experimental workspace used to
generate prediction columns consumed by the VEX metric pipeline.

This is a research release rather than a packaged training library. Most
configuration is stored directly inside scripts, so paths and model identifiers
should be checked before running experiments.

## Directory Layout

```text
models/
├── classical_asag/
│   └── tfidf/
├── encoder/
│   ├── bert/
│   └── deberta/
├── hf_key/
├── joint_models/
│   ├── gemma/
│   └── llama/
└── prior_template/
    ├── global_prior/
    └── qa_overlap_template/
```

## Shared Data Assumptions

Most scripts expect the released parquet dataset:

```text
dataset/vex/v1_0_release/v1_0_stable.parquet
```

The expected core columns are:

| Column | Purpose |
|---|---|
| `grading_id` | Unique grading instance identifier |
| `question` | Question text |
| `answer` | Student answer text |
| `grade` | Ordinal target grade on the normalized five-point scale |
| `split` | Data split, typically `train` or `test` |
| `label_type` | Label provenance, typically `silver` or `gold` |

The normalized grade scale is:

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

Training is generally performed on silver-labeled training rows. Evaluation and
prediction export target the gold test rows.

## Model Families

### 1. TF-IDF Baselines

Location:

- [classical_asag/tfidf/tfidf.py](classical_asag/tfidf/tfidf.py)

This script implements sparse text baselines using `TfidfVectorizer` features
and a `Ridge` regressor.

Implemented variants include:

| Variant | Description | Output column |
|---|---|---|
| `v1_answer_word_unigram` | Answer-only word unigrams | `pred_tfidf_v1_answer_word_unigram` |
| `v2_answer_word_uni_bigram` | Answer-only word unigrams and bigrams | `pred_tfidf_v2_answer_word_uni_bigram` |
| `v3_qa_concat_word_uni_bigram` | Concatenated question-answer word features | `pred_tfidf_v3_qa_concat_word_uni_bigram` |
| `v4_question_and_answer_separate` | Separate question and answer vectorizers | `pred_tfidf_v4_question_and_answer_separate` |
| `v5_answer_char_3_5` | Answer-only character `char_wb` n-grams | `pred_tfidf_v5_answer_char_3_5` |
| `v6_mixed_word_char_qa` | Mixed question word and answer word/character features | `pred_tfidf_v6_mixed_word_char_qa` |

The VEX metric configuration currently consumes the `v1`, `v4`, and `v5`
prediction columns.

### 2. Prior and Template Baselines

Locations:

- [prior_template/global_prior/global_prior.py](prior_template/global_prior/global_prior.py)
- [prior_template/qa_overlap_template/QA_overlap_template.py](prior_template/qa_overlap_template/QA_overlap_template.py)

`global_prior.py` learns a single constant grade from the training split and
snaps it to the valid grade scale.

Output column:

```text
grade_prior_global
```

`QA_overlap_template.py` implements a deterministic baseline using empty-answer
detection, answer length, and token overlap between question and answer.

Output column:

```text
grade_prior_template_overlap
```

### 3. Encoder Regressors

Locations:

- [encoder/bert/](encoder/bert/)
- [encoder/deberta/](encoder/deberta/)

The encoder pipelines treat grading as scalar regression over the normalized
five-point scale. Inputs are constructed as a single text sequence:

```text
QUESTION: <question> [SEP] STUDENT_ANSWER: <answer>
```

BERT scripts:

- [encoder/bert/prepare_bert.py](encoder/bert/prepare_bert.py)
- [encoder/bert/train_bert.py](encoder/bert/train_bert.py)
- [encoder/bert/infer_bert.py](encoder/bert/infer_bert.py)

mDeBERTa scripts:

- [encoder/deberta/prepare_deberta.py](encoder/deberta/prepare_deberta.py)
- [encoder/deberta/train_deberta.py](encoder/deberta/train_deberta.py)
- [encoder/deberta/infer_deberta.py](encoder/deberta/infer_deberta.py)

Current released training configurations:

| Family | Model | Epochs | LR | Per-device batch | Output column |
|---|---|---:|---:|---:|---|
| BERT | `bert-base-german-cased` | 2 | `2e-5` | 8 | `grade_bert_ft` |
| mDeBERTa | `microsoft/mdeberta-v3-base` | 2 | `5e-6` | 4 | `grade_mdeberta_ft` |

Inference scripts can also run a base/random-head mode and write columns such as
`grade_bert_base` or `grade_mdeberta_base`, but the current VEX metric config
uses the fine-tuned encoder columns.

The inference exports include:

```text
grade_<model>_<mode>_raw
grade_<model>_<mode>_clipped
grade_<model>_<mode>
```

Important note: the lightweight `prepare_*` helper scripts are convenience
download utilities. Verify model identifiers in the train and inference scripts
before downloading weights.

### 4. Joint LLM Graders

Locations:

- [joint_models/gemma/](joint_models/gemma/)
- [joint_models/llama/](joint_models/llama/)

These scripts frame grading as instruction-following generation over
chat-formatted examples. The model is asked to grade a question-answer pair and
return structured output.

Gemma scripts:

- [joint_models/gemma/prepare_gemma.py](joint_models/gemma/prepare_gemma.py)
- [joint_models/gemma/train_gemma.py](joint_models/gemma/train_gemma.py)
- [joint_models/gemma/infer_gemma.py](joint_models/gemma/infer_gemma.py)

The released Gemma setup uses:

- `google/gemma-4-E4B-it` as local base model path,
- Unsloth for loading and fine-tuning,
- LoRA adaptation,
- 2 epochs on silver labels,
- JSON-style grade targets during training.

The inference script stores raw generations and prompt-fit metadata, including:

```text
raw_prediction_gemma_e4
gemma_e4_input_tokens
gemma_e4_prompt_was_truncated
gemma_e4_generation_hit_limit
gemma_e4_prompt_fit_status
```

The `joint_models/llama/` folder is preserved as part of the experimental
workspace. The public files in that directory currently mirror Gemma-oriented
filenames and configuration, so it should be interpreted as a release artifact
rather than a clean standalone Llama pipeline.

## Credentialed Model Downloads

The [hf_key/](hf_key/) folder documents the local Hugging Face token helper used
for gated model downloads.

Do not commit real tokens. Prefer the `HF_TOKEN` environment variable for shared
or automated setups.

## VEX Metric Prediction Columns

The current VEX metric configuration consumes these columns from
`dataset/additional/vex_metric_dataset/merged_model_predictions.parquet`:

```text
new_grade_deepseek/deepseek-v3.2-thinking
new_grade_deepseek/deepseek-v3.2
new_grade_google/gemini-2.5-pro
new_grade_anthropic/claude-sonnet-4.6
new_grade_openai/gpt-5.4
new_grade_llama32_3b_base
new_grade_gemma_e4_base
new_grade_llama32_3b_ft
new_grade_gemma_e4_ft
grade_bert_ft
grade_mdeberta_ft
grade_prior_global
grade_prior_template_overlap
pred_tfidf_v5_answer_char_3_5
pred_tfidf_v1_answer_word_unigram
pred_tfidf_v4_question_and_answer_separate
```

Some result scripts intentionally filter subsets of these columns. For example,
the significance script excludes prior/template and encoder columns so the
paper-facing significance comparisons focus on LLM and TF-IDF systems.

## Practical Usage Notes

Before running scripts:

1. Review path constants at the top of the script.
2. Confirm that `dataset/vex/v1_0_release/v1_0_stable.parquet` exists locally.
3. Verify GPU availability for encoder and joint-model experiments.
4. Check output column names before merging predictions into downstream evaluation files.
5. Treat `requirements.txt` as a practical starting point rather than a fully pinned environment.

Core dependencies used across the folder include:

- `pandas`
- `numpy`
- `pyarrow`
- `scikit-learn`
- `torch`
- `transformers`
- `datasets`
- `huggingface_hub`
- `trl`
- `unsloth`
- `scipy`
- `tqdm`

## Release Scope

This directory prioritizes transparent experimental traceability over polished
library design. Some scripts preserve intermediate or imperfect workspace
artifacts; this README documents those cases explicitly so the release remains
faithful to the project state.
