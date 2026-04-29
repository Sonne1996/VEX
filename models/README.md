# Models

This directory contains the modeling code released with **VEX**, our benchmark and evaluation framework for automatic short-answer grading (ASAG). The release is organized by model family and reflects the experimental workspace used to generate prediction columns consumed by the VEX evaluation pipeline.

The codebase covers four families of graders:

1. Classical sparse baselines based on TF-IDF features.
2. Lightweight non-neural priors and rule-based templates.
3. Encoder-only Transformer regressors.
4. Joint instruction-following language models for grading and feedback generation.

The scripts in this folder are intended as a research release rather than a polished training library. Configuration is stored directly inside each script, and paths should be reviewed before running experiments.

## Directory Layout

```text
models/
├── classical_asag/
│   └── tfidf/
├── hf_key/
├── joint_models/
│   ├── gemma/
│   └── llama/
├── prior_template/
│   ├── global_prior/
│   └── qa_overlap_template/
└── transformers/
    ├── bert/
    └── deberta/
```

## Shared Data Assumptions

Across model families, the scripts expect a released parquet dataset derived from the VEX data pipeline, typically configured as:

```text
dataset/vex/v1_0_release/v1_0_stable.parquet
```

The exact path is specified near the top of each script and should be verified before execution.

The released code assumes a table with at least the following columns:

| Column | Purpose |
|---|---|
| `grading_id` | Unique grading instance identifier |
| `question` | Question text |
| `answer` | Student answer text |
| `grade` | Ordinal target grade on the normalized five-point scale |
| `split` | Data split, typically `train` or `test` |
| `label_type` | Label provenance, typically `silver` or `gold` |

The grade scale used throughout this directory is:

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

In the released experiments, training is generally performed on the `silver` portion of the `train` split, while evaluation and prediction export target the `gold` portion of the `test` split.

## Model Families

### 1. Classical ASAG Baselines

Location: [classical_asag/tfidf/tfidf.py](classical_asag/tfidf/tfidf.py#L1)

This script implements several sparse text baselines using `TfidfVectorizer` features and a `Ridge` regressor. It trains on the silver-labeled training portion and writes predictions for the test portion back into a parquet file.

Implemented variants:

| Variant | Description | Output column |
|---|---|---|
| `v1_answer_word_unigram` | Answer-only word unigrams | `pred_tfidf_v1_answer_word_unigram` |
| `v2_answer_word_uni_bigram` | Answer-only word unigrams and bigrams | `pred_tfidf_v2_answer_word_uni_bigram` |
| `v3_qa_concat_word_uni_bigram` | Concatenated question-answer text with word unigrams and bigrams | `pred_tfidf_v3_qa_concat_word_uni_bigram` |
| `v4_question_and_answer_separate` | Separate question and answer vectorizers, then concatenation | `pred_tfidf_v4_question_and_answer_separate` |
| `v5_answer_char_3_5` | Answer-only character `char_wb` n-grams of width 3-5 | `pred_tfidf_v5_answer_char_3_5` |
| `v6_mixed_word_char_qa` | Mixed question word features and answer word/character features | `pred_tfidf_v6_mixed_word_char_qa` |

Predictions are rounded to the nearest valid grade on the five-point scale and saved to `tfidf_results.parquet`.

### 2. Prior and Template Baselines

Location:

- [prior_template/global_prior/global_prior.py](prior_template/global_prior/global_prior.py#L1)
- [prior_template/qa_overlap_template/QA_overlap_template.py](prior_template/qa_overlap_template/QA_overlap_template.py#L1)

These scripts provide simple interpretable baselines.

`global_prior.py` learns a single constant grade from the training split by averaging observed training labels and snapping the result to the nearest allowed grade. The exported prediction column is:

```text
grade_prior_global
```

`QA_overlap_template.py` implements a deterministic rule-based baseline using:

- empty-answer detection,
- answer length,
- token overlap between question and answer.

The exported prediction column is:

```text
grade_prior_template_overlap
```

These baselines are intentionally simple and serve as low-cost reference points for the benchmark.

### 3. Encoder-Only Transformer Regressors

Location:

- [transformers/bert/](transformers/bert/)
- [transformers/deberta/](transformers/deberta/)

The Transformer pipelines treat grading as scalar regression over the five-point normalized scale. Both families build a single encoder input of the form:

```text
QUESTION: <question> [SEP] STUDENT_ANSWER: <answer>
```

Training scripts:

- [transformers/bert/train_bert.py](transformers/bert/train_bert.py#L1)
- [transformers/deberta/train_deberta.py](transformers/deberta/train_deberta.py#L1)

Download helpers:

- [transformers/bert/prepare_bert.py](transformers/bert/prepare_bert.py#L1)
- [transformers/deberta/prepare_deberta.py](transformers/deberta/prepare_deberta.py#L1)

Inference scripts:

- [transformers/bert/infer_bert.py](transformers/bert/infer_bert.py#L1)
- [transformers/deberta/infer_deberta.py](transformers/deberta/infer_deberta.py#L1)

Released encoder configurations:

| Family | Base model | Fine-tuning regime | Exported rounded column |
|---|---|---|---|
| BERT | `bert-base-german-cased` | 2 epochs on silver labels | `grade_bert_base` or `grade_bert_ft` |
| mDeBERTa | `microsoft/mdeberta-v3-base` | 2 epochs on silver labels | `grade_mdeberta_base` or `grade_mdeberta_ft` |

For each inference run, the scripts export:

- a raw continuous prediction column,
- a clipped continuous prediction column in `[0, 1]`,
- a rounded five-point prediction column.

For example, the BERT inference script writes columns such as:

```text
grade_bert_base_raw
grade_bert_base_clipped
grade_bert_base
```

or

```text
grade_bert_ft_raw
grade_bert_ft_clipped
grade_bert_ft
```

The training scripts save checkpoints during training and also export a `final_model/` directory for downstream inference.

Important release note:

The download helper scripts should be treated as convenience utilities, not as perfectly aligned canonical configuration files. In the current release:

- `prepare_bert.py` downloads `google-bert/bert-base-uncased`, while the training and inference scripts are configured around `bert-base-german-cased`.
- `prepare_deberta.py` downloads `microsoft/deberta-v3-large`, while the training and inference scripts are configured around `microsoft/mdeberta-v3-base`.

Users reproducing the reported model family behavior should therefore verify the model identifiers in the training and inference scripts before downloading weights.

### 4. Joint LLM Graders

Location:

- [joint_models/gemma/](joint_models/gemma/)
- [joint_models/llama/](joint_models/llama/)

These scripts frame grading as instruction-following generation. The training setup uses supervised fine-tuning over chat-formatted examples, where the model is asked to grade a question-answer pair and return JSON.

Gemma training and preparation:

- [joint_models/gemma/prepare_gemma.py](joint_models/gemma/prepare_gemma.py#L1)
- [joint_models/gemma/train_gemma.py](joint_models/gemma/train_gemma.py#L1)
- [joint_models/gemma/infer_gemma.py](joint_models/gemma/infer_gemma.py#L1)

The released Gemma configuration uses:

- `google/gemma-4-E4B-it` as the base chat model,
- Unsloth for loading and fine-tuning,
- LoRA adaptation,
- training on silver labels with a JSON target containing only the grade.

At inference time, the prompt asks the model to return both:

- a grade on the five-point normalized scale,
- a short constructive feedback message in the student answer language.

Important release note:

The current inference script stores raw model generations and prompt-fit metadata, but does not itself parse those generations into finalized VEX-compatible prediction columns. In other words, `infer_gemma.py` is presently a raw-generation export script, not a finished metric-ready prediction writer.

The Gemma inference output includes columns such as:

```text
raw_prediction_gemma_e4
gemma_e4_input_tokens
gemma_e4_prompt_was_truncated
gemma_e4_generation_hit_limit
gemma_e4_prompt_fit_status
```

The training script saves checkpoints in `outputs_silver_grade_only_gemma_e4/`, and the released inference configuration points to a specific checkpoint path.

#### Note on `joint_models/llama/`

The `llama/` directory is included to preserve the released experimental workspace, but the files currently mirror the Gemma-oriented naming and configuration. In particular, the public scripts in this folder still reference Gemma-specific filenames, paths, and model identifiers. Readers should therefore interpret `joint_models/llama/` as a release artifact rather than a clean standalone Llama pipeline.

## Credentialed Model Downloads

The `hf_key/` folder is a local helper location for gated Hugging Face downloads used by the joint LLM scripts. The release contains:

- [hf_key/hf_api_key.example.txt](hf_key/hf_api_key.example.txt)
- [hf_key/README.md](hf_key/README.md#L1)

For public or shared deployments, users should keep access tokens local and avoid committing real credentials to version control.

## Practical Usage Notes

This directory is best understood as a reproducibility release rather than a packaged toolkit. A few points are worth checking before running any script:

1. Review path constants at the top of each file.
2. Confirm that the expected parquet release file exists locally.
3. Verify GPU availability for the larger Transformer and Gemma experiments.
4. Inspect model output column names before merging predictions into downstream evaluation tables.
5. Treat the repository-level `requirements.txt` as a practical starting point; dependency versions are not fully pinned in the current release.

From the imports used across scripts, the core Python stack includes:

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

## Relation to the VEX Evaluation Pipeline

The VEX metric pipeline in [vex_metric/](../vex_metric/) expects prediction columns with stable names. The most directly aligned model outputs in this directory are:

- `pred_tfidf_v1_answer_word_unigram`
- `pred_tfidf_v4_question_and_answer_separate`
- `pred_tfidf_v5_answer_char_3_5`
- `grade_prior_global`
- `grade_prior_template_overlap`
- `grade_bert_base`
- `grade_bert_ft`
- `grade_mdeberta_base`
- `grade_mdeberta_ft`

The joint LLM directories document the training and raw inference setup underlying the release, but additional post-processing is required before those outputs can be consumed as fully standardized VEX metric columns.

## Release Scope

This release is designed to make the modeling side of VEX inspectable and reproducible for the research community. It prioritizes:

- transparent baselines,
- script-level experimental traceability,
- explicit model-family separation,
- compatibility with the downstream virtual-exam evaluation pipeline.

Where the public directory structure still exposes intermediate or imperfect experiment artifacts, this README documents them explicitly so the released repository remains faithful to the original research workflow.
