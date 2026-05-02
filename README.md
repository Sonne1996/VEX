# VEX: Virtual Exam Benchmark for Automatic Short Answer Grading

VEX is a research release for Automatic Short Answer Grading (ASAG). It combines
a real-world short-answer dataset, human annotation workflows, model baselines,
and a virtual-exam evaluation pipeline that reports both item-level and
exam-level behavior.

Main entry points:

- [Dataset](./dataset)
- [Models](./models)
- [Results](./results)
- [VEX Metric](./vex_metric)
- [Annotation](./annotation)

## Current Snapshot

- The main public dataset release is stored under `dataset/vex/v1_0_release/`.
- Additional task-specific datasets are stored under `dataset/additional/`.
- Encoder models now live under `models/encoder/`.
- The VEX metric input is `dataset/additional/vex_metric_dataset/merged_model_predictions.parquet`.
- The current VEX evaluation configuration samples 500 virtual exams for test sizes `5`, `10`, `15`, and `20`.
- Generated result folders under `results/` and generated VEX environments under `vex_metric/vex_test_env/` are treated as reproducible outputs and should not be committed.

## Overview

The project combines four parts:

- a curated short-answer dataset release,
- human annotation, consensus, and audit workflows,
- classical, encoder, and joint LLM grading models,
- a virtual-exam evaluation protocol for item-level and exam-level benchmarking.

The dataset is based on university-level short-answer responses. The stable
release contains answer text, question metadata, human and silver labels, split
information, model predictions, and derived artifacts for feedback and
evaluation studies.

## Repository Structure

```text
VEX/
├── annotation/
│   ├── dataset/
│   └── feedback/
├── dataset/
│   ├── additional/
│   └── vex/
├── feedback/
├── models/
│   ├── classical_asag/
│   ├── encoder/
│   ├── joint_models/
│   ├── prior_template/
│   └── hf_key/
├── results/
└── vex_metric/
```

## Dataset

The dataset release lives in [dataset/](./dataset).

Important folders:

- `dataset/vex/`: main raw-to-release dataset lineage.
- `dataset/vex/v1_0_release/`: frozen public release with `v1_0_stable.parquet`, `train.parquet`, and `test.parquet`.
- `dataset/additional/`: derived datasets for audit analysis, feedback studies, teacher selection, and VEX metric evaluation.

Start here:

- [dataset/README.md](./dataset/README.md)
- [dataset/vex/README.md](./dataset/vex/README.md)
- [dataset/additional/README.md](./dataset/additional/README.md)

## Annotation

The annotation workflows live in [annotation/](./annotation).

They document:

- initial gold-subset sampling,
- dual human annotation,
- consensus and manual audit,
- feedback-analysis sampling.

Start here:

- [annotation/dataset/README.md](./annotation/dataset/README.md)
- [annotation/feedback/README.md](./annotation/feedback/README.md)

## Models

The released model code lives in [models/](./models).

The model families are:

- TF-IDF baselines,
- prior and template baselines,
- BERT and mDeBERTa encoder regressors,
- joint LLM graders for grade and feedback generation.

Start here:

- [models/README.md](./models/README.md)

## VEX Metric

The virtual-exam evaluation pipeline lives in [vex_metric/](./vex_metric).

It creates reproducible sampled exam environments, builds joined evaluation
tables, computes item-level metrics on the original held-out input, and computes
exam-level metrics after aggregating question-level scores into virtual exams.

Run the full pipeline from the repository root:

```bash
python vex_metric/run_vex.py
```

Start here:

- [vex_metric/README.md](./vex_metric/README.md)

## Results

Paper-facing analysis scripts live in [results/](./results).

They generate dataset statistics, confusion matrices, significance tests, and
the figure data/plots used for item-vs-exam and grading-scale analyses.

Start here:

- [results/README.md](./results/README.md)

## Release Scope

This repository is a research release, not a packaged production framework.
Some scripts intentionally preserve the experimental workspace structure used
during the project. The README files document those limitations where they
matter for reproduction.

## License

Code licensing is described in [LICENSE](./LICENSE).

Dataset licensing is documented in
[dataset/vex/metadata/DATA_LICENSE.md](./dataset/vex/metadata/DATA_LICENSE.md).
The dataset license also covers the task-specific datasets under
`dataset/additional/`.

## Citation

Citation details will be added after paper acceptance.
