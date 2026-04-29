# VEX: Virtual Exam Benchmark for Automatic Short Answer Grading

<p align="center">
  A research release for <b>Automatic Short Answer Grading (ASAG)</b> with
  a real-world dataset, human annotation workflows, model baselines, and
  a virtual-exam evaluation pipeline.
</p>

<p align="center">
  <a href="./dataset">Dataset</a> |
  <a href="./models">Models</a> |
  <a href="./vex_metric">VEX Metric</a> |
  <a href="./annotation">Annotation</a>
</p>

## News

- `2026-04`: Initial repository cleanup and release documentation for the public VEX research snapshot.
- `2026-04`: Added dataset release metadata, changelog, and Croissant metadata for the stable VEX dataset line.
- `2026-04`: Added model-family documentation and task-specific dataset cards for the accompanying benchmark assets.

## Overview

This repository contains the research release for **VEX**, a benchmark and evaluation framework for
**Automatic Short Answer Grading (ASAG)**.

The project combines four parts:

- a real-world short-answer dataset release,
- human annotation and audit workflows,
- baseline and neural grading models,
- a virtual-exam evaluation pipeline for item-level and exam-level benchmarking.

The main dataset release currently centers on **ASAG2026**, which contains approximately **31,500 student answers**
with structured metadata, human and silver labels, model predictions, and derived resources for feedback and evaluation studies.

## What Is New in VEX?

### 1. A real-world ASAG dataset release

VEX is built around a real educational short-answer corpus rather than a synthetic benchmark.
The dataset release includes question text, student answers, grades, metadata, and derived artifacts for downstream experiments.

### 2. Human-grounded gold evaluation data

The repository documents the dual-annotation, consensus, and manual-audit workflow used to create a frozen gold subset for trustworthy evaluation.

### 3. Multiple model families in one release

The modeling code spans:

- TF-IDF baselines,
- prior and template baselines,
- encoder-only Transformer regressors,
- joint LLM-based grading and feedback models.

### 4. Virtual-exam evaluation instead of item-only reporting

Beyond standard item-level metrics, VEX evaluates models in a **virtual exam** setting, where predictions are aggregated over sampled exams and compared at the student outcome level.

This is intended to reflect educational use more closely than isolated answer-level scoring alone.

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
├── results/
└── vex_metric/
```

## Repository Guide

### Dataset

The dataset release lives in [dataset/](./dataset).

It includes:

- the main VEX dataset lineage from raw export to release-ready tables,
- metadata such as the dataset changelog and license,
- additional derived datasets for audit analysis, feedback evaluation, teacher selection, and VEX metric experiments.

Start here:

- [dataset/README.md](./dataset/README.md)
- [dataset/vex/README.md](./dataset/vex/README.md)

### Annotation

The annotation workflows live in [annotation/](./annotation).

These folders document:

- how the initial gold subset was sampled,
- how dual annotation and consensus were performed,
- how audit decisions were incorporated,
- how the feedback-analysis subset was sampled.

Start here:

- [annotation/dataset/README.md](./annotation/dataset/README.md)
- [annotation/feedback/README.md](./annotation/feedback/README.md)

### Models

The released model code lives in [models/](./models).

This folder contains the baseline and experimental grading pipelines used throughout the project, including:

- sparse TF-IDF regressors,
- simple prior/template baselines,
- BERT and mDeBERTa regressors,
- instruction-tuned LLM graders for grade and feedback generation.

Start here:

- [models/README.md](./models/README.md)

### VEX Metric

The virtual-exam evaluation pipeline lives in [vex_metric/](./vex_metric).

It creates reproducible sampled exam environments, builds joined evaluation tables, and computes both item-level and exam-level metrics.

Start here:

- [vex_metric/README.md](./vex_metric/README.md)

## Quick Start

### 1. Inspect the dataset release

Review the dataset documentation first:

```bash
cd dataset
```

Key entry points:

- `dataset/README.md`
- `dataset/vex/README.md`
- `dataset/vex/metadata/CHANGE_LOG.md`

### 2. Inspect or run baseline models

Model scripts are organized by family under `models/`.
Most scripts expect a local parquet release file and use in-file path constants that should be checked before execution.

### 3. Run the virtual-exam evaluation

From the repository root:

```bash
cd vex_metric
python run_vex.py
```

The evaluation pipeline creates a reproducible virtual test environment and writes metric reports under `vex_metric/vex_test_env/`.

## Release Scope

This repository is a **research release**, not a packaged production framework.

The goal is to make the project inspectable and reproducible by releasing:

- the dataset lineage and metadata,
- the annotation methodology,
- the modeling scripts used in experiments,
- the evaluation pipeline used for virtual-exam benchmarking.

Some directories intentionally preserve the original experimental workspace structure. Where scripts or helpers are not fully polished, the corresponding README files document those limitations explicitly.

## Citation

Citation details will be added after paper acceptance.

## License

Code licensing is described in the repository [LICENSE](./LICENSE).

Dataset licensing for the VEX release is documented separately in:

- [dataset/vex/metadata/DATA_LICENSE.md](./dataset/vex/metadata/DATA_LICENSE.md)

## Acknowledgment

If you use this repository, please cite the accompanying paper once bibliographic details are available and reference the released VEX dataset and evaluation framework accordingly.
