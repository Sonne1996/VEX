# Results

This directory contains paper-facing analysis scripts for the VEX release.

Generated figures, sanity checks, confusion matrices, dataset statistics, and
significance outputs are written into subfolders under `results/`. Those output
folders are reproducible artifacts and should not be committed unless explicitly
needed for a release package. The Python scripts themselves are source files and
should remain versioned.

## Scripts

### `data_set_metrics.py`

Computes core dataset statistics for the VEX release and writes text/CSV outputs
under:

```text
results/dataset_metrics/
```

The script reports statistics separately for:

- the full release dataset,
- the gold subset.

It also reports question-language counts based on question text.

### `confusen_matrix.py`

Creates confusion matrices for item-level and exam-level model comparisons.

Outputs are written under:

```text
results/confusion_matrices/
```

The script consumes the VEX metric dataframe and writes CSV matrices for linear
and Bologna grade scales.

### `create_plot_2_qwk_vs_tau.py`

Creates Figure 2 style plots comparing item-level performance with exam-level
performance.

Current behavior:

- x-axis: item-level QWK,
- y-axis: exam-level QWK,
- separate outputs for absolute linear and Bologna grading,
- exam-level metrics are split by virtual exam size,
- writes separate q5, q10, q15, and q20 plot/data/sanity outputs when those test sizes are present,
- sanity-check text with the exact loaded values.

Outputs are written under:

```text
results/figures_plot_2/
```

### `create_plot_3_tau_vs_acc.py`

Creates Figure 3 style sensitivity plots for exam-level performance as a
function of virtual exam size.

Current behavior:

- uses precomputed exam-level metrics from `vex_metric/vex_test_env/2_metrics/`,
- compares `EL-tau-b`, `EL-Acc`, and `EL-QWK`,
- reports both absolute linear and Bologna variants where configured,
- writes a sanity-check file with plotted values.

Outputs are written under:

```text
results/figures_plot_3/
```

### `create_plot_4_a_b_qwk_linear_qwk_distro.py`

Creates grading-granularity plots for exam-level performance.

Current behavior:

- input: `vex_metric/vex_test_env/4_dataframe/dataframe_env.parquet`,
- writes separate q5, q10, q15, and q20 plot/data/sanity outputs when those test sizes are present,
- evaluates pass/fail and then 2 through 10 ordered grade categories,
- compares absolute threshold and Bologna distribution grading,
- writes per-exam data, model summaries, and sanity-check text.

Outputs are written under:

```text
results/figures_plot_4/
```

### `calc_statistical_significance.py`

Runs q10 statistical significance analyses.

Current behavior:

- input: `vex_metric/vex_test_env/4_dataframe/dataframe_env.parquet`,
- fixed `TEST_SIZE = 10`,
- model ranking is metric-specific,
- `EL-Acc` significance uses exact McNemar tests,
- `EL-QWK` significance uses paired permutation tests,
- QWK is recomputed per virtual exam and then averaged.

The script currently excludes prior/template and encoder columns from the
significance comparison set. Outputs are written under:

```text
results/statistical_significance_q10/
```

## Relation to `vex_metric/`

The VEX metric pipeline computes the reusable metric tables:

```text
vex_metric/vex_test_env/2_metrics/exam_level_precomputed_metrics.parquet
vex_metric/vex_test_env/2_metrics/exam_level_summary_by_test_size.parquet
vex_metric/vex_test_env/2_metrics/exam_level_summary_all.parquet
vex_metric/vex_test_env/4_dataframe/dataframe_env_exam_metrics_wide.parquet
```

Plot scripts should prefer those precomputed files where possible instead of
reimplementing exam-level metrics independently.

## Expected Workflow

From the repository root:

```bash
python vex_metric/run_vex.py
python results/create_plot_2_qwk_vs_tau.py
python results/create_plot_3_tau_vs_acc.py
python results/create_plot_4_a_b_qwk_linear_qwk_distro.py
python results/data_set_metrics.py
```

Run the significance script separately because it is more expensive:

```bash
python results/calc_statistical_significance.py
```

## Notes

- `test_10` under `vex_metric/vex_test_env/3_tests/` means virtual test run number 10, not q10.
- q10 means `test_size = 10`.
- Item-level metrics are computed on the original held-out input parquet, not on `dataframe_env.parquet`, because the virtual-exam dataframe repeats answers across sampled exams.
- Exam-level metrics are computed per virtual exam first and then averaged over runs.
