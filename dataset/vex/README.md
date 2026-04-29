# VEX Dataset

This folder contains the main VEX dataset lineage, from the first stable export to the benchmark-facing release versions.

It is the core dataset directory for the NeurIPS release and should be read as the canonical source for:

- dataset version history,
- release metadata,
- main stable parquet artifacts,
- reproducible dataset-construction scripts from the public processing stages onward.

## Structure

```text
vex/
├── metadata/
├── raw/
├── v0_1_sqlite_export/
├── v0_2_cleaning/
├── v0_3_silver_labels/
└── v1_0_release/
```

## Version Overview

### `raw/`

Contains the helper script for downloading the SQLite source database. The raw institutional database itself is not part of the public release.

### `v0_1_sqlite_export/`

First stable export from the institutional SQLite source into a unified parquet format.

### `v0_2_cleaning/`

Cleaning and restructuring stage. This version extracts plain-text question and answer content, merges question metadata such as Bloom level and topic, normalizes gold labels, and creates the canonical `grade`, `label_type`, and `split` fields.

### `v0_3_silver_labels/`

Extends the cleaned dataset with silver-label information used for scalable training and aligns the schema with the downstream modeling pipeline. The result is `v0.3_stable`, the first stable modeling-oriented version.

### `v1_0_release/`

Public release stage used by the released model scripts as `v1_0_stable.parquet`.

## Metadata

The release metadata is stored in [metadata/](C:/Git/Bachelor/VEX/dataset/vex/metadata):

- [CHANGE_LOG.md](C:/Git/Bachelor/VEX/dataset/vex/metadata/CHANGE_LOG.md:1)
- [DATA_LICENSE.md](C:/Git/Bachelor/VEX/dataset/vex/metadata/DATA_LICENSE.md:1)
- [croissant_metadata.json](C:/Git/Bachelor/VEX/dataset/vex/metadata/croissant_metadata.json:1)

These files describe the version transitions, release license, and machine-readable dataset metadata.

## Reproducibility Scope

The fully raw institutional source database is not public. For that reason:

- `v0_1` is the first stable public artifact in the lineage,
- the public release includes the transformation scripts from `v0_2` onward,
- the later stable versions can be understood and reproduced from the released processing logic and metadata.

This matches the note in [v0_1_sqlite_export/README.md](C:/Git/Bachelor/VEX/dataset/vex/v0_1_sqlite_export/README.md:1).

## Relation to Other Folders

- The human gold-label workflow is documented in [../../annotation/dataset/README.md](C:/Git/Bachelor/VEX/annotation/dataset/README.md:1).
- The model scripts consume the later stable release files documented in [../../models/README.md](C:/Git/Bachelor/VEX/models/README.md:1).
- The VEX evaluation pipeline consumes the merged prediction dataset documented in [../additional/vex_metric_dataset/README.md](C:/Git/Bachelor/VEX/dataset/additional/vex_metric_dataset/README.md:1).
