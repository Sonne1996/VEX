# Raw Source

This folder documents the raw institutional source used at the beginning of the
VEX dataset lineage.

The original SQLite database is not part of the public release. The first stable
public artifact is the exported parquet produced in `v0_1_sqlite_export/`.

## Role in the Pipeline

The raw source contains the original institutional answer and question data. In
the public repository, this stage is represented only by documentation and
helper code, because the raw database may contain institutional context that is
not released directly.

The public lineage therefore starts at:

```text
dataset/vex/v0_1_sqlite_export/
```

## Notes

- Do not expect the raw SQLite file to be present in a clean public checkout.
- Use the stable parquet artifacts in later version folders for reproduction.
- The released processing path is documented in [../README.md](../README.md).
