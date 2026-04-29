#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


# =========================================================
# CONFIG
# =========================================================

# Hugging Face dataset repo.
# Example: "your-username/vex-metric-dataset"
REPO_ID = "YOUR_USERNAME/YOUR_DATASET_REPO"

# Name/path of the SQLite file inside the Hugging Face repo.
# Example: "vex_metric_dataset.sqlite"
FILENAME = "vex_metric_dataset.sqlite"

# Local output path.
OUTPUT_PATH = Path("vex_metric_dataset.sqlite")

# Hugging Face repo type. For datasets, keep this as "dataset".
REPO_TYPE = "dataset"


# =========================================================
# DOWNLOAD
# =========================================================

def download_sqlite_file() -> Path:
    """
    Downloads the SQLite dataset file from Hugging Face and copies it into
    the current dataset folder.

    hf_hub_download returns a cached file path. We copy it to OUTPUT_PATH so
    the repository has a stable local file name.
    """
    print(f"Downloading {FILENAME} from Hugging Face repo {REPO_ID}...")

    cached_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type=REPO_TYPE,
    )

    cached_path = Path(cached_path)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, OUTPUT_PATH)

    print("Download complete.")
    print(f"Cached file: {cached_path.resolve()}")
    print(f"Local file:  {OUTPUT_PATH.resolve()}")

    return OUTPUT_PATH


def main() -> None:
    download_sqlite_file()


if __name__ == "__main__":
    main()