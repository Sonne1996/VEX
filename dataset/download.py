#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download the VEX dataset release files from Hugging Face.

This script is intended for reviewers and users who clone the repository
without the large dataset files. It downloads the released SQLite and Parquet
files from the Hugging Face dataset repository and places them into the
expected local folder structure.

Expected local structure after running this script:

dataset/
├── additional/
│   ├── audit_dataset/
│   │   └── audit_dataset.parquet
│   ├── feedback_dataset/
│   │   └── merged_feedback_long.parquet
│   ├── teacher_selection_dataset/
│   │   └── gold_with_all_models.parquet
│   └── vex_metric_dataset/
│       └── merged_model_predictions.parquet
└── vex/
    ├── raw/
    │   └── raw.sqlite
    └── v1_0_release/
        └── v1_0_stable.parquet

Usage:
    python download.py

Optional:
    python download.py --repo-id VEX19/VEX
    python download.py --output-dir /path/to/dataset
    python download.py --force
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "VEX19/VEX"


@dataclass(frozen=True)
class DownloadItem:
    """
    Represents one file to download from Hugging Face.

    remote_candidates:
        Possible paths inside the Hugging Face dataset repository.
        The first existing remote path is used.

    local_path:
        Target path relative to the local dataset root.
    """

    remote_candidates: tuple[str, ...]
    local_path: Path


DOWNLOAD_ITEMS: tuple[DownloadItem, ...] = (
    DownloadItem(
        remote_candidates=("vex/raw/raw.sqlite",),
        local_path=Path("vex/raw/raw.sqlite"),
    ),
    DownloadItem(
        # Supports both possible HF layouts.
        # Locally we always place the file into vex/v1_0_release/.
        remote_candidates=(
            "vex/v1_0_release/v1_0_stable.parquet",
            "vex/v1_0/v1_0_stable.parquet",
        ),
        local_path=Path("vex/v1_0_release/v1_0_stable.parquet"),
    ),
    DownloadItem(
        remote_candidates=("additional/audit_dataset/audit_dataset.parquet",),
        local_path=Path("additional/audit_dataset/audit_dataset.parquet"),
    ),
    DownloadItem(
        remote_candidates=("additional/feedback_dataset/merged_feedback_long.parquet",),
        local_path=Path("additional/feedback_dataset/merged_feedback_long.parquet"),
    ),
    DownloadItem(
        remote_candidates=("additional/teacher_selection_dataset/gold_with_all_models.parquet",),
        local_path=Path("additional/teacher_selection_dataset/gold_with_all_models.parquet"),
    ),
    DownloadItem(
        remote_candidates=("additional/vex_metric_dataset/merged_model_predictions.parquet",),
        local_path=Path("additional/vex_metric_dataset/merged_model_predictions.parquet"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download VEX release files from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository ID. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Local dataset root. Default: directory containing this script.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite local files if they already exist.",
    )
    return parser.parse_args()


def download_first_existing_remote(
    *,
    repo_id: str,
    remote_candidates: tuple[str, ...],
    cache_dir: Path | None = None,
) -> tuple[Path, str]:
    """
    Try all candidate remote paths and return the first one that exists.

    Returns:
        downloaded_cache_path, selected_remote_path
    """
    last_error: Exception | None = None

    for remote_path in remote_candidates:
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            return Path(downloaded_path), remote_path
        except Exception as exc:
            last_error = exc

    candidates = "\n".join(f"  - {path}" for path in remote_candidates)
    raise RuntimeError(
        "Could not download any of the following remote candidates:\n"
        f"{candidates}\n\n"
        f"Last error: {last_error}"
    )


def copy_file(source: Path, target: Path, *, force: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        print(f"[SKIP] {target} already exists. Use --force to overwrite.")
        return

    shutil.copy2(source, target)
    print(f"[OK]   {target}")


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("VEX DATASET DOWNLOAD")
    print("=" * 100)
    print(f"HF repo:     {args.repo_id}")
    print(f"Output dir:  {output_dir}")
    print(f"Overwrite:   {args.force}")
    print()

    for item in DOWNLOAD_ITEMS:
        target_path = output_dir / item.local_path

        if target_path.exists() and not args.force:
            print(f"[SKIP] {target_path} already exists. Use --force to overwrite.")
            continue

        print(f"[GET]  {item.local_path}")
        downloaded_cache_path, selected_remote_path = download_first_existing_remote(
            repo_id=args.repo_id,
            remote_candidates=item.remote_candidates,
        )

        print(f"       remote: {selected_remote_path}")
        copy_file(downloaded_cache_path, target_path, force=args.force)

    print()
    print("=" * 100)
    print("DOWNLOAD FINISHED")
    print("=" * 100)
    print()
    print("You can now verify the files with:")
    print("  python verify_checksums.py")


if __name__ == "__main__":
    main()