#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# do at the end verify_checksums.py --write

"""
Create or verify SHA256 checksums for the VEX dataset release files.

Default mode:
    Verify all files listed in checksums.sha256.

Write mode:
    Create or overwrite checksums.sha256 for the expected release files.

Usage:
    python verify_checksums.py
    python verify_checksums.py --write
    python verify_checksums.py --root /path/to/dataset
    python verify_checksums.py --manifest checksums.sha256
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


DEFAULT_MANIFEST = "checksums.sha256"


EXPECTED_FILES: tuple[Path, ...] = (
    Path("vex/raw/raw.sqlite"),
    Path("vex/v1_0_release/v1_0_stable.parquet"),
    Path("additional/audit_dataset/audit_dataset.parquet"),
    Path("additional/feedback_dataset/merged_feedback_long.parquet"),
    Path("additional/teacher_selection_dataset/gold_with_all_models.parquet"),
    Path("additional/vex_metric_dataset/merged_model_predictions.parquet"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or verify SHA256 checksums for the VEX dataset release."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset root directory. Default: directory containing this script.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(DEFAULT_MANIFEST),
        help=f"Checksum manifest path. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write a new checksum manifest instead of verifying.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if extra release-like files are found outside EXPECTED_FILES.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def resolve_manifest_path(root: Path, manifest: Path) -> Path:
    if manifest.is_absolute():
        return manifest
    return root / manifest


def write_manifest(root: Path, manifest_path: Path) -> int:
    missing_files: list[Path] = []

    lines: list[str] = []

    print("=" * 100)
    print("WRITING SHA256 MANIFEST")
    print("=" * 100)
    print(f"Root:     {root}")
    print(f"Manifest: {manifest_path}")
    print()

    for relative_path in EXPECTED_FILES:
        absolute_path = root / relative_path

        if not absolute_path.exists():
            missing_files.append(relative_path)
            print(f"[MISSING] {relative_path}")
            continue

        if not absolute_path.is_file():
            missing_files.append(relative_path)
            print(f"[NOT FILE] {relative_path}")
            continue

        digest = sha256_file(absolute_path)
        lines.append(f"{digest}  {relative_path.as_posix()}")
        print(f"[OK]      {relative_path}")

    if missing_files:
        print()
        print("ERROR: Cannot write complete manifest because files are missing:")
        for path in missing_files:
            print(f"  - {path}")
        return 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"Checksum manifest written to: {manifest_path}")
    return 0


def read_manifest(manifest_path: Path) -> dict[Path, str]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Checksum manifest not found: {manifest_path}\n"
            "Create it with:\n"
            "  python verify_checksums.py --write"
        )

    expected: dict[Path, str] = {}

    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            raise ValueError(
                f"Invalid manifest line {line_number}: {raw_line!r}"
            )

        digest, path_text = parts
        relative_path = Path(path_text.strip())

        if len(digest) != 64:
            raise ValueError(
                f"Invalid SHA256 digest on line {line_number}: {digest!r}"
            )

        expected[relative_path] = digest.lower()

    return expected


def verify_manifest(root: Path, manifest_path: Path) -> int:
    print("=" * 100)
    print("VERIFYING SHA256 CHECKSUMS")
    print("=" * 100)
    print(f"Root:     {root}")
    print(f"Manifest: {manifest_path}")
    print()

    try:
        expected = read_manifest(manifest_path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    if not expected:
        print("ERROR: Checksum manifest is empty.")
        return 1

    failed = False

    for relative_path, expected_digest in expected.items():
        absolute_path = root / relative_path

        if not absolute_path.exists():
            print(f"[MISSING] {relative_path}")
            failed = True
            continue

        if not absolute_path.is_file():
            print(f"[NOT FILE] {relative_path}")
            failed = True
            continue

        actual_digest = sha256_file(absolute_path)

        if actual_digest == expected_digest:
            print(f"[OK]      {relative_path}")
        else:
            print(f"[FAILED]  {relative_path}")
            print(f"          expected: {expected_digest}")
            print(f"          actual:   {actual_digest}")
            failed = True

    print()

    if failed:
        print("Checksum verification FAILED.")
        return 1

    print("Checksum verification PASSED.")
    return 0


def find_release_like_files(root: Path) -> set[Path]:
    suffixes = {".parquet", ".sqlite", ".sqlite3", ".db"}
    found: set[Path] = set()

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if ".git" in path.parts:
            continue

        if path.suffix.lower() in suffixes:
            found.add(path.relative_to(root))

    return found


def strict_check(root: Path) -> int:
    expected = set(EXPECTED_FILES)
    found = find_release_like_files(root)

    extra = sorted(found - expected)

    if not extra:
        return 0

    print()
    print("STRICT CHECK FAILED: Extra release-like files found:")
    for path in extra:
        print(f"  - {path}")
    print()
    print("Either remove these files or add them intentionally to EXPECTED_FILES.")
    return 1


def main() -> int:
    args = parse_args()

    root: Path = args.root.resolve()
    manifest_path: Path = resolve_manifest_path(root, args.manifest)

    if args.write:
        status = write_manifest(root, manifest_path)
    else:
        status = verify_manifest(root, manifest_path)

    if status != 0:
        return status

    if args.strict:
        strict_status = strict_check(root)
        if strict_status != 0:
            return strict_status

    return 0


if __name__ == "__main__":
    sys.exit(main())