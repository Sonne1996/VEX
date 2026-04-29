#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import traceback


def main() -> None:
    """
    Gesamte VEX-Pipeline:

    1. Virtuelle Testumgebung erstellen oder validieren
    2. Dataframe-Umgebung erzeugen
    3. Evaluation ausführen
    """
    print("=" * 100)
    print("STARTING VEX PIPELINE")
    print("=" * 100)

    try:
        print("\n[1/3] Creating or validating virtual test environment...")
        from create_vex_test_env import create_virtual_test_env

        create_virtual_test_env()
        print("[1/3] Virtual test environment step finished.")

        print("\n[2/3] Preparing dataframe environment...")
        from create_dataframe import create_dataframe

        create_dataframe()
        print("[2/3] Data preparation finished.")

        print("\n[3/3] Evaluating prepared dataframe environment...")
        from evaluate_dataframe import main as evaluate_main

        evaluate_main()
        print("[3/3] Evaluation finished.")

        print("\n" + "=" * 100)
        print("VEX PIPELINE FINISHED SUCCESSFULLY")
        print("=" * 100)

    except Exception as exc:
        print("\n" + "=" * 100)
        print("VEX PIPELINE FAILED")
        print("=" * 100)
        print(f"Error: {exc}")
        print("\nFull traceback:\n")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()