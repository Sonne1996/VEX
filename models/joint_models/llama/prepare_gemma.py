#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

# Set HF_TOKEN, or keep a local untracked token file at models/hf_key/hf_api_key.txt.
HF_TOKEN_FILE = PROJECT_ROOT / "models" / "hf_key" / "hf_api_key.txt"

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN and HF_TOKEN_FILE.exists():
    HF_TOKEN = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()

# Das ist das offizielle Modell, das zu deinem Pfad
# model_weights/gemma-4-E4B-it passt.
MODEL_ID = "google/gemma-4-E4B-it"

SAVE_DIR = (BASE_DIR / "model_weights" / "gemma-4-E4B-it").resolve()

# Optional:
# Falls du stattdessen die Unsloth-4bit-Version direkt lokal speichern willst,
# dann nimm:
# MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
# SAVE_DIR = (BASE_DIR / "model_weights" / "gemma-4-E4B-it-unsloth-bnb-4bit").resolve()


# =========================================================
# DOWNLOAD
# =========================================================

def ensure_token() -> str:
    token = HF_TOKEN.strip()
    if not token:
        raise ValueError(
            "Kein Hugging Face Token gefunden. "
            "Setze HF_TOKEN als Umgebungsvariable oder trage ihn im Script ein."
        )
    return token


def download_model(model_id: str, save_dir: Path, token: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starte Download für: {model_id}")
    print(f"Zielordner: {save_dir}")

    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=str(save_dir),
            token=token,
            local_dir_use_symlinks=False,
            ignore_patterns=[
                "*.bin",
                "*.pth",
                "*.pt",
                "*.msgpack",
                "*.h5",
                "original/*",
            ],
            resume_download=True,
        )
    except Exception as e:
        print(f"Fehler beim Download: {e}")
        print("Prüfe:")
        print("- ob dein HF Token korrekt ist")
        print("- ob du auf Hugging Face die Gemma-Nutzungsbedingungen akzeptiert hast")
        raise

    print("Download abgeschlossen.")
    print(f"Gespeichert in: {path}")


def main() -> None:
    token = ensure_token()
    download_model(MODEL_ID, SAVE_DIR, token)


if __name__ == "__main__":
    main()
