from huggingface_hub import snapshot_download
import os

# Konfiguration
# Für DeBERTa: "microsoft/deberta-v3-large"
# Für Mistral: "mistralai/Mistral-Nemo-Instruct-2407"
MODEL_ID = "google-bert/bert-base-uncased"
SAVE_DIR = "./model_weights/bert-base-uncased"

def download_model(model_id, save_path):
    print(f"--- Starte Download für: {model_id} ---")
    
    # Erstelle das Verzeichnis, falls es nicht existiert
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Lädt alle notwendigen Dateien (Weights, Config, Tokenizer) herunter
    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=save_path,
            local_dir_use_symlinks=False, # Kopiert die echten Dateien in den Ordner
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Spart Platz: Keine Flax/TF/Rust Gewichte
        )
        print(f"\nErfolgreich heruntergeladen nach: {path}")
    except Exception as e:
        print(f"Fehler beim Download: {e}")

if __name__ == "__main__":
    download_model(MODEL_ID, SAVE_DIR)