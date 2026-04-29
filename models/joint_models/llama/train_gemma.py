#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_PARQUET = "../dataset/v1_0_release/v1_0_stable.parquet"
OUTPUT_DIR = (BASE_DIR / "outputs_silver_grade_only_gemma_e4").resolve()
LOG_FILE = OUTPUT_DIR / "training.log"

# ROBUSTERE WAHL:
# MODEL_NAME = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

# Falls du trotzdem lokal testen willst:
MODEL_DIR = (BASE_DIR / "model_weights" / "gemma-4-E4B-it").resolve()
USE_LOCAL_MODEL = True

QUESTION_COL = "question"
ANSWER_COL = "answer"
GRADE_COL = "grade"
SPLIT_COL = "split"
LABEL_TYPE_COL = "label_type"

TRAIN_SPLIT_VALUE = "train"
TEST_SPLIT_VALUE = "test"

TRAIN_LABEL_TYPE = "silver"
TEST_LABEL_TYPE = "gold"

MAX_SEQ_LENGTH = 1024
RANDOM_SEED = 42

# LoRA / Training
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 2
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
SAVE_STEPS = 250
SAVE_TOTAL_LIMIT = 3

ALLOWED_GRADES = {0.0, 0.25, 0.5, 0.75, 1.0}


# =========================================================
# PROMPT
# =========================================================

PROMPT_INSTRUCTION = (
    "You are an expert grader for short-answer responses in a university-level database systems course.\n"
    "Task:\n"
    "Grade the STUDENT_ANSWER based only on the QUESTION and the correctness of the answer content.\n\n"

    "Score scale:\n"
    "0.0 = incorrect or seriously flawed\n"
    "0.25 = mostly incorrect, with limited relevant content\n"
    "0.5 = partially correct, but incomplete or imprecise\n"
    "0.75 = mostly correct, with only minor issues\n"
    "1.0 = fully correct, precise, and complete\n\n"

    "Important Instructions:\n"
    "The context is a university-level database systems course.\n"
    "Evaluate technical meaning, not fluency or grammar.\n"
    "Accept paraphrases if the meaning is correct.\n"
    "Do not reward answer length by itself.\n"
    "Penalize missing key concepts, misconceptions, contradictions, and vague statements.\n"
    "Use only the five allowed scores.\n"
    "Return only valid JSON.\n"
    "Do not use markdown fences.\n\n"

    "QUESTION:\n"
    "{QUESTION}\n\n"

    "STUDENT_ANSWER:\n"
    "{STUDENT_ANSWER}\n\n"

    "Output Format:\n"
    "{\n"
    '  "grade": <one of [0.0, 0.25, 0.5, 0.75, 1.0]>\n'
    "}\n"
)


# =========================================================
# HELPERS
# =========================================================

def assert_local_model_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Local model directory not found: {path}")

    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    missing = [name for name in required_files if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Local model directory is missing required files: {missing}\n"
            f"Checked path: {path}"
        )

    if not list(path.glob("*.safetensors")):
        raise FileNotFoundError(f"No .safetensors file found in: {path}")


def open_parquet_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Expected parquet file, got: {path}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    return pd.read_parquet(path)


def normalize_grade(value) -> float | None:
    if pd.isna(value):
        return None

    try:
        value = float(value)
    except Exception:
        return None

    if value not in ALLOWED_GRADES:
        return None

    return value


def build_user_prompt(question: str, answer: str) -> str:
    question = "" if pd.isna(question) else str(question).strip()
    answer = "" if pd.isna(answer) else str(answer).strip()

    prompt = PROMPT_INSTRUCTION
    prompt = prompt.replace("{QUESTION}", question)
    prompt = prompt.replace("{STUDENT_ANSWER}", answer)
    return prompt


def build_target_json(grade: float) -> str:
    return json.dumps({"grade": grade}, ensure_ascii=False)


def make_chat_messages(question: str, answer: str, grade: float) -> list[dict[str, str]]:
    user_content = build_user_prompt(question, answer)
    assistant_content = build_target_json(grade)

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def apply_chat_template(messages: list[dict[str, str]], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def prepare_train_dataframe(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    required_cols = [QUESTION_COL, ANSWER_COL, GRADE_COL, SPLIT_COL, LABEL_TYPE_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out[GRADE_COL] = out[GRADE_COL].apply(normalize_grade)

    train_df = out[
        (out[SPLIT_COL] == TRAIN_SPLIT_VALUE) &
        (out[LABEL_TYPE_COL] == TRAIN_LABEL_TYPE) &
        (out[GRADE_COL].notna())
    ].copy()

    if train_df.empty:
        raise ValueError("No usable silver train rows found.")

    train_df["text"] = train_df.apply(
        lambda row: apply_chat_template(
            make_chat_messages(
                question=row[QUESTION_COL],
                answer=row[ANSWER_COL],
                grade=row[GRADE_COL],
            ),
            tokenizer=tokenizer,
        ),
        axis=1,
    )

    return train_df


def prepare_reserved_test_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [QUESTION_COL, ANSWER_COL, GRADE_COL, SPLIT_COL, LABEL_TYPE_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out[GRADE_COL] = out[GRADE_COL].apply(normalize_grade)

    test_df = out[
        (out[SPLIT_COL] == TEST_SPLIT_VALUE) &
        (out[LABEL_TYPE_COL] == TEST_LABEL_TYPE) &
        (out[GRADE_COL].notna())
    ].copy()

    if test_df.empty:
        raise ValueError("No usable gold test rows found.")

    return test_df


def get_latest_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None

    checkpoints = []
    for path in output_dir.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint-"):
            step_str = path.name.replace("checkpoint-", "")
            if step_str.isdigit():
                checkpoints.append((int(step_str), path))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return str(checkpoints[-1][1])


# =========================================================
# CALLBACKS
# =========================================================

class EpochSaveAndFileLoggerCallback(TrainerCallback):
    def __init__(self, log_file: Path):
        self.log_file = Path(log_file)

    def _write(self, message: str) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self._write("Training started.")
        self._write(f"Output dir: {args.output_dir}")
        self._write(f"Max steps: {state.max_steps}")
        self._write(f"Num train epochs: {getattr(args, 'num_train_epochs', 'unknown')}")
        self._write(
            f"Save strategy: {args.save_strategy} | "
            f"save_steps: {getattr(args, 'save_steps', 'n/a')} | "
            f"save_total_limit: {args.save_total_limit}"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero:
            return

        logs = dict(logs or {})
        logs.pop("total_flos", None)

        step = state.global_step
        max_steps = state.max_steps
        epoch = logs.get("epoch", state.epoch)

        parts = [
            f"step={step}/{max_steps}",
            f"epoch={epoch}",
        ]

        if "loss" in logs:
            parts.append(f"loss={logs['loss']}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']}")
        if "grad_norm" in logs:
            parts.append(f"grad_norm={logs['grad_norm']}")
        if "train_loss" in logs:
            parts.append(f"train_loss={logs['train_loss']}")

        self._write(" | ".join(parts))

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self._write(
                f"Epoch ended: epoch={state.epoch}, step={state.global_step}/{state.max_steps}. "
                "Forcing checkpoint save."
            )
        control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self._write(f"Checkpoint saved at global_step={state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        self._write("Training finished.")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Neues Run-Header ins Logfile schreiben
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"NEW RUN STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n")

    df = open_parquet_file(INPUT_PARQUET)

    if USE_LOCAL_MODEL:
        assert_local_model_dir(MODEL_DIR)
        model_name = str(MODEL_DIR)
        print("Using local model directory:")
        print(MODEL_DIR)
    else:
        model_name = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
        print("Using official Unsloth model:")
        print(model_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        local_files_only=USE_LOCAL_MODEL,
    )

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # WICHTIG:
    # NICHT model.config.use_cache = False setzen.
    # Genau das war bei Gemma 4 problematisch.

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=False,   # zuerst stabil testen
        random_state=RANDOM_SEED,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    train_df = prepare_train_dataframe(df, tokenizer)
    test_df = prepare_reserved_test_dataframe(df)

    print("Dataset summary:")
    print(f"Silver train rows: {len(train_df)}")
    print(f"Gold test rows:    {len(test_df)}")
    print("Gold test is reserved and NOT used during training.")

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"Silver train rows: {len(train_df)}\n")
        f.write(f"Gold test rows: {len(test_df)}\n")
        f.write("Gold test is reserved and NOT used during training.\n")

    train_dataset = Dataset.from_pandas(
        train_df[["text"]],
        preserve_index=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        callbacks=[EpochSaveAndFileLoggerCallback(LOG_FILE)],
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            warmup_steps=WARMUP_STEPS,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type="cosine",
            logging_steps=LOGGING_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to="none",
            optim="adamw_torch",
            seed=RANDOM_SEED,
            dataset_num_proc=1,
            packing=False,
            max_grad_norm=1.0,
        ),
    )

    resume_checkpoint = get_latest_checkpoint(OUTPUT_DIR)

    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"Resuming from checkpoint: {resume_checkpoint}\n")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        print("No checkpoint found. Starting training from scratch.")
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write("No checkpoint found. Starting training from scratch.\n")
        trainer.train()

    print("\nTraining finished.")
    print(f"Model/checkpoints saved in: {OUTPUT_DIR}")
    print(f"Training log saved in:      {LOG_FILE}")
    print(f"Gold test rows kept untouched for final evaluation: {len(test_df)}")

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write("Training finished.\n")
        f.write(f"Model/checkpoints saved in: {OUTPUT_DIR}\n")
        f.write(f"Training log saved in: {LOG_FILE}\n")
        f.write(f"Gold test rows kept untouched for final evaluation: {len(test_df)}\n")


if __name__ == "__main__":
    main()