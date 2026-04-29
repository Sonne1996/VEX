#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

INPUT_PARQUET = PROJECT_ROOT / "dataset" / "vex" / "v1_0_release" / "v1_0_stable.parquet"
OUTPUT_PARQUET = BASE_DIR / "gemma_results_ft.parquet"
LOG_FILE = BASE_DIR / "infer_gemma_e4_test_only_ft.log"

QUESTION_COL = "question"
ANSWER_COL = "answer"
ID_COL = "grading_id"
SPLIT_COL = "split"

TEST_SPLIT_VALUE = "test"

ROW_LIMIT = None  # z.B. 20 zum Testen

# -----------------------------------------
# MODEL SWITCH
# -----------------------------------------
USE_FINETUNED_MODEL = True

BASE_MODEL_PATH = BASE_DIR / "model_weights" / "gemma-4-E4B-it"
FINETUNED_MODEL_PATH = BASE_DIR / "outputs_silver_grade_only_gemma_e4" / "checkpoint-6866"

MAX_SEQ_LENGTH = 8192
LOAD_IN_4BIT = True
MODEL_CONTEXT_LIMIT = 8192

RAW_OUTPUT_COL = "raw_prediction_gemma_e4"

INPUT_TOKENS_COL = "gemma_e4_input_tokens"
PROMPT_WAS_TRUNCATED_COL = "gemma_e4_prompt_was_truncated"
GENERATION_HIT_LIMIT_COL = "gemma_e4_generation_hit_limit"
PROMPT_FIT_STATUS_COL = "gemma_e4_prompt_fit_status"

MAX_NEW_TOKENS = 512
DO_SAMPLE = False
TEMPERATURE = 0.0

# -----------------------------------------
# PERFORMANCE SETTINGS
# -----------------------------------------
BATCH_SIZE = 32
MIN_BATCH_SIZE = 1
SAVE_EVERY_N_BATCHES = 20
EMPTY_CACHE_EVERY_N_BATCHES = 10

ENABLE_TF32 = True


# =========================================================
# PROMPT
# =========================================================

PROMPT_INSTRUCTION = (
    "You are an expert evaluator for short-answer responses in higher education.\n\n"

    "The task is set in a university-level database systems course.\n\n"

    "Your task is to evaluate a student's answer to a question and produce:\n"
    "1. a single ordinal score on a 0-1 scale\n"
    "2. a short, constructive feedback message\n\n"

    "Evaluation principles:\n"
    "Judge the answer based on its relevance to the question, semantic correctness, completeness, and precision.\n"
    "Reward answers that address the core meaning of the question clearly and correctly.\n"
    "Penalize answers that are incorrect, vague, incomplete, misleading, or off-topic.\n"
    "Be especially careful with partially correct answers: do not over-reward keyword overlap if the meaning is weak or incomplete.\n"
    "Use a consistent internal standard across all responses.\n"
    "The feedback must be specific to the student's answer and must justify the score.\n"
    "The feedback should be constructive and useful for learning.\n"
    "Do not mention that you are an AI model.\n"
    "Do not include hidden reasoning, step-by-step chains of thought, or extra commentary outside the JSON.\n"
    "The feedback must be written in the same language as the student answer.\n\n"

    "Score scale:\n"
    "0.0 = incorrect or seriously flawed\n"
    "0.25 = mostly incorrect, with limited relevant content\n"
    "0.5 = partially correct, but incomplete or imprecise\n"
    "0.75 = mostly correct, with only minor issues\n"
    "1.0 = fully correct, precise, and complete\n\n"

    "Question:\n"
    "{QUESTION}\n\n"

    "Student Answer:\n"
    "{STUDENT_ANSWER}\n\n"

    "Return ONLY valid JSON in the following format:\n\n"
    "{\n"
    '  "grade": <one of [0.0, 0.25, 0.5, 0.75, 1.0]>,\n'
    '  "feedback": "<2-4 sentences of constructive feedback written in the same language as the student answer>"\n'
    "}"
)


# =========================================================
# HELPERS
# =========================================================

def log_message(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def open_parquet_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Expected parquet file, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_parquet(path)


def save_parquet(path: Path, df: pd.DataFrame) -> None:
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    log_message(f"Saved parquet to: {path.resolve()}")


def build_user_prompt(question: str, answer: str) -> str:
    question = "" if pd.isna(question) else str(question).strip()
    answer = "" if pd.isna(answer) else str(answer).strip()

    prompt = PROMPT_INSTRUCTION
    prompt = prompt.replace("{QUESTION}", question)
    prompt = prompt.replace("{STUDENT_ANSWER}", answer)

    return prompt


def build_chat_messages(question: str, answer: str) -> list[dict[str, str]]:
    user_content = build_user_prompt(question, answer)
    return [{"role": "user", "content": user_content}]


def build_chat_prompt(tokenizer, question: str, answer: str) -> str:
    messages = build_chat_messages(question, answer)

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if rendered is None:
        raise ValueError("apply_chat_template returned None.")

    return rendered


def build_chat_prompts_batch(tokenizer, questions: list[str], answers: list[str]) -> list[str]:
    prompts: list[str] = []
    for question, answer in zip(questions, answers):
        prompts.append(build_chat_prompt(tokenizer, question, answer))
    return prompts


def get_eos_token_id(tokenizer) -> int | None:
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id

    if hasattr(tokenizer, "tokenizer"):
        inner = tokenizer.tokenizer
        if hasattr(inner, "eos_token_id") and inner.eos_token_id is not None:
            return inner.eos_token_id

    return None


def get_pad_token_id(tokenizer) -> int | None:
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id

    if hasattr(tokenizer, "tokenizer"):
        inner = tokenizer.tokenizer
        if hasattr(inner, "pad_token_id") and inner.pad_token_id is not None:
            return inner.pad_token_id

    eos_token_id = get_eos_token_id(tokenizer)
    return eos_token_id


def safe_strip_special_tokens(token_ids: list[int], pad_token_id: int | None, eos_token_id: int | None) -> list[int]:
    cleaned = list(token_ids)

    while cleaned and pad_token_id is not None and cleaned[-1] == pad_token_id:
        cleaned.pop()

    while cleaned and eos_token_id is not None and cleaned[-1] == eos_token_id:
        cleaned.pop()

    return cleaned


# =========================================================
# MODEL LOADING
# =========================================================

def load_model_and_tokenizer():
    model_path = FINETUNED_MODEL_PATH if USE_FINETUNED_MODEL else BASE_MODEL_PATH

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path not found: {Path(model_path).resolve()}")

    log_message(f"Loading model from: {Path(model_path).resolve()}")
    log_message(
        f"MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}, "
        f"MODEL_CONTEXT_LIMIT={MODEL_CONTEXT_LIMIT}, "
        f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}, "
        f"BATCH_SIZE={BATCH_SIZE}"
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


# =========================================================
# TOKEN / CONTEXT CHECK
# =========================================================

def prepare_batch_inputs_with_context_check(
    tokenizer,
    prompts: list[str],
) -> tuple[dict, list[int], list[bool], list[str]]:
    max_prompt_tokens = MODEL_CONTEXT_LIMIT - MAX_NEW_TOKENS

    if max_prompt_tokens <= 0:
        raise ValueError("MODEL_CONTEXT_LIMIT must be larger than MAX_NEW_TOKENS.")

    tokenized_no_trunc = tokenizer(
        text=prompts,
        add_special_tokens=False,
        truncation=False,
    )

    input_token_counts = [len(ids) for ids in tokenized_no_trunc["input_ids"]]

    fit_statuses: list[str] = []
    trunc_flags: list[bool] = []

    for count in input_token_counts:
        if count < max_prompt_tokens * 0.9:
            fit_statuses.append("ok")
            trunc_flags.append(False)
        elif count <= max_prompt_tokens:
            fit_statuses.append("near_limit")
            trunc_flags.append(False)
        else:
            fit_statuses.append("too_long")
            trunc_flags.append(True)

    inputs = tokenizer(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )

    return inputs, input_token_counts, trunc_flags, fit_statuses


# =========================================================
# CORE INFERENCE
# =========================================================

def generate_batch_predictions(
    model,
    tokenizer,
    questions: list[str],
    answers: list[str],
    row_identifiers: list[str],
) -> list[dict]:
    prompts = build_chat_prompts_batch(tokenizer, questions, answers)

    inputs, input_token_counts, prompt_was_truncated_list, fit_status_list = prepare_batch_inputs_with_context_check(
        tokenizer=tokenizer,
        prompts=prompts,
    )

    max_prompt_tokens = MODEL_CONTEXT_LIMIT - MAX_NEW_TOKENS

    for row_identifier, input_token_count, fit_status, prompt_was_truncated in zip(
        row_identifiers, input_token_counts, fit_status_list, prompt_was_truncated_list
    ):
        if fit_status == "near_limit":
            log_message(
                f"[WARNING] row={row_identifier} prompt near context limit "
                f"(input_tokens={input_token_count}, max_prompt_tokens={max_prompt_tokens})"
            )
        if prompt_was_truncated:
            log_message(
                f"[WARNING] row={row_identifier} prompt exceeded context limit and was truncated "
                f"(input_tokens={input_token_count}, max_prompt_tokens={max_prompt_tokens})"
            )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    eos_token_id = get_eos_token_id(tokenizer)
    pad_token_id = get_pad_token_id(tokenizer)

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        use_cache=True,
    )

    if DO_SAMPLE:
        generate_kwargs["temperature"] = TEMPERATURE

    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    elif eos_token_id is not None:
        generate_kwargs["pad_token_id"] = eos_token_id

    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)

    # WICHTIG:
    # Bei gepaddeten Batch-Inputs muss ab der gemeinsamen gepaddeten
    # Eingabelänge gesliced werden, nicht mit der individuellen Prompt-Länge.
    padded_input_width = int(inputs["input_ids"].shape[1])

    results: list[dict] = []

    for i in range(outputs.shape[0]):
        generated_ids = outputs[i, padded_input_width:].tolist()

        cleaned_generated_ids = safe_strip_special_tokens(
            token_ids=generated_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        raw_output = tokenizer.decode(cleaned_generated_ids, skip_special_tokens=True).strip()

        generated_new_tokens = len(cleaned_generated_ids)
        generation_hit_limit = generated_new_tokens >= MAX_NEW_TOKENS

        if generation_hit_limit:
            log_message(
                f"[WARNING] row={row_identifiers[i]} generation hit MAX_NEW_TOKENS "
                f"(generated_new_tokens={generated_new_tokens}, max_new_tokens={MAX_NEW_TOKENS})"
            )

        results.append(
            {
                "raw_output": raw_output,
                "input_tokens": int(input_token_counts[i]),
                "prompt_was_truncated": bool(prompt_was_truncated_list[i]),
                "generation_hit_limit": bool(generation_hit_limit),
                "fit_status": fit_status_list[i],
            }
        )

    return results


def write_results_to_dataframe(
    out: pd.DataFrame,
    batch_indices: list[int],
    batch_results: list[dict],
) -> None:
    for idx, result in zip(batch_indices, batch_results):
        out.at[idx, RAW_OUTPUT_COL] = result["raw_output"]
        out.at[idx, INPUT_TOKENS_COL] = result["input_tokens"]
        out.at[idx, PROMPT_WAS_TRUNCATED_COL] = result["prompt_was_truncated"]
        out.at[idx, GENERATION_HIT_LIMIT_COL] = result["generation_hit_limit"]
        out.at[idx, PROMPT_FIT_STATUS_COL] = result["fit_status"]


def run_inference(df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    out = df.copy()

    out[RAW_OUTPUT_COL] = None
    out[INPUT_TOKENS_COL] = pd.NA
    out[PROMPT_WAS_TRUNCATED_COL] = False
    out[GENERATION_HIT_LIMIT_COL] = False
    out[PROMPT_FIT_STATUS_COL] = None

    all_indices = list(out.index)
    total_rows = len(all_indices)

    start_time = time.perf_counter()
    processed_rows = 0
    batch_counter = 0

    progress_bar = tqdm(total=total_rows, desc="Running batched inference on test split")

    current_pos = 0
    current_batch_size = BATCH_SIZE

    while current_pos < total_rows:
        batch_indices = all_indices[current_pos:current_pos + current_batch_size]

        batch_questions = [out.at[idx, QUESTION_COL] for idx in batch_indices]
        batch_answers = [out.at[idx, ANSWER_COL] for idx in batch_indices]
        batch_row_ids = [str(out.at[idx, ID_COL]) for idx in batch_indices]

        batch_start = time.perf_counter()

        try:
            batch_results = generate_batch_predictions(
                model=model,
                tokenizer=tokenizer,
                questions=batch_questions,
                answers=batch_answers,
                row_identifiers=batch_row_ids,
            )

            write_results_to_dataframe(out, batch_indices, batch_results)

            batch_time = time.perf_counter() - batch_start
            processed_rows += len(batch_indices)
            batch_counter += 1
            current_pos += len(batch_indices)
            progress_bar.update(len(batch_indices))

            elapsed = time.perf_counter() - start_time
            rows_per_sec = processed_rows / elapsed if elapsed > 0 else 0.0
            sec_per_row = elapsed / processed_rows if processed_rows > 0 else 0.0

            log_message(
                f"Batch done: batch_size={len(batch_indices)}, "
                f"batch_time={batch_time:.2f}s, processed={processed_rows}/{total_rows}, "
                f"rows_per_sec={rows_per_sec:.4f}, sec_per_row={sec_per_row:.2f}"
            )

            if batch_counter % SAVE_EVERY_N_BATCHES == 0:
                save_parquet(OUTPUT_PARQUET, out)

            if torch.cuda.is_available() and batch_counter % EMPTY_CACHE_EVERY_N_BATCHES == 0:
                torch.cuda.empty_cache()

            if current_batch_size < BATCH_SIZE:
                current_batch_size = min(BATCH_SIZE, current_batch_size * 2)

        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if current_batch_size <= MIN_BATCH_SIZE:
                raise RuntimeError(
                    f"CUDA OOM even at batch_size={current_batch_size}. "
                    f"Reduce MAX_NEW_TOKENS or use a smaller model."
                ) from None

            new_batch_size = max(MIN_BATCH_SIZE, current_batch_size // 2)
            log_message(
                f"[WARNING] CUDA OOM at batch_size={current_batch_size}. "
                f"Retrying with smaller batch_size={new_batch_size}."
            )
            current_batch_size = new_batch_size

    progress_bar.close()
    return out


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    LOG_FILE.write_text("", encoding="utf-8")

    if torch.cuda.is_available() and ENABLE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    df = open_parquet_file(INPUT_PARQUET)

    required_cols = [ID_COL, QUESTION_COL, ANSWER_COL, SPLIT_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[df[SPLIT_COL] == TEST_SPLIT_VALUE].copy()

    if df.empty:
        raise ValueError(f"No rows found with {SPLIT_COL} == '{TEST_SPLIT_VALUE}'")

    if ROW_LIMIT is not None:
        df = df.head(ROW_LIMIT).copy()

    log_message(f"Rows to process after split filter: {len(df)}")
    log_message(f"Using only rows where {SPLIT_COL} == '{TEST_SPLIT_VALUE}'")
    log_message(f"USE_FINETUNED_MODEL={USE_FINETUNED_MODEL}")
    log_message(f"BATCH_SIZE={BATCH_SIZE}")
    log_message(f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
    log_message(f"DO_SAMPLE={DO_SAMPLE}")

    model, tokenizer = load_model_and_tokenizer()

    total_start = time.perf_counter()
    result_df = run_inference(df, model, tokenizer)
    total_elapsed = time.perf_counter() - total_start

    save_parquet(OUTPUT_PARQUET, result_df)

    truncated_count = int(result_df[PROMPT_WAS_TRUNCATED_COL].sum())
    hit_limit_count = int(result_df[GENERATION_HIT_LIMIT_COL].sum())

    rows_per_sec = len(result_df) / total_elapsed if total_elapsed > 0 else 0.0
    sec_per_row = total_elapsed / len(result_df) if len(result_df) > 0 else 0.0

    log_message("Done.")
    log_message(f"Raw outputs: {len(result_df)} / {len(result_df)}")
    log_message(f"Prompts truncated: {truncated_count}")
    log_message(f"Generations hit MAX_NEW_TOKENS: {hit_limit_count}")
    log_message(f"Total runtime: {total_elapsed:.2f}s")
    log_message(f"Rows/sec: {rows_per_sec:.4f}")
    log_message(f"Sec/row: {sec_per_row:.2f}")


if __name__ == "__main__":
    main()
