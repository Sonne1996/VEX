# Script that creates version 0.3 from version 0.2 stable
import pandas as pd
import sys
from pathlib import Path
import datetime
import re
import json
import numpy as np

grade_mapping = {
    "incorrect": 0.0,
    "mostly incorrect": 0.25,
    "partially correct": 0.5,
    "mostly correct": 0.75,
    "correct": 1.0,
} 

llm_mapping = { 
    "Yes": "1", 
    "No": "0" 
}

def open_parquet_file(path: Path):

    if(not(path.suffix.lower() == ".parquet")):
        print("You did not input a path for a parquet file.")
        sys.exit()

    if not path.exists():
        print(f"File not found at: {path.absolute()}")
        sys.exit()
        
    df = pd.read_parquet(path)
    return df

def remove_rows_from_parquet(df: pd.DataFrame, row_names):
    df_cleaned = df.drop(row_names, axis='columns', inplace=False)
    return df_cleaned

def save_parquet(file_name: str, df: pd.DataFrame):
    target_file = f"{file_name}.parquet"

    df.to_parquet(target_file, engine='pyarrow', compression='snappy', index=False)
    print("Parquet saved to: " + target_file)

def file_information(df: pd.DataFrame, file_name: str, join_stats=None):
    info_file = f"{file_name}_metadata.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"REPORT: METADATA FOR {file_name.upper()}\n")
        f.write("="*80 + "\n")
        f.write(f"Created on:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Rows:       {len(df)}\n")
        f.write(f"Total Columns:    {len(df.columns)}\n")
        f.write("-" * 80 + "\n")
        
        # Header für die Tabelle
        header = f"{'Column Name':<65} | {'Type':<12} | {'Missing':<8} | {'Distinct':<8} | {'Unique %':<8}"
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        
        for col in df.columns:
            # Metriken berechnen
            null_count = df[col].isnull().sum()
            dtype = str(df[col].dtype)
            distinct_count = df[col].nunique()
            
            # Uniqueness % berechnen (wie viel % der nicht-leeren Werte sind einzigartig)
            total_non_null = len(df) - null_count
            uniqueness_pct = (distinct_count / total_non_null * 100) if total_non_null > 0 else 0
            
            # Zeile formatieren
            row = f"{col[:60]:<65} | {dtype:<12} | {null_count:<8} | {distinct_count:<8} | {uniqueness_pct:>7.1f}%"
            f.write(row + "\n")
            
        f.write("-" * 80 + "\n")
        f.write(f"RAM Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        
        if join_stats:
            f.write("="*80 + "\n")
            f.write("ADDITIONAL JOIN STATS:\n")
            for key, value in join_stats.items():
                f.write(f"- {key:<35}: {value}\n")
        
        f.write("="*80 + "\n")
    print(f"Detailed metadata report created: {info_file}")

def add_columns(df, column_names):
    """
    Takes a list of column names and adds each to the DataFrame 
    if it doesn't already exist, initializing with None.
    """
    for col in column_names:
        if col not in df.columns:
            df[col] = None
        else:
            print(f"Column '{col}' already exists. Skipping.")
    
    return df

def drop_columns(df, column_name):
    # We use errors='ignore' so the code doesn't crash if a column 
    # was already dropped or doesn't exist.
    df = df.drop(columns=column_name, errors='ignore')
    return df

def merge_labeled_data(labeled_path, master_path):
    # 1. Load the DataFrames
    df_labeled = pd.read_parquet(labeled_path)  # The ~3,318 labels
    df_master = pd.read_parquet(master_path)    # The 100% dataset

    # --- SHADOW KEY LOGIC ---
    # This helper removes all special characters, whitespace, and case sensitivity.
    # It ensures that '{"a": 1}' matches '{"a":1}' or '{"a": 1\n}'.
    def create_shadow_key(s):
        if s is None: return ""
        # Keep only letters and numbers, lowercase everything
        return re.sub(r'[^a-z0-9]', '', str(s).lower())

    print("Creating temporary shadow keys for robust matching...")
    df_labeled['_tmp_shadow_key'] = df_labeled['model_response_with_metadata'].apply(create_shadow_key)
    df_master['_tmp_shadow_key'] = df_master['model_response_with_metadata'].apply(create_shadow_key)

    # 4. Perform the merge
    # We join on question_id and our temporary shadow key.
    # We only bring over the necessary columns from the labeled data.
    merged_df = pd.merge(
        df_master, 
        df_labeled[['_tmp_shadow_key', 'question_id', 'human_grade 1', 'is_llm 1', 'grader_name 1', 'human_grade 2', 'is_llm 2', 'grader_name 2', 
        'gold_label_after_human_audit', 'gold_is_llm_after_human_audit', 'consensus_status_audit', 'human_audit_comment']], 
        on=['question_id', '_tmp_shadow_key'], 
        how='left'
    )

    # CLEANUP: Remove the temporary shadow key before returning
    # This ensures version 0.3 looks exactly like the master format.
    merged_df = merged_df.drop(columns=['_tmp_shadow_key'])

    return merged_df


def merge_silver_labels(df_vercel_input, df_master_input, merge_key="grading_id"):
    """
    Merges only true silver-label result columns from the Vercel dataframe
    into the master dataframe using grading_id.

    Prevents duplicate columns like question_x/question_y, answer_x/answer_y, split_x/split_y.
    """

    if isinstance(df_vercel_input, (str, Path)):
        df_vercel = pd.read_parquet(df_vercel_input)
    else:
        df_vercel = df_vercel_input.copy()

    if isinstance(df_master_input, (str, Path)):
        df_master = pd.read_parquet(df_master_input)
    else:
        df_master = df_master_input.copy()

    if merge_key not in df_vercel.columns:
        raise ValueError(f"merge_key '{merge_key}' fehlt im Vercel-DataFrame")

    if merge_key not in df_master.columns:
        raise ValueError(f"merge_key '{merge_key}' fehlt im Master-DataFrame")

    # Nur neue Silver-Result-Spalten aus Vercel behalten
    cols_to_merge = [merge_key] + [
        col for col in df_vercel.columns
        if col != merge_key and col not in df_master.columns
    ]

    df_vercel_small = df_vercel[cols_to_merge].copy()

    # Falls grading_id im Vercel-Result doppelt vorkommt: first wins
    df_vercel_small = df_vercel_small.drop_duplicates(subset=[merge_key], keep="first")

    merged_df = pd.merge(
        df_master,
        df_vercel_small,
        on=merge_key,
        how="left"
    )

    return merged_df
  

def remove_members(df, member_ids_to_remove):
    return df[~df["member_id"].isin(member_ids_to_remove)].copy()

def clamp_grade_column(df: pd.DataFrame, column_name: str = "grade") -> pd.DataFrame:
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found. No clamping performed.")
        return df

    df = df.copy()

    numeric_grade = pd.to_numeric(df[column_name], errors="coerce")

    changed_count = 0

    for idx, value in numeric_grade.items():
        if pd.isna(value):
            continue

        clamped_value = min(max(value, 0.0), 1.0)

        if clamped_value != value:
            print(
                f"Clamped row index {idx}: {column_name} {value} -> {clamped_value}"
            )
            df.at[idx, column_name] = clamped_value
            changed_count += 1

    print(f"Total clamped values in column '{column_name}': {changed_count}")
    return df
    


def version_0_3():
    BASE_DIR = Path(__file__).resolve().parent

    path = BASE_DIR / ".." / "v0_3" / "v0.26.parquet"

    df = pd.read_parquet(path)

    df = df.rename(columns={
        'name': 'student_name',
        'bloom': 'bloom_level',
        'topic' : 'question_topic'
    })

    save_path = Path("v0.3_stable")

    save_parquet(save_path, df)

    file_information(df, "v0.3")

    return

def version_0_26():
    BASE_DIR = Path(__file__).resolve().parent

    path = BASE_DIR / ".." / "v0_3" / "v0.25.parquet"

    df = pd.read_parquet(path)

    columns_to_remove = ["rating"]

    df = remove_rows_from_parquet(df, columns_to_remove)

    save_path = Path("v0.26")

    save_parquet(save_path, df)

    file_information(df, "v0.26")


def version_0_25():

    BASE_DIR = Path(__file__).resolve().parent

    path = BASE_DIR / ".." / "v0_3" / "v0.24.parquet"

    df = pd.read_parquet(path)

    columns_to_remove = ["human_grade 1", "is_llm 1", "grader_name 1", "human_grade 2", "is_llm 2", "grader_name 2", "gold_label_after_human_audit",
    "consensus_status_audit", "human_audit_comment"]
   
    df = remove_rows_from_parquet(df,  columns_to_remove)

    save_path = Path("v0.25")

    save_parquet(save_path, df)

    file_information(df, "v0.25")

def version_0_24():
    BASE_DIR = Path(__file__).resolve().parent

    path = BASE_DIR / ".." / "v0_3" / "v0.23.parquet"

    df = pd.read_parquet(path)

    columns_to_remove = ["model_response_with_metadata", "new_grade_google/gemini-2.5-pro", "raw_output_google/gemini-2.5-pro", "metadata_google/gemini-2.5-pro"]

    df = remove_rows_from_parquet(df, columns_to_remove)

    save_path = Path("v0.24")

    save_parquet(save_path, df)

    file_information(df, "v0.24")


def version_0_23():
    BASE_DIR = Path(__file__).resolve().parent

    path = BASE_DIR / ".." / "v0_3" / "v0.22.parquet"

    df = open_parquet_file(path)

    df = clamp_grade_column(df, "grade")

    save_path = Path("v0.23")

    save_parquet(save_path, df)

    file_information(df, "v0.23")

def version_0_22():
    BASE_DIR = Path(__file__).resolve().parent

    path = BASE_DIR / ".." / "v0_3" / "v0.21.parquet"

    df = open_parquet_file(path)

    df["grade"] = df["grade"].combine_first(df["new_grade_google/gemini-2.5-pro"])

    save_path = Path("v0.22")

    save_parquet(save_path, df)

    file_information(df, "v0.22")


def version_0_21():
    BASE_DIR = Path(__file__).resolve().parent

    path = BASE_DIR / ".." / "v0_2" / "v0.2_stable.parquet"

    df = open_parquet_file(path)

    df = merge_silver_labels("../../vercel_api_results/requests_for_silver_labels/results_silver_labels.parquet", path, "grading_id")

    save_path = Path("v0.21")

    save_parquet(save_path, df)

    file_information(df, "v0.21")

def main():

    version_0_21()

    version_0_22()

    version_0_23()

    version_0_24()

    version_0_25()

    version_0_26()

    version_0_3()


main()
