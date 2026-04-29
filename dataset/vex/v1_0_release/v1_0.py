import pandas as pd
import sys
from pathlib import Path
import datetime
import re
import json
import numpy as np

def open_parquet_file(path: Path):

    if(not(path.suffix.lower() == ".parquet")):
        print("You did not input a path for a parquet file.")
        sys.exit()

    if not path.exists():
        print(f"File not found at: {path.absolute()}")
        sys.exit()
        
    df = pd.read_parquet(path)
    return df


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

def remove_rows_from_parquet(df: pd.DataFrame, row_names):
    df_cleaned = df.drop(row_names, axis='columns', inplace=False)
    return df_cleaned

def version_1_0():
    BASE_DIR = Path(__file__).resolve().parent

    path = BASE_DIR / ".." / "v0_3" / "v0.3_stable.parquet"

    df = open_parquet_file(path)

    

    save_path = Path("v0.3_stable")

    save_parquet(save_path, df)

    file_information(df, "v0.3_stable")

def main():

    version_1_0()

main()
