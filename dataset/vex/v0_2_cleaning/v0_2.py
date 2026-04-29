#Script that creates version 0.2 from version 0.1 stable
import pandas as pd
import sys
from pathlib import Path
import datetime
import json
import re

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

def map_categories_to_ordinal(df):
    
    if "gold_label_after_human_audit" not in df.columns:
        raise ValueError(
            "Spalte 'gold_label_after_human_audit' fehlt"
        )

    def map_gold_label(value):
        if pd.isna(value):
            return pd.NA

        value_str = str(value).strip().lower()

        if value_str not in grade_mapping:
            raise ValueError(f"Unbekannter Label-Wert in gold_label_after_human_audit: {value!r}")

        return grade_mapping[value_str]

    df["gold_label_after_human_audit_num"] = df["gold_label_after_human_audit"].apply(map_gold_label)

    return df

def map_is_llm_to_binary(df):
    if "gold_is_llm_after_human_audit" not in df.columns:
        raise ValueError(
            "Spalte 'gold_is_llm_after_human_audit' fehlt"
        )

    def map_is_llm(value):
        if pd.isna(value):
            return pd.NA

        value_str = str(value).strip()

        if value_str not in llm_mapping:
            raise ValueError(
                f"Unbekannter Label-Wert in gold_is_llm_after_human_audit: {value!r}"
            )

        return llm_mapping[value_str]

    df["gold_is_llm_after_human_audit_num"] = df["gold_is_llm_after_human_audit"].apply(map_is_llm)

    return df

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
        f.write("="*90 + "\n")
        f.write(f"Created on:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Rows:       {len(df)}\n")
        f.write(f"Total Columns:    {len(df.columns)}\n")
        f.write("-" * 90 + "\n")
        
        # Header für die Tabelle
        header = f"{'Column Name':<40} | {'Type':<12} | {'Missing':<8} | {'Distinct':<8} | {'Unique %':<8}"
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        
        for col in df.columns:
            # Metriken berechnen
            null_count = df[col].isnull().sum()
            dtype = str(df[col].dtype)
            distinct_count = df[col].nunique()
            
            # Uniqueness % berechnen (wie viel % der nicht-leeren Werte sind einzigartig)
            total_non_null = len(df) - null_count
            uniqueness_pct = (distinct_count / total_non_null * 100) if total_non_null > 0 else 0
            
            # Zeile formatieren
            row = f"{col[:40]:<40} | {dtype:<12} | {null_count:<8} | {distinct_count:<8} | {uniqueness_pct:>7.1f}%"
            f.write(row + "\n")
            
        f.write("-" * 90 + "\n")
        f.write(f"RAM Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        
        if join_stats:
            f.write("="*90 + "\n")
            f.write("ADDITIONAL JOIN STATS:\n")
            for key, value in join_stats.items():
                f.write(f"- {key:<35}: {value}\n")
        
        f.write("="*90 + "\n")
    print(f"Detailed metadata report created: {info_file}")


def extract_answer(df: pd.DataFrame, column_name: str):
    """
    Extrahiert Text aus einer Tiptap-JSON-Struktur und erstellt eine neue Spalte 'cleaned_answer'.
    """
    
    def get_text_recursive(node):
        """Hilfsfunktion, um alle 'text' Felder aus dem Tiptap-JSON zu fischen."""
        if not isinstance(node, dict):
            return ""
        
        text = ""
        # Wenn der Knoten vom Typ 'text' ist, nimm den Inhalt
        if node.get("type") == "text":
            text += node.get("text", "")
        
        # Wenn der Knoten 'content' hat (Liste), geh tiefer
        if "content" in node and isinstance(node["content"], list):
            for child in node["content"]:
                text += get_text_recursive(child)
        
        # Zeilenumbruch für Paragraphs oder ListItems hinzufügen, damit es lesbar bleibt
        if node.get("type") in ["paragraph", "listItem", "heading"]:
            text += "\n"
        elif node.get("type") == "hardBreak":
            text += "\n"
            
        return text

    def process_row(json_str):
        if not json_str or pd.isna(json_str):
            return ""
        try:
            # Falls es bereits ein Dict ist, nicht laden, sonst parsen
            data = json.loads(json_str) if isinstance(json_str, str) else json_str
            # Starte die rekursive Textextraktion
            cleaned = get_text_recursive(data)
            # Entferne doppelte Zeilenumbrüche am Ende
            return cleaned.strip()
        except Exception as e:
            print("something went wrong in extract answer function")
            sys.exit()

    # Neue Spalte erstellen
    df['cleaned_answer'] = df[column_name].apply(process_row)
    return df

def extract_question_details(df: pd.DataFrame, column_name: str):
    """
    Extrahiert Question, Rubric und Examples aus der komplexen Question-JSON-Struktur.
    Erstellt drei neue Spalten.
    """
    
    def get_text_recursive(node):
        if not isinstance(node, dict):
            return ""
        text = ""
        if node.get("type") == "text":
            text += node.get("text", "")
        if "content" in node and isinstance(node["content"], list):
            for child in node["content"]:
                text += get_text_recursive(child)
        if node.get("type") in ["paragraph", "listItem", "heading"]:
            text += "\n"
        elif node.get("type") == "hardBreak":
            text += "\n"
        return text

    def process_row(json_str):
        # Fallback für leere Zeilen
        res = {"q": "", "r": "", "e": ""}
        if not json_str or pd.isna(json_str):
            return pd.Series(res)
        
        try:
            data = json.loads(json_str) if isinstance(json_str, str) else json_str
            
            # 1. Eigentliche Fragestellung (aus 'content')
            res["q"] = get_text_recursive(data.get("content", {})).strip()
            
            # 2. Rubrik / Bewertungskriterien (aus 'rubric')
            res["r"] = get_text_recursive(data.get("rubric", {})).strip()
            
            # 3. Beispiele (Liste von Objekten mit 'content' und 'accuracy')
            example_list = data.get("examples", [])
            example_texts = []
            for i, ex in enumerate(example_list, 1):
                content_text = get_text_recursive(ex.get("content", {})).strip()
                acc = ex.get("accuracy", "N/A")
                example_texts.append(f"Ex {i} (Acc: {acc}): {content_text}")
            
            res["e"] = "\n---\n".join(example_texts)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            sys.exit()
            
        return pd.Series(res)

    # Wir wenden die Funktion an und expandieren das Ergebnis in 3 Spalten
    new_cols = df[column_name].apply(process_row)
    df['cleaned_question'] = new_cols['q']
    df['cleaned_rubric'] = new_cols['r']
    df['cleaned_examples'] = new_cols['e']
    
    return df


def add_columns(df, column_names):
    """
    Takes a list of column names and adds each to the DataFrame 
    if it doesn't already exist, initializing with None.
    """
    for col in column_names:
        if col not in df.columns:
            df[col] = ""
        else:
            print(f"Column '{col}' already exists. Skipping.")
    
    return df

def clean_parquet_duplicates(file_path, column_name):
    """
    Identifies rows with non-unique values in a specific column,
    removes the duplicates (keeping the first occurrence), 
    and returns the cleaned DataFrame.
    """
    try:
        # Load the parquet file
        df = pd.read_parquet(file_path)
        
        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found.")
            return df  # Return original if column isn't there

        # Find duplicates for reporting/visibility
        duplicates = df[df.duplicated(subset=[column_name], keep=False)]

        if duplicates.empty:
            print(f"All values in '{column_name}' are unique.")
            return df
        else:
            print(f"Found {len(duplicates)} rows with non-unique values in '{column_name}'.")
            print("The following rows (including duplicates) were identified:")
            print(duplicates)

            # Drop duplicates: keeps the first instance, deletes the rest
            df_cleaned = df.drop_duplicates(subset=[column_name], keep='first')
            
            print(f"\nSuccessfully removed {len(df) - len(df_cleaned)} rows.")
            return df_cleaned
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def merge_bloom_to_questions(df, path):
    if "question_id" not in df.columns:
        raise ValueError("Input df must contain column 'question_id'.")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata parquet not found: {path}")

    meta = pd.read_parquet(path)

    required_cols = {"id", "bloom", "topic"}
    missing_cols = required_cols - set(meta.columns)
    if missing_cols:
        raise ValueError(
            f"Metadata parquet missing required columns: {sorted(missing_cols)}"
        )

    meta_small = meta[["id", "bloom", "topic"]].copy()

    dup_counts = meta_small["id"].value_counts()
    dup_ids = dup_counts[dup_counts > 1].index.tolist()
    if dup_ids:
        raise ValueError(
            f"Metadata parquet contains duplicate ids. "
            f"Example duplicate ids: {dup_ids[:10]}"
        )

    out = df.copy()

    df_ids = set(out["question_id"].dropna().astype(str).unique())
    meta_ids = set(meta_small["id"].dropna().astype(str).unique())
    overlap = len(df_ids & meta_ids)

    print("ID overlap stats:")
    print(f"Unique question_id in df: {len(df_ids)}")
    print(f"Unique id in metadata: {len(meta_ids)}")
    print(f"Overlapping ids: {overlap}")

    out["question_id"] = out["question_id"].astype(str)
    meta_small["id"] = meta_small["id"].astype(str)

    out = out.merge(
        meta_small,
        left_on="question_id",
        right_on="id",
        how="left"
    ).drop(columns=["id"])

    missing_bloom = int(out["bloom"].isna().sum())
    missing_topic = int(out["topic"].isna().sum())

    print("\nBloom/topic merge complete.")
    print(f"Rows total: {len(out)}")
    print(f"Missing bloom: {missing_bloom}")
    print(f"Missing topic: {missing_topic}")

    return out

def remove_question(df, question_id):
    return df[df["question_id"] != question_id].copy()

def remove_members(df, member_ids_to_remove):
    return df[~df["member_id"].isin(member_ids_to_remove)].copy()

def remove_answers(df, answer_ids_to_remove):
    return df[~df["answer_id"].isin(answer_ids_to_remove)].copy()

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
        
def version_0_2():
    path = Path("v0.19.parquet")

    #drop json tables of answers, questions, model_prediciton 
    df = open_parquet_file(path)

    new_names = ["grade", "label_type"]

    df = add_columns(df, new_names)

    # Werte übernehmen
    df["grade"] = df["gold_label_after_human_audit_num"]
    df["label_type"] = df["gold_label_after_human_audit_num"].apply(
        lambda x: "gold" if not pd.isna(x) else "silver"
    )

    df = remove_rows_from_parquet(df, "gold_label_after_human_audit_num")

    df["split"] = df["grade"].apply(
        lambda x: "test" if pd.notna(x) else "train")

    #reorder columns
    new_order = ['member_id', 'subject_id', 'answer_id', 'question_id', 'grading_id', 
    'name', 'cleaned_question', 'bloom', 'topic', 'cleaned_answer', 'grade', 'label_type', 'gold_is_llm_after_human_audit', 'split', 'rating',
    'human_grade 1', 'is_llm 1', 'grader_name 1', 'human_grade 2', 'is_llm 2', 'grader_name 2', 'gold_label_after_human_audit',
    'consensus_status_audit', 'human_audit_comment', 'model_response_with_metadata']

    df = df[new_order]

    #rename columns
    df = df.rename(columns={
        'cleaned_question': 'question',
        'cleaned_answer': 'answer',
        'gold_is_llm_after_human_audit' : 'gold_is_llm'
    })

    save_path = Path("v0.2_stable")

    save_parquet(save_path, df)

    file_information(df, "v0.2_stable")

def version_0_19():
    path = Path("v0.18.parquet")

    df = open_parquet_file(path)

    df = df = merge_labeled_data("../../human_feedback_google_sheet/human_labels_create_gold_labels/download_final_gold/frozen_gold.parquet", path)

    df = map_categories_to_ordinal(df)

    df = map_is_llm_to_binary(df)

    save_path = Path("v0.19")

    save_parquet(save_path, df)

    file_information(df, "v0.19")

def version_0_18():
    path = Path("v0.17.parquet")

    df = open_parquet_file(path)

    df = remove_members(df, ["VJiFivvV0Um_1yyTPCmc0", "13pzmlMzmkoZBejyDzFl_", "Ina8QxMqgA6Ya-MeDHTCY", "9-jA_OsemUsthIpvs1Qxw"])

    df = remove_answers(df, ["T6eFhshJK_KrB8t7I7uHS", "SOSvnNsyZYnQkG3-0h4-s", "aWZ4uVDrn6HhfzXqLSo8n", "Qn9WWcNmG6Z0q0nZwevmq"])

    save_path = Path("v0.18")

    save_parquet(save_path, df)

    file_information(df, "v0.18")


def version_0_17():
    path = Path("v0.16.parquet")

    df = open_parquet_file(path)

    df = clean_parquet_duplicates(path, 'answer_id')

    save_path = Path("v0.17")

    save_parquet(save_path, df)

    file_information(df, "v0.17")


def version_0_16():
    path = Path("v0.15.parquet")

    df = open_parquet_file(path)

    row_names = ["used_model", "accuracy", "cleaned_rubric", "cleaned_examples"]

    df = remove_rows_from_parquet(df, row_names)

    save_path = Path("v0.16")

    save_parquet(save_path, df)

    file_information(df, "v0.16")


def version_0_15():
    path = Path("v0.14.parquet")

    df = open_parquet_file(path)

    row_names = ["answer", "question", "model_prediction", "used_rubric", "used_examples"]

    df = remove_rows_from_parquet(df, row_names)

    save_path = Path("v0.15")

    save_parquet(save_path, df)

    file_information(df, "v0.15")

def version_0_14():
    path = Path("v0.13.parquet")

    df = open_parquet_file(path)

    df = merge_bloom_to_questions(df, Path("../../human_feedback_google_sheet/questions/questions_with_bloom.parquet"))

    df = remove_question(df, "cGrXsW0ziGlQfLDRK_16W")

    save_path = Path("v0.14")

    save_parquet(save_path, df)

    file_information(df, "v0.14")

def version_0_13():
    path = Path("v0.12.parquet")

    df = open_parquet_file(path)

    row_names = ["comment"]

    df = remove_rows_from_parquet(df, row_names)

    save_path = Path("v0.13")

    save_parquet(save_path, df)

    file_information(df, "v0.13")

def version_0_12():
    path = Path("v0.11.parquet")

    df = open_parquet_file(path)

    extract_question_details(df, "question")

    save_path = Path("v0.12")

    save_parquet(save_path, df)

    file_information(df, "v0.12")

def version_0_11():
    BASE_DIR = Path(__file__).resolve().parent

    path = BASE_DIR / ".." / "v0_1" / "v0.1_stable.parquet"

    df = open_parquet_file(path)

    extract_answer(df, "answer")

    save_path = Path("v0.11")

    save_parquet(save_path, df)

    file_information(df, "v0.11")

def main():

    version_0_11()


    version_0_12()


    version_0_13()


    version_0_14()


    version_0_15()


    version_0_16()


    version_0_17()


    version_0_18()


    version_0_19()

   
    version_0_2()

main()
