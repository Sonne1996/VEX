#SCript that creates version 0.1 from raw db data
import sqlite3
import pandas as pd
import datetime
import os
from pathlib import Path

def create_anonymous_sqlite(source_db='raw_not_anomyzed.sqlite', target_db='..raw/raw.sqlite'):
    """
    Creates a new SQLite database and copies only the relevant tables.
    """
    print(f"Starting anonymization: {source_db} -> {target_db}...")
    
    # Establish connection to source and target databases
    src_conn = sqlite3.connect(source_db)
    tgt_conn = sqlite3.connect(target_db)
    
    # List of tables to be transferred
    allowed_tables = [
        'shared_members',
        'shared_member_answers',
        'shared_member_gradings',
        'shared_grading_feedback'
    ]
    
    try:
        for table in allowed_tables:
            print(f"Copying table: {table}...")
            # Load entire table into a DataFrame
            df = pd.read_sql_query(f"SELECT * FROM {table}", src_conn)
            
            # Write to the new database
            # if_exists='replace' ensures the file is recreated on every run
            df.to_sql(table, tgt_conn, index=False, if_exists='replace')
            
        print("Anonymization completed successfully.")
        
    except Exception as e:
        print(f"Error during anonymization: {e}")
        
    finally:
        src_conn.close()
        tgt_conn.close()

def try_anonymize_data():
    source_db = 'raw_not_anomyzed.sqlite'
    target_db = 'raw.sqlite'
    
    # Check if the source database exists
    if os.path.exists(source_db):
        print(f"Source file '{source_db}' found. Starting anonymization...")
        create_anonymous_sqlite(source_db, target_db)
    else:
        print(f"Note: '{source_db}' not found. Skipping anonymization.")
        
    # Check if the working database (raw.sqlite) exists to proceed
    if not os.path.exists(target_db):
        print(f"Error: '{target_db}' does not exist. Script will be aborted.")
        return

def open_connection():
    conn = sqlite3.connect('raw.sqlite')
    return conn

def close_connection(conn):
    conn.close()

def save_to_parquet(file_name, df):
    target_file = f"{file_name}.parquet"

    df.to_parquet(target_file, engine='pyarrow', compression='snappy', index=False)
    print(f"Parquet file created successfully: {target_file}")
    
    check_df = pd.read_parquet(target_file)
    print(f"Validation: {len(check_df)} rows found in {target_file}.")

def file_information(file_name, df, join_stats=None):
    info_file = f"{file_name}_metadata.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"REPORT: METADATA FOR {file_name.upper()}\n")
        f.write("="*40 + "\n")
        f.write(f"Created on:      {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Entries in result (Rows): {len(df)}\n")
        f.write(f"Columns (Cols):  {len(df.columns)}\n")
        
        if join_stats:
            f.write("-" * 40 + "\n")
            f.write("TABLE SIZES & JOIN ANALYSIS:\n")
            for key, value in join_stats.items():
                # Formats the output: Table names or status messages
                f.write(f"- {key:<35}: {value}\n")
        
        f.write("-" * 40 + "\n")
        f.write("COLUMN DETAILS:\n")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            dtype = df[col].dtype
            f.write(f"- {col:<40} | Type: {str(dtype):<10} | Missing: {null_count}\n")
            
        f.write("-" * 40 + "\n")
        f.write(f"RAM Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        f.write("="*40 + "\n")
    print(f"Metadata report created: {info_file}")

def get_table_size(conn, table_name):
    """Returns the number of rows in a table."""
    query = f"SELECT COUNT(*) FROM {table_name}"
    return pd.read_sql_query(query, conn).iloc[0, 0]

def get_lost_rows_count(conn, base_table, join_table, base_key, join_key):
    """Counts IDs that are missing in the target table."""
    query = f"""
    SELECT COUNT(*) 
    FROM {base_table} 
    WHERE {base_key} NOT IN (SELECT {join_key} FROM {join_table} WHERE {join_key} IS NOT NULL)
    """
    return pd.read_sql_query(query, conn).iloc[0, 0]

def fallback(p_name, df, stats):
    if not df.empty:
            save_to_parquet(p_name, df)
            file_information(p_name, df, join_stats=stats)
    else:
        print("Warning: Join resulted in no data!")

# Joins shared_members with shared_members_answers
# Original rows of shared_members / (kept/not kept):
#   - ID / kept as member_id
#   - name / kept
#   - subject_id / kept
# Original rows of shared_members_answers / (kept/not kept):
#   - question_id / kept
#   - content / kept as answer
#   - question / kept
def join_one(conn):
    stats = {
        "Size 'shared_members'": get_table_size(conn, "shared_members"),
        "Size 'shared_member_answers'": get_table_size(conn, "shared_member_answers"),
        "Members without answers": get_lost_rows_count(conn, "shared_members", "shared_member_answers", "id", "shared_member_id")
    }
    
    query = """
    SELECT  m.id AS member_id, 
            m.name, 
            m.subject_id, 
            a.id AS answer_id,
            a.question_id, 
            a.content AS answer, 
            a.question 
    FROM shared_members m 
    JOIN shared_member_answers a ON m.id = a.shared_member_id
    """
    return pd.read_sql_query(query, conn), stats

def join_two(conn):
    stats = {
        "Size 'shared_member_answers'": get_table_size(conn, "shared_member_answers"),
        "Size 'shared_member_gradings'": get_table_size(conn, "shared_member_gradings"),
        "Answers without AI grading": get_lost_rows_count(conn, "shared_member_answers", "shared_member_gradings", "id", "answer_id")
    }
    
    query = """
    SELECT  m.id AS member_id, 
            m.name, 
            m.subject_id, 
            a.id AS answer_id,
            a.question_id, 
            a.content AS answer, 
            a.question,
            g.accuracy, 
            g.content AS model_prediction, 
            g.used_model, 
            g.used_rubric, 
            g.used_examples,
            g.response AS model_response_with_metadata
    FROM shared_members m
    JOIN shared_member_answers a ON m.id = a.shared_member_id
    JOIN shared_member_gradings g ON a.id = g.answer_id
    """
    return pd.read_sql_query(query, conn), stats

def join_three(conn):

    stats = {
        "Size 'shared_member_gradings'": get_table_size(conn, "shared_member_gradings"),
        "Size 'shared_grading_feedback'": get_table_size(conn, "shared_grading_feedback"),
        "Gradings without feedback (will be NULL)": get_lost_rows_count(conn, "shared_member_gradings", "shared_grading_feedback", "id", "grading_id")
    }

    query = """
    SELECT 
        m.id AS member_id,
        m.name,
        m.subject_id,
        a.id AS answer_id,
        a.question_id,
        a.content AS answer,
        a.question,
        g.id AS grading_id,
        g.accuracy,
        g.content AS model_prediction,
        g.used_model,
        g.used_rubric,
        g.used_examples,
        g.response AS model_response_with_metadata,
        f.rating,
        f.comment
    FROM shared_members m
    JOIN shared_member_answers a ON m.id = a.shared_member_id
    JOIN shared_member_gradings g ON a.id = g.answer_id
    LEFT JOIN shared_grading_feedback f ON g.id = f.grading_id
    """

    df = pd.read_sql_query(query, conn)

    return df, stats

def main():

    target_name = "v0_1"

    current_folder_name = os.path.basename(os.getcwd())

    if current_folder_name != target_name:

        if not os.path.exists(target_name):
            os.makedirs(target_name)
            print(f"Folder '{target_name}' was created.")
        
        os.chdir(target_name)
        print(f"Moved into folder '{target_name}'.")
    else:
        print(f"Already in the correct folder: {target_name}")

    print(f"Current path: {os.getcwd()}")

    try_anonymize_data()

    conn = None
    try:
        conn = open_connection()
        
        # join one
        p_name = 'v0.01'
        df, stats = join_one(conn)
        
        fallback(p_name, df, stats)

        # join two
        p_name = 'v0.02'
        df, stats = join_two(conn)
        
        fallback(p_name, df, stats)

        # join three
        p_name = 'v0.1_stable'
        df, stats = join_three(conn)
        
        fallback(p_name, df, stats)

    finally:
        if conn:
            close_connection(conn)

main()

print("Finished! The Parquet files have been created.")