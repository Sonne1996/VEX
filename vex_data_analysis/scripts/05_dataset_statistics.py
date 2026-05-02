import os
import pandas as pd


def load_data():
    files = [f for f in os.listdir("input") if f.endswith(".parquet")]
    path = os.path.join("input", files[0])
    return pd.read_parquet(path)


def main():
    df = load_data()

    # === BASIC COUNTS ===
    total_responses = len(df)
    students = df["member_id"].nunique()
    questions = df["question_id"].nunique()
    gold = (df["label_type"] == "gold").sum()

    # === GROUPED ===
    responses_per_student = df.groupby("member_id").size()
    responses_per_question = df.groupby("question_id").size()

    avg_per_student = responses_per_student.mean()
    median_per_student = responses_per_student.median()

    avg_per_question = responses_per_question.mean()
    median_per_question = responses_per_question.median()

    # === RESPONSE LENGTH ===
    df["answer_length"] = df["answer"].str.split().str.len()

    avg_len = df["answer_length"].mean()
    median_len = df["answer_length"].median()

    # === WRITE TXT ===
    os.makedirs("output", exist_ok=True)

    with open("output/dataset_statistics.txt", "w") as f:
        f.write("=== DATASET STATISTICS ===\n\n")

        f.write(f"Students (final release): {students}\n")
        f.write(f"Total responses: {total_responses}\n")
        f.write(f"Avg responses per question: {avg_per_question:.2f}\n")
        f.write(f"Avg responses per student: {avg_per_student:.2f}\n")
        f.write(f"Avg response length (tokens): {avg_len:.2f}\n\n")

        f.write(f"Unique questions: {questions}\n")
        f.write(f"Gold-labeled responses: {gold}\n")
        f.write(f"Median responses per question: {median_per_question:.2f}\n")
        f.write(f"Median responses per student: {median_per_student:.2f}\n")
        f.write(f"Median response length (tokens): {median_len:.2f}\n")

    print("Saved: output/dataset_statistics.txt")


if __name__ == "__main__":
    main()
