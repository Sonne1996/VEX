import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    files = [f for f in os.listdir("input") if f.endswith(".parquet")]
    path = os.path.join("input", files[0])
    return pd.read_parquet(path)


def main():
    df = load_data()

    # Antworten pro Student
    responses_per_student = df.groupby("member_id").size()
    values = responses_per_student.sort_values().values

    # ===== STATS =====
    total_students = len(responses_per_student)

    high_coverage = (responses_per_student >= 190).sum()
    medium_coverage = ((responses_per_student > 0) & (responses_per_student < 190)).sum()
    low_coverage = (responses_per_student == 0).sum()

    print("\n=== STUDENT COVERAGE STATS ===")
    print(f"Total students: {total_students}")
    print(f">=190 responses: {high_coverage} ({high_coverage/total_students*100:.2f}%)")
    print(f"1–189 responses: {medium_coverage} ({medium_coverage/total_students*100:.2f}%)")
    print(f"0 responses: {low_coverage} ({low_coverage/total_students*100:.2f}%)")

    print("\nDistribution summary:")
    print(responses_per_student.describe())

    # ===== PLOT =====
    fig, ax = plt.subplots(figsize=(3.25, 2.2))

    ax.plot(values, color="#4C72B0", linewidth=1.5)

    ax.set_xlabel("Students (sorted)", fontsize=10)
    ax.set_ylabel("Responses per student", fontsize=10)

    ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    os.makedirs("output", exist_ok=True)
    fig.savefig("output/student_coverage.pdf", bbox_inches="tight")

    print("\nSaved: output/student_coverage.pdf")

if __name__ == "__main__":
    main()
