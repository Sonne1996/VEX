import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    files = [f for f in os.listdir("input") if f.endswith(".parquet")]
    path = os.path.join("input", files[0])
    return pd.read_parquet(path)


def main():
    df = load_data()

    responses_per_student = df.groupby("member_id").size()

    # Kategorien definieren
    low = (responses_per_student < 50).sum()
    medium = ((responses_per_student >= 50) & (responses_per_student < 190)).sum()
    high = (responses_per_student >= 190).sum()

    # Labels mit Zeilenumbruch (kein Overlap)
    labels = ["Low\n(<50)", "Partial\n(50–189)", "High\n(≥190)"]
    values = [low, medium, high]

    fig, ax = plt.subplots(figsize=(3.25, 2.2))

    ax.bar(labels, values, color="#4C72B0", width=0.6)

    ax.set_ylabel("Number of students", fontsize=10)

    # Grid im gleichen Stil wie andere Plots
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    # Cleaner Look
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Tick-Label etwas kleiner für bessere Lesbarkeit
    plt.xticks(fontsize=9)

    fig.tight_layout()

    os.makedirs("output", exist_ok=True)
    fig.savefig("output/student_categories.pdf", bbox_inches="tight")

    print("Saved: output/student_categories.pdf")


if __name__ == "__main__":
    main()
