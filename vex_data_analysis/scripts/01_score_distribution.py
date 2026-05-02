import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    files = [f for f in os.listdir("input") if f.endswith(".parquet")]
    path = os.path.join("input", files[0])
    return pd.read_parquet(path)


def main():
    df = load_data()

    score_order = [0, 0.25, 0.5, 0.75, 1]

    counts = df["grade"].value_counts().reindex(score_order)
    percentages = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(3.25, 2.2))

    ax.bar(
        [str(s) for s in score_order],
        percentages.values,
        width=0.6,
        color="#4C72B0"
    )

    ax.set_xlabel("Ordinal score", fontsize=10)
    ax.set_ylabel("Responses (%)", fontsize=10)

    ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    os.makedirs("output", exist_ok=True)
    fig.savefig("output/score_distribution.pdf", bbox_inches="tight")

    print("Saved: output/score_distribution.pdf")


if __name__ == "__main__":
    main()
