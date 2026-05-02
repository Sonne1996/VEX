import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


DB_PATH = "input/data.sqlite"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT created_at FROM shared_member_answers"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def main():
    df = load_data()

    if df.empty:
        print("No data found.")
        return

    # UNIX → datetime
    df["created_at"] = pd.to_datetime(df["created_at"], unit="s")

    # Startdatum
    start_date = df["created_at"].min()

    # Wochen berechnen
    df["week"] = ((df["created_at"] - start_date).dt.days // 7) + 1

    print("Max week in dataset:", df["week"].max())

    # auf 18 Wochen begrenzen
    df = df[df["week"] <= 18]

    # Aggregation
    counts = df.groupby("week").size()
    counts = counts.reindex(range(1, 19), fill_value=0)

    # Plot
    fig, ax = plt.subplots(figsize=(3.25, 2.2))

    ax.bar(counts.index, counts.values, color="#4C72B0", width=0.8)

    ax.set_xlabel("Week", fontsize=10)
    ax.set_ylabel("Number of responses", fontsize=10)

    # 🔥 nur jede 2. Woche anzeigen
    xticks = list(range(1, 19, 2))  # 1,3,5,...17
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=8)

    # Grid
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    # Clean look
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    os.makedirs("output", exist_ok=True)
    fig.savefig("output/engagement_over_time.pdf", bbox_inches="tight")

    print("Saved: output/engagement_over_time.pdf")


if __name__ == "__main__":
    main()
