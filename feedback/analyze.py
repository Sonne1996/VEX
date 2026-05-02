import pandas as pd

INPUT = "results.parquet"
OUTPUT = "results.txt"

df = pd.read_parquet(INPUT)

# --- Derived Metrics ---
DIM_COLS = ["diag", "ground", "align", "action", "spec", "tone"]
CONTENT_COLS = ["diag", "ground", "align", "action", "spec"]

df["mean"] = df[DIM_COLS].mean(axis=1)
df["content"] = df[CONTENT_COLS].mean(axis=1)
df["gap"] = df["tone"] - df["content"]
df["low_quality"] = ((df["diag"] <= 2) & (df["align"] <= 2)).astype(int)

# --- Group by Model ---
summary = df.groupby("model").mean(numeric_only=True)

summary = summary[["diag", "ground", "align", "action", "spec", "tone", "mean", "content", "gap", "low_quality"]]
summary = summary.rename(columns={"low_quality": "low_quality_rate"})
summary["low_quality_rate"] = summary["low_quality_rate"] * 100

summary = summary.round(2)

# --- Overall ---
overall_content = df["content"].mean()
overall_tone = df["tone"].mean()
overall_gap = df["gap"].mean()
overall_low = df["low_quality"].mean() * 100

# --- Write TXT ---
with open(OUTPUT, "w") as f:
    f.write("\n")
    f.write("FEEDBACK EVALUATION RESULTS\n")
    f.write("=" * 90 + "\n\n")

    f.write("PER MODEL RESULTS\n")
    f.write("-" * 90 + "\n")
    f.write(summary.to_string())
    f.write("\n\n")

    f.write("OVERALL SUMMARY\n")
    f.write("-" * 90 + "\n")
    f.write(f"Content Mean: {overall_content:.2f}\n")
    f.write(f"Tone Mean:    {overall_tone:.2f}\n")
    f.write(f"Gap:          {overall_gap:.2f}\n")
    f.write(f"Low Quality %: {overall_low:.2f}\n")

print("Saved:", OUTPUT)

# --- SANITY CHECK ---
try:
    sanity = pd.read_parquet("sanity.parquet")

    with open(OUTPUT, "a") as f:
        f.write("\n\n")
        f.write("SANITY CHECK\n")
        f.write("=" * 90 + "\n\n")

        f.write(sanity.to_string())
        f.write("\n")

except Exception as e:
    print("Sanity check skipped:", e)
