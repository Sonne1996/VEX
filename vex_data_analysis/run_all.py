import subprocess

scripts = [
    "scripts/01_score_distribution.py",
    "scripts/02_student_coverage.py",
    "scripts/03_student_categories.py",
    "scripts/04_engagement_over_time.py",
    "scripts/05_dataset_statistics.py",
]

for script in scripts:
    print(f"Running {script}")
    subprocess.run(["python", script], check=True)

print("Done.")
