import subprocess
import sys
import time
from pathlib import Path

scripts = [
    "v0_1_sqlite_export/v0_1.py",
    "v0_2_cleaning/v0_2.py",
    "v0_3_silver_labels/v0_3.py",
    "v1_0_release/v1_0.py"
]

def run_pipeline():
    start_total = time.time()
    
    for script_path in scripts:
        script = Path(script_path)
        
        if not script.exists():
            print(f"File not found: {script.absolute()}")
            sys.exit(1)

        print(f"\nStarts: {script.name}...")
        start_script = time.time()
        
        result = subprocess.run(
            [sys.executable, script.name],
            cwd=script.parent
        )
        
        duration = time.time() - start_script

        if result.returncode == 0:
            print(f"{script.name} finished in {duration:.2f}s")
        else:
            print(f"Error in {script.name} (Exit Code: {result.returncode})")
            sys.exit(1)

    total_duration = time.time() - start_total
    print(f"\nPipeline succesfull in {total_duration:.2f}s")

if __name__ == "__main__":
    run_pipeline()