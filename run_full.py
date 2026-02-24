"""
run the full mess
"""
import subprocess
import sys
from pathlib import Path

from config import ensure_dirs

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    ensure_dirs()
    scripts = [
        "1_download_data.py",
        "2_build_graph.py",
        "3_run_walks.py",
        "4_train_embeddings.py",
        "5_project_3d.py",
    ]
    for script in scripts:
        path = PROJECT_ROOT / script
        print(f"\n--- {script} ---")
        result = subprocess.run([sys.executable, str(path)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"ERROR: {script} exited with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)
    print("\nPipeline complete. Output: output/layout.tsv")


if __name__ == "__main__":
    main()
