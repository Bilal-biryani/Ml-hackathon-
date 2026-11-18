"""Re-run the complete pipeline in order (about 1-2 hours on a 16-core laptop)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["02_build_dataset.py", "01_eda.py", "03_train_compare.py", "04_bayes_opt.py",
          "05_hmm_eval.py", "05b_hmm_error_analysis.py", "06_maps.py", "07_build_report.py"]

if __name__ == "__main__":
    for s in STAGES:
        print(f"\n######## {s} ########", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / s)], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"{s} failed with exit code {r.returncode}")
