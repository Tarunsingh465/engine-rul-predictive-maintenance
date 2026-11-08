"""Copy the chosen datasets' final results from experiment folders into results/summary.json + summary.csv.
Usage:  python merge_summary.py exp_caps:FD002 exp_fd004_150:FD004"""
import json
import sys
from pathlib import Path

import pandas as pd

main = Path("results/summary.json")
main.parent.mkdir(exist_ok=True)
allsum = json.loads(main.read_text()) if main.exists() else {}
for arg in sys.argv[1:]:
    folder, fd = arg.split(":")
    allsum[fd] = json.loads((Path(folder) / "results" / "summary.json").read_text())[fd]
allsum = dict(sorted(allsum.items()))
main.write_text(json.dumps(allsum, indent=2))
df = pd.DataFrame(allsum).T
df.to_csv("results/summary.csv")
cols = [c for c in ["best_model", "best_cap", "best_window", "val_rmse", "test_rmse", "test_rmse_capped",
                    "test_rmse_last30", "test_score"] if c in df.columns]
print(df[cols].round(2))
