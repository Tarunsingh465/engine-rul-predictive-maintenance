"""Where does the test error come from?  Reads results/FD00X_test_predictions.csv (made by run_experiments.py).
Run:  python diagnose.py            (from the project folder)"""
from pathlib import Path

import numpy as np
import pandas as pd

CAP = 125
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
rows, buckets = [], []
for f in sorted(Path("results").glob("FD*_test_predictions.csv")):
    fd = f.name.split("_")[0]
    d = pd.read_csv(f)
    t, p = d.true_rul.to_numpy(), d.pred_rul.to_numpy()
    low = t <= CAP
    rows.append(dict(dataset=fd, engines=len(d), true_above_cap=int((~low).sum()),
                     rmse_official=rm(p - t), rmse_true_clipped_at_cap=rm(p - np.minimum(t, CAP)),
                     rmse_both_clipped=rm(np.minimum(p, CAP) - np.minimum(t, CAP)),  # same measure run_experiments prints as "capped"
                     rmse_only_engines_within_cap=rm((p - t)[low]),
                     mean_error_all=float(np.mean(p - t))))
    for name, m in [("0-30", t < 30), ("30-60", (t >= 30) & (t < 60)), ("60-125", (t >= 60) & (t <= CAP)), (">125", t > CAP)]:
        if m.sum():
            buckets.append(dict(dataset=fd, true_rul=name, engines=int(m.sum()), rmse=rm((p - t)[m]), mean_error=float(np.mean((p - t)[m]))))
pd.set_option("display.width", 200)
print("\n=== Test RMSE under different treatments of the RUL cap ===")
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\n=== Error by true remaining life (mean_error > 0 means predicted too optimistic / late) ===")
print(pd.DataFrame(buckets).round(2).to_string(index=False))
