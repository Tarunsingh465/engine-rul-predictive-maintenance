"""Does the model actually perform worse on the points eda.py flagged as outliers?

Reuses the exact outlier-detection functions from eda.py (applied to NORMALISED test-set sensor values, the
version eda.py itself found most trustworthy), gets a prediction for EVERY cycle of every test engine (not just
the last one), reconstructs an approximate true RUL for each cycle the same way the dashboard does, and compares
prediction error on flagged rows against unflagged rows -- with a Mann-Whitney U test, since errors are not
Normal (see distribution_analysis.py), to check whether any difference is real or just noise.

Needs a trained model (models/<FD>_best.joblib). Does not retrain or change anything.

Run:  python outlier_impact.py --data_dir data --fds FD001 FD002 FD003 FD004
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from eda import iqr_outliers, jump_outliers
from src.data import load_fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--fds", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--out_dir", default="outlier_impact_results")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    rows = []

    for fd in a.fds:
        mpath = Path(a.models_dir) / f"{fd}_best.joblib"
        if not mpath.exists():
            print(f"\n===== {fd}: no trained model at {mpath}, skipping =====")
            continue
        print(f"\n===== {fd} =====")
        pred = joblib.load(mpath)
        _, test_raw, rul_true = load_fd(a.data_dir, fd)

        # 1) flag test-set rows, on NORMALISED values (same method eda.py found trustworthy), using the
        #    predictor's OWN fitted preprocessor so this matches exactly what the model actually saw
        norm = pred.pre.transform(test_raw)
        cols = pred.pre.cols_
        iqr_f = norm[cols].apply(iqr_outliers).any(axis=1)
        jump_f = norm.groupby(test_raw["unit"].values)[cols].transform(lambda s: jump_outliers(s)).any(axis=1)
        flagged = (iqr_f | jump_f).to_numpy()
        print(f"  {int(flagged.sum())} / {len(test_raw)} test rows flagged ({100 * flagged.mean():.1f}%)")

        # 2) a prediction for every cycle (not just the last), and an approximate true RUL for every cycle,
        #    reconstructed the same way the dashboard does: true_rul = true_at_last_cycle + (last_cycle - cycle),
        #    capped at the model's own cap so this compares like-for-like with what the model could ever predict
        #    (otherwise the already-documented, SEPARATE cap effect would dominate and hide the outlier effect)
        hist = pred.predict_raw(test_raw, last_only=False)
        last_cycle = test_raw.groupby("unit").cycle.max()
        true_last = pd.Series(rul_true, index=np.sort(test_raw.unit.unique()))
        hist["true_rul"] = np.minimum(
            true_last.loc[hist.unit].to_numpy() + (last_cycle.loc[hist.unit].to_numpy() - hist.cycle.to_numpy()),
            pred.cap)
        err = (hist["pred_rul"] - hist["true_rul"]).to_numpy()
        abs_err = np.abs(err)

        if len(flagged) != len(hist):
            print(f"  WARNING: row count mismatch ({len(flagged)} flags vs {len(hist)} predictions); skipping")
            continue

        e_flag, e_clean = abs_err[flagged], abs_err[~flagged]
        u_stat, p_val = stats.mannwhitneyu(e_flag, e_clean, alternative="two-sided") if (len(e_flag) and len(e_clean)) else (np.nan, np.nan)
        row = dict(fd=fd, n_flagged=int(flagged.sum()), n_clean=int((~flagged).sum()),
                  rmse_flagged=float(np.sqrt(np.mean(err[flagged] ** 2))) if flagged.any() else np.nan,
                  rmse_clean=float(np.sqrt(np.mean(err[~flagged] ** 2))),
                  mean_abs_err_flagged=float(e_flag.mean()) if len(e_flag) else np.nan,
                  mean_abs_err_clean=float(e_clean.mean()),
                  mann_whitney_p=float(p_val) if np.isfinite(p_val) else None)
        rows.append(row)
        print(f"  RMSE on flagged rows:   {row['rmse_flagged']:.2f}  (n={row['n_flagged']})")
        print(f"  RMSE on unflagged rows: {row['rmse_clean']:.2f}  (n={row['n_clean']})")
        if row["mann_whitney_p"] is not None:
            sig = "YES - a real difference" if row["mann_whitney_p"] < 0.05 else "no - not statistically distinguishable"
            print(f"  Mann-Whitney U test p-value: {row['mann_whitney_p']:.4f}  -> significant difference? {sig}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out / "outlier_impact.csv", index=False)
        print("\n=== Summary: does prediction error differ on flagged vs unflagged rows? ===")
        print(df.round(3).to_string(index=False))
        print(f"\nSaved: {out}/outlier_impact.csv")
    else:
        print("\nNo datasets had both a trained model and matching data -- nothing to summarise.")


if __name__ == "__main__":
    main()
