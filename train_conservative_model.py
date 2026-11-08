"""Train a SEPARATE, risk-adjusted ("conservative") model alongside the main deployed one: instead of predicting
average RUL, it predicts a LOW quantile (biased toward "early"), chosen from the cost ratio of a late vs an early
maintenance call (tau = cost_early / (cost_early + cost_late), see src/quantile.py). Does NOT touch or overwrite
the main models/<FD>_best.joblib -- saves to models/<FD>_conservative.joblib, a different file, and writes a
side-by-side comparison to results/conservative_comparison.csv for the dashboard to show both models' numbers.

Run:  python train_conservative_model.py --data_dir data --fds FD001 FD002 FD003 FD004 --cost_ratio 4
"""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import N_COND, Preprocessor, add_rul, build_windows, load_fd, tabular_features
from src.metrics import nasa_score, rmse
from src.quantile import cost_to_quantile, make_quantile_model
from src.quantile_predict import QuantilePredictor


def eval_model(pred, test_raw, rul_true, cap_for_capped_metric=125):
    p = pred.predict_raw(test_raw, last_only=True)["pred_rul"].to_numpy()
    mid = (rul_true >= 60) & (rul_true <= 125)  # the range where the main model's bias was documented
    return dict(rmse=rmse(rul_true, p), nasa_score=nasa_score(rul_true, p), mean_error=float(np.mean(p - rul_true)),
               pct_late=float(np.mean(p > rul_true) * 100),
               mean_error_midlife=(float(np.mean((p - rul_true)[mid])) if mid.any() else float("nan")),
               rmse_last30=(rmse(rul_true[rul_true < 30], p[rul_true < 30]) if (rul_true < 30).any() else float("nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--fds", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--cost_ratio", type=float, default=4.0,
                    help="how many times worse a late prediction is than an early one; sets the quantile "
                         "(tau = 1/(1+cost_ratio)) the conservative model predicts")
    ap.add_argument("--model", default="hgb", choices=["hgb", "xgb"])
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tau = cost_to_quantile(a.cost_ratio)
    print(f"cost_ratio={a.cost_ratio} -> predicting the tau={tau:.2f} quantile of RUL "
          f"(deliberately biased {int((0.5 - tau) * 100)} points toward 'early', relative to the median)")

    models_dir = Path(a.models_dir)
    rows = []
    for fd in a.fds:
        main_path = models_dir / f"{fd}_best.joblib"
        if not main_path.exists():
            print(f"\n===== {fd}: no main model at {main_path}, skipping =====")
            continue
        print(f"\n===== {fd} =====")
        main_pred = joblib.load(main_path)
        cap, window = main_pred.cap, main_pred.window  # match the deployed model exactly: fair, apples-to-apples

        train_raw, test_raw, rul_true = load_fd(a.data_dir, fd)
        pre = Preprocessor(N_COND[fd], seed=a.seed).fit(train_raw, cap)
        tr = add_rul(pre.transform(train_raw))
        Xtr, mtr = build_windows(tr, pre.cols_, window)
        Ftr = tabular_features(Xtr, mtr.cycle)
        y = np.minimum(mtr["rul"].to_numpy(), cap)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = make_quantile_model(a.model, tau, a.seed, a.n_estimators, a.n_jobs)
            m.fit(Ftr, y)  # refit on ALL training engines, same discipline as the main pipeline

        cons_pred = QuantilePredictor(fd, pre, window, cap, tau, m, cost_ratio=a.cost_ratio)
        joblib.dump(cons_pred, models_dir / f"{fd}_conservative.joblib")

        main_m = eval_model(main_pred, test_raw, rul_true, cap)
        cons_m = eval_model(cons_pred, test_raw, rul_true, cap)
        print(f"  {'metric':<20}{'main (accuracy-optimal)':>26}{'conservative (risk-adjusted)':>30}")
        for k in main_m:
            print(f"  {k:<20}{main_m[k]:>26.2f}{cons_m[k]:>30.2f}")
        rows.append(dict(fd=fd, tau=tau, cost_ratio=a.cost_ratio,
                         **{f"main_{k}": v for k, v in main_m.items()},
                         **{f"conservative_{k}": v for k, v in cons_m.items()}))

    if rows:
        out = Path("results")
        out.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "conservative_comparison.csv", index=False)
        print(f"\nSaved: {out}/conservative_comparison.csv, models/FDxxx_conservative.joblib")
    else:
        print("\nNo datasets had a main model to compare against.")


if __name__ == "__main__":
    main()
