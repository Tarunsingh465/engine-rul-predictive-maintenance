"""Distributional / uncertainty-quantification study for RUL prediction.

Instead of a single point prediction, this trains a FAMILY of quantile models per dataset and checks:
  1. Calibration: does the tau-quantile model's prediction actually get exceeded (1-tau) of the time?
  2. Prediction intervals: how wide are they, and how often do they really contain the truth (PICP)?
  3. Cost-based quantile choice: given a cost ratio for late vs early maintenance, which quantile minimises
     expected cost (tau = c_early / (c_early + c_late)), and how does the resulting bias compare with the
     existing point model?
Does NOT touch run_experiments.py or your saved *_best.joblib models -- this is a separate, additional analysis.

Run:  python quantile_study.py --data_dir data --fds FD001 FD002 FD003 FD004
"""
import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import N_COND, Preprocessor, add_rul, build_windows, load_fd, tabular_features
from src.metrics import rmse
from src.quantile import (
    conformalize_interval, coverage, cost_to_quantile, interval_stats, make_quantile_model, pinball_loss,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--fds", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--cap", type=int, default=125, help="target cap used for TRAINING the quantile models")
    ap.add_argument("--quantiles", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
    ap.add_argument("--cost_ratios", nargs="+", type=float, default=[2, 3, 4, 6, 9],
                    help="how many times worse a late prediction is than an early one, by this many multiples")
    ap.add_argument("--model", default="hgb", choices=["hgb", "xgb"])
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--min_val_cycle", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="quantile_results")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    rows, cost_rows, conformal_rows = [], [], []

    for fd in a.fds:
        print(f"\n===== {fd}: distributional study (quantiles trained at cap={a.cap}) =====")
        train_raw, test_raw, rul_true = load_fd(a.data_dir, fd)
        units = np.sort(train_raw["unit"].unique())
        rng = np.random.default_rng(a.seed)
        val_units = rng.choice(units, size=max(1, int(a.val_frac * len(units))), replace=False)
        # the validation engines are split in two: half to check calibration, half held out purely for
        # conformal interval calibration, so the final test-set PICP is an honest, un-peeked-at number
        cal_units = rng.choice(val_units, size=max(1, len(val_units) // 2), replace=False)
        val_units_check = np.setdiff1d(val_units, cal_units)

        pre = Preprocessor(N_COND[fd], seed=a.seed).fit(train_raw, a.cap)
        tr = add_rul(pre.transform(train_raw))
        te = pre.transform(test_raw)
        Xtr, mtr = build_windows(tr, pre.cols_, a.window)
        Xte, mte = build_windows(te, pre.cols_, a.window, last_only=True)
        Ftr, Fte = tabular_features(Xtr, mtr.cycle), tabular_features(Xte, mte.cycle)
        y = np.minimum(mtr["rul"].to_numpy(), a.cap)
        isv = mtr["unit"].isin(val_units).to_numpy()
        vsel = isv & mtr["unit"].isin(val_units_check).to_numpy() & (mtr["cycle"].to_numpy() >= a.min_val_cycle)
        csel = isv & mtr["unit"].isin(cal_units).to_numpy() & (mtr["cycle"].to_numpy() >= a.min_val_cycle)
        yv_raw = mtr["rul"].to_numpy()[vsel].astype(float)  # RAW (uncapped) validation target, like the real test metric
        yc_raw = mtr["rul"].to_numpy()[csel].astype(float)  # separate conformal-calibration target

        preds_va, preds_te, preds_ca, pinballs = {}, {}, {}, {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tau in a.quantiles:
                m = make_quantile_model(a.model, tau, a.seed, a.n_estimators, a.n_jobs)
                m.fit(Ftr[~isv], y[~isv])
                pv, pt = m.predict(Ftr[vsel]), np.clip(m.predict(Fte), 0, a.cap)
                pc = m.predict(Ftr[csel])  # held-out calibration predictions, used only for conformal widening
                preds_va[tau], preds_te[tau], preds_ca[tau] = pv, pt, pc
                pinballs[tau] = pinball_loss(yv_raw, pv, tau)
                cov = coverage(yv_raw, pv, tau)
                rows.append(dict(fd=fd, tau=tau, pinball_val=pinballs[tau], coverage_val=round(cov, 3),
                                 target_coverage=tau, calibration_gap=round(cov - tau, 3),
                                 rmse_test=rmse(rul_true, pt), mean_error_test=float(np.mean(pt - rul_true))))
                print(f"  tau={tau:<4} coverage={cov:.2f} (target {tau:.2f})  pinball={pinballs[tau]:7.3f}  "
                      f"test_rmse={rows[-1]['rmse_test']:6.2f}  mean_error={rows[-1]['mean_error_test']:+6.2f}")

        # 10-90 interval sanity check: does the interval really contain ~80% of true values?
        if 0.1 in preds_va and 0.9 in preds_va:
            st = interval_stats(yv_raw, preds_va[0.1], preds_va[0.9])
            print(f"  [10%,90%] interval (raw):        PICP={st['picp']:.2f} (target 0.80)  mean width={st['mean_width']:.1f} cycles")
            lo_adj, hi_adj, Q = conformalize_interval(preds_ca[0.1], preds_ca[0.9], yc_raw,
                                                       preds_te[0.1], preds_te[0.9], alpha=0.2)
            st_test_raw = interval_stats(rul_true, preds_te[0.1], preds_te[0.9])
            st_test_adj = interval_stats(rul_true, lo_adj, hi_adj)
            print(f"  [10%,90%] interval on TEST, before calibration: PICP={st_test_raw['picp']:.2f}  width={st_test_raw['mean_width']:.1f}")
            print(f"  [10%,90%] interval on TEST, after  calibration: PICP={st_test_adj['picp']:.2f}  width={st_test_adj['mean_width']:.1f}  (widened by Q={Q:.1f} cycles each side)")
            conformal_rows.append(dict(fd=fd, picp_test_raw=st_test_raw["picp"], width_test_raw=st_test_raw["mean_width"],
                                       picp_test_calibrated=st_test_adj["picp"], width_test_calibrated=st_test_adj["mean_width"], Q=Q))

        # which quantile would a given late/early cost ratio recommend, and what bias does that give on test?
        for ratio in a.cost_ratios:
            tau = cost_to_quantile(ratio)
            nearest = min(a.quantiles, key=lambda q: abs(q - tau))
            pt = preds_te[nearest]
            cost_rows.append(dict(fd=fd, cost_ratio=f"{ratio:.0f}x", tau_optimal=round(tau, 2), tau_used=nearest,
                                  rmse_test=rmse(rul_true, pt), mean_error_test=float(np.mean(pt - rul_true)),
                                  pct_late=float(np.mean(pt > rul_true) * 100)))

        # reliability diagram + interval plot
        taus = sorted(preds_va)
        cov_obs = [coverage(yv_raw, preds_va[t], t) for t in taus]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax[0].plot(taus, cov_obs, "o-", color="#d62728")
        ax[0].set(xlabel="Target quantile (tau)", ylabel="Observed coverage", title=f"{fd}: calibration",
                  xlim=(0, 1), ylim=(0, 1))
        ax[0].legend()
        if 0.1 in preds_te and 0.9 in preds_te:
            o = np.argsort(rul_true)
            ax[1].fill_between(np.arange(len(o)), preds_te[0.1][o], preds_te[0.9][o], alpha=.25, color="#1f77b4",
                               label="10-90% interval")
            ax[1].plot(rul_true[o], color="k", lw=1.2, label="true RUL")
            ax[1].set(xlabel="Engine (sorted by true RUL)", ylabel="RUL (cycles)", title=f"{fd}: prediction interval")
            ax[1].legend()
        fig.tight_layout()
        fig.savefig(out / f"{fd}_quantile_study.png", dpi=120)
        plt.close(fig)

    cal = pd.DataFrame(rows)
    cal.to_csv(out / "calibration.csv", index=False)
    cst = pd.DataFrame(cost_rows)
    cst.to_csv(out / "cost_based_quantile_choice.csv", index=False)
    if conformal_rows:
        cf = pd.DataFrame(conformal_rows)
        cf.to_csv(out / "conformal_calibration.csv", index=False)
    print("\n=== Calibration summary (coverage should be close to target_coverage / tau) ===")
    print(cal[["fd", "tau", "target_coverage", "coverage_val", "calibration_gap", "mean_error_test"]].to_string(index=False))
    print("\n=== Which quantile a given late/early cost ratio recommends, and its effect on test bias ===")
    print(cst.to_string(index=False))
    if conformal_rows:
        print("\n=== Conformal calibration: does the widened interval reach 80% coverage on the TEST set? ===")
        print(cf.round(2).to_string(index=False))
    print(f"\nSaved: {out}/calibration.csv, {out}/cost_based_quantile_choice.csv, {out}/FDxxx_quantile_study.png")


if __name__ == "__main__":
    main()
