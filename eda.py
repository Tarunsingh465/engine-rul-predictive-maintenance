"""Exploratory data analysis and outlier detection for C-MAPSS, run BEFORE modelling choices are trusted.
Produces plots + a CSV flagging outlier readings; does not change run_experiments.py or any saved model.

Run:  python eda.py --data_dir data --fds FD001 FD002 FD003 FD004
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import N_COND, OP, SENS, Preprocessor, load_fd


def iqr_outliers(s, k=3.0):
    """Classic Tukey rule: flag points more than k * IQR beyond the 25th/75th percentile."""
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (s < lo) | (s > hi)


def jump_outliers(s, z=6.0):
    """Within one engine's own sensor trace: flag CYCLE-TO-CYCLE JUMPS that are large relative to the engine's
    typical jump size (robust MAD of first differences). Works on the differenced series, not the raw level, so
    it does not trip on a smooth monotonic degradation trend -- only on a sudden, inconsistent step."""
    d = s.diff()
    mad = (d - d.median()).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=s.index)
    return ((d - d.median()).abs() / (1.4826 * mad) > z).fillna(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--fds", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--out_dir", default="eda_results")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    profile_rows, outlier_rows = [], []

    for fd in a.fds:
        print(f"\n===== {fd}: EDA =====")
        train, test, rul_true = load_fd(a.data_dir, fd)
        n_units = train.unit.nunique()
        print(f"  train: {len(train)} rows, {n_units} engines, {train.cycle.max()} max cycles")
        print(f"  missing values in train: {int(train.isna().sum().sum())}  "
              f"| duplicate (unit, cycle) rows: {int(train.duplicated(['unit', 'cycle']).sum())}")

        # 1) sensor profile: range, std, and how many are ~constant (candidates the modelling step already drops)
        prof = train[SENS].agg(["min", "max", "mean", "std"]).T
        prof["near_constant"] = prof["std"] < 1e-6
        prof.insert(0, "fd", fd)
        profile_rows.append(prof.reset_index().rename(columns={"index": "sensor"}))
        print(f"  near-constant sensors ({int(prof['near_constant'].sum())}): "
              f"{list(prof.index[prof['near_constant']])}")

        # 2) outliers -- run on RAW sensor level AND on per-condition normalised values, and compare.
        # On multi-condition datasets (FD002/FD004), a raw-level check cannot tell a genuine anomaly apart from
        # an engine simply switching to a different (also normal) operating regime, which looks like a huge jump
        # in raw units. Normalising within each regime first (the same step run_experiments.py already does)
        # removes that artifact, so what's left over is a much more trustworthy outlier flag.
        live_sens = [s for s in SENS if train[s].std() > 1e-6]  # skip near-constant sensors (nothing to flag)

        def run_checks(df, cols):
            iqr_f = df[cols].apply(iqr_outliers)
            jump_f = df.groupby(train["unit"].values)[cols].transform(lambda s: jump_outliers(s))
            return iqr_f.any(axis=1), jump_f.any(axis=1)

        raw_iqr, raw_jump = run_checks(train, live_sens)
        print(f"  RAW sensor units  -- IQR flagged: {int(raw_iqr.sum())} ({100 * raw_iqr.mean():.1f}%)  "
              f"| jump flagged: {int(raw_jump.sum())} ({100 * raw_jump.mean():.1f}%)")

        pre_chk = Preprocessor(N_COND[fd], min_corr=0.0, seed=0).fit(train, cap=125)  # min_corr=0: keep all for EDA
        norm = pre_chk.transform(train)
        any_iqr, any_jump = run_checks(norm, pre_chk.cols_)
        print(f"  NORMALISED (per-condition z-score) -- IQR flagged: {int(any_iqr.sum())} ({100 * any_iqr.mean():.1f}%)  "
              f"| jump flagged: {int(any_jump.sum())} ({100 * any_jump.mean():.1f}%)")
        if N_COND[fd] > 1:
            print(f"  -> gap between raw and normalised flags is the effect of operating-condition switches "
                  f"being mistaken for outliers, not real anomalies")
        print(f"  flagged by BOTH, on normalised data (most trustworthy): {int((any_iqr & any_jump).sum())}")
        flagged = train.loc[any_iqr | any_jump, ["unit", "cycle"]].copy()
        flagged["fd"] = fd
        flagged["iqr_flag"] = any_iqr[any_iqr | any_jump].values
        flagged["jump_flag"] = any_jump[any_iqr | any_jump].values
        outlier_rows.append(flagged)

        # 3) is degradation visible yet? correlation of each sensor with cycle, within the dominant operating condition
        mask = np.ones(len(train), dtype=bool)
        if N_COND[fd] > 1:
            from sklearn.cluster import KMeans
            c = KMeans(N_COND[fd], n_init=10, random_state=0).fit_predict(train[OP])
            mask = c == pd.Series(c).mode()[0]
        live_in_mask = [s for s in live_sens if train.loc[mask, s].std() > 1e-6]  # also live within this regime
        corrs = train.loc[mask, live_in_mask].corrwith(train.loc[mask, "cycle"]).sort_values(key=abs, ascending=False)
        print("  sensors most correlated with cycle (within one operating condition, top 5):")
        print("   " + ", ".join(f"{s}={v:+.2f}" for s, v in corrs.head(5).items()))

        # plots
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        top_sensor = corrs.index[0]
        for u in train.unit.unique()[:12]:
            g = train[train.unit == u]
            ax[0, 0].plot(g.cycle, g[top_sensor], alpha=.5, lw=.8)
        ax[0, 0].set(title=f"{fd}: {top_sensor} over life (12 engines)", xlabel="cycle", ylabel=top_sensor)
        life = train.groupby("unit").cycle.max()
        ax[0, 1].hist(life, bins=20, color="#1f77b4")
        ax[0, 1].set(title=f"{fd}: engine lifetimes", xlabel="cycles to failure", ylabel="count")
        ax[1, 0].bar(range(len(corrs)), corrs.values, color="#2ca02c")
        ax[1, 0].set(title=f"{fd}: sensor correlation with cycle", xlabel="sensor (sorted)", ylabel="corr")
        flagged_pts = train.loc[any_iqr | any_jump]
        ax[1, 1].scatter(train.cycle, train[top_sensor], s=4, alpha=.15, label="all")
        ax[1, 1].scatter(flagged_pts.cycle, flagged_pts[top_sensor], s=10, color="red", label="flagged")
        ax[1, 1].set(title=f"{fd}: flagged points on {top_sensor}", xlabel="cycle", ylabel=top_sensor)
        ax[1, 1].legend()
        fig.tight_layout()
        fig.savefig(out / f"{fd}_eda.png", dpi=120)
        plt.close(fig)

    pd.concat(profile_rows, ignore_index=True).to_csv(out / "sensor_profile.csv", index=False)
    pd.concat(outlier_rows, ignore_index=True).to_csv(out / "flagged_rows.csv", index=False)
    print(f"\nSaved: {out}/sensor_profile.csv, {out}/flagged_rows.csv, {out}/FDxxx_eda.png")


if __name__ == "__main__":
    main()
