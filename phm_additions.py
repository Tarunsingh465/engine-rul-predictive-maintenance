"""Industry-PHM-standard additions this project was missing: a physics-based baseline, feature-importance
explainability, a composite PCA health indicator, and a log-target experiment.

Does NOT change run_experiments.py or overwrite any saved *_best.joblib model; results go to a separate folder.

Run:  python phm_additions.py --data_dir data --fds FD001 FD002 FD003 FD004
"""
import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.data import N_COND, Preprocessor, add_rul, build_windows, load_fd, tabular_feature_names, tabular_features
from src.metrics import nasa_score, rmse
from src.models import make_tab


# ---------------------------------------------------------------------------------------------------------------
# 1) Baselines: is the real model actually earning its complexity?
# ---------------------------------------------------------------------------------------------------------------
def baseline_scores(train_raw, test_raw, rul_true, cap):
    """Two trivial baselines any real model must beat:
    - constant: predict the median training RUL for every test engine (ignores all sensor/cycle data)
    - physics-linear: predict (median training life - current cycle), i.e. assume every engine lives exactly the
      average lifetime and degrades linearly -- this is literally the assumption the capped-RUL target encodes,
      so beating it demonstrates the sensors are adding real information, not just the cap/window machinery."""
    med_life = train_raw.groupby("unit").cycle.max().median()
    last_cycle = test_raw.groupby("unit").cycle.max().to_numpy()
    const_pred = np.full(len(rul_true), min(med_life, cap))
    linear_pred = np.clip(med_life - last_cycle, 0, cap)
    return {
        "constant (median train life)": dict(rmse=rmse(rul_true, const_pred), score=nasa_score(rul_true, const_pred)),
        "physics-linear (median life - cycle)": dict(rmse=rmse(rul_true, linear_pred), score=nasa_score(rul_true, linear_pred)),
    }


# ---------------------------------------------------------------------------------------------------------------
# 2) Feature importance: which sensors / features actually drove the final model's predictions
# ---------------------------------------------------------------------------------------------------------------
def feature_importance_table(fd, pre, window, feat_version, cap, train_raw, val_frac=0.2, seed=0,
                             n_estimators=300, n_jobs=-1):
    """Train one interpretable HistGradientBoosting model (not the full ensemble) on the SAME preprocessing as
    the saved model and report its impurity/gain-style feature importances. A separate, simple model is used for
    importances rather than inspecting the saved ensemble directly, because the saved ensemble can mix tree
    models (which expose importances) with neural networks (which don't) -- this keeps the explanation honest
    about which model it describes."""
    tr = add_rul(pre.transform(train_raw))
    Xtr, mtr = build_windows(tr, pre.cols_, window)
    F = tabular_features(Xtr, mtr.cycle, feat_version)
    y = np.minimum(mtr["rul"].to_numpy(), cap)
    units = np.sort(train_raw["unit"].unique())
    val_units = np.random.default_rng(seed).choice(units, size=max(1, int(val_frac * len(units))), replace=False)
    isv = mtr["unit"].isin(val_units).to_numpy()
    m = make_tab("hgb", seed, n_estimators, n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(F[~isv], y[~isv])
    names = tabular_feature_names(pre.cols_, feat_version)
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(m, F[isv], y[isv], n_repeats=5, random_state=seed, n_jobs=1)
    df = pd.DataFrame({"feature": names, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std})
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------
# 3) Composite health indicator: the standard PHM alternative to raw multi-sensor input
# ---------------------------------------------------------------------------------------------------------------
def health_indicator(pre, train_raw):
    """A single 1-D 'degradation index' built with PCA on the (already per-condition-normalised) sensors: the
    first principal component, oriented so it INCREASES as the engine degrades (checked against cycle number).
    This is the classic PHM alternative/complement to feeding raw multi-sensor windows straight into a regressor."""
    norm = pre.transform(train_raw)
    pca = PCA(n_components=1, random_state=0).fit(norm[pre.cols_])
    hi_all = pca.transform(norm[pre.cols_])[:, 0]
    if np.corrcoef(hi_all, norm["cycle"])[0, 1] < 0:  # orient so HI rises with degradation, not falls
        pca.components_ *= -1
        hi_all = -hi_all
    explained = float(pca.explained_variance_ratio_[0])
    corr_with_cycle = float(np.corrcoef(hi_all, norm["cycle"])[0, 1])
    return pca, norm.assign(health_indicator=hi_all), explained, corr_with_cycle


# ---------------------------------------------------------------------------------------------------------------
# 4) Log-target experiment: engine lifetime was found to be closer to Lognormal than Normal -- does training on
#    log1p(RUL) instead of raw RUL improve the model?
# ---------------------------------------------------------------------------------------------------------------
def log_target_experiment(pre, window, feat_version, cap, train_raw, test_raw, rul_true, seed=0,
                          n_estimators=300, n_jobs=-1, val_frac=0.2):
    tr = add_rul(pre.transform(train_raw))
    te = pre.transform(test_raw)
    Xtr, mtr = build_windows(tr, pre.cols_, window)
    Xte, mte = build_windows(te, pre.cols_, window, last_only=True)
    Ftr, Fte = tabular_features(Xtr, mtr.cycle, feat_version), tabular_features(Xte, mte.cycle, feat_version)
    units = np.sort(train_raw["unit"].unique())
    val_units = np.random.default_rng(seed).choice(units, size=max(1, int(val_frac * len(units))), replace=False)
    isv = mtr["unit"].isin(val_units).to_numpy()
    y_raw = np.minimum(mtr["rul"].to_numpy(), cap)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, ytr, back in [("raw target", y_raw, lambda p: p),
                                ("log1p target", np.log1p(y_raw), lambda p: np.expm1(p))]:
            m = make_tab("hgb", seed, n_estimators, n_jobs)
            m.fit(Ftr[~isv], ytr[~isv])
            pv = np.clip(back(m.predict(Ftr[isv])), 0, cap)
            pt = np.clip(back(m.predict(Fte)), 0, cap)
            yv = y_raw[isv]
            rows.append(dict(target=name, val_rmse=rmse(yv, pv), test_rmse=rmse(rul_true, pt),
                             test_score=nasa_score(rul_true, pt)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--fds", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--feat_version", type=int, default=1)
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="phm_results")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    all_baseline, all_log = [], []

    for fd in a.fds:
        print(f"\n===== {fd} =====")
        cap = a.cap
        info_path = Path(a.models_dir) / f"{fd}_best.joblib"
        if info_path.exists():  # reuse the real deployed cap/window so comparisons are apples-to-apples
            saved = joblib.load(info_path)
            cap, window = saved.cap, saved.window
        else:
            window = a.window
        train_raw, test_raw, rul_true = load_fd(a.data_dir, fd)
        pre = Preprocessor(N_COND[fd], seed=a.seed).fit(train_raw, cap)

        # 1) baselines
        print(f"  --- Baselines (cap={cap}) ---")
        base = baseline_scores(train_raw, test_raw, rul_true, cap)
        for name, v in base.items():
            print(f"  {name:<38} RMSE={v['rmse']:6.2f}  score={v['score']:8.0f}")
            all_baseline.append(dict(fd=fd, baseline=name, **v))
        if info_path.exists():
            real_rmse = rmse(rul_true, saved.predict_raw(test_raw, last_only=True)["pred_rul"].to_numpy())
            print(f"  {'deployed model':<38} RMSE={real_rmse:6.2f}  "
                  f"(beats constant baseline by {base['constant (median train life)']['rmse'] - real_rmse:+.1f}, "
                  f"beats physics baseline by {base['physics-linear (median life - cycle)']['rmse'] - real_rmse:+.1f})")

        # 2) feature importance
        print("  --- Feature importance (permutation, on a HistGradientBoosting model) ---")
        fi = feature_importance_table(fd, pre, window, a.feat_version, cap, train_raw, seed=a.seed,
                                      n_estimators=a.n_estimators, n_jobs=a.n_jobs)
        fi.insert(0, "fd", fd)
        fi.to_csv(out / f"{fd}_feature_importance.csv", index=False)
        print("  top 8 features:\n  " + fi.head(8).drop(columns="fd").to_string(index=False).replace("\n", "\n  "))
        fig, ax = plt.subplots(figsize=(7, 4))
        top = fi.head(12).iloc[::-1]
        ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#2ca02c")
        ax.set(title=f"{fd}: top 12 features (permutation importance)", xlabel="importance (drop in R^2 when shuffled)")
        fig.tight_layout()
        fig.savefig(out / f"{fd}_feature_importance.png", dpi=120)
        plt.close(fig)

        # 3) health indicator
        pca, hi_df, explained, corr = health_indicator(pre, train_raw)
        print(f"  --- Health indicator (PCA, 1 component) ---")
        print(f"  explains {explained * 100:.1f}% of normalised-sensor variance; "
              f"correlation with cycle (degradation proxy) = {corr:+.2f}")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        for u in hi_df.unit.unique()[:15]:
            g = hi_df[hi_df.unit == u]
            ax2.plot(g.cycle, g.health_indicator, alpha=.5, lw=.8)
        ax2.set(title=f"{fd}: PCA health indicator over engine life (15 engines)", xlabel="cycle",
               ylabel="health indicator (higher = more degraded)")
        fig2.tight_layout()
        fig2.savefig(out / f"{fd}_health_indicator.png", dpi=120)
        plt.close(fig2)

        # 4) log-target experiment
        print("  --- Log-target experiment (raw RUL vs log1p(RUL), same HistGradientBoosting model) ---")
        lt = log_target_experiment(pre, window, a.feat_version, cap, train_raw, test_raw, rul_true, seed=a.seed,
                                   n_estimators=a.n_estimators, n_jobs=a.n_jobs)
        lt.insert(0, "fd", fd)
        all_log.append(lt)
        print("  " + lt.drop(columns="fd").to_string(index=False).replace("\n", "\n  "))

    pd.DataFrame(all_baseline).to_csv(out / "baselines_summary.csv", index=False)
    pd.concat(all_log, ignore_index=True).to_csv(out / "log_target_summary.csv", index=False)
    print(f"\nSaved to {out}/: baselines_summary.csv, log_target_summary.csv, FDxxx_feature_importance.csv/png, "
          f"FDxxx_health_indicator.png")


if __name__ == "__main__":
    main()
