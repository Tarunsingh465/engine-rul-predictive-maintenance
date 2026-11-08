"""Statistical/distributional analysis: the "why" behind the modelling choices, with the math made explicit.

1. ENGINE LIFETIME DISTRIBUTION. In reliability engineering, time-to-failure is classically modelled with a
   Weibull distribution (its hazard rate can increase with age, matching wear-out failure). We fit Weibull,
   Lognormal and Normal to the engines' lifetimes by maximum likelihood and compare them with a KS test and AIC,
   instead of assuming one.
2. RESIDUAL DISTRIBUTION of the final saved model. Minimising RMSE (squared error) is the maximum-likelihood
   loss ONLY if residuals are Normal. If they are skewed, that is a mathematical justification (not just an
   empirical one) for using quantile regression instead, which connects to the next point.
3. PINBALL LOSS <-> ASYMMETRIC LAPLACE DISTRIBUTION. Minimising the pinball loss at level tau is equivalent to
   maximum-likelihood estimation under an Asymmetric Laplace distribution with that same tau. This is the
   mathematical reason quantile regression works, not a heuristic.
4. OPERATING-CONDITION CLUSTERS. run_experiments.py assumes 6 clusters for FD002/FD004 (from NASA's own
   documentation). We check that statistically with a silhouette-score sweep over k, instead of only trusting it.

Does NOT touch run_experiments.py, saved models, or any other script. Needs a trained model
(models/<FD>_best.joblib) for part 2; parts 1, 3 and 4 only need the raw data.

Run:  python distribution_analysis.py --data_dir data --fds FD001 FD002 FD003 FD004
"""
import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data import N_COND, OP, load_fd


def fit_lifetime_distributions(life):
    """Fit Weibull, Lognormal and Normal to engine lifetimes by MLE; compare with KS test p-value and AIC
    (lower AIC = better fit after penalising for parameter count; higher KS p-value = more consistent with
    that distribution -- NOT proof it IS that distribution, just that it isn't rejected)."""
    candidates = {"Weibull": (stats.weibull_min, 3), "Lognormal": (stats.lognorm, 3), "Normal": (stats.norm, 2)}
    rows = []
    for name, (dist, k) in candidates.items():
        params = dist.fit(life)
        ll = np.sum(dist.logpdf(life, *params))
        aic = 2 * k - 2 * ll
        ks_stat, ks_p = stats.kstest(life, dist.cdf, args=params)
        rows.append(dict(distribution=name, params=tuple(round(float(p), 3) for p in params),
                         log_likelihood=round(ll, 1), AIC=round(aic, 1), ks_p_value=round(ks_p, 4)))
    df = pd.DataFrame(rows).sort_values("AIC")
    return df, candidates


def residual_stats(resid):
    """Skewness and excess kurtosis of the residuals, plus a normality test. Normal: skew=0, kurtosis=0."""
    sh_stat, sh_p = stats.shapiro(resid) if len(resid) <= 5000 else (np.nan, np.nan)
    return dict(n=len(resid), mean=round(float(np.mean(resid)), 2), skewness=round(float(stats.skew(resid)), 2),
                excess_kurtosis=round(float(stats.kurtosis(resid)), 2),
                shapiro_p_value=(round(float(sh_p), 4) if np.isfinite(sh_p) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--fds", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--out_dir", default="distribution_results")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    life_rows, resid_rows, cluster_rows = [], [], []

    for fd in a.fds:
        print(f"\n===== {fd} =====")
        train, test, rul_true = load_fd(a.data_dir, fd)

        # 1) lifetime distribution
        life = train.groupby("unit").cycle.max().to_numpy(dtype=float)
        fits, candidates = fit_lifetime_distributions(life)
        fits.insert(0, "fd", fd)
        life_rows.append(fits)
        print("  Engine lifetime distribution fit (lower AIC = better; KS p>0.05 = not rejected):")
        print("  " + fits.to_string(index=False).replace("\n", "\n  "))
        best = fits.iloc[0]["distribution"]
        print(f"  -> best fit by AIC: {best}"
              + ("  (the classic reliability-engineering choice for failure times)" if best == "Weibull" else ""))

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(life, bins=20, density=True, alpha=.5, color="#1f77b4", label="observed")
        xs = np.linspace(life.min(), life.max(), 200)
        for name, (dist, _) in candidates.items():
            params = dist.fit(life)
            ax[0].plot(xs, dist.pdf(xs, *params), label=name)
        ax[0].set(title=f"{fd}: engine lifetime distribution", xlabel="cycles to failure", ylabel="density")
        ax[0].legend(fontsize=8)
        best_dist, _ = candidates[best]
        stats.probplot(life, dist=best_dist, sparams=best_dist.fit(life), plot=ax[1])
        ax[1].set_title(f"{fd}: Q-Q plot vs best fit ({best})")

        # 2) residual distribution of the saved model (if trained)
        mpath = Path(a.models_dir) / f"{fd}_best.joblib"
        if mpath.exists():
            pred = joblib.load(mpath)
            p = pred.predict_raw(test, last_only=True)["pred_rul"].to_numpy()
            resid = p - rul_true  # predicted - true, same convention used throughout this project
            rs = residual_stats(resid)
            rs["fd"] = fd
            resid_rows.append(rs)
            print(f"  Residual distribution (predicted - true, official test set): {rs}")
            if abs(rs["skewness"]) > 0.5:
                print(f"  -> |skew|={abs(rs['skewness']):.2f} > 0.5: residuals are meaningfully skewed, so RMSE "
                      f"(the right loss under Normal residuals) is not the theoretically ideal loss here -- "
                      f"this is the mathematical case for the quantile-regression work, not just an empirical one")
            else:
                print(f"  -> residuals are close to symmetric; RMSE is a reasonable loss choice here")

            fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
            ax2[0].hist(resid, bins=25, density=True, alpha=.6, color="#d62728")
            xs2 = np.linspace(resid.min(), resid.max(), 200)
            ax2[0].plot(xs2, stats.norm.pdf(xs2, resid.mean(), resid.std()), "k--", label="Normal fit")
            ax2[0].axvline(0, color="gray", lw=1)
            ax2[0].set(title=f"{fd}: residuals (pred - true)", xlabel="error (cycles)", ylabel="density")
            ax2[0].legend(fontsize=8)
            stats.probplot(resid, dist="norm", plot=ax2[1])
            ax2[1].set_title(f"{fd}: residual Q-Q plot vs Normal")
            fig2.tight_layout()
            fig2.savefig(out / f"{fd}_residuals.png", dpi=120)
            plt.close(fig2)
        else:
            print(f"  (no trained model at {mpath}, skipping residual analysis)")

        # 3) validate the operating-condition cluster count with silhouette score, instead of assuming it.
        # Only meaningful for datasets NASA documents as multi-regime (FD002/FD004, N_COND>1); on single-regime
        # datasets (FD001/FD003) the op settings only vary by sensor noise, so clustering them is meaningless.
        if N_COND[fd] > 1:
            ops = train[OP].to_numpy()
            ks = list(range(2, 9))
            sample = ops[np.random.default_rng(0).choice(len(ops), size=min(5000, len(ops)), replace=False)]
            sils = [silhouette_score(sample, KMeans(k, n_init=10, random_state=0).fit_predict(sample)) for k in ks]
            best_k = ks[int(np.argmax(sils))]
            cluster_rows.append(dict(fd=fd, best_k_by_silhouette=best_k, nasa_documented_k=N_COND[fd]))
            print(f"  Operating-condition clusters: silhouette score picks k={best_k} "
                  f"(NASA documents {N_COND[fd]}; scores: {[round(s, 3) for s in sils]})")
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            ax3.plot(ks, sils, "o-")
            ax3.axvline(best_k, color="red", ls="--", lw=1, label=f"best k={best_k}")
            ax3.set(title=f"{fd}: silhouette score vs k", xlabel="k (clusters)", ylabel="silhouette score")
            ax3.legend()
            fig3.tight_layout()
            fig3.savefig(out / f"{fd}_cluster_k.png", dpi=120)
            plt.close(fig3)

        fig.tight_layout()
        fig.savefig(out / f"{fd}_lifetime.png", dpi=120)
        plt.close(fig)

    pd.concat(life_rows, ignore_index=True).to_csv(out / "lifetime_distribution_fits.csv", index=False)
    if resid_rows:
        pd.DataFrame(resid_rows).to_csv(out / "residual_stats.csv", index=False)
    if cluster_rows:
        pd.DataFrame(cluster_rows).to_csv(out / "cluster_validation.csv", index=False)
    print(f"\nSaved to {out}/: lifetime_distribution_fits.csv, residual_stats.csv, cluster_validation.csv, "
          f"FDxxx_lifetime.png, FDxxx_residuals.png, FDxxx_cluster_k.png")


if __name__ == "__main__":
    main()
