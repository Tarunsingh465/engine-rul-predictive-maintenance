# Statistical appendix: distributions, residuals, and why each modelling choice was made

This appendix is the "why", in statistical terms, behind choices in the main README. It is built from three
scripts: `eda.py` (outlier detection and sensor correlation), `distribution_analysis.py` (lifetime and residual
distributions, cluster validation), and `quantile_study.py` / `src/quantile.py` (quantile regression and
conformal calibration). Numbers below are from real runs on the official NASA data.

## 1. Engine lifetime distribution
Time-to-failure in reliability engineering is classically modelled with a Weibull or Lognormal distribution,
not a Normal one (failure processes are rarely symmetric and cannot go below zero). This was tested, not assumed:
Weibull, Lognormal and Normal were each fit to engine lifetimes by maximum likelihood and compared with AIC and
a Kolmogorov-Smirnov test.

| Dataset | Best fit (by AIC) | Normal KS p-value |
|---|---|---|
| FD001 | Lognormal | 0.039 (rejected) |
| FD002 | Lognormal | 0.003 (rejected) |
| FD003 | Weibull | 0.005 (rejected) |
| FD004 | Weibull | 0.025 (rejected) |

**The Normal distribution is formally rejected (p<0.05) for engine lifetime on all four datasets.** Lognormal or
Weibull always fits better by AIC, consistent with standard reliability-engineering treatment of failure times.

## 2. Residual distribution of the final models
Minimising RMSE (squared error) is the maximum-likelihood loss only when residuals are Normal. This was checked
directly on the saved models' test-set residuals (predicted - true RUL):

| Dataset | Skewness | Excess kurtosis | Shapiro-Wilk p-value | Normal rejected? |
|---|---|---|---|---|
| FD001 | 0.01 (symmetric) | 0.93 (heavy-tailed) | 0.016 | Yes |
| FD002 | -0.67 (skewed) | 1.53 (heavy-tailed) | <0.001 | Yes |
| FD003 | 0.56 (skewed) | 1.32 (heavy-tailed) | 0.008 | Yes |
| FD004 | -0.36 (near-symmetric) | 0.74 (heavy-tailed) | <0.001 | Yes |

**Residuals are not Normal on any dataset.** FD002 and FD003 are meaningfully skewed; FD001 and FD004 are closer
to symmetric but still heavy-tailed (positive excess kurtosis everywhere). The heavy tails are consistent with
the earlier finding (`diagnose.py`) that a minority of engines beyond the RUL cap produce large occasional
errors, which inflates kurtosis without necessarily skewing the whole distribution. Both the skew (FD002/FD003)
and the heavy tails (FD001/FD004) are reasons RMSE alone understates the full picture, and both motivate the
quantile-regression and conformal-calibration work below rather than relying on RMSE in isolation.

## 3. Quantile regression and the pinball loss
Minimising the pinball loss at level tau is the maximum-likelihood loss under an **Asymmetric Laplace
distribution** parameterised by tau - this is the mathematical reason quantile regression is the right tool once
residuals are shown not to be Normal (section 2), not just an empirical trick. Quantile models were trained at
multiple tau and checked for calibration:

- **Raw quantile models were overconfident everywhere**: the nominal 80% interval (10th-90th percentile) actually
  covered only 30-54% of true test values across the four datasets, a known failure mode of tree-based quantile
  regression (quantile crossing/compression).
- **Conformalized quantile regression (CQR)** fixed this on the larger datasets: FD002 reached 89% coverage
  (target 80%) and FD004 reached 96%, at the cost of a wider interval. On the smaller datasets (FD001/FD003,
  ~10 calibration engines) the correction overshot to 100% coverage with an unusably wide interval - a known
  limitation of split conformal prediction with small calibration sets, not a flaw in the method itself.
- **Cost-based quantile choice**: the quantile that minimises expected cost is tau = c_early / (c_early +
  c_late). As the assumed late-vs-early cost ratio rises from 2x to 9x, the recommended operating point moves
  from the median toward the 10th percentile, and the late-prediction rate falls consistently (e.g. FD002: ~50%
  to 20%) at a documented RMSE cost - turning "how cautious should the model be" into a calculable business
  decision rather than a guess.

## 4. Validating the 6-operating-condition assumption
`run_experiments.py` assumes 6 operating regimes for FD002/FD004, following NASA's own documentation. This was
checked independently with a silhouette-score sweep over k=2..8 clusters on the raw operating settings:

| Dataset | Silhouette-optimal k | NASA-documented k |
|---|---|---|
| FD002 | 6 | 6 |
| FD004 | 6 | 6 |

Both datasets independently converge on exactly 6, confirming the assumption statistically rather than only by
citation.

## 5. Outlier detection and the operating-condition artifact
Two outlier checks (global IQR on sensor level, and a within-engine cycle-to-cycle jump detector) were run twice:
once on raw sensor units, once after the pipeline's own per-condition normalisation.

| Dataset | Raw jump-flag rate | Normalised jump-flag rate |
|---|---|---|
| FD001 (1 condition) | 0.01% | 0.01% (unchanged, as expected - normalisation is a no-op for one condition) |
| FD003 (1 condition) | 1.6% | 1.6% (unchanged, as expected) |
| FD002 (6 conditions) | 25.4% | **8.9%** (confirms: most "jumps" were engines switching operating regime, not anomalies) |
| FD004 (6 conditions) | 25.3% | **44.2%** (the opposite direction - see note below) |

FD001/FD003 being unchanged is a useful internal check: for single-condition data, z-scoring cannot change which
points an IQR or jump detector flags, so identical numbers confirm the method is behaving correctly. FD002
behaved as hypothesised. FD004 did not - a plausible explanation is that FD004 uniquely has two overlapping
fault modes, so the 6-cluster regime model (fit only on operating settings) mixes two different degradation
populations within each cluster, which can inflate rather than remove apparent outliers. This is a documented
open question, not a resolved one; the next step would be clustering on a richer feature set that can separate
fault mode from operating regime.
