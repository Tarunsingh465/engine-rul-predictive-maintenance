# Interview cheat sheet: RUL prediction project

## 30-second pitch
"I built a predictive-maintenance system on NASA's turbofan engine data. It predicts how many cycles an engine has left
before failure and flags engines for maintenance. I handled six different operating conditions with per-condition
normalisation, compared tree models and LSTM variants, and chose models on held-out engines to avoid leakage. On the
official test sets I get an RMSE of about 11 to 13 on the single-condition datasets and about 20 on the harder
multi-condition ones, and within 30 cycles of failure the error is only about 2 to 5 cycles."

## What to call the model (use real names, never "ens")
- FD001: an equal-weight ensemble of XGBoost, HistGradientBoosting and three LSTM variants (LSTM, CNN-LSTM, CNN-LSTM with attention).
- FD002, FD003, FD004: an equal-weight ensemble of XGBoost and HistGradientBoosting.
- Say: "a heterogeneous averaging ensemble, trees on window statistics plus sequence models on raw windows, where the
  validation set showed it helped."

## Final numbers (official test set)
| Dataset | Official RMSE | NASA score | RMSE within 30 cycles of failure |
|---|---|---|---|
| FD001 | 12.73 | 263 | 3.12 |
| FD002 | 20.50 | 2595 | 3.18 |
| FD003 | 11.07 | 201 | 2.27 |
| FD004 | 20.37 | 2379 | 5.30 |
Say "about 12.7", not "12.73": one run per dataset, and run-to-run variation is a few tenths.

## Likely questions and honest answers
**Walk me through the project.** Data and goal, preprocessing (condition normalisation, sensor selection, capped RUL target,
windows), models, validation by engine, selection on validation, refit, test, dashboard. Then results and limits.

**Why cap the RUL?** Early in life there is no visible degradation, so a flat target stops the model chasing noise. Cost: it
cannot predict beyond the cap. On FD002/FD004 about a quarter of test engines have more life than 125, which is why their
official RMSE is higher. Raising the cap to 150 cut official RMSE from about 23.5 to 20.5 and 24.9 to 20.4.

**What is the downside of the higher cap?** Mid-life predictions (60 to 125 cycles left) became about 8 cycles optimistic on
average. Near-failure accuracy was not hurt. I would mitigate it with an asymmetric loss or prediction intervals.

**Why split validation by engine, not by row?** Rows from one engine are near-duplicates. A row split leaks information and
gives falsely good scores.

**Why per-condition normalisation?** In FD002/FD004 six flight regimes shift every sensor more than wear does. Standardising
within each regime exposes the degradation signal.

**How did you avoid test-set leakage?** Preprocessing is fit on training engines only; all automatic selection used held-out
validation engines; the test set was scored once for the chosen configuration. Honest caveat: I looked at test results
while deciding what to try next (for example window length, and deploying cap 150 instead of 175 for FD004), so the
test numbers are slightly optimistic.

**Why an ensemble?** Trees see summary statistics and networks see the raw sequence, so their errors differ and averaging
reduced error on FD001. On the other datasets the networks were worse and validation chose the trees-only ensemble,
so the choice is made per dataset on validation.

**Why did trees beat the neural networks?** Probably small data (100 to 260 engines) and lightly tuned networks. I did not
test that explanation, so I would not claim networks are worse in general.

**What is the NASA score?** An asymmetric error that punishes late predictions more than early ones, because telling an
engine near failure that it is healthy is the costly mistake. It is sensitive to a few large errors, so I report RMSE as well.

**Why is the FD002/FD004 error higher?** Mostly the cap: engines with more than the cap left are under-predicted by design.
Both sides capped, FD002 is 11.29 and FD004 is 12.73, close to the single-condition sets.

**How accurate is it near failure?** Within 30 cycles of failure: 3.2 (FD002), 2.3 (FD003), 3.1 (FD001), 5.3 (FD004).

**Did you try the LSTM vs CNN-LSTM vs attention comparison?** Yes, as an ablation on FD001. No consistent winner; the
differences were smaller than run-to-run noise, so I make no claim that convolution or attention helps.

**What did not work?** Per-engine baseline normalisation (from the literature) made results worse, especially close to
failure (FD003 11.07 to 12.76). Extra short-window features gave no consistent gain. Neither was adopted.

**What would you improve?** Asymmetric loss and prediction intervals to cut late predictions, cross-validation over
engines, properly tuned sequence models, and validation on real fleet data.

**Is it overfitted?** Validation and test errors were close on all datasets, results were stable between runs, and
the split is by engine. I did not run engine-level cross-validation, so I cannot rule out mild overfitting.

**How does it compare with published results?** Roughly in the range of well-known deep-learning baselines, but protocols
differ (for example whether test labels are capped), so I do not claim to beat any specific paper.

**How would this work at a company like L&T?** The same pipeline applies to any asset with run-to-failure histories and
sensor streams. Real data would need handling of sensor drift, missing values and maintenance events, plus separate
validation, because C-MAPSS is simulated.

**Explain the dashboard.** Pick a dataset; the fleet table ranks engines by predicted remaining life with
critical/warning thresholds; the engine view shows predicted vs true RUL over the engine's life and normalised sensors;
the benchmark tab shows every configuration tried.

## Things not to say
- "My model is a CNN-LSTM" (it is an ensemble; CNN-LSTM was one candidate).
- "Error is 12.73" as if exact; "I beat paper X"; "neural networks are worse"; "the model is not overfitted" (say what you checked).
- Do not hide that FD002/FD004 official error is about 20 and why.

---

## Addendum: statistical / distributional depth (added after EDA + distribution analysis)
See STATISTICAL_APPENDIX.md for full numbers. Key points to have ready:

**"Walk me through your EDA."** Checked for missing values and duplicates (none, on all four). Checked
sensor-vs-cycle correlation within operating condition to sanity-check automatic sensor selection. Ran two
outlier detectors (global IQR, within-engine jump detection) twice - raw and after per-condition normalisation -
because on multi-condition data, a raw-unit check cannot distinguish a real anomaly from an engine simply
switching operating regime. Confirmed that on FD002 most apparent "outliers" were this artifact (25% to 9% after
normalising); found the opposite on FD004, which I attribute (untested) to its two overlapping fault modes
confusing the 6-cluster regime model.

**"What distribution does RUL/lifetime follow, and why does it matter?"** Tested Weibull, Lognormal and Normal
by maximum likelihood; Normal is formally rejected (KS p<0.05) for engine lifetime on all four datasets.
Lognormal or Weibull fits best, matching how reliability engineering treats time-to-failure.

**"Why RMSE, and why also quantile regression?"** Checked whether residuals were Normal - minimising RMSE is
only the maximum-likelihood loss under that assumption. Shapiro's test rejects normality on all four datasets:
FD002/FD003 show real skew (0.56-0.67), FD001/FD004 are closer to symmetric but heavy-tailed (positive excess
kurtosis everywhere, consistent with the rare large errors from engines beyond the RUL cap). That is the
mathematical case for quantile regression, on top of the empirical gain already shown.

**"What is quantile regression actually optimising?"** The pinball loss at level tau is the maximum-likelihood
loss under an Asymmetric Laplace distribution parameterised by tau - so picking a quantile is equivalent to
assuming a specific error distribution shaped by the chosen late/early cost ratio.

**"Were your quantile models actually calibrated?"** No, not initially - the nominal 80% interval covered only
30-54% of true values (overconfident), a known tree-based quantile regression issue (quantile crossing). Fixed
with conformalized quantile regression: reached 89% and 96% coverage on the two larger datasets. On the two
smaller ones (fewer calibration engines) it overcorrected to 100% coverage with a too-wide interval - a known
small-sample limitation of split conformal prediction, which I documented rather than hid.

**"Did you just use NASA's 6-regime number, or check it?"** Checked it with a silhouette-score sweep over
k=2..8 on the raw operating settings; both FD002 and FD004 independently picked k=6.

**Don't say:** "my data has no outliers" (say what you checked and found, including the FD004 open question);
"RMSE was the right choice" without the residual-distribution reasoning behind it; "I used Weibull because
papers do" (say you tested it against alternatives and it won).
