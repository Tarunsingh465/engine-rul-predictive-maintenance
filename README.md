# Predictive Maintenance: Remaining Useful Life (RUL) of Turbofan Engines

An end-to-end machine-learning project on the NASA C-MAPSS benchmark: predict how many operating cycles an
engine has left before failure, flag engines that need maintenance, and explore them in a dashboard.

The point of the project is as much **how the results were obtained and checked** as the numbers themselves:
engine-level validation, model selection on held-out engines (never on the test set), several honest diagnostics,
and a documented list of experiments that did and did not help.

## Results (official NASA test set, final saved models)

| Dataset | Operating conditions / fault modes | Final model | Official RMSE | NASA score | RMSE, engines within 30 cycles of failure |
|---|---|---|---|---|---|
| FD001 | 1 / 1 | Ensemble: XGBoost, HistGradientBoosting, LSTM, CNN-LSTM, CNN-LSTM-attention (cap 125, window 60) | 12.73 | 263 | 3.12 |
| FD002 | 6 / 1 | Ensemble: XGBoost + HistGradientBoosting (cap 150, window 60) | 20.50 | 2595 | 3.18 |
| FD003 | 1 / 2 | Ensemble: XGBoost + HistGradientBoosting (cap 125, window 60) | 11.07 | 201 | 2.27 |
| FD004 | 6 / 2 | Ensemble: XGBoost + HistGradientBoosting (cap 150, window 60) | 20.37 | 2379 | 5.30 |

**How to read these numbers**
- "Official RMSE" compares predictions with the **uncapped** official test labels. Models are trained on RUL labels
  capped at 125 (or 150), so engines that truly have more life left than the cap are under-predicted by design.
  On FD002/FD004 about a quarter of test engines are in that group, which is why their official RMSE is higher.
  Comparing both sides capped at 125 gives 11.29 (FD002) and 12.73 (FD004); FD001 and FD003 give 11.54 and 9.38.
- Accuracy is best where it matters most: within 30 cycles of failure the error is roughly 2 to 5 cycles on all four sets.
- Results come from one training run per dataset. Run-to-run variation is on the order of a few tenths of an RMSE point,
  so treat the numbers as "about 12.7", not as exact values.
- Published results use differing protocols (for example whether test labels are capped), so these numbers are
  **not** directly comparable with leaderboards, and no claim of beating any published method is made.

## Method
1. **Per-operating-condition normalisation.** FD002/FD004 have 6 flight regimes that shift every sensor. Regimes are
   found with KMeans on the three operating settings, and each sensor is standardised within its regime.
2. **Sensor selection** (training data only): drop constant sensors and those with |correlation| < 0.05 with RUL.
3. **Piecewise-linear RUL target.** Early in life there is no visible degradation, so the target is flat at a cap.
4. **Sliding windows** of 60 cycles per engine, left-padded so short engines work.
5. **Models.** XGBoost and HistGradientBoosting on window features (last value, mean, std, min, max, slope, change,
   engine age); and, for FD001, an LSTM, a CNN-LSTM and a CNN-LSTM with attention (PyTorch), each averaged over 3 seeds.
6. **Ensembling.** Equal-weight average of the members.
7. **Validation without leakage.** Engines (not rows) are split 80/20. Neural networks early-stop on a separate small group
   of training engines so the validation engines stay unseen. Configurations are chosen on validation only, then the
   chosen one is refit on all training engines and scored once on the test set.
8. **Selection rule.** A single model must beat the ensemble by more than 0.5 validation RMSE to be preferred, because
   differences smaller than that are noise on ~20 validation engines.

## Experiments, including what did not work
- **Window length:** 60 beat 40 on every dataset.
- **Neural networks vs trees:** after fixing optimistic validation scores, tree models were as good or better on test on
  all four datasets. My networks are lightly tuned, so this says little about what well-tuned networks could do.
- **Ablation LSTM -> CNN-LSTM -> + attention (FD001):** no consistent winner; differences were smaller than run-to-run noise.
- **Higher RUL cap on FD002/FD004** (validated against raw labels): cap 150 cut official RMSE from 23.5 to 20.5 and from
  24.9 to 20.4, and cut the NASA score. **Cost:** for engines with 60 to 125 cycles left, predictions became about
  8 cycles optimistic on average (up from about 3.5), which is the risky direction. Near-failure accuracy was not hurt.
- **Per-engine baseline normalisation** (idea from the literature): made results worse, especially close to failure
  (FD003 11.07 -> 12.76). **Extra short-window features:** no consistent gain. Neither was adopted.

## Known limitations
- Mid-life optimism on FD002/FD004 (see above). A late prediction is the costly mistake, so a production system should
  add an asymmetric loss or prediction intervals.
- Some design choices (window length, which cap to deploy for FD004) were made after seeing test results, so test
  numbers are slightly optimistic even though every automatic selection used validation only.
- C-MAPSS is simulated data: no sensor drift, missing data or maintenance interventions. Real fleet data would need
  separate validation.
- Single split and single seed per configuration; no cross-validation over engines.

## Repository
```
run_experiments.py   sweep window / RUL cap / model, select on validation, refit, evaluate, save
app.py               Streamlit dashboard (dataset switcher, fleet table, per-engine view, benchmark tab)
diagnose.py          error analysis: effect of the RUL cap, error by remaining-life range
merge_summary.py     copy chosen experiment results into results/summary.csv
src/                 data.py (preprocessing, windows, features), models.py, predict.py, metrics.py, synthetic.py
```

## Setup and reproduction
```
pip install -r requirements.txt            # for GPU PyTorch use the command from pytorch.org
# put the NASA C-MAPSS files (train_/test_/RUL_FD00X.txt) in data/
python run_experiments.py --data_dir data --fds FD001 FD003 --windows 40 60 --caps 125 --models hgb xgb lstm cnn_lstm cnn_lstm_attn
python run_experiments.py --data_dir data --fds FD002 FD004 --windows 60 --caps 125 150 175 --val_mode test_like --val_max_rul 195 --models hgb xgb --out_dir exp_caps
python diagnose.py
streamlit run app.py
```
`python -m src.synthetic --out data_synthetic` creates fake data in the same format for a quick smoke test; never
report numbers from it.

Data: NASA Prognostics Center of Excellence, Turbofan Engine Degradation Simulation (C-MAPSS).
