"""Sweep window size / RUL cap / model on each FD set, pick the best config by VALIDATION RMSE
(never by test), refit it on all training engines, evaluate on the official test set, save everything.

    python run_experiments.py --data_dir data --fds FD001 FD002 FD003 FD004
    python run_experiments.py --data_dir data --fds FD001 --quick
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import N_COND, Preprocessor, add_rul, build_windows, load_fd, tabular_features
from src.metrics import nasa_score, rmse
from src.models import DL_MODELS, TAB_MODELS, make_tab
from src.predict import Predictor


CAP_REPORT = 125  # the cap used only for the extra "capped" comparison metric


def fit_member(name, Xw, F, y, cap, seed, args, Xw_va=None, F_va=None, y_va=None, epochs=None):
    """Fit one model. Returns (kind, fitted_model, n_epochs_or_None)."""
    if name in TAB_MODELS:
        m = make_tab(name, seed, args.n_estimators, args.n_jobs, strong_reg=args.strong_reg)
        m.fit(F, np.log1p(y) if args.log_target else y)  # log1p(RUL): lifetimes were found Lognormal, see
                                                         # distribution_analysis.py; expm1 undoes it at predict time
        return "tab", m, None
    from src.models import AvgNet, fit_dl
    ep = epochs or args.epochs
    nets, eps = [], []
    for k in range(args.dl_seeds):  # average several seeds: deep nets are noisy run to run
        net, best_ep = fit_dl(Xw, y, cap, Xw_va, y_va, epochs=ep, seed=seed + k, arch=name)
        nets.append(net)
        eps.append(best_ep)
    return "dl", (AvgNet(nets).eval() if len(nets) > 1 else nets[0]), int(round(float(np.mean(eps))))


def predict_member(kind, m, Xw, F, cap, log_target=False):
    if kind == "tab":
        p = m.predict(F)
        return np.expm1(p) if log_target else p
    from src.models import dl_predict
    return dl_predict(m, Xw, cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--fds", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--windows", nargs="+", type=int, default=[20, 30, 40])
    ap.add_argument("--caps", nargs="+", type=int, default=[110, 125, 140])
    ap.add_argument("--models", nargs="+", default=TAB_MODELS + DL_MODELS)
    ap.add_argument("--smooth", type=int, default=0, help="EWM span for sensor smoothing (0=off)")
    ap.add_argument("--baseline_cycles", type=int, default=0,
                    help="subtract each engine's own mean over its first N cycles (0=off; try 10)")
    ap.add_argument("--feat_version", type=int, default=1, choices=[1, 2],
                    help="2 = add last-10-cycle statistics to the tree features")
    ap.add_argument("--val_mode", default="capped", choices=["capped", "test_like"],
                    help="capped: val vs RUL clipped at --eval_cap. test_like: val vs RAW RUL on windows with RUL<=--val_max_rul "
                         "(mimics the official test metric, lets higher RUL caps compete fairly)")
    ap.add_argument("--val_max_rul", type=int, default=195)
    ap.add_argument("--min_corr", type=float, default=0.05)
    ap.add_argument("--log_target", action="store_true",
                    help="train tree models (hgb/xgb/rf/et) on log1p(RUL) instead of raw RUL -- motivated by "
                         "distribution_analysis.py finding engine lifetime fits Lognormal better than Normal. "
                         "TREE MODELS ONLY: not supported for lstm/cnn_lstm/cnn_lstm_attn in this version.")
    ap.add_argument("--strong_reg", action="store_true",
                    help="tighter regularisation preset for hgb/xgb (lower learning rate, higher L2, more "
                         "aggressive subsampling) -- trades training time for less overfitting risk")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--dl_seeds", type=int, default=3, help="seeds averaged per deep model (reduces run-to-run noise)")
    ap.add_argument("--ens_margin", type=float, default=0.5,
                    help="prefer the ensemble unless a single model beats it by more than this val RMSE")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--min_val_cycle", type=int, default=30)
    ap.add_argument("--eval_cap", type=int, default=125,
                    help="common RUL ceiling used to compare configs on validation (keeps different caps comparable)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="one config only (window 30, cap 125)")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    if args.quick:
        args.windows, args.caps = [30], [125]
    models = [m for m in args.models if m in TAB_MODELS + DL_MODELS]
    skipped = set(args.models) - set(models)
    if skipped:
        print(f"[skip] not available in this environment: {sorted(skipped)}")
    if args.log_target:
        dl_in_run = [m for m in models if m in DL_MODELS]
        if dl_in_run:
            print(f"[skip] --log_target only supports tree models in this version; dropping {dl_in_run}")
            models = [m for m in models if m not in DL_MODELS]
    out = Path(args.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    summary = {}

    for fd in args.fds:
        t0 = time.time()
        train_raw, test_raw, rul_true = load_fd(args.data_dir, fd)
        units = np.sort(train_raw["unit"].unique())
        rng = np.random.default_rng(args.seed)
        val_units = rng.choice(units, size=max(1, int(args.val_frac * len(units))), replace=False)
        # deep nets early-stop on their OWN small group of training engines, so the validation engines stay unseen
        train_units = np.setdiff1d(units, val_units)
        es_units = np.random.default_rng(args.seed + 1).choice(train_units, size=max(1, int(0.15 * len(train_units))), replace=False)
        rows, epochs_used, best = [], {}, None
        print(f"\n===== {fd}: {len(units)} train engines, {test_raw.unit.nunique()} test engines =====")

        for cap in args.caps:
            pre = Preprocessor(N_COND[fd], args.min_corr, args.smooth, args.seed, args.baseline_cycles).fit(train_raw, cap)
            tr = add_rul(pre.transform(train_raw))
            te = pre.transform(test_raw)
            for W in args.windows:
                Xtr, mtr = build_windows(tr, pre.cols_, W)
                Xte, mte = build_windows(te, pre.cols_, W, last_only=True)
                Ftr, Fte = tabular_features(Xtr, mtr.cycle, args.feat_version), tabular_features(Xte, mte.cycle, args.feat_version)
                y = np.minimum(mtr["rul"].to_numpy(), cap)
                isv = mtr["unit"].isin(val_units).to_numpy()
                ises = mtr["unit"].isin(es_units).to_numpy()
                vsel = isv & (mtr["cycle"].to_numpy() >= args.min_val_cycle)
                if args.val_mode == "test_like":
                    vsel &= mtr["rul"].to_numpy() <= args.val_max_rul
                    yv, ev_cap = mtr["rul"].to_numpy()[vsel].astype(float), None  # raw labels, like the official test metric
                else:
                    yv, ev_cap = np.minimum(mtr["rul"].to_numpy()[vsel], args.eval_cap), args.eval_cap
                preds_va, preds_te = {}, {}
                for name in models:
                    tm = ~isv if name in TAB_MODELS else (~isv & ~ises)  # trees use all non-val engines
                    kind, m, ep = fit_member(name, Xtr[tm], Ftr[tm], y[tm], cap, args.seed, args,
                                             Xtr[ises], Ftr[ises], y[ises])
                    if ep:
                        epochs_used[(cap, W, name)] = ep
                    preds_va[name] = np.clip(predict_member(kind, m, Xtr[vsel], Ftr[vsel], cap, args.log_target), 0, cap)
                    preds_te[name] = np.clip(predict_member(kind, m, Xte, Fte, cap, args.log_target), 0, cap)
                if len(models) > 1:
                    preds_va["ens"] = np.mean([preds_va[n] for n in models], axis=0)
                    preds_te["ens"] = np.mean([preds_te[n] for n in models], axis=0)
                tabm = [n for n in models if n in TAB_MODELS]
                if len(tabm) > 1 and len(tabm) < len(models):  # trees-only ensemble competes with the full one
                    preds_va["ens_trees"] = np.mean([preds_va[n] for n in tabm], axis=0)
                    preds_te["ens_trees"] = np.mean([preds_te[n] for n in tabm], axis=0)
                for name in preds_va:
                    r = dict(fd=fd, cap=cap, window=W, model=name, n_sensors=len(pre.cols_),
                             val_rmse=rmse(yv, preds_va[name] if ev_cap is None else np.minimum(preds_va[name], ev_cap)),
                             test_rmse=rmse(rul_true, preds_te[name]),
                             test_rmse_capped=rmse(np.minimum(rul_true, CAP_REPORT), np.minimum(preds_te[name], CAP_REPORT)),
                             test_score=nasa_score(rul_true, preds_te[name]))
                    rows.append(r)
                    print(f"  cap={cap:<3} W={W:<2} {name:<4} val_rmse={r['val_rmse']:6.2f}  "
                          f"test_rmse={r['test_rmse']:6.2f} (capped {r['test_rmse_capped']:5.2f})  score={r['test_score']:9.0f}")

        best = min(rows, key=lambda q: q["val_rmse"])
        ens_rows = [q for q in rows if q["model"] in ("ens", "ens_trees")]
        if ens_rows and best["model"] not in ("ens", "ens_trees"):
            be = min(ens_rows, key=lambda q: q["val_rmse"])
            if be["val_rmse"] <= best["val_rmse"] + args.ens_margin:  # difference is within noise -> prefer ensemble
                best = be
        res = pd.DataFrame(rows).sort_values("val_rmse")
        res.to_csv(out / "results" / f"sweep_{fd}.csv", index=False)

        # ---- refit the validation-selected config on ALL training engines ----
        cap, W, name = best["cap"], best["window"], best["model"]
        names = (models if name == "ens" else [n for n in models if n in TAB_MODELS] if name == "ens_trees" else [name])
        pre = Preprocessor(N_COND[fd], args.min_corr, args.smooth, args.seed, args.baseline_cycles).fit(train_raw, cap)
        tr, te = add_rul(pre.transform(train_raw)), pre.transform(test_raw)
        Xtr, mtr = build_windows(tr, pre.cols_, W)
        Xte, mte = build_windows(te, pre.cols_, W, last_only=True)
        Ftr, Fte = tabular_features(Xtr, mtr.cycle, args.feat_version), tabular_features(Xte, mte.cycle, args.feat_version)
        y = np.minimum(mtr["rul"].to_numpy(), cap)
        members = []
        for n in names:
            kind, m, _ = fit_member(n, Xtr, Ftr, y, cap, args.seed, args, epochs=epochs_used.get((cap, W, n)))
            members.append((kind, n, m))
        pred = Predictor(fd, pre, W, cap, members, args.feat_version, args.log_target)
        p = pred.predict_windows(Xte, mte.cycle.values)
        final = dict(fd=fd, best_cap=cap, best_window=W, best_model=name, n_sensors=len(pre.cols_),
                     val_rmse=best["val_rmse"], test_rmse=rmse(rul_true, p), test_score=nasa_score(rul_true, p),
                     test_rmse_capped=rmse(np.minimum(rul_true, CAP_REPORT), np.minimum(p, CAP_REPORT)),
                     test_rmse_last30=(rmse(rul_true[rul_true < 30], p[rul_true < 30]) if (rul_true < 30).any() else float("nan")),
                     seconds=round(time.time() - t0))
        summary[fd] = final
        joblib.dump(pred, out / "models" / f"{fd}_best.joblib")
        print(f"  >> FINAL {fd}: {name} cap={cap} W={W} | test RMSE {final['test_rmse']:.2f} | "
              f"NASA score {final['test_score']:.0f}")

        # ---- plots ----
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].scatter(rul_true, p, s=14, alpha=.7)
        lim = [0, max(rul_true.max(), p.max()) + 5]
        ax[0].plot(lim, lim, "r--", lw=1)
        ax[0].set(xlabel="True RUL", ylabel="Predicted RUL", title=f"{fd} test: predicted vs true")
        o = np.argsort(rul_true)
        ax[1].plot(rul_true[o], label="true")
        ax[1].plot(p[o], label="predicted")
        ax[1].set(xlabel="Engine (sorted by true RUL)", ylabel="RUL (cycles)", title=f"RMSE {final['test_rmse']:.1f}")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out / "results" / f"{fd}_test_predictions.png", dpi=120)
        plt.close(fig)
        pd.DataFrame({"unit": mte["unit"], "true_rul": rul_true, "pred_rul": p}).to_csv(
            out / "results" / f"{fd}_test_predictions.csv", index=False)

    sp = out / "results" / "summary.json"  # merge with earlier runs so all four datasets accumulate
    allsum = json.loads(sp.read_text()) if sp.exists() else {}
    allsum.update(summary)
    allsum = dict(sorted(allsum.items()))
    pd.DataFrame(allsum).T.to_csv(out / "results" / "summary.csv")
    sp.write_text(json.dumps(allsum, indent=2, default=float))
    print("\n", pd.DataFrame(allsum).T[["best_model", "best_cap", "best_window", "val_rmse", "test_rmse", "test_rmse_capped", "test_rmse_last30", "test_score"]])


if __name__ == "__main__":
    main()
