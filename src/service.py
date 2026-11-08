"""Prediction service logic used by api.py. Plain Python (no web framework), so it is easy to test."""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .data import COLS, load_fd

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
REQUIRED = [c for c in COLS if c != "unit"]  # cycle, op1-op3, s1-s21 (one row per operating cycle)


class ServiceError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code, self.message = status_code, message


def trained_datasets():
    return [fd for fd in DATASETS if (MODELS_DIR / f"{fd}_best.joblib").exists()]


@lru_cache(maxsize=8)
def get_model(fd):
    """Load a trained Predictor once and keep it in memory."""
    if fd not in DATASETS:
        raise ServiceError(404, f"Unknown dataset '{fd}'. Choose one of {DATASETS}.")
    path = MODELS_DIR / f"{fd}_best.joblib"
    if not path.exists():
        raise ServiceError(404, f"No trained model for {fd}. Train it first with run_experiments.py.")
    return joblib.load(path)


def model_info(fd):
    p = get_model(fd)
    return {"dataset": fd, "members": p.names, "window": p.window, "rul_cap": p.cap,
            "sensors_used": list(p.pre.cols_)}


def classify(rul, critical, warning):
    return "CRITICAL" if rul < critical else ("WARNING" if rul < warning else "OK")


def predict_engine(fd, readings, critical=20, warning=50, include_history=False):
    """readings: list of dicts, one per operating cycle of ONE engine (keys = REQUIRED columns)."""
    pred = get_model(fd)
    if critical >= warning:
        raise ServiceError(422, "critical_threshold must be smaller than warning_threshold.")
    if not readings:
        raise ServiceError(422, "readings is empty: send at least one operating cycle.")
    df = pd.DataFrame(readings)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ServiceError(422, f"Missing columns in readings: {missing}")
    df = df[REQUIRED]
    if df.isna().any().any():
        raise ServiceError(422, "readings contain empty values.")
    if df["cycle"].duplicated().any():
        raise ServiceError(422, "Duplicate 'cycle' values: send one row per operating cycle of a single engine.")
    df = df.astype({c: float for c in REQUIRED if c != "cycle"}).sort_values("cycle")
    df.insert(0, "unit", 1)

    out = pred.predict_raw(df, last_only=not include_history)
    last_cycle = int(df["cycle"].iloc[-1])
    rul = float(out["pred_rul"].iloc[-1])
    warnings = []
    if len(df) < pred.window:
        warnings.append(f"Only {len(df)} cycles received; the model was built for a {pred.window}-cycle window, "
                        "so the estimate is less reliable. Send more history if you have it.")
    if int(df["cycle"].iloc[0]) != 1:
        warnings.append(f"Readings start at cycle {int(df['cycle'].iloc[0])}. The model uses 'cycle' as the engine's "
                        "age, so make sure it is the true operating-cycle count since the engine started.")
    at_cap = rul >= pred.cap - 0.5
    if at_cap:
        warnings.append(f"Prediction is at the model's cap ({pred.cap}): read it as 'at least about {pred.cap} cycles "
                        "left'. The model does not estimate remaining life beyond the cap.")
    result = {"dataset": fd, "model_members": pred.names, "rul_cap": pred.cap, "cycles_received": len(df),
              "last_cycle": last_cycle, "predicted_rul": round(rul, 1), "at_cap": at_cap,
              "status": classify(rul, critical, warning), "warnings": warnings}
    if include_history:
        result["history"] = [{"cycle": int(c), "predicted_rul": round(float(p), 1)}
                             for c, p in zip(out["cycle"], out["pred_rul"])]
    return result


@lru_cache(maxsize=4)
def _test_data(fd):
    try:
        return load_fd(DATA_DIR, fd)
    except FileNotFoundError:
        raise ServiceError(404, f"NASA files for {fd} not found in '{DATA_DIR}'.")


def demo_engine(fd, unit, critical=20, warning=50, include_history=False):
    """Predict for one engine of the official NASA test set (handy for demos); also returns the true RUL."""
    get_model(fd)  # fail early with a clear message if not trained
    _, test, rul_true = _test_data(fd)
    eng = test[test["unit"] == unit]
    if eng.empty:
        raise ServiceError(404, f"Engine {unit} not found in the {fd} test set (units 1 to {int(test['unit'].max())}).")
    res = predict_engine(fd, eng[REQUIRED].to_dict("records"), critical, warning, include_history)
    res["unit"] = int(unit)
    res["true_rul_official"] = int(rul_true[int(unit) - 1])
    return res
