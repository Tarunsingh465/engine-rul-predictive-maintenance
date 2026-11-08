"""A predictor wrapping a single quantile-regression model -- the "risk-adjusted" counterpart to the main
accuracy-optimal Predictor (src/predict.py). Deliberately biases predictions toward "early" rather than "late",
by predicting a LOW quantile of RUL instead of the average. Kept entirely separate from Predictor/predict.py
so the main, already-validated models are never touched."""
import numpy as np

from .data import build_windows, tabular_features


class QuantilePredictor:
    """Same predict_raw(df, last_only) interface as Predictor, so the dashboard can treat both the same way."""

    def __init__(self, fd, pre, window, cap, tau, model, feat_version=1, cost_ratio=None):
        self.fd, self.pre, self.window, self.cap = fd, pre, window, cap
        self.tau, self.model, self.feat_version, self.cost_ratio = tau, model, feat_version, cost_ratio

    @property
    def names(self):
        """So the dashboard can show model info the same way it does for the main Predictor."""
        return [f"quantile(tau={self.tau:.2f})"]

    def predict_raw(self, df_raw, last_only=True):
        d = self.pre.transform(df_raw)
        X, meta = build_windows(d, self.pre.cols_, self.window, last_only=last_only)
        F = tabular_features(X, meta["cycle"].values, self.feat_version)
        out = meta[["unit", "cycle"]].copy()
        out["pred_rul"] = np.clip(self.model.predict(F), 0, self.cap)
        return out
