"""Predictor: preprocessing + windowing + (ensemble of) trained models in one picklable object."""
import numpy as np

from .data import build_windows, tabular_features


class Predictor:
    def __init__(self, fd, pre, window, cap, members, feat_version=1, log_target=False):
        self.fd, self.pre, self.window, self.cap = fd, pre, window, cap
        self.feat_version = feat_version
        self.log_target = log_target  # tab members were trained on log1p(RUL); expm1 their raw predict() output
        self.members = members  # list of (kind, name, fitted_model); kind in {"tab", "dl"}

    @property
    def names(self):
        return [m[1] for m in self.members]

    def predict_windows(self, X, cycles):
        preds = []
        for kind, _, obj in self.members:
            if kind == "tab":
                p = obj.predict(tabular_features(X, cycles, getattr(self, "feat_version", 1)))
                preds.append(np.expm1(p) if getattr(self, "log_target", False) else p)
            else:
                from .models import dl_predict
                preds.append(dl_predict(obj, X, self.cap))
        return np.clip(np.mean(preds, axis=0), 0, self.cap)

    def predict_raw(self, df_raw, last_only=True):
        """df_raw: raw C-MAPSS dataframe (unit, cycle, op1.., s1..). Returns DataFrame(unit, cycle, pred_rul)."""
        d = self.pre.transform(df_raw)
        X, meta = build_windows(d, self.pre.cols_, self.window, last_only=last_only)
        out = meta[["unit", "cycle"]].copy()
        out["pred_rul"] = self.predict_windows(X, meta["cycle"].values)
        return out
