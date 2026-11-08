"""Loading, preprocessing and windowing for NASA C-MAPSS (FD001-FD004)."""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans

OP = ["op1", "op2", "op3"]
SENS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle"] + OP + SENS
N_COND = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}  # operating conditions


def load_fd(data_dir, fd):
    """Return (train_df, test_df, true_rul_array) for one sub-dataset."""
    d = Path(data_dir)
    rd = lambda n: pd.read_csv(d / n, sep=r"\s+", header=None, names=COLS)
    train, test = rd(f"train_{fd}.txt"), rd(f"test_{fd}.txt")
    rul = pd.read_csv(d / f"RUL_{fd}.txt", sep=r"\s+", header=None)[0].to_numpy()
    return train, test, rul


class Preprocessor:
    """Per-operating-condition standardisation + automatic sensor selection.

    FD002/FD004 have 6 operating conditions that shift every sensor, so sensors
    are standardised *within* each condition (found by KMeans on op settings).
    Sensors that are constant, or nearly uncorrelated with (capped) RUL on the
    training data, are dropped. Everything is fit on training data only.
    """

    def __init__(self, n_cond=1, min_corr=0.05, smooth=0, seed=0, baseline_cycles=0):
        self.n_cond, self.min_corr, self.smooth, self.seed = n_cond, min_corr, smooth, seed
        self.baseline_cycles = baseline_cycles  # >0: subtract each engine's own mean over its first N cycles

    def _cond(self, df):
        if self.km_ is None:
            return np.zeros(len(df), dtype=int)
        return self.km_.predict(df[OP].to_numpy())

    def _z(self, df):
        c = self._cond(df)
        return (df[SENS].to_numpy() - self.mean_[c]) / self.std_[c]

    def fit(self, train, cap):
        self.km_ = (KMeans(self.n_cond, n_init=10, random_state=self.seed).fit(train[OP].to_numpy())
                    if self.n_cond > 1 else None)
        c = self._cond(train)
        g = train[SENS].groupby(c)
        mean = g.mean().reindex(range(self.n_cond)).to_numpy()
        std = np.nan_to_num(g.std(ddof=0).reindex(range(self.n_cond)).to_numpy())
        self.mean_, self.std_ = mean, np.where(std < 1e-8, 1.0, std)
        Z = self._z(train)
        rul = np.minimum(train.groupby("unit")["cycle"].transform("max") - train["cycle"], cap).to_numpy()
        keep = []
        for j, s in enumerate(SENS):
            if (std[:, j] < 1e-8).all():
                continue  # exactly constant sensor
            r = abs(np.corrcoef(Z[:, j], rul)[0, 1])
            if np.isfinite(r) and r >= self.min_corr:
                keep.append(s)
        self.cols_ = keep
        return self

    def transform(self, df):
        df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
        out = pd.DataFrame(self._z(df), columns=SENS)[self.cols_]
        n_base = getattr(self, "baseline_cycles", 0)  # getattr keeps older saved models loadable
        if n_base > 0:  # healthy-baseline normalisation using only the engine's own first cycles
            first = (df["cycle"].to_numpy() <= n_base)
            units = df["unit"].to_numpy()
            base = out[first].groupby(units[first]).mean()
            out = out - base.reindex(units).to_numpy()
        if self.smooth > 1:  # causal exponential smoothing per engine
            out = out.groupby(df["unit"].values).transform(lambda s: s.ewm(span=self.smooth, adjust=False).mean())
        out.insert(0, "cycle", df["cycle"].values)
        out.insert(0, "unit", df["unit"].values)
        return out


def add_rul(df):
    df = df.copy()
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df


def build_windows(df, cols, window, last_only=False):
    """Sliding windows (N, window, n_features) ending at each cycle.
    Early cycles are left-padded by repeating the first row, so short test
    engines still work. last_only=True returns one window per engine."""
    Xs, metas = [], []
    keep = [c for c in ("unit", "cycle", "rul") if c in df.columns]
    for _, g in df.groupby("unit", sort=True):
        a = g[cols].to_numpy(np.float32)
        n = len(a)
        pad = np.concatenate([np.repeat(a[:1], window - 1, axis=0), a]) if window > 1 else a
        w = np.transpose(sliding_window_view(pad, window, axis=0), (0, 2, 1))
        sel = slice(n - 1, n) if last_only else slice(0, n)
        Xs.append(w[sel])
        metas.append(g.iloc[sel][keep])
    return np.ascontiguousarray(np.concatenate(Xs)), pd.concat(metas, ignore_index=True)


def tabular_features(X, cycles, version=1):
    """Summarise each window: last value, mean, std, min, max, slope, delta + engine age.
    version=2 also adds mean / std / slope over the last 10 cycles (short-term trend)."""
    W = X.shape[1]
    last = X[:, -1, :]
    if W > 1:
        t = np.arange(W, dtype=np.float32) - (W - 1) / 2
        slope = (X * t[None, :, None]).sum(1) / (t ** 2).sum()
    else:
        slope = np.zeros_like(last)
    feats = [last, X.mean(1), X.std(1), X.min(1), X.max(1), slope, last - X[:, 0, :],
             np.asarray(cycles, dtype=np.float32)[:, None]]
    if version >= 2:
        k = min(10, W)
        Xs = X[:, -k:, :]
        if k > 1:
            t = np.arange(k, dtype=np.float32) - (k - 1) / 2
            s10 = (Xs * t[None, :, None]).sum(1) / (t ** 2).sum()
        else:
            s10 = np.zeros_like(last)
        feats += [Xs.mean(1), Xs.std(1), s10]
    return np.hstack(feats)


def tabular_feature_names(sensor_cols, version=1):
    """Column names matching tabular_features(), in the same order, for feature-importance reporting."""
    blocks = ["last", "mean", "std", "min", "max", "slope", "delta_from_window_start"]
    names = [f"{s}_{b}" for b in blocks for s in sensor_cols] + ["engine_age_cycles"]
    if version >= 2:
        names += [f"{s}_{b}" for b in ("mean_last10", "std_last10", "slope_last10") for s in sensor_cols]
    return names
