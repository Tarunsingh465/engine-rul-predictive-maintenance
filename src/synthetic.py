"""Generate C-MAPSS-like synthetic files (same format as the NASA .txt files) to smoke-test the pipeline.
NOT real data: never report numbers obtained from it."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import COLS, N_COND

CONDS = np.array([[0, 0, 100], [10, 0.25, 100], [20, 0.7, 100], [25, 0.62, 60], [35, 0.84, 100], [42, 0.84, 100]], float)
CONST = [0, 4, 5, 9, 15, 17, 18]  # s1,s5,s6,s10,s16,s18,s19 are flat in the real FD001


def make_fd(fd, n_train, n_test, seed=0):
    rng = np.random.default_rng(seed)
    nc = N_COND[fd]
    base = rng.uniform(50, 1000, 21)
    direction = rng.choice([-1, 1], 21)
    amp = rng.uniform(0.02, 0.06, 21) * base
    noise = rng.uniform(0.004, 0.012, 21) * base
    scale = rng.uniform(0.5, 1.6, (nc, 21)) if nc > 1 else np.ones((1, 21))
    for j in CONST:
        amp[j] = noise[j] = 0

    def engine(uid, L, cut):
        t = np.arange(1, L + 1)
        h = (t / L) ** rng.uniform(1.6, 2.6)
        c = rng.integers(0, nc, L)
        ops = CONDS[c] if nc > 1 else CONDS[[0] * L] + rng.normal(0, [0.002, 0.0002, 0], (L, 3))
        s = base * scale[c] + direction * amp * h[:, None] * scale[c] + rng.normal(0, 1, (L, 21)) * noise
        df = pd.DataFrame(np.hstack([np.full((L, 1), uid), t[:, None], ops, s]), columns=COLS)
        return df.iloc[:cut]

    lives = lambda n: np.clip(rng.normal(205, 45, n).astype(int), 130, 360)
    train = pd.concat([engine(i + 1, L, L) for i, L in enumerate(lives(n_train))], ignore_index=True)
    tests, ruls = [], []
    for i, L in enumerate(lives(n_test)):
        cut = int(rng.integers(max(31, L - 150), L))  # like real data: test RUL <= ~150
        tests.append(engine(i + 1, L, cut))
        ruls.append(L - cut)
    return train, pd.concat(tests, ignore_index=True), np.array(ruls)


def write_fd(out, fd, n_train, n_test, seed):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tr, te, rul = make_fd(fd, n_train, n_test, seed)
    for name, df in (("train", tr), ("test", te)):
        df = df.astype({"unit": int, "cycle": int})
        df.to_csv(out / f"{name}_{fd}.txt", sep=" ", header=False, index=False, float_format="%.4f")
    pd.Series(rul).to_csv(out / f"RUL_{fd}.txt", header=False, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synthetic")
    ap.add_argument("--fds", nargs="+", default=["FD001", "FD002"])
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=100)
    a = ap.parse_args()
    for k, fd in enumerate(a.fds):
        write_fd(a.out, fd, a.n_train, a.n_test, seed=k)
    print("written to", a.out)
