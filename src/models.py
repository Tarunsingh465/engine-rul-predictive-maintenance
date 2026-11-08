"""Model zoo: tree models (sklearn / xgboost) and optional PyTorch LSTM / CNN-LSTM / CNN-LSTM-attention nets."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # optional
    XGBRegressor = None
try:
    import torch
    import torch.nn as nn
except Exception:  # optional
    torch = None

TAB_MODELS = ["rf", "et", "hgb"] + (["xgb"] if XGBRegressor else [])
# Deep-model variants for an ablation study: plain LSTM -> + CNN front-end -> + attention pooling
ARCH = {
    "lstm": dict(conv=False, attn=False),           # plain 2-layer LSTM
    "cnn_lstm": dict(conv=True, attn=False),        # Conv1d feature extractor + LSTM
    "cnn_lstm_attn": dict(conv=True, attn=True),    # Conv1d + LSTM + attention pooling
}
DL_MODELS = list(ARCH) if torch is not None else []


def make_tab(name, seed=0, n_estimators=300, n_jobs=-1, strong_reg=False):
    """strong_reg=True: a more tightly regularised preset for xgb/hgb (lower learning rate, more trees so there
    is still enough capacity, higher L2, more aggressive subsampling) -- trades training time for a model less
    likely to overfit the training engines. rf/et are unaffected (their regularisation knobs are the min-leaf-size
    and max-features set below, already conservative)."""
    if name == "rf":
        return RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3, max_features=0.5,
                                     n_jobs=n_jobs, random_state=seed)
    if name == "et":
        return ExtraTreesRegressor(n_estimators=n_estimators, min_samples_leaf=2, max_features=0.5,
                                   n_jobs=n_jobs, random_state=seed)
    if name == "hgb":
        if strong_reg:
            return HistGradientBoostingRegressor(max_iter=1200, learning_rate=0.02, max_leaf_nodes=20,
                                                 min_samples_leaf=25, l2_regularization=4.0, early_stopping=True,
                                                 n_iter_no_change=30, validation_fraction=0.15, random_state=seed)
        return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=True,
                                             validation_fraction=0.1, random_state=seed)
    if name == "xgb":
        if strong_reg:
            return XGBRegressor(n_estimators=1500, learning_rate=0.015, max_depth=4, min_child_weight=5,
                                subsample=0.6, colsample_bytree=0.5, reg_lambda=6.0, reg_alpha=0.5,
                                n_jobs=n_jobs, random_state=seed)
        return XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8,
                            colsample_bytree=0.7, reg_lambda=2.0, n_jobs=n_jobs, random_state=seed)
    raise ValueError(name)


if torch is not None:
    class RULNet(nn.Module):
        """Configurable RUL network. conv=True adds a Conv1d front-end; attn=True adds attention pooling."""

        def __init__(self, n_feat, hidden=64, layers=2, dropout=0.2, conv=True, attn=True):
            super().__init__()
            self.use_conv, self.use_attn = conv, attn
            if conv:
                self.conv = nn.Sequential(nn.Conv1d(n_feat, 64, 3, padding=1), nn.ReLU(),
                                          nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
            self.rnn = nn.LSTM(64 if conv else n_feat, hidden, layers, batch_first=True, dropout=dropout)
            if attn:
                self.attn = nn.Linear(hidden, 1)
            self.head = nn.Sequential(nn.Linear(hidden * (2 if attn else 1), 64), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(64, 1))

        def forward(self, x):  # x: (B, W, F)
            if self.use_conv:
                x = self.conv(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.rnn(x)
            if self.use_attn:
                a = torch.softmax(self.attn(h), dim=1)
                z = torch.cat([(a * h).sum(1), h[:, -1]], dim=1)
            else:
                z = h[:, -1]
            return self.head(z).squeeze(-1)


    class AvgNet(nn.Module):
        """Averages the outputs of several independently trained networks (seed ensemble)."""

        def __init__(self, nets):
            super().__init__()
            self.nets = nn.ModuleList(nets)

        def forward(self, x):
            return torch.stack([n(x) for n in self.nets]).mean(0)


    def fit_dl(Xtr, ytr, cap, Xva=None, yva=None, epochs=60, patience=10, seed=0, lr=1e-3, bs=256,
               arch="cnn_lstm_attn"):
        """Train RULNet variant `arch` (see ARCH). With a validation set: early stopping, returns (model, best_epoch_count)."""
        torch.manual_seed(seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = RULNet(Xtr.shape[2], **ARCH[arch]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
        Xt, yt = torch.tensor(Xtr), torch.tensor(ytr / cap, dtype=torch.float32)
        best, best_state, best_ep, bad = float("inf"), None, epochs, 0
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(Xt[idx].to(dev)), yt[idx].to(dev))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if Xva is not None:
                v = float(np.mean((dl_predict(model, Xva, 1.0, dev) - yva / cap) ** 2))
                sched.step(v)
                if v < best - 1e-6:
                    best, best_ep, bad = v, ep + 1, 0
                    best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model.cpu().eval(), best_ep


    def dl_predict(model, X, cap, dev="cpu", bs=1024):
        model.eval().to(dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                out.append(model(torch.tensor(X[i:i + bs]).to(dev)).cpu().numpy())
        return np.concatenate(out) * cap
