"""Quantile models for RUL: instead of one number, predict a distribution of plausible remaining-life values.

Why this matters for maintenance: a late prediction (engine fails before the model expects) is much more
costly than an early one (engine serviced sooner than strictly needed). The ratio of these two costs implies
an OPTIMAL quantile to plan against (see cost_to_quantile below) -- this turns "how cautious should the model
be" from a guess into a calculable number.
"""
import numpy as np

from .models import XGBRegressor, make_tab


def cost_to_quantile(cost_late, cost_early=1.0):
    """The quantile that minimises expected cost under the pinball loss, given the cost of a late vs an early
    prediction. E.g. late 4x worse than early -> tau = cost_early / (cost_early + cost_late) = 0.20.
    This is the News-vendor / pinball-loss result: tau = c_early / (c_early + c_late)."""
    return cost_early / (cost_early + cost_late)


def make_quantile_model(name, tau, seed=0, n_estimators=300, n_jobs=-1):
    """A tree model trained to predict the tau-quantile of RUL (e.g. tau=0.1 -> a cautious/low estimate)."""
    if name == "hgb":
        m = make_tab("hgb", seed, n_estimators, n_jobs)
        m.set_params(loss="quantile", quantile=tau)
        return m
    if name == "xgb":
        if XGBRegressor is None:
            raise ValueError("xgboost is not installed")
        m = make_tab("xgb", seed, n_estimators, n_jobs)
        m.set_params(objective="reg:quantileerror", quantile_alpha=tau)
        return m
    raise ValueError(f"{name}: quantile regression only supported for 'hgb' and 'xgb' here")


def pinball_loss(y_true, y_pred, tau):
    """The loss each quantile model is trained to minimise. Lower is better. Used to check calibration quality."""
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def coverage(y_true, y_pred_quantile, tau):
    """What fraction of true values actually fall BELOW this quantile's prediction. A well-calibrated
    tau-quantile model should have coverage close to tau (e.g. the 20% quantile should be exceeded 80% of the time,
    i.e. true < predicted about 20% of the time)."""
    return float(np.mean(np.asarray(y_true) <= np.asarray(y_pred_quantile)))


def interval_stats(y_true, lo, hi):
    """For a [lo, hi] prediction interval: how often it actually contains the truth, and how wide it is."""
    y_true, lo, hi = np.asarray(y_true), np.asarray(lo), np.asarray(hi)
    return {"picp": float(np.mean((y_true >= lo) & (y_true <= hi))),  # Prediction Interval Coverage Probability
            "mean_width": float(np.mean(hi - lo))}


def conformalize_interval(lo_cal, hi_cal, y_cal, lo_test, hi_test, alpha=0.2):
    """Conformalized Quantile Regression (CQR, Romano et al. 2019): widen a [lo, hi] interval using a
    held-out calibration set so it reaches the target coverage (1 - alpha), with a distribution-free guarantee
    (assuming the calibration and test engines are exchangeable). Fixes under-coverage from tree quantile models,
    which tend to produce intervals that are too narrow (overconfident).

    lo_cal, hi_cal, y_cal : the low/high quantile predictions and true RUL on a CALIBRATION set, not used to fit
        the quantile models and not the test set.
    Returns: (lo_test_adjusted, hi_test_adjusted, Q) where Q is the single number added/subtracted from the interval.
    """
    lo_cal, hi_cal, y_cal = np.asarray(lo_cal), np.asarray(hi_cal), np.asarray(y_cal)
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)  # how far outside [lo, hi] each true value fell
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)  # finite-sample correction
    Q = float(np.quantile(scores, q_level))
    return np.asarray(lo_test) - Q, np.asarray(hi_test) + Q, Q
