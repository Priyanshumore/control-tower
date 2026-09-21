"""
Module 3: demand forecasting engine.

Every method is implemented from first principles so the arithmetic is fully
auditable against a textbook worked example. Each method returns a one step
ahead fitted series (aligned to the history) plus a forward forecast.

Selection logic: use manually supplied parameters or search candidates against a
held-out block. Rank methods by the chosen error metric, then refit on full
history. Automatic selection scores are validation scores, not an independent
estimate from a second untouched test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

METHODS = ["Naive", "Moving Average", "Weighted Moving Average",
           "Exponential Smoothing", "Holt Linear Trend", "Holt-Winters"]


@dataclass
class FitResult:
    method: str
    params: Dict[str, float]
    fitted: np.ndarray          # one step ahead fit aligned to history (nan where undefined)
    forecast: np.ndarray        # forward forecast
    metrics: Dict[str, float]   # evaluated on the holdout block
    note: str = ""


# ---------------------------------------------------------------------------
# Individual methods.  Each returns (fitted, forecast_fn)
# ---------------------------------------------------------------------------
def naive(y: np.ndarray, h: int) -> Tuple[np.ndarray, np.ndarray]:
    f = np.full(len(y), np.nan)
    f[1:] = y[:-1]
    return f, np.full(h, y[-1])


def moving_average(y: np.ndarray, h: int, n: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    n = max(1, int(n))
    f = np.full(len(y), np.nan)
    for t in range(n, len(y)):
        f[t] = y[t - n:t].mean()
    fc = float(y[-n:].mean())
    return f, np.full(h, fc)


def weighted_moving_average(y: np.ndarray, h: int,
                            weights: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    w = np.array(weights if weights else [0.5, 0.3, 0.2], dtype=float)
    if len(y) < len(w):
        raise ValueError("Weight window exceeds the available history")
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative and sum to a positive value")
    w = w / w.sum()
    n = len(w)
    f = np.full(len(y), np.nan)
    for t in range(n, len(y)):
        # w[0] is the weight on the most recent period
        f[t] = float(np.dot(y[t - n:t][::-1], w))
    fc = float(np.dot(y[-n:][::-1], w))
    return f, np.full(h, fc)


def ses(y: np.ndarray, h: int, alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    f = np.full(len(y), np.nan)
    level = float(y[0])
    f[0] = np.nan
    for t in range(1, len(y)):
        f[t] = level
        level = alpha * y[t] + (1 - alpha) * level
    return f, np.full(h, level)


def holt(y: np.ndarray, h: int, alpha: float = 0.2,
         beta: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y)
    f = np.full(n, np.nan)
    level = float(y[0])
    trend = float(y[1] - y[0]) if n > 1 else 0.0
    for t in range(1, n):
        f[t] = level + trend
        prev_level = level
        level = alpha * y[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    fc = np.array([level + (i + 1) * trend for i in range(h)])
    return f, fc


def holt_winters(y: np.ndarray, h: int, alpha: float = 0.3, beta: float = 0.1,
                 gamma: float = 0.2, s: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    """Additive Holt-Winters. Needs at least two full seasons of history."""
    n = len(y)
    if n < 2 * s:
        raise ValueError("not enough history for the seasonal cycle")

    # initial level and trend from the first two seasons
    s1 = y[:s].mean()
    s2 = y[s:2 * s].mean()
    level = float(s1)
    trend = float((s2 - s1) / s)
    season = np.array([y[i] - s1 for i in range(s)], dtype=float)

    f = np.full(n, np.nan)
    for t in range(n):
        idx = t % s
        if t >= s:
            f[t] = level + trend + season[idx]
        prev_level = level
        level = alpha * (y[t] - season[idx]) + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        season[idx] = gamma * (y[t] - level) + (1 - gamma) * season[idx]

    fc = np.array([level + (i + 1) * trend + season[(n + i) % s] for i in range(h)])
    return f, fc


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------
def metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(p)
    a, p = a[mask], p[mask]
    if len(a) == 0:
        return {"MAD": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE": np.nan,
                "Bias": np.nan, "Tracking_Signal": np.nan, "n": 0}
    e = a - p
    mad = float(np.mean(np.abs(e)))
    mse = float(np.mean(e ** 2))
    nz = a != 0
    mape = float(np.mean(np.abs(e[nz] / a[nz])) * 100) if nz.any() else np.nan
    bias = float(np.mean(e))
    ts = float(np.sum(e) / mad) if mad > 0 else 0.0
    return {"MAD": mad, "MSE": mse, "RMSE": float(np.sqrt(mse)), "MAPE": mape,
            "Bias": bias, "Tracking_Signal": ts, "n": int(len(a))}


# ---------------------------------------------------------------------------
# Fit every method for one item
# ---------------------------------------------------------------------------
def fit_all(y: np.ndarray, horizon: int, season_length: int = 13,
            holdout: int = 8, ma_n: int = 3,
            wma_weights: Optional[List[float]] = None,
            manual: bool = False, alpha: float = 0.2, beta: float = 0.1,
            gamma: float = 0.2, criterion: str = "MAPE") -> List[FitResult]:
    """Evaluate manual or searched parameters on holdout, refit on full history."""
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all() or len(y) < 4 or (y < 0).any():
        raise ValueError("Forecasting needs at least four finite, nonnegative demand observations")
    if any(not 0 <= v <= 1 for v in (alpha, beta, gamma)):
        raise ValueError("Smoothing constants must be between zero and one")
    n = len(y)
    holdout = int(min(max(2, holdout), max(2, n // 3)))
    train, test = y[:n - holdout], y[n - holdout:]

    grid_a = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    grid_bg = [round(x, 2) for x in np.arange(0.05, 0.61, 0.05)]

    if manual:
        grid_a, grid_bg = [alpha], [beta]

    results: List[FitResult] = []

    def evaluate(name: str, params: Dict[str, float], fn) -> Optional[FitResult]:
        """fn(series, h) -> (fitted, forecast)"""
        try:
            _, pred_test = fn(train, holdout)          # forecast the holdout block
            m = metrics(test, pred_test)
            fitted_full, fc = fn(y, horizon)           # refit on everything
        except Exception as exc:
            return FitResult(name, params, np.full(n, np.nan), np.full(horizon, np.nan),
                             metrics(np.array([]), np.array([])), note=str(exc))
        return FitResult(name, params, fitted_full, fc, m)

    results.append(evaluate("Naive", {}, lambda s, h: naive(s, h)))

    best_ma = None
    for nn in ([ma_n] if manual else [2, 3, 4, 5, 6, 8]):
        if nn >= len(train):
            continue
        r = evaluate("Moving Average", {"n": nn}, lambda s, h, nn=nn: moving_average(s, h, nn))
        if r and (best_ma is None or _better(r, best_ma, criterion)):
            best_ma = r
    if best_ma is None:
        best_ma = evaluate("Moving Average", {"n": ma_n},
                           lambda s, h: moving_average(s, h, ma_n))
    results.append(best_ma)

    w = wma_weights if wma_weights else [0.5, 0.3, 0.2]
    results.append(evaluate("Weighted Moving Average", {"w": ",".join(str(x) for x in w)},
                            lambda s, h, w=w: weighted_moving_average(s, h, w)))

    best = None
    for a in grid_a:
        r = evaluate("Exponential Smoothing", {"alpha": a}, lambda s, h, a=a: ses(s, h, a))
        if r and (best is None or _better(r, best, criterion)):
            best = r
    results.append(best)

    best = None
    for a in grid_a:
        for b in grid_bg:
            r = evaluate("Holt Linear Trend", {"alpha": a, "beta": b},
                         lambda s, h, a=a, b=b: holt(s, h, a, b))
            if r and (best is None or _better(r, best, criterion)):
                best = r
    results.append(best)

    if n >= 2 * season_length + 2 and len(train) >= 2 * season_length:
        best = None
        for a in ([alpha] if manual else [0.1, 0.2, 0.3, 0.4, 0.5]):
            for b in ([beta] if manual else [0.05, 0.1, 0.2]):
                for g in ([gamma] if manual else [0.1, 0.2, 0.3, 0.5]):
                    r = evaluate("Holt-Winters",
                                 {"alpha": a, "beta": b, "gamma": g, "s": season_length},
                                 lambda s, h, a=a, b=b, g=g: holt_winters(s, h, a, b, g, season_length))
                    if r and not np.isnan(r.metrics.get("MAPE", np.nan)) and (best is None or _better(r, best, criterion)):
                        best = r
        if best is not None:
            results.append(best)

    return [r for r in results if r is not None]


def _better(a: FitResult, b: FitResult, criterion: str = "MAPE") -> bool:
    ma, mb = a.metrics.get(criterion, np.nan), b.metrics.get(criterion, np.nan)
    if np.isnan(ma):
        return False
    if np.isnan(mb):
        return True
    return ma < mb


def select_best(results: List[FitResult], criterion: str = "MAPE") -> FitResult:
    valid = [r for r in results if np.isfinite(r.metrics.get(criterion, np.nan))
             and np.isfinite(r.forecast).all()]
    if not valid:
        return results[0]
    return min(valid, key=lambda r: r.metrics[criterion])


# ---------------------------------------------------------------------------
# Item level driver
# ---------------------------------------------------------------------------
def run_forecast(demand_history: pd.DataFrame, horizon: int, plan_start: int,
                 season_length: int = 13, holdout: int = 8,
                 criterion: str = "MAPE",
                 method_override: Optional[Dict[str, str]] = None,
                 mape_threshold: float = 20.0,
                 forecast_options: Optional[dict] = None) -> Dict[str, object]:
    """Forecast every item in the demand history.

    Returns a dict with:
      forecast   : tidy frame item, period, forecast
      accuracy   : one row per item and method with all error metrics
      fit        : actual vs fitted vs forecast for charting
      selected   : chosen method per item
      exceptions : items whose accuracy breaches the threshold
    """
    method_override = method_override or {}
    fc_rows, acc_rows, fit_rows, sel_rows, exc_rows = [], [], [], [], []

    for item, g in demand_history.groupby("item"):
        g = g.sort_values("period")
        y = g["demand"].to_numpy(dtype=float)
        periods = g["period"].to_numpy()
        if len(y) < 4:
            continue

        results = fit_all(y, horizon, season_length=season_length, holdout=holdout,
                          criterion=criterion, **(forecast_options or {}))
        for r in results:
            row = {"item": item, "method": r.method,
                   "params": ", ".join(f"{k}={v}" for k, v in r.params.items()), "note": r.note}
            row.update({k: v for k, v in r.metrics.items()})
            acc_rows.append(row)

        forced = method_override.get(item)
        if forced and forced != "Auto":
            cand = [r for r in results if r.method == forced and np.isfinite(r.forecast).all()]
            best = cand[0] if cand else select_best(results, criterion)
        else:
            best = select_best(results, criterion)

        sel_rows.append({
            "item": item, "method": best.method,
            "params": ", ".join(f"{k}={v}" for k, v in best.params.items()),
            "MAPE": best.metrics.get("MAPE"), "MAD": best.metrics.get("MAD"),
            "RMSE": best.metrics.get("RMSE"), "Bias": best.metrics.get("Bias"),
            "Tracking_Signal": best.metrics.get("Tracking_Signal"),
            "selection": ("Manual override" if forced == best.method else
                          f"{forced} unavailable; lowest {criterion}" if forced and forced != "Auto"
                          else f"Lowest {criterion}"),
        })

        for i, p in enumerate(periods):
            fit_rows.append({"item": item, "period": int(p), "actual": float(y[i]),
                             "fitted": float(best.fitted[i]) if not np.isnan(best.fitted[i]) else None,
                             "type": "History"})
        for i in range(horizon):
            p = plan_start + i
            v = float(max(0.0, best.forecast[i])) if not np.isnan(best.forecast[i]) else 0.0
            fc_rows.append({"item": item, "period": p, "forecast": round(v, 2)})
            fit_rows.append({"item": item, "period": p, "actual": None,
                             "fitted": round(v, 2), "type": "Forecast"})

        mape = best.metrics.get("MAPE", np.nan)
        ts = best.metrics.get("Tracking_Signal", 0.0)
        if not np.isnan(mape) and mape > mape_threshold:
            exc_rows.append({"item": item, "type": "Forecast accuracy",
                             "severity": "High" if mape > 1.5 * mape_threshold else "Medium",
                             "detail": f"{best.method} holdout MAPE {mape:.1f}% is above the "
                                       f"{mape_threshold:.0f}% threshold",
                             "recommended_action": "Review the demand pattern, add causal factors "
                                                   "such as promotions, or shorten the review cycle"})
        if abs(ts) > 4:
            exc_rows.append({"item": item, "type": "Forecast bias",
                             "severity": "Medium",
                             "detail": f"Tracking signal {ts:.1f} indicates a persistent "
                                       f"{'under' if ts > 0 else 'over'} forecast",
                             "recommended_action": "Re-tune smoothing constants or reset the level"})

    return {
        "forecast": pd.DataFrame(fc_rows),
        "accuracy": pd.DataFrame(acc_rows),
        "fit": pd.DataFrame(fit_rows),
        "selected": pd.DataFrame(sel_rows),
        "exceptions": pd.DataFrame(exc_rows),
    }
