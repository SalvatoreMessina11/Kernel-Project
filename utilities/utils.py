from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utilities.config import ANNUALIZATION, ROOT


def safe_log(x: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return np.log(x.where(x > 0))


def project_calendar(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="D")


def value_flag_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "Date" and not c.endswith("_tradable")]


def read_alternating_csv(path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    value_cols = value_flag_columns(df.reset_index())
    values = df[value_cols].copy()
    flags = pd.DataFrame(index=df.index)
    for col in value_cols:
        flag_col = f"{col}_tradable"
        flags[col] = df[flag_col].astype(bool) if flag_col in df.columns else True
    return values, flags


def write_alternating_csv(
    values: pd.DataFrame,
    flags: pd.DataFrame,
    path: Path | str,
    float_format: str = "%.10f",
) -> None:
    out = pd.DataFrame({"Date": values.index})
    for col in values.columns:
        out[col] = values[col].to_numpy()
        out[f"{col}_tradable"] = flags[col].astype(bool).to_numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format=float_format)


def simple_returns_from_log(log_returns: pd.Series) -> pd.Series:
    return np.expm1(log_returns.astype(float))


def portfolio_log_return(asset_log_returns: pd.Series, weights: pd.Series) -> float:
    aligned = asset_log_returns.reindex(weights.index).fillna(0.0).astype(float)
    simple = np.expm1(aligned.to_numpy())
    gross = float(weights.to_numpy() @ simple)
    if not np.isfinite(gross) or gross <= -1.0:
        return np.nan
    return float(np.log1p(gross))


def drift_weights(weights: pd.Series, asset_log_returns: pd.Series) -> pd.Series:
    aligned = asset_log_returns.reindex(weights.index).fillna(0.0).astype(float)
    simple = np.expm1(aligned.to_numpy())
    arr = weights.reindex(aligned.index).fillna(0.0).to_numpy(dtype=float)
    gross_value = float(arr @ (1.0 + simple))
    if not np.isfinite(gross_value) or gross_value <= 0:
        return normalize_positive(weights)
    new_weights = arr * (1.0 + simple) / gross_value
    return pd.Series(new_weights, index=aligned.index)


def normalize_positive(weights: pd.Series) -> pd.Series:
    weights = weights.clip(lower=0).astype(float)
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        return pd.Series(1.0 / len(weights), index=weights.index)
    return weights / total


def regularize_covariance(covariance: pd.DataFrame | np.ndarray, gamma: float) -> pd.DataFrame:
    """Return Sigma + gamma I as a symmetric numeric DataFrame."""
    if isinstance(covariance, pd.DataFrame):
        index = covariance.index
        values = covariance.reindex(index=index, columns=index).to_numpy(dtype=float)
    else:
        values = np.asarray(covariance, dtype=float)
        index = pd.RangeIndex(values.shape[0])
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = 0.5 * (values + values.T)
    values = values + float(gamma) * np.eye(values.shape[0])
    return pd.DataFrame(values, index=index, columns=index)


def equal_weight_feasible(index: Iterable, min_weight: float, max_weight: float) -> pd.Series:
    index = pd.Index(index)
    if len(index) == 0:
        return pd.Series(dtype=float)
    weight = 1.0 / len(index)
    if min_weight - 1e-12 <= weight <= max_weight + 1e-12:
        return pd.Series(weight, index=index, dtype=float)
    lower_budget = len(index) * min_weight
    upper_budget = len(index) * max_weight
    if lower_budget > 1.0 + 1e-12 or upper_budget < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible weight bounds for {len(index)} assets: "
            f"{min_weight:.4f} <= w <= {max_weight:.4f}, sum(w)=1."
        )
    return pd.Series(weight, index=index, dtype=float).clip(min_weight, max_weight)


def project_to_box_simplex(values: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    """Project values onto {w: sum(w)=1, min_weight <= w_i <= max_weight}."""
    v = np.asarray(values, dtype=float)
    n_assets = v.size
    lower_budget = n_assets * min_weight
    upper_budget = n_assets * max_weight
    if lower_budget > 1.0 + 1e-12 or upper_budget < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible weight bounds for {n_assets} assets: "
            f"{min_weight:.4f} <= w <= {max_weight:.4f}, sum(w)=1."
        )
    low = float(np.min(v - max_weight))
    high = float(np.max(v - min_weight))
    for _ in range(100):
        mid = 0.5 * (low + high)
        projected = np.clip(v - mid, min_weight, max_weight)
        if projected.sum() > 1.0:
            low = mid
        else:
            high = mid
    projected = np.clip(v - 0.5 * (low + high), min_weight, max_weight)
    residual = 1.0 - float(projected.sum())
    if abs(residual) > 1e-12:
        if residual > 0:
            room = np.maximum(max_weight - projected, 0.0)
            total_room = float(room.sum())
            if total_room > 0:
                projected += room / total_room * min(residual, total_room)
        else:
            room = np.maximum(projected - min_weight, 0.0)
            total_room = float(room.sum())
            if total_room > 0:
                projected -= room / total_room * min(-residual, total_room)
    return projected


def project_rows_to_box_simplex(values: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    """Project each row onto {w: sum(w)=1, min_weight <= w_i <= max_weight}."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 2:
        raise ValueError("values must be a 2D array")
    if v.shape[0] == 0:
        return v.copy()
    n_assets = v.shape[1]
    lower_budget = n_assets * min_weight
    upper_budget = n_assets * max_weight
    if lower_budget > 1.0 + 1e-12 or upper_budget < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible weight bounds for {n_assets} assets: "
            f"{min_weight:.4f} <= w <= {max_weight:.4f}, sum(w)=1."
        )

    low = np.min(v - max_weight, axis=1)
    high = np.max(v - min_weight, axis=1)
    for _ in range(80):
        mid = 0.5 * (low + high)
        projected = np.clip(v - mid[:, None], min_weight, max_weight)
        too_large = projected.sum(axis=1) > 1.0
        low = np.where(too_large, mid, low)
        high = np.where(too_large, high, mid)

    projected = np.clip(v - (0.5 * (low + high))[:, None], min_weight, max_weight)
    residual = 1.0 - projected.sum(axis=1)

    positive = np.where(residual > 1e-12)[0]
    if positive.size:
        room = np.maximum(max_weight - projected[positive], 0.0)
        total_room = room.sum(axis=1)
        valid = total_room > 0
        if valid.any():
            rows = positive[valid]
            add = np.minimum(residual[rows], total_room[valid])
            projected[rows] += room[valid] / total_room[valid, None] * add[:, None]

    negative = np.where(residual < -1e-12)[0]
    if negative.size:
        room = np.maximum(projected[negative] - min_weight, 0.0)
        total_room = room.sum(axis=1)
        valid = total_room > 0
        if valid.any():
            rows = negative[valid]
            remove = np.minimum(-residual[rows], total_room[valid])
            projected[rows] -= room[valid] / total_room[valid, None] * remove[:, None]

    return projected


def constrained_mean_variance_weights_batch(
    expected_returns: pd.DataFrame,
    covariance: pd.DataFrame | np.ndarray,
    risk_aversion: float,
    min_weight: float,
    max_weight: float,
    tolerance: float = 1e-8,
    max_iterations: int = 80,
) -> tuple[pd.DataFrame, dict]:
    """Solve constrained mean-variance weights for many dates at once."""
    mu = expected_returns.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tickers = pd.Index(mu.columns)
    n_dates, n_assets = mu.shape
    if n_assets == 0:
        empty = pd.DataFrame(index=mu.index, columns=tickers, dtype=float)
        return empty, {
            "success": pd.Series(False, index=mu.index),
            "message": pd.Series("no assets", index=mu.index),
            "fallback": pd.Series(True, index=mu.index),
            "iterations": 0,
        }
    if n_dates == 0:
        empty = pd.DataFrame(index=mu.index, columns=tickers, dtype=float)
        return empty, {
            "success": pd.Series(dtype=bool, index=mu.index),
            "message": pd.Series(dtype=str, index=mu.index),
            "fallback": pd.Series(dtype=bool, index=mu.index),
            "iterations": 0,
        }

    if isinstance(covariance, pd.DataFrame):
        sigma = covariance.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    else:
        sigma = np.asarray(covariance, dtype=float)
        if sigma.shape != (n_assets, n_assets):
            raise ValueError(f"Covariance shape {sigma.shape} does not match {n_assets} assets.")
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = 0.5 * (sigma + sigma.T)
    sigma += 1e-12 * np.eye(n_assets)

    fallback = equal_weight_feasible(tickers, min_weight, max_weight).to_numpy(dtype=float)
    weights = np.tile(fallback, (n_dates, 1))
    mu_arr = mu.to_numpy(dtype=float)
    iteration = 0

    try:
        largest_eigenvalue = float(np.linalg.eigvalsh(sigma).max())
        step = 1.0 / max(risk_aversion * largest_eigenvalue, 1e-12)
        weights = project_rows_to_box_simplex(weights, min_weight, max_weight)
        converged = np.zeros(n_dates, dtype=bool)
        for iteration in range(1, max_iterations + 1):
            gradient = risk_aversion * (weights @ sigma) - mu_arr
            next_weights = project_rows_to_box_simplex(weights - step * gradient, min_weight, max_weight)
            converged = np.max(np.abs(next_weights - weights), axis=1) <= tolerance
            weights = next_weights
            if bool(converged.all()):
                break

        valid = (
            np.abs(weights.sum(axis=1) - 1.0) <= 1e-7
        ) & (
            weights.min(axis=1) >= min_weight - 1e-7
        ) & (
            weights.max(axis=1) <= max_weight + 1e-7
        )
        if not bool(valid.all()):
            weights[~valid] = fallback
        message = "batch_projected_gradient_converged" if bool(converged.all()) else "batch_projected_gradient_max_iter"
        return pd.DataFrame(weights, index=mu.index, columns=tickers), {
            "success": pd.Series(valid, index=mu.index),
            "message": pd.Series(message, index=mu.index),
            "fallback": pd.Series(~valid, index=mu.index),
            "iterations": iteration,
        }
    except Exception as exc:
        weights = np.tile(fallback, (n_dates, 1))
        return pd.DataFrame(weights, index=mu.index, columns=tickers), {
            "success": pd.Series(False, index=mu.index),
            "message": pd.Series(str(exc), index=mu.index),
            "fallback": pd.Series(True, index=mu.index),
            "iterations": iteration,
        }


def constrained_mean_variance_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame | np.ndarray,
    risk_aversion: float,
    min_weight: float,
    max_weight: float,
    initial_weights: pd.Series | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> tuple[pd.Series, dict]:
    """Solve the long-only constrained regularized mean-variance portfolio.

    The optimizer maximizes mu'w - A/2 w'Sigma w subject to sum(w)=1 and
    min_weight <= w_i <= max_weight. It returns equal weights only after
    explicit optimizer failures, with the failure recorded in the metadata.
    """
    mu = expected_returns.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tickers = pd.Index(mu.index)
    n_assets = len(tickers)
    if n_assets == 0:
        return pd.Series(dtype=float), {"success": False, "message": "no assets", "fallback": True}

    if isinstance(covariance, pd.DataFrame):
        sigma = covariance.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    else:
        sigma = np.asarray(covariance, dtype=float)
        if sigma.shape != (n_assets, n_assets):
            raise ValueError(f"Covariance shape {sigma.shape} does not match {n_assets} assets.")
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = 0.5 * (sigma + sigma.T)
    sigma += 1e-12 * np.eye(n_assets)
    mu_arr = mu.to_numpy(dtype=float)

    fallback = equal_weight_feasible(tickers, min_weight, max_weight)
    if initial_weights is not None:
        x0 = initial_weights.reindex(tickers).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x0 = x0.clip(min_weight, max_weight)
        total = float(x0.sum())
        if np.isfinite(total) and total > 0:
            x0 = (x0 / total).clip(min_weight, max_weight)
            residual = 1.0 - float(x0.sum())
            if abs(residual) <= 1e-8:
                start = x0.to_numpy(dtype=float)
            else:
                start = fallback.to_numpy(dtype=float)
        else:
            start = fallback.to_numpy(dtype=float)
    else:
        start = fallback.to_numpy(dtype=float)

    def objective(weights: np.ndarray) -> float:
        return float(0.5 * risk_aversion * weights @ sigma @ weights - mu_arr @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return risk_aversion * sigma @ weights - mu_arr

    try:
        largest_eigenvalue = float(np.linalg.eigvalsh(sigma).max())
        step = 1.0 / max(risk_aversion * largest_eigenvalue, 1e-12)
        weights = project_to_box_simplex(start, min_weight, max_weight)
        converged = False
        for iteration in range(1, max_iterations + 1):
            next_weights = project_to_box_simplex(weights - step * gradient(weights), min_weight, max_weight)
            if float(np.linalg.norm(next_weights - weights, ord=np.inf)) <= tolerance:
                weights = next_weights
                converged = True
                break
            weights = next_weights
        candidate = pd.Series(weights, index=tickers, dtype=float)
        if (
            abs(float(candidate.sum()) - 1.0) <= 1e-8
            and float(candidate.min()) >= min_weight - 1e-8
            and float(candidate.max()) <= max_weight + 1e-8
        ):
            return candidate, {
                "success": True,
                "message": "projected_gradient_converged" if converged else "projected_gradient_max_iter",
                "fallback": False,
                "objective": -objective(candidate.to_numpy(dtype=float)),
                "iterations": iteration,
            }
    except Exception:
        pass

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones_like(w)}]
    bounds = [(min_weight, max_weight)] * n_assets
    attempts = [
        {"ftol": tolerance, "maxiter": 500},
        {"ftol": max(tolerance, 1e-9), "maxiter": 1000},
    ]
    last_result = None
    for options in attempts:
        result = minimize(
            objective,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"disp": False, **options},
        )
        last_result = result
        candidate = pd.Series(result.x, index=tickers, dtype=float)
        if (
            result.success
            and abs(float(candidate.sum()) - 1.0) <= 1e-7
            and float(candidate.min()) >= min_weight - 1e-7
            and float(candidate.max()) <= max_weight + 1e-7
        ):
            candidate = candidate.clip(min_weight, max_weight)
            candidate += (1.0 - float(candidate.sum())) / n_assets
            return candidate, {
                "success": True,
                "message": str(result.message),
                "fallback": False,
                "objective": -objective(candidate.to_numpy(dtype=float)),
                "iterations": int(getattr(result, "nit", 0)),
            }
        start = fallback.to_numpy(dtype=float)

    message = "optimizer failed"
    if last_result is not None:
        message = str(last_result.message)
    return fallback, {
        "success": False,
        "message": message,
        "fallback": True,
        "objective": -objective(fallback.to_numpy(dtype=float)),
        "iterations": int(getattr(last_result, "nit", 0)) if last_result is not None else 0,
    }


def rebalance_equal_weight_open(prev_weights: pd.Series, tradable: pd.Series) -> pd.Series:
    tradable = tradable.reindex(prev_weights.index).fillna(False).astype(bool)
    closed_weight = prev_weights[~tradable].sum()
    out = prev_weights.copy()
    open_assets = tradable[tradable].index
    if len(open_assets) == 0:
        return normalize_positive(out)
    budget = max(0.0, 1.0 - float(closed_weight))
    out.loc[open_assets] = budget / len(open_assets)
    return normalize_positive(out)


def max_drawdown(log_returns: pd.Series) -> float:
    cumulative = np.exp(log_returns.fillna(0).cumsum())
    drawdown = cumulative / cumulative.cummax() - 1.0
    return float(drawdown.min()) if len(drawdown) else np.nan


def performance_stats(log_returns: pd.Series) -> dict:
    r = log_returns.dropna().astype(float)
    if r.empty:
        return {
            "mean_daily": np.nan,
            "vol_daily": np.nan,
            "sharpe_annualized": np.nan,
            "cumulative_return": np.nan,
            "max_drawdown": np.nan,
            "n_days": 0,
        }
    vol = float(r.std(ddof=1))
    mean = float(r.mean())
    sharpe = math.sqrt(ANNUALIZATION) * mean / vol if vol > 0 else np.nan
    return {
        "mean_daily": mean,
        "vol_daily": vol,
        "sharpe_annualized": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "cumulative_return": float(np.expm1(r.sum())),
        "max_drawdown": max_drawdown(r),
        "n_days": int(len(r)),
    }


def load_region_map() -> pd.DataFrame:
    from utilities.config import CANDIDATE_BANKS

    return pd.DataFrame([a.__dict__ for a in CANDIDATE_BANKS])[["ticker", "name", "country", "region", "currency"]]


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)
