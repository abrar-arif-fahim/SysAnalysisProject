from typing import Sequence
import numpy as np


def portfolio_return(weights: Sequence[float], mu: Sequence[float]) -> float:
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return float(w @ mu)


def portfolio_variance(weights: Sequence[float], covariance: Sequence[Sequence[float]]) -> float:
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    return float(w @ (cov @ w))


def portfolio_risk(weights: Sequence[float], covariance: Sequence[Sequence[float]]) -> float:
    var = portfolio_variance(weights, covariance)
    return float(np.sqrt(max(var, 0.0)))


def validate_weights(weights: Sequence[float], tol: float = 1e-6) -> bool:
    w = np.asarray(weights, dtype=float)
    if np.any(w < -tol):
        return False
    s = float(w.sum())
    return abs(s - 1.0) <= 1e-5


def sharpe_ratio(weights: Sequence[float], mu: Sequence[float], covariance: Sequence[Sequence[float]], rf_rate: float, commission_rates: Sequence[float] = None) -> float:
    from .commission import commission_cost

    pr = portfolio_return(weights, mu)
    comm = commission_cost(weights, commission_rates) if commission_rates is not None else 0.0
    risk = portfolio_risk(weights, covariance)
    if risk == 0:
        return -1e9
    return (pr - comm - rf_rate) / risk
