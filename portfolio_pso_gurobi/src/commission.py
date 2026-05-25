from typing import Sequence
import numpy as np


def commission_cost(weights: Sequence[float], commission_rates: Sequence[float]) -> float:
    """Return commission as fraction of portfolio value (i.e., summed rate * weight)."""
    if commission_rates is None:
        return 0.0
    w = np.asarray(weights, dtype=float)
    cr = np.asarray(commission_rates, dtype=float)
    return float(np.dot(w, cr))


def commission_amount(weights: Sequence[float], budget: float, commission_rates: Sequence[float]) -> float:
    return commission_cost(weights, commission_rates) * budget


def net_portfolio_return(weights: Sequence[float], mu: Sequence[float], commission_rates: Sequence[float]) -> float:
    from .portfolio import portfolio_return
    pr = portfolio_return(weights, mu)
    comm = commission_cost(weights, commission_rates)
    return pr - comm
