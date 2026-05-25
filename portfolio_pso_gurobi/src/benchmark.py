from typing import List, Tuple
import numpy as np
from .portfolio import sharpe_ratio


def equal_weight_baseline(mu, covariance, rf_rate, commission_rates, max_assets: int):
    n = len(mu)
    if max_assets >= n:
        weights = np.ones(n) / n
    else:
        # choose top expected returns
        idx = np.argsort(mu)[::-1][:max_assets]
        w = np.zeros(n)
        w[idx] = 1.0 / len(idx)
        weights = w
    s = sharpe_ratio(weights, mu, covariance, rf_rate, commission_rates)
    return {"weights": weights.tolist(), "sharpe": float(s)}


def random_search_baseline(mu, covariance, rf_rate, commission_rates, n_samples: int, max_assets: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(mu)
    best = {"sharpe": -1e9, "weights": None}
    for _ in range(n_samples):
        k = rng.randint(1, max_assets + 1)
        idx = rng.choice(n, size=k, replace=False)
        x = rng.dirichlet(np.ones(k))
        w = np.zeros(n)
        w[idx] = x
        s = sharpe_ratio(w, mu, covariance, rf_rate, commission_rates)
        if s > best["sharpe"]:
            best["sharpe"] = float(s)
            best["weights"] = w.tolist()
    return best
