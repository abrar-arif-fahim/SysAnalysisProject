import numpy as np
from src.portfolio import portfolio_return, portfolio_risk, sharpe_ratio, validate_weights


def test_validate_weights_sum_and_nonneg():
    w = np.array([0.5, 0.5, 0.0])
    assert validate_weights(w)


def test_portfolio_return_and_risk_nonnegative():
    mu = np.array([0.1, 0.04, 0.18])
    cov = np.eye(3) * 0.1
    w = np.array([0.6, 0.4, 0.0])
    r = portfolio_return(w, mu)
    risk = portfolio_risk(w, cov)
    assert r >= 0
    assert risk >= 0


def test_sharpe_includes_commission():
    mu = np.array([0.1, 0.04, 0.18])
    cov = np.eye(3) * 0.01
    w = np.array([0.5, 0.5, 0.0])
    s1 = sharpe_ratio(w, mu, cov, 0.01, commission_rates=[0.001, 0.001, 0.001])
    s2 = sharpe_ratio(w, mu, cov, 0.01, commission_rates=[0.0, 0.0, 0.0])
    assert s1 <= s2
