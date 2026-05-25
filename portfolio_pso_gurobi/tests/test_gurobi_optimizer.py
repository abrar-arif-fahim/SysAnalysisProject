import numpy as np
import pytest

try:
    import gurobipy  # type: ignore
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False

from src.gurobi_optimizer import solve_gurobi_portfolio


@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
def test_gurobi_solution_properties():
    mu = np.array([0.1, 0.04, 0.18])
    cov = np.eye(3) * 0.05
    res = solve_gurobi_portfolio(mu, cov, [0.001, 0.001, 0.002], max_assets=2, risk_aversion=3.0, gurobi_config={})
    assert res is not None
    w = np.array(res["weights"])
    assert abs(w.sum() - 1.0) < 1e-6
    assert len(res["selected"]) <= 2
