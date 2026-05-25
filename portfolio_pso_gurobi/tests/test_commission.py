import numpy as np
from src.commission import commission_cost, commission_amount


def test_commission_cost_and_amount():
    w = np.array([0.5, 0.5, 0.0])
    rates = [0.001, 0.002, 0.005]
    c = commission_cost(w, rates)
    assert abs(c - (0.5 * 0.001 + 0.5 * 0.002)) < 1e-8
    amt = commission_amount(w, 10000, rates)
    assert abs(amt - c * 10000) < 1e-6
