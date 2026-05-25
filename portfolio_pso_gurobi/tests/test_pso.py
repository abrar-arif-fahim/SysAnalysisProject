import numpy as np
from src.pso import PSO


def test_pso_basic_properties():
    mu = np.array([0.1, 0.04, 0.18])
    cov = np.eye(3) * 0.05
    pso = PSO(mu=mu, covariance=cov, commission_rates=[0.001, 0.001, 0.002], rf_rate=0.02, max_assets=2,
              omega=0.7, c1=1.4, c2=1.4, n_particles=10, n_iter=5, seed=1)
    w, s, history = pso.run()
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-8).all()
    assert len(history) == 5
