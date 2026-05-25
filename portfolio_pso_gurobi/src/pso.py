from typing import Optional, Tuple, List
import numpy as np
from .utils import project_to_simplex
from .portfolio import sharpe_ratio
from .errors import PSOError
from .logger import logger


class Particle:
    def __init__(self, dim: int, seed: Optional[int] = None):
        rng = np.random.RandomState(seed)
        # initialize on simplex using Dirichlet
        a = rng.dirichlet(np.ones(dim))
        self.position = a
        self.velocity = rng.randn(dim) * 0.01
        self.best_position = self.position.copy()
        self.best_score = -1e9


class PSO:
    def __init__(self, mu: np.ndarray, covariance: np.ndarray, commission_rates: list, rf_rate: float,
                 max_assets: int, omega: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 n_particles: int = 30, n_iter: int = 100, seed: int = 42):
        self.mu = np.asarray(mu, dtype=float)
        self.cov = np.asarray(covariance, dtype=float)
        self.cr = commission_rates
        self.rf = rf_rate
        self.max_assets = max_assets
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.seed = seed
        self.dim = len(self.mu)
        self.rng = np.random.RandomState(seed)

    def _enforce_cardinality(self, weights: np.ndarray) -> np.ndarray:
        if self.max_assets >= len(weights):
            return weights
        # keep top-k entries
        idx = np.argsort(weights)[::-1]
        mask = np.zeros_like(weights, dtype=bool)
        mask[idx[: self.max_assets]] = True
        w = weights * mask
        if w.sum() == 0:
            # fallback uniform on chosen
            w[mask] = 1.0 / mask.sum()
            return w
        return w / w.sum()

    def _fitness(self, weights: np.ndarray) -> float:
        # enforce cardinality before evaluating
        w = self._enforce_cardinality(weights)
        return sharpe_ratio(w, self.mu, self.cov, self.rf, self.cr)

    def run(self) -> Tuple[np.ndarray, float, List[float]]:
        try:
            swarm = [Particle(self.dim, seed=int(self.seed + i)) for i in range(self.n_particles)]
        except Exception as e:
            logger.exception("Failed to initialize PSO particles")
            raise PSOError(f"Failed to initialize particles: {e}")
        # evaluate initial
        gbest_pos = None
        gbest_score = -1e9
        for p in swarm:
            p.position = project_to_simplex(p.position)
            score = self._fitness(p.position)
            p.best_score = score
            p.best_position = p.position.copy()
            if score > gbest_score:
                gbest_score = score
                gbest_pos = p.position.copy()

        history = []
        for it in range(self.n_iter):
            for p in swarm:
                r1 = self.rng.rand(self.dim)
                r2 = self.rng.rand(self.dim)
                p.velocity = (self.omega * p.velocity
                              + self.c1 * r1 * (p.best_position - p.position)
                              + self.c2 * r2 * (gbest_pos - p.position))
                p.position = p.position + p.velocity
                # project to simplex
                p.position = project_to_simplex(p.position)
                score = self._fitness(p.position)
                if score > p.best_score:
                    p.best_score = score
                    p.best_position = p.position.copy()
                if score > gbest_score:
                    gbest_score = score
                    gbest_pos = p.position.copy()
            history.append(gbest_score)
        logger.info("PSO finished: best_sharpe=%s", gbest_score)
        return gbest_pos, gbest_score, history
