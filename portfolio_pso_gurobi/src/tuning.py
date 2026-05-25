import csv
from pathlib import Path
from typing import List
import numpy as np

from .pso import PSO


def tune_pso(config: dict, mu: np.ndarray, covariance: np.ndarray, commission_rates: List[float]) -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "tuning_report.csv"
    # small grid
    omegas = [0.6, 0.8]
    c1s = [0.8, 1.5]
    c2s = [0.8, 1.5]

    rows = []
    for o in omegas:
        for c1 in c1s:
            for c2 in c2s:
                pso = PSO(mu=mu, covariance=covariance, commission_rates=commission_rates, rf_rate=config["assets"]["rf_rate"],
                          max_assets=config["portfolio"]["max_assets"], omega=o, c1=c1, c2=c2,
                          n_particles=10, n_iter=20, seed=config["pso"]["seed"])
                _, best_sharpe, _ = pso.run()
                rows.append({"omega": o, "c1": c1, "c2": c2, "sharpe": float(best_sharpe)})

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["omega", "c1", "c2", "sharpe"])
        writer.writeheader()
        writer.writerows(rows)
