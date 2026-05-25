import numpy as np
from src.tuning import tune_pso
from src.utils import load_config
from pathlib import Path


def test_tuning_writes_csv(tmp_path):
    # small smoke test using config
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")
    mu = np.array(cfg["assets"]["mu"])
    cov = np.eye(len(mu)) * 0.05
    tune_pso(cfg, mu, cov, cfg["commission"]["rates"])
    out = root / "results" / "tuning_report.csv"
    assert out.exists()
