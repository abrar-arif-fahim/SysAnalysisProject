from pathlib import Path
import json
import random
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .errors import ConfigError
from .logger import logger


def load_config(path: Path) -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.exception("Failed to load config %s", path)
        raise ConfigError(f"Failed to load config {path}: {e}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories(root: Path) -> None:
    (root / "results" / "plots").mkdir(parents=True, exist_ok=True)
    (root / "results" / "logs").mkdir(parents=True, exist_ok=True)


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    # projection onto simplex {x >=0, sum x =1}
    v = np.asarray(v, dtype=float)
    if v.sum() == 1 and np.all(v >= 0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if rho.size == 0:
        theta = 0
    else:
        rho = rho[-1]
        theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    # numerically fix
    w = w / (w.sum() + 1e-12)
    return w


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_csv_matrix(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    arr = df.values
    return arr.squeeze()


def annualize_return(value: float, periods_per_year: int = 252) -> float:
    return value * periods_per_year


def annualize_risk(value: float, periods_per_year: int = 252) -> float:
    return value * (periods_per_year ** 0.5)
