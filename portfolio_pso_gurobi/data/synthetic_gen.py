from pathlib import Path
import numpy as np
import pandas as pd

from src.errors import DataError
from src.logger import logger


def generate_synthetic_returns(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # 3 assets: stocks, bonds, crypto
    mu = np.array([0.10, 0.04, 0.18])
    # simple covariance matrix (annualized)
    sigma = np.array([0.18, 0.06, 0.45])
    cov = np.diag(sigma ** 2)
    # add some correlation
    cov[0, 2] = cov[2, 0] = 0.04
    cov[0, 1] = cov[1, 0] = 0.01
    cov[1, 2] = cov[2, 1] = 0.005

    returns_path = output_dir / "returns.csv"
    cov_path = output_dir / "covariance.csv"
    try:
        pd.DataFrame(mu).to_csv(returns_path, index=False, header=False)
        pd.DataFrame(cov).to_csv(cov_path, index=False, header=False)
    except Exception as e:
        logger.exception("Failed to write synthetic data to %s", output_dir)
        raise DataError(f"Failed to write synthetic data: {e}")
    logger.info("Generated synthetic returns and covariance into %s", output_dir)
