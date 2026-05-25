from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from src.errors import DataError
from src.logger import logger


def fetch_market_data(output_dir: Path, assets: List[str], tickers_map: Optional[Dict[str, str]] = None,
                      start: str = "2019-01-01", end: Optional[str] = None) -> bool:
    """Fetch adjusted close prices using yfinance and write returns and covariance CSVs.

    Returns True on success, False on failure (e.g., yfinance not installed or network error).
    """
    try:
        import yfinance as yf
    except Exception as e:
        logger.warning("yfinance not installed or could not be imported: %s", e)
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    if tickers_map is None:
        # reasonable defaults
        tickers_map = {
            "stocks": "SPY",
            "bonds": "AGG",
            "crypto": "BTC-USD",
        }

    tickers = [tickers_map.get(a, a) for a in assets]
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    except Exception as e:
        logger.warning("Error downloading data for tickers %s: %s", tickers, e)
        return False

    # If single ticker, make DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # ensure columns in same order as assets
    cols = []
    for a in assets:
        tk = tickers_map.get(a, a)
        if tk in data.columns:
            cols.append(tk)
        else:
            logger.warning("Ticker %s not found in downloaded data", tk)
            return False

    data = data[cols]
    # compute daily returns
    rets = data.pct_change().dropna()
    mu = rets.mean().values * 252
    cov = rets.cov().values * 252

    # save
    try:
        pd.DataFrame(mu).to_csv(output_dir / "returns.csv", index=False, header=False)
        pd.DataFrame(cov).to_csv(output_dir / "covariance.csv", index=False, header=False)
    except Exception as e:
        logger.exception("Failed to save fetched data to %s", output_dir)
        raise DataError(f"Failed to save data: {e}")
    logger.info("Fetched market data and saved to %s", output_dir)
    return True
