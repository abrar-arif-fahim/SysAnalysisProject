from pathlib import Path
import json
import sys

from src.utils import load_config, set_seed, ensure_directories, load_csv_matrix, save_json
from data.synthetic_gen import generate_synthetic_returns
from data.fetch_yfinance import fetch_market_data
from src.pso import PSO
from src.gurobi_optimizer import solve_gurobi_portfolio
from src.benchmark import equal_weight_baseline, random_search_baseline
from src.visualize import plot_convergence, plot_weights_bar, plot_portfolio_comparison
from src.errors import DataError, GurobiError, PSOError
from src.logger import logger


def _format_weights(weights, assets):
    return ", ".join(f"{asset}: {float(weight):.4f}" for asset, weight in zip(assets, weights))


def _print_portfolio_result(title, result, assets):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Sharpe: {float(result['sharpe']):.6f}")
    if "return" in result:
        print(f"Return: {float(result['return']):.6f}")
    if "risk" in result:
        print(f"Risk: {float(result['risk']):.6f}")
    if "commission" in result:
        print(f"Commission: {float(result['commission']):.6f}")
    if "objective" in result:
        print(f"Objective: {float(result['objective']):.6f}")
    print(f"Weights: {_format_weights(result['weights'], assets)}")


def main():
    root = Path(__file__).parent
    cfg = load_config(root / "config.yaml")
    ensure_directories(root)
    set_seed(cfg.get("pso", {}).get("seed", 42))

    data_dir = root / "data"
    returns_path = data_dir / "returns.csv"
    cov_path = data_dir / "covariance.csv"
    if not returns_path.exists() or not cov_path.exists():
        # try fetching real market data via yfinance first, fall back to synthetic
        logger.info("Attempting to fetch market data via yfinance...")
        assets = cfg["assets"]["names"]
        try:
            fetch_market_data(data_dir, assets)
        except DataError:
            logger.warning("Falling back to synthetic data generation...")
            generate_synthetic_returns(data_dir)

    mu = load_csv_matrix(returns_path)
    covariance = load_csv_matrix(cov_path)
    # expecting mu as 1D
    if mu.ndim > 1:
        mu = mu.flatten()

    assets = cfg["assets"]["names"]
    commission_rates = cfg["commission"]["rates"]
    rf = cfg["assets"].get("rf_rate", 0.02)

    # Run PSO
    logger.info("Running PSO...")
    pso_cfg = cfg["pso"]
    pso = PSO(mu=mu, covariance=covariance, commission_rates=commission_rates,
              rf_rate=rf, max_assets=cfg["portfolio"]["max_assets"],
              omega=pso_cfg["omega"], c1=pso_cfg["c1"], c2=pso_cfg["c2"],
              n_particles=pso_cfg["n_particles"], n_iter=pso_cfg["n_iter"], seed=pso_cfg["seed"])
    best_w, best_sharpe, history = pso.run()
    try:
        save_json(root / "results" / "best_pso_portfolio.json", {"weights": best_w.tolist(), "sharpe": float(best_sharpe)})
        plot_convergence(history, root / "results" / "plots" / "pso_convergence.png")
    except Exception:
        logger.exception("Failed saving PSO results or plotting")

    # Run Gurobi
    logger.info("Running Gurobi (if available)...")
    gurobi_cfg = cfg.get("gurobi", {})
    gurobi_res = None
    gurobi_unavailable_reason = "gurobipy is not installed or unavailable."
    try:
        gurobi_res = solve_gurobi_portfolio(mu, covariance, commission_rates, cfg["portfolio"]["max_assets"], cfg["portfolio"]["risk_aversion"], gurobi_cfg)
        if gurobi_res is not None:
            save_json(root / "results" / "best_gurobi_portfolio.json", gurobi_res)
        else:
            logger.warning("Gurobi unavailable; skipping Gurobi result output.")
    except GurobiError as e:
        logger.warning("Gurobi step skipped: %s", e)
        gurobi_unavailable_reason = str(e)
        gurobi_res = None

    # Equal-weight baseline
    logger.info("Computing baselines...")
    ew = equal_weight_baseline(mu, covariance, rf, commission_rates, cfg["portfolio"]["max_assets"]) 
    rs = random_search_baseline(mu, covariance, rf, commission_rates, cfg["random_search"]["n_samples"], cfg["portfolio"]["max_assets"], cfg["pso"]["seed"]) 

    # Comparison CSV
    import pandas as pd
    rows = []
    if gurobi_res is not None:
        rows.append({"method": "Gurobi", "sharpe": float(gurobi_res.get("sharpe", 0.0)), "weights": gurobi_res.get("weights", [])})
    rows.append({"method": "PSO", "sharpe": float(best_sharpe), "weights": best_w.tolist()})
    rows.append({"method": "Equal Weight", "sharpe": ew["sharpe"], "weights": ew["weights"]})
    rows.append({"method": "Random Search", "sharpe": rs["sharpe"], "weights": rs["weights"]})
    df = pd.DataFrame(rows)
    df.to_csv(root / "results" / "comparison_report.csv", index=False)

    # Terminal report
    if gurobi_res is not None:
        _print_portfolio_result("Gurobi optimized result", gurobi_res, assets)
    else:
        print(f"\nGurobi optimized result\n-----------------------\nGurobi result unavailable because {gurobi_unavailable_reason}")
    _print_portfolio_result("PSO optimized result", {"weights": best_w.tolist(), "sharpe": float(best_sharpe)}, assets)
    print("\nFinal comparison report")
    print("-----------------------")
    print(df.to_string(index=False))

    # Plot weights bar
    result_paths = []
    if gurobi_res is not None:
        result_paths.append(root / "results" / "best_gurobi_portfolio.json")
    result_paths.append(root / "results" / "best_pso_portfolio.json")
    plot_weights_bar(result_paths, assets, root / "results" / "plots" / "weights.png")
    plot_portfolio_comparison(root / "results" / "comparison_report.csv", root / "results" / "plots" / "comparison.png")

    logger.info("Done. Results in results/ folder.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Unhandled error running main: %s", e)
        raise
