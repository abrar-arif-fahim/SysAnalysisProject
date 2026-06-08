# Portfolio PSO Gurobi Project Notes

## Project Overview
- Project folder: `portfolio_pso_gurobi`
- Purpose: compare portfolio optimization using:
  - Particle Swarm Optimization (PSO)
  - Gurobi quadratic optimization benchmark
  - Equal-weight baseline
  - Random-search baseline
- Portfolio assets: `stocks`, `bonds`, `crypto`
- Commission and knapsack-style asset selection are included.

## Data
- Default real-data fetch uses `yfinance` with tickers:
  - `stocks` → `SPY`
  - `bonds` → `AGG`
  - `crypto` → `BTC-USD`
- Default fetch window in code: start `2019-01-01`, end = latest available date.
- If `yfinance` is unavailable, the code falls back to synthetic data.
- Synthetic data is generated with realistic characteristics:
  - Stocks: medium return, medium risk
  - Bonds: lower return, lower risk
  - Crypto: higher return, higher risk

## Configured Parameters
### Assets
- Names: `stocks`, `bonds`, `crypto`
- Expected returns (`mu`): `[0.10, 0.04, 0.18]`
- Risk-free rate: `0.02`

### Commission
- Commission rates: `[0.002, 0.001, 0.005]`
  - stocks: `0.20%`
  - bonds: `0.10%`
  - crypto: `0.50%`

### Portfolio
- Total budget: `10000`
- Maximum selected assets: `2`
- Risk aversion multiplier (`lambda`): `3.0`

### PSO parameters
- Inertia weight: `omega = 0.72`
- Cognitive factor: `c1 = 1.49`
- Social factor: `c2 = 1.49`
- Particles: `n_particles = 30`
- Iterations: `n_iter = 100`
- Seed: `42`

### Gurobi parameters
- Time limit: `60` seconds
- MIP gap: `0.001`

### Random search
- Samples: `5000`

## Key Variables and Concepts
- `w_i`: portfolio weight of asset `i`
- `z_i`: binary selection variable for asset `i`
- Constraints:
  - `sum(w_i) = 1`
  - `w_i >= 0`
  - `w_i <= z_i`
  - `sum(z_i) <= max_assets`
- Commission cost is treated as a linear penalty on weights.

## Optimization Objectives
### PSO objective
- Maximizes adjusted Sharpe ratio:
  - `Sharpe = (portfolio_return - commission_cost - rf_rate) / portfolio_risk`
- Uses the same commission rates in net return.

### Gurobi objective
- Maximizes quadratic risk-adjusted return instead of fractional Sharpe directly:
  - `maximize mu^T w - lambda * (w^T Sigma w) - commission_cost`
- Sharpe ratio is computed after solving for comparison.

## Output Files
- `results/best_pso_portfolio.json`
- `results/best_gurobi_portfolio.json` (only if Gurobi is available)
- `results/comparison_report.csv`
- `results/tuning_report.csv`
- `results/plots/` (convergence, comparison, weights)
- `results/logs/app.log`

## Current Environment Status
- Date of note: `2026-05-26`
- Current run behavior recorded in `results/logs/app.log`:
  - `yfinance` was not installed, so synthetic data was used.
  - `gurobipy` was not installed, so the Gurobi step was skipped.
- The PSO step completed successfully and generated results.

## Useful files for AI questions
- `main.py` — orchestration and workflow
- `config.yaml` — active parameter values
- `src/portfolio.py` — return, risk, Sharpe implementations
- `src/commission.py` — commission logic
- `src/pso.py` — PSO optimizer
- `src/gurobi_optimizer.py` — Gurobi model and solver
- `src/benchmark.py` — baselines
- `src/tuning.py` — PSO tuning
- `src/visualize.py` — plotting
- `data/fetch_yfinance.py` — market data fetcher
- `data/synthetic_gen.py` — fallback synthetic data generator

## Notes for asking AI questions
- Ask about how the knapsack-style `z_i` variables are implemented.
- Ask how commission is included in the adjusted Sharpe ratio.
- Ask how the Gurobi objective differs from PSO's Sharpe objective.
- Ask how to add more assets or change the ticker mapping.
- Ask how to make the project installable or how to fix missing dependencies.
