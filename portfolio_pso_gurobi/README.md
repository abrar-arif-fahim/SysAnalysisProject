# portfolio_pso_gurobi

This project compares Particle Swarm Optimization (PSO) and Gurobi for a 3-asset portfolio (stocks, bonds, crypto) while accounting for transaction commissions and a knapsack-style asset selection (maximum number of selected assets).

Overview
- PSO: metaheuristic maximizing an adjusted Sharpe ratio (includes commission cost).
- Gurobi: quadratic optimization maximizing expected return minus risk penalty (lambda * variance) and commission cost; Sharpe is computed after solving for comparison.
- Baselines: equal-weight and random-search.

Knapsack-style constraint: binary selection variables ensure at most `max_assets` are used and `w_i <= z_i` enforces weight only if asset selected.

Commission modeling: proportional commission rates per asset are subtracted from returns when computing net return.

Usage
1. Create venv and install dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the project:

```powershell
python main.py
```

3. Run tests:

```powershell
pytest
```

Files
- `main.py`: orchestrates experiments and saves results into `results/`.
- `config.yaml`: parameters for assets, PSO, Gurobi, etc.
- `data/synthetic_gen.py`: generates `returns.csv` and `covariance.csv` if missing.
- `src/`: implementation modules.
- `tests/`: pytest tests.
