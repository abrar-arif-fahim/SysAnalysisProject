# Portfolio Optimization: PSO vs Gurobi

A comparative study of Particle Swarm Optimization (PSO) and Gurobi for multi-asset portfolio optimization with transaction costs and knapsack-style asset selection constraints.

## Project Overview

This project implements and evaluates two optimization approaches for portfolio allocation across a 3-asset portfolio (stocks, bonds, and cryptocurrencies):

- **Particle Swarm Optimization (PSO)**: A metaheuristic algorithm that maximizes the adjusted Sharpe ratio after accounting for transaction costs
- **Gurobi**: A commercial solver using quadratic programming to maximize risk-adjusted returns
- **Baselines**: Equal-weight and random search for performance context

Both approaches are subject to knapsack-style constraints that limit the number of active assets (maximum 2 out of 3) and enforce weight dependencies via binary selection variables.

## Key Features

- **Asset Selection Constraints**: Binary variables enforce which assets can be included in the portfolio
- **Commission Modeling**: Per-asset transaction costs (stocks: 0.2%, bonds: 0.1%, crypto: 0.5%) are incorporated into return calculations
- **Multiple Data Sources**: Automatically fetches real market data via yfinance; falls back to synthetic data if unavailable
  - Real data: SPY (stocks), AGG (bonds), BTC-USD (crypto)
  - Synthetic data: Realistic parameters for each asset class
- **Comprehensive Performance Comparison**: Includes execution time, Sharpe ratio, and portfolio allocations
- **Configurable Parameters**: All settings defined in `config.yaml`

## Asset Configuration

| Asset | Expected Return | Volatility | Commission | Real Ticker |
|-------|-----------------|------------|-----------|------------|
| Stocks | 10% | Medium | 0.2% | SPY |
| Bonds | 4% | Low | 0.1% | AGG |
| Crypto | 18% | High | 0.5% | BTC-USD |

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Optional Dependencies

- **Gurobi**: For commercial solver optimization, install `gurobipy` and obtain a license (academic licenses available)
- **yfinance**: For real market data fetching (fallback to synthetic data if unavailable)

## Usage

### Running Experiments

```bash
python main.py
```

The script will:
1. Load configuration from `config.yaml`
2. Fetch real market data (via yfinance) or generate synthetic data
3. Run PSO optimization
4. Run Gurobi optimization (if available)
5. Compute equal-weight and random search baselines
6. Generate comparison report and visualizations

Results are saved to the `results/` directory.

### Running Tests

```bash
pytest
```

## Project Structure

```
portfolio_pso_gurobi/
├── main.py                    # Experiment orchestration and workflow
├── config.yaml                # Configuration parameters (assets, PSO, Gurobi settings)
├── requirements.txt           # Python dependencies
├── NOTES.md                   # Detailed project notes and documentation
├── data/
│   ├── synthetic_gen.py       # Generates synthetic market data
│   └── fetch_yfinance.py      # Fetches real market data from yfinance
├── src/
│   ├── pso.py                 # PSO optimizer implementation
│   ├── gurobi_optimizer.py    # Gurobi model and solver
│   ├── portfolio.py           # Portfolio metrics (return, risk, Sharpe)
│   ├── commission.py          # Commission cost calculations
│   ├── benchmark.py           # Baseline methods (equal-weight, random search)
│   ├── visualize.py           # Plotting and visualization utilities
│   ├── utils.py               # Configuration loading and file utilities
│   ├── logger.py              # Logging setup
│   └── errors.py              # Custom exception classes
├── tests/                     # Unit tests
├── results/                   # Output directory (auto-created)
│   ├── best_pso_portfolio.json
│   ├── best_gurobi_portfolio.json
│   ├── comparison_report.csv
│   ├── plots/
│   │   ├── pso_convergence.png
│   │   ├── weights.png
│   │   └── comparison.png
│   └── logs/
│       └── app.log
└── __pycache__/

```

## Configuration

Edit `config.yaml` to adjust optimization behavior:

### Assets
```yaml
assets:
  names: ["stocks", "bonds", "crypto"]
  mu: [0.10, 0.04, 0.18]           # Expected returns
  rf_rate: 0.02                     # Risk-free rate
```

### Commission Rates
```yaml
commission:
  rates: [0.002, 0.001, 0.005]      # Per-asset transaction costs
```

### Portfolio Constraints
```yaml
portfolio:
  budget: 10000                     # Total investment budget
  max_assets: 2                     # Maximum number of active assets
  risk_aversion: 3.0                # Lambda parameter for Gurobi
```

### PSO Hyperparameters
```yaml
pso:
  omega: 0.72                       # Inertia weight
  c1: 1.49                          # Cognitive coefficient
  c2: 1.49                          # Social coefficient
  n_particles: 30                   # Swarm size
  n_iter: 100                       # Number of iterations
  seed: 42                          # Random seed
```

### Gurobi Settings
```yaml
gurobi:
  time_limit: 60                    # Time limit in seconds
  mip_gap: 0.001                    # MIP optimality gap
```

### Random Search
```yaml
random_search:
  n_samples: 5000                   # Number of random portfolios
```

## Mathematical Formulation

### Decision Variables
- `w_i`: Portfolio weight of asset `i`
- `z_i`: Binary selection variable (1 if asset `i` is included, 0 otherwise)

### Constraints
- `sum(w_i) = 1` — Weights sum to 1 (fully invested)
- `w_i >= 0` — Non-negative weights
- `w_i <= z_i` — Weight depends on selection
- `sum(z_i) <= max_assets` — Maximum active assets constraint

### PSO Objective
Maximize adjusted Sharpe ratio:
```
Sharpe = (portfolio_return - commission_cost - rf_rate) / portfolio_risk
```

### Gurobi Objective
Maximize quadratic risk-adjusted return:
```
maximize: μᵀw - λ(wᵀΣw) - commission_cost
```

## Data

### Fetching Real Data

Synthetic market data is generated automatically on first run if yfinance is unavailable. To manually generate synthetic data:

```bash
python data/synthetic_gen.py
```

Generated files:
- `data/returns.csv` — Expected returns vector
- `data/covariance.csv` — Covariance matrix

### Data Characteristics

**Synthetic Data** (if yfinance unavailable):
- **Stocks**: 10% return, medium volatility
- **Bonds**: 4% return, low volatility
- **Crypto**: 18% return, high volatility

**Real Data** (via yfinance):
- Historical prices from 2019-01-01 to present
- Daily returns computed and annualized
- Empirical covariance matrix

## Results

Optimization results are exported to `results/` including:

- `best_pso_portfolio.json` — Optimal PSO allocation and Sharpe ratio
- `best_gurobi_portfolio.json` — Optimal Gurobi allocation and metrics (if available)
- `comparison_report.csv` — Side-by-side comparison of all methods
- `plots/pso_convergence.png` — PSO convergence history
- `plots/weights.png` — Portfolio weight comparison
- `plots/comparison.png` — Performance comparison chart
- `logs/app.log` — Detailed execution log

## Performance Comparison Output

Example comparison report:

| Method | Sharpe | Weights |
|--------|--------|---------|
| Gurobi | 0.65 | [0.3, 0.0, 0.7] |
| PSO | 0.64 | [0.25, 0.05, 0.7] |
| Equal Weight | 0.42 | [0.333, 0.333, 0.333] |
| Random Search | 0.35 | [0.2, 0.1, 0.7] |

## References

- Kennedy, J., & Eberhart, R. (1995). "Particle Swarm Optimization." IEEE Transactions on Neural Networks.
- Markowitz, H. (1952). "Portfolio Selection." Journal of Finance.
- Gurobi Optimization. https://www.gurobi.com
- yfinance Documentation. https://github.com/ranaroussi/yfinance

## Troubleshooting

### Gurobi Not Available
If Gurobi is not installed, the optimization will skip the Gurobi step and proceed with PSO and baselines.

### yfinance Data Fetch Fails
If market data cannot be fetched, the system automatically falls back to synthetic data generation.

### Missing Dependencies
Ensure all requirements are installed:
```bash
pip install -r requirements.txt
```

## Author

Abrar Arif Fahim

## License

This project is part of the SysAnalysisProject repository.
