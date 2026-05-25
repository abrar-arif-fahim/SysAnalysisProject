# Portfolio Optimization: PSO vs Gurobi

A comparative study of Particle Swarm Optimization and Gurobi for multi-asset portfolio optimization with transaction costs and asset selection constraints.

## Project Overview

This project implements and evaluates two optimization approaches for portfolio allocation:

- **Particle Swarm Optimization (PSO)**: A metaheuristic algorithm that maximizes the Sharpe ratio after accounting for transaction costs
- **Gurobi**: A commercial solver using quadratic programming to maximize risk-adjusted returns

Both approaches are tested on a 3-asset portfolio (stocks, bonds, cryptocurrencies) subject to knapsack-style constraints that limit the number of active assets and enforce weight restrictions.

## Key Features

- **Constraint Modeling**: Binary selection variables enforce asset selection limits and weight dependencies
- **Transaction Costs**: Per-asset commission rates are incorporated into return calculations
- **Performance Comparison**: Includes baselines (equal-weight, random search) for context
- **Configurable Parameters**: All settings defined in `config.yaml`

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

## Usage

### Running Experiments

```bash
python main.py
```

Results are saved to the `results/` directory.

### Running Tests

```bash
pytest
```

## Project Structure

```
portfolio_pso_gurobi/
├── main.py              # Experiment orchestration
├── config.yaml          # Configuration parameters
├── requirements.txt
├── data/
│   └── synthetic_gen.py # Generates synthetic market data
├── src/                 # Core implementation modules
└── tests/               # Unit tests
```

## Configuration

Edit `config.yaml` to adjust:

- Asset parameters (expected returns, volatility, commissions)
- PSO hyperparameters (swarm size, iterations, inertia)
- Gurobi solver settings
- Portfolio constraints (max assets, weight bounds)

## Data

Synthetic market data (`returns.csv`, `covariance.csv`) is generated automatically on first run. To regenerate:

```bash
python data/synthetic_gen.py
```

## Results

Optimization results are exported to `results/` including:

- Optimal allocations
- Risk and return metrics
- Sharpe ratios
- Solver performance metrics

## References

- PSO Algorithm: [Kennedy & Eberhart, 1995]
- Portfolio Theory: [Markowitz, 1952]
