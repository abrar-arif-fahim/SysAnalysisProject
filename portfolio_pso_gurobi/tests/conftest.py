import sys
from pathlib import Path

# Ensure the package root (portfolio_pso_gurobi) is on sys.path so `src` imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
