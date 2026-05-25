from typing import Optional, Dict, Any, List
import numpy as np


from .errors import GurobiError
from .logger import logger


def solve_gurobi_portfolio(mu: np.ndarray, covariance: np.ndarray, commission_rates: List[float], max_assets: int, risk_aversion: float, gurobi_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        logger.warning("gurobipy not installed or unavailable: %s", e)
        return None

    n = len(mu)
    m = gp.Model("portfolio")
    # silent
    m.setParam('OutputFlag', 0)
    if "time_limit" in gurobi_config:
        m.setParam('TimeLimit', gurobi_config["time_limit"])
    if "mip_gap" in gurobi_config:
        m.setParam('MIPGap', gurobi_config["mip_gap"])

    w = m.addVars(range(n), lb=0.0, ub=1.0, name="w")
    z = m.addVars(range(n), vtype=GRB.BINARY, name="z")

    m.addConstr(gp.quicksum(w[i] for i in range(n)) == 1.0)
    for i in range(n):
        m.addConstr(w[i] <= z[i])
    m.addConstr(gp.quicksum(z[i] for i in range(n)) <= max_assets)

    # quadratic variance term: build as sum_{i,j} covariance[i,j] * w_i * w_j
    quad = gp.quicksum(covariance[i][j] * w[i] * w[j] for i in range(n) for j in range(n))

    linear = gp.quicksum(mu[i] * w[i] for i in range(n)) - gp.quicksum(commission_rates[i] * w[i] for i in range(n))
    obj = linear - risk_aversion * quad
    m.setObjective(obj, GRB.MAXIMIZE)

    try:
        m.optimize()
    except gp.GurobiError as e:
        logger.exception("Gurobi optimization failed")
        raise GurobiError(f"Gurobi optimization failed: {e}")

    if m.status != GRB.OPTIMAL and m.status != GRB.TIME_LIMIT and m.status != GRB.SUBOPTIMAL:
        logger.error("Gurobi did not find a usable solution. Status: %s", m.status)
        raise GurobiError(f"Gurobi did not find a usable solution. Status: {m.status}")

    weights = np.array([w[i].X for i in range(n)])
    selected = [i for i in range(n) if weights[i] > 1e-6]
    ret = float(np.dot(weights, mu))
    risk = float(weights @ (np.array(covariance) @ weights))
    commission = float(np.dot(weights, commission_rates))
    sharpe = (ret - commission - 0.0) / (np.sqrt(max(risk, 1e-12)))

    res = {
        "weights": weights.tolist(),
        "selected": selected,
        "objective": m.ObjVal,
        "return": ret,
        "risk": float(np.sqrt(max(risk, 0.0))),
        "commission": commission,
        "sharpe": float(sharpe),
    }
    logger.info("Gurobi solved: obj=%s, sharpe=%s", m.ObjVal, res["sharpe"])
    return res
