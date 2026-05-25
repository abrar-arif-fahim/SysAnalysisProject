from pathlib import Path
from typing import List
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_convergence(history: List[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(history, marker=".")
    plt.xlabel("Iteration")
    plt.ylabel("Best Sharpe")
    plt.title("PSO Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_portfolio_comparison(comparison_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(comparison_csv)
    metrics = df.set_index("method")["sharpe"]
    plt.figure()
    metrics.plot(kind="bar")
    plt.ylabel("Sharpe")
    plt.title("Method Sharpe Comparison")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_weights_bar(result_json_paths: List[Path], asset_names: List[str], output_path: Path) -> None:
    data = []
    labels = []
    for p in result_json_paths:
        if not Path(p).exists():
            continue
        with open(p, "r") as f:
            obj = json.load(f)
        data.append(obj.get("weights", []))
        labels.append(Path(p).stem)

    if not data:
        return
    arr = np.array(data)
    n_methods, n_assets = arr.shape
    x = np.arange(n_assets)
    width = 0.7
    plt.figure(figsize=(6, 4))
    for i in range(n_methods):
        plt.bar(x + (i - n_methods / 2) * (width / n_methods), arr[i], width / n_methods, label=labels[i])
    plt.xticks(x, asset_names)
    plt.ylabel("Weight")
    plt.title("Portfolio Weights")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
