"""Plot recorded API cost against aggregated accuracy for the main cohort.

Run ``python3 -m analysis.performance`` and ``python3 -m analysis.costs`` first,
then ``python3 -m analysis.figure_generators.plot_cost_vs_accuracy``.
The 48-company standalone Gemini experiment is excluded from this 49-company
comparison. No model API calls are made.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from ..build_results import OUTPUT_DIR


METHODS = ("Single", "Mistral worker", "GPT worker", "Gemini worker", "Worker vote", "LLMC")
MARKERS = {"Single": "o", "Mistral worker": "v", "GPT worker": "P",
           "Gemini worker": "D", "Worker vote": "^", "LLMC": "s"}
COLORS = {"Single": "#7FAED5", "Mistral worker": "#9AC2E1", "GPT worker": "#60A5D4",
          "Gemini worker": "#398BC1", "Worker vote": "#155485", "LLMC": "#203A62"}
# Keep labels beside their observations without connector lines.
LABEL_POSITIONS = {
    "Single": (12, 20, "left"),
    "Mistral worker": (12, -23, "left"),
    "GPT worker": (12, 8, "left"),
    "Gemini worker": (-12, 8, "right"),
    "Worker vote": (0, 24, "center"),
    "LLMC": (0, -27, "center"),
}


def load_points(cost_path: Path, performance_path: Path) -> dict[str, tuple[float, float]]:
    if not cost_path.is_file():
        raise FileNotFoundError(f"Missing {cost_path}; run python3 -m analysis.costs first")
    if not performance_path.is_file():
        raise FileNotFoundError(f"Missing {performance_path}; run python3 -m analysis.performance first")
    with cost_path.open(encoding="utf-8", newline="") as handle:
        costs = [row for row in csv.DictReader(handle)
                 if row["scenario"] == "main_three_runs" and row["method"] in METHODS]
    with performance_path.open(encoding="utf-8", newline="") as handle:
        performance = {row["method"]: row for row in csv.DictReader(handle)
                       if row["evaluation"] == "aggregated" and row["method"] in METHODS}
    if len(costs) != len(METHODS) or {row["method"] for row in costs} != set(METHODS):
        raise ValueError("Expected one main-cohort cost row per plotted method")
    if set(performance) != set(METHODS):
        raise ValueError("Expected one aggregated performance row per plotted method")
    points: dict[str, tuple[float, float]] = {}
    for row in costs:
        method = row["method"]
        if (int(row["companies"]) != 49 or int(row["evaluations_per_company"]) != 3
                or int(row["company_evaluations"]) != 147):
            raise ValueError(f"Wrong cohort or evaluation count for {method}")
        cost = float(row["mean_cost_per_company_evaluation_usd"])
        accuracy = float(row["aggregated_accuracy"])
        performance_row = performance[method]
        if (not math.isfinite(cost) or cost < 0 or not 0 <= accuracy <= 1
                or int(row["aggregated_correct"]) != int(performance_row["correct"])
                or int(performance_row["n"]) != 49
                or not math.isclose(accuracy, float(performance_row["accuracy"]), abs_tol=1e-12)
                or not math.isclose(cost * 147, float(row["total_recorded_cost_usd"]), abs_tol=1e-8)):
            raise ValueError(f"Cost or accuracy does not reconcile for {method}")
        points[method] = (cost, accuracy)
    return points


def plot(points: dict[str, tuple[float, float]]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.6, 4.6), layout="constrained")
    for method in METHODS:
        cost, accuracy = points[method]
        axis.scatter(cost, accuracy, marker=MARKERS[method], color=COLORS[method],
                     s=100, zorder=3)
        x, y, horizontal = LABEL_POSITIONS[method]
        label = f"{method if method != 'Single' else 'Single classifier'}\n${cost:.5f}; {accuracy:.1%}"
        axis.annotate(label, (cost, accuracy), xytext=(x, y), textcoords="offset points",
                      ha=horizontal, va="center", fontsize=10,
                      bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.95})
    maximum_cost = max(cost for cost, _ in points.values())
    accuracies = [accuracy for _, accuracy in points.values()]
    axis.set_xlim(0, max(0.105, maximum_cost * 1.25))
    axis.set_ylim(max(0, min(0.66, min(accuracies) - 0.07)),
                  min(1, max(0.87, max(accuracies) + 0.07)))
    axis.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.10])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:.2f}"))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("Mean API cost per company evaluation (USD)", fontsize=10)
    axis.set_ylabel("Aggregated classification accuracy", fontsize=10)
    axis.set_axisbelow(True)
    axis.grid(color="#E5E8EB", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=10)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costs", type=Path, default=OUTPUT_DIR / "cost_summary.csv")
    parser.add_argument("--performance", type=Path, default=OUTPUT_DIR / "performance_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR.parent.parent / "figures")
    args = parser.parse_args()
    figure = plot(load_points(args.costs, args.performance))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = args.output_dir / f"cost_vs_accuracy.{extension}"
        figure.savefig(path, dpi=300, bbox_inches=None, facecolor="white")
        print(path)
    plt.close(figure)


if __name__ == "__main__":
    main()
