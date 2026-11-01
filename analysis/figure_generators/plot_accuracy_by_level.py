"""Plot aggregated Single and LLMC accuracy by reviewed involvement level.

Run ``python3 -m analysis.performance`` first, then
``python3 -m analysis.figure_generators.plot_accuracy_by_level``.
No model API calls are made.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from ..build_results import LABELS, OUTPUT_DIR


METHODS = ("Single", "LLMC")
COLORS = {"Single": "#7FAED5", "LLMC": "#203A62"}


def load_counts(path: Path) -> dict[str, list[tuple[int, int]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python3 -m analysis.performance first")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["evaluation"] == "aggregated" and row["method"] in METHODS]
    if len(rows) != 2 or {row["method"] for row in rows} != set(METHODS):
        raise ValueError(f"Expected one aggregated row for each architecture in {path}")
    result: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        method = row["method"]
        counts = [(int(row[f"{label.lower()}_correct"]), int(row[f"{label.lower()}_reference_count"]))
                  for label in LABELS]
        if any(total <= 0 or not 0 <= correct <= total for correct, total in counts):
            raise ValueError(f"Invalid category counts for {method}")
        if sum(total for _, total in counts) != 49 or sum(correct for correct, _ in counts) != int(row["correct"]):
            raise ValueError(f"Category counts do not reconcile for {method}")
        result[method] = counts
    if [total for _, total in result["Single"]] != [total for _, total in result["LLMC"]]:
        raise ValueError("Reference-label totals differ between Single and LLMC")
    return result


def plot(counts: dict[str, list[tuple[int, int]]]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.6, 4.6), layout="constrained")
    centers = list(range(len(LABELS)))
    width = 0.34
    for method, offset in (("Single", -width / 2), ("LLMC", width / 2)):
        values = [correct / total for correct, total in counts[method]]
        bars = axis.bar([x + offset for x in centers], values, width=width,
                        color=COLORS[method], label="Single classifier" if method == "Single" else "LLMC")
        for index, (bar, (correct, total), value) in enumerate(zip(bars, counts[method], values, strict=True)):
            # Equal-height pairs share a centered label, including the High pair.
            shared = counts["Single"][index] == counts["LLMC"][index]
            if shared and method == "Single":
                continue
            label = f"{value:.1%}\n({correct}/{total})" if index else f"{value:.0%}\n({correct}/{total})"
            label_x = centers[index] if shared else bar.get_x() + bar.get_width() / 2
            axis.text(label_x, value + 0.025, label,
                      ha="center", va="bottom", fontsize=10)
    axis.set_xticks(centers, LABELS)
    axis.set_xlim(-0.55, 3.55)
    axis.set_ylim(0, 1.16)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_ylabel("Accuracy", fontsize=10)
    axis.set_xlabel("Reviewed involvement level", fontsize=10)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=10)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E5E8EB", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=10)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=OUTPUT_DIR / "performance_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR.parent.parent / "figures")
    args = parser.parse_args()
    figure = plot(load_counts(args.source))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = args.output_dir / f"accuracy_by_involvement_level.{extension}"
        figure.savefig(path, dpi=300, bbox_inches=None, facecolor="white")
        print(path)
    plt.close(figure)


if __name__ == "__main__":
    main()
