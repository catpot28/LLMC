"""Plot the aggregated Single and LLMC confusion matrices from saved counts.

Run ``python3 -m analysis.performance`` first, then
``python3 -m analysis.figure_generators.plot_confusion_matrices``.
No model calls are made.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ..build_results import LABELS, OUTPUT_DIR


METHODS = ("Single", "LLMC")
PALETTE = ("#F7FAFC", "#DCEAF5", "#79ACD6", "#2F75B5", "#17365D")


def load_matrices(path: Path) -> dict[str, list[list[int]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python3 -m analysis.performance first")
    matrices = {method: [[0] * len(LABELS) for _ in LABELS] for method in METHODS}
    seen: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in METHODS:
                continue
            actual, predicted = row["reviewed_label"], row["predicted_label"]
            if actual not in LABELS or predicted not in LABELS:
                raise ValueError(f"Invalid label in {path}: {actual}, {predicted}")
            key = (method, actual, predicted)
            if key in seen:
                raise ValueError(f"Duplicate confusion cell {key} in {path}")
            seen.add(key)
            count = int(row["count"])
            if count < 0:
                raise ValueError(f"Negative confusion count in {path}")
            matrices[method][LABELS.index(actual)][LABELS.index(predicted)] = count
    for method in METHODS:
        if sum(key[0] == method for key in seen) != len(LABELS) ** 2:
            raise ValueError(f"Incomplete {method} confusion matrix in {path}")
        if sum(map(sum, matrices[method])) != 49:
            raise ValueError(f"{method} confusion matrix does not contain 49 companies")
    for row_index in range(len(LABELS)):
        if sum(matrices["Single"][row_index]) != sum(matrices["LLMC"][row_index]):
            raise ValueError("Reference-label totals differ between matrices")
    return matrices


def draw_matrix(axis: plt.Axes, method: str, matrix: list[list[int]], maximum: int):
    cmap = LinearSegmentedColormap.from_list("thesis_blue", PALETTE)
    image = axis.imshow(matrix, cmap=cmap, vmin=0, vmax=maximum, aspect="equal")
    axis.set_xlabel("Aggregated prediction", fontsize=10, labelpad=8)
    axis.set_ylabel("Reviewed label", fontsize=10, labelpad=8)
    axis.set_xticks(range(4), LABELS)
    axis.set_yticks(range(4), LABELS)
    axis.tick_params(length=0, labelsize=10)
    axis.set_xticks([i - 0.5 for i in range(1, 4)], minor=True)
    axis.set_yticks([i - 0.5 for i in range(1, 4)], minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index, row_index, str(value),
                ha="center", va="center", fontsize=10,
                fontweight="bold" if row_index == column_index else "normal",
                color="white" if value >= maximum * 0.45 else "#17365D",
            )
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def make_figure(matrices: dict[str, list[list[int]]], methods: tuple[str, ...], maximum: int) -> plt.Figure:
    figure, axes = plt.subplots(
        1, len(methods),
        figsize=(6.6, 3.8) if len(methods) == 2 else (6.6, 4.6),
        layout="constrained",
    )
    if len(methods) == 1:
        axes = [axes]
    for axis, method in zip(axes, methods, strict=True):
        image = draw_matrix(axis, method, matrices[method], maximum)
        if len(methods) > 1:
            name = "Single classifier" if method == "Single" else "LLMC"
            axis.set_xlabel(f"{name}\nAggregated prediction", fontsize=10)
            axis.tick_params(axis="x", labelrotation=30)
            plt.setp(axis.get_xticklabels(), ha="right")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.86, pad=0.03)
    colorbar.set_label("Number of companies", fontsize=10)
    colorbar.ax.tick_params(labelsize=10)
    colorbar.outline.set_visible(False)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=OUTPUT_DIR / "confusion_matrices.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR.parent.parent / "figures")
    args = parser.parse_args()
    matrices = load_matrices(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    maximum = max(value for matrix in matrices.values() for row in matrix for value in row)
    for stem, methods in (
        ("confusion_matrices", METHODS),
        ("confusion_matrix_single", ("Single",)),
        ("confusion_matrix_llmc", ("LLMC",)),
    ):
        figure = make_figure(matrices, methods, maximum)
        for extension in ("png", "pdf"):
            path = args.output_dir / f"{stem}.{extension}"
            figure.savefig(path, dpi=300, bbox_inches=None, facecolor="white")
            print(path)
        plt.close(figure)
    print("Correct on diagonal:", {
        method: sum(matrices[method][i][i] for i in range(4)) for method in METHODS
    })


if __name__ == "__main__":
    main()
