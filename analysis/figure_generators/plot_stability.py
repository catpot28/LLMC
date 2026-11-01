"""Plot stability figures from saved analysis tables; make no model API calls.

Run ``python3 -m analysis.build_results`` and ``python3 -m analysis.stability``
before ``python3 -m analysis.figure_generators.plot_stability``.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from ..build_results import OUTPUT_DIR, RUNS


ARCHITECTURES = ("Single", "LLMC")
PREFIXES = {"Single": "single", "LLMC": "llmc"}
DISPLAY = {"Single": "Single classifier", "LLMC": "LLMC"}
COLORS = {"Single": "#7FAED5", "LLMC": "#203A62"}


def load_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python3 -m analysis.stability first")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2 or {row["architecture"] for row in rows} != set(ARCHITECTURES):
        raise ValueError(f"Expected one Single and one LLMC row in {path}")
    summary = {row["architecture"]: row for row in rows}
    for architecture in ARCHITECTURES:
        row = summary[architecture]
        companies, runs = int(row["companies"]), int(row["runs"])
        comparisons = int(row["comparisons"])
        changes = int(row["label_changes"])
        if companies != 49 or runs != len(RUNS) or comparisons != companies * (runs - 1):
            raise ValueError(f"Unexpected study size for {architecture}")
        if not math.isclose(changes / comparisons, float(row["mean_label_flip_rate"]), abs_tol=1e-12):
            raise ValueError(f"Incorrect label flip rate for {architecture}")
        distribution = [int(row[f"companies_with_{count}_{'change' if count == 1 else 'changes'}"])
                        for count in range(3)]
        if sum(distribution) != companies or sum(count * number for count, number in enumerate(distribution)) != changes:
            raise ValueError(f"Label-change distribution does not match total for {architecture}")
        if int(row["total_confidence_outputs"]) != companies * runs:
            raise ValueError(f"Unexpected number of confidence scores for {architecture}")
    return summary


def load_confidences(path: Path, summary: dict[str, dict[str, str]],
                     runs: tuple[str, ...] = RUNS) -> dict[str, Counter[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python3 -m analysis.build_results first")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 49 or len({row["company_id"] for row in rows}) != 49:
        raise ValueError(f"Expected 49 distinct companies in {path}")
    counts: dict[str, Counter[float]] = {}
    for architecture in ARCHITECTURES:
        prefix = PREFIXES[architecture]
        values = [float(row[f"{prefix}_{run}_confidence"]) for row in rows for run in runs]
        if any(not math.isfinite(value) or not 0 <= value <= 100 for value in values):
            raise ValueError(f"Invalid {architecture} confidence score")
        counts[architecture] = Counter(values)
        saved = summary[architecture]
        if (len(values) != int(saved["total_confidence_outputs"])
                or counts[architecture][100] != int(saved["outputs_reporting_100"])
                or min(values) != float(saved["minimum_reported_confidence"])
                or max(values) != float(saved["maximum_reported_confidence"])):
            raise ValueError(f"{architecture} confidence scores disagree with stability summary")
    return counts


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E5E8EB", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#333333")
    axis.spines["bottom"].set_color("#333333")
    axis.tick_params(labelsize=10)


def confidence_dispersion(summary: dict[str, dict[str, str]]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.6, 4.6), layout="constrained")
    values = [float(summary[name]["mean_confidence_dispersion_points"]) for name in ARCHITECTURES]
    bars = axis.bar([DISPLAY[name] for name in ARCHITECTURES], values,
                    color=[COLORS[name] for name in ARCHITECTURES], width=0.55)
    axis.bar_label(bars, fmt="%.3f", padding=5, fontsize=10)
    axis.set_ylim(0, max(4.1, max(values) * 1.3))
    axis.set_ylabel("Mean confidence score dispersion (points)", fontsize=10)
    style_axis(axis)
    return figure


def confidence_distribution(counts: dict[str, Counter[float]], summary: dict[str, dict[str, str]]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.6, 4.6), layout="constrained")
    scores = sorted(set(counts["Single"]) | set(counts["LLMC"]))
    for architecture, offset in (("Single", -0.20), ("LLMC", 0.20)):
        total = int(summary[architecture]["total_confidence_outputs"])
        shares = [counts[architecture][score] / total for score in scores]
        axis.bar([score + offset for score in scores], shares, width=0.38,
                 color=COLORS[architecture], label=DISPLAY[architecture])
    axis.set_xlim(min(68.5, min(scores) - 1.5), 101.5)
    largest_share = max(count / int(summary[name]["total_confidence_outputs"])
                        for name in ARCHITECTURES for count in counts[name].values())
    axis.set_ylim(0, max(0.55, largest_share + 0.08))
    axis.set_xticks(list(range(min(70, int(min(scores) // 5) * 5), 101, 5)))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("Reported confidence score", fontsize=10)
    axis.set_ylabel("Share of recorded scores", fontsize=10)
    axis.legend(frameon=False, loc="upper left", fontsize=10)
    style_axis(axis)
    return figure


def label_flip_rate(summary: dict[str, dict[str, str]]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.6, 4.6), layout="constrained")
    values = [float(summary[name]["mean_label_flip_rate"]) for name in ARCHITECTURES]
    bars = axis.bar([DISPLAY[name] for name in ARCHITECTURES], values,
                    color=[COLORS[name] for name in ARCHITECTURES], width=0.55)
    axis.bar_label(bars, labels=[f"{value:.2%}" for value in values], padding=5, fontsize=10)
    axis.set_ylim(0, max(0.16, max(values) * 1.3))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_ylabel("Mean label flip rate", fontsize=10)
    style_axis(axis)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=OUTPUT_DIR / "stability_summary.csv")
    parser.add_argument("--company-results", type=Path, default=OUTPUT_DIR / "company_results.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR.parent.parent / "figures")
    parser.add_argument("--runs", nargs=3, default=RUNS, metavar="EXPERIMENT")
    args = parser.parse_args()
    summary = load_summary(args.summary)
    confidences = load_confidences(args.company_results, summary, tuple(args.runs))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stem, figure in (
        ("confidence_dispersion", confidence_dispersion(summary)),
        ("confidence_score_distribution", confidence_distribution(confidences, summary)),
        ("label_flip_rate", label_flip_rate(summary)),
    ):
        for extension in ("png", "pdf"):
            path = args.output_dir / f"{stem}.{extension}"
            figure.savefig(path, dpi=300, bbox_inches=None, facecolor="white")
            print(path)
        plt.close(figure)


if __name__ == "__main__":
    main()
