"""Calculate the exact paired McNemar test for aggregated Single and LLMC labels.

Run ``python3 -m analysis.build_results`` first, then
``python3 -m analysis.figure_generators.mcnemar``.
This script makes no model API calls.
"""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..build_results import LABELS, OUTPUT_DIR
from ..performance import load_company_results


def exact_two_sided_p(single_only: int, llmc_only: int) -> float:
    """Exact two-sided binomial test conditional on discordant pairs."""
    discordant = single_only + llmc_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(comb(discordant, k) for k in range(min(single_only, llmc_only) + 1))
    return min(1.0, 2 * lower_tail / 2**discordant)


def calculate(path: Path) -> dict[str, int | float]:
    companies = load_company_results(path)
    both_correct = single_only = llmc_only = both_incorrect = 0
    for company_id, row in companies.items():
        reference = row["reviewed_label"]
        single = row["single_aggregated_label"]
        llmc = row["llmc_aggregated_label"]
        if any(label not in LABELS for label in (reference, single, llmc)):
            raise ValueError(f"Invalid label for company {company_id}")
        single_correct = single == reference
        llmc_correct = llmc == reference
        if row["single_aggregated_correct"] != str(single_correct):
            raise ValueError(f"Single correctness mismatch for company {company_id}")
        if row["llmc_aggregated_correct"] != str(llmc_correct):
            raise ValueError(f"LLMC correctness mismatch for company {company_id}")
        if single_correct and llmc_correct:
            both_correct += 1
        elif single_correct:
            single_only += 1
        elif llmc_correct:
            llmc_only += 1
        else:
            both_incorrect += 1
    result = {
        "both_correct": both_correct,
        "single_only_correct": single_only,
        "llmc_only_correct": llmc_only,
        "both_incorrect": both_incorrect,
        "single_correct_total": both_correct + single_only,
        "llmc_correct_total": both_correct + llmc_only,
        "companies": len(companies),
        "discordant_pairs": single_only + llmc_only,
        "exact_two_sided_p": exact_two_sided_p(single_only, llmc_only),
    }
    if sum((both_correct, single_only, llmc_only, both_incorrect)) != len(companies):
        raise ValueError("Paired outcome counts do not sum to the cohort size")
    return result


def draw_table(result: dict[str, int | float]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.6, 3.3), layout="constrained")
    axis.set_xlim(0, 1)
    axis.set_ylim(0.12, 0.91)
    axis.axis("off")
    font = {"family": "sans-serif", "size": 10}
    axis.text(0.70, 0.83, "LLMC", ha="center", va="center", fontdict=font)
    xs = (0.09, 0.48, 0.70, 0.91)
    for x, label in zip(xs, ("Single classifier", "Correct", "Incorrect", "Total"), strict=True):
        axis.text(x, 0.74, label, ha="left" if x == xs[0] else "center",
                  va="center", fontdict=font)
    row_data = (
        ("Correct", result["both_correct"], result["single_only_correct"], result["single_correct_total"]),
        ("Incorrect", result["llmc_only_correct"], result["both_incorrect"],
         result["companies"] - result["single_correct_total"]),
        ("Total", result["llmc_correct_total"],
         result["companies"] - result["llmc_correct_total"], result["companies"]),
    )
    for y, values in zip((0.63, 0.54, 0.45), row_data, strict=True):
        for x, value in zip(xs, values, strict=True):
            axis.text(x, y, str(value), ha="left" if x == xs[0] else "center",
                      va="center", fontdict=font)
    for y, linewidth in ((0.88, 1.2), (0.79, 0.6), (0.70, 0.6), (0.39, 1.2)):
        axis.plot((0.07, 0.94), (y, y), color="#222222", linewidth=linewidth)
    axis.text(0.09, 0.30, "Exact two-sided McNemar p-value", ha="left", va="center", fontdict=font)
    axis.text(0.91, 0.30, f"{result['exact_two_sided_p']:.4f}", ha="center", va="center", fontdict=font)
    axis.text(0.09, 0.19,
              f"Note: {result['discordant_pairs']} discordant pairs from {result['companies']} companies,\n"
              "with one aggregated label per architecture.",
              ha="left", va="center", fontdict={**font, "size": 9}, style="italic")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-results", type=Path, default=OUTPUT_DIR / "company_results.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=OUTPUT_DIR.parent.parent / "figures")
    args = parser.parse_args()
    result = calculate(args.company_results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "mcnemar_test.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(csv_path)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure = draw_table(result)
    for extension in ("png", "pdf"):
        path = args.figure_dir / f"mcnemar_test.{extension}"
        figure.savefig(path, dpi=300, bbox_inches=None, facecolor="white")
        print(path)
    plt.close(figure)
    print(f"Single-only correct: {result['single_only_correct']}; "
          f"LLMC-only correct: {result['llmc_only_correct']}; "
          f"exact two-sided p = {result['exact_two_sided_p']:.6f}")


if __name__ == "__main__":
    main()
