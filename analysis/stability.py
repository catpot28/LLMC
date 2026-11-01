"""Summarize label changes and reported-confidence variation across three runs.

Run ``python3 -m analysis.build_results`` first, then
``python3 -m analysis.stability``. The cohort contains three V2 evaluations
with matching saved prompt hashes. Provider behavior may change over time.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .build_results import LABELS, OUTPUT_DIR, RUNS
from .performance import load_company_results


ARCHITECTURES = {"single": "Single", "llmc": "LLMC"}


def build_stability(company_results: Path, runs: tuple[str, ...] = RUNS) -> list[dict[str, Any]]:
    companies = load_company_results(company_results)
    summaries = []
    for architecture, display in ARCHITECTURES.items():
        changes_per_company: list[int] = []
        dispersions: list[float] = []
        confidence_scores: list[float] = []
        for company_id, company in companies.items():
            labels = [company[f"{architecture}_{run}_label"] for run in runs]
            scores = [float(company[f"{architecture}_{run}_confidence"]) for run in runs]
            if any(label not in LABELS for label in labels):
                raise ValueError(f"Invalid label for company {company_id}")
            if any(not math.isfinite(score) or not 0 <= score <= 100 for score in scores):
                raise ValueError(f"Invalid confidence for company {company_id}")
            changes = sum(left != right for left, right in zip(labels, labels[1:]))
            dispersion = statistics.stdev(scores)
            saved_lfr = float(company[f"{architecture}_label_flip_rate"])
            saved_dispersion = float(company[f"{architecture}_confidence_score_dispersion"])
            if not math.isclose(changes / 2, saved_lfr, abs_tol=1e-12):
                raise ValueError(f"Label flip rate mismatch for company {company_id}")
            if not math.isclose(dispersion, saved_dispersion, abs_tol=1e-12):
                raise ValueError(f"Confidence dispersion mismatch for company {company_id}")
            changes_per_company.append(changes)
            dispersions.append(dispersion)
            confidence_scores.extend(scores)

        changes_distribution = Counter(changes_per_company)
        comparisons = len(companies) * (len(runs) - 1)
        total_changes = sum(changes_per_company)
        summaries.append({
            "architecture": display,
            "companies": len(companies),
            "runs": len(runs),
            "comparisons": comparisons,
            "label_changes": total_changes,
            "mean_label_flip_rate": total_changes / comparisons,
            "companies_with_0_changes": changes_distribution[0],
            "companies_with_1_change": changes_distribution[1],
            "companies_with_2_changes": changes_distribution[2],
            "mean_confidence_dispersion_points": statistics.mean(dispersions),
            "companies_with_0_confidence_dispersion": sum(value == 0 for value in dispersions),
            "minimum_reported_confidence": min(confidence_scores),
            "maximum_reported_confidence": max(confidence_scores),
            "outputs_reporting_100": sum(value == 100 for value in confidence_scores),
            "total_confidence_outputs": len(confidence_scores),
        })
    return summaries


def label_flip_distribution(summaries: list[dict[str, Any]]) -> list[dict[str, str | int]]:
    by_name = {row["architecture"]: row for row in summaries}
    if set(by_name) != set(ARCHITECTURES.values()):
        raise ValueError("Expected one Single and one LLMC stability summary")
    output: list[dict[str, str | int]] = []
    for changes in range(3):
        suffix = "change" if changes == 1 else "changes"
        output.append({
            "label_changes_per_company": str(changes),
            **{name: by_name[name][f"companies_with_{changes}_{suffix}"]
               for name in ARCHITECTURES.values()},
        })
    output.append({
        "label_changes_per_company": "Total label changes",
        **{name: by_name[name]["label_changes"] for name in ARCHITECTURES.values()},
    })
    output.append({
        "label_changes_per_company": "Mean label flip rate",
        **{name: f"{by_name[name]['mean_label_flip_rate']:.2%}"
           for name in ARCHITECTURES.values()},
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-results", type=Path, default=OUTPUT_DIR / "company_results.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--runs", nargs=3, default=RUNS, metavar="EXPERIMENT")
    args = parser.parse_args()
    summaries = build_stability(args.company_results, tuple(args.runs))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "stability_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved {len(summaries)} stability rows to {path}")
    distribution_path = args.output_dir / "label_flip_distribution.csv"
    with distribution_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("label_changes_per_company", *ARCHITECTURES.values()))
        writer.writeheader()
        writer.writerows(label_flip_distribution(summaries))
    print(f"Saved label-flip distribution to {distribution_path}")
    for row in summaries:
        print(
            f"{row['architecture']}: "
            f"{row['label_changes']}/{row['comparisons']} label changes, "
            f"mean confidence dispersion {row['mean_confidence_dispersion_points']:.3f}"
        )


if __name__ == "__main__":
    main()
