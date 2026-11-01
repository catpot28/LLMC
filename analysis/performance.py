"""Summarize accuracy and errors for Single, LLMC, workers, and worker voting.

Run ``python3 -m analysis.build_results`` first, then
``python3 -m analysis.performance``. This script makes no API requests.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .build_results import EXPERIMENTS, LABELS, OUTPUT_DIR, RUNS, aggregate


METHODS = (
    "Single", "GPT worker", "Mistral worker", "Gemini worker", "Worker vote", "LLMC"
)
WORKER_NAMES = {1: "GPT worker", 2: "Mistral worker", 3: "Gemini worker"}


def load_company_results(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python3 -m analysis.build_results first")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {int(row["company_id"]): row for row in rows}
    if len(rows) != 49 or len(keyed) != 49:
        raise ValueError("Expected 49 unique rows in company_results.csv")
    return keyed


def worker_predictions(
    companies: dict[int, dict[str, str]], runs: tuple[str, ...] = RUNS,
) -> dict[tuple[str, str], dict[int, str]]:
    predictions: dict[tuple[str, str], dict[int, str]] = {}
    ids = set(companies)
    for run in runs:
        base = EXPERIMENTS / run
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        path = base / "council/worker_outputs.jsonl"
        with path.open(encoding="utf-8") as handle:
            records: list[dict[str, Any]] = [json.loads(line) for line in handle if line.strip()]
        if len(records) != 4 * len(ids):
            raise ValueError(f"Expected four council records per company in {path}")

        by_company: dict[int, list[dict[str, Any]]] = {company_id: [] for company_id in ids}
        for record in records:
            company_id = int(record["company_id"])
            if company_id not in by_company:
                raise ValueError(f"Unexpected company {company_id} in {path}")
            if int(record["run_number"]) != manifest["run_number"]:
                raise ValueError(f"Recorded run number mismatch for {company_id} in {path}")
            if record["ai_label"] not in LABELS:
                raise ValueError(f"Invalid label for company {company_id} in {path}")
            by_company[company_id].append(record)

        for name in (*WORKER_NAMES.values(), "Worker vote"):
            predictions[(name, run)] = {}
        for company_id, company_records in by_company.items():
            workers = [record for record in company_records if record["role"] == "Worker"]
            judges = [record for record in company_records if record["role"] == "Judge"]
            if len(workers) != 3 or len(judges) != 1:
                raise ValueError(f"Expected three workers and one judge for {company_id} in {path}")
            indexed = {int(record["worker_number"]): record for record in workers}
            if set(indexed) != set(WORKER_NAMES):
                raise ValueError(f"Worker numbers are incomplete or duplicated for {company_id} in {path}")
            for number, name in WORKER_NAMES.items():
                record = indexed[number]
                if record["model_name"] != manifest["models"]["workers"][number - 1]:
                    raise ValueError(f"Worker model mismatch for {company_id} in {path}")
                predictions[(name, run)][company_id] = record["ai_label"]
            judge = judges[0]
            if judge["model_name"] != manifest["models"]["judge"]:
                raise ValueError(f"Judge model mismatch for {company_id} in {path}")
            if judge["ai_label"] != companies[company_id][f"llmc_{run}_label"]:
                raise ValueError(f"Judge and final LLMC labels differ for {company_id} in {path}")
            ordered_labels = [indexed[number]["ai_label"] for number in WORKER_NAMES]
            predictions[("Worker vote", run)][company_id] = aggregate(ordered_labels)[0]
    return predictions


def metrics(predicted: dict[int, str], reference: dict[int, str]) -> tuple[dict[str, Any], Counter[tuple[str, str]]]:
    if predicted.keys() != reference.keys():
        raise ValueError("Prediction and reference company IDs differ")
    confusion = Counter((reference[company_id], predicted[company_id]) for company_id in reference)
    correct = sum(predicted[company_id] == reference[company_id] for company_id in reference)
    distances = Counter(
        abs(LABELS.index(predicted[company_id]) - LABELS.index(reference[company_id]))
        for company_id in reference
    )
    precisions = []
    recalls = []
    f1_scores = []
    result: dict[str, Any] = {
        "n": len(reference),
        "correct": correct,
        "accuracy": correct / len(reference),
        "ordinal_mae": sum(distance * count for distance, count in distances.items()) / len(reference),
    }
    for label in LABELS:
        true_positive = confusion[label, label]
        reference_count = sum(confusion[label, candidate] for candidate in LABELS)
        predicted_count = sum(confusion[candidate, label] for candidate in LABELS)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / reference_count if reference_count else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        result[f"{label.lower()}_correct"] = true_positive
        result[f"{label.lower()}_reference_count"] = reference_count
    result["macro_precision"] = sum(precisions) / len(precisions)
    result["macro_recall"] = sum(recalls) / len(recalls)
    result["macro_f1"] = sum(f1_scores) / len(f1_scores)
    for distance in range(4):
        result[f"distance_{distance}"] = distances[distance]
    return result, confusion


def build_performance(company_results: Path, runs: tuple[str, ...] = RUNS) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    companies = load_company_results(company_results)
    reference = {company_id: row["reviewed_label"] for company_id, row in companies.items()}
    if any(label not in LABELS for label in reference.values()):
        raise ValueError("Invalid reviewed reference label")
    predictions: dict[tuple[str, str], dict[int, str]] = {}
    for run in runs:
        for method, prefix in (("Single", "single"), ("LLMC", "llmc")):
            predictions[(method, run)] = {
                company_id: row[f"{prefix}_{run}_label"] for company_id, row in companies.items()
            }
    predictions.update(worker_predictions(companies, runs))

    for method in METHODS:
        predictions[(method, "aggregated")] = {
            company_id: aggregate([predictions[(method, run)][company_id] for run in runs])[0]
            for company_id in reference
        }
    for method, prefix in (("Single", "single"), ("LLMC", "llmc")):
        if any(
            predictions[(method, "aggregated")][company_id] != row[f"{prefix}_aggregated_label"]
            for company_id, row in companies.items()
        ):
            raise ValueError(f"{method} aggregation differs from company_results.csv")

    summary: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for evaluation in (*runs, "aggregated"):
            figures, confusion = metrics(predictions[(method, evaluation)], reference)
            summary.append({"method": method, "evaluation": evaluation, **figures})
            if evaluation == "aggregated":
                for actual in LABELS:
                    for predicted in LABELS:
                        confusion_rows.append({
                            "method": method,
                            "reviewed_label": actual,
                            "predicted_label": predicted,
                            "count": confusion[actual, predicted],
                        })

    vote = predictions[("Worker vote", "aggregated")]
    judge = predictions[("LLMC", "aggregated")]
    judge_changes = {
        "corrected_vote_errors": sum(
            vote[c] != reference[c] and judge[c] == reference[c] for c in reference
        ),
        "introduced_errors": sum(
            vote[c] == reference[c] and judge[c] != reference[c] for c in reference
        ),
        "changed_both_incorrect": sum(
            vote[c] != judge[c] and vote[c] != reference[c] and judge[c] != reference[c]
            for c in reference
        ),
    }
    return summary, confusion_rows, judge_changes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-results", type=Path, default=OUTPUT_DIR / "company_results.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--runs", nargs=3, default=RUNS, metavar="EXPERIMENT")
    args = parser.parse_args()
    summary, confusion, judge_changes = build_performance(args.company_results, tuple(args.runs))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "performance_summary.csv", summary)
    write_csv(args.output_dir / "confusion_matrices.csv", confusion)
    print(f"Saved {len(summary)} performance rows and {len(confusion)} confusion counts to {args.output_dir}")
    print("Aggregated correct:", {
        row["method"]: row["correct"] for row in summary if row["evaluation"] == "aggregated"
    })
    print("Judge versus worker vote:", judge_changes)


if __name__ == "__main__":
    main()
