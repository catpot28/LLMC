"""Summarize recorded API costs alongside observed classification accuracy.

Run the company-results and performance scripts first, then
``python3 -m analysis.costs``. Main-run costs cover three evaluations of each
of 49 companies. The separate Gemini Single comparison has 48 recorded outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from .build_results import EXPERIMENTS, OUTPUT_DIR, REFERENCE, RUNS, keyed_rows
from .performance import load_company_results


WORKER_NAMES = {1: "GPT worker", 2: "Mistral worker", 3: "Gemini worker"}
MAIN_METHODS = ("Single", *WORKER_NAMES.values(), "Worker vote", "LLMC")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recorded_cost(record: dict[str, Any], path: Path) -> Decimal:
    cost = Decimal(str(record["cost"]))
    if not cost.is_finite() or cost < 0:
        raise ValueError(f"Invalid recorded cost in {path}")
    return cost


def aggregated_accuracy(path: Path) -> dict[str, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python3 -m analysis.performance first")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    aggregated = {
        row["method"]: (int(row["correct"]), int(row["n"]))
        for row in rows if row["evaluation"] == "aggregated"
    }
    if any(method not in aggregated or aggregated[method][1] != 49 for method in MAIN_METHODS):
        raise ValueError("Missing or invalid aggregated performance for main methods")
    return aggregated


def build_costs(company_results: Path, performance_summary: Path, runs: tuple[str, ...] = RUNS) -> list[dict[str, Any]]:
    companies = load_company_results(company_results)
    accuracy = aggregated_accuracy(performance_summary)
    main_ids = set(companies)
    totals: dict[str, Decimal] = defaultdict(Decimal)
    model_ids: dict[str, str] = {}
    for run in runs:
        base = EXPERIMENTS / run
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        if manifest["experiment_id"] != run:
            raise ValueError(f"Experiment ID mismatch for {run}")
        model_ids["Single"] = manifest["models"]["single"]
        model_ids["LLMC"] = "three workers + judge"
        model_ids["Worker vote"] = "three workers; no judge call"
        model_ids["Qwen judge"] = manifest["models"]["judge"]
        for number, name in WORKER_NAMES.items():
            model_ids[name] = manifest["models"]["workers"][number - 1]

        single_path = base / "single/results.jsonl"
        council_path = base / "council/results.jsonl"
        worker_path = base / "council/worker_outputs.jsonl"
        single = keyed_rows(single_path)
        council = keyed_rows(council_path)
        worker_rows = read_jsonl(worker_path)
        if single.keys() != main_ids or council.keys() != main_ids:
            raise ValueError(f"Main result company IDs differ in {run}")
        if len(worker_rows) != 4 * len(main_ids):
            raise ValueError(f"Expected four council outputs per company in {worker_path}")

        by_company: dict[int, list[dict[str, Any]]] = {company_id: [] for company_id in main_ids}
        for record in worker_rows:
            company_id = int(record["company_id"])
            if company_id not in by_company:
                raise ValueError(f"Unexpected company {company_id} in {worker_path}")
            if int(record["run_number"]) != manifest["run_number"]:
                raise ValueError(f"Recorded run number mismatch in {worker_path}")
            by_company[company_id].append(record)

        for company_id in main_ids:
            single_cost = recorded_cost(single[company_id], single_path)
            council_cost = recorded_cost(council[company_id], council_path)
            saved = companies[company_id]
            if abs(single_cost - Decimal(saved[f"single_{run}_cost_usd"])) > Decimal("0.00000001"):
                raise ValueError(f"Single cost differs from company_results.csv for {company_id}")
            if abs(council_cost - Decimal(saved[f"llmc_{run}_cost_usd"])) > Decimal("0.00000001"):
                raise ValueError(f"LLMC cost differs from company_results.csv for {company_id}")
            totals["Single"] += single_cost
            totals["LLMC"] += council_cost

            records = by_company[company_id]
            workers = [record for record in records if record["role"] == "Worker"]
            judges = [record for record in records if record["role"] == "Judge"]
            if len(workers) != 3 or len(judges) != 1:
                raise ValueError(f"Expected three workers and one judge for {company_id} in {run}")
            indexed = {int(record["worker_number"]): record for record in workers}
            if set(indexed) != set(WORKER_NAMES):
                raise ValueError(f"Worker numbers are incomplete or duplicated for {company_id} in {run}")
            component_total = Decimal(0)
            for number, name in WORKER_NAMES.items():
                worker = indexed[number]
                if worker["model_name"] != model_ids[name]:
                    raise ValueError(f"Worker model mismatch for {company_id} in {run}")
                cost = recorded_cost(worker, worker_path)
                totals[name] += cost
                component_total += cost
            judge = judges[0]
            if judge["model_name"] != model_ids["Qwen judge"]:
                raise ValueError(f"Judge model mismatch for {company_id} in {run}")
            judge_cost = recorded_cost(judge, worker_path)
            totals["Qwen judge"] += judge_cost
            if abs(component_total + judge_cost - council_cost) > Decimal("0.00000001"):
                raise ValueError(f"Council components do not reconcile for {company_id} in {run}")
    totals["Worker vote"] = sum((totals[name] for name in WORKER_NAMES.values()), Decimal(0))
    if abs(totals["Worker vote"] + totals["Qwen judge"] - totals["LLMC"]) > Decimal("0.00000001"):
        raise ValueError("Council total does not reconcile to worker and judge totals")

    output = []
    for method in ("Single", "LLMC", *WORKER_NAMES.values(), "Worker vote", "Qwen judge"):
        correct, n = accuracy.get(method, (None, 49))
        output.append({
            "scenario": "main_three_runs",
            "method": method,
            "model_or_components": model_ids[method],
            "companies": n,
            "evaluations_per_company": 3,
            "company_evaluations": 49 * 3,
            "model_calls": 49 * 3 * (4 if method == "LLMC" else 3 if method == "Worker vote" else 1),
            "total_recorded_cost_usd": str(totals[method]),
            "mean_cost_per_company_evaluation_usd": str(totals[method] / Decimal(147)),
            "aggregated_correct": "" if correct is None else correct,
            "aggregated_accuracy": "" if correct is None else correct / n,
        })

    gemini_base = EXPERIMENTS / "model_comparison_gemini_v2"
    gemini_manifest = json.loads((gemini_base / "manifest.json").read_text(encoding="utf-8"))
    gemini_path = gemini_base / "single/results.jsonl"
    gemini = keyed_rows(gemini_path)
    reference = keyed_rows(REFERENCE)
    if len(gemini) != 48 or not gemini.keys() <= main_ids:
        raise ValueError("Expected 48 unique Gemini comparison outputs from the main cohort")
    if sorted(main_ids - gemini.keys()) != gemini_manifest["missing_company_ids"]:
        raise ValueError("Gemini comparison missing-company IDs differ from its manifest")
    if gemini_manifest["model"]["id"] != model_ids["Gemini worker"]:
        raise ValueError("Gemini comparison model differs from council Gemini worker")
    gemini_total = sum((recorded_cost(record, gemini_path) for record in gemini.values()), Decimal(0))
    gemini_correct = sum(
        record["ai_label"] == reference[company_id]["manual_ai_label"]
        for company_id, record in gemini.items()
    )
    output.append({
        "scenario": "separate_gemini_single_run",
        "method": "Gemini standalone",
        "model_or_components": gemini_manifest["model"]["id"],
        "companies": len(gemini),
        "evaluations_per_company": 1,
        "company_evaluations": len(gemini),
        "model_calls": len(gemini),
        "total_recorded_cost_usd": str(gemini_total),
        "mean_cost_per_company_evaluation_usd": str(gemini_total / Decimal(len(gemini))),
        "aggregated_correct": gemini_correct,
        "aggregated_accuracy": gemini_correct / len(gemini),
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-results", type=Path, default=OUTPUT_DIR / "company_results.csv")
    parser.add_argument("--performance-summary", type=Path, default=OUTPUT_DIR / "performance_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--runs", nargs=3, default=RUNS, metavar="EXPERIMENT")
    args = parser.parse_args()
    rows = build_costs(args.company_results, args.performance_summary, tuple(args.runs))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "cost_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} cost rows to {path}")
    for row in rows:
        print(
            f"{row['method']}: ${Decimal(row['total_recorded_cost_usd']):.6f} "
            f"over {row['company_evaluations']} company evaluations"
        )


if __name__ == "__main__":
    main()
