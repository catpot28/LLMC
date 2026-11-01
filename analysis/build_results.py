"""Join saved model outputs and reviewed labels into one company-level table.

Run from the workspace root with ``python3 -m analysis.build_results``.
The table includes Single, three workers, worker vote, LLMC, and judge costs
for all three runs. No model API calls are made; source files are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "data/classification_inputs/companies.jsonl"
REFERENCE = ROOT / "data/human_labels/reviewed_reference_companies.jsonl"
EXPERIMENTS = ROOT / "data/experiments"
OUTPUT_DIR = ROOT / "data/analysis"
RUNS = ("run_2_v2", "run_3_v2", "run_4_v2")
ARCHITECTURES = {"single": "Single", "llmc": "LLMC"}
WORKERS = {1: "gpt_worker", 2: "mistral_worker", 3: "gemini_worker"}
LABELS = ("None", "Low", "Medium", "High")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(path)
    keyed = {int(row["company_id"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate company ID in {path}")
    return keyed


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(labels: list[str]) -> tuple[str, str]:
    """Use a majority of three, or the ordinal median if all labels differ."""
    if len(labels) != 3 or any(label not in LABELS for label in labels):
        raise ValueError(f"Expected three valid labels, got {labels}")
    label, count = Counter(labels).most_common(1)[0]
    if count >= 2:
        return label, "majority"
    return sorted(labels, key=LABELS.index)[1], "ordinal_median_tie_break"


def build_table(runs: tuple[str, ...] = RUNS) -> list[dict[str, Any]]:
    if len(runs) != 3 or len(set(runs)) != 3:
        raise ValueError("Expected three distinct V2 experiment folders")
    if any(not name or Path(name).name != name or name in (".", "..") for name in runs):
        raise ValueError("Experiment folders must be names directly under data/experiments")
    # Check the whole cohort before computing or publishing any replacement table.
    cohort_models = None
    for run_number, run in enumerate(runs, start=1):
        base = EXPERIMENTS / run
        for relative in ("manifest.json", "single/results.jsonl",
                         "council/worker_outputs.jsonl", "council/results.jsonl"):
            if not (base / relative).is_file():
                raise FileNotFoundError(
                    f"Three-V2 analysis requires completed {run}: missing {base / relative}. "
                    "Existing analysis outputs have not been replaced."
                )
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("prompt_version") != "v2" or manifest.get("run_number") != run_number:
            raise ValueError(f"Expected V2 repetition {run_number} in {base}")
        for role, filename in (("workers_and_single", "single_v2.txt"),
                               ("judge", "judge_v2.txt")):
            if manifest["prompts"][role]["sha256"] != sha256(ROOT / "prompts" / filename):
                raise ValueError(f"V2 prompt hash mismatch for {role} in {base}")
        if cohort_models is None:
            cohort_models = manifest["models"]
        elif manifest["models"] != cohort_models:
            raise ValueError(f"Model settings differ across the V2 cohort in {base}")

    inputs = keyed_rows(INPUTS)
    reference = keyed_rows(REFERENCE)
    if len(inputs) != 49 or len(reference) != 51:
        raise ValueError("Expected 49 classifier inputs and 51 reviewed labels")
    if not inputs.keys() <= reference.keys():
        raise ValueError("Some classifier inputs have no reviewed reference label")

    results: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    components: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for run in runs:
        base = EXPERIMENTS / run
        manifest_path = base / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["experiment_id"] != run:
            raise ValueError(f"Experiment ID mismatch in {manifest_path}")
        if manifest["inputs"]["sha256"] != sha256(INPUTS):
            raise ValueError(f"Classifier input hash mismatch in {manifest_path}")
        if manifest["inputs"]["companies"] != len(inputs):
            raise ValueError(f"Company count mismatch in {manifest_path}")
        for architecture, expected_name in ARCHITECTURES.items():
            relative = (
                "single/results.jsonl" if architecture == "single"
                else "council/results.jsonl"
            )
            path = base / relative
            records = keyed_rows(path)
            if records.keys() != inputs.keys():
                raise ValueError(f"Company IDs differ from classifier inputs in {path}")
            for company_id, record in records.items():
                if record["architecture"] != expected_name:
                    raise ValueError(f"Wrong architecture for company {company_id} in {path}")
                if int(record["run_number"]) != manifest["run_number"]:
                    raise ValueError(f"Wrong recorded run number for company {company_id} in {path}")
                if record["ai_label"] not in LABELS:
                    raise ValueError(f"Invalid label for company {company_id} in {path}")
                if not math.isfinite(float(record["confidence_score"])) or not 0 <= float(record["confidence_score"]) <= 100:
                    raise ValueError(f"Invalid confidence for company {company_id} in {path}")
                if not math.isfinite(float(record["cost"])) or float(record["cost"]) < 0:
                    raise ValueError(f"Invalid cost for company {company_id} in {path}")
            results[(run, architecture)] = records

        worker_path = base / "council/worker_outputs.jsonl"
        worker_records = read_jsonl(worker_path)
        if len(worker_records) != 4 * len(inputs):
            raise ValueError(f"Expected three workers and one judge per company in {worker_path}")
        for record in worker_records:
            company_id = int(record["company_id"])
            if company_id not in inputs:
                raise ValueError(f"Unexpected company ID {company_id} in {worker_path}")
            if int(record["run_number"]) != manifest["run_number"]:
                raise ValueError(f"Wrong recorded run number for company {company_id} in {worker_path}")
            if record["ai_label"] not in LABELS:
                raise ValueError(f"Invalid label for company {company_id} in {worker_path}")
            confidence = float(record["confidence_score"])
            cost = float(record["cost"])
            if not math.isfinite(confidence) or not 0 <= confidence <= 100:
                raise ValueError(f"Invalid confidence for company {company_id} in {worker_path}")
            if not math.isfinite(cost) or cost < 0:
                raise ValueError(f"Invalid cost for company {company_id} in {worker_path}")
            if record["role"] == "Worker":
                number = int(record["worker_number"])
                if number not in WORKERS:
                    raise ValueError(f"Unknown worker number for company {company_id} in {worker_path}")
                name = WORKERS[number]
                expected_model = manifest["models"]["workers"][number - 1]
            elif record["role"] == "Judge" and record["worker_number"] is None:
                name = "judge"
                expected_model = manifest["models"]["judge"]
            else:
                raise ValueError(f"Unknown council role for company {company_id} in {worker_path}")
            if record["model_name"] != expected_model:
                raise ValueError(f"Model mismatch for company {company_id} in {worker_path}")
            group = components.setdefault((run, company_id), {})
            if name in group:
                raise ValueError(f"Duplicate {name} result for company {company_id} in {worker_path}")
            group[name] = record
        for company_id in inputs:
            if set(components.get((run, company_id), {})) != {*WORKERS.values(), "judge"}:
                raise ValueError(f"Incomplete council outputs for company {company_id} in {worker_path}")

    table = []
    for company_id, company in sorted(inputs.items()):
        reviewed = reference[company_id]
        if company["company_name"] != reviewed["company_name"]:
            raise ValueError(f"Company name mismatch for ID {company_id}")
        label = reviewed["manual_ai_label"]
        if label not in LABELS:
            raise ValueError(f"Invalid reviewed label for company {company_id}")
        row: dict[str, Any] = {
            "company_id": company_id,
            "company_name": company["company_name"],
            "reviewed_label": label,
        }
        for architecture in ARCHITECTURES:
            predictions: list[str] = []
            confidences: list[float] = []
            costs: list[float] = []
            for run in runs:
                record = results[(run, architecture)][company_id]
                prefix = f"{architecture}_{run}"
                row[f"{prefix}_label"] = record["ai_label"]
                row[f"{prefix}_confidence"] = record["confidence_score"]
                row[f"{prefix}_cost_usd"] = record["cost"]
                predictions.append(record["ai_label"])
                confidences.append(float(record["confidence_score"]))
                costs.append(float(record["cost"]))
            aggregated, method = aggregate(predictions)
            row[f"{architecture}_aggregated_label"] = aggregated
            row[f"{architecture}_aggregation_method"] = method
            row[f"{architecture}_aggregated_correct"] = aggregated == label
            row[f"{architecture}_label_flip_rate"] = sum(
                left != right for left, right in zip(predictions, predictions[1:])
            ) / 2
            row[f"{architecture}_confidence_score_dispersion"] = statistics.stdev(confidences)
            row[f"{architecture}_three_run_total_cost_usd"] = sum(costs)

        worker_labels: dict[str, list[str]] = {name: [] for name in WORKERS.values()}
        worker_costs: dict[str, list[float]] = {name: [] for name in WORKERS.values()}
        vote_labels: list[str] = []
        vote_costs: list[float] = []
        judge_costs: list[float] = []
        for run in runs:
            council = components[(run, company_id)]
            run_worker_labels = []
            run_worker_costs = []
            for number, name in WORKERS.items():
                record = council[name]
                prefix = f"{name}_{run}"
                row[f"{prefix}_label"] = record["ai_label"]
                row[f"{prefix}_confidence"] = record["confidence_score"]
                row[f"{prefix}_cost_usd"] = record["cost"]
                worker_labels[name].append(record["ai_label"])
                worker_costs[name].append(float(record["cost"]))
                run_worker_labels.append(record["ai_label"])
                run_worker_costs.append(float(record["cost"]))

            vote_label, vote_method = aggregate(run_worker_labels)
            vote_cost = sum(run_worker_costs)
            row[f"worker_vote_{run}_label"] = vote_label
            row[f"worker_vote_{run}_method"] = vote_method
            row[f"worker_vote_{run}_cost_usd"] = vote_cost
            vote_labels.append(vote_label)
            vote_costs.append(vote_cost)

            judge = council["judge"]
            if judge["ai_label"] != row[f"llmc_{run}_label"]:
                raise ValueError(f"Judge and final LLMC labels differ for company {company_id} in {run}")
            judge_cost = float(judge["cost"])
            row[f"judge_{run}_cost_usd"] = judge_cost
            judge_costs.append(judge_cost)
            if abs(vote_cost + judge_cost - float(row[f"llmc_{run}_cost_usd"])) > 1e-8:
                raise ValueError(f"Worker and judge costs do not reconcile for company {company_id} in {run}")

        for name in WORKERS.values():
            aggregated, method = aggregate(worker_labels[name])
            row[f"{name}_aggregated_label"] = aggregated
            row[f"{name}_aggregation_method"] = method
            row[f"{name}_aggregated_correct"] = aggregated == label
            row[f"{name}_three_run_total_cost_usd"] = sum(worker_costs[name])
        aggregated_vote, vote_method = aggregate(vote_labels)
        row["worker_vote_aggregated_label"] = aggregated_vote
        row["worker_vote_aggregation_method"] = vote_method
        row["worker_vote_aggregated_correct"] = aggregated_vote == label
        row["worker_vote_three_run_total_cost_usd"] = sum(vote_costs)
        row["judge_three_run_total_cost_usd"] = sum(judge_costs)
        table.append(row)

    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--runs", nargs=3, default=RUNS, metavar="EXPERIMENT")
    args = parser.parse_args()
    table = build_table(tuple(args.runs))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "company_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"Saved {len(table)} company rows to {csv_path}")
    correct = {
        name: sum(row[f"{name}_aggregated_correct"] for row in table)
        for name in (*ARCHITECTURES, *WORKERS.values(), "worker_vote")
    }
    print(f"Aggregated correct: {correct}")


if __name__ == "__main__":
    main()
