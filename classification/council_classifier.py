"""Resumable three-worker-and-judge classification pipeline.

Importing this module makes no API calls. The experiment runner must
explicitly call ``run_all_councils`` with output paths and prompt versions.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

import requests

from config.models import (
    LLMC_JUDGE_MODEL,
    LLMC_WORKER_MODELS,
    MODELS,
    ClassificationOutput,
    ModelResult,
    classify_text,
)
from .single_classifier import build_prompt, iterate_local_companies


Classifier = Callable[[str, str], ModelResult]
OutputKey = tuple[int, int, str, int | None]
ResultKey = tuple[int, str, int]
_APPEND_LOCK = Lock()


def append_jsonl(row: dict[str, Any], path: Path) -> None:
    """Durably append one JSON record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND_LOCK, path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file, rejecting malformed records."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error
    return rows


def output_key(row: dict[str, Any]) -> OutputKey:
    worker_number = row.get("worker_number")
    return (
        int(row["company_id"]),
        int(row["run_number"]),
        str(row["role"]),
        int(worker_number) if worker_number is not None else None,
    )


def result_key(row: dict[str, Any]) -> ResultKey:
    return (
        int(row["company_id"]),
        str(row["architecture"]),
        int(row["run_number"]),
    )


def load_outputs(path: Path) -> dict[OutputKey, dict[str, Any]]:
    """Load intermediate outputs and reject duplicate role keys."""

    outputs: dict[OutputKey, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = output_key(row)
        if key in outputs:
            raise ValueError(f"Duplicate LLMC output key: {key}")
        classification_fields(row)
        if row["role"] not in {"Worker", "Judge"}:
            raise ValueError(f"Invalid LLMC role for key {key}: {row['role']}")
        if row["role"] == "Worker" and row.get("worker_number") not in {1, 2, 3}:
            raise ValueError(f"Invalid worker number for key {key}")
        if row["role"] == "Judge" and row.get("worker_number") is not None:
            raise ValueError(f"Judge worker_number must be null for key {key}")
        outputs[key] = row
    return outputs


def load_results(path: Path) -> dict[ResultKey, dict[str, Any]]:
    """Load final LLMC results and reject duplicate result keys."""

    results: dict[ResultKey, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = result_key(row)
        if key in results:
            raise ValueError(f"Duplicate LLMC result key: {key}")
        if row.get("architecture") != "LLMC":
            raise ValueError(f"Invalid architecture for LLMC result: {key}")
        classification_fields(row)
        results[key] = row
    return results


def classification_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Return only the anonymous four-field model output."""

    fields = {
        "ai_label": row.get("ai_label"),
        "confidence_score": row.get("confidence_score"),
        "ai_products_services": row.get("ai_products_services"),
        "supporting_evidence": row.get("supporting_evidence"),
    }
    return ClassificationOutput.model_validate(fields).model_dump()


def build_judge_prompt(
    worker_rows: list[dict[str, Any]],
    prompt_path: Path,
) -> str:
    """Insert three anonymous worker outputs into the Judge prompt."""

    if len(worker_rows) != 3:
        raise ValueError("The Judge requires exactly three worker outputs")
    template = prompt_path.read_text(encoding="utf-8")
    placeholder = "{{WORKER_OUTPUTS}}"
    if placeholder not in template:
        raise ValueError(f"Judge prompt is missing placeholder: {placeholder}")
    anonymous_outputs = {
        f"worker_{index}": classification_fields(row)
        for index, row in enumerate(worker_rows, start=1)
    }
    return template.replace(
        placeholder,
        json.dumps(anonymous_outputs, indent=2, ensure_ascii=False),
    )


def make_output_row(
    company_id: int,
    run_number: int,
    role: str,
    worker_number: int | None,
    model_name: str,
    result: ModelResult,
) -> dict[str, Any]:
    """Create one database-bound llmc_outputs record."""

    return {
        "company_id": company_id,
        "run_number": run_number,
        "role": role,
        "worker_number": worker_number,
        "model_name": MODELS[model_name],
        **result.classification.model_dump(),
        "cost": result.cost,
    }


def sum_costs(rows: Iterable[dict[str, Any]]) -> float:
    """Sum monetary costs without binary floating-point accumulation error."""

    return float(sum((Decimal(str(row["cost"])) for row in rows), Decimal("0")))


def run_council(
    company: dict[str, Any],
    run_number: int,
    outputs: dict[OutputKey, dict[str, Any]],
    results: dict[ResultKey, dict[str, Any]],
    *,
    outputs_path: Path,
    results_path: Path,
    worker_prompt_path: Path,
    judge_prompt_path: Path,
    classifier: Classifier = classify_text,
) -> tuple[str, dict[str, Any], float]:
    """Run or resume one council and return status, final row, and new API cost."""

    company_id = int(company["company_id"])
    final_key = (company_id, "LLMC", run_number)
    if final_key in results:
        return "skipped", results[final_key], 0.0

    website_text = str(company.get("website_text") or "").strip()
    if not website_text:
        raise ValueError(f"Company {company_id} has no website_text")
    worker_prompt = build_prompt(website_text, worker_prompt_path)
    new_api_cost = 0.0
    worker_rows: list[dict[str, Any]] = []

    for worker_number, model_name in enumerate(LLMC_WORKER_MODELS, start=1):
        key = (company_id, run_number, "Worker", worker_number)
        row = outputs.get(key)
        expected_model = MODELS[model_name]
        if row is not None:
            if row.get("model_name") != expected_model:
                raise ValueError(
                    f"Existing output {key} uses {row.get('model_name')}, "
                    f"expected {expected_model}"
                )
        else:
            model_result = classifier(model_name, worker_prompt)
            row = make_output_row(
                company_id,
                run_number,
                "Worker",
                worker_number,
                model_name,
                model_result,
            )
            append_jsonl(row, outputs_path)
            outputs[key] = row
            new_api_cost += model_result.cost
        worker_rows.append(row)

    judge_key = (company_id, run_number, "Judge", None)
    judge_row = outputs.get(judge_key)
    expected_judge_model = MODELS[LLMC_JUDGE_MODEL]
    if judge_row is not None:
        if judge_row.get("model_name") != expected_judge_model:
            raise ValueError(
                f"Existing output {judge_key} uses {judge_row.get('model_name')}, "
                f"expected {expected_judge_model}"
            )
    else:
        judge_result = classifier(
            LLMC_JUDGE_MODEL,
            build_judge_prompt(worker_rows, judge_prompt_path),
        )
        judge_row = make_output_row(
            company_id,
            run_number,
            "Judge",
            None,
            LLMC_JUDGE_MODEL,
            judge_result,
        )
        append_jsonl(judge_row, outputs_path)
        outputs[judge_key] = judge_row
        new_api_cost += judge_result.cost

    total_cost = sum_costs([*worker_rows, judge_row])
    final_row = {
        "company_id": company_id,
        "architecture": "LLMC",
        "run_number": run_number,
        **classification_fields(judge_row),
        "cost": total_cost,
    }
    append_jsonl(final_row, results_path)
    results[final_key] = final_row
    return "completed", final_row, new_api_cost


def run_all_councils(
    run_number: int,
    outputs_path: Path,
    results_path: Path,
    worker_prompt_path: Path,
    judge_prompt_path: Path,
    companies: Iterable[dict[str, Any]] | None = None,
    classifier: Classifier = classify_text,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Run or resume the LLMC pipeline for every company."""

    outputs = load_outputs(outputs_path)
    results = load_results(results_path)
    source = list(companies if companies is not None else iterate_local_companies())
    if concurrency < 1 or concurrency > 3:
        raise ValueError("Council concurrency must be between 1 and 3")
    if len({int(row['company_id']) for row in source}) != len(source):
        raise ValueError("Duplicate company IDs in council input cohort")
    summary: dict[str, Any] = {
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "failed_company_ids": [],
        "new_api_cost": 0.0,
    }

    def evaluate(company: dict[str, Any]):
        company_id = int(company["company_id"])
        print(f"LLMC company {company_id}: run {run_number}")
        try:
            status, final_row, new_cost = run_council(
                company,
                run_number,
                outputs,
                results,
                outputs_path=outputs_path,
                results_path=results_path,
                worker_prompt_path=worker_prompt_path,
                judge_prompt_path=judge_prompt_path,
                classifier=classifier,
            )
        except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
            print(f"FAIL company {company_id}: {error}")
            return company_id, "failed", None, 0.0
        return company_id, status, final_row, new_cost

    def record(outcome):
        company_id, status, final_row, new_cost = outcome
        summary[status] += 1
        if status == "failed":
            summary["failed_company_ids"].append(company_id)
            return
        summary["new_api_cost"] += new_cost
        if status == "skipped":
            print(f"SKIP company {company_id}: final run already exists")
        else:
            print(
                f"DONE company {company_id}: {final_row['ai_label']} "
                f"(council cost ${final_row['cost']:.8f})"
            )

    if concurrency == 1:
        for company in source:
            record(evaluate(company))
    else:
        # A company belongs to exactly one task. Only durable appends share a lock;
        # each company's workers and judge still execute in their original order.
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(evaluate, company) for company in source]
            try:
                for future in as_completed(futures):
                    record(future.result())
            except BaseException:
                # Do not start queued paid requests after an interrupted batch.
                for future in futures:
                    future.cancel()
                raise

    return summary
