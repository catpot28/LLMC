"""Single-model classification using the frozen company input snapshot.

This module defines the classifier but makes no API calls when imported. The
experiment runner explicitly calls ``classify_all_companies`` with
an output path and the intended prompt version.
"""

import json
import os
from pathlib import Path
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import requests

from config.models import SINGLE_MODEL, ModelResult, classify_text


ROOT = Path(__file__).resolve().parent.parent
COMPANIES_PATH = ROOT / "data" / "classification_inputs" / "companies.jsonl"


ResultKey = tuple[int, str, int]
Classifier = Callable[[str, str], ModelResult]


def iterate_local_companies() -> Iterator[dict[str, Any]]:
    """Yield companies from the fixed local JSON Lines snapshot."""

    with COMPANIES_PATH.open(encoding="utf-8") as companies_file:
        for line_number, line in enumerate(companies_file, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {COMPANIES_PATH} at line {line_number}"
                    ) from error


def select_company(company_id: int | None) -> dict[str, Any]:
    """Select one company by ID, or the first company when no ID is supplied."""

    for company in iterate_local_companies():
        if company_id is None or company.get("company_id") == company_id:
            return company
    raise ValueError(f"Company {company_id} was not found in {COMPANIES_PATH}")


def build_prompt(website_text: str, prompt_path: Path) -> str:
    """Insert one company's website text into the versioned prompt."""

    template = prompt_path.read_text(encoding="utf-8")
    placeholder = "{{WEBSITE_TEXT}}"
    if placeholder not in template:
        raise ValueError(f"Prompt is missing placeholder: {placeholder}")
    return template.replace(placeholder, website_text)


def load_result_keys(path: Path) -> set[ResultKey]:
    """Load completed company/architecture/run keys from local results."""

    if not path.exists():
        return set()
    keys: set[ResultKey] = set()
    with path.open(encoding="utf-8") as results_file:
        for line_number, line in enumerate(results_file, start=1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from error
            try:
                keys.add(
                    (
                        int(result["company_id"]),
                        str(result["architecture"]),
                        int(result["run_number"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid result fields in {path} at line {line_number}"
                ) from error
    return keys


def result_exists(company_id: int, run_number: int, path: Path) -> bool:
    """Return whether this company/run already has a local Single result."""

    return (company_id, "Single", run_number) in load_result_keys(path)


def save_result(result: dict[str, Any], path: Path) -> None:
    """Append one validated result to the local JSON Lines results file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as results_file:
        results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        results_file.flush()
        os.fsync(results_file.fileno())


def classify_company(
    company: dict[str, Any],
    run_number: int,
    prompt_path: Path,
    model_name: str = SINGLE_MODEL,
    classifier: Classifier = classify_text,
) -> dict[str, Any]:
    """Classify one company and return its database-bound result record."""

    company_id = int(company["company_id"])
    website_text = str(company.get("website_text") or "").strip()
    if not website_text:
        raise ValueError(f"Company {company_id} has no website_text")

    model_result = classifier(model_name, build_prompt(website_text, prompt_path))
    return {
        "company_id": company_id,
        "architecture": "Single",
        "run_number": run_number,
        **model_result.classification.model_dump(),
        "cost": model_result.cost,
    }


def classify_all_companies(
    run_number: int,
    results_path: Path,
    prompt_path: Path,
    companies: Iterable[dict[str, Any]] | None = None,
    classifier: Classifier = classify_text,
    model_name: str = SINGLE_MODEL,
) -> dict[str, Any]:
    """Classify all unfinished companies sequentially and save each success."""

    completed_keys = load_result_keys(results_path)
    source = companies if companies is not None else iterate_local_companies()
    summary: dict[str, Any] = {
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "failed_company_ids": [],
        "new_cost": 0.0,
    }

    for company in source:
        company_id = int(company["company_id"])
        key = (company_id, "Single", run_number)
        if key in completed_keys:
            summary["skipped"] += 1
            print(f"SKIP company {company_id}: run {run_number} already exists")
            continue

        print(f"RUN  company {company_id}: Single run {run_number}")
        try:
            result = classify_company(
                company,
                run_number,
                model_name=model_name,
                classifier=classifier,
                prompt_path=prompt_path,
            )
            save_result(result, results_path)
        except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
            summary["failed"] += 1
            summary["failed_company_ids"].append(company_id)
            print(f"FAIL company {company_id}: {error}")
            continue

        completed_keys.add(key)
        summary["completed"] += 1
        summary["new_cost"] += result["cost"]
        print(
            f"DONE company {company_id}: {result['ai_label']} "
            f"(cost ${result['cost']:.8f})"
        )

    return summary
