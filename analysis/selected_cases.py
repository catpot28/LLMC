"""Reproduce thesis Tables 7 and 8 from the selected-case ID mapping.

Run ``python3 -m analysis.build_results`` first, then
``python3 -m analysis.selected_cases``. The mapping records only case IDs and
company identities; all labels come from saved, aggregated classifications.
No model API calls are made.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .build_results import LABELS, OUTPUT_DIR
from .performance import load_company_results


ABBREVIATIONS = {"None": "N", "Low": "L", "Medium": "M", "High": "H"}
EXPECTED_CASES = tuple(f"C{number:02d}" for number in range(1, 9))
TABLE_7_FIELDS = ("Case", "Reference", "Single", "LLMC")
TABLE_8_FIELDS = ("Case", "Reference", "Single", "GPT", "Mistral worker", "Gemini", "Judge")


def load_mapping(path: Path, companies: dict[int, dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected-case mapping: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != len(EXPECTED_CASES):
        raise ValueError(f"Expected {len(EXPECTED_CASES)} case mappings in {path}")
    by_case: dict[str, dict[str, str]] = {}
    used_ids: set[int] = set()
    for record in records:
        case_id = record["case_id"]
        company_id = int(record["company_id"])
        if case_id in by_case or company_id in used_ids:
            raise ValueError(f"Duplicate case or company ID in {path}")
        if company_id not in companies:
            raise ValueError(f"Company ID {company_id} is absent from company_results.csv")
        company = companies[company_id]
        if record["company_name"] != company["company_name"]:
            raise ValueError(f"Company name mismatch for {case_id}")
        by_case[case_id] = company
        used_ids.add(company_id)
    if set(by_case) != set(EXPECTED_CASES):
        raise ValueError(f"Expected case IDs {EXPECTED_CASES} in {path}")
    return [(case_id, by_case[case_id]) for case_id in EXPECTED_CASES]


def abbreviated(row: dict[str, str], field: str) -> str:
    label = row[field]
    if label not in LABELS:
        raise ValueError(f"Invalid {field} label: {label}")
    return ABBREVIATIONS[label]


def build_tables(mapping: list[tuple[str, dict[str, str]]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    table_7 = []
    table_8 = []
    for case_id, company in mapping:
        reference = abbreviated(company, "reviewed_label")
        single = abbreviated(company, "single_aggregated_label")
        llmc = abbreviated(company, "llmc_aggregated_label")
        # Preserve C01-C07 from the earlier analysis; C08 adds Adobe from
        # the current three-V2 comparison. Labels always come from saved results.
        table_7.append({"Case": case_id, "Reference": reference, "Single": single, "LLMC": llmc})
        table_8.append({
            "Case": case_id,
            "Reference": reference,
            "Single": single,
            "GPT": abbreviated(company, "gpt_worker_aggregated_label"),
            "Mistral worker": abbreviated(company, "mistral_worker_aggregated_label"),
            "Gemini": abbreviated(company, "gemini_worker_aggregated_label"),
            "Judge": llmc,
        })
    return table_7, table_8


def save_table(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=OUTPUT_DIR / "case_mapping.csv")
    parser.add_argument("--company-results", type=Path, default=OUTPUT_DIR / "company_results.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    companies = load_company_results(args.company_results)
    mapping = load_mapping(args.mapping, companies)
    table_7, table_8 = build_tables(mapping)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows, fields in (
        ("table_7_selected_cases.csv", table_7, TABLE_7_FIELDS),
        ("table_8_worker_outputs.csv", table_8, TABLE_8_FIELDS),
    ):
        path = args.output_dir / filename
        save_table(path, rows, fields)
        print(f"Saved {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
