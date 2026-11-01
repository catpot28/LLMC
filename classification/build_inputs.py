"""Build the 49-company classifier input snapshot from preprocessing outputs.

The original run created pilot inputs with ``pilot.py``, remaining-company
inputs with ``remaining_run.py``, then combined them with
``combine_experiments.py``. This script rebuilds the same combined input file
directly from the final preprocessed text and reviewer-1 company metadata.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = ROOT / "data" / "preprocessed" / "final"
COMPANY_METADATA = ROOT / "data" / "human_labels" / "reviewer_1_companies.jsonl"
OUTPUT_PATH = ROOT / "data" / "classification_inputs" / "companies.jsonl"


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _read_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Company metadata missing: {path}")
    companies: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            company = json.loads(line)
            key = _normalize_name(company["company_name"])
            if not key or key in companies:
                raise ValueError(f"Missing or duplicate company name at {path}:{line_number}")
            companies[key] = company
    return companies


def build_rows(final_dir: Path = FINAL_DIR, metadata_path: Path = COMPANY_METADATA) -> tuple[list[dict[str, Any]], list[str]]:
    """Join each final text to its company metadata; skip zero-content sites."""
    companies = _read_metadata(metadata_path)
    paths = sorted(final_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No final company text files found in {final_dir}")

    rows: list[dict[str, Any]] = []
    excluded: list[str] = []
    seen_ids: set[int] = set()
    for path in paths:
        final = json.loads(path.read_text(encoding="utf-8"))
        slug = final["slug"]
        if slug != path.stem:
            raise ValueError(f"Slug and filename disagree: {path}")
        key = _normalize_name(str(final.get("company") or slug))
        if key not in companies:
            raise ValueError(f"No metadata for {path.name}")
        company = companies[key]
        company_id = int(company["company_id"])
        if company_id in seen_ids:
            raise ValueError(f"Duplicate final text for company ID {company_id}")
        seen_ids.add(company_id)

        text = final["text"]
        tokens = int(final["content_token_count"])
        if not isinstance(text, str) or (tokens == 0) != (not text.strip()):
            raise ValueError(f"Inconsistent text and token count in {path}")
        if tokens == 0:
            excluded.append(company["company_name"])
            continue
        rows.append({
            "company_id": company_id,
            "company_name": company["company_name"],
            "company_domain": company.get("company_domain"),
            "assembled_text_slug": slug,
            "assembled_text_content_tokens": tokens,
            "website_text": text,
        })

    rows.sort(key=lambda row: row["company_id"])
    return rows, excluded


def save_or_verify(rows: list[dict[str, Any]], output_path: Path = OUTPUT_PATH) -> str:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Generated classification inputs differ from {output_path}; saved file was left unchanged")
        return "verified"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return "saved"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=FINAL_DIR)
    parser.add_argument("--metadata", type=Path, default=COMPANY_METADATA)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    rows, excluded = build_rows(args.final_dir, args.metadata)
    status = save_or_verify(rows, args.output)
    print(f"{args.output}: {status} ({len(rows)} companies)")
    print(f"Excluded zero-content companies: {', '.join(excluded) if excluded else 'none'}")


if __name__ == "__main__":
    main()
