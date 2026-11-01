"""Rebuild the saved preprocessing data from the scraped pages.

The processing stages write working files to a temporary directory. Selected
handoff checkpoints, plus Phase 5's ``final`` and ``selected_pages`` outputs,
are kept in ``data/preprocessed``. Existing files are checked, never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from itertools import zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory

from .assemble_company_text import select_all
from .clean_deduplicate import process_phase3_all
from .filter_pages import DEFAULT_INPUT_DIR, DEFAULT_SELECTED_CSV, process_all
from .select_rank_pages import (
    process_budget_candidates_all,
    process_depth_all,
    process_language_all,
    process_locale_all,
    process_page_types_all,
)


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "preprocessed"
INTERMEDIATE_OUTPUTS = (
    ("intermediate/filter_pages/retained", "phase_2/retained"),
    ("intermediate/clean_deduplicate/cleaned", "phase_3/cleaned"),
    ("intermediate/select_rank_pages/budget_primary", "phase_4/budget_primary"),
    ("intermediate/select_rank_pages/budget_depth3_reserve", "phase_4/budget_depth3_reserve"),
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(directory: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(directory)): path
        for path in directory.rglob("*")
        if path.is_file()
    }


def _same_json(left: object, right: object) -> bool:
    """Ignore only negligible float rounding in generated metadata."""
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json(a, b) for a, b in zip(left, right)
        )
    return left == right


def _same_file(generated: Path, saved: Path) -> bool:
    if _file_hash(generated) == _file_hash(saved):
        return True
    if generated.suffix == ".json":
        with generated.open(encoding="utf-8") as left, saved.open(encoding="utf-8") as right:
            return _same_json(json.load(left), json.load(right))
    if generated.suffix == ".jsonl":
        missing = object()
        with generated.open(encoding="utf-8") as left, saved.open(encoding="utf-8") as right:
            for left_line, right_line in zip_longest(left, right, fillvalue=missing):
                if left_line is missing or right_line is missing:
                    return False
                if left_line != right_line and not _same_json(
                    json.loads(left_line), json.loads(right_line)
                ):
                    return False
        return True
    return False


def save_or_verify(source: Path, destination: Path) -> str:
    """Save a new result, or verify it matches the existing frozen result."""
    generated = _files(source)
    if not generated:
        raise ValueError(f"No generated files found in {source}")
    if destination.exists():
        existing = _files(destination)
        if generated.keys() != existing.keys() or any(
            not _same_file(generated[name], existing[name]) for name in generated
        ):
            raise ValueError(
                f"Generated {source.name} differs from saved data in {destination} "
                "beyond negligible float rounding; "
                "existing files were left unchanged."
            )
        return "verified"
    shutil.copytree(source, destination)
    return "saved"


def run_pipeline(
    scraped_dir: Path = DEFAULT_INPUT_DIR,
    selected_csv: Path = DEFAULT_SELECTED_CSV,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, str]:
    if not scraped_dir.is_dir():
        raise FileNotFoundError(f"Scraped pages missing: {scraped_dir}")
    if not selected_csv.is_file():
        raise FileNotFoundError(f"Selected-company file missing: {selected_csv}")

    with TemporaryDirectory(prefix="thesis-preprocessing-") as temporary:
        work = Path(temporary)
        phase2 = work / "phase_2"
        phase3 = work / "phase_3"
        phase4 = work / "phase_4"
        phase5 = work / "phase_5"

        print("1/8 Filtering scraped pages...", flush=True)
        process_all(scraped_dir, selected_csv, phase2)
        print("2/8 Cleaning and deduplicating pages...", flush=True)
        process_phase3_all(phase2 / "retained", phase3)
        print("3/8 Selecting a language for each company...", flush=True)
        process_language_all(phase3 / "cleaned", phase4)
        print("4/8 Consolidating locale variants...", flush=True)
        process_locale_all(phase4 / "language_selected", phase4)
        print("5/8 Separating primary and reserve pages...", flush=True)
        process_depth_all(phase4 / "locale_consolidated", phase4)
        print("6/8 Identifying page types...", flush=True)
        process_page_types_all(
            phase4 / "depth_primary", phase4 / "depth3_reserve", phase4
        )
        print("7/8 Counting tokens and preparing candidates...", flush=True)
        process_budget_candidates_all(
            phase4 / "typed_primary", phase4 / "typed_depth3_reserve", phase4
        )
        print("8/8 Assembling final company texts...", flush=True)
        select_all(
            phase4 / "budget_primary", phase4 / "budget_depth3_reserve", phase5
        )

        saved = {
            destination: work / source
            for destination, source in INTERMEDIATE_OUTPUTS
        }
        saved.update({
            "final": phase5 / "final",
            "selected_pages": phase5 / "selected_pages",
        })

        print("Checking generated files against saved results...", flush=True)
        # Check all existing snapshots before copying any new one.
        for name, source in saved.items():
            destination = output_dir / name
            if destination.exists():
                save_or_verify(source, destination)

        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            name: save_or_verify(source, output_dir / name)
            for name, source in saved.items()
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scraped-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    for name, status in run_pipeline(
        args.scraped_dir, args.selected_csv, args.output_dir
    ).items():
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()
