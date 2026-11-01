"""Preview or explicitly run Single and LLMC classifications via OpenRouter."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from config.models import LLMC_JUDGE_MODEL, LLMC_WORKER_MODELS, MODELS, SINGLE_MODEL
from config.settings import load_openrouter_config

from .council_classifier import run_all_councils
from .single_classifier import COMPANIES_PATH, classify_all_companies, iterate_local_companies


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "data" / "experiments"
EXPERIMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(
    experiment_id: str,
    run_number: int,
    prompt_version: str,
    worker_prompt: Path,
    judge_prompt: Path,
    company_count: int,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "run_number": run_number,
        "prompt_version": prompt_version,
        "inputs": {
            "path": str(COMPANIES_PATH.relative_to(ROOT)),
            "sha256": _sha256(COMPANIES_PATH),
            "companies": company_count,
        },
        "prompts": {
            "workers_and_single": {
                "path": str(worker_prompt.relative_to(ROOT)),
                "sha256": _sha256(worker_prompt),
            },
            "judge": {
                "path": str(judge_prompt.relative_to(ROOT)),
                "sha256": _sha256(judge_prompt),
            },
        },
        "models": {
            "single": MODELS[SINGLE_MODEL],
            "workers": [MODELS[name] for name in LLMC_WORKER_MODELS],
            "judge": MODELS[LLMC_JUDGE_MODEL],
        },
    }


def _check_or_save_manifest(path: Path, expected: dict[str, Any], save: bool) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError(f"Experiment settings differ from existing {path}")
    elif save:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True, help="Name for a separate run under data/experiments/")
    parser.add_argument("--architecture", choices=("single", "council", "both"), required=True)
    parser.add_argument("--prompt-version", choices=("v1", "v2"), required=True)
    parser.add_argument("--run-number", type=int, choices=(1, 2, 3), required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--company-id", type=int)
    scope.add_argument("--all", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Make paid model API requests")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=1,
                        help="Independent companies processed concurrently by LLMC (default: 1)")
    args = parser.parse_args()

    if not EXPERIMENT_ID_RE.fullmatch(args.experiment_id):
        parser.error("--experiment-id must use only letters, numbers, underscores, or hyphens")
    worker_prompt = ROOT / "prompts" / f"single_{args.prompt_version}.txt"
    judge_prompt = ROOT / "prompts" / f"judge_{args.prompt_version}.txt"
    for path in (COMPANIES_PATH, worker_prompt, judge_prompt):
        if not path.is_file():
            raise SystemExit(f"Required input missing: {path}")

    all_companies = list(iterate_local_companies())
    companies = (
        all_companies if args.all else
        [row for row in all_companies if int(row["company_id"]) == args.company_id]
    )
    if not companies:
        raise SystemExit(f"Company ID {args.company_id} is not in {COMPANIES_PATH}")

    experiment_dir = EXPERIMENTS_DIR / args.experiment_id
    manifest = _manifest(
        args.experiment_id, args.run_number, args.prompt_version,
        worker_prompt, judge_prompt, len(all_companies),
    )
    _check_or_save_manifest(experiment_dir / "manifest.json", manifest, save=False)
    print(f"Experiment: {experiment_dir}")
    print(f"Architecture: {args.architecture}; prompt: {args.prompt_version}; run: {args.run_number}")
    print(f"Selected companies: {len(companies)} of {len(all_companies)}")
    if not args.execute:
        print("Preview only: no files written and no API requests made.")
        print("Add --execute to make model requests and save results.")
        return

    load_openrouter_config()  # Check the local key before writing results or making requests.
    _check_or_save_manifest(experiment_dir / "manifest.json", manifest, save=True)
    if args.architecture in {"single", "both"}:
        summary = classify_all_companies(
            args.run_number,
            experiment_dir / "single" / "results.jsonl",
            worker_prompt,
            companies=companies,
        )
        print("Single summary:", json.dumps(summary, sort_keys=True))
        if summary["failed"]:
            raise SystemExit("Single classification had failures; see the summary above")
    if args.architecture in {"council", "both"}:
        summary = run_all_councils(
            args.run_number,
            experiment_dir / "council" / "worker_outputs.jsonl",
            experiment_dir / "council" / "results.jsonl",
            worker_prompt,
            judge_prompt,
            companies=companies,
            concurrency=args.concurrency,
        )
        print("LLMC summary:", json.dumps(summary, sort_keys=True))
        if summary["failed"]:
            raise SystemExit("LLMC classification had failures; see the summary above")


if __name__ == "__main__":
    main()
