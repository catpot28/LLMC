"""Analyse a completed three-run classification evaluation without API requests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from run_classification import DEFAULT_EXPERIMENT, ROOT, experiment_runs


TABLES = {
    "company_results.csv", "performance_summary.csv", "confusion_matrices.csv",
    "stability_summary.csv", "label_flip_distribution.csv", "cost_summary.csv",
    "table_7_selected_cases.csv", "table_8_worker_outputs.csv", "mcnemar_test.csv",
}
FIGURES = {
    f"{name}.{extension}"
    for name in ("confusion_matrices", "confusion_matrix_single", "confusion_matrix_llmc",
                 "confidence_dispersion", "confidence_score_distribution", "label_flip_rate",
                 "mcnemar_test", "accuracy_by_involvement_level", "cost_vs_accuracy")
    for extension in ("png", "pdf")
}


def run_module(module: str, *args: object, cache_dir: Path) -> None:
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(cache_dir)
    subprocess.run([sys.executable, "-B", "-m", module, *map(str, args)],
                   cwd=ROOT, check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    cohort = parser.add_mutually_exclusive_group()
    cohort.add_argument("--experiment-id", help="Same ID used by run_classification.py")
    cohort.add_argument("--saved", action="store_true",
                        help="Verify and analyse the original saved thesis runs instead")
    args = parser.parse_args()
    try:
        runs = (("run_2_v2", "run_3_v2", "run_4_v2") if args.saved else
                experiment_runs(args.experiment_id or DEFAULT_EXPERIMENT))
    except ValueError as error:
        parser.error(str(error))
    name = "saved_thesis" if args.saved else (args.experiment_id or DEFAULT_EXPERIMENT)
    destination = ROOT / "outputs" / name
    print("Analysing:", ", ".join(runs), flush=True)
    print("No model requests will be made. Original thesis tables and figures are preserved.",
          flush=True)

    with TemporaryDirectory(prefix=".analysis-rebuild-", dir=ROOT) as temporary:
        work = Path(temporary)
        tables, figures = work / "tables", work / "figures"

        def run(module: str, *arguments: object) -> None:
            run_module(module, *arguments, cache_dir=work / "matplotlib-cache")

        run("analysis.build_results", "--runs", *runs, "--output-dir", tables)
        companies = tables / "company_results.csv"
        for module in ("performance", "stability"):
            run(f"analysis.{module}", "--runs", *runs,
                "--company-results", companies, "--output-dir", tables)
        run("analysis.selected_cases", "--company-results", companies, "--output-dir", tables)
        run("analysis.costs", "--runs", *runs, "--company-results", companies,
            "--performance-summary", tables / "performance_summary.csv", "--output-dir", tables)
        run("analysis.figure_generators.mcnemar", "--company-results", companies,
            "--output-dir", tables, "--figure-dir", figures)
        for module, source in (("plot_confusion_matrices", "confusion_matrices.csv"),
                               ("plot_accuracy_by_level", "performance_summary.csv")):
            run(f"analysis.figure_generators.{module}", "--source", tables / source,
                "--output-dir", figures)
        run("analysis.figure_generators.plot_stability", "--runs", *runs,
            "--summary", tables / "stability_summary.csv",
            "--company-results", companies, "--output-dir", figures)
        run("analysis.figure_generators.plot_cost_vs_accuracy", "--costs", tables / "cost_summary.csv",
            "--performance", tables / "performance_summary.csv", "--output-dir", figures)

        if {p.name for p in tables.iterdir()} != TABLES:
            raise ValueError("Expected all nine numerical tables")
        if {p.name for p in figures.iterdir()} != FIGURES:
            raise ValueError("Expected nine PNG/PDF figure pairs")
        if any(p.stat().st_size == 0 for directory in (tables, figures) for p in directory.iterdir()):
            raise ValueError("An empty output was generated")
        if args.saved:
            for filename in sorted(TABLES):
                if (tables / filename).read_bytes() != (ROOT / "data/analysis" / filename).read_bytes():
                    raise ValueError(f"Saved thesis results do not match the rebuild: {filename}")
            print("PASS: all nine tables match the submitted thesis results.", flush=True)

        # Publish only after every analysis and figure has completed successfully.
        for source in (tables, figures):
            target = destination / source.name
            target.mkdir(parents=True, exist_ok=True)
            for path in source.iterdir():
                shutil.copy2(path, target / path.name)
        (destination / "cohort.json").write_text(json.dumps({
            "runs": list(runs), "prompt_version": "v2", "companies": 49,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "verified_against_saved_thesis": args.saved,
            "case_mapping": "data/analysis/case_mapping.csv",
            "separate_gemini_comparison": "data/experiments/model_comparison_gemini_v2",
        }, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: nine tables and 18 figures saved to {destination}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Analysis stopped: {error}. Check that all three classification "
                         "repetitions are complete. No fallback to another cohort was used.") from error
    except KeyboardInterrupt:
        raise SystemExit("Analysis interrupted.")
