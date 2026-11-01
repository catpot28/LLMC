"""Run Single and council classification for 49 companies over three V2 repetitions."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = "new_evaluation"


def experiment_runs(experiment_id: str) -> tuple[str, ...]:
    """Keep generated names within the experiment runner's 80-character limit."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,73}", experiment_id):
        raise ValueError("Experiment ID must be 1-74 letters, numbers, underscores, or hyphens")
    return tuple(f"{experiment_id}_run_{number}" for number in (1, 2, 3))


def run_module(module: str, *args: object) -> None:
    subprocess.run([sys.executable, "-B", "-m", module, *map(str, args)],
                   cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT,
                        help="Name for a new three-run evaluation (default: new_evaluation)")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--execute", action="store_true",
                        help="Make paid requests; without this flag, only preview")
    args = parser.parse_args()
    try:
        runs = experiment_runs(args.experiment_id)
    except ValueError as error:
        parser.error(str(error))

    print("Scope: 49 companies, Single plus three workers and judge, three V2 repetitions.",
          flush=True)
    print("A fresh evaluation requires 735 successful paid model calls (245 per repetition).",
          flush=True)
    run_module("classification.build_inputs")
    commands = [
        ("--experiment-id", name, "--architecture", "both", "--prompt-version", "v2",
         "--run-number", number, "--all", "--concurrency", args.concurrency)
        for number, name in enumerate(runs, start=1)
    ]
    # Check all three manifests before any paid request, including on resume.
    for command in commands:
        run_module("classification.run_experiment", *command)
    if not args.execute:
        print("Preview complete. Add --execute to run classification. No model requests made.")
        return

    for command in commands:
        run_module("classification.run_experiment", *command, "--execute")
    print("Classification complete. Next run:", flush=True)
    print(f"python run_analysis.py --experiment-id {args.experiment_id}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit("Classification stopped. Resolve the error and repeat the same command "
                         "to resume saved results.") from error
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Repeat the same command to resume saved results.")
