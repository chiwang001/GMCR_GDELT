#!/usr/bin/env python3
"""Run ``solve_core_preference_model.py`` independently for many seeds.

Each seed gets a private solver output directory under its experiment folder.
This avoids races when workers run concurrently and makes it impossible for a
result from one seed to be attributed to another seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DIR = SCRIPT_DIR.parent
SOLVER = SCRIPT_DIR / "solve_core_preference_model.py"
CHECKER = SCRIPT_DIR / "check_stability_states.py"
DEFAULT_SEEDS = tuple(range(20250333, 20250334))
SUMMARY_FIELDS = (
    "seed",
    "status",
    "returncode",
    "elapsed_seconds",
    "run_dir",
    "loss",
    "fit_objective",
    "log_transition_objective",
    "seq_stability_penalty",
    "forward_shortfall_penalty",
    "regularization_a",
    "regularization_mu",
    "regularization_alpha",
    "transition_probability_sum",
    "parameter_file",
    "summary_file",
    "postprocess_status",
    "error",
)


def parse_int_list(value: str, *, name: str = "seeds") -> tuple[int, ...]:
    """Parse a comma-separated integer list, rejecting empty/duplicate values."""
    values: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{name} must be a comma-separated list of integers."
            ) from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} must not contain duplicates.")
    return tuple(values)


def make_unique_dir(parent: Path, name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / name
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DynamicGMCR preference estimation across independent random seeds."
    )
    parser.add_argument(
        "--seeds",
        type=lambda value: parse_int_list(value, name="seeds"),
        default=DEFAULT_SEEDS,
        help="Comma-separated optimizer seeds (default: 20250722,...,20250731).",
    )
    parser.add_argument(
        "--preset",
        choices=("quick", "standard", "full", "two_hour", "deep"),
        default="two_hour",
    )
    parser.add_argument("--input-dir", type=Path, default=DEMO_DIR / "input")
    parser.add_argument(
        "--solver-output-dir",
        type=Path,
        default=DEMO_DIR / "output",
        help=(
            "Solver output root. Each seed uses a private "
            "<root>/seed_<seed>/<day-hour-minute> directory."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEMO_DIR / "output" / "multiseed",
        help="Root directory for multi-seed logs and aggregate summaries.",
    )
    parser.add_argument("--experiment-label", default="")
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--force", action="store_true", help="Rerun seeds with complete results."
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only rebuild summaries from existing seed directories.",
    )
    parser.add_argument("--perturbation-seed", type=int, default=20260701)
    parser.add_argument("--de-early-stop-patience", type=int, default=0)
    parser.add_argument("--local-search-early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--postprocess-stability",
        action="store_true",
        help="Run the historical/202002 stability checker after each successful solve.",
    )
    parser.add_argument("--prediction-month", type=int, default=202002)
    return parser.parse_args(argv)


def child_command(
    args: argparse.Namespace, seed: int, solver_output_dir: Path
) -> list[str]:
    return [
        sys.executable,
        str(SOLVER),
        "--input-dir",
        str(Path(args.input_dir).resolve()),
        "--output-dir",
        str(solver_output_dir.resolve()),
        "--preset",
        str(args.preset),
        "--seed",
        str(seed),
        "--perturbation-seed",
        str(args.perturbation_seed),
        "--de-early-stop-patience",
        str(max(0, args.de_early_stop_patience)),
        "--local-search-early-stop-patience",
        str(max(0, args.local_search_early_stop_patience)),
    ]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def write_text_file(path: Path, content: str) -> None:
    """Write a run log even if its seed directory was not present anymore."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or "", encoding="utf-8")


def solver_run_dirs(solver_output_dir: Path) -> list[Path]:
    if not solver_output_dir.exists():
        return []
    return sorted(
        (path for path in solver_output_dir.iterdir() if path.is_dir()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def output_files(run_dir: Path) -> tuple[Path | None, Path | None]:
    parameter_files = sorted(run_dir.glob("parameters_*.json"))
    summary_files = sorted(run_dir.glob("summary_*.json"))
    if len(parameter_files) != 1 or len(summary_files) != 1:
        return None, None
    parameter_stamp = parameter_files[0].stem.removeprefix("parameters_")
    summary_stamp = summary_files[0].stem.removeprefix("summary_")
    if parameter_stamp != summary_stamp:
        return None, None
    return (
        parameter_files[0],
        summary_files[0],
    )


def result_is_complete(run_dir: Path, seed: int) -> bool:
    parameter_file, summary_file = output_files(run_dir)
    if parameter_file is None or summary_file is None:
        return False
    try:
        summary = read_json(summary_file)
        config = summary.get("config", {})
        objective = summary.get("objective", {})
        vector = read_json(parameter_file).get("vector", [])
        loss = float(objective.get("loss"))
        return (
            int(config.get("seed", -1)) == int(seed)
            and len(vector) == 42
            and all(math.isfinite(float(value)) for value in vector)
            and math.isfinite(loss)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
        return False


def find_complete_run(solver_output_dir: Path, seed: int) -> Path | None:
    for run_dir in reversed(solver_run_dirs(solver_output_dir)):
        if result_is_complete(run_dir, seed):
            return run_dir
    return None


def find_recorded_run(seed_dir: Path, seed: int) -> Path | None:
    """Find this experiment's result without claiming unrelated global runs."""
    status_path = seed_dir / "run_status.json"
    if status_path.exists():
        try:
            raw_path = read_json(status_path).get("run_dir", "")
            if raw_path:
                run_dir = Path(raw_path)
                if result_is_complete(run_dir, seed):
                    return run_dir
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    # Backward compatibility with results written by the first runner version.
    return find_complete_run(seed_dir / "solver_output", seed)


def _objective_fields(summary: dict[str, Any]) -> dict[str, Any]:
    objective = summary.get("objective", {})
    return {key: objective.get(key, "") for key in SUMMARY_FIELDS if key in objective}


def summarize_seed(
    seed_dir: Path,
    seed: int,
    *,
    run_dir: Path | None = None,
    discover_run: bool = True,
    status: str | None = None,
    returncode: int = 0,
    elapsed_seconds: float = 0.0,
    postprocess_status: str = "not_requested",
    error: str = "",
) -> dict[str, Any]:
    if run_dir is None and discover_run:
        run_dir = find_recorded_run(seed_dir, seed)
    parameter_file: Path | None = None
    summary_file: Path | None = None
    summary: dict[str, Any] = {}
    if run_dir is not None:
        parameter_file, summary_file = output_files(run_dir)
        if summary_file is not None:
            try:
                summary = read_json(summary_file)
            except (OSError, ValueError, json.JSONDecodeError):
                summary = {}
    row: dict[str, Any] = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "seed": seed,
            "status": status or ("completed" if run_dir else "failed"),
            "returncode": returncode,
            "elapsed_seconds": elapsed_seconds,
            "run_dir": str(run_dir) if run_dir else "",
            "parameter_file": str(parameter_file) if parameter_file else "",
            "summary_file": str(summary_file) if summary_file else "",
            "postprocess_status": postprocess_status,
            "error": error,
        }
    )
    row.update(_objective_fields(summary))
    return row


def run_one(args: argparse.Namespace, experiment_dir: Path, seed: int) -> dict[str, Any]:
    seed_dir = experiment_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    # Never share the solver's timestamped output directory between seeds.
    # ``solver-output-dir`` remains the user-selected root; the seed folder is
    # added below it for a stable one-to-one mapping.
    solver_output_dir = (
        Path(args.solver_output_dir).resolve() / f"seed_{seed}"
    )
    solver_output_dir.mkdir(parents=True, exist_ok=True)

    existing = find_recorded_run(seed_dir, seed)
    if existing is None:
        # Recover a completed result if the process was interrupted before it
        # could write run_status.json.
        existing = find_complete_run(solver_output_dir, seed)
    if existing is not None and not args.force:
        row = summarize_seed(seed_dir, seed, status="skipped_existing")
        (seed_dir / "run_status.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return row

    before = {path.resolve() for path in solver_run_dirs(solver_output_dir)}
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Starting seed {seed}", flush=True)
    started = time.perf_counter()
    env = os.environ.copy()
    if args.max_workers > 1:
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[name] = "1"
    completed = subprocess.run(
        child_command(args, seed, solver_output_dir),
        cwd=str(DEMO_DIR),
        text=True,
        capture_output=True,
        env=env,
    )
    elapsed = time.perf_counter() - started
    write_text_file(seed_dir / "run.stdout.log", completed.stdout)
    write_text_file(seed_dir / "run.stderr.log", completed.stderr)

    new_dirs = [
        path
        for path in solver_run_dirs(solver_output_dir)
        if path.resolve() not in before
    ]
    complete_new_dirs = [path for path in new_dirs if result_is_complete(path, seed)]
    run_dir = complete_new_dirs[-1] if len(complete_new_dirs) == 1 else None
    status = "completed" if completed.returncode == 0 and run_dir is not None else "failed"
    postprocess_status = "not_requested"
    error = "" if status == "completed" else (
        f"solver return code {completed.returncode}"
        if completed.returncode
        else f"expected one complete new run directory, found {len(complete_new_dirs)}"
    )
    if status == "completed" and args.postprocess_stability and run_dir is not None:
        parameter_file, _ = output_files(run_dir)
        assert parameter_file is not None
        post = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--parameters",
                str(parameter_file),
                "--input-dir",
                str(Path(args.input_dir).resolve()),
                "--output-dir",
                str(run_dir.resolve()),
                "--prediction-month",
                str(args.prediction_month),
            ],
            cwd=str(DEMO_DIR),
            text=True,
            capture_output=True,
            env=env,
        )
        write_text_file(seed_dir / "stability.stdout.log", post.stdout)
        write_text_file(seed_dir / "stability.stderr.log", post.stderr)
        postprocess_status = "completed" if post.returncode == 0 else "failed"

    row = summarize_seed(
        seed_dir,
        seed,
        run_dir=run_dir,
        discover_run=False,
        status=status,
        returncode=int(completed.returncode),
        elapsed_seconds=float(elapsed),
        postprocess_status=postprocess_status,
        error=error,
    )
    (seed_dir / "run_status.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] Seed {seed}: {status} "
        f"({elapsed:.1f} s)",
        flush=True,
    )
    return row


def run_one_safely(
    args: argparse.Namespace, experiment_dir: Path, seed: int
) -> dict[str, Any]:
    try:
        return run_one(args, experiment_dir, seed)
    except Exception as exc:
        seed_dir = experiment_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        row = summarize_seed(
            seed_dir,
            seed,
            discover_run=False,
            status="failed",
            returncode=-1,
            error=f"{type(exc).__name__}: {exc}",
        )
        write_text_file(seed_dir / "run.stderr.log", traceback.format_exc())
        (seed_dir / "run_status.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Seed {seed}: failed ({row['error']})", flush=True)
        return row


def write_aggregate(
    experiment_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    with (experiment_dir / "run_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_dir": str(experiment_dir.resolve()),
        "seeds": [int(seed) for seed in args.seeds],
        "preset": args.preset,
        "input_dir": str(Path(args.input_dir).resolve()),
        "solver_output_dir": str(Path(args.solver_output_dir).resolve()),
        "perturbation_seed": int(args.perturbation_seed),
        "postprocess_stability": bool(args.postprocess_stability),
        "rows": rows,
    }
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_experiment_dir(args: argparse.Namespace) -> Path:
    if args.experiment_dir is not None:
        path = args.experiment_dir.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    label = args.experiment_label.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = (args.output_root / label).resolve()
    if args.experiment_label.strip():
        path.mkdir(parents=True, exist_ok=True)
        return path
    return make_unique_dir(args.output_root.resolve(), label)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")
    experiment_dir = resolve_experiment_dir(args)
    if args.summarize_only:
        rows = [summarize_seed(experiment_dir / f"seed_{seed}", seed) for seed in args.seeds]
    elif args.max_workers == 1:
        rows = [run_one_safely(args, experiment_dir, seed) for seed in args.seeds]
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=min(args.max_workers, len(args.seeds))) as pool:
            futures = {
                pool.submit(run_one_safely, args, experiment_dir, seed): seed
                for seed in args.seeds
            }
            for future in as_completed(futures):
                rows.append(future.result())
    write_aggregate(experiment_dir, rows, args)
    failed = [row for row in rows if row["status"] == "failed"]
    print(f"Experiment directory: {experiment_dir}")
    print(f"Completed/skipped: {len(rows) - len(failed)}; failed: {len(failed)}")
    print(f"Summary: {experiment_dir / 'run_summary.csv'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
