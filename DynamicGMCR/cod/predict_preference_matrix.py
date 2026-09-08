"""Estimate next-month pairwise preference probabilities across optimizer seeds.

For actor ``i``, row state ``s`` and column state ``q``, the reported matrix is

    P(s >_i q) = count_seed(u_i(s) > u_i(q)) / M.

Only completed ``seed_*`` directories containing fitted parameter JSON files
contribute to ``M``.  The newest parameter file is selected if a seed directory
contains more than one completed run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import check_stability_states as stability
import solve_core_preference_model as solver


SEED_DIRECTORY_PATTERN = re.compile(r"seed_(\d+)$")


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Calculate next-month pairwise preference-probability matrices from "
            "independent optimizer-seed results."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=demo_dir / "output",
        help="Directory containing seed_<number> result directories.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=demo_dir / "input",
        help="Directory containing the DynamicGMCR model inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=demo_dir / "output" / "preference_prediction",
        help="Directory for probability matrices and audit outputs.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated optimizer seeds; default uses every seed_* directory.",
    )
    parser.add_argument(
        "--prediction-month",
        type=int,
        default=None,
        help="Forecast month label; default is the month after the final observation.",
    )
    parser.add_argument(
        "--utility-mode",
        choices=("model", "base"),
        default="model",
        help=(
            "model uses the fitted model's fixed state-score perturbation; base uses "
            "only the state-score table."
        ),
    )
    return parser.parse_args()


def parse_seed_filter(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--seeds must contain at least one integer seed.")
    try:
        seeds = {int(value) for value in values}
    except ValueError as exc:
        raise ValueError("--seeds must be a comma-separated list of integers.") from exc
    if len(seeds) != len(values):
        raise ValueError("--seeds contains duplicate values.")
    return seeds


def discover_parameter_files(
    results_dir: Path,
    requested_seeds: set[int] | None = None,
) -> tuple[list[tuple[int, Path]], list[dict[str, Any]]]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    selected: list[tuple[int, Path]] = []
    skipped: list[dict[str, Any]] = []
    found_seed_directories: set[int] = set()
    for seed_dir in sorted(results_dir.iterdir(), key=lambda path: path.name):
        match = SEED_DIRECTORY_PATTERN.fullmatch(seed_dir.name)
        if not seed_dir.is_dir() or match is None:
            continue
        seed = int(match.group(1))
        found_seed_directories.add(seed)
        if requested_seeds is not None and seed not in requested_seeds:
            continue

        candidates = list(seed_dir.rglob("parameters_*.json"))
        if not candidates:
            skipped.append(
                {"seed": seed, "seed_directory": str(seed_dir.resolve()), "reason": "no parameter file"}
            )
            continue
        candidates.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)))
        chosen = candidates[-1]
        selected.append((seed, chosen))
        if len(candidates) > 1:
            skipped.append(
                {
                    "seed": seed,
                    "seed_directory": str(seed_dir.resolve()),
                    "reason": "multiple parameter files; newest selected",
                    "selected": str(chosen.resolve()),
                    "candidate_count": len(candidates),
                }
            )

    if requested_seeds is not None:
        absent = sorted(requested_seeds - found_seed_directories)
        if absent:
            raise FileNotFoundError(
                "Requested seed directories were not found: "
                + ", ".join(str(seed) for seed in absent)
            )
    if not selected:
        raise ValueError(f"No completed seed results found under {results_dir}.")
    selected.sort(key=lambda item: item[0])
    return selected, skipped


def compute_pairwise_statistics(
    utilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strict-win counts, exact-tie counts and strict-win probabilities."""
    values = np.asarray(utilities, dtype=float)
    if values.ndim != 3:
        raise ValueError(
            "utilities must have shape (seeds, actors, states); "
            f"received {values.shape}."
        )
    if values.shape[0] == 0:
        raise ValueError("At least one seed utility array is required.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Predicted utilities contain non-finite values.")

    left = values[:, :, :, None]
    right = values[:, :, None, :]
    preferred_counts = np.count_nonzero(left > right, axis=0)
    tie_counts = np.count_nonzero(left == right, axis=0)
    probabilities = preferred_counts.astype(float) / values.shape[0]
    return preferred_counts, tie_counts, probabilities


def probability_matrix_frame(
    state_ids: Sequence[int], probabilities: np.ndarray
) -> pd.DataFrame:
    state_values = [int(value) for value in state_ids]
    frame = pd.DataFrame(
        np.asarray(probabilities, dtype=float),
        columns=[f"state_{state_id}" for state_id in state_values],
    )
    frame.insert(0, "state_id", state_values)
    return frame


def validate_seed_in_summary(seed: int, parameter_path: Path) -> None:
    summary_path = stability.find_sibling_json(parameter_path, "summary")
    if summary_path is None:
        return
    payload = stability.read_json(summary_path)
    configured_seed = payload.get("config", {}).get("seed")
    if configured_seed is not None and int(configured_seed) != seed:
        raise ValueError(
            f"Seed directory seed_{seed} contains a result configured with seed "
            f"{configured_seed}: {parameter_path}"
        )


def predict_one_seed(
    seed: int,
    parameter_path: Path,
    input_dir: Path,
    data: solver.ModelData,
    utility_mode: str,
) -> tuple[np.ndarray, np.ndarray, solver.SolverConfig]:
    validate_seed_in_summary(seed, parameter_path)
    config = stability.load_config(parameter_path, input_dir)
    params = stability.load_parameters(parameter_path, config)

    placeholder = np.zeros_like(data.impacts_scaled[:1])
    extended_impacts = np.concatenate([data.impacts_scaled, placeholder], axis=0)
    forecast_preferences = solver.compute_h(
        params,
        extended_impacts,
        nonlinear_saturation=config.nonlinear_saturation,
    )[-1]
    if utility_mode == "model":
        forecast_perturbation = solver.generate_score_perturbations(
            len(data.months) + 1, len(data.state_ids), config
        )[-1]
        forecast_utilities = solver.perturbed_utilities(
            data.omega, forecast_perturbation, forecast_preferences
        )
    else:
        forecast_utilities = solver.base_utilities(data.omega, forecast_preferences)
    return forecast_preferences, forecast_utilities, config


def write_outputs(
    output_dir: Path,
    prediction_month: int,
    utility_mode: str,
    state_ids: np.ndarray,
    parameter_files: list[tuple[int, Path]],
    skipped: list[dict[str, Any]],
    preferences: np.ndarray,
    utilities: np.ndarray,
    configs: list[solver.SolverConfig],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preferred_counts, tie_counts, probabilities = compute_pairwise_statistics(utilities)
    seed_count = len(parameter_files)
    written: list[Path] = []

    for actor_index, actor in enumerate(solver.COUNTRIES):
        path = output_dir / f"predicted_preference_matrix_{actor}_{prediction_month}.csv"
        probability_matrix_frame(state_ids, probabilities[actor_index]).to_csv(
            path, index=False, encoding="utf-8-sig", float_format="%.10f"
        )
        written.append(path)

    preference_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    for run_index, (seed, parameter_path) in enumerate(parameter_files):
        for actor_index, actor in enumerate(solver.COUNTRIES):
            for driver_index, driver in enumerate(solver.DRIVERS_BY_ACTOR[actor_index]):
                preference_rows.append(
                    {
                        "seed": seed,
                        "parameter_file": str(parameter_path.resolve()),
                        "prediction_month": prediction_month,
                        "actor": actor,
                        "driver_index": driver_index + 1,
                        "driver": driver,
                        "forecast_preference": preferences[run_index, actor_index, driver_index],
                    }
                )
            for state_index, state_id in enumerate(state_ids):
                utility_rows.append(
                    {
                        "seed": seed,
                        "parameter_file": str(parameter_path.resolve()),
                        "prediction_month": prediction_month,
                        "actor": actor,
                        "state_id": int(state_id),
                        "forecast_utility": utilities[run_index, actor_index, state_index],
                    }
                )

    preference_path = output_dir / f"predicted_preferences_by_seed_{prediction_month}.csv"
    pd.DataFrame(preference_rows).to_csv(
        preference_path, index=False, encoding="utf-8-sig", float_format="%.10f"
    )
    written.append(preference_path)
    utility_path = output_dir / f"predicted_utilities_by_seed_{prediction_month}.csv"
    pd.DataFrame(utility_rows).to_csv(
        utility_path, index=False, encoding="utf-8-sig", float_format="%.10f"
    )
    written.append(utility_path)

    pairwise_rows: list[dict[str, Any]] = []
    for actor_index, actor in enumerate(solver.COUNTRIES):
        for row_index, state_s in enumerate(state_ids):
            for column_index, state_q in enumerate(state_ids):
                pairwise_rows.append(
                    {
                        "prediction_month": prediction_month,
                        "actor": actor,
                        "state_s": int(state_s),
                        "state_q": int(state_q),
                        "preferred_seed_count": int(
                            preferred_counts[actor_index, row_index, column_index]
                        ),
                        "tie_seed_count": int(tie_counts[actor_index, row_index, column_index]),
                        "seed_count": seed_count,
                        "preference_probability": probabilities[
                            actor_index, row_index, column_index
                        ],
                    }
                )
    pairwise_path = output_dir / f"pairwise_preference_probabilities_{prediction_month}.csv"
    pd.DataFrame(pairwise_rows).to_csv(
        pairwise_path, index=False, encoding="utf-8-sig", float_format="%.10f"
    )
    written.append(pairwise_path)

    selected_runs = []
    for index, ((seed, parameter_path), config) in enumerate(zip(parameter_files, configs)):
        selected_runs.append(
            {
                "seed": seed,
                "parameter_file": str(parameter_path.resolve()),
                "forecast_preferences": {
                    actor: [float(value) for value in preferences[index, actor_index]]
                    for actor_index, actor in enumerate(solver.COUNTRIES)
                },
                "score_perturbation": {
                    "applied": utility_mode == "model",
                    "delta": float(config.delta),
                    "rho": float(config.delta_persistence),
                    "seed": int(config.perturbation_seed),
                },
            }
        )

    manifest = {
        "method": "P(s >_i q) = Count_seed(u_i(s) > u_i(q)) / M",
        "comparison": "strict_greater_than",
        "matrix_orientation": "row state s is compared with column state q",
        "prediction_month": prediction_month,
        "historical_end_month": int(prediction_month) - 1
        if prediction_month % 100 > 1
        else (prediction_month // 100 - 1) * 100 + 12,
        "forecast_assumption": (
            "One-step preference forecast uses observed impacts through the final "
            "historical month; no prediction-month impact is imputed."
        ),
        "utility_mode": utility_mode,
        "utility_definition": (
            "u=(omega+eta)h, including each result's fixed score perturbation"
            if utility_mode == "model"
            else "u=omega*h, excluding the score perturbation"
        ),
        "seed_count_M": seed_count,
        "state_count": len(state_ids),
        "actors": list(solver.COUNTRIES),
        "selected_runs": selected_runs,
        "skipped_or_noted_seed_directories": skipped,
        "generated_files": [path.name for path in written],
    }
    manifest_path = output_dir / f"preference_prediction_manifest_{prediction_month}.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    written.append(manifest_path)
    return written


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    try:
        requested_seeds = parse_seed_filter(args.seeds)
        parameter_files, skipped = discover_parameter_files(results_dir, requested_seeds)

        first_config = stability.load_config(parameter_files[0][1], input_dir)
        data = solver.load_model_data(first_config)
        inferred_prediction_month = stability.next_month(int(data.months[-1]))
        prediction_month = (
            inferred_prediction_month
            if args.prediction_month is None
            else int(args.prediction_month)
        )
        if prediction_month != inferred_prediction_month:
            raise ValueError(
                "This model supports one-step prediction only: expected prediction month "
                f"{inferred_prediction_month}, received {prediction_month}."
            )

        preference_results = []
        utility_results = []
        configs = []
        for seed, parameter_path in parameter_files:
            forecast_preferences, forecast_utilities, config = predict_one_seed(
                seed, parameter_path, input_dir, data, args.utility_mode
            )
            preference_results.append(forecast_preferences)
            utility_results.append(forecast_utilities)
            configs.append(config)

        preferences = np.stack(preference_results, axis=0)
        utilities = np.stack(utility_results, axis=0)
        written = write_outputs(
            output_dir=output_dir,
            prediction_month=prediction_month,
            utility_mode=args.utility_mode,
            state_ids=data.state_ids,
            parameter_files=parameter_files,
            skipped=skipped,
            preferences=preferences,
            utilities=utilities,
            configs=configs,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(
        f"Calculated {len(solver.COUNTRIES)} preference matrices for "
        f"{len(parameter_files)} seeds and {len(data.state_ids)} states."
    )
    for note in skipped:
        print(f"Note: seed {note['seed']}: {note['reason']}")
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
