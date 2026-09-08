"""Compare preference matrices estimated from different numbers of seeds.

For actor ``i`` and two nested seed counts ``n`` and ``m``, the distance is

    D_i(n, m) = sum_s sum_q (P_i_n(s, q) - P_i_m(s, q)) ** 2.

Seed results are ordered by numeric seed and each larger sample contains every
seed in the smaller sample.  This makes changes attributable to the additional
seed results instead of to unrelated resampling differences.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import predict_preference_matrix as prediction


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Compare next-month preference matrices calculated from nested sets "
            "of optimizer seeds using the sum of squared element differences."
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
        default=demo_dir / "output" / "preference_matrix_convergence",
        help="Directory for matrices, distances, and the audit manifest.",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed filter; default uses every completed seed.",
    )
    seed_group.add_argument(
        "--all-seeds",
        action="store_true",
        help="Explicitly use every completed seed directory (the default).",
    )
    count_group = parser.add_mutually_exclusive_group()
    count_group.add_argument(
        "--seed-counts",
        default=None,
        help=(
            "Comma-separated nested sample sizes, for example 10,20,50,100. "
            "The full available count is always appended."
        ),
    )
    count_group.add_argument(
        "--step",
        type=int,
        default=10,
        help="Default count increment (default: 10); the full count is appended.",
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
            "model includes the fitted model's fixed state-score perturbation; "
            "base uses only the state-score table."
        ),
    )
    return parser.parse_args()


def resolve_seed_counts(raw: str | None, total: int, step: int = 10) -> list[int]:
    if total < 2:
        raise ValueError("At least two completed seeds are required for comparison.")
    if raw is None:
        if step <= 0:
            raise ValueError("--step must be a positive integer.")
        counts = list(range(step, total + 1, step))
    else:
        values = [item.strip() for item in raw.split(",") if item.strip()]
        if not values:
            raise ValueError("--seed-counts must contain at least one integer.")
        try:
            counts = [int(value) for value in values]
        except ValueError as exc:
            raise ValueError(
                "--seed-counts must be a comma-separated list of integers."
            ) from exc
        if any(count <= 0 for count in counts):
            raise ValueError("Every seed count must be positive.")
        if len(set(counts)) != len(counts):
            raise ValueError("--seed-counts contains duplicate values.")
        if any(count > total for count in counts):
            invalid = sorted(count for count in counts if count > total)
            raise ValueError(
                f"Seed counts exceed the {total} completed results: {invalid}."
            )

    counts = sorted(set(counts + [total]))
    if len(counts) < 2:
        raise ValueError(
            "At least two distinct seed counts are required; specify a smaller count."
        )
    return counts


def cumulative_preference_matrices(
    utilities: np.ndarray, seed_counts: Sequence[int]
) -> dict[int, np.ndarray]:
    """Return probability matrices for nested prefixes of the seed utilities."""
    values = np.asarray(utilities, dtype=float)
    if values.ndim != 3:
        raise ValueError(
            "utilities must have shape (seeds, actors, states); "
            f"received {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Predicted utilities contain non-finite values.")
    counts = [int(count) for count in seed_counts]
    if not counts or min(counts) <= 0 or max(counts) > len(values):
        raise ValueError("seed_counts must lie between 1 and the number of seeds.")

    strict_wins = values[:, :, :, None] > values[:, :, None, :]
    cumulative_wins = np.cumsum(strict_wins, axis=0, dtype=np.int64)
    return {
        count: cumulative_wins[count - 1].astype(float) / count for count in counts
    }


def squared_difference_sums(
    matrix_a: np.ndarray, matrix_b: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return per-actor and combined sums of squared corresponding differences."""
    first = np.asarray(matrix_a, dtype=float)
    second = np.asarray(matrix_b, dtype=float)
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError(
            "Both matrices must have the same (actors, states, states) shape."
        )
    differences = first - second
    actor_sums = np.sum(differences * differences, axis=(1, 2))
    return actor_sums, float(np.sum(actor_sums))


def distance_row(
    count_a: int,
    count_b: int,
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
) -> dict[str, Any]:
    actor_sums, combined = squared_difference_sums(matrix_a, matrix_b)
    if len(actor_sums) != 2:
        raise ValueError(
            f"DynamicGMCR distance tables require two actors, received {len(actor_sums)}."
        )
    state_count = matrix_a.shape[1]
    actor_element_count = state_count * state_count
    return {
        "seed_count_a": int(count_a),
        "seed_count_b": int(count_b),
        "added_seed_count": int(count_b - count_a),
        "CN_squared_difference_sum": float(actor_sums[0]),
        "US_squared_difference_sum": float(actor_sums[1]),
        "combined_squared_difference_sum": combined,
        "CN_mean_squared_difference": float(actor_sums[0] / actor_element_count),
        "US_mean_squared_difference": float(actor_sums[1] / actor_element_count),
        "combined_mean_squared_difference": float(
            combined / (len(actor_sums) * actor_element_count)
        ),
    }


def build_distance_tables(
    matrices: dict[int, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = sorted(matrices)
    all_rows = [
        distance_row(count_a, count_b, matrices[count_a], matrices[count_b])
        for count_a, count_b in combinations(counts, 2)
    ]
    consecutive_rows = [
        distance_row(count_a, count_b, matrices[count_a], matrices[count_b])
        for count_a, count_b in zip(counts, counts[1:])
    ]
    full_count = counts[-1]
    reference_rows = [
        distance_row(count, full_count, matrices[count], matrices[full_count])
        for count in counts
    ]
    return (
        pd.DataFrame(all_rows),
        pd.DataFrame(consecutive_rows),
        pd.DataFrame(reference_rows),
    )


def load_seed_utilities(
    parameter_files: list[tuple[int, Path]],
    input_dir: Path,
    utility_mode: str,
) -> tuple[np.ndarray, Any, int, list[Any]]:
    first_config = prediction.stability.load_config(parameter_files[0][1], input_dir)
    data = prediction.solver.load_model_data(first_config)
    prediction_month = prediction.stability.next_month(int(data.months[-1]))
    utilities = []
    configs = []
    for seed, parameter_path in parameter_files:
        _, forecast_utilities, config = prediction.predict_one_seed(
            seed, parameter_path, input_dir, data, utility_mode
        )
        utilities.append(forecast_utilities)
        configs.append(config)
    return np.stack(utilities, axis=0), data, prediction_month, configs


def write_outputs(
    output_dir: Path,
    state_ids: Sequence[int],
    prediction_month: int,
    utility_mode: str,
    parameter_files: list[tuple[int, Path]],
    skipped: list[dict[str, Any]],
    seed_counts: list[int],
    matrices: dict[int, np.ndarray],
    configs: list[Any],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = output_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    expected_matrix_names = {
        f"predicted_preference_matrix_{actor}_{prediction_month}_seeds_{count}.csv"
        for count in seed_counts
        for actor in prediction.solver.COUNTRIES
    }
    stale_matrix_files = sorted(
        path
        for path in matrix_dir.glob(
            f"predicted_preference_matrix_*_{prediction_month}_seeds_*.csv"
        )
        if path.name not in expected_matrix_names
    )
    for path in stale_matrix_files:
        path.unlink()
    all_distances, consecutive_distances, reference_distances = (
        build_distance_tables(matrices)
    )
    written: list[Path] = []

    all_path = output_dir / f"matrix_distances_all_pairs_{prediction_month}.csv"
    all_distances.to_csv(
        all_path, index=False, encoding="utf-8-sig", float_format="%.12f"
    )
    written.append(all_path)
    consecutive_path = (
        output_dir / f"matrix_distances_consecutive_seed_counts_{prediction_month}.csv"
    )
    consecutive_distances.to_csv(
        consecutive_path, index=False, encoding="utf-8-sig", float_format="%.12f"
    )
    written.append(consecutive_path)
    reference_path = (
        output_dir / f"matrix_distances_to_full_seed_matrix_{prediction_month}.csv"
    )
    reference_distances.to_csv(
        reference_path, index=False, encoding="utf-8-sig", float_format="%.12f"
    )
    written.append(reference_path)

    for count in seed_counts:
        for actor_index, actor in enumerate(prediction.solver.COUNTRIES):
            path = (
                matrix_dir
                / f"predicted_preference_matrix_{actor}_{prediction_month}_seeds_{count}.csv"
            )
            prediction.probability_matrix_frame(
                state_ids, matrices[count][actor_index]
            ).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12f")
            written.append(path)

    included_seeds = [seed for seed, _ in parameter_files]
    count_membership = {
        str(count): included_seeds[:count] for count in seed_counts
    }
    config_signatures = sorted(
        {
            (
                int(config.population_size),
                int(config.generations),
                int(config.local_search_steps),
                int(config.perturbation_seed),
                float(config.delta),
            )
            for config in configs
        }
    )
    manifest = {
        "method": (
            "D_i(n,m) = sum_s sum_q "
            "(P_i_n(s,q) - P_i_m(s,q))^2"
        ),
        "matrix_scope": (
            "All directional matrix elements are included. Diagonal elements are "
            "zero in every matrix and therefore add zero to the distance."
        ),
        "combined_distance": "CN squared-difference sum + US squared-difference sum",
        "nested_sampling": True,
        "seed_order": "ascending numeric optimizer seed",
        "prediction_month": int(prediction_month),
        "utility_mode": utility_mode,
        "completed_seed_count": len(parameter_files),
        "seed_counts": seed_counts,
        "distance_table_row_counts": {
            "all_pairs": len(all_distances),
            "consecutive_counts": len(consecutive_distances),
            "to_full_matrix": len(reference_distances),
        },
        "state_count": len(state_ids),
        "actor_count": len(prediction.solver.COUNTRIES),
        "matrix_element_count_per_actor": len(state_ids) ** 2,
        "included_seeds_by_count": count_membership,
        "selected_runs": [
            {"seed": seed, "parameter_file": str(path.resolve())}
            for seed, path in parameter_files
        ],
        "skipped_or_noted_seed_directories": skipped,
        "removed_stale_matrix_files": [path.name for path in stale_matrix_files],
        "config_signatures": [
            {
                "population_size": signature[0],
                "generations": signature[1],
                "local_search_steps": signature[2],
                "perturbation_seed": signature[3],
                "delta": signature[4],
            }
            for signature in config_signatures
        ],
        "generated_files": [str(path.relative_to(output_dir)) for path in written],
    }
    manifest_path = output_dir / f"matrix_distance_manifest_{prediction_month}.json"
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
        requested_seeds = prediction.parse_seed_filter(args.seeds)
        parameter_files, skipped = prediction.discover_parameter_files(
            results_dir, requested_seeds
        )
        seed_counts = resolve_seed_counts(
            args.seed_counts, len(parameter_files), int(args.step)
        )
        utilities, data, inferred_prediction_month, configs = load_seed_utilities(
            parameter_files, input_dir, args.utility_mode
        )
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
        matrices = cumulative_preference_matrices(utilities, seed_counts)
        written = write_outputs(
            output_dir=output_dir,
            state_ids=data.state_ids,
            prediction_month=prediction_month,
            utility_mode=args.utility_mode,
            parameter_files=parameter_files,
            skipped=skipped,
            seed_counts=seed_counts,
            matrices=matrices,
            configs=configs,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(
        f"Compared {len(seed_counts)} nested seed counts using "
        f"{len(parameter_files)} completed seeds and {len(data.state_ids)} states."
    )
    print("Seed counts: " + ", ".join(str(count) for count in seed_counts))
    for note in skipped:
        print(f"Note: seed {note['seed']}: {note['reason']}")
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
