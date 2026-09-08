"""Estimate next-month preference matrices with independent seed sampling.

The existing ``compare_seed_count_preference_matrices.py`` uses nested prefixes
of the numerically ordered optimizer seeds.  This script instead draws an
independent subset of completed seeds for every replicate and every requested
sample size.  Sampling is without replacement within a replicate and is
independent across replicates and sample sizes.

For a sampled subset ``I`` of size ``n`` the matrix entry is

    P_i(s > q) = mean(1[u_i,k(s) > u_i,k(q)] for k in I).

The full completed-seed matrix is used only as an internal reference for
distance and uncertainty diagnostics; it is not treated as ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import predict_preference_matrix as prediction


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Calculate preference-probability matrices from independent random "
            "subsets of optimizer seeds."
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
        default=demo_dir / "output" / "preference_matrix_independent_sampling",
        help="Directory for matrices, replicate metrics, and the audit manifest.",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated optimizer seed filter (for smoke tests).",
    )
    seed_group.add_argument(
        "--all-seeds",
        action="store_true",
        help=(
            "Explicitly use every completed seed_<number> directory. This is "
            "also the default when --seeds is omitted."
        ),
    )
    count_group = parser.add_mutually_exclusive_group()
    count_group.add_argument(
        "--seed-counts",
        default=None,
        help=(
            "Comma-separated subset sizes, for example 10,20,50. "
            "The full completed-seed count is always appended."
        ),
    )
    count_group.add_argument(
        "--step",
        type=int,
        default=10,
        help="Default subset-size increment (default: 10).",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=100,
        help="Independent subsets drawn per sample size (default: 100).",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=20260831,
        help="Random seed controlling the independent subset draws.",
    )
    parser.add_argument(
        "--prediction-month",
        type=int,
        default=None,
        help="Forecast month; default is the month after the final observation.",
    )
    parser.add_argument(
        "--utility-mode",
        choices=("model", "base"),
        default="model",
        help=(
            "model includes each result's fixed score perturbation; "
            "base uses only the state-score table."
        ),
    )
    parser.add_argument(
        "--save-replicate-matrices",
        action="store_true",
        help="Also save one CN and one US matrix for every replicate.",
    )
    return parser.parse_args()


def resolve_seed_counts(raw: str | None, total: int, step: int = 10) -> list[int]:
    if total < 2:
        raise ValueError("At least two completed seeds are required.")
    if raw is None:
        if step <= 0:
            raise ValueError("--step must be a positive integer.")
        counts = list(range(step, total + 1, step))
        if not counts:
            # Keep the default usable for very small seed collections.
            counts = [total - 1]
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


def squared_difference_sums(
    matrix_a: np.ndarray, matrix_b: np.ndarray
) -> tuple[np.ndarray, float]:
    first = np.asarray(matrix_a, dtype=float)
    second = np.asarray(matrix_b, dtype=float)
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError(
            "Both matrices must have the same (actors, states, states) shape."
        )
    differences = first - second
    actor_sums = np.sum(differences * differences, axis=(1, 2))
    return actor_sums, float(np.sum(actor_sums))


def sample_matrix_replicates(
    utilities: np.ndarray,
    seed_counts: list[int],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return independent subset matrices and their selected row indices."""
    values = np.asarray(utilities, dtype=float)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("utilities must be finite with shape (seeds, actors, states).")
    if replicates <= 0:
        raise ValueError("--replicates must be a positive integer.")

    matrices: dict[int, np.ndarray] = {}
    memberships: dict[int, np.ndarray] = {}
    for count in seed_counts:
        sampled_indices = np.stack(
            [rng.choice(len(values), size=count, replace=False) for _ in range(replicates)],
            axis=0,
        )
        # Process one replicate at a time so the largest comparison tensor is
        # only (sampled_seed, actor, state, state), rather than retaining all
        # replicate comparisons simultaneously.
        matrices[count] = np.stack(
            [
                (
                    values[indices, :, :, None]
                    > values[indices, :, None, :]
                ).mean(axis=0, dtype=float)
                for indices in sampled_indices
            ],
            axis=0,
        )
        memberships[count] = sampled_indices
    return matrices, memberships


def config_signature(config: Any) -> dict[str, Any]:
    """Keep model settings relevant to comparability, excluding paths and seed."""
    ignored = {"input_dir", "output_dir", "seed"}
    return {
        field.name: getattr(config, field.name)
        for field in fields(config)
        if field.name not in ignored
    }


def build_replicate_table(
    matrices: dict[int, np.ndarray],
    memberships: dict[int, np.ndarray],
    reference: np.ndarray,
    included_seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for count in sorted(matrices):
        current = matrices[count]
        actor_distances = []
        combined_distances = []
        for replicate_index, matrix in enumerate(current):
            actor_sums, combined = squared_difference_sums(matrix, reference)
            actor_distances.append(actor_sums)
            combined_distances.append(combined)
            rows.append(
                {
                    "seed_count": int(count),
                    "replicate": int(replicate_index + 1),
                    "CN_squared_distance_to_full": float(actor_sums[0]),
                    "US_squared_distance_to_full": float(actor_sums[1]),
                    "combined_squared_distance_to_full": float(combined),
                    "sampled_seeds": ",".join(
                        str(included_seeds[index])
                        for index in memberships[count][replicate_index]
                    ),
                }
            )
        actor_distances_array = np.asarray(actor_distances, dtype=float)
        combined_array = np.asarray(combined_distances, dtype=float)
        element_sd = np.std(current, axis=0, ddof=1 if len(current) > 1 else 0)
        summary_rows.append(
            {
                "seed_count": int(count),
                "replicates": int(len(current)),
                "CN_mean_squared_distance_to_full": float(actor_distances_array[:, 0].mean()),
                "CN_sd_squared_distance_to_full": float(actor_distances_array[:, 0].std(ddof=1) if len(current) > 1 else 0.0),
                "US_mean_squared_distance_to_full": float(actor_distances_array[:, 1].mean()),
                "US_sd_squared_distance_to_full": float(actor_distances_array[:, 1].std(ddof=1) if len(current) > 1 else 0.0),
                "combined_mean_squared_distance_to_full": float(combined_array.mean()),
                "combined_sd_squared_distance_to_full": float(combined_array.std(ddof=1) if len(current) > 1 else 0.0),
                "combined_p95_squared_distance_to_full": float(np.quantile(combined_array, 0.95)),
                "CN_mean_element_sd": float(element_sd[0].mean()),
                "US_mean_element_sd": float(element_sd[1].mean()),
                "CN_max_element_sd": float(element_sd[0].max()),
                "US_max_element_sd": float(element_sd[1].max()),
                "sample_mean_matrix_matches_full_reference": bool(
                    np.allclose(current.mean(axis=0), reference, atol=1.0e-12)
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def write_matrix(
    path: Path, state_ids: np.ndarray, matrix: np.ndarray
) -> None:
    prediction.probability_matrix_frame(state_ids, matrix).to_csv(
        path, index=False, encoding="utf-8-sig", float_format="%.12f"
    )


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
        utilities, data, inferred_month, configs = _load_seed_utilities(
            parameter_files, input_dir, args.utility_mode
        )
        signatures = {
            json.dumps(config_signature(config), sort_keys=True, default=str)
            for config in configs
        }
        if len(signatures) > 1:
            raise ValueError(
                "Selected seed results use inconsistent model configurations; "
                "independent seed sampling requires one common configuration."
            )
        prediction_month = inferred_month if args.prediction_month is None else int(args.prediction_month)
        if prediction_month != inferred_month:
            raise ValueError(
                "This model supports one-step prediction only: expected prediction month "
                f"{inferred_month}, received {prediction_month}."
            )
        seed_counts = resolve_seed_counts(args.seed_counts, len(parameter_files), int(args.step))
        rng = np.random.default_rng(int(args.sampling_seed))
        _, _, reference = prediction.compute_pairwise_statistics(utilities)
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix_dir = output_dir / "matrices"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        replicate_tables: list[pd.DataFrame] = []
        summary_tables: list[pd.DataFrame] = []
        included_seeds = [seed for seed, _ in parameter_files]
        for count in seed_counts:
            matrices, memberships = sample_matrix_replicates(
                utilities, [count], int(args.replicates), rng
            )
            replicate_table, summary_table = build_replicate_table(
                matrices, memberships, reference, included_seeds
            )
            replicate_tables.append(replicate_table)
            summary_tables.append(summary_table)
            sampled_matrices = matrices[count]
            mean_matrix = sampled_matrices.mean(axis=0)
            sd_matrix = np.std(
                sampled_matrices, axis=0,
                ddof=1 if sampled_matrices.shape[0] > 1 else 0,
            )
            for actor_index, actor in enumerate(prediction.solver.COUNTRIES):
                mean_path = matrix_dir / f"mean_preference_matrix_{actor}_{prediction_month}_seeds_{count}.csv"
                sd_path = matrix_dir / f"sd_preference_matrix_{actor}_{prediction_month}_seeds_{count}.csv"
                write_matrix(mean_path, data.state_ids, mean_matrix[actor_index])
                write_matrix(sd_path, data.state_ids, sd_matrix[actor_index])
                written.extend([mean_path, sd_path])
                if args.save_replicate_matrices:
                    for replicate_index, matrix in enumerate(sampled_matrices, start=1):
                        replicate_path = matrix_dir / (
                            f"preference_matrix_{actor}_{prediction_month}_seeds_{count}_"
                            f"replicate_{replicate_index:04d}.csv"
                        )
                        write_matrix(replicate_path, data.state_ids, matrix[actor_index])
                        written.append(replicate_path)

        replicate_table = pd.concat(replicate_tables, ignore_index=True)
        summary_table = pd.concat(summary_tables, ignore_index=True)
        replicate_path = output_dir / f"independent_sampling_replicates_{prediction_month}.csv"
        summary_path = output_dir / f"independent_sampling_summary_{prediction_month}.csv"
        replicate_table.to_csv(replicate_path, index=False, encoding="utf-8-sig")
        summary_table.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.12f")
        written.extend([replicate_path, summary_path])
        manifest = {
            "method": "independent without-replacement subset sampling per replicate and sample size",
            "matrix_definition": "P_i(s > q) = mean_k 1[u_i,k(s) > u_i,k(q)]",
            "prediction_month": int(prediction_month),
            "utility_mode": args.utility_mode,
            "seed_selection": "all_completed_seed_directories" if args.seeds is None else "requested_seed_filter",
            "all_seeds_requested": bool(args.seeds is None or args.all_seeds),
            "sampling_seed": int(args.sampling_seed),
            "replicates_per_count": int(args.replicates),
            "completed_seed_count": len(parameter_files),
            "seed_counts": seed_counts,
            "state_count": len(data.state_ids),
            "actor_count": len(prediction.solver.COUNTRIES),
            "full_reference_definition": "all completed seeds, used for internal distance diagnostics only",
            "selected_runs": [
                {"seed": seed, "parameter_file": str(path.resolve())}
                for seed, path in parameter_files
            ],
            "skipped_or_noted_seed_directories": skipped,
            "config_signatures": [config_signature(config) for config in configs],
            "generated_files": [str(path.relative_to(output_dir)) for path in written],
        }
        manifest_path = output_dir / f"independent_sampling_manifest_{prediction_month}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(manifest_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(
        f"Sampled {len(seed_counts)} seed counts with {args.replicates} independent "
        f"replicates, using {len(parameter_files)} completed seeds."
    )
    print("Seed counts: " + ", ".join(str(count) for count in seed_counts))
    for path in written:
        print(f"Wrote {path}")


def _load_seed_utilities(
    parameter_files: list[tuple[int, Path]],
    input_dir: Path,
    utility_mode: str,
) -> tuple[np.ndarray, Any, int, list[Any]]:
    """Load seed utilities while keeping the comparison script's model path."""
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


if __name__ == "__main__":
    main()
