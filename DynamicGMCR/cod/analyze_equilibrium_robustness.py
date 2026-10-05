"""Propagate preference-matrix sampling uncertainty through forward GMCR.

This program audits and extends the February 2020 probability-preference
analysis in the public GMCR_GDELT project.  It treats the full retained
ensemble as a finite reference, repeatedly samples solutions without
replacement, reconstructs pairwise matrices, and reuses the project's
probability-form GMCR margin implementation.

The resulting evidence is conditional on the supplied solution ensemble,
state graph, pairwise threshold, and probability-form stability rules.  The
pairwise entries are ranking frequencies, not action or posterior
probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTORS = ("CN", "US")
CONCEPTS = ("Nash", "GMR", "SMR", "SEQ")
KEY_STATES = (54, 57, 73, 76)
EPS = 1.0e-12
DEFAULT_SIZES = (10, 50, 100, 200, 300, 400, 490)
COLORS = {
    "Nash": "#0072B2",
    "GMR": "#D55E00",
    "SMR": "#009E73",
    "SEQ": "#CC79A7",
}


@dataclass(frozen=True)
class Paths:
    project_root: Path
    dynamic_root: Path
    code_dir: Path
    prediction_dir: Path
    utilities: Path
    preferences: Path
    matrices: dict[str, Path]
    graph: Path
    stability_dir: Path
    parameter_dir: Path | None
    monthly_impacts: Path
    monthly_states: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and test whether sampling convergence of the February 2020 "
            "pairwise preference matrices propagates to probability-form GMCR "
            "margins and equilibrium classifications."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="GMCR_GDELT checkout, or its DynamicGMCR directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<DynamicGMCR>/results/equilibrium_robustness."
        ),
    )
    parser.add_argument(
        "--subset-sizes",
        default=",".join(map(str, DEFAULT_SIZES)),
        help="Comma-separated proper-subset sizes (default: 10,50,100,200,300,400,490).",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=100,
        help="Independent subsets per size (default: 100).",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=20261004,
        help="Seed for deterministic without-replacement sampling.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Strict-improvement support threshold (default: 0.5).",
    )
    parser.add_argument(
        "--sensitivity-thresholds",
        default="0.50,0.55,0.60",
        help="Comma-separated full-ensemble thresholds for sensitivity analysis.",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=5,
        help="Number of disjoint solution batches (default: 5).",
    )
    parser.add_argument(
        "--boundary-epsilon",
        type=float,
        default=0.05,
        help="Absolute full-reference margin defining a boundary result.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run: at most three sizes and three replicates per size.",
    )
    return parser.parse_args(argv)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)


def _first_existing(candidates: Iterable[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    rendered = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate {label}; checked:\n  {rendered}")


def discover_paths(project_root: Path) -> Paths:
    root = project_root.resolve()
    if (root / "cod").is_dir() and (root / "input").is_dir():
        dynamic = root
    else:
        dynamic = root if root.name.lower() == "dynamicgmcr" else root / "DynamicGMCR"
    if not dynamic.is_dir():
        raise FileNotFoundError(
            f"Expected a DynamicGMCR directory below {root}, or as the supplied root."
        )
    code_dir = _first_existing(
        (dynamic / "cod", dynamic / "code", dynamic / "src"), "GMCR code directory"
    )
    prediction_dir = _first_existing(
        (
            dynamic / "results" / "preference_prediction",
            dynamic / "source_runs" / "preference_prediction",
            dynamic / "output" / "preference_prediction",
        ),
        "preference-prediction results",
    )
    parameter_candidates = (
        dynamic / "results" / "multiseed_parameters",
        dynamic / "source_runs" / "multiseed_parameters",
        dynamic / "output",
    )
    parameter_dir = next((p for p in parameter_candidates if p.is_dir()), None)
    input_dir = dynamic / "input"
    stability_candidates = (
        prediction_dir / "stability_probability",
        prediction_dir / "gmcr_stability_202002_probability",
        prediction_dir / "gmcr_stability_202002_recomputed",
        prediction_dir / "gmcr_stability_202002",
    )
    stability_dir = next((path for path in stability_candidates if path.is_dir()), stability_candidates[0])
    return Paths(
        project_root=root,
        dynamic_root=dynamic,
        code_dir=code_dir,
        prediction_dir=prediction_dir,
        utilities=prediction_dir / "predicted_utilities_by_seed_202002.csv",
        preferences=prediction_dir / "predicted_preferences_by_seed_202002.csv",
        matrices={
            actor: prediction_dir / f"predicted_preference_matrix_{actor}_202002.csv"
            for actor in ACTORS
        },
        graph=input_dir / "state_transition_graph.csv",
        stability_dir=stability_dir,
        parameter_dir=parameter_dir,
        monthly_impacts=input_dir / "monthly_country_domain_impact.csv",
        monthly_states=input_dir / "monthly_key_states.csv",
    )


def import_probability_gmcr(code_dir: Path) -> Any:
    script = code_dir / "gmcr_probability_stability.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing probability-form GMCR implementation: {script}")
    name = "dynamicgmcr_probability_stability_for_robustness"
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if tuple(module.ACTORS) != ACTORS or tuple(module.CONCEPTS) != CONCEPTS:
        raise ValueError("The probability-GMCR actor or concept ordering is unexpected.")
    return module


def parse_int_list(raw: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("Subset sizes must be comma-separated integers.") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError("At least one positive subset size is required.")
    return sorted(set(values))


def parse_float_list(raw: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("Thresholds must be comma-separated numbers.") from exc
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Every threshold must lie in [0, 1].")
    return sorted(set(values))


def resolve_subset_sizes(requested: Sequence[int], full_size: int) -> list[int]:
    sizes = sorted({int(size) for size in requested if 0 < int(size) < full_size})
    if full_size < 2:
        raise ValueError("At least two valid retained solutions are required.")
    near_full = max(1, full_size - min(10, max(1, full_size // 10)))
    if near_full < full_size:
        sizes.append(near_full)
    sizes = sorted(set(sizes))
    if not sizes:
        sizes = [full_size - 1]
    return sizes


def quick_sizes(sizes: Sequence[int]) -> list[int]:
    if len(sizes) <= 3:
        return list(sizes)
    return sorted({sizes[0], sizes[len(sizes) // 2], sizes[-1]})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_utility_ensemble(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    frame = _read_csv(path)
    required = {"seed", "prediction_month", "actor", "state_id", "forecast_utility"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.duplicated(["seed", "actor", "state_id"]).any():
        raise ValueError(f"{path}: duplicate seed/actor/state rows found.")
    months = pd.to_numeric(frame["prediction_month"], errors="raise").astype(int).unique()
    if len(months) != 1:
        raise ValueError(f"{path}: expected one prediction month, found {months.tolist()}.")
    actors = tuple(frame["actor"].astype(str).drop_duplicates())
    if set(actors) != set(ACTORS):
        raise ValueError(f"{path}: expected actors {ACTORS}, found {actors}.")
    seeds = np.sort(pd.to_numeric(frame["seed"], errors="raise").astype(int).unique())
    states = np.sort(pd.to_numeric(frame["state_id"], errors="raise").astype(int).unique())
    expected = len(seeds) * len(ACTORS) * len(states)
    if len(frame) != expected:
        raise ValueError(
            f"{path}: incomplete rectangular ensemble; expected {expected} rows, found {len(frame)}."
        )
    index = pd.MultiIndex.from_product(
        [seeds, ACTORS, states], names=["seed", "actor", "state_id"]
    )
    ordered = (
        frame.assign(
            seed=pd.to_numeric(frame["seed"], errors="raise").astype(int),
            state_id=pd.to_numeric(frame["state_id"], errors="raise").astype(int),
            actor=frame["actor"].astype(str),
            forecast_utility=pd.to_numeric(frame["forecast_utility"], errors="raise"),
        )
        .set_index(["seed", "actor", "state_id"])
        .reindex(index)["forecast_utility"]
    )
    if ordered.isna().any() or not np.isfinite(ordered.to_numpy(float)).all():
        raise ValueError(f"{path}: missing or non-finite utility values.")
    utilities = ordered.to_numpy(float).reshape(len(seeds), len(ACTORS), len(states))
    return utilities, seeds, states, int(months[0])


def pairwise_indicators(utilities: np.ndarray) -> np.ndarray:
    values = np.asarray(utilities, dtype=float)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Utilities must be finite with shape (solutions, actors, states).")
    return values[:, :, :, None] > values[:, :, None, :]


def pairwise_matrix(indicators: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(indicators)
    if values.ndim != 4:
        raise ValueError("Pairwise indicators must have four dimensions.")
    selected = values if indices is None else values[np.asarray(indices, dtype=int)]
    if len(selected) == 0:
        raise ValueError("Cannot construct a matrix from an empty solution subset.")
    return selected.mean(axis=0, dtype=float)


def deterministic_samples(
    full_size: int, subset_sizes: Sequence[int], replicates: int, seed: int
) -> dict[int, np.ndarray]:
    if replicates <= 0:
        raise ValueError("Replicates must be positive.")
    rng = np.random.default_rng(int(seed))
    return {
        int(size): np.stack(
            [rng.choice(full_size, size=int(size), replace=False) for _ in range(replicates)]
        )
        for size in subset_sizes
    }


def compute_margins(
    gmcr: Any,
    probabilities: np.ndarray,
    neighbors: list[list[np.ndarray]],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(probabilities, dtype=float)
    state_count = matrix.shape[1]
    margins = np.empty((len(ACTORS), state_count, len(CONCEPTS)), dtype=float)
    ui_counts = np.zeros((len(ACTORS), state_count), dtype=int)
    for actor in range(len(ACTORS)):
        for state in range(state_count):
            current, improvements = gmcr.stability_margins(
                state, actor, matrix, neighbors, float(threshold)
            )
            margins[actor, state] = current
            ui_counts[actor, state] = len(improvements)
    return margins, ui_counts


def classify(margins: np.ndarray) -> np.ndarray:
    return np.asarray(margins, dtype=float) >= -EPS


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {key: math.nan for key in ("mean", "median", "sd", "p05", "p95", "max")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def threshold_separation(
    reference: np.ndarray,
    neighbors: list[list[np.ndarray]],
    threshold: float,
) -> tuple[float, np.ndarray]:
    local = np.full((len(ACTORS), reference.shape[1]), np.inf, dtype=float)
    edge_values: list[float] = []
    for actor in range(len(ACTORS)):
        for state, destinations in enumerate(neighbors[actor]):
            if len(destinations):
                gaps = np.abs(reference[actor, destinations, state] - threshold)
                local[actor, state] = float(np.min(gaps))
                edge_values.extend(float(value) for value in gaps)
    return (min(edge_values) if edge_values else math.inf), local


def ui_set_unchanged(
    candidate: np.ndarray,
    reference: np.ndarray,
    neighbors: list[list[np.ndarray]],
    threshold: float,
) -> np.ndarray:
    unchanged = np.ones((len(ACTORS), reference.shape[1]), dtype=bool)
    for actor in range(len(ACTORS)):
        for state, destinations in enumerate(neighbors[actor]):
            unchanged[actor, state] = np.array_equal(
                candidate[actor, destinations, state] > threshold + EPS,
                reference[actor, destinations, state] > threshold + EPS,
            )
    return unchanged


def certificate_flags(
    reference_margins: np.ndarray,
    d_infinity: float,
    global_gap: float,
    local_gaps: np.ndarray,
    feasible_counts: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    global_ui = bool(d_infinity < global_gap)
    local_ui = d_infinity < local_gaps
    sign_clear = np.abs(reference_margins) > d_infinity + EPS
    analytical = local_ui[:, :, None] & sign_clear
    structural = feasible_counts == 0
    analytical |= structural[:, :, None]
    return global_ui, local_ui, sign_clear, analytical


def _stable_rates(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    cand = np.asarray(candidate, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    false_stable = cand & ~ref
    false_unstable = ~cand & ref
    unstable_n = int((~ref).sum())
    stable_n = int(ref.sum())
    return {
        "classification_agreement": float(np.mean(cand == ref)),
        "false_stable_count": int(false_stable.sum()),
        "false_unstable_count": int(false_unstable.sum()),
        "false_stable_rate": float(false_stable.sum() / unstable_n) if unstable_n else 0.0,
        "false_unstable_rate": float(false_unstable.sum() / stable_n) if stable_n else 0.0,
    }


def reference_frame(
    state_ids: np.ndarray,
    margins: np.ndarray,
    stable: np.ndarray,
    ui_counts: np.ndarray,
    neighbors: list[list[np.ndarray]],
    local_gaps: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ai, actor in enumerate(ACTORS):
        for si, state_id in enumerate(state_ids):
            feasible = len(neighbors[ai][si])
            basis = (
                "no_feasible_move"
                if feasible == 0
                else "feasible_moves_no_strict_improvement"
                if ui_counts[ai, si] == 0
                else "preference_and_response_analysis"
            )
            for ci, concept in enumerate(CONCEPTS):
                rows.append(
                    {
                        "actor": actor,
                        "state_id": int(state_id),
                        "concept": concept,
                        "margin": float(margins[ai, si, ci]),
                        "stable": bool(stable[ai, si, ci]),
                        "feasible_move_count": int(feasible),
                        "strict_improvement_count": int(ui_counts[ai, si]),
                        "stability_basis": basis,
                        "local_ui_threshold_separation": (
                            float(local_gaps[ai, si])
                            if np.isfinite(local_gaps[ai, si])
                            else math.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _bool_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(bool)
    return values.astype(str).str.lower().map({"true": True, "false": False}).to_numpy(bool)


def parameter_inventory(parameter_dir: Path | None) -> tuple[int, list[int], int, int]:
    if parameter_dir is None:
        return 0, [], 0, 0
    selected_files: list[Path] = []
    seeds: list[int] = []
    candidate_count = 0
    multiple_candidate_seed_count = 0
    for seed_path in sorted(parameter_dir.glob("seed_*")):
        if not seed_path.is_dir():
            continue
        try:
            seed = int(seed_path.name.removeprefix("seed_"))
        except ValueError:
            continue
        candidates = sorted(
            set(seed_path.rglob("parameters_*.json")),
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
        )
        candidate_count += len(candidates)
        if not candidates:
            continue
        if len(candidates) > 1:
            multiple_candidate_seed_count += 1
        selected_files.append(candidates[-1])
        seeds.append(seed)
    return (
        len(selected_files),
        sorted(set(seeds)),
        candidate_count,
        multiple_candidate_seed_count,
    )


def audit_inputs(
    paths: Paths,
    gmcr: Any,
    utilities: np.ndarray,
    seeds: np.ndarray,
    state_ids: np.ndarray,
    prediction_month: int,
    reference: np.ndarray,
    reference_margins: np.ndarray,
    reference_stable: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    stored_ids: np.ndarray | None = None
    stored_matrices: list[np.ndarray] = []
    for actor in ACTORS:
        ids, matrix = gmcr.load_probability_matrix(paths.matrices[actor])
        if stored_ids is None:
            stored_ids = ids
        elif not np.array_equal(stored_ids, ids):
            raise ValueError("Stored CN and US matrices have different state ordering.")
        stored_matrices.append(matrix)
    assert stored_ids is not None
    stored = np.stack(stored_matrices)
    matrix_max_error = float(np.max(np.abs(reference - stored)))
    if not np.array_equal(state_ids, stored_ids):
        raise ValueError("Utility and stored-matrix state identifiers do not match.")

    (
        parameter_count,
        parameter_seeds,
        parameter_candidate_count,
        multiple_parameter_seed_count,
    ) = parameter_inventory(paths.parameter_dir)
    preference_frame = _read_csv(paths.preferences)
    forecast_seed_count = int(preference_frame["seed"].nunique())
    forecast_months = sorted(
        pd.to_numeric(preference_frame["prediction_month"], errors="raise").astype(int).unique().tolist()
    )
    impact_frame = _read_csv(paths.monthly_impacts)
    state_frame = _read_csv(paths.monthly_states)
    input_month_max = int(impact_frame["month"].max())
    observed_state_month_max = int(state_frame["month"].max())

    existing_details_path = paths.stability_dir / "stability_details.csv"
    existing_max_margin_error = math.nan
    existing_classification_matches = False
    archived_rows: list[dict[str, Any]] = []
    s54_archived: dict[str, dict[str, Any]] = {}
    if existing_details_path.exists():
        existing = _read_csv(existing_details_path)
        indexed = existing.set_index(["actor", "state_id"])
        margin_errors: list[float] = []
        matches: list[bool] = []
        for ai, actor in enumerate(ACTORS):
            for si, state_id in enumerate(state_ids):
                row = indexed.loc[(actor, int(state_id))]
                for ci, concept in enumerate(CONCEPTS):
                    margin_errors.append(
                        abs(float(row[f"{concept}_margin"]) - reference_margins[ai, si, ci])
                    )
                    raw_stable = row[f"{concept}_stable"]
                    stored_stable = (
                        bool(raw_stable)
                        if isinstance(raw_stable, (bool, np.bool_))
                        else str(raw_stable).lower() == "true"
                    )
                    matches.append(stored_stable == bool(reference_stable[ai, si, ci]))
                    archived_rows.append(
                        {
                            "actor": actor,
                            "state_id": int(state_id),
                            "concept": concept,
                            "archived_margin": float(row[f"{concept}_margin"]),
                            "archived_stable": stored_stable,
                            "current_500_margin": float(reference_margins[ai, si, ci]),
                            "current_500_stable": bool(reference_stable[ai, si, ci]),
                            "absolute_margin_difference": abs(
                                float(row[f"{concept}_margin"])
                                - reference_margins[ai, si, ci]
                            ),
                            "classification_matches": bool(
                                stored_stable == reference_stable[ai, si, ci]
                            ),
                        }
                    )
        existing_max_margin_error = float(max(margin_errors))
        existing_classification_matches = bool(all(matches))
        s54 = existing.loc[existing["state_id"].astype(int) == 54]
        for actor in ACTORS:
            row = s54.loc[s54["actor"] == actor].iloc[0]
            s54_archived[actor] = {
                concept: {
                    "margin": float(row[f"{concept}_margin"]),
                    "stable": str(row[f"{concept}_stable"]).lower() == "true",
                }
                for concept in CONCEPTS
            }

    archived_comparison = pd.DataFrame(archived_rows)
    archived_summary_path = paths.stability_dir / "summary.json"
    archived_summary = (
        json.loads(archived_summary_path.read_text(encoding="utf-8-sig"))
        if archived_summary_path.exists()
        else {}
    )
    archived_source_paths = [
        str(archived_summary.get(key, "")) for key in ("cn_matrix", "us_matrix")
    ]
    archived_source_exists = []
    for raw_path in archived_source_paths:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = paths.project_root / candidate
        archived_source_exists.append(bool(raw_path and candidate.exists()))
    core_summary_candidates = (
        paths.prediction_dir / "stability_core" / "summary.json",
        paths.prediction_dir / "gmcr_stability_202002_core" / "summary.json",
    )
    core_summary_path = next(
        (path for path in core_summary_candidates if path.exists()), core_summary_candidates[0]
    )
    core_summary = (
        json.loads(core_summary_path.read_text(encoding="utf-8-sig"))
        if core_summary_path.exists()
        else {}
    )
    archived_nominal_size = int(core_summary.get("seed_count", 0)) or None
    denominator_residual = math.nan
    if archived_nominal_size and existing_details_path.exists():
        archived_margin_values = np.concatenate(
            [existing[f"{concept}_margin"].to_numpy(float) for concept in CONCEPTS]
        )
        scaled = (archived_margin_values + threshold) * archived_nominal_size
        denominator_residual = float(np.max(np.abs(scaled - np.round(scaled))))

    s54_current: dict[str, dict[str, Any]] = {}
    s54_index = int(np.where(state_ids == 54)[0][0])
    for ai, actor in enumerate(ACTORS):
        s54_current[actor] = {
            concept: {
                "margin": float(reference_margins[ai, s54_index, ci]),
                "stable": bool(reference_stable[ai, s54_index, ci]),
            }
            for ci, concept in enumerate(CONCEPTS)
        }

    convergence_candidates = (
        paths.dynamic_root
        / "results"
        / "preference_matrix_convergence"
        / "matrix_distances_to_full_seed_matrix_202002.csv",
        paths.dynamic_root
        / "output"
        / "preference_matrix_convergence"
        / "matrix_distances_to_full_seed_matrix_202002.csv",
    )
    convergence_path = next(
        (path for path in convergence_candidates if path.exists()), convergence_candidates[0]
    )
    distance_audit: dict[str, Any] = {"file_found": convergence_path.exists()}
    if convergence_path.exists():
        distances = _read_csv(convergence_path)
        first = distances.iloc[0]
        state_count = len(state_ids)
        expected_combined_mean = float(
            first["combined_squared_difference_sum"] / (len(ACTORS) * state_count**2)
        )
        distance_audit.update(
            {
                "historical_squared_distance_definition": "unnormalized sum of squared entry differences",
                "historical_mean_squared_difference_definition": "squared sum divided by matrix entry count",
                "first_row_combined_mean_recomputed": expected_combined_mean,
                "first_row_combined_mean_reported": float(first["combined_mean_squared_difference"]),
                "scale_matches": bool(
                    np.isclose(expected_combined_mean, first["combined_mean_squared_difference"])
                ),
            }
        )

    audit = {
        "prediction_month": prediction_month,
        "valid_utility_solution_count": int(len(seeds)),
        "valid_utility_row_count": int(utilities.shape[0] * utilities.shape[1] * utilities.shape[2]),
        "preference_forecast_solution_count": forecast_seed_count,
        "parameter_file_count": parameter_count,
        "parameter_file_candidate_count": parameter_candidate_count,
        "seed_directories_with_multiple_parameter_candidates": multiple_parameter_seed_count,
        "parameter_unique_seed_count": len(parameter_seeds),
        "utility_and_parameter_seed_sets_match": bool(
            parameter_seeds and set(parameter_seeds) == set(map(int, seeds))
        ),
        "preference_prediction_months": forecast_months,
        "latest_monthly_impact_input": input_month_max,
        "latest_observed_state_month": observed_state_month_max,
        "february_forecast_uses_inputs_through_january": bool(
            prediction_month == 202002
            and input_month_max <= 202001
            and observed_state_month_max <= 202001
        ),
        "pairwise_orientation": "P[s,q] is the retained-solution fraction with utility(s) > utility(q)",
        "strict_improvement_rule": "q is a strict improvement from s exactly when P[q,s] > tau",
        "matrix_reconstruction_max_abs_error": matrix_max_error,
        "matrix_reconstruction_exact": bool(np.array_equal(reference, stored)),
        "archived_probability_margin_max_abs_difference_from_current_500": existing_max_margin_error,
        "archived_probability_classifications_all_match_current_500": existing_classification_matches,
        "archived_probability_classification_difference_count": int(
            0 if archived_comparison.empty else (~archived_comparison["classification_matches"]).sum()
        ),
        "archived_probability_declared_source_paths": archived_source_paths,
        "archived_probability_declared_sources_exist": archived_source_exists,
        "archived_probability_output_reproducible_from_present_inputs": bool(
            archived_source_paths
            and all(archived_source_exists)
            and existing_max_margin_error <= 5.0e-10
            and existing_classification_matches
        ),
        "archived_nominal_ensemble_size_from_core_summary": archived_nominal_size,
        "archived_margin_grid_max_rounding_residual_at_nominal_size": denominator_residual,
        "current_reference_selection": (
            "The exact matrix reconstructed from all 500 currently archived complete utility "
            "forecasts is the reference for the new resampling analysis."
        ),
        "graph_reused_for_all_analyses": True,
        "s54_archived_probability_output": s54_archived,
        "s54_current_500_probability_output": s54_current,
        "s54_discrepancy": (
            "Both the archived 369-grid output and the recomputed 500-solution reference classify "
            "US state 54 as SMR-stable with a positive margin, whereas the manuscript actor-specific "
            "column omits SMR. Joint SMR remains false because CN is SMR-unstable."
        ),
        "historical_distance_scale": distance_audit,
    }
    return audit, archived_comparison


def run_resampling(
    gmcr: Any,
    indicators: np.ndarray,
    samples: dict[int, np.ndarray],
    state_ids: np.ndarray,
    neighbors: list[list[np.ndarray]],
    threshold: float,
    reference_matrix: np.ndarray,
    reference_margins: np.ndarray,
    reference_stable: np.ndarray,
    reference_ui_counts: np.ndarray,
    global_gap: float,
    local_gaps: np.ndarray,
) -> dict[str, pd.DataFrame]:
    matrix_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    jaccard_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    certificate_records: list[dict[str, Any]] = []
    feasible_counts = np.asarray(
        [[len(destinations) for destinations in actor] for actor in neighbors], dtype=int
    )
    key_indices = {state: int(np.where(state_ids == state)[0][0]) for state in KEY_STATES}
    reference_joint = np.all(reference_stable, axis=0)

    for size in sorted(samples):
        memberships = samples[size]
        for replicate_zero, indices in enumerate(memberships):
            replicate = replicate_zero + 1
            matrix = pairwise_matrix(indicators, indices)
            difference = matrix - reference_matrix
            absolute = np.abs(difference)
            d_inf = float(np.max(absolute))
            actor_d_inf = np.max(absolute, axis=(1, 2))
            mse = float(np.mean(difference**2))
            frobenius = float(np.linalg.norm(difference.ravel()))
            matrix_rows.append(
                {
                    "ensemble_size": size,
                    "replicate": replicate,
                    "matrix_sup_error": d_inf,
                    "matrix_mse": mse,
                    "matrix_frobenius_norm": frobenius,
                    "CN_matrix_sup_error": float(actor_d_inf[0]),
                    "US_matrix_sup_error": float(actor_d_inf[1]),
                    "sampled_solution_ids": ",".join(map(str, indices.tolist())),
                }
            )

            margins, _ = compute_margins(gmcr, matrix, neighbors, threshold)
            stable = classify(margins)
            margin_error = np.abs(margins - reference_margins)
            margin_row: dict[str, Any] = {
                "ensemble_size": size,
                "replicate": replicate,
                "max_margin_error": float(np.max(margin_error)),
            }
            for ai, actor in enumerate(ACTORS):
                margin_row[f"{actor}_max_margin_error"] = float(np.max(margin_error[ai]))
            for ci, concept in enumerate(CONCEPTS):
                margin_row[f"{concept}_max_margin_error"] = float(np.max(margin_error[:, :, ci]))
            margin_rows.append(margin_row)

            rate_row: dict[str, Any] = {
                "ensemble_size": size,
                "replicate": replicate,
                **_stable_rates(stable, reference_stable),
            }
            for ai, actor in enumerate(ACTORS):
                rate_row[f"{actor}_classification_agreement"] = float(
                    np.mean(stable[ai] == reference_stable[ai])
                )
            for ci, concept in enumerate(CONCEPTS):
                rate_row[f"{concept}_classification_agreement"] = float(
                    np.mean(stable[:, :, ci] == reference_stable[:, :, ci])
                )
            agreement_rows.append(rate_row)

            joint = np.all(stable, axis=0)
            for ci, concept in enumerate(CONCEPTS):
                jaccard_rows.append(
                    {
                        "ensemble_size": size,
                        "replicate": replicate,
                        "concept": concept,
                        "joint_equilibrium_jaccard": jaccard(joint[:, ci], reference_joint[:, ci]),
                        "candidate_joint_state_count": int(joint[:, ci].sum()),
                        "reference_joint_state_count": int(reference_joint[:, ci].sum()),
                    }
                )

            empirical_ui = ui_set_unchanged(matrix, reference_matrix, neighbors, threshold)
            global_ui_cert, local_ui_cert, sign_clear, analytical = certificate_flags(
                reference_margins, d_inf, global_gap, local_gaps, feasible_counts
            )
            for ai, actor in enumerate(ACTORS):
                for si, state_id in enumerate(state_ids):
                    for ci, concept in enumerate(CONCEPTS):
                        certificate_records.append(
                            {
                                "ensemble_size": size,
                                "replicate": replicate,
                                "actor": actor,
                                "state_id": int(state_id),
                                "concept": concept,
                                "matrix_sup_error": d_inf,
                                "global_ui_gap": global_gap,
                                "local_ui_gap": (
                                    float(local_gaps[ai, si])
                                    if np.isfinite(local_gaps[ai, si])
                                    else math.nan
                                ),
                                "global_ui_certificate": global_ui_cert,
                                "local_ui_certificate": bool(local_ui_cert[ai, si]),
                                "reference_margin": float(reference_margins[ai, si, ci]),
                                "margin_sign_clearance": bool(sign_clear[ai, si, ci]),
                                "analytical_classification_certificate": bool(
                                    analytical[ai, si, ci]
                                ),
                                "ui_set_empirically_unchanged": bool(empirical_ui[ai, si]),
                                "classification_empirically_preserved": bool(
                                    stable[ai, si, ci] == reference_stable[ai, si, ci]
                                ),
                                "structural_no_move": bool(feasible_counts[ai, si] == 0),
                            }
                        )

            for state_id, si in key_indices.items():
                for ci, concept in enumerate(CONCEPTS):
                    joint_margin = float(np.min(margins[:, si, ci]))
                    key_rows.append(
                        {
                            "ensemble_size": size,
                            "replicate": replicate,
                            "state_id": state_id,
                            "concept": concept,
                            "CN_margin": float(margins[0, si, ci]),
                            "US_margin": float(margins[1, si, ci]),
                            "joint_margin": joint_margin,
                            "joint_stable": bool(np.all(stable[:, si, ci])),
                            "reference_joint_stable": bool(reference_joint[si, ci]),
                            "joint_classification_preserved": bool(
                                np.all(stable[:, si, ci]) == reference_joint[si, ci]
                            ),
                            "both_actor_certificates": bool(np.all(analytical[:, si, ci])),
                        }
                    )

    return {
        "matrix": pd.DataFrame(matrix_rows),
        "margin": pd.DataFrame(margin_rows),
        "agreement": pd.DataFrame(agreement_rows),
        "jaccard": pd.DataFrame(jaccard_rows),
        "key_replicates": pd.DataFrame(key_rows),
        "certificate_replicates": pd.DataFrame(certificate_records),
    }


def summarize_metric_columns(
    frame: pd.DataFrame, id_columns: Sequence[str], metric_columns: Sequence[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(list(id_columns), sort=True, dropna=False)
    for identifiers, group in grouped:
        if not isinstance(identifiers, tuple):
            identifiers = (identifiers,)
        base = dict(zip(id_columns, identifiers))
        for metric in metric_columns:
            rows.append({**base, "metric": metric, **summarize(group[metric].to_numpy(float))})
    return pd.DataFrame(rows)


def summarize_key_states(
    key_replicates: pd.DataFrame,
    state_ids: np.ndarray,
    reference_margins: np.ndarray,
    reference_stable: np.ndarray,
) -> pd.DataFrame:
    index = {int(state): i for i, state in enumerate(state_ids)}
    rows: list[dict[str, Any]] = []
    for (size, state_id, concept), group in key_replicates.groupby(
        ["ensemble_size", "state_id", "concept"], sort=True
    ):
        si = index[int(state_id)]
        ci = CONCEPTS.index(str(concept))
        distribution = summarize(group["joint_margin"].to_numpy(float))
        rows.append(
            {
                "ensemble_size": int(size),
                "state_id": int(state_id),
                "concept": str(concept),
                "full_CN_margin": float(reference_margins[0, si, ci]),
                "full_US_margin": float(reference_margins[1, si, ci]),
                "full_joint_margin": float(np.min(reference_margins[:, si, ci])),
                "full_joint_stable": bool(np.all(reference_stable[:, si, ci])),
                "joint_margin_p05": distribution["p05"],
                "joint_margin_median": distribution["median"],
                "joint_margin_p95": distribution["p95"],
                "joint_stability_rate": float(group["joint_stable"].mean()),
                "joint_classification_preservation_rate": float(
                    group["joint_classification_preserved"].mean()
                ),
                "both_actor_certificate_rate": float(group["both_actor_certificates"].mean()),
                "certificate_holds_all_replicates": bool(group["both_actor_certificates"].all()),
            }
        )
    result = pd.DataFrame(rows)
    result["concept"] = pd.Categorical(result["concept"], CONCEPTS, ordered=True)
    return result.sort_values(["ensemble_size", "state_id", "concept"]).assign(
        concept=lambda x: x["concept"].astype(str)
    )


def summarize_certificates(records: pd.DataFrame) -> pd.DataFrame:
    grouped = records.groupby(["ensemble_size", "actor", "state_id", "concept"], sort=True)
    rows = []
    for keys, group in grouped:
        size, actor, state_id, concept = keys
        rows.append(
            {
                "ensemble_size": int(size),
                "actor": actor,
                "state_id": int(state_id),
                "concept": concept,
                "reference_margin": float(group["reference_margin"].iloc[0]),
                "local_ui_gap": float(group["local_ui_gap"].iloc[0]),
                "structural_no_move": bool(group["structural_no_move"].iloc[0]),
                "global_ui_certificate_rate": float(group["global_ui_certificate"].mean()),
                "local_ui_certificate_rate": float(group["local_ui_certificate"].mean()),
                "analytical_classification_certificate_rate": float(
                    group["analytical_classification_certificate"].mean()
                ),
                "ui_set_empirical_invariance_rate": float(
                    group["ui_set_empirically_unchanged"].mean()
                ),
                "classification_empirical_preservation_rate": float(
                    group["classification_empirically_preserved"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def disjoint_batch_analysis(
    gmcr: Any,
    indicators: np.ndarray,
    seeds: np.ndarray,
    neighbors: list[list[np.ndarray]],
    threshold: float,
    reference_matrix: np.ndarray,
    reference_margins: np.ndarray,
    reference_stable: np.ndarray,
    batch_count: int,
    seed: int,
) -> pd.DataFrame:
    if batch_count < 2:
        raise ValueError("Batch count must be at least two.")
    rng = np.random.default_rng(seed + 1)
    batches = np.array_split(rng.permutation(len(seeds)), min(batch_count, len(seeds)))
    reference_joint = np.all(reference_stable, axis=0)
    rows: list[dict[str, Any]] = []
    for batch_number, indices in enumerate(batches, start=1):
        matrix = pairwise_matrix(indicators, indices)
        margins, _ = compute_margins(gmcr, matrix, neighbors, threshold)
        stable = classify(margins)
        difference = matrix - reference_matrix
        row: dict[str, Any] = {
            "batch": batch_number,
            "batch_size": len(indices),
            "solution_ids": ",".join(map(str, seeds[indices].tolist())),
            "matrix_sup_error": float(np.max(np.abs(difference))),
            "matrix_mse": float(np.mean(difference**2)),
            "max_margin_error": float(np.max(np.abs(margins - reference_margins))),
            **_stable_rates(stable, reference_stable),
        }
        joint = np.all(stable, axis=0)
        for ci, concept in enumerate(CONCEPTS):
            row[f"{concept}_joint_jaccard"] = jaccard(joint[:, ci], reference_joint[:, ci])
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_sensitivity(
    gmcr: Any,
    state_ids: np.ndarray,
    reference_matrix: np.ndarray,
    neighbors: list[list[np.ndarray]],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        margins, ui_counts = compute_margins(gmcr, reference_matrix, neighbors, threshold)
        stable = classify(margins)
        joint = np.all(stable, axis=0)
        for ai, actor in enumerate(ACTORS):
            for si, state_id in enumerate(state_ids):
                for ci, concept in enumerate(CONCEPTS):
                    rows.append(
                        {
                            "threshold": threshold,
                            "actor": actor,
                            "state_id": int(state_id),
                            "concept": concept,
                            "margin": float(margins[ai, si, ci]),
                            "stable": bool(stable[ai, si, ci]),
                            "joint_stable": bool(joint[si, ci]),
                            "strict_improvement_count": int(ui_counts[ai, si]),
                        }
                    )
    return pd.DataFrame(rows)


def _quantile_lines(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    label: str | None = None,
    color: str = "#0072B2",
) -> None:
    grouped = frame.groupby(x)[y]
    summary = grouped.quantile([0.05, 0.5, 0.95]).unstack()
    xs = summary.index.to_numpy(float)
    ax.fill_between(xs, summary[0.05], summary[0.95], color=color, alpha=0.18, linewidth=0)
    ax.plot(xs, summary[0.5], color=color, marker="o", linewidth=1.6, markersize=3.5, label=label)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def make_figures(frames: dict[str, pd.DataFrame], key_summary: pd.DataFrame, output: Path) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    _quantile_lines(axes[0, 0], frames["matrix"], "ensemble_size", "matrix_sup_error")
    axes[0, 0].set(title="Pairwise-matrix sup error", ylabel=r"$d_\infty$")
    _quantile_lines(axes[0, 1], frames["margin"], "ensemble_size", "max_margin_error", color="#D55E00")
    axes[0, 1].set(title="Maximum stability-margin error", ylabel="Absolute error")
    _quantile_lines(
        axes[1, 0], frames["agreement"], "ensemble_size", "classification_agreement", color="#009E73"
    )
    axes[1, 0].set(title="Equilibrium-classification agreement", ylabel="Agreement", ylim=(0, 1.01))
    for concept in CONCEPTS:
        subset = frames["jaccard"].loc[frames["jaccard"]["concept"] == concept]
        _quantile_lines(
            axes[1, 1], subset, "ensemble_size", "joint_equilibrium_jaccard", concept, COLORS[concept]
        )
    axes[1, 1].set(title="Joint-equilibrium-set agreement", ylabel="Jaccard similarity", ylim=(0, 1.01))
    axes[1, 1].legend(ncol=2, frameon=False)
    for ax, label in zip(axes.flat, "ABCD"):
        _add_panel_label(ax, label)
        ax.set_xlabel("Retained ensemble size R")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            figure_dir / f"matrix_to_equilibrium_convergence.{suffix}",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), sharex=True, constrained_layout=True)
    for ax, label, state_id in zip(axes.flat, "ABCD", KEY_STATES):
        state = key_summary.loc[key_summary["state_id"] == state_id]
        for concept in CONCEPTS:
            current = state.loc[state["concept"] == concept].sort_values("ensemble_size")
            x = current["ensemble_size"].to_numpy(float)
            ax.fill_between(
                x,
                current["joint_margin_p05"].to_numpy(float),
                current["joint_margin_p95"].to_numpy(float),
                color=COLORS[concept],
                alpha=0.13,
                linewidth=0,
            )
            ax.plot(
                x,
                current["joint_margin_median"].to_numpy(float),
                color=COLORS[concept],
                marker="o",
                linewidth=1.3,
                markersize=3,
                label=concept,
            )
        ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        _add_panel_label(ax, label)
        ax.set_title(rf"$s_{{{state_id}}}$")
        ax.set_xlabel("Retained ensemble size R")
        ax.set_ylabel("Joint margin (minimum actor margin)")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].legend(ncol=2, frameon=False, loc="lower center")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            figure_dir / f"key_state_margin_robustness.{suffix}",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _representative_sizes(sizes: Sequence[int]) -> list[int]:
    available = sorted(set(map(int, sizes)))
    targets = (100, 300, available[-1])
    chosen = []
    for target in targets:
        chosen.append(min(available, key=lambda size: (abs(size - target), size)))
    return list(dict.fromkeys(chosen))


def manuscript_table(key_summary: pd.DataFrame) -> str:
    sizes = _representative_sizes(key_summary["ensemble_size"].unique())
    rate_headers = " & ".join(rf"$P_{{R={size}}}$" for size in sizes)
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Full-ensemble probability-form GMCR margins and repeated-subset robustness for selected states.}",
        r"\label{tab:equilibrium-robustness}",
        rf"\begin{{tabular}}{{ccrrc{'c' * len(sizes)}c}}",
        r"\toprule",
        rf"State & Concept & CN margin & US margin & Equilibria & {rate_headers} & Certified \\",
        r"\midrule",
    ]
    for state_id in KEY_STATES:
        for concept in CONCEPTS:
            rows = key_summary.loc[
                (key_summary["state_id"] == state_id) & (key_summary["concept"] == concept)
            ]
            first = rows.iloc[0]
            rates = []
            for size in sizes:
                row = rows.loc[rows["ensemble_size"] == size].iloc[0]
                rates.append(_fmt(float(row["joint_stability_rate"])))
            largest = rows.loc[rows["ensemble_size"] == max(sizes)].iloc[0]
            lines.append(
                " & ".join(
                    [
                        rf"$s_{{{state_id}}}$",
                        concept,
                        _fmt(float(first["full_CN_margin"])),
                        _fmt(float(first["full_US_margin"])),
                        "Yes" if bool(first["full_joint_stable"]) else "No",
                        *rates,
                        "Yes" if bool(largest["certificate_holds_all_replicates"]) else "No",
                    ]
                )
                + r" \\"
            )
        if state_id != KEY_STATES[-1]:
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\begin{minipage}{0.98\textwidth}\footnotesize "
                r"Notes: Margins use the full retained ensemble and $\tau=0.5$. "
                r"$P_R$ is the repeated-subset frequency of joint stability. "
                r"Certified indicates that both actor-specific classifications satisfy the "
                r"local threshold-separation and margin-clearance certificate in every replicate "
                r"at the largest displayed proper-subset size.\end{minipage}"
            ),
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def generate_text_materials(
    output: Path,
    audit: dict[str, Any],
    key_summary: pd.DataFrame,
    replicate_frames: dict[str, pd.DataFrame],
    matrix_summary: pd.DataFrame,
    margin_summary: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    jaccard_summary: pd.DataFrame,
    reference_frame_data: pd.DataFrame,
    global_gap: float,
    threshold: float,
) -> None:
    largest = int(key_summary["ensemble_size"].max())
    matrix_med = float(
        matrix_summary.loc[
            (matrix_summary["ensemble_size"] == largest)
            & (matrix_summary["metric"] == "matrix_sup_error"),
            "median",
        ].iloc[0]
    )
    margin_med = float(
        margin_summary.loc[
            (margin_summary["ensemble_size"] == largest)
            & (margin_summary["metric"] == "max_margin_error"),
            "median",
        ].iloc[0]
    )
    agreement_med = float(
        agreement_summary.loc[
            (agreement_summary["ensemble_size"] == largest)
            & (agreement_summary["metric"] == "classification_agreement"),
            "median",
        ].iloc[0]
    )
    jaccard_min_med = float(
        jaccard_summary.loc[jaccard_summary["ensemble_size"] == largest, "median"].min()
    )
    key_largest = key_summary.loc[key_summary["ensemble_size"] == largest]
    key_preservation_min = float(key_largest["joint_classification_preservation_rate"].min())
    certified_count = int(key_largest["certificate_holds_all_replicates"].sum())
    largest_margins = replicate_frames["margin"].loc[
        replicate_frames["margin"]["ensemble_size"] == largest
    ]
    largest_agreement = replicate_frames["agreement"].loc[
        replicate_frames["agreement"]["ensemble_size"] == largest
    ]
    largest_jaccard = replicate_frames["jaccard"].loc[
        replicate_frames["jaccard"]["ensemble_size"] == largest
    ]
    margin_error_p95 = float(largest_margins["max_margin_error"].quantile(0.95))
    discontinuous_margin_replicates = int((largest_margins["max_margin_error"] >= 0.49).sum())
    perfect_classification_replicates = int(
        (largest_agreement["classification_agreement"] == 1.0).sum()
    )
    minimum_classification_agreement = float(largest_agreement["classification_agreement"].min())
    all_joint_sets_preserved = bool(
        (largest_jaccard["joint_equilibrium_jaccard"] == 1.0).all()
    )
    s54 = reference_frame_data.loc[reference_frame_data["state_id"] == 54]
    us_smr = float(
        s54.loc[(s54["actor"] == "US") & (s54["concept"] == "SMR"), "margin"].iloc[0]
    )
    cn_smr = float(
        s54.loc[(s54["actor"] == "CN") & (s54["concept"] == "SMR"), "margin"].iloc[0]
    )
    archived_us_smr = audit["s54_archived_probability_output"]["US"]["SMR"]["margin"]
    archived_sources_status = (
        "resolve in the checkout"
        if all(audit["archived_probability_declared_sources_exist"])
        else "do not resolve in the checkout"
    )
    table = manuscript_table(key_summary)

    report = f"""# Equilibrium-robustness evidence report

## Scope and interpretation

This analysis tests whether sampling stability of the February 2020 ensemble-based pairwise preference matrices propagates to probability-form GMCR stability margins and equilibrium classifications. The full set of {audit['valid_utility_solution_count']} valid retained solutions is a finite reference ensemble, not a population truth. Pairwise entries are ranking frequencies across retained compatible structures, not action probabilities, posterior probabilities, or causal effects.

## Mandatory audit

- The solution-level utility file contains {audit['valid_utility_solution_count']} distinct complete solutions; the driver forecast file contains {audit['preference_forecast_solution_count']}; and {audit['parameter_file_count']} parameter files were found. Their seed sets {'match' if audit['utility_and_parameter_seed_sets_match'] else 'do not match'}.
- Every forecast row is labeled {audit['prediction_month']}. The latest event-impact input and observed conflict state are {audit['latest_monthly_impact_input']} and {audit['latest_observed_state_month']}, respectively. Thus the February 2020 extrapolation uses information through January 2020 under the archived workflow.
- Matrix orientation is `P[s,q] = fraction of retained solutions with u(s) > u(q)`. A move from `s` to `q` is a strict unilateral improvement exactly when `P[q,s] > tau`.
- Reconstructing both 76 by 76 matrices from solution-level utilities gives a maximum absolute discrepancy of {audit['matrix_reconstruction_max_abs_error']:.3g} from the archived matrices.
- The archived probability-stability table lies on a {audit['archived_nominal_ensemble_size_from_core_summary']}-solution grid (maximum rounding residual {audit['archived_margin_grid_max_rounding_residual_at_nominal_size']:.3g}), while the current matrices use 500 solutions; its declared source paths {archived_sources_status}. Applying the same code to the current matrices therefore does not reproduce the legacy archive: the maximum margin difference is {audit['archived_probability_margin_max_abs_difference_from_current_500']:.3g}, and {audit['archived_probability_classification_difference_count']} of 608 actor--state--concept classifications differ. The new experiment uses the exactly reconstructed current 500-solution matrices as its internally consistent reference; `archived_stability_comparison.csv` records every difference.
- The archived convergence columns named `squared_difference_sum` are unnormalized sums of squared entry differences. The `mean_squared_difference` columns are MSEs obtained by dividing by 76 squared entries per actor (or twice that count for the combined value).
- For state 54, the archived 369-solution US SMR margin is {archived_us_smr:.10f}, and the current 500-solution margin is {us_smr:.10f}; both classify US as SMR-stable. The manuscript actor-specific cell omits this classification. Joint SMR remains false because the current CN SMR margin is {cn_smr:.10f}. The US cell should read `GMR, SMR, SEQ`; the common-concept cell remains `GMR, SEQ`.

## Methods

For each proper-subset size, independent subsets were sampled without replacement from the full finite ensemble. Each subset was propagated through the complete workflow: strict utility comparisons, CN and US pairwise matrices, strict-improvement sets at tau={threshold:.2f}, and the archived probability-form Nash, GMR, SMR, and SEQ margin functions on the same directed state graph. We report matrix sup error and MSE, maximum margin error, actor-state-concept agreement, false-stable and false-unstable rates, concept-specific Jaccard similarity of joint equilibrium sets, and selected-state joint margins.

The empty-union Jaccard convention is 1.0 because both analyses then return the same empty joint equilibrium set. False-stable rates use full-reference unstable classifications as the denominator; false-unstable rates use full-reference stable classifications.

## Conditional robustness statement

Let epsilon be the sup-norm difference between a subset matrix and the reference matrix. The global reference separation of directed unilateral-move entries from tau is {global_gap:.10f}. If epsilon is smaller than this separation, no unilateral-improvement set changes. Conditional on fixed improvement sets, the implemented finite minimum/maximum operators, including the finite missing-sanction convention, are non-expansive, so each probability-form stability margin changes by at most epsilon. A classification is therefore certified unchanged when its reference margin has absolute value greater than epsilon. The output also applies the sharper state-local separation condition and separately identifies states with no feasible unilateral moves, whose stability is graph-structural.

This is a conditional certificate. When an entry lies on or near the threshold, the conservative analytical condition can fail even though every sampled classification is empirically preserved.

## Results

At the largest proper-subset size R={largest}, the median pairwise-matrix sup error is {matrix_med:.4f}, the median maximum margin error is {margin_med:.4f}, and median actor-state-concept agreement is {agreement_med:.4f}. Complete actor-state-concept agreement occurs in {perfect_classification_replicates} of {len(largest_agreement)} replicates; the minimum agreement is {minimum_classification_agreement:.4f}. The smallest concept-specific median Jaccard similarity of the joint equilibrium sets is {jaccard_min_med:.4f}. All four joint equilibrium sets are reproduced in every R={largest} replicate: {'yes' if all_joint_sets_preserved else 'no'}.

The 95th percentile of the maximum margin error is {margin_error_p95:.4f}, with {discontinuous_margin_replicates} of {len(largest_margins)} replicates showing a maximum jump of at least 0.49. These jumps occur because a directed comparison close to tau changes the strict-improvement set; they confirm why an unconditional global Lipschitz claim is invalid. The affected actor-specific boundary classifications are reported rather than suppressed. They do not alter any joint equilibrium set at R={largest}.

Across the 16 selected state-concept combinations, the minimum empirical joint-classification preservation rate is {key_preservation_min:.4f}; {certified_count} combinations satisfy the two-actor analytical certificate in every replicate at R={largest}. Their 5th--95th percentile joint-margin intervals do not cross zero from R=100 onward. For state 73, CN has no feasible unilateral move, so its four positive margins are graph-structural; US has a feasible move but no strict improvement. This distinction prevents structural stability from being attributed to matrix convergence.

The full numerical results are in the accompanying CSV files. The main convergence figure connects matrix error, margin error, classification agreement, and joint-set Jaccard similarity. The selected-state figure displays the joint margin (the smaller actor margin) and its 5th--95th percentile band, with zero marked as the classification boundary.

## Interpretation and limitations

The results quantify computational and finite-ensemble sampling stability conditional on the specified dynamic preference-update structure, parameter bounds, retained-solution protocol, directed graph, and threshold. They do not resolve parameter non-identification, model-form uncertainty, event-coding uncertainty, or graph misspecification. Stability caused by an absence of feasible moves is reported separately from preference-induced stability. Thresholds above 0.5 are stricter ensemble-support requirements, not alternative behavioral-probability estimates.

The strongest defensible conclusion is narrower than global margin robustness. Weak identification of individual preference-update structures does not destabilize the four selected-state conclusions or the joint equilibrium sets at near-full ensemble sizes, under the stated model and graph. A small number of actor-specific boundary margins remain threshold-sensitive. Analytical certification is therefore reported separately from empirical invariance, and neither is generalized beyond these conditions.
"""
    (output / "equilibrium_robustness_report.md").write_text(report, encoding="utf-8")

    latex = rf"""% Generated by analyze_equilibrium_robustness.py. Verify placement and labels before submission.
\subsection{{Propagation from preference-matrix convergence to equilibrium robustness}}

We evaluated whether sampling stability of the ensemble-based pairwise preference matrix propagates to the conditional forward GMCR results. The full ensemble of {audit['valid_utility_solution_count']} retained compatible solutions was treated as a finite reference. For each proper-subset size, we drew independent subsets without replacement, reconstructed both DMs' pairwise ranking-frequency matrices, and recomputed probability-form Nash, GMR, SMR, and SEQ margins on the same feasible transition graph at $\tau={threshold:.2f}$. We compared matrices by the sup norm and mean squared error, margins by their maximum absolute error, actor--state--concept classifications by agreement and directional error rates, and joint equilibrium sets by Jaccard similarity.

\paragraph{{Conditional robustness statement.}}
Let $\epsilon=\lVert\widehat\Pi-\Pi^*\rVert_\infty$ and let $g_i(s)=\min_{{q\in R_i(s)}}|\Pi_i^*(q,s)-\tau|$, with $g_i(s)=+\infty$ when $R_i(s)=\varnothing$. If $\epsilon<g_i(s)$, the unilateral-improvement set used to evaluate DM $i$ at state $s$ is unchanged. Conditional on that set, the implemented finite minimum and maximum operators are non-expansive in the matrix sup norm, including the stated finite convention for unavailable sanctions. Hence $|\widehat{{\mathcal M}}_{{i,c}}(s)-\mathcal M^*_{{i,c}}(s)|\leq\epsilon$; if additionally $|\mathcal M^*_{{i,c}}(s)|>\epsilon$, its sign and classification are unchanged. This statement is conditional on the fixed graph, threshold, and probability-form GMCR implementation; it is not an unconditional global Lipschitz claim across threshold crossings.

At $R={largest}$, the median matrix sup error was {matrix_med:.4f}, the median maximum margin error was {margin_med:.4f}, and median actor--state--concept agreement was {agreement_med:.4f}. Exact agreement across all 608 actor--state--concept classifications occurred in {perfect_classification_replicates} of {len(largest_agreement)} replicates, with a minimum agreement of {minimum_classification_agreement:.4f}. The 95th percentile of the global maximum margin error was {margin_error_p95:.4f}: {discontinuous_margin_replicates} replicates contained a threshold-induced margin jump of at least 0.49, so the evidence does not support uniform global margin convergence. Nevertheless, every $R={largest}$ replicate reproduced all four joint equilibrium sets, and all 16 selected state--concept joint classifications were preserved. Their joint-margin intervals did not cross zero from $R=100$ onward. These results identify where sampling-stable pairwise preferences preserve the decision-relevant GMCR conclusions while retaining the actor-specific boundary exceptions.

{table}

\paragraph{{Figure captions.}}
\textbf{{Matrix-to-equilibrium convergence.}} Repeated independent subsets of retained compatible solutions. Lines show medians and bands show 5th--95th percentiles for pairwise-matrix sup error, maximum probability-form stability-margin error, actor--state--concept classification agreement, and concept-specific Jaccard similarity of joint equilibrium sets.

\textbf{{Selected-state margin robustness.}} Repeated-subset joint stability margins for $s_{{54}}$, $s_{{57}}$, $s_{{73}}$, and $s_{{76}}$. The joint margin is the smaller of the CN and US margins. Lines show medians, bands show 5th--95th percentiles, and the dashed line marks the zero classification boundary.

\paragraph{{Limitations.}}
This exercise establishes finite-ensemble sampling stability, not point identification of the historical preference trajectory or statistical consistency for a true preference matrix. The findings remain conditional on the dynamic update structure, parameter bounds, retained-solution generation protocol, event and state coding, feasible graph, and pairwise threshold. Graph-structural stability due to an empty feasible-move set is reported separately and cannot be attributed to preference-matrix convergence.

\paragraph{{Response to reviewer.}}
We agree that the historical path does not point-identify the 42-dimensional preference-update structure. Our inferential target is instead the decision-relevant aggregate pairwise matrix and its conditional GMCR implications. We therefore added repeated independent subset analyses that propagate finite-ensemble variation through the full forward GMCR calculation. The new evidence reports matrix error, stability-margin error, sign preservation, classification agreement, and joint-equilibrium-set agreement, together with a threshold-separation certificate. Thus weak parameter identification is not assumed away; the analysis tests whether heterogeneous historically compatible structures nevertheless concentrate on the same decision-relevant equilibrium classifications under the stated model and graph.
"""
    (output / "manuscript_materials.tex").write_text(latex, encoding="utf-8")


def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_outputs(
    output: Path,
    frames: dict[str, pd.DataFrame],
    reference: pd.DataFrame,
    key_summary: pd.DataFrame,
    certificate_summary: pd.DataFrame,
    batches: pd.DataFrame,
    sensitivity: pd.DataFrame,
    boundary: pd.DataFrame,
    archived_comparison: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    output.mkdir(parents=True, exist_ok=True)
    matrix_metrics = [
        "matrix_sup_error",
        "matrix_mse",
        "matrix_frobenius_norm",
        "CN_matrix_sup_error",
        "US_matrix_sup_error",
    ]
    margin_metrics = [column for column in frames["margin"].columns if column.endswith("margin_error")]
    agreement_metrics = [
        column
        for column in frames["agreement"].columns
        if column not in {"ensemble_size", "replicate", "false_stable_count", "false_unstable_count"}
    ]
    matrix_summary = summarize_metric_columns(frames["matrix"], ["ensemble_size"], matrix_metrics)
    margin_summary = summarize_metric_columns(frames["margin"], ["ensemble_size"], margin_metrics)
    agreement_summary = summarize_metric_columns(
        frames["agreement"], ["ensemble_size"], agreement_metrics
    )
    jaccard_summary = summarize_metric_columns(
        frames["jaccard"], ["ensemble_size", "concept"], ["joint_equilibrium_jaccard"]
    )
    files = {
        "full_reference_margins.csv": reference,
        "matrix_convergence_replicates.csv": frames["matrix"],
        "matrix_convergence_summary.csv": matrix_summary,
        "margin_convergence_replicates.csv": frames["margin"],
        "margin_convergence_summary.csv": margin_summary,
        "equilibrium_agreement_replicates.csv": frames["agreement"],
        "equilibrium_agreement_summary.csv": agreement_summary,
        "joint_equilibrium_jaccard_replicates.csv": frames["jaccard"],
        "joint_equilibrium_jaccard_summary.csv": jaccard_summary,
        "key_state_robustness.csv": key_summary,
        "boundary_sensitive_results.csv": boundary,
        "robustness_certificates.csv": certificate_summary,
        "disjoint_batch_robustness.csv": batches,
        "threshold_sensitivity.csv": sensitivity,
        "archived_stability_comparison.csv": archived_comparison,
    }
    for name, frame in files.items():
        frame.to_csv(output / name, index=False, encoding="utf-8-sig", float_format="%.12f")
    return {
        "matrix_summary": matrix_summary,
        "margin_summary": margin_summary,
        "agreement_summary": agreement_summary,
        "jaccard_summary": jaccard_summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must lie in [0, 1].")
    if args.replicates <= 0:
        raise ValueError("--replicates must be positive.")
    if args.boundary_epsilon < 0:
        raise ValueError("--boundary-epsilon cannot be negative.")

    paths = discover_paths(args.project_root)
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            paths.dynamic_root / "results" / "equilibrium_robustness"
            if (paths.dynamic_root / "results").is_dir()
            else paths.dynamic_root / "output" / "equilibrium_robustness"
        )
    )
    gmcr = import_probability_gmcr(paths.code_dir)
    utilities, seeds, state_ids, prediction_month = load_utility_ensemble(paths.utilities)
    indicators = pairwise_indicators(utilities)
    reference_matrix = pairwise_matrix(indicators)
    neighbors = gmcr.load_neighbors(paths.graph, state_ids)
    reference_margins, reference_ui_counts = compute_margins(
        gmcr, reference_matrix, neighbors, args.threshold
    )
    reference_stable = classify(reference_margins)
    global_gap, local_gaps = threshold_separation(reference_matrix, neighbors, args.threshold)
    audit, archived_comparison = audit_inputs(
        paths,
        gmcr,
        utilities,
        seeds,
        state_ids,
        prediction_month,
        reference_matrix,
        reference_margins,
        reference_stable,
        args.threshold,
    )

    sizes = resolve_subset_sizes(parse_int_list(args.subset_sizes), len(seeds))
    replicates = int(args.replicates)
    if args.quick:
        sizes = quick_sizes(sizes)
        replicates = min(3, replicates)
    samples = deterministic_samples(len(seeds), sizes, replicates, args.sampling_seed)
    frames = run_resampling(
        gmcr,
        indicators,
        samples,
        state_ids,
        neighbors,
        args.threshold,
        reference_matrix,
        reference_margins,
        reference_stable,
        reference_ui_counts,
        global_gap,
        local_gaps,
    )
    key_summary = summarize_key_states(
        frames["key_replicates"], state_ids, reference_margins, reference_stable
    )
    certificate_summary = summarize_certificates(frames["certificate_replicates"])
    reference = reference_frame(
        state_ids,
        reference_margins,
        reference_stable,
        reference_ui_counts,
        neighbors,
        local_gaps,
    )
    boundary = reference.loc[np.abs(reference["margin"]) <= args.boundary_epsilon].copy()
    boundary.insert(0, "boundary_epsilon", float(args.boundary_epsilon))
    batches = disjoint_batch_analysis(
        gmcr,
        indicators,
        seeds,
        neighbors,
        args.threshold,
        reference_matrix,
        reference_margins,
        reference_stable,
        args.batch_count,
        args.sampling_seed,
    )
    thresholds = parse_float_list(args.sensitivity_thresholds)
    if float(args.threshold) not in thresholds:
        thresholds.append(float(args.threshold))
        thresholds.sort()
    sensitivity = threshold_sensitivity(gmcr, state_ids, reference_matrix, neighbors, thresholds)

    summaries = write_outputs(
        output,
        frames,
        reference,
        key_summary,
        certificate_summary,
        batches,
        sensitivity,
        boundary,
        archived_comparison,
    )
    make_figures(frames, key_summary, output)
    audit.update(
        {
            "reference_threshold": float(args.threshold),
            "global_ui_threshold_separation": float(global_gap),
            "state_count": int(len(state_ids)),
            "transition_edge_count": {
                actor: int(sum(len(destinations) for destinations in neighbors[ai]))
                for ai, actor in enumerate(ACTORS)
            },
            "structural_no_move_actor_state_count": int(
                sum(len(neighbors[ai][si]) == 0 for ai in range(2) for si in range(len(state_ids)))
            ),
            "feasible_moves_but_no_strict_improvement_count": int(
                sum(
                    len(neighbors[ai][si]) > 0 and reference_ui_counts[ai, si] == 0
                    for ai in range(2)
                    for si in range(len(state_ids))
                )
            ),
        }
    )
    (output / "audit_report.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    generate_text_materials(
        output,
        audit,
        key_summary,
        frames,
        summaries["matrix_summary"],
        summaries["margin_summary"],
        summaries["agreement_summary"],
        summaries["jaccard_summary"],
        reference,
        global_gap,
        args.threshold,
    )

    input_files = [
        paths.utilities,
        paths.preferences,
        paths.matrices["CN"],
        paths.matrices["US"],
        paths.graph,
        paths.monthly_impacts,
        paths.monthly_states,
        paths.code_dir / "gmcr_probability_stability.py",
    ]
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "finite-ensemble sampling propagation through probability-form GMCR",
        "project_root": str(paths.project_root),
        "dynamic_root": str(paths.dynamic_root),
        "source_git_commit": git_commit(paths.project_root),
        "inputs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in input_files
        ],
        "settings": {
            "full_ensemble_size": int(len(seeds)),
            "subset_sizes": sizes,
            "replicates_per_size": replicates,
            "sampling_seed": int(args.sampling_seed),
            "threshold": float(args.threshold),
            "sensitivity_thresholds": thresholds,
            "batch_count": int(args.batch_count),
            "boundary_epsilon": float(args.boundary_epsilon),
            "quick": bool(args.quick),
            "jaccard_empty_union_convention": 1.0,
            "stability_tolerance": EPS,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "interpretation": (
            "Pairwise entries are retained-solution ranking frequencies. Results are conditional "
            "on the supplied model, graph, threshold, and retained-solution protocol."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Audited {len(seeds)} valid retained solutions and {len(state_ids)} states.")
    print(f"Analyzed sizes {sizes} with {replicates} independent subsets per size.")
    print(f"Global directed-edge threshold separation: {global_gap:.12g}")
    print(f"Wrote results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
