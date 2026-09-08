"""Evaluate historical and one-step-ahead GMCR stability for fitted parameters."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import solve_core_preference_model as solver


CONCEPTS = ("Nash", "GMR", "SMR", "SEQ")


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "List every Nash/GMR/SMR/SEQ-stable state for CN and US in each "
            "historical month, then evaluate the next forecast month."
        )
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        required=True,
        help="Fitted parameters_*.json file.",
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
        default=None,
        help="Output directory; defaults to the parameter file's directory.",
    )
    parser.add_argument(
        "--prediction-month",
        type=int,
        default=202002,
        help="One-step-ahead month in YYYYMM form (default: 202002).",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def next_month(month: int) -> int:
    year, value = divmod(int(month), 100)
    if value < 1 or value > 12:
        raise ValueError(f"Invalid YYYYMM month: {month}")
    return (year + 1) * 100 + 1 if value == 12 else year * 100 + value + 1


def find_sibling_json(parameter_path: Path, stem: str) -> Path | None:
    timestamp = parameter_path.stem.removeprefix("parameters_")
    exact = parameter_path.parent / f"{stem}_{timestamp}.json"
    if exact.exists():
        return exact
    matches = sorted(parameter_path.parent.glob(f"{stem}_*.json"))
    return matches[-1] if matches else None


def load_config(parameter_path: Path, input_dir: Path) -> solver.SolverConfig:
    summary_path = find_sibling_json(parameter_path, "summary")
    raw_config: dict[str, Any] = {}
    if summary_path is not None:
        raw_config = read_json(summary_path).get("config", {})

    allowed = {field.name for field in fields(solver.SolverConfig)}
    config_values = {key: value for key, value in raw_config.items() if key in allowed}
    config_values["input_dir"] = str(input_dir.resolve())
    config_values["output_dir"] = str(parameter_path.parent.resolve())
    return solver.SolverConfig(**config_values)


def load_parameters(
    parameter_path: Path, config: solver.SolverConfig
) -> solver.Parameters:
    payload = read_json(parameter_path)
    vector = np.asarray(payload.get("vector", []), dtype=float)
    if vector.shape != (solver.PARAMETER_DIM,):
        raise ValueError(
            f"Expected a {solver.PARAMETER_DIM}-element parameter vector in "
            f"{parameter_path}, got shape {vector.shape}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("The parameter vector contains non-finite values.")

    lower, upper = solver.parameter_bounds(config)
    if np.any(vector < lower - 1.0e-10) or np.any(vector > upper + 1.0e-10):
        raise ValueError("The parameter vector violates the model bounds.")
    return solver.decode(vector, lower, upper)


def build_extended_paths(
    data: solver.ModelData,
    params: solver.Parameters,
    config: solver.SolverConfig,
) -> tuple[np.ndarray, np.ndarray]:
    # The placeholder forecast impact is never consumed: h_(T+1) uses impacts
    # only through T under the model's lagged update equation.
    placeholder = np.zeros_like(data.impacts_scaled[:1])
    extended_impacts = np.concatenate([data.impacts_scaled, placeholder], axis=0)
    preferences = solver.compute_h(
        params,
        extended_impacts,
        nonlinear_saturation=config.nonlinear_saturation,
    )
    perturbations = solver.generate_score_perturbations(
        len(data.months) + 1, len(data.state_ids), config
    )
    return preferences, perturbations


def evaluate_all_states(
    data: solver.ModelData,
    preferences: np.ndarray,
    perturbations: np.ndarray,
    config: solver.SolverConfig,
    prediction_month: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = [int(month) for month in data.months] + [int(prediction_month)]
    detail_rows: list[dict[str, Any]] = []
    set_rows: list[dict[str, Any]] = []
    equilibrium_rows: list[dict[str, Any]] = []

    for t, month in enumerate(months):
        period = "prediction" if t == len(data.months) else "historical"
        observed_state_id = (
            int(data.state_ids[data.observed_indices[t]]) if period == "historical" else None
        )
        raw_utilities = solver.perturbed_utilities(
            data.omega, perturbations[t], preferences[t]
        )
        utilities = solver.standardize_utilities(raw_utilities)
        flags_by_actor = np.zeros(
            (solver.N_ACTORS, len(data.state_ids), len(CONCEPTS)), dtype=bool
        )

        for actor, country in enumerate(solver.COUNTRIES):
            for state_index, state_id in enumerate(data.state_ids):
                margins = solver.stability_margins(
                    state_index, actor, utilities, data.neighbors, config
                )
                flags = solver.stability_flags(margins)
                flags_by_actor[actor, state_index] = flags
                row: dict[str, Any] = {
                    "month": month,
                    "period": period,
                    "actor": country,
                    "state_id": int(state_id),
                    "is_observed_state": (
                        int(state_id) == observed_state_id if observed_state_id is not None else False
                    ),
                    "utility_raw": float(raw_utilities[actor, state_index]),
                    "utility_standardized": float(utilities[actor, state_index]),
                }
                for concept_index, concept in enumerate(CONCEPTS):
                    row[f"{concept}_margin"] = float(margins[concept_index])
                    row[f"{concept}_stable"] = bool(flags[concept_index])
                detail_rows.append(row)

            for concept_index, concept in enumerate(CONCEPTS):
                ids = data.state_ids[flags_by_actor[actor, :, concept_index]].astype(int)
                set_rows.append(
                    {
                        "month": month,
                        "period": period,
                        "actor": country,
                        "stability_concept": concept,
                        "stable_state_count": int(len(ids)),
                        "stable_state_ids": "|".join(str(value) for value in ids),
                        "observed_state_id": observed_state_id,
                        "observed_state_stable": (
                            bool(
                                flags_by_actor[
                                    actor, data.state_id_to_index[observed_state_id], concept_index
                                ]
                            )
                            if observed_state_id is not None
                            else None
                        ),
                    }
                )

        for concept_index, concept in enumerate(CONCEPTS):
            joint_flags = np.all(flags_by_actor[:, :, concept_index], axis=0)
            ids = data.state_ids[joint_flags].astype(int)
            equilibrium_rows.append(
                {
                    "month": month,
                    "period": period,
                    "stability_concept": concept,
                    "equilibrium_state_count": int(len(ids)),
                    "equilibrium_state_ids": "|".join(str(value) for value in ids),
                    "observed_state_id": observed_state_id,
                    "observed_state_is_equilibrium": (
                        bool(joint_flags[data.state_id_to_index[observed_state_id]])
                        if observed_state_id is not None
                        else None
                    ),
                }
            )

    return (
        pd.DataFrame(detail_rows),
        pd.DataFrame(set_rows),
        pd.DataFrame(equilibrium_rows),
    )


def build_state_classification(details: pd.DataFrame) -> pd.DataFrame:
    """Return one row per month/state with both actors' stability types."""
    rows: list[dict[str, Any]] = []
    group_columns = ["month", "period", "state_id", "is_observed_state"]
    for keys, group in details.groupby(group_columns, sort=False, dropna=False):
        month, period, state_id, is_observed = keys
        row: dict[str, Any] = {
            "month": int(month),
            "period": str(period),
            "state_id": int(state_id),
            "is_observed_state": bool(is_observed),
        }
        actor_types: dict[str, set[str]] = {}
        for actor in solver.COUNTRIES:
            actor_row = group.loc[group["actor"] == actor]
            if len(actor_row) != 1:
                raise ValueError(
                    f"Expected one {actor} row for month {month}, state {state_id}."
                )
            values = actor_row.iloc[0]
            satisfied = {
                concept for concept in CONCEPTS if bool(values[f"{concept}_stable"])
            }
            actor_types[actor] = satisfied
            row[f"{actor}_stability_types"] = (
                "|".join(concept for concept in CONCEPTS if concept in satisfied)
                if satisfied
                else "None"
            )
            for concept in CONCEPTS:
                row[f"{actor}_{concept}_stable"] = bool(values[f"{concept}_stable"])
                row[f"{actor}_{concept}_margin"] = float(values[f"{concept}_margin"])

        joint_types = [
            concept
            for concept in CONCEPTS
            if all(concept in actor_types[actor] for actor in solver.COUNTRIES)
        ]
        row["joint_equilibrium_types"] = "|".join(joint_types) if joint_types else "None"
        for concept in CONCEPTS:
            row[f"joint_{concept}_equilibrium"] = concept in joint_types
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_payload(
    parameter_path: Path,
    config: solver.SolverConfig,
    params: solver.Parameters,
    prediction_month: int,
    preferences: np.ndarray,
    details: pd.DataFrame,
    stable_sets: pd.DataFrame,
    equilibria: pd.DataFrame,
) -> dict[str, Any]:
    forecast_sets = stable_sets.loc[stable_sets["period"] == "prediction"]
    forecast_equilibria = equilibria.loc[equilibria["period"] == "prediction"]
    reference_state_id = int(
        stable_sets.loc[stable_sets["period"] == "historical", "observed_state_id"].iloc[-1]
    )
    reference_rows = details.loc[
        (details["period"] == "prediction")
        & (details["state_id"] == reference_state_id)
    ]
    actors: dict[str, Any] = {}
    for actor in solver.COUNTRIES:
        actor_rows = forecast_sets.loc[forecast_sets["actor"] == actor]
        actors[actor] = {
            str(row.stability_concept): {
                "count": int(row.stable_state_count),
                "state_ids": (
                    [int(value) for value in str(row.stable_state_ids).split("|")]
                    if row.stable_state_ids
                    else []
                ),
            }
            for row in actor_rows.itertuples(index=False)
        }

    joint = {
        str(row.stability_concept): {
            "count": int(row.equilibrium_state_count),
            "state_ids": (
                [int(value) for value in str(row.equilibrium_state_ids).split("|")]
                if row.equilibrium_state_ids
                else []
            ),
        }
        for row in forecast_equilibria.itertuples(index=False)
    }
    reference_stability: dict[str, Any] = {}
    for row in reference_rows.itertuples(index=False):
        reference_stability[str(row.actor)] = {
            concept: {
                "stable": bool(getattr(row, f"{concept}_stable")),
                "margin": float(getattr(row, f"{concept}_margin")),
            }
            for concept in CONCEPTS
        }
    reference_equilibrium = {
        str(row.stability_concept): reference_state_id
        in (
            [int(value) for value in str(row.equilibrium_state_ids).split("|")]
            if row.equilibrium_state_ids
            else []
        )
        for row in forecast_equilibria.itertuples(index=False)
    }
    return {
        "parameters_file": str(parameter_path.resolve()),
        "prediction_month": int(prediction_month),
        "forecast_assumption": (
            "One-step preference forecast uses observed impacts through the final "
            "historical month; no prediction-month impact is imputed."
        ),
        "stability_rule": "A state is stable when its model stability margin is >= 0.",
        "theta": float(config.theta),
        "score_perturbation": {
            "delta": float(config.delta),
            "rho": float(config.delta_persistence),
            "seed": int(config.perturbation_seed),
        },
        "forecast_preferences": {
            country: [float(value) for value in preferences[-1, actor]]
            for actor, country in enumerate(solver.COUNTRIES)
        },
        "last_observed_state_assessment": {
            "state_id": reference_state_id,
            "stability_by_actor": reference_stability,
            "joint_equilibrium": reference_equilibrium,
        },
        "stable_states_by_actor": actors,
        "joint_equilibrium_states": joint,
        "parameter_alpha": {
            country: float(params.alpha[actor])
            for actor, country in enumerate(solver.COUNTRIES)
        },
    }


def main() -> None:
    args = parse_args()
    parameter_path = args.parameters.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or parameter_path.parent).resolve()

    config = load_config(parameter_path, input_dir)
    data = solver.load_model_data(config)
    expected_month = next_month(int(data.months[-1]))
    if int(args.prediction_month) != expected_month:
        raise ValueError(
            "This script performs a one-step forecast. Expected prediction month "
            f"{expected_month} after {int(data.months[-1])}, got {args.prediction_month}."
        )

    params = load_parameters(parameter_path, config)
    preferences, perturbations = build_extended_paths(data, params, config)
    details, stable_sets, equilibria = evaluate_all_states(
        data, preferences, perturbations, config, int(args.prediction_month)
    )
    classifications = build_state_classification(details)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = parameter_path.stem.removeprefix("parameters_")
    detail_path = output_dir / f"stability_by_state_{timestamp}.csv"
    classification_path = output_dir / f"state_stability_types_{timestamp}.csv"
    prediction_table_path = (
        output_dir / f"prediction_{args.prediction_month}_state_stability_{timestamp}.csv"
    )
    sets_path = output_dir / f"stable_state_sets_{timestamp}.csv"
    equilibria_path = output_dir / f"equilibrium_state_sets_{timestamp}.csv"
    prediction_path = output_dir / f"prediction_{args.prediction_month}_stability_{timestamp}.json"
    details.to_csv(detail_path, index=False, encoding="utf-8-sig")
    classifications.to_csv(classification_path, index=False, encoding="utf-8-sig")
    classifications.loc[classifications["period"] == "prediction"].to_csv(
        prediction_table_path, index=False, encoding="utf-8-sig"
    )
    stable_sets.to_csv(sets_path, index=False, encoding="utf-8-sig")
    equilibria.to_csv(equilibria_path, index=False, encoding="utf-8-sig")
    write_json(
        prediction_path,
        prediction_payload(
            parameter_path,
            config,
            params,
            int(args.prediction_month),
            preferences,
            details,
            stable_sets,
            equilibria,
        ),
    )

    print(f"Wrote {detail_path}")
    print(f"Wrote {classification_path}")
    print(f"Wrote {prediction_table_path}")
    print(f"Wrote {sets_path}")
    print(f"Wrote {equilibria_path}")
    print(f"Wrote {prediction_path}")
    print("\nPrediction joint equilibria:")
    for row in equilibria.loc[equilibria["period"] == "prediction"].itertuples(index=False):
        state_ids = row.equilibrium_state_ids or "(none)"
        print(f"  {row.stability_concept}: {state_ids}")

    last_state = int(data.state_ids[data.observed_indices[-1]])
    print(f"\nLast observed state {last_state} in {args.prediction_month}:")
    forecast_reference = details.loc[
        (details["period"] == "prediction") & (details["state_id"] == last_state)
    ]
    for row in forecast_reference.itertuples(index=False):
        stable = [concept for concept in CONCEPTS if getattr(row, f"{concept}_stable")]
        print(f"  {row.actor}: {', '.join(stable) if stable else '(none)'}")


if __name__ == "__main__":
    main()
