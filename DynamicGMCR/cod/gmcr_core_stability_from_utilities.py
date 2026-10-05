"""Apply the core utility-based GMCR stability rules to forecast utilities.

The probability matrices are derived from an ensemble of forecast utilities.
Core-model stability is therefore evaluated for every seed and summarized
across seeds; it is not reconstructed from pairwise probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import solve_core_preference_model as solver


CONCEPTS = ("Nash", "GMR", "SMR", "SEQ")


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Compute core-model GMCR margins for every forecast-utility seed."
    )
    parser.add_argument(
        "--utilities",
        type=Path,
        default=demo_dir / "output" / "preference_prediction" / "predicted_utilities_by_seed_202002.csv",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=demo_dir / "input",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=demo_dir / "output" / "preference_prediction" / "gmcr_stability_202002_core",
    )
    return parser.parse_args()


def load_utilities(path: Path, state_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"seed", "actor", "state_id", "forecast_utility"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame["seed"] = frame["seed"].astype(int)
    frame["state_id"] = frame["state_id"].astype(int)
    frame["forecast_utility"] = pd.to_numeric(frame["forecast_utility"], errors="coerce")
    if frame["forecast_utility"].isna().any():
        raise ValueError(f"{path}: forecast_utility contains non-numeric values")
    actor_order = list(solver.COUNTRIES)
    seed_ids = np.sort(frame["seed"].unique())
    state_order = np.asarray(state_ids, dtype=int)
    expected = len(seed_ids) * len(actor_order) * len(state_order)
    if len(frame) != expected:
        raise ValueError(f"{path}: expected {expected} rows, found {len(frame)}")
    index = pd.MultiIndex.from_product(
        [seed_ids, actor_order, state_order], names=["seed", "actor", "state_id"]
    )
    values = frame.set_index(["seed", "actor", "state_id"])["forecast_utility"].reindex(index)
    if values.isna().any():
        raise ValueError(f"{path}: seed/actor/state combinations are incomplete")
    utilities = values.to_numpy(float).reshape(len(seed_ids), len(actor_order), len(state_order))
    return seed_ids, utilities


def main() -> None:
    args = parse_args()
    state_ids, _, state_id_to_index = solver.load_state_tables(args.input_dir)
    neighbors = solver.load_neighbors(args.input_dir, len(state_ids), state_id_to_index)
    seed_ids, raw_utilities = load_utilities(args.utilities, state_ids)

    # These are the exact defaults used by solve_core_preference_model.py.
    config = solver.SolverConfig(
        input_dir=str(args.input_dir.resolve()),
        output_dir=str(args.output_dir.resolve()),
        theta=0.10,
        empty_neighbor_margin=1.0,
        margin_cap=10.0,
    )

    detail_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(seed_ids):
        standardized = solver.standardize_utilities(raw_utilities[seed_index])
        for actor_index, actor in enumerate(solver.COUNTRIES):
            for state_index, state_id in enumerate(state_ids):
                margins = solver.stability_margins(
                    state_index, actor_index, standardized, neighbors, config
                )
                for concept, margin in zip(CONCEPTS, margins):
                    detail_rows.append(
                        {
                            "seed": int(seed),
                            "actor": actor,
                            "state_id": int(state_id),
                            "stability_concept": concept,
                            "margin": float(margin),
                            "stable": bool(margin >= 0.0),
                        }
                    )

    details = pd.DataFrame(detail_rows)
    grouped = details.groupby(["actor", "state_id", "stability_concept"], sort=True)
    aggregate = grouped["margin"].agg(
        margin_mean="mean",
        margin_median="median",
        margin_q05=lambda x: float(np.quantile(x, 0.05)),
        margin_q95=lambda x: float(np.quantile(x, 0.95)),
        margin_min="min",
        margin_max="max",
    ).reset_index()
    rates = grouped["stable"].mean().rename("stable_rate").reset_index()
    aggregate = aggregate.merge(rates, on=["actor", "state_id", "stability_concept"])
    aggregate["majority_stable"] = aggregate["stable_rate"] >= 0.5

    joint = details.pivot_table(
        index=["seed", "state_id"], columns=["actor", "stability_concept"], values="stable", aggfunc="first"
    )
    joint_rows: list[dict[str, object]] = []
    for state_id in state_ids:
        row: dict[str, object] = {"state_id": int(state_id)}
        for concept in CONCEPTS:
            state_joint = joint.xs(int(state_id), level="state_id")
            cn = state_joint[("CN", concept)].to_numpy(bool)
            us = state_joint[("US", concept)].to_numpy(bool)
            both = cn & us
            row[f"joint_{concept}_stable_rate"] = float(np.mean(both))
            row[f"joint_{concept}_majority_stable"] = bool(np.mean(both) >= 0.5)
        joint_rows.append(row)
    joint_frame = pd.DataFrame(joint_rows)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "stability_by_seed_state.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    aggregate.to_csv(output_dir / "stability_margin_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10f")
    joint_frame.to_csv(output_dir / "joint_stability_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10f")

    summary = {
        "utilities": str(args.utilities.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "seed_count": int(len(seed_ids)),
        "state_count": int(len(state_ids)),
        "actors": list(solver.COUNTRIES),
        "concepts": list(CONCEPTS),
        "core_parameters": {
            "theta": 0.10,
            "empty_neighbor_margin": 1.0,
            "margin_cap": 10.0,
            "utility_standardization": "within seed and actor across all states",
        },
        "majority_stable_state_counts": {
            actor: {
                concept: int(
                    aggregate.loc[
                        (aggregate.actor == actor)
                        & (aggregate.stability_concept == concept),
                        "majority_stable",
                    ].sum()
                )
                for concept in CONCEPTS
            }
            for actor in solver.COUNTRIES
        },
        "joint_majority_stable_state_counts": {
            concept: int(joint_frame[f"joint_{concept}_majority_stable"].sum())
            for concept in CONCEPTS
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'stability_by_seed_state.csv'}")
    print(f"Wrote {output_dir / 'stability_margin_summary.csv'}")
    print(f"Wrote {output_dir / 'joint_stability_summary.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
