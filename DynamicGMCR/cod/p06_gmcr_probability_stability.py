from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONCEPTS = ("Nash", "GMR", "SMR", "SEQ")
ACTORS = ("CN", "US")
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent.parent
    default_matrix_dir = demo_dir / "output" / "preference_prediction"
    parser = argparse.ArgumentParser(
        description="Apply two-player GMCR Nash/GMR/SMR/SEQ tests to probability preference matrices."
    )
    parser.add_argument(
        "--cn-matrix",
        type=Path,
        default=default_matrix_dir / "predicted_preference_matrix_CN_202002.csv",
        help="CN probability preference matrix (rows are preferred state, columns reference state).",
    )
    parser.add_argument(
        "--us-matrix",
        type=Path,
        default=default_matrix_dir / "predicted_preference_matrix_US_202002.csv",
        help="US probability preference matrix.",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=demo_dir / "input" / "state_transition_graph.csv",
        help="GMCR unilateral transition graph CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_matrix_dir / "gmcr_stability_202002",
        help="Directory for generated CSV and JSON files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for a preference relation (default: 0.5).",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_probability_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = _read_csv(path)
    state_column = frame.columns[0]
    state_ids = frame[state_column].astype(int).to_numpy()

    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    values = np.clip(values, 0.0, 1.0)
    return state_ids, values


def load_neighbors(
    path: Path, state_ids: np.ndarray
) -> list[list[np.ndarray]]:
    graph = _read_csv(path)
    index = {int(state_id): i for i, state_id in enumerate(state_ids)}
    raw: list[list[list[int]]] = [[[] for _ in state_ids] for _ in ACTORS]
    actor_index = {actor: i for i, actor in enumerate(ACTORS)}
    for row in graph.itertuples(index=False):
        actor = str(getattr(row, "actor"))
        if actor not in actor_index:
            continue
        source = int(getattr(row, "from_state_id"))
        destination = int(getattr(row, "to_state_id"))
        if source == destination:
            continue
        raw[actor_index[actor]][index[source]].append(index[destination])

    return [
        [np.asarray(sorted(set(destinations)), dtype=int) for destinations in actor_raw]
        for actor_raw in raw
    ]


def _max_or_fallback(values: Iterable[float], fallback: float) -> float:
    values = list(values)
    return max(values) if values else fallback


def stability_margins(
    state: int,
    actor: int,
    probabilities: np.ndarray,
    neighbors: list[list[np.ndarray]],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    opponent = 1 - actor
    choices = neighbors[actor][state]
    gains = probabilities[actor, choices, state] - threshold
    improvements = choices[gains > EPS]
    nash = float(threshold if len(choices) == 0 else np.min(-gains))


    if len(choices) == 0:
        nash = threshold
    if len(improvements) == 0:
        return np.full(4, nash, dtype=float), improvements

    gmr_terms: list[float] = []
    smr_terms: list[float] = []
    seq_terms: list[float] = []
    for q in improvements:
        sanctions = neighbors[opponent][int(q)]
        if len(sanctions) == 0:
            gmr_terms.append(-threshold)
            smr_terms.append(-threshold)
            seq_terms.append(-threshold)
            continue


        block = probabilities[actor, state, sanctions] - threshold
        gmr_terms.append(float(np.max(block)))

        smr_by_sanction: list[float] = []
        seq_by_sanction: list[float] = []
        for k in sanctions:
            counter_states = np.concatenate(
                (np.asarray([k], dtype=int), neighbors[actor][int(k)])
            )
            counter_block = probabilities[actor, state, counter_states] - threshold
            smr_by_sanction.append(float(np.min(counter_block)))


            credibility = probabilities[opponent, int(k), int(q)] - threshold
            deterrence = float(probabilities[actor, state, int(k)] - threshold)
            seq_by_sanction.append(float(min(credibility, deterrence)))

        smr_terms.append(_max_or_fallback(smr_by_sanction, -threshold))
        seq_terms.append(_max_or_fallback(seq_by_sanction, -threshold))

    margins = np.asarray(
        [nash, min(gmr_terms), min(smr_terms), min(seq_terms)], dtype=float
    )
    return margins, improvements


def analyze(
    state_ids: np.ndarray,
    probabilities: np.ndarray,
    neighbors: list[list[np.ndarray]],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    detail_rows: list[dict[str, object]] = []
    compact: dict[int, dict[str, object]] = {int(state_id): {"state_id": int(state_id)} for state_id in state_ids}
    set_rows: list[dict[str, object]] = []

    for actor_index, actor in enumerate(ACTORS):
        for state_index, state_id in enumerate(state_ids):
            margins, improvements = stability_margins(
                state_index, actor_index, probabilities, neighbors, threshold
            )
            flags = margins >= -EPS
            row: dict[str, object] = {
                "state_id": int(state_id),
                "actor": actor,
                "improved_move_ids": "|".join(str(int(state_ids[i])) for i in improvements),
            }
            for concept, margin, stable in zip(CONCEPTS, margins, flags):
                row[f"{concept}_margin"] = float(margin)
                row[f"{concept}_stable"] = bool(stable)
                compact[int(state_id)][f"{actor}_{concept}_margin"] = float(margin)
                compact[int(state_id)][f"{actor}_{concept}_stable"] = bool(stable)
            detail_rows.append(row)

        for concept_index, concept in enumerate(CONCEPTS):
            stable_ids = [
                int(state_id)
                for state_id, row in compact.items()
                if bool(row[f"{actor}_{concept}_stable"])
            ]
            set_rows.append(
                {
                    "actor": actor,
                    "stability_concept": concept,
                    "stable_state_count": len(stable_ids),
                    "stable_state_ids": "|".join(map(str, stable_ids)),
                }
            )

    compact_rows: list[dict[str, object]] = []
    for state_id in state_ids:
        row = compact[int(state_id)]
        joint_types = [
            concept
            for concept in CONCEPTS
            if bool(row[f"CN_{concept}_stable"]) and bool(row[f"US_{concept}_stable"])
        ]
        row["joint_stability_types"] = "|".join(joint_types) if joint_types else "None"
        for concept in CONCEPTS:
            row[f"joint_{concept}_stable"] = concept in joint_types
        compact_rows.append(row)

    compact_frame = pd.DataFrame(compact_rows)
    detail_frame = pd.DataFrame(detail_rows)
    summary: dict[str, object] = {
        "threshold": threshold,
        "state_count": int(len(state_ids)),
        "actors": list(ACTORS),
        "concepts": list(CONCEPTS),
        "transition_edge_count": {
            actor: int(sum(len(destinations) for destinations in neighbors[i]))
            for i, actor in enumerate(ACTORS)
        },
        "stable_state_counts": {
            actor: {
                concept: int((detail_frame.loc[(detail_frame.actor == actor), f"{concept}_stable"]).sum())
                for concept in CONCEPTS
            }
            for actor in ACTORS
        },
        "joint_stable_state_counts": {
            concept: int(compact_frame[f"joint_{concept}_stable"].sum())
            for concept in CONCEPTS
        },
    }
    return detail_frame, compact_frame, {"sets": set_rows, "summary": summary}


def main() -> None:
    args = parse_args()
    threshold = float(args.threshold)

    cn_ids, cn = load_probability_matrix(args.cn_matrix.resolve())
    us_ids, us = load_probability_matrix(args.us_matrix.resolve())
    neighbors = load_neighbors(args.graph.resolve(), cn_ids)
    probabilities = np.stack((cn, us), axis=0)
    details, states, metadata = analyze(cn_ids, probabilities, neighbors, threshold)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "stability_details.csv"
    states_path = output_dir / "stability_by_state.csv"
    sets_path = output_dir / "stable_state_sets.csv"
    summary_path = output_dir / "summary.json"
    details.to_csv(details_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    states.to_csv(states_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    pd.DataFrame(metadata["sets"]).to_csv(sets_path, index=False, encoding="utf-8-sig")
    payload = {
        "cn_matrix": str(args.cn_matrix.resolve()),
        "us_matrix": str(args.us_matrix.resolve()),
        "graph": str(args.graph.resolve()),
        **metadata["summary"],
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {details_path}")
    print(f"Wrote {states_path}")
    print(f"Wrote {sets_path}")
    print(f"Wrote {summary_path}")
    print("\nJoint stable states:")
    for concept in CONCEPTS:
        ids = states.loc[states[f"joint_{concept}_stable"], "state_id"].astype(int).tolist()
        print(f"  {concept}: {ids if ids else '(none)'}")


if __name__ == "__main__":
    main()
