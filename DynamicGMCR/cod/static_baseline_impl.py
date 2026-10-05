#!/usr/bin/env python3
"""Static strategy-prioritization baseline for the DynamicGMCR GMCR state graph.

The baseline keeps the DynamicGMCR states, options, and transition graph fixed, but
replaces dynamic preference updating with GMCR-style static strategy priority
declarations for CN and US. It reports GMCR stability and the static model's
ability to explain the historical state sequence.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COUNTRIES = ("CN", "US")
OPTION_COLUMNS = (
    "U1[US]",
    "U2[US]",
    "U3[US]",
    "U4[US]",
    "C1[C0]",
    "C2[C0]",
    "C3[C0]",
    "C4[C0]",
)
OPTION_NAMES = ("U1", "U2", "U3", "U4", "C1", "C2", "C3", "C4")
OWNED_OPTION_INDICES = (
    np.arange(4, 8),  # CN controls C1-C4.
    np.arange(0, 4),  # US controls U1-U4.
)
EPS = 1e-10


@dataclass(frozen=True)
class BaselineConfig:
    input_dir: Path
    output_dir: Path
    improvement_tolerance: float = 1e-9
    path_score_floor: float = 1e-8


@dataclass(frozen=True)
class StrategyPriority:
    country: str
    priority: int
    statement_id: str
    statement: str
    required: tuple[tuple[str, int], ...]
    explanation: str


@dataclass
class BaselineData:
    state_ids: np.ndarray
    options: np.ndarray
    observed_months: np.ndarray
    observed_indices: np.ndarray
    neighbors: list[list[np.ndarray]]
    transition_types: np.ndarray
    state_tables: dict[str, pd.DataFrame]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    project = here.parent
    parser = argparse.ArgumentParser(
        description="Run a static strategy-prioritization GMCR baseline on DynamicGMCR inputs."
    )
    parser.add_argument("--input-dir", type=Path, default=project / "input")
    parser.add_argument("--output-dir", type=Path, default=project / "output" / "static_baseline")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def make_timestamped_dir(parent: Path) -> Path:
    timestamp = datetime.now().strftime("%d-%H-%M")
    candidate = parent / timestamp
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    suffix = 1
    while True:
        candidate = parent / f"{timestamp}_{suffix:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def load_transition_graph(
    path: Path,
    state_ids: np.ndarray,
    options: np.ndarray,
) -> list[list[np.ndarray]]:
    frame = read_csv(path)
    required = {"actor", "from_state_id", "to_state_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Transition graph is missing {required - set(frame.columns)}")
    state_to_index = {int(state_id): idx for idx, state_id in enumerate(state_ids)}
    result_sets: list[list[set[int]]] = [
        [set() for _ in range(len(state_ids))],
        [set() for _ in range(len(state_ids))],
    ]
    for row in frame.itertuples(index=False):
        actor_label = str(getattr(row, "actor")).upper()
        if actor_label not in COUNTRIES:
            raise ValueError(f"Unknown actor in transition graph: {actor_label}")
        actor = 0 if actor_label == "CN" else 1
        start_id = int(getattr(row, "from_state_id"))
        dest_id = int(getattr(row, "to_state_id"))
        if start_id not in state_to_index or dest_id not in state_to_index:
            raise ValueError(f"Unknown transition state: {start_id}->{dest_id}")
        start = state_to_index[start_id]
        destination = state_to_index[dest_id]
        changed = options[start] != options[destination]
        owned = OWNED_OPTION_INDICES[actor]
        opponent = np.setdiff1d(np.arange(options.shape[1]), owned)
        if not np.any(changed[owned]) or np.any(changed[opponent]):
            raise ValueError(f"Transition {start_id}->{dest_id} is not unilateral.")
        result_sets[actor][start].add(destination)
    return [
        [np.array(sorted(destinations), dtype=int) for destinations in actor_sets]
        for actor_sets in result_sets
    ]


def build_transition_types(options: np.ndarray) -> np.ndarray:
    changed = options[:, None, :] != options[None, :, :]
    us_changed = np.any(changed[:, :, :4], axis=2)
    cn_changed = np.any(changed[:, :, 4:8], axis=2)
    return cn_changed.astype(np.int8) + 2 * us_changed.astype(np.int8)


def load_data(input_dir: Path) -> BaselineData:
    cn_table = read_csv(input_dir / "state_table_cn.csv")
    us_table = read_csv(input_dir / "state_table_us.csv")
    observed = read_csv(input_dir / "monthly_key_states.csv")
    if len(cn_table) != len(us_table):
        raise ValueError("CN and US state tables must have the same length.")
    state_ids = cn_table.iloc[:, 0].astype(int).to_numpy()
    if not np.array_equal(state_ids, us_table.iloc[:, 0].astype(int).to_numpy()):
        raise ValueError("CN and US state tables do not use the same state order.")
    options = cn_table.loc[:, OPTION_COLUMNS].astype(int).to_numpy()
    if not np.array_equal(options, us_table.loc[:, OPTION_COLUMNS].astype(int).to_numpy()):
        raise ValueError("CN and US option configurations differ.")
    if not np.isin(options, [0, 1]).all():
        raise ValueError("Options must be binary.")
    state_to_index = {int(state_id): idx for idx, state_id in enumerate(state_ids)}
    observed["month"] = observed["month"].astype(int)
    observed["state_id"] = observed["state_id"].astype(int)
    unknown = set(observed["state_id"]) - set(state_to_index)
    if unknown:
        raise ValueError(f"Observed states not in state table: {sorted(unknown)}")
    observed_indices = np.array(
        [state_to_index[int(state_id)] for state_id in observed["state_id"]],
        dtype=int,
    )
    neighbors = load_transition_graph(input_dir / "state_transition_graph.csv", state_ids, options)
    return BaselineData(
        state_ids=state_ids,
        options=options,
        observed_months=observed["month"].to_numpy(dtype=int),
        observed_indices=observed_indices,
        neighbors=neighbors,
        transition_types=build_transition_types(options),
        state_tables={"CN": cn_table, "US": us_table},
    )


def default_strategy_priorities() -> tuple[StrategyPriority, ...]:
    """Return static GMCR option-prioritization statements.

    Each statement is a desired option pattern. States are compared
    lexicographically by satisfaction of these statements: priority 1 dominates
    all lower priorities, priority 2 breaks ties, and so on.
    """
    return (
        StrategyPriority(
            "CN",
            1,
            "CN-P1",
            "U1 = 0",
            (("U1", 0),),
            "中国首先偏好美国停止或不启动关税升级，因为 U1 直接增加出口成本、市场不确定性和被强制谈判压力。",
        ),
        StrategyPriority(
            "CN",
            2,
            "CN-P2",
            "U4 = 1 and C4 = 1",
            (("U4", 1), ("C4", 1)),
            "若进入阶段性协议，中国偏好双方都接受书面、可核验承诺，以把谈判成果制度化并降低反复升级风险。",
        ),
        StrategyPriority(
            "CN",
            3,
            "CN-P3",
            "U1 = 1 and C1 = 1",
            (("U1", 1), ("C1", 1)),
            "当美国升级关税施压已经存在时，中国偏好实施对等反制，以提高美国单边施压成本并维护战略自主。",
        ),
        StrategyPriority(
            "CN",
            4,
            "CN-P4",
            "C3 = 1",
            (("C3", 1),),
            "中国偏好保持正式谈判通道，因为制度化沟通有助于控制冲突烈度并保留议价空间。",
        ),
        StrategyPriority(
            "CN",
            5,
            "CN-P5",
            "U2 = 1",
            (("U2", 1),),
            "中国偏好美国释放关税缓和信号，因为这表明冲突可控，也为让步、谈判和协议安排提供政治空间。",
        ),
        StrategyPriority(
            "CN",
            6,
            "CN-P6",
            "U3 = 1 and C2 = 1",
            (("U3", 1), ("C2", 1)),
            "中国只在正式谈判存在时偏好提供可量化经贸让步，避免在缺乏谈判框架时单方面让步。",
        ),
        StrategyPriority(
            "CN",
            7,
            "CN-P7",
            "U1 = 0 and C1 = 0",
            (("U1", 0), ("C1", 0)),
            "在美国关税压力消失时，中国偏好不继续反制，以减少报复循环并恢复经贸稳定。",
        ),
        StrategyPriority(
            "CN",
            8,
            "CN-P8",
            "C4 = 1",
            (("C4", 1),),
            "在较低优先级上，中国偏好接受书面可核验承诺，因为它提高协议可信度，但该偏好低于避免被施压和保留反制能力。",
        ),
        StrategyPriority(
            "US",
            1,
            "US-P1",
            "C2 = 1 and C4 = 1",
            (("C2", 1), ("C4", 1)),
            "美国首先偏好中国同时提供可量化经贸让步和书面可核验承诺，因为这是结构性施压最可展示的政策成果。",
        ),
        StrategyPriority(
            "US",
            2,
            "US-P2",
            "U4 = 1",
            (("U4", 1),),
            "在获得成果基础上，美国偏好接受部分协议路线，以形成可执行的阶段性成果并降低市场不确定性。",
        ),
        StrategyPriority(
            "US",
            3,
            "US-P3",
            "C2 = 0 and U1 = 1",
            (("C2", 0), ("U1", 1)),
            "若中国尚未提供可量化让步，美国偏好保留或升级关税施压，把关税作为强制性谈判工具。",
        ),
        StrategyPriority(
            "US",
            4,
            "US-P4",
            "U3 = 1",
            (("U3", 1),),
            "美国偏好保持正式谈判通道，以保留议程控制、危机管理机制和继续施压后的交易出口。",
        ),
        StrategyPriority(
            "US",
            5,
            "US-P5",
            "C1 = 0",
            (("C1", 0),),
            "美国偏好中国不进行对等反制，因为 C1 会提高美国企业、农业和消费者承受的冲突成本。",
        ),
        StrategyPriority(
            "US",
            6,
            "US-P6",
            "U2 = 1",
            (("U2", 1),),
            "在较低优先级上，美国偏好释放关税缓和信号，用于管理市场预期并显示冲突仍可控。",
        ),
        StrategyPriority(
            "US",
            7,
            "US-P7",
            "C3 = 1",
            (("C3", 1),),
            "美国偏好中国保持正式谈判通道，因为这降低误判风险，也使美国议程控制和施压收益更容易转化为协议。",
        ),
        StrategyPriority(
            "US",
            8,
            "US-P8",
            "U1 = 0 and U2 = 1",
            (("U1", 0), ("U2", 1)),
            "若已进入缓和阶段，美国偏好不再同时保持升级关税姿态，以避免缓和信号与强制施压相互抵消。",
        ),
    )


def priority_satisfaction(
    options: np.ndarray,
    priorities: tuple[StrategyPriority, ...],
) -> dict[str, np.ndarray]:
    option_index = {option: idx for idx, option in enumerate(OPTION_NAMES)}
    result: dict[str, np.ndarray] = {}
    for country in COUNTRIES:
        country_priorities = [p for p in priorities if p.country == country]
        matrix = np.zeros((len(options), len(country_priorities)), dtype=int)
        for col, priority in enumerate(country_priorities):
            satisfied = np.ones(len(options), dtype=bool)
            for option, desired in priority.required:
                satisfied &= options[:, option_index[option]] == int(desired)
            matrix[:, col] = satisfied.astype(int)
        result[country] = matrix
    return result


def compute_utilities(
    options: np.ndarray,
    priorities: tuple[StrategyPriority, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert lexicographic priority satisfaction to ordinal utility scores."""
    satisfaction = priority_satisfaction(options, priorities)
    utilities = np.zeros((len(COUNTRIES), len(options)), dtype=float)
    for country_idx, country in enumerate(COUNTRIES):
        matrix = satisfaction[country]
        # Binary powers preserve lexicographic priority ordering for boolean
        # statements and provide a numerical score for GMCR improvement tests.
        powers = 2.0 ** np.arange(matrix.shape[1] - 1, -1, -1)
        utilities[country_idx] = matrix @ powers
    return utilities, satisfaction


def player_stability(
    state: int,
    player: int,
    utilities: np.ndarray,
    neighbors: list[list[np.ndarray]],
    tolerance: float,
) -> tuple[bool, bool, bool, bool]:
    opponent = 1 - player
    baseline = utilities[player, state]
    moves = neighbors[player][state]
    ui = moves[utilities[player, moves] > baseline + tolerance]
    if len(ui) == 0:
        return True, True, True, True

    gmr_ok = True
    smr_ok = True
    seq_ok = True
    for improved in ui:
        sanctions = neighbors[opponent][int(improved)]
        harmful = sanctions[utilities[player, sanctions] <= baseline + tolerance]
        if len(harmful) == 0:
            gmr_ok = False
            smr_ok = False
        else:
            protected = False
            for sanctioned in harmful:
                counters = neighbors[player][int(sanctioned)]
                if not np.any(utilities[player, counters] > baseline + tolerance):
                    protected = True
                    break
            if not protected:
                smr_ok = False

        opponent_improvements = sanctions[
            utilities[opponent, sanctions]
            > utilities[opponent, int(improved)] + tolerance
        ]
        seq_sanctions = opponent_improvements[
            utilities[player, opponent_improvements] <= baseline + tolerance
        ]
        if len(seq_sanctions) == 0:
            seq_ok = False
    return False, gmr_ok, smr_ok, seq_ok


def build_stability_table(
    data: BaselineData,
    utilities: np.ndarray,
    tolerance: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    flags = np.zeros((2, len(data.state_ids), 4), dtype=bool)
    for player in range(2):
        for state in range(len(data.state_ids)):
            flags[player, state] = player_stability(
                state, player, utilities, data.neighbors, tolerance
            )
    rows: list[dict[str, Any]] = []
    for state in range(len(data.state_ids)):
        row: dict[str, Any] = {
            "state_id": int(data.state_ids[state]),
            "cn_utility": float(utilities[0, state]),
            "us_utility": float(utilities[1, state]),
            "joint_stability_count": int(flags[:, state, :].all(axis=0).sum()),
        }
        for player_idx, country in enumerate(COUNTRIES):
            row[f"{country.lower()}_stable_any"] = bool(flags[player_idx, state].any())
            for concept_idx, concept in enumerate(("nash", "gmr", "smr", "seq")):
                row[f"{country.lower()}_{concept}"] = bool(
                    flags[player_idx, state, concept_idx]
                )
        for concept_idx, concept in enumerate(("nash", "gmr", "smr", "seq")):
            row[f"joint_{concept}"] = bool(flags[:, state, concept_idx].all())
        rows.append(row)
    return pd.DataFrame(rows), flags


def transition_matrix(
    utilities: np.ndarray,
    neighbors: list[list[np.ndarray]],
    tolerance: float,
) -> np.ndarray:
    n_states = utilities.shape[1]
    matrix = np.zeros((2, n_states, n_states), dtype=float)
    for actor in range(2):
        for state in range(n_states):
            choices = neighbors[actor][state]
            gains = utilities[actor, choices] - utilities[actor, state]
            improving = gains > tolerance
            total_gain = float(gains[improving].sum())
            if total_gain <= EPS:
                continue
            matrix[actor, state, choices[improving]] = gains[improving] / total_gain
    return matrix


def state_transition_matrix(
    transition: np.ndarray,
    transition_types: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered_joint = np.stack(
        [transition[0] @ transition[1], transition[1] @ transition[0]], axis=0
    )
    joint_average = ordered_joint.mean(axis=0)
    raw = np.zeros_like(joint_average)
    raw[transition_types == 1] = transition[0][transition_types == 1]
    raw[transition_types == 2] = transition[1][transition_types == 2]
    raw[transition_types == 3] = joint_average[transition_types == 3]
    np.fill_diagonal(raw, 0.0)
    row_totals = raw.sum(axis=1)
    probability = np.divide(
        raw,
        row_totals[:, None],
        out=np.zeros_like(raw),
        where=row_totals[:, None] > EPS,
    )
    return probability, raw, ordered_joint, row_totals


def supported_rank(scores: np.ndarray, target: int, score_floor: float) -> int:
    target_score = float(scores[int(target)])
    if target_score <= score_floor:
        return int(len(scores) + 1)
    return int(np.sum(scores > target_score + EPS)) + 1


def observed_path_bottleneck(
    utilities: np.ndarray,
    neighbors: list[list[np.ndarray]],
    transition_types: np.ndarray,
    start: int,
    destination: int,
) -> float:
    transition_type = int(transition_types[start, destination])
    if transition_type in (1, 2):
        actor = transition_type - 1
        if destination not in neighbors[actor][start]:
            return -1.0
        return float(utilities[actor, destination] - utilities[actor, start])
    if transition_type != 3:
        return -1.0
    best = -math.inf
    for first, second in ((0, 1), (1, 0)):
        for middle_value in neighbors[first][start]:
            middle = int(middle_value)
            if destination not in neighbors[second][middle]:
                continue
            first_gain = float(utilities[first, middle] - utilities[first, start])
            second_gain = float(utilities[second, destination] - utilities[second, middle])
            best = max(best, min(first_gain, second_gain))
    return best if math.isfinite(best) else -1.0


def explain_history(
    data: BaselineData,
    utilities: np.ndarray,
    flags: np.ndarray,
    config: BaselineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    transition = transition_matrix(utilities, data.neighbors, config.improvement_tolerance)
    probabilities, raw_scores, ordered_joint, row_totals = state_transition_matrix(
        transition, data.transition_types
    )
    rows: list[dict[str, Any]] = []
    transition_scores: list[float] = []
    for pos in range(1, len(data.observed_months)):
        start = int(data.observed_indices[pos - 1])
        observed = int(data.observed_indices[pos])
        changed = start != observed
        stay_cn = bool(flags[0, start].any())
        stay_us = bool(flags[1, start].any())
        stay_ok = stay_cn and stay_us
        transition_score = float(probabilities[start, observed]) if changed else math.nan
        if changed:
            transition_scores.append(transition_score)
        path_scores = probabilities[start].copy()
        combined_scores = path_scores.copy()
        combined_scores[start] = 1.0 if not changed else 0.0
        full_order = np.argsort(-combined_scores, kind="stable")
        destination_order = np.argsort(-path_scores, kind="stable")
        transition_type = int(data.transition_types[start, observed])
        if transition_type == 1:
            mechanism = "CN"
        elif transition_type == 2:
            mechanism = "US"
        elif transition_type == 3:
            mechanism = "CN-US|US-CN"
        elif changed:
            mechanism = "none"
        else:
            mechanism = "stay"
        immediate_support = bool(changed and transition_score > config.path_score_floor)
        explanation = (
            "stable_status_quo"
            if not changed and stay_ok
            else "unstable_status_quo"
            if not changed
            else "immediate_utility_support"
            if immediate_support
            else "unsupported_transition"
        )
        row: dict[str, Any] = {
            "month": int(data.observed_months[pos]),
            "start_state_id": int(data.state_ids[start]),
            "observed_state_id": int(data.state_ids[observed]),
            "observed_change": int(changed),
            "transition_explanation": explanation,
            "observed_transition_probability": transition_score,
            "observed_raw_transition_score": (
                float(raw_scores[start, observed]) if changed else math.nan
            ),
            "observed_rank": (
                supported_rank(combined_scores, observed, config.path_score_floor)
                if changed
                else 1
            ),
            "conditional_observed_rank": (
                supported_rank(path_scores, observed, config.path_score_floor)
                if changed
                else math.nan
            ),
            "predicted_state_id": int(data.state_ids[int(full_order[0])]),
            "predicted_score": float(combined_scores[int(full_order[0])]),
            "conditional_predicted_state_id": int(data.state_ids[int(destination_order[0])]),
            "conditional_predicted_score": float(path_scores[int(destination_order[0])]),
            "transition_mechanism": mechanism,
            "pre_normalization_probability_sum": float(row_totals[start]),
            "observed_path_bottleneck": (
                observed_path_bottleneck(
                    utilities,
                    data.neighbors,
                    data.transition_types,
                    start,
                    observed,
                )
                if changed
                else math.nan
            ),
            "stay_cn_stable_any": stay_cn if not changed else math.nan,
            "stay_us_stable_any": stay_us if not changed else math.nan,
            "stay_constraint_satisfied": stay_ok if not changed else math.nan,
            "cn_utility_start": float(utilities[0, start]),
            "us_utility_start": float(utilities[1, start]),
            "cn_utility_observed": float(utilities[0, observed]),
            "us_utility_observed": float(utilities[1, observed]),
        }
        if changed and transition_type == 3:
            row["cn_then_us_raw_component"] = float(ordered_joint[0, start, observed])
            row["us_then_cn_raw_component"] = float(ordered_joint[1, start, observed])
        else:
            row["cn_then_us_raw_component"] = 0.0
            row["us_then_cn_raw_component"] = 0.0
        rows.append(row)

    frame = pd.DataFrame(rows)
    changed = frame[frame["observed_change"] == 1]
    unchanged = frame[frame["observed_change"] == 0]
    positive = changed["observed_transition_probability"] > config.path_score_floor
    summary = {
        "model": "static_strategy_priority_gmcr_baseline",
        "input_source": str(config.input_dir),
        "preference_method": "GMCR strategy priority declarations",
        "months": int(len(data.observed_months)),
        "transitions": int(len(data.observed_months) - 1),
        "feasible_states": int(len(data.state_ids)),
        "cn_edges": int(sum(len(x) for x in data.neighbors[0])),
        "us_edges": int(sum(len(x) for x in data.neighbors[1])),
        "changed_months": int(len(changed)),
        "unchanged_months": int(len(unchanged)),
        "unchanged_stability_satisfied": int(
            unchanged["stay_constraint_satisfied"].eq(True).sum()
        ),
        "unchanged_stability_violation_count": int(
            unchanged["stay_constraint_satisfied"].eq(False).sum()
        ),
        "unchanged_stability_satisfaction_rate": float(
            unchanged["stay_constraint_satisfied"].eq(True).mean()
        )
        if len(unchanged)
        else math.nan,
        "changed_positive_probability_count": int(positive.sum()),
        "changed_zero_probability_count": int((~positive).sum()),
        "changed_positive_probability_rate": float(positive.mean())
        if len(changed)
        else math.nan,
        "changed_probability_mean": float(
            changed["observed_transition_probability"].mean()
        )
        if len(changed)
        else math.nan,
        "changed_log_probability_sum": float(
            np.log(
                np.maximum(
                    changed["observed_transition_probability"].to_numpy(dtype=float),
                    config.path_score_floor,
                )
            ).sum()
        )
        if len(changed)
        else 0.0,
        "conditional_change_top1_accuracy": float(
            (changed["conditional_observed_rank"] <= 1).mean()
        )
        if len(changed)
        else math.nan,
        "conditional_change_top3_accuracy": float(
            (changed["conditional_observed_rank"] <= 3).mean()
        )
        if len(changed)
        else math.nan,
        "conditional_change_top5_accuracy": float(
            (changed["conditional_observed_rank"] <= 5).mean()
        )
        if len(changed)
        else math.nan,
        "conditional_change_mean_rank": float(
            changed["conditional_observed_rank"].mean()
        )
        if len(changed)
        else math.nan,
        "explanation_counts": {
            str(key): int(value)
            for key, value in frame["transition_explanation"].value_counts().items()
        },
        "interpretation": (
            "Static strategy priority declarations are fixed for the full sample. "
            "No monthly event shocks, dynamic preference states, option-weight "
            "estimation, or optimized parameters are used."
        ),
    }
    return frame, summary


def write_outputs(
    data: BaselineData,
    config: BaselineConfig,
    priorities: tuple[StrategyPriority, ...],
    satisfaction: dict[str, np.ndarray],
    utilities: np.ndarray,
    stability: pd.DataFrame,
    history: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    priority_rows = []
    for priority in priorities:
        priority_rows.append(
            {
                "country": priority.country,
                "priority": int(priority.priority),
                "statement_id": priority.statement_id,
                "statement": priority.statement,
                "required_literals": ";".join(
                    f"{option}={desired}" for option, desired in priority.required
                ),
                "realistic_explanation": priority.explanation,
            }
        )
    pd.DataFrame(priority_rows).to_csv(
        config.output_dir / "strategy_priority_declarations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    satisfaction_rows = []
    for state in range(len(data.state_ids)):
        row: dict[str, Any] = {"state_id": int(data.state_ids[state])}
        for option_idx, option in enumerate(OPTION_NAMES):
            row[option] = int(data.options[state, option_idx])
        for country in COUNTRIES:
            country_priorities = [p for p in priorities if p.country == country]
            for idx, priority in enumerate(country_priorities):
                row[f"{priority.statement_id}_satisfied"] = int(
                    satisfaction[country][state, idx]
                )
        satisfaction_rows.append(row)
    pd.DataFrame(satisfaction_rows).to_csv(
        config.output_dir / "state_priority_satisfaction.csv",
        index=False,
        encoding="utf-8-sig",
    )

    utility_rows = []
    cn_order = np.argsort(-utilities[0], kind="stable")
    us_order = np.argsort(-utilities[1], kind="stable")
    cn_rank = np.empty(len(data.state_ids), dtype=int)
    us_rank = np.empty(len(data.state_ids), dtype=int)
    cn_rank[cn_order] = np.arange(1, len(data.state_ids) + 1)
    us_rank[us_order] = np.arange(1, len(data.state_ids) + 1)
    for state in range(len(data.state_ids)):
        row: dict[str, Any] = {
            "state_id": int(data.state_ids[state]),
            "cn_utility": float(utilities[0, state]),
            "us_utility": float(utilities[1, state]),
            "cn_rank": int(cn_rank[state]),
            "us_rank": int(us_rank[state]),
            "cn_priority_vector": "".join(str(x) for x in satisfaction["CN"][state]),
            "us_priority_vector": "".join(str(x) for x in satisfaction["US"][state]),
        }
        for option_idx, option in enumerate(OPTION_NAMES):
            row[option] = int(data.options[state, option_idx])
        utility_rows.append(row)
    pd.DataFrame(utility_rows).to_csv(
        config.output_dir / "static_state_utilities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stability.to_csv(
        config.output_dir / "static_state_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    history.to_csv(
        config.output_dir / "historical_explanation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    document = dict(summary)
    document["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    (config.output_dir / "static_baseline_summary.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_baseline(config: BaselineConfig) -> dict[str, Any]:
    data = load_data(config.input_dir.resolve())
    priorities = default_strategy_priorities()
    utilities, satisfaction = compute_utilities(data.options, priorities)
    stability, flags = build_stability_table(
        data, utilities, config.improvement_tolerance
    )
    history, summary = explain_history(data, utilities, flags, config)
    write_outputs(
        data,
        config,
        priorities,
        satisfaction,
        utilities,
        stability,
        history,
        summary,
    )
    return summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.no_timestamp else make_timestamped_dir(args.output_dir)
    config = BaselineConfig(
        input_dir=args.input_dir,
        output_dir=output_dir,
    )
    summary = run_baseline(config)
    if not args.quiet:
        print(
            "Static baseline completed; "
            f"stay_satisfied={summary['unchanged_stability_satisfied']}/"
            f"{summary['unchanged_months']}; "
            f"changed_positive={summary['changed_positive_probability_count']}/"
            f"{summary['changed_months']}; "
            f"output={config.output_dir.resolve()}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
