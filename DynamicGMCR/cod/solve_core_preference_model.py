"""Estimate dynamic preferences for subsequent GMCR analysis.

Historical state evolution calibrates the 42-dimensional preference process.  The
model does not directly forecast whether a state will persist or which transition
will occur.  Future-state assessment is performed separately by applying GMCR to
utilities computed from forecast preferences.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COUNTRIES = ("CN", "US")
DOMAINS = (
    "diplomacy",
    "economy",
    "law",
    "politics",
    "security",
    "society_humanitarian",
)
CN_DRIVERS = (
    "D1中国经济福利与市场准入",
    "D2中国战略自主与体系韧性",
    "D3中国国内政治合法性",
    "D4中国冲突升级与风险控制",
    "D5中国谈判杠杆与达约收益",
)
US_DRIVERS = (
    "D1美国经济福利与企业成本",
    "D2美国结构性议题施压收益",
    "D3美国国内政治支持",
    "D4美国政策可信度与冲突控制",
    "D5美国谈判控制与达约收益",
)
DRIVERS_BY_ACTOR = (CN_DRIVERS, US_DRIVERS)
N_ACTORS = 2
N_DOMAINS = 6
N_DRIVERS = 5
EPS = 1e-12
LARGE_NEGATIVE = -1.0e9


# Sign table for A_i(d, k). 1 => >=0, -1 => <=0, 0 => fixed zero,
# 9 => free. Rows follow DOMAINS; columns follow actor-specific D1-D5.
CN_A_SIGNS = np.array(
    [
        [1, 1, 9, 1, 1],    # diplomacy
        [1, 1, 1, 9, 1],    # economy
        [-1, 1, 1, 1, 1],   # law
        [9, 1, 1, 1, 1],    # politics
        [-1, 1, 1, 1, 1],   # security
        [9, 9, 1, 1, 9],    # society_humanitarian
    ],
    dtype=int,
)
US_A_SIGNS = np.array(
    [
        [1, 9, 9, 1, 1],    # diplomacy
        [1, 1, 1, 9, 1],    # economy
        [-1, 1, 1, 1, 1],   # law
        [9, 1, 1, 1, 1],    # politics
        [-1, 1, 1, 1, 1],   # security
        [9, 1, 1, 1, 9],    # society_humanitarian
    ],
    dtype=int,
)
A_SIGNS = np.stack([CN_A_SIGNS, US_A_SIGNS], axis=0)

# Theory-prespecified direct event-to-driver connections. The topology is fixed
# before model fitting and is never selected from the historical state sequence.
# Rows follow DOMAINS and columns follow actor-specific D1-D5.
THEORETICAL_A_MASK = np.array(
    [
        [1, 0, 0, 1, 1],  # diplomacy -> welfare, risk control, negotiation
        [1, 1, 1, 0, 0],  # economy -> welfare, strategic, domestic politics
        [1, 1, 0, 1, 0],  # law -> welfare, strategic, risk control
        [0, 1, 1, 0, 0],  # politics -> strategic, domestic politics
        [1, 1, 0, 1, 0],  # security -> welfare, strategic, risk control
        [0, 0, 1, 0, 0],  # society/humanitarian -> domestic politics
    ],
    dtype=bool,
)
A_ACTIVE_MASK = np.stack(
    [THEORETICAL_A_MASK.copy(), THEORETICAL_A_MASK.copy()], axis=0
)
N_ACTIVE_A_PARAMETERS = int(np.sum(A_ACTIVE_MASK))
ALPHA_SLICE = slice(N_ACTIVE_A_PARAMETERS, N_ACTIVE_A_PARAMETERS + N_ACTORS)
MU_SLICE = slice(ALPHA_SLICE.stop, ALPHA_SLICE.stop + N_ACTORS * N_DRIVERS)
PARAMETER_DIM = int(MU_SLICE.stop)


@dataclass(frozen=True)
class SolverConfig:
    input_dir: str
    output_dir: str
    seed: int = 20250728    
    population_size: int = 160
    generations: int = 100
    differential_weight: float = 0.65
    crossover_rate: float = 0.85
    local_search_steps: int = 500
    local_search_scale: float = 0.08
    progress_interval: int = 25
    a_abs_bound: float = 5.0
    mu_abs_bound: float = 2.0
    alpha_upper: float = 0.98
    delta: float = 0.05
    delta_persistence: float = 0.90
    perturbation_seed: int = 20260701
    theta: float = 0.10
    transition_temperature: float = 0.35
    theta_forward: float = 0.05
    stable_margin: float = 0.02
    epsilon_p: float = 1.0e-8
    forward_horizon: int = 4
    lambda_forward: float = 1.0
    lambda_stable: float = 0.25
    nonlinear_saturation: bool = True
    lambda_a: float = 0.01
    lambda_mu: float = 0.01
    lambda_alpha: float = 0.01
    burn_in_months: int = 5
    margin_cap: float = 10.0
    empty_neighbor_margin: float = 1.0
    de_early_stop_patience: int = 0
    local_search_early_stop_patience: int = 0


@dataclass(frozen=True)
class EvaluationPlan:
    month_index: int
    start: int
    observed: int
    changed: bool
    direct_actors: tuple[int, ...] = ()
    ordered_paths: tuple[tuple[int, int, tuple[int, ...]], ...] = ()
    forward_candidates: tuple[int, ...] = ()


@dataclass
class ModelData:
    months: np.ndarray
    state_ids: np.ndarray
    observed_indices: np.ndarray
    impacts_raw: np.ndarray
    impacts_scaled: np.ndarray
    omega: np.ndarray
    neighbors: list[list[np.ndarray]]
    state_id_to_index: dict[int, int]
    score_perturbations: np.ndarray | None = None
    evaluation_plans: tuple[EvaluationPlan, ...] | None = None
    lagged_impacts: np.ndarray | None = None
    evaluation_forward_horizon: int | None = None


@dataclass
class Parameters:
    a: np.ndarray
    alpha: np.ndarray
    mu: np.ndarray
    vector: np.ndarray


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    demo_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the DynamicGMCR 42-dimensional sparse preference process for subsequent "
            "utility-based GMCR analysis."
        )
    )
    parser.add_argument("--input-dir", default=str(demo_dir / "input"))
    parser.add_argument("--output-dir", default=str(demo_dir / "output"))
    parser.add_argument(
        "--preset",
        choices=("quick", "standard", "full", "two_hour", "deep"),
        default="two_hour",
        help=(
            "Use quick for a short pipeline check, standard for moderate solving, "
            "full for longer optimization, two_hour for the default large run, "
            "and deep for an extended search."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--population-size", type=int, default=60)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument(
        "--local-search-steps",
        type=int,
        default=500,
        help="Number of local-search proposals after differential evolution.",
    )
    parser.add_argument(
        "--de-early-stop-patience",
        type=int,
        default=0,
        help=(
            "Stop differential evolution after this many generations without a new "
            "best loss; 0 disables early stopping."
        ),
    )
    parser.add_argument(
        "--local-search-early-stop-patience",
        type=int,
        default=0,
        help=(
            "Stop local search after this many proposals without a new best loss; "
            "0 disables early stopping."
        ),
    )
    parser.add_argument(
        "--a-abs-bound",
        type=float,
        default=5.0,
        help="Absolute bound for active event-to-driver mapping coefficients.",
    )
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument(
        "--delta-persistence",
        type=float,
        default=0.90,
        help="AR(1) persistence of the ex ante state-score perturbation process.",
    )
    parser.add_argument(
        "--perturbation-seed",
        type=int,
        default=20260701,
        help="Seed for score perturbations; independent of the optimizer seed.",
    )
    parser.add_argument("--theta", type=float, default=0.10)
    parser.add_argument(
        "--transition-temperature",
        type=float,
        default=0.35,
        help="Softmax temperature for conditional transition-preference compatibility.",
    )
    parser.add_argument("--theta-forward", type=float, default=0.05)
    parser.add_argument("--stable-margin", type=float, default=0.02)
    parser.add_argument("--lambda-forward", type=float, default=1.0)
    parser.add_argument("--lambda-stable", type=float, default=0.25)
    parser.add_argument("--lambda-a", type=float, default=0.01)
    parser.add_argument("--lambda-mu", type=float, default=0.01)
    parser.add_argument("--lambda-alpha", type=float, default=0.01)
    parser.add_argument(
        "--burn-in-months",
        type=int,
        default=5,
        help="Initial months used only to warm up the dynamic preference state.",
    )
    parser.add_argument("--progress-interval", type=int, default=5)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> SolverConfig:
    if args.preset == "quick":
        population_size = 24
        generations = 20
        local_search_steps = 20
        progress_interval = 2
    elif args.preset == "full":
        population_size = 180
        generations = 250
        local_search_steps = 500
        progress_interval = 25
    elif args.preset == "two_hour":
        population_size = 320
        generations = 800
        local_search_steps = 2000
        progress_interval = 25
    elif args.preset == "deep":
        population_size = 420
        generations = 1200
        local_search_steps = 4000
        progress_interval = 50
    else:
        population_size = args.population_size
        generations = args.generations
        local_search_steps = args.local_search_steps
        progress_interval = args.progress_interval

    return SolverConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        population_size=population_size,
        generations=generations,
        local_search_steps=local_search_steps,
        a_abs_bound=max(0.0, float(args.a_abs_bound)),
        delta=max(0.0, args.delta),
        delta_persistence=min(max(float(args.delta_persistence), 0.0), 1.0),
        perturbation_seed=int(args.perturbation_seed),
        theta=max(0.0, args.theta),
        transition_temperature=max(float(args.transition_temperature), EPS),
        theta_forward=max(0.0, args.theta_forward),
        stable_margin=max(0.0, args.stable_margin),
        lambda_forward=max(0.0, args.lambda_forward),
        lambda_stable=max(0.0, args.lambda_stable),
        lambda_a=max(0.0, args.lambda_a),
        lambda_mu=max(0.0, args.lambda_mu),
        lambda_alpha=max(0.0, args.lambda_alpha),
        burn_in_months=max(0, int(args.burn_in_months)),
        progress_interval=max(1, progress_interval),
        de_early_stop_patience=max(0, int(args.de_early_stop_patience)),
        local_search_early_stop_patience=max(
            0, int(args.local_search_early_stop_patience)
        ),
    )


def log_progress(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def load_state_tables(input_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    cn = read_csv(input_dir / "state_table_cn.csv")
    us = read_csv(input_dir / "state_table_us.csv")
    cn = cn.sort_values("状态编号").reset_index(drop=True)
    us = us.sort_values("状态编号").reset_index(drop=True)

    cn_ids = cn["状态编号"].astype(int).to_numpy()
    us_ids = us["状态编号"].astype(int).to_numpy()
    if not np.array_equal(cn_ids, us_ids):
        raise ValueError("CN and US state tables must contain the same ordered state IDs.")

    omega = np.zeros((N_ACTORS, len(cn_ids), N_DRIVERS), dtype=float)
    omega[0] = cn.loc[:, CN_DRIVERS].astype(float).to_numpy()
    omega[1] = us.loc[:, US_DRIVERS].astype(float).to_numpy()
    state_id_to_index = {int(state_id): idx for idx, state_id in enumerate(cn_ids)}
    return cn_ids, omega, state_id_to_index


def load_monthly_states(
    input_dir: Path, state_id_to_index: dict[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    states = read_csv(input_dir / "monthly_key_states.csv")
    states = states.sort_values("month").reset_index(drop=True)
    months = states["month"].astype(int).to_numpy()
    observed_ids = states["state_id"].astype(int).to_numpy()
    missing = [int(state_id) for state_id in observed_ids if int(state_id) not in state_id_to_index]
    if missing:
        raise ValueError(f"Observed states are absent from state tables: {missing}")
    observed_indices = np.array(
        [state_id_to_index[int(state_id)] for state_id in observed_ids], dtype=int
    )
    return months, observed_indices


def load_impacts(input_dir: Path, months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    impacts = read_csv(input_dir / "monthly_country_domain_impact.csv")
    values = np.zeros((len(months), N_ACTORS, N_DOMAINS), dtype=float)
    month_index = {int(month): idx for idx, month in enumerate(months)}
    country_index = {country: idx for idx, country in enumerate(COUNTRIES)}
    domain_index = {domain: idx for idx, domain in enumerate(DOMAINS)}

    for row in impacts.itertuples(index=False):
        month = int(getattr(row, "month"))
        country = str(getattr(row, "country"))
        domain = str(getattr(row, "domain"))
        if month not in month_index or country not in country_index or domain not in domain_index:
            continue
        values[
            month_index[month],
            country_index[country],
            domain_index[domain],
        ] = float(getattr(row, "preference_update_impact"))

    mean = values.mean(axis=0, keepdims=True)
    sd = values.std(axis=0, keepdims=True)
    scaled = (values - mean) / np.maximum(sd, EPS)
    return values, scaled


def load_neighbors(
    input_dir: Path, n_states: int, state_id_to_index: dict[int, int]
) -> list[list[np.ndarray]]:
    graph = read_csv(input_dir / "state_transition_graph.csv")
    raw: list[list[list[int]]] = [[[] for _ in range(n_states)] for _ in range(N_ACTORS)]
    actor_index = {country: idx for idx, country in enumerate(COUNTRIES)}

    for row in graph.itertuples(index=False):
        actor = str(getattr(row, "actor"))
        if actor not in actor_index:
            continue
        from_state = int(getattr(row, "from_state_id"))
        to_state = int(getattr(row, "to_state_id"))
        if from_state not in state_id_to_index or to_state not in state_id_to_index:
            continue
        raw[actor_index[actor]][state_id_to_index[from_state]].append(
            state_id_to_index[to_state]
        )

    neighbors: list[list[np.ndarray]] = []
    for actor in range(N_ACTORS):
        actor_neighbors = []
        for state in range(n_states):
            actor_neighbors.append(np.array(sorted(set(raw[actor][state])), dtype=int))
        neighbors.append(actor_neighbors)
    return neighbors


def generate_score_perturbations(
    n_months: int,
    n_states: int,
    config: SolverConfig,
) -> np.ndarray:
    """Generate a fixed, bounded disturbance path without using state outcomes.

    A latent stationary AR(1) process is generated independently for every
    actor-state-driver relation and mapped to ``[-delta, delta]`` with ``tanh``.
    The perturbation seed is intentionally separate from the optimizer seed so
    multi-seed optimizer checks do not silently change the assumed measurement
    process.
    """
    shape = (n_months, N_ACTORS, n_states, N_DRIVERS)
    if n_months <= 0 or n_states <= 0 or config.delta <= 0.0:
        return np.zeros(shape, dtype=float)

    rho = min(max(float(config.delta_persistence), 0.0), 1.0)
    innovation_scale = math.sqrt(max(1.0 - rho * rho, 0.0))
    rng = np.random.default_rng(config.perturbation_seed)
    innovations = rng.standard_normal(shape)
    latent = np.empty(shape, dtype=float)
    latent[0] = innovations[0]
    for t in range(1, n_months):
        latent[t] = rho * latent[t - 1] + innovation_scale * innovations[t]
    return float(config.delta) * np.tanh(latent)


def load_model_data(config: SolverConfig) -> ModelData:
    input_dir = Path(config.input_dir)
    state_ids, omega, state_id_to_index = load_state_tables(input_dir)
    months, observed_indices = load_monthly_states(input_dir, state_id_to_index)
    impacts_raw, impacts_scaled = load_impacts(input_dir, months)
    neighbors = load_neighbors(input_dir, len(state_ids), state_id_to_index)
    score_perturbations = generate_score_perturbations(
        len(months), len(state_ids), config
    )
    data = ModelData(
        months=months,
        state_ids=state_ids,
        observed_indices=observed_indices,
        impacts_raw=impacts_raw,
        impacts_scaled=impacts_scaled,
        omega=omega,
        neighbors=neighbors,
        state_id_to_index=state_id_to_index,
        score_perturbations=score_perturbations,
    )
    data.evaluation_plans = build_evaluation_plans(data, config.forward_horizon)
    data.lagged_impacts = precompute_lagged_impacts(data.impacts_scaled)
    data.evaluation_forward_horizon = int(config.forward_horizon)
    return data


def parameter_bounds(config: SolverConfig) -> tuple[np.ndarray, np.ndarray]:
    lower: list[float] = []
    upper: list[float] = []
    for actor in range(N_ACTORS):
        for domain in range(N_DOMAINS):
            for driver in range(N_DRIVERS):
                if not bool(A_ACTIVE_MASK[actor, domain, driver]):
                    continue
                sign = int(A_SIGNS[actor, domain, driver])
                if sign == 1:
                    lower.append(0.0)
                    upper.append(config.a_abs_bound)
                elif sign == -1:
                    lower.append(-config.a_abs_bound)
                    upper.append(0.0)
                elif sign == 0:
                    lower.append(0.0)
                    upper.append(0.0)
                else:
                    lower.append(-config.a_abs_bound)
                    upper.append(config.a_abs_bound)

    lower.extend([0.0, 0.0])
    upper.extend([config.alpha_upper, config.alpha_upper])
    lower.extend([-config.mu_abs_bound] * (N_ACTORS * N_DRIVERS))
    upper.extend([config.mu_abs_bound] * (N_ACTORS * N_DRIVERS))
    return np.array(lower, dtype=float), np.array(upper, dtype=float)


def clip_to_bounds(vector: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(vector, lower), upper)


def decode(vector: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Parameters:
    clipped = clip_to_bounds(np.asarray(vector, dtype=float), lower, upper)
    if len(clipped) != PARAMETER_DIM:
        raise ValueError(f"Expected parameter vector length {PARAMETER_DIM}, got {len(clipped)}.")
    a = np.zeros((N_ACTORS, N_DOMAINS, N_DRIVERS), dtype=float)
    offset = 0
    for actor in range(N_ACTORS):
        for domain in range(N_DOMAINS):
            for driver in range(N_DRIVERS):
                if not bool(A_ACTIVE_MASK[actor, domain, driver]):
                    continue
                a[actor, domain, driver] = clipped[offset]
                offset += 1
    alpha = clipped[ALPHA_SLICE]
    mu = clipped[MU_SLICE].reshape(N_ACTORS, N_DRIVERS)
    return Parameters(a=a, alpha=alpha, mu=mu, vector=clipped.copy())


def distributed_lag_impacts(impacts_scaled: np.ndarray, end: int) -> np.ndarray:
    """Return a fixed 0.6/0.3/0.1 lag aggregate ending at ``end``.

    Available weights are renormalized at the beginning of the sample. The
    mechanism is fixed ex ante and therefore adds no estimated parameters.
    """
    lag_weights = (0.6, 0.3, 0.1)
    available = min(end + 1, len(lag_weights))
    weights = np.asarray(lag_weights[:available], dtype=float)
    weights /= float(np.sum(weights))
    aggregate = np.zeros_like(impacts_scaled[0], dtype=float)
    for lag, weight in enumerate(weights):
        aggregate += weight * impacts_scaled[end - lag]
    return aggregate


def precompute_lagged_impacts(impacts_scaled: np.ndarray) -> np.ndarray:
    """Cache the fixed distributed-lag inputs used by every parameter vector."""
    if len(impacts_scaled) <= 1:
        return np.empty((0,) + impacts_scaled.shape[1:], dtype=float)
    return np.stack(
        [
            distributed_lag_impacts(impacts_scaled, end)
            for end in range(len(impacts_scaled) - 1)
        ],
        axis=0,
    )


def compute_h(
    params: Parameters,
    impacts_scaled: np.ndarray,
    lagged_impacts: np.ndarray | None = None,
    nonlinear_saturation: bool = True,
) -> np.ndarray:
    n_months = len(impacts_scaled)
    h = np.zeros((n_months, N_ACTORS, N_DRIVERS), dtype=float)
    h[0] = params.mu
    if lagged_impacts is not None and lagged_impacts.shape != (
        max(n_months - 1, 0),
        N_ACTORS,
        N_DOMAINS,
    ):
        raise ValueError(
            f"lagged_impacts has shape {lagged_impacts.shape}, expected "
            f"{(max(n_months - 1, 0), N_ACTORS, N_DOMAINS)}."
        )
    for t in range(1, n_months):
        lagged_month = (
            lagged_impacts[t - 1]
            if lagged_impacts is not None
            else distributed_lag_impacts(impacts_scaled, t - 1)
        )
        innovation = np.einsum("ad,adk->ak", lagged_month, params.a)
        response = np.tanh(innovation) if nonlinear_saturation else innovation
        target = params.mu + response
        h[t] = (
            params.alpha[:, None] * h[t - 1]
            + (1.0 - params.alpha[:, None]) * target
        )
    return h


def base_utilities(omega: np.ndarray, h_month: np.ndarray) -> np.ndarray:
    return np.einsum("ask,ak->as", omega, h_month)


def perturbations_for_data(data: ModelData, config: SolverConfig) -> np.ndarray:
    expected_shape = (
        len(data.months),
        N_ACTORS,
        len(data.state_ids),
        N_DRIVERS,
    )
    if data.score_perturbations is not None:
        perturbations = np.asarray(data.score_perturbations, dtype=float)
        if perturbations.shape != expected_shape:
            raise ValueError(
                "score_perturbations has shape "
                f"{perturbations.shape}, expected {expected_shape}."
            )
        return perturbations
    return generate_score_perturbations(len(data.months), len(data.state_ids), config)


def perturbed_utilities(
    omega: np.ndarray,
    perturbation_month: np.ndarray,
    h_month: np.ndarray,
) -> np.ndarray:
    """Compute utilities from an ex ante score disturbance fixed for the month."""
    return base_utilities(omega + perturbation_month, h_month)


def standardize_utilities(utilities: np.ndarray) -> np.ndarray:
    """Standardize each actor's utilities across all feasible states."""
    means = np.mean(utilities, axis=1, keepdims=True)
    scales = np.std(utilities, axis=1, keepdims=True)
    return (utilities - means) / np.maximum(scales, EPS)


def all_perturbed_utilities(
    omega: np.ndarray,
    score_perturbations: np.ndarray,
    preferences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw and standardized utilities for every month in one operation."""
    raw = np.einsum(
        "task,tak->tas",
        omega[None, :, :, :] + score_perturbations,
        preferences,
        optimize=True,
    )
    means = np.mean(raw, axis=2, keepdims=True)
    scales = np.std(raw, axis=2, keepdims=True)
    standardized = (raw - means) / np.maximum(scales, EPS)
    return raw, standardized


def transition_matrix(
    utilities: np.ndarray,
    neighbors: list[list[np.ndarray]],
    temperature: float,
) -> np.ndarray:
    """Conditional compatibility over unilateral destinations, given a change.

    The current state is deliberately excluded. Every graph-valid move has a
    strictly positive softmax weight, irrespective of whether its immediate
    utility gain is positive. This is a calibration measure, not a direct state
    transition forecast.
    """
    n_states = utilities.shape[1]
    matrix = np.zeros((N_ACTORS, n_states, n_states), dtype=float)
    tau = max(float(temperature), EPS)
    for actor in range(N_ACTORS):
        for state in range(n_states):
            choices = neighbors[actor][state]
            if len(choices) == 0:
                continue
            logits = (utilities[actor, choices] - utilities[actor, state]) / tau
            logits -= float(np.max(logits))
            weights = np.exp(logits)
            matrix[actor, state, choices] = weights / float(np.sum(weights))
    return matrix


def direct_actors(neighbors: list[list[np.ndarray]], start: int, dest: int) -> list[int]:
    actors = []
    for actor in range(N_ACTORS):
        if int(dest) in set(int(x) for x in neighbors[actor][int(start)]):
            actors.append(actor)
    return actors


def two_step_intermediates(
    neighbors: list[list[np.ndarray]], start: int, dest: int, first: int, second: int
) -> list[int]:
    mids = []
    for mid in neighbors[first][int(start)]:
        if int(dest) in set(int(x) for x in neighbors[second][int(mid)]):
            mids.append(int(mid))
    return mids


def transition_probability(
    matrix: np.ndarray, neighbors: list[list[np.ndarray]], start: int, dest: int
) -> tuple[float, str, int]:
    direct = direct_actors(neighbors, start, dest)
    direct_terms = [float(matrix[actor, start, dest]) for actor in direct]
    if direct_terms:
        return float(np.mean(direct_terms)), "|".join(COUNTRIES[a] for a in direct), len(direct_terms)

    ordered_terms = []
    valid_path_count = 0
    labels = []
    for first, second in ((0, 1), (1, 0)):
        mids = two_step_intermediates(neighbors, start, dest, first, second)
        score = 0.0
        for mid in mids:
            product = float(matrix[first, start, mid] * matrix[second, mid, dest])
            score += product
            if product > 0.0:
                valid_path_count += 1
        if mids:
            labels.append(f"{COUNTRIES[first]}->{COUNTRIES[second]}")
            ordered_terms.append(score)

    if ordered_terms:
        return float(np.mean(ordered_terms)), "|".join(labels), valid_path_count
    return 0.0, "none", 0


def targeted_transition_probability(
    utilities: np.ndarray,
    neighbors: list[list[np.ndarray]],
    plan: EvaluationPlan,
    temperature: float,
) -> tuple[float, str, int]:
    """Evaluate only softmax rows used by one observed transition path."""
    n_states = utilities.shape[1]
    tau = max(float(temperature), EPS)
    row_cache: dict[tuple[int, int], np.ndarray] = {}

    def row_probabilities(actor: int, state: int) -> np.ndarray:
        key = (actor, state)
        cached = row_cache.get(key)
        if cached is not None:
            return cached
        probabilities = np.zeros(n_states, dtype=float)
        choices = neighbors[actor][state]
        if len(choices) > 0:
            logits = (utilities[actor, choices] - utilities[actor, state]) / tau
            logits -= float(np.max(logits))
            weights = np.exp(logits)
            probabilities[choices] = weights / float(np.sum(weights))
        row_cache[key] = probabilities
        return probabilities

    direct_terms = [
        float(row_probabilities(actor, plan.start)[plan.observed])
        for actor in plan.direct_actors
    ]
    if direct_terms:
        label = "|".join(COUNTRIES[actor] for actor in plan.direct_actors)
        return float(np.mean(direct_terms)), label, len(direct_terms)

    ordered_terms: list[float] = []
    labels: list[str] = []
    valid_path_count = 0
    for first, second, mids in plan.ordered_paths:
        score = 0.0
        first_row = row_probabilities(first, plan.start)
        for mid in mids:
            product = float(
                first_row[mid] * row_probabilities(second, mid)[plan.observed]
            )
            score += product
            if product > 0.0:
                valid_path_count += 1
        ordered_terms.append(score)
        labels.append(f"{COUNTRIES[first]}->{COUNTRIES[second]}")

    if ordered_terms:
        return float(np.mean(ordered_terms)), "|".join(labels), valid_path_count
    return 0.0, "none", 0


def reachable_within_horizon(
    neighbors: list[list[np.ndarray]], start: int, horizon: int
) -> set[int]:
    visited = {int(start)}
    queue: deque[tuple[int, int]] = deque([(int(start), 0)])
    while queue:
        state, depth = queue.popleft()
        if depth >= horizon:
            continue
        destinations = set()
        for actor in range(N_ACTORS):
            destinations.update(int(x) for x in neighbors[actor][state])
        for dest in destinations:
            if dest in visited:
                continue
            visited.add(dest)
            queue.append((dest, depth + 1))
    visited.discard(int(start))
    return visited


def build_evaluation_plans(
    data: ModelData, forward_horizon: int
) -> tuple[EvaluationPlan, ...]:
    """Precompute graph-only work shared by every objective evaluation."""
    plans: list[EvaluationPlan] = []
    for t in range(1, len(data.months)):
        start = int(data.observed_indices[t - 1])
        observed = int(data.observed_indices[t])
        changed = start != observed
        if not changed:
            plans.append(
                EvaluationPlan(
                    month_index=t,
                    start=start,
                    observed=observed,
                    changed=False,
                )
            )
            continue

        direct = tuple(direct_actors(data.neighbors, start, observed))
        ordered_paths: list[tuple[int, int, tuple[int, ...]]] = []
        if not direct:
            for first, second in ((0, 1), (1, 0)):
                mids = tuple(
                    two_step_intermediates(
                        data.neighbors, start, observed, first, second
                    )
                )
                if mids:
                    ordered_paths.append((first, second, mids))
        candidates = tuple(
            reachable_within_horizon(
                data.neighbors, observed, max(0, int(forward_horizon))
            )
        )
        plans.append(
            EvaluationPlan(
                month_index=t,
                start=start,
                observed=observed,
                changed=True,
                direct_actors=direct,
                ordered_paths=tuple(ordered_paths),
                forward_candidates=candidates,
            )
        )
    return tuple(plans)


def nash_margin(
    state: int,
    actor: int,
    utilities: np.ndarray,
    neighbors: list[list[np.ndarray]],
    theta: float,
    empty_margin: float,
) -> float:
    choices = neighbors[actor][state]
    if len(choices) == 0:
        return empty_margin
    return float(np.min(utilities[actor, state] - utilities[actor, choices] + theta))


def improved_moves(
    state: int,
    actor: int,
    utilities: np.ndarray,
    neighbors: list[list[np.ndarray]],
    theta: float,
) -> np.ndarray:
    choices = neighbors[actor][state]
    if len(choices) == 0:
        return np.array([], dtype=int)
    gains = utilities[actor, choices] - utilities[actor, state]
    return choices[gains > theta]


def stability_margins(
    state: int,
    actor: int,
    utilities: np.ndarray,
    neighbors: list[list[np.ndarray]],
    config: SolverConfig,
) -> np.ndarray:
    theta = config.theta
    opponent = 1 - actor
    nash = nash_margin(
        state, actor, utilities, neighbors, theta, config.empty_neighbor_margin
    )
    ui = improved_moves(state, actor, utilities, neighbors, theta)
    if len(ui) == 0:
        return np.array([nash, nash, nash, nash], dtype=float)

    gmr_terms = []
    smr_terms = []
    seq_terms = []
    for q_value in ui:
        q = int(q_value)
        sanctions = neighbors[opponent][q]
        if len(sanctions) == 0:
            gmr_terms.append(LARGE_NEGATIVE)
            smr_terms.append(LARGE_NEGATIVE)
            seq_terms.append(LARGE_NEGATIVE)
            continue

        gmr_by_sanction = utilities[actor, state] - utilities[actor, sanctions] + theta
        gmr_terms.append(float(np.max(gmr_by_sanction)))

        smr_by_sanction = []
        seq_by_sanction = []
        for k_value in sanctions:
            k = int(k_value)
            counter_states = np.concatenate(
                [np.array([k], dtype=int), neighbors[actor][k]]
            )
            smr_by_sanction.append(
                float(
                    np.min(
                        utilities[actor, state]
                        - utilities[actor, counter_states]
                        + theta
                    )
                )
            )
            credibility = float(utilities[opponent, k] - utilities[opponent, q] - theta)
            deterrence = float(utilities[actor, state] - utilities[actor, k] + theta)
            seq_by_sanction.append(min(credibility, deterrence))

        smr_terms.append(float(np.max(smr_by_sanction)))
        seq_terms.append(float(np.max(seq_by_sanction)))

    margins = np.array(
        [
            nash,
            min(gmr_terms),
            min(smr_terms),
            min(seq_terms),
        ],
        dtype=float,
    )
    return np.maximum(np.minimum(margins, config.margin_cap), -config.margin_cap)


def stability_flags(margins: np.ndarray) -> np.ndarray:
    return margins >= 0.0


def forward_shortfall(
    utilities: np.ndarray,
    start: int,
    observed: int,
    data: ModelData,
    config: SolverConfig,
) -> tuple[float, int, float]:
    """Return bounded explanatory shortfall and the best common-gain state."""
    candidates = reachable_within_horizon(
        data.neighbors, observed, config.forward_horizon
    )
    if not candidates:
        return float(config.theta_forward**2), int(observed), 0.0

    best_raw_gain = -math.inf
    best_state = int(observed)
    for target in candidates:
        common_gain = float(np.min(utilities[:, target] - utilities[:, start]))
        if common_gain > best_raw_gain:
            best_state = int(target)
            best_raw_gain = common_gain
    shortfall = max(config.theta_forward - best_raw_gain, 0.0) ** 2
    return float(shortfall), best_state, float(best_raw_gain)


def forward_shortfall_from_candidates(
    utilities: np.ndarray,
    start: int,
    observed: int,
    candidates: tuple[int, ...],
    config: SolverConfig,
) -> tuple[float, int, float]:
    """Vectorized forward shortfall over a precomputed reachable-state set."""
    if not candidates:
        best_raw_gain = LARGE_NEGATIVE
        best_state = int(observed)
    else:
        candidate_array = np.fromiter(candidates, dtype=int)
        common_gains = np.min(
            utilities[:, candidate_array] - utilities[:, start, None], axis=0
        )
        best_position = int(np.argmax(common_gains))
        best_raw_gain = float(common_gains[best_position])
        best_state = int(candidate_array[best_position])
    shortfall = max(config.theta_forward - best_raw_gain, 0.0) ** 2
    return float(shortfall), best_state, float(best_raw_gain)


def regularization_components(params: Parameters) -> tuple[float, float, float]:
    active_a = params.a[A_ACTIVE_MASK]
    return (
        float(np.mean(active_a**2)),
        float(np.mean(params.mu**2)),
        float(np.mean(params.alpha**2)),
    )


def evaluate(
    vector: np.ndarray,
    data: ModelData,
    config: SolverConfig,
    lower: np.ndarray,
    upper: np.ndarray,
    return_rows: bool = False,
) -> tuple[float, dict[str, Any]]:
    params = decode(vector, lower, upper)
    h = compute_h(
        params,
        data.impacts_scaled,
        data.lagged_impacts,
        nonlinear_saturation=config.nonlinear_saturation,
    )
    score_perturbations = perturbations_for_data(data, config)
    raw_utilities_all, utilities_all = all_perturbed_utilities(
        data.omega, score_perturbations, h
    )
    if (
        data.evaluation_plans is None
        or data.evaluation_forward_horizon != int(config.forward_horizon)
    ):
        data.evaluation_plans = build_evaluation_plans(data, config.forward_horizon)
        data.evaluation_forward_horizon = int(config.forward_horizon)
    transition_log_terms: list[float] = []
    forward_penalties: list[float] = []
    stability_penalties: list[float] = []
    transition_probability_sum = 0.0
    rows: list[dict[str, Any]] = []

    for plan in data.evaluation_plans:
        t = plan.month_index
        start = plan.start
        observed = plan.observed
        changed = plan.changed
        raw_utilities = raw_utilities_all[t]
        utilities = utilities_all[t]

        if changed:
            probability, mechanism, path_count = targeted_transition_probability(
                utilities,
                data.neighbors,
                plan,
                config.transition_temperature,
            )
            probability = max(float(probability), 0.0)
            transition_probability_sum += probability
            log_term = math.log(max(probability, config.epsilon_p))
            transition_log_terms.append(log_term)
            fwd_penalty, fwd_state, fwd_raw_gain = forward_shortfall_from_candidates(
                utilities,
                start,
                observed,
                plan.forward_candidates,
                config,
            )
            forward_penalties.append(fwd_penalty)
            if return_rows:
                rows.append(
                    {
                        "month": int(data.months[t]),
                        "start_state_id": int(data.state_ids[start]),
                        "observed_state_id": int(data.state_ids[observed]),
                        "observed_change": 1,
                        "transition_probability": probability,
                        "transition_mechanism": mechanism,
                        "valid_path_count": int(path_count),
                        "log_transition_term": log_term,
                        "forward_shortfall_penalty": fwd_penalty,
                        "forward_target_state_id": int(data.state_ids[fwd_state]),
                        "forward_common_gain": fwd_raw_gain,
                        "cn_primary_seq_margin": "",
                        "us_primary_seq_margin": "",
                        "cn_stability_penalty": "",
                        "us_stability_penalty": "",
                    }
                )
        else:
            actor_margins = []
            actor_penalties = []
            for actor in range(N_ACTORS):
                margins = stability_margins(
                    start, actor, utilities, data.neighbors, config
                )
                primary_seq_margin = float(margins[3])
                penalty = max(-primary_seq_margin, 0.0) ** 2
                actor_margins.append(margins)
                actor_penalties.append(penalty)
                stability_penalties.append(penalty)
            if return_rows:
                actor_flags = [stability_flags(margins) for margins in actor_margins]
                rows.append(
                    {
                        "month": int(data.months[t]),
                        "start_state_id": int(data.state_ids[start]),
                        "observed_state_id": int(data.state_ids[observed]),
                        "observed_change": 0,
                        "transition_probability": "",
                        "transition_mechanism": "stay",
                        "valid_path_count": "",
                        "log_transition_term": "",
                        "forward_shortfall_penalty": "",
                        "forward_target_state_id": "",
                        "forward_common_gain": "",
                        "cn_stable_any": int(np.any(actor_flags[0])),
                        "us_stable_any": int(np.any(actor_flags[1])),
                        "cn_best_margin": float(np.max(actor_margins[0])),
                        "us_best_margin": float(np.max(actor_margins[1])),
                        "cn_primary_seq_margin": float(actor_margins[0][3]),
                        "us_primary_seq_margin": float(actor_margins[1][3]),
                        "cn_stability_penalty": float(actor_penalties[0]),
                        "us_stability_penalty": float(actor_penalties[1]),
                        "seq_stable_both": int(
                            actor_margins[0][3] >= 0.0
                            and actor_margins[1][3] >= 0.0
                        ),
                        "cn_margins_Nash_GMR_SMR_SEQ": "|".join(
                            f"{x:.8g}" for x in actor_margins[0]
                        ),
                        "us_margins_Nash_GMR_SMR_SEQ": "|".join(
                            f"{x:.8g}" for x in actor_margins[1]
                        ),
                    }
                )

    mean_log_transition = (
        float(np.mean(transition_log_terms)) if transition_log_terms else 0.0
    )
    mean_forward_penalty = (
        float(np.mean(forward_penalties)) if forward_penalties else 0.0
    )
    mean_stability_penalty = (
        float(np.mean(stability_penalties)) if stability_penalties else 0.0
    )
    reg_a, reg_mu, reg_alpha = regularization_components(params)
    fit_objective = (
        mean_log_transition
        - config.lambda_forward * mean_forward_penalty
        - config.lambda_stable * mean_stability_penalty
        - config.lambda_a * reg_a
        - config.lambda_mu * reg_mu
        - config.lambda_alpha * reg_alpha
    )
    loss = -fit_objective

    details = {
        "loss": float(loss),
        "fit_objective": float(fit_objective),
        "hard_feasibility_constraints": False,
        "log_transition_objective": mean_log_transition,
        "transition_observation_count": len(transition_log_terms),
        "forward_shortfall_penalty": mean_forward_penalty,
        "forward_observation_count": len(forward_penalties),
        "seq_stability_penalty": mean_stability_penalty,
        "stability_actor_observation_count": len(stability_penalties),
        "regularization_a": reg_a,
        "regularization_mu": reg_mu,
        "regularization_alpha": reg_alpha,
        "transition_probability_sum": float(transition_probability_sum),
        "params": params,
        "h": h,
    }
    if return_rows:
        details["monthly_rows"] = rows
    return float(loss), details


def initialize_population(
    rng: np.random.Generator, lower: np.ndarray, upper: np.ndarray, size: int
) -> np.ndarray:
    span = upper - lower
    random_part = rng.random((size, len(lower))) * span + lower
    fixed = span <= EPS
    random_part[:, fixed] = lower[fixed]
    return random_part


def differential_evolution(
    data: ModelData, config: SolverConfig, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    rng = np.random.default_rng(config.seed)
    pop_size = max(config.population_size, 4)
    log_progress(
        "Starting differential evolution: "
        f"population={pop_size}, generations={config.generations}, "
        f"dimension={PARAMETER_DIM}"
    )
    population = initialize_population(rng, lower, upper, pop_size)
    losses = np.empty(pop_size, dtype=float)
    initial_log_interval = max(1, min(config.progress_interval, pop_size // 10 or 1))
    for idx in range(pop_size):
        losses[idx], _ = evaluate(population[idx], data, config, lower, upper)
        if (idx + 1) % initial_log_interval == 0 or idx == pop_size - 1:
            log_progress(
                "Initial population evaluated: "
                f"{idx + 1}/{pop_size}, current_best_loss={float(np.min(losses[:idx + 1])):.6f}"
            )

    history: list[dict[str, Any]] = []
    generations_without_improvement = 0
    for generation in range(config.generations):
        previous_best = float(np.min(losses))
        for idx in range(pop_size):
            choices = [x for x in range(pop_size) if x != idx]
            a_idx, b_idx, c_idx = rng.choice(choices, size=3, replace=False)
            mutant = (
                population[a_idx]
                + config.differential_weight * (population[b_idx] - population[c_idx])
            )
            mutant = clip_to_bounds(mutant, lower, upper)
            crossover = rng.random(PARAMETER_DIM) < config.crossover_rate
            crossover[int(rng.integers(0, PARAMETER_DIM))] = True
            trial = np.where(crossover, mutant, population[idx])
            trial = clip_to_bounds(trial, lower, upper)
            trial_loss, _ = evaluate(trial, data, config, lower, upper)
            if trial_loss <= losses[idx]:
                population[idx] = trial
                losses[idx] = trial_loss

        best_idx = int(np.argmin(losses))
        if float(losses[best_idx]) < previous_best:
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        history.append(
            {
                "phase": "differential_evolution",
                "generation": int(generation),
                "best_loss": float(losses[best_idx]),
                "median_loss": float(np.median(losses)),
                "worst_loss": float(np.max(losses)),
            }
        )
        if (
            generation == 0
            or (generation + 1) % config.progress_interval == 0
            or generation == config.generations - 1
        ):
            _, best_details = evaluate(
                population[best_idx], data, config, lower, upper
            )
            log_progress(
                "Differential evolution progress: "
                f"generation={generation + 1}/{config.generations}, "
                f"best_loss={float(losses[best_idx]):.6f}, "
                f"median_loss={float(np.median(losses)):.6f}, "
                f"fit_objective={best_details['fit_objective']:.6f}"
            )
        if (
            config.de_early_stop_patience > 0
            and generations_without_improvement
            >= config.de_early_stop_patience
        ):
            log_progress(
                "Stopping differential evolution early: "
                f"no best-loss improvement for {generations_without_improvement} "
                "generations."
            )
            break

    best_idx = int(np.argmin(losses))
    log_progress(
        "Finished differential evolution: "
        f"best_loss={float(losses[best_idx]):.6f}"
    )
    return population[best_idx].copy(), float(losses[best_idx]), history


def local_search(
    start: np.ndarray,
    data: ModelData,
    config: SolverConfig,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    rng = np.random.default_rng(config.seed + 1009)
    current = start.copy()
    current_loss, _ = evaluate(current, data, config, lower, upper)
    best = current.copy()
    best_loss = current_loss
    span = np.maximum(upper - lower, EPS)
    history: list[dict[str, Any]] = []
    log_progress(
        "Starting local search: "
        f"steps={config.local_search_steps}, initial_loss={best_loss:.6f}"
    )

    steps_without_improvement = 0
    for step in range(config.local_search_steps):
        scale = config.local_search_scale * (1.0 - step / max(config.local_search_steps, 1))
        proposal = current + rng.normal(0.0, scale, size=PARAMETER_DIM) * span
        proposal = clip_to_bounds(proposal, lower, upper)
        proposal_loss, _ = evaluate(proposal, data, config, lower, upper)
        temperature = max(1.0e-6, 0.05 * (1.0 - step / max(config.local_search_steps, 1)))
        accept = proposal_loss <= current_loss
        if not accept:
            accept = rng.random() < math.exp(-(proposal_loss - current_loss) / temperature)
        if accept:
            current = proposal
            current_loss = proposal_loss
        if current_loss < best_loss:
            best = current.copy()
            best_loss = current_loss
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
        if step % config.progress_interval == 0 or step == config.local_search_steps - 1:
            history.append(
                {
                    "phase": "local_search",
                    "step": int(step),
                    "current_loss": float(current_loss),
                    "best_loss": float(best_loss),
                }
            )
        if (
            step == 0
            or (step + 1) % config.progress_interval == 0
            or step == config.local_search_steps - 1
        ):
            _, best_details = evaluate(best, data, config, lower, upper)
            log_progress(
                "Local search progress: "
                f"step={step + 1}/{config.local_search_steps}, "
                f"current_loss={current_loss:.6f}, "
                f"best_loss={best_loss:.6f}, "
                f"fit_objective={best_details['fit_objective']:.6f}"
            )
        if (
            config.local_search_early_stop_patience > 0
            and steps_without_improvement
            >= config.local_search_early_stop_patience
        ):
            log_progress(
                "Stopping local search early: "
                f"no best-loss improvement for {steps_without_improvement} proposals."
            )
            break
    log_progress(f"Finished local search: best_loss={best_loss:.6f}")
    return best, float(best_loss), history


def make_run_dir(output_dir: Path) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%d-%H-%M")
    run_dir = output_dir / timestamp
    try:
        # Atomic creation avoids two concurrent multi-seed processes selecting
        # the same timestamp directory.
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir, timestamp
    except FileExistsError:
        pass
    suffix = 1
    while True:
        candidate = output_dir / f"{timestamp}_{suffix:02d}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate, timestamp
        except FileExistsError:
            suffix += 1


def matrix_to_table(matrix: np.ndarray, actor: int) -> list[dict[str, Any]]:
    rows = []
    drivers = DRIVERS_BY_ACTOR[actor]
    for d_idx, domain in enumerate(DOMAINS):
        row: dict[str, Any] = {"domain": domain}
        for k_idx, driver in enumerate(drivers):
            row[driver] = float(matrix[d_idx, k_idx])
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parameter_rows(params: Parameters) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for actor_idx, country in enumerate(COUNTRIES):
        for domain_idx, domain in enumerate(DOMAINS):
            for driver_idx, driver in enumerate(DRIVERS_BY_ACTOR[actor_idx]):
                estimated = bool(A_ACTIVE_MASK[actor_idx, domain_idx, driver_idx])
                sign = int(A_SIGNS[actor_idx, domain_idx, driver_idx])
                rows.append(
                    {
                        "index": offset if estimated else "",
                        "group": "A",
                        "country": country,
                        "domain": domain,
                        "driver": driver,
                        "value": float(params.a[actor_idx, domain_idx, driver_idx]),
                        "estimated": int(estimated),
                        "constraint": (
                            "nonnegative"
                            if estimated and sign == 1
                            else "nonpositive"
                            if estimated and sign == -1
                            else "free"
                            if estimated
                            else "fixed_zero"
                        ),
                    }
                )
                if estimated:
                    offset += 1
    for actor_idx, country in enumerate(COUNTRIES):
        rows.append(
            {
                "index": offset,
                "group": "alpha",
                "country": country,
                "domain": "",
                "driver": "",
                "value": float(params.alpha[actor_idx]),
                "estimated": 1,
                "constraint": "bounded",
            }
        )
        offset += 1
    for actor_idx, country in enumerate(COUNTRIES):
        for driver_idx, driver in enumerate(DRIVERS_BY_ACTOR[actor_idx]):
            rows.append(
                {
                    "index": offset,
                    "group": "mu",
                    "country": country,
                    "domain": "",
                    "driver": driver,
                    "value": float(params.mu[actor_idx, driver_idx]),
                    "estimated": 1,
                    "constraint": "bounded",
                }
            )
            offset += 1
    return rows


def build_theoretical_mapping_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for actor_idx, country in enumerate(COUNTRIES):
        for domain_idx, domain in enumerate(DOMAINS):
            for driver_idx, driver in enumerate(DRIVERS_BY_ACTOR[actor_idx]):
                active = bool(A_ACTIVE_MASK[actor_idx, domain_idx, driver_idx])
                sign = int(A_SIGNS[actor_idx, domain_idx, driver_idx])
                rows.append(
                    {
                        "country": country,
                        "domain": domain,
                        "driver_index": int(driver_idx + 1),
                        "driver": driver,
                        "direct_connection": int(active),
                        "coefficient_status": (
                            "estimated_nonnegative"
                            if active and sign == 1
                            else "estimated_nonpositive"
                            if active and sign == -1
                            else "estimated_free"
                            if active
                            else "fixed_zero"
                        ),
                        "selection_basis": "theory_prespecified",
                    }
                )
    return rows


def build_preference_rows(data: ModelData, h: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t, month in enumerate(data.months):
        for actor_idx, country in enumerate(COUNTRIES):
            for driver_idx, driver in enumerate(DRIVERS_BY_ACTOR[actor_idx]):
                rows.append(
                    {
                        "month": int(month),
                        "country": country,
                        "driver_index": int(driver_idx + 1),
                        "driver": driver,
                        "preference_value": float(h[t, actor_idx, driver_idx]),
                    }
                )
    return rows


def build_observed_utility_rows(
    data: ModelData, h: np.ndarray, config: SolverConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    score_perturbations = perturbations_for_data(data, config)
    for t, month in enumerate(data.months):
        state = int(data.observed_indices[t])
        base = base_utilities(data.omega, h[t])
        total = perturbed_utilities(data.omega, score_perturbations[t], h[t])
        standardized = standardize_utilities(total)
        perturbation_utility = total - base
        for actor_idx, country in enumerate(COUNTRIES):
            rows.append(
                {
                    "month": int(month),
                    "country": country,
                    "observed_state_id": int(data.state_ids[state]),
                    "base_utility": float(base[actor_idx, state]),
                    "perturbation_utility": float(
                        perturbation_utility[actor_idx, state]
                    ),
                    "perturbed_utility": float(total[actor_idx, state]),
                    "standardized_utility": float(
                        standardized[actor_idx, state]
                    ),
                    "score_perturbation_max_abs": float(
                        np.max(np.abs(score_perturbations[t, actor_idx, state]))
                    ),
                }
            )
    return rows


def build_score_perturbation_rows(
    data: ModelData, config: SolverConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    score_perturbations = perturbations_for_data(data, config)
    for t, month in enumerate(data.months):
        for actor_idx, country in enumerate(COUNTRIES):
            for state_idx, state_id in enumerate(data.state_ids):
                for driver_idx, driver in enumerate(DRIVERS_BY_ACTOR[actor_idx]):
                    perturbation = float(
                        score_perturbations[t, actor_idx, state_idx, driver_idx]
                    )
                    base_score = float(data.omega[actor_idx, state_idx, driver_idx])
                    rows.append(
                        {
                            "month": int(month),
                            "country": country,
                            "state_id": int(state_id),
                            "driver_index": int(driver_idx + 1),
                            "driver": driver,
                            "base_score": base_score,
                            "score_perturbation": perturbation,
                            "perturbed_score": base_score + perturbation,
                        }
                    )
    return rows


def sparse_positions(count: int, max_ticks: int = 10) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=int)
    if count <= max_ticks:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, max_ticks, dtype=int))


def save_figure(fig: Any, path: Path) -> None:
    fig.tight_layout()
    for suffix in (".png", ".pdf", ".svg"):
        output_path = path.with_suffix(suffix)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")


def save_visualizations(
    run_dir: Path,
    timestamp: str,
    data: ModelData,
    config: SolverConfig,
    params: Parameters,
    h: np.ndarray,
    monthly_rows: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log_progress("matplotlib is unavailable; skipping visualization output.")
        return []

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 9,
        }
    )

    history_df = pd.DataFrame(history)
    if not history_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        de = history_df[history_df["phase"] == "differential_evolution"].copy()
        ls = history_df[history_df["phase"] == "local_search"].copy()
        if not de.empty:
            ax.plot(
                de["generation"],
                de["best_loss"],
                label="DE best loss",
                color="#1D3557",
                linewidth=1.8,
            )
            ax.plot(
                de["generation"],
                de["median_loss"],
                label="DE median loss",
                color="#457B9D",
                linewidth=1.1,
                alpha=0.8,
            )
        if not ls.empty:
            x_offset = float(de["generation"].max() + 1) if not de.empty else 0.0
            x_values = x_offset + ls["step"] / max(float(config.local_search_steps), 1.0) * max(
                float(config.generations), 1.0
            )
            ax.plot(
                x_values,
                ls["best_loss"],
                label="Local search best loss",
                color="#E76F51",
                linewidth=1.8,
            )
        ax.set_title("Optimization Progress")
        ax.set_xlabel("DE generation scale")
        ax.set_ylabel("Loss (lower is better)")
        ax.legend(frameon=False)
        path = figures_dir / f"optimization_progress_{timestamp}.png"
        save_figure(fig, path)
        plt.close(fig)
        created.append(str(path.relative_to(run_dir)))

    fit_df = pd.DataFrame(monthly_rows)
    if not fit_df.empty:
        changed = fit_df[fit_df["observed_change"] == 1].copy()
        stay = fit_df[fit_df["observed_change"] == 0].copy()

        if not changed.empty:
            changed["label"] = changed["month"].astype(str)
            x = np.arange(len(changed))
            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            axes[0].bar(
                x,
                changed["transition_probability"].astype(float),
                color="#2A9D8F",
                width=0.72,
            )
            axes[0].set_ylabel("Compatibility score")
            axes[0].set_title("Historical Transition-Preference Compatibility")
            axes[0].set_ylim(0.0, 1.05)

            axes[1].bar(
                x,
                changed["forward_shortfall_penalty"].astype(float),
                color="#E9C46A",
                width=0.72,
            )
            axes[1].set_ylabel("Squared shortfall")
            axes[1].set_title("Bounded Forward-Explanation Shortfall")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(changed["label"], rotation=45, ha="right")
            axes[1].set_xlabel("Month")
            path = figures_dir / f"transition_fit_{timestamp}.png"
            save_figure(fig, path)
            plt.close(fig)
            created.append(str(path.relative_to(run_dir)))

        if not stay.empty:
            stay["label"] = stay["month"].astype(str)
            x = np.arange(len(stay))
            fig, ax = plt.subplots(figsize=(11, 4.8))
            ax.plot(
                x,
                stay["cn_primary_seq_margin"].astype(float),
                marker="o",
                markersize=3,
                label="CN SEQ margin",
                color="#1D3557",
            )
            ax.plot(
                x,
                stay["us_primary_seq_margin"].astype(float),
                marker="o",
                markersize=3,
                label="US SEQ margin",
                color="#E76F51",
            )
            ax.axhline(config.stable_margin, color="#5C677D", linestyle="--", linewidth=1.0)
            ax.set_title("Stable-Month Primary SEQ Margin")
            ax.set_ylabel("SEQ stability margin")
            ticks = sparse_positions(len(stay))
            ax.set_xticks(ticks)
            ax.set_xticklabels(stay["label"].iloc[ticks], rotation=45, ha="right")
            ax.set_xlabel("Month")
            ax.legend(frameon=False)
            path = figures_dir / f"stability_margins_{timestamp}.png"
            save_figure(fig, path)
            plt.close(fig)
            created.append(str(path.relative_to(run_dir)))

    month_labels = np.array([str(int(month)) for month in data.months])
    x_month = np.arange(len(data.months))
    ticks = sparse_positions(len(data.months))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for actor_idx, country in enumerate(COUNTRIES):
        for driver_idx in range(N_DRIVERS):
            axes[actor_idx].plot(
                x_month,
                h[:, actor_idx, driver_idx],
                linewidth=1.4,
                label=f"D{driver_idx + 1}",
            )
        axes[actor_idx].set_title(f"{country} Dynamic Preference Trajectory")
        axes[actor_idx].set_ylabel("h value")
        axes[actor_idx].legend(ncol=5, frameon=False, loc="upper left")
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(month_labels[ticks], rotation=45, ha="right")
    axes[-1].set_xlabel("Month")
    path = figures_dir / f"preference_trajectories_{timestamp}.png"
    save_figure(fig, path)
    plt.close(fig)
    created.append(str(path.relative_to(run_dir)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for actor_idx, country in enumerate(COUNTRIES):
        image = axes[actor_idx].imshow(params.a[actor_idx], cmap="RdBu_r", aspect="auto")
        axes[actor_idx].set_title(f"{country} Event-to-Driver Matrix A")
        axes[actor_idx].set_yticks(np.arange(N_DOMAINS))
        axes[actor_idx].set_yticklabels(DOMAINS)
        axes[actor_idx].set_xticks(np.arange(N_DRIVERS))
        axes[actor_idx].set_xticklabels([f"D{i + 1}" for i in range(N_DRIVERS)])
        for row in range(N_DOMAINS):
            for col in range(N_DRIVERS):
                axes[actor_idx].text(
                    col,
                    row,
                    f"{params.a[actor_idx, row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        fig.colorbar(image, ax=axes[actor_idx], shrink=0.8)
    path = figures_dir / f"parameter_matrix_A_{timestamp}.png"
    save_figure(fig, path)
    plt.close(fig)
    created.append(str(path.relative_to(run_dir)))

    if fit_df is not None and not fit_df.empty:
        observed_ids = [int(data.state_ids[idx]) for idx in data.observed_indices]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.step(x_month, observed_ids, where="post", color="#1D3557", linewidth=1.8)
        change_months = fit_df[fit_df["observed_change"] == 1]
        if not change_months.empty:
            month_to_x = {int(month): idx for idx, month in enumerate(data.months)}
            xs = [month_to_x[int(month)] for month in change_months["month"]]
            ys = [
                int(row["observed_state_id"])
                for _, row in change_months.iterrows()
            ]
            ax.scatter(xs, ys, color="#E76F51", s=30, zorder=3, label="Observed change")
            ax.legend(frameon=False)
        ax.set_title("Observed State Path")
        ax.set_ylabel("State ID")
        ax.set_xticks(ticks)
        ax.set_xticklabels(month_labels[ticks], rotation=45, ha="right")
        ax.set_xlabel("Month")
        path = figures_dir / f"observed_state_path_{timestamp}.png"
        save_figure(fig, path)
        plt.close(fig)
        created.append(str(path.relative_to(run_dir)))

    log_progress(f"Finished writing {len(created)} visualization files.")
    return created


def save_outputs(
    run_dir: Path,
    timestamp: str,
    data: ModelData,
    config: SolverConfig,
    best_vector: np.ndarray,
    best_details: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    log_progress(f"Writing outputs to {run_dir}")
    params: Parameters = best_details["params"]
    h: np.ndarray = best_details["h"]
    details_for_json = {
        key: value
        for key, value in best_details.items()
        if key not in {"params", "h", "monthly_rows"}
    }
    summary = {
        "timestamp_day_hour_minute": timestamp,
        "model": "DynamicGMCR 42-dimensional smooth dynamic-preference estimation model",
        "model_role": (
            "Estimate and forecast preferences from historical conflict evolution; "
            "future state assessment requires separate GMCR analysis of forecast utilities."
        ),
        "parameter_dimension": PARAMETER_DIM,
        "parameter_structure": {
            "active_A_connections": N_ACTIVE_A_PARAMETERS,
            "active_A_connections_per_country": {
                country: int(np.sum(A_ACTIVE_MASK[idx]))
                for idx, country in enumerate(COUNTRIES)
            },
            "alpha_parameters": N_ACTORS,
            "long_run_mean_parameters": N_ACTORS * N_DRIVERS,
            "inactive_A_connections_fixed_zero": int(
                A_ACTIVE_MASK.size - N_ACTIVE_A_PARAMETERS
            ),
            "connection_selection": "theory_prespecified_before_fitting",
        },
        "preference_process": {
            "long_run_mean": "mu",
            "update": (
                "h_t = alpha*h_(t-1) + (1-alpha)*(mu+tanh(z_(t-1)A))"
                if config.nonlinear_saturation
                else "h_t = alpha*h_(t-1) + (1-alpha)*(mu+z_(t-1)A)"
            ),
            "nonlinear_saturation": bool(config.nonlinear_saturation),
            "distributed_lag": "z_(t-1) = 0.6*x_(t-1)+0.3*x_(t-2)+0.1*x_(t-3), with boundary renormalization",
        },
        "calibration_role": {
            "direct_state_forecast": False,
            "compatibility_condition": "conditional_on_an_observed_change",
            "future_assessment": "apply GMCR separately to utilities from forecast preferences",
        },
        "countries": COUNTRIES,
        "domains": DOMAINS,
        "months": {
            "start": int(data.months[0]),
            "end": int(data.months[-1]),
            "count": int(len(data.months)),
        },
        "states": int(len(data.state_ids)),
        "score_perturbation_process": {
            "generation": "eta_t = delta * tanh(z_t)",
            "latent_process": (
                "z_t = rho * z_(t-1) + sqrt(1-rho^2) * epsilon_t, "
                "epsilon_t ~ N(0,1)"
            ),
            "outcome_conditioned": False,
            "fixed_during_optimization": True,
            "delta": float(config.delta),
            "rho": float(config.delta_persistence),
            "seed": int(config.perturbation_seed),
        },
        "config": asdict(config),
        "objective": details_for_json,
    }
    write_json(run_dir / f"summary_{timestamp}.json", summary)
    write_json(
        run_dir / f"parameters_{timestamp}.json",
        {
            "timestamp_day_hour_minute": timestamp,
            "alpha": {
                country: float(params.alpha[idx]) for idx, country in enumerate(COUNTRIES)
            },
            "mu": {
                country: {
                    driver: float(params.mu[idx, driver_idx])
                    for driver_idx, driver in enumerate(DRIVERS_BY_ACTOR[idx])
                }
                for idx, country in enumerate(COUNTRIES)
            },
            "A": {
                "CN": matrix_to_table(params.a[0], 0),
                "US": matrix_to_table(params.a[1], 1),
            },
            "vector": [float(x) for x in best_vector],
        },
    )
    write_csv(run_dir / f"parameter_vector_{timestamp}.csv", build_parameter_rows(params))
    write_csv(
        run_dir / f"theoretical_mapping_{timestamp}.csv",
        build_theoretical_mapping_rows(),
    )
    write_csv(run_dir / f"monthly_fit_{timestamp}.csv", best_details["monthly_rows"])
    write_csv(run_dir / f"monthly_preferences_{timestamp}.csv", build_preference_rows(data, h))
    write_csv(
        run_dir / f"observed_state_utilities_{timestamp}.csv",
        build_observed_utility_rows(data, h, config),
    )
    write_csv(
        run_dir / f"state_score_perturbations_{timestamp}.csv",
        build_score_perturbation_rows(data, config),
    )
    write_csv(run_dir / f"optimization_history_{timestamp}.csv", history)
    figure_files = save_visualizations(
        run_dir=run_dir,
        timestamp=timestamp,
        data=data,
        config=config,
        params=params,
        h=h,
        monthly_rows=best_details["monthly_rows"],
        history=history,
    )
    write_json(
        run_dir / f"visualization_manifest_{timestamp}.json",
        {
            "timestamp_day_hour_minute": timestamp,
            "figure_count": len(figure_files),
            "figures": figure_files,
        },
    )
    log_progress("Finished writing output files.")


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    random.seed(config.seed)
    np.random.seed(config.seed)

    log_progress(
        "Starting DynamicGMCR 42-dimensional smooth preference estimator for subsequent GMCR analysis."
    )
    log_progress(f"Input directory: {config.input_dir}")
    log_progress(f"Output directory: {config.output_dir}")
    log_progress(
        "Hyperparameters: "
        f"delta={config.delta}, delta_persistence={config.delta_persistence}, "
        f"perturbation_seed={config.perturbation_seed}, theta={config.theta}, "
        f"transition_temperature={config.transition_temperature}, "
        f"theta_forward={config.theta_forward}, stable_margin={config.stable_margin}, "
        f"forward_horizon={config.forward_horizon}"
    )
    estimated_evaluations = (
        config.population_size
        + config.population_size * config.generations
        + config.local_search_steps
    )
    log_progress(
        "Solver scale: "
        f"population={config.population_size}, generations={config.generations}, "
        f"local_search_steps={config.local_search_steps}, "
        f"estimated_objective_evaluations={estimated_evaluations}"
    )

    log_progress("Loading model input files.")
    data = load_model_data(config)
    log_progress(
        "Loaded inputs: "
        f"months={len(data.months)}, states={len(data.state_ids)}, "
        f"historical_state_steps={max(len(data.months) - 1, 0)}"
    )
    lower, upper = parameter_bounds(config)
    log_progress(
        "Prepared parameter bounds: "
        f"dimension={len(lower)}, fixed_parameters={int(np.sum((upper - lower) <= EPS))}"
    )

    best, _, de_history = differential_evolution(data, config, lower, upper)
    best, _, ls_history = local_search(best, data, config, lower, upper)
    log_progress("Evaluating final solution with detailed monthly output.")
    _, best_details = evaluate(best, data, config, lower, upper, return_rows=True)
    log_progress(
        "Final solution summary: "
        f"loss={best_details['loss']:.6f}, "
        f"fit_objective={best_details['fit_objective']:.6f}, "
        f"log_transition={best_details['log_transition_objective']:.6f}, "
        f"forward_shortfall={best_details['forward_shortfall_penalty']:.6f}, "
        f"seq_stability_penalty={best_details['seq_stability_penalty']:.6f}"
    )

    run_dir, timestamp = make_run_dir(Path(config.output_dir))
    log_progress(f"Created timestamped run directory: {run_dir}")
    save_outputs(
        run_dir=run_dir,
        timestamp=timestamp,
        data=data,
        config=config,
        best_vector=best,
        best_details=best_details,
        history=de_history + ls_history,
    )
    log_progress("Solver finished.")


if __name__ == "__main__":
    main()
