from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


SOLVER_PATH = Path(__file__).resolve().parents[1] / "cod" / "solve_core_preference_model.py"
SPEC = importlib.util.spec_from_file_location("demo4_solver", SOLVER_PATH)
assert SPEC is not None and SPEC.loader is not None
solver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = solver
SPEC.loader.exec_module(solver)


def test_parameter_structure_remains_42_dimensional() -> None:
    assert solver.PARAMETER_DIM == 42
    assert solver.N_ACTIVE_A_PARAMETERS == 30
    assert np.all(np.sum(solver.A_ACTIVE_MASK, axis=(1, 2)) == 15)


def test_inactive_mapping_positions_decode_to_zero() -> None:
    config = solver.SolverConfig(input_dir=".", output_dir=".")
    lower, upper = solver.parameter_bounds(config)
    params = solver.decode((lower + upper) / 2.0, lower, upper)
    assert np.all(params.a[~solver.A_ACTIVE_MASK] == 0.0)


def test_mean_reverting_preferences_are_bounded() -> None:
    config = solver.SolverConfig(input_dir=".", output_dir=".")
    lower, upper = solver.parameter_bounds(config)
    params = solver.decode(upper, lower, upper)
    impacts = np.full((30, solver.N_ACTORS, solver.N_DOMAINS), 20.0)
    h = solver.compute_h(params, impacts)
    assert np.all(h <= params.mu[None, :, :] + 1.0 + 1.0e-12)
    assert np.all(h >= params.mu[None, :, :] - 1.0 - 1.0e-12)


def test_nonlinear_saturation_switch_controls_innovation_response() -> None:
    config = solver.SolverConfig(input_dir=".", output_dir=".")
    lower, upper = solver.parameter_bounds(config)
    params = solver.decode((lower + upper) / 2.0, lower, upper)
    params.a.fill(0.0)
    params.alpha.fill(0.0)
    params.mu.fill(0.0)
    params.a[0, 0, 0] = 1.0
    impacts = np.zeros((2, solver.N_ACTORS, solver.N_DOMAINS))
    impacts[0, 0, 0] = 2.0

    saturated = solver.compute_h(params, impacts, nonlinear_saturation=True)
    linear = solver.compute_h(params, impacts, nonlinear_saturation=False)
    assert np.isclose(saturated[1, 0, 0], np.tanh(2.0))
    assert np.isclose(linear[1, 0, 0], 2.0)


def test_smooth_transition_rows_are_positive_and_normalized() -> None:
    utilities = np.array([[0.0, 1.0, -1.0], [0.2, -0.3, 0.5]])
    neighbors = [
        [np.array([1, 2]), np.array([0]), np.array([], dtype=int)],
        [np.array([1, 2]), np.array([0]), np.array([], dtype=int)],
    ]
    matrix = solver.transition_matrix(utilities, neighbors, temperature=0.35)
    assert np.all(matrix[:, 0, [1, 2]] > 0.0)
    assert np.allclose(np.sum(matrix[:, 0], axis=1), 1.0)
    assert np.allclose(np.sum(matrix[:, 1], axis=1), 1.0)
    assert np.allclose(np.sum(matrix[:, 2], axis=1), 0.0)


def test_vectorized_utilities_match_month_by_month_computation() -> None:
    rng = np.random.default_rng(17)
    omega = rng.normal(size=(solver.N_ACTORS, 7, solver.N_DRIVERS))
    perturbations = rng.normal(
        size=(4, solver.N_ACTORS, 7, solver.N_DRIVERS)
    )
    preferences = rng.normal(size=(4, solver.N_ACTORS, solver.N_DRIVERS))

    raw, standardized = solver.all_perturbed_utilities(
        omega, perturbations, preferences
    )
    for t in range(len(preferences)):
        expected_raw = solver.perturbed_utilities(
            omega, perturbations[t], preferences[t]
        )
        assert np.allclose(raw[t], expected_raw)
        assert np.allclose(
            standardized[t], solver.standardize_utilities(expected_raw)
        )


def reference_loss(
    vector: np.ndarray,
    data: solver.ModelData,
    config: solver.SolverConfig,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    params = solver.decode(vector, lower, upper)
    preferences = solver.compute_h(params, data.impacts_scaled)
    perturbations = solver.perturbations_for_data(data, config)
    transition_terms = []
    forward_penalties = []
    stability_penalties = []

    for t in range(1, len(data.months)):
        start = int(data.observed_indices[t - 1])
        observed = int(data.observed_indices[t])
        raw = solver.perturbed_utilities(
            data.omega, perturbations[t], preferences[t]
        )
        utilities = solver.standardize_utilities(raw)
        if start != observed:
            matrix = solver.transition_matrix(
                utilities, data.neighbors, config.transition_temperature
            )
            probability, _, _ = solver.transition_probability(
                matrix, data.neighbors, start, observed
            )
            transition_terms.append(
                math.log(max(float(probability), config.epsilon_p))
            )
            penalty, _, _ = solver.forward_shortfall(
                utilities, start, observed, data, config
            )
            forward_penalties.append(penalty)
        else:
            for actor in range(solver.N_ACTORS):
                margins = solver.stability_margins(
                    start, actor, utilities, data.neighbors, config
                )
                stability_penalties.append(max(-float(margins[3]), 0.0) ** 2)

    reg_a, reg_mu, reg_alpha = solver.regularization_components(params)
    objective = (
        np.mean(transition_terms)
        - config.lambda_forward * np.mean(forward_penalties)
        - config.lambda_stable * np.mean(stability_penalties)
        - config.lambda_a * reg_a
        - config.lambda_mu * reg_mu
        - config.lambda_alpha * reg_alpha
    )
    return float(-objective)


def test_optimized_objective_matches_full_matrix_reference() -> None:
    input_dir = Path(__file__).resolve().parents[1] / "input"
    config = solver.SolverConfig(
        input_dir=str(input_dir), output_dir=".", seed=123
    )
    data = solver.load_model_data(config)
    lower, upper = solver.parameter_bounds(config)
    rng = np.random.default_rng(23)
    vectors = [
        (lower + upper) / 2.0,
        lower + rng.random(len(lower)) * (upper - lower),
        lower + rng.random(len(lower)) * (upper - lower),
    ]

    for vector in vectors:
        optimized, _ = solver.evaluate(vector, data, config, lower, upper)
        reference = reference_loss(vector, data, config, lower, upper)
        assert optimized == reference


def test_targeted_transition_and_forward_paths_match_full_computation() -> None:
    input_dir = Path(__file__).resolve().parents[1] / "input"
    config = solver.SolverConfig(input_dir=str(input_dir), output_dir=".")
    data = solver.load_model_data(config)
    rng = np.random.default_rng(29)

    for plan in data.evaluation_plans or ():
        if not plan.changed:
            continue
        utilities = solver.standardize_utilities(
            rng.normal(size=(solver.N_ACTORS, len(data.state_ids)))
        )
        matrix = solver.transition_matrix(
            utilities, data.neighbors, config.transition_temperature
        )
        expected = solver.transition_probability(
            matrix, data.neighbors, plan.start, plan.observed
        )
        actual = solver.targeted_transition_probability(
            utilities, data.neighbors, plan, config.transition_temperature
        )
        assert actual == expected

        expected_forward = solver.forward_shortfall(
            utilities, plan.start, plan.observed, data, config
        )
        actual_forward = solver.forward_shortfall_from_candidates(
            utilities,
            plan.start,
            plan.observed,
            plan.forward_candidates,
            config,
        )
        assert actual_forward == expected_forward
