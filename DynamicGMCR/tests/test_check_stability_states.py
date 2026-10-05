from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1] / "cod"
sys.path.insert(0, str(CODE_DIR))
SCRIPT_PATH = CODE_DIR / "check_stability_states.py"
SPEC = importlib.util.spec_from_file_location("demo4_stability_check", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def test_next_month_handles_year_boundary() -> None:
    assert checker.next_month(202001) == 202002
    assert checker.next_month(202012) == 202101


def test_extended_path_preserves_historical_values() -> None:
    solver = checker.solver
    config = solver.SolverConfig(input_dir=".", output_dir=".")
    rng = np.random.default_rng(42)
    impacts = rng.normal(size=(5, solver.N_ACTORS, solver.N_DOMAINS))
    lower, upper = solver.parameter_bounds(config)
    params = solver.decode((lower + upper) / 2.0, lower, upper)

    historical = solver.compute_h(params, impacts)
    extended = solver.compute_h(
        params, np.concatenate([impacts, np.zeros_like(impacts[:1])], axis=0)
    )

    assert np.allclose(extended[:-1], historical)
    expected_lag = solver.distributed_lag_impacts(impacts, len(impacts) - 1)
    innovation = np.einsum("ad,adk->ak", expected_lag, params.a)
    expected_target = params.mu + np.tanh(innovation)
    expected_forecast = (
        params.alpha[:, None] * historical[-1]
        + (1.0 - params.alpha[:, None]) * expected_target
    )
    assert np.allclose(extended[-1], expected_forecast)


def test_state_classification_lists_each_actors_satisfied_concepts() -> None:
    rows = []
    actor_flags = {
        "CN": (True, True, False, False),
        "US": (True, False, True, False),
    }
    for actor, flags in actor_flags.items():
        row = {
            "month": 202002,
            "period": "prediction",
            "actor": actor,
            "state_id": 7,
            "is_observed_state": False,
        }
        for concept, stable in zip(checker.CONCEPTS, flags):
            row[f"{concept}_stable"] = stable
            row[f"{concept}_margin"] = 1.0 if stable else -1.0
        rows.append(row)

    result = checker.build_state_classification(pd.DataFrame(rows)).iloc[0]

    assert result["CN_stability_types"] == "Nash|GMR"
    assert result["US_stability_types"] == "Nash|SMR"
    assert result["joint_equilibrium_types"] == "Nash"
