from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "cod" / "visualize_stability_states.py"
)
SPEC = importlib.util.spec_from_file_location("demo4_stability_visualization", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
visualizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = visualizer
SPEC.loader.exec_module(visualizer)


def classification_frame() -> pd.DataFrame:
    rows = []
    for state_id in (2, 1):
        row = {
            "month": 202002,
            "period": "prediction",
            "state_id": state_id,
            "is_observed_state": False,
        }
        for actor_index, actor in enumerate(visualizer.ACTORS):
            for concept_index, concept in enumerate(visualizer.CONCEPTS):
                stable = (state_id + actor_index + concept_index) % 2 == 0
                row[f"{actor}_{concept}_stable"] = stable
                row[f"{actor}_{concept}_margin"] = 1.0 if stable else -1.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_prediction_matrix_orders_states_and_actor_concepts() -> None:
    data = classification_frame()
    visualizer.validate_classification(data)

    values, states, labels = visualizer.prediction_matrix(data, "_stable")

    assert states == [1, 2]
    assert labels == [
        "CN\nNash",
        "CN\nGMR",
        "CN\nSMR",
        "CN\nSEQ",
        "US\nNash",
        "US\nGMR",
        "US\nSMR",
        "US\nSEQ",
    ]
    assert values.shape == (2, 8)
    assert np.array_equal(values[0, :4], [False, True, False, True])


def test_bool_matrix_uses_requested_state_order() -> None:
    data = classification_frame()
    values = visualizer.bool_matrix(
        data, "CN_Nash_stable", months=[202002], states=[1, 2]
    )
    assert np.array_equal(values[:, 0], [0, 1])
