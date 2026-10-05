from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "cod"
sys.path.insert(0, str(CODE_DIR))
SCRIPT_PATH = CODE_DIR / "predict_preference_matrix.py"
SPEC = importlib.util.spec_from_file_location("demo4_preference_prediction", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
prediction = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prediction
SPEC.loader.exec_module(prediction)


def test_pairwise_probabilities_use_strict_seed_counting() -> None:
    utilities = np.array(
        [
            [[3.0, 2.0, 2.0]],
            [[1.0, 2.0, 3.0]],
            [[4.0, 2.0, 1.0]],
        ]
    )

    preferred, ties, probabilities = prediction.compute_pairwise_statistics(utilities)

    assert preferred[0, 0, 1] == 2
    assert preferred[0, 1, 0] == 1
    assert probabilities[0, 0, 1] == 2 / 3
    assert ties[0, 1, 2] == 1
    assert np.all(np.diag(probabilities[0]) == 0.0)


def test_probability_matrix_rows_are_s_and_columns_are_q() -> None:
    probabilities = np.array([[0.0, 0.75], [0.25, 0.0]])

    frame = prediction.probability_matrix_frame([10, 20], probabilities)

    assert frame.columns.tolist() == ["state_id", "state_10", "state_20"]
    assert frame.loc[0, "state_id"] == 10
    assert frame.loc[0, "state_20"] == 0.75
    assert frame.loc[1, "state_10"] == 0.25


def test_discovery_skips_incomplete_seed_and_selects_completed_seed(tmp_path: Path) -> None:
    complete = tmp_path / "seed_11" / "run"
    complete.mkdir(parents=True)
    parameter_file = complete / "parameters_run.json"
    parameter_file.write_text("{}", encoding="utf-8")
    (tmp_path / "seed_12").mkdir()
    (tmp_path / "not_a_seed").mkdir()

    selected, skipped = prediction.discover_parameter_files(tmp_path)

    assert selected == [(11, parameter_file)]
    assert skipped[0]["seed"] == 12
    assert skipped[0]["reason"] == "no parameter file"
