from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "cod"
    / "visualize_preference_matrix_convergence.py"
)
SPEC = importlib.util.spec_from_file_location("demo4_convergence_visualization", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
visualization = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = visualization
SPEC.loader.exec_module(visualization)


def test_combined_distance_grid_is_symmetric() -> None:
    table = pd.DataFrame(
        {
            "seed_count_a": [10, 10, 20],
            "seed_count_b": [20, 30, 30],
            "combined_squared_difference_sum": [4.0, 9.0, 1.0],
        }
    )

    counts, grid = visualization.build_combined_distance_grid(table)

    assert counts == [10, 20, 30]
    assert np.isnan(np.diag(grid)).all()
    assert np.allclose(grid[0, 1], grid[1, 0])
    assert grid[0, 2] == 9.0


def test_sparse_ticks_preserve_endpoints() -> None:
    ticks = visualization.sparse_ticks(range(10, 281, 10), maximum=8)

    assert len(ticks) <= 8
    assert ticks[0] == 10
    assert ticks[-1] == 280
