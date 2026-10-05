from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "cod"
sys.path.insert(0, str(CODE_DIR))
SCRIPT_PATH = CODE_DIR / "compare_seed_count_preference_matrices.py"
SPEC = importlib.util.spec_from_file_location("demo4_matrix_comparison", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def test_resolve_seed_counts_appends_full_count() -> None:
    assert comparison.resolve_seed_counts(None, total=23, step=10) == [10, 20, 23]
    assert comparison.resolve_seed_counts("5,2", total=8) == [2, 5, 8]


def test_cumulative_matrices_use_nested_seed_prefixes() -> None:
    utilities = np.array(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 0.0]],
        ]
    )

    matrices = comparison.cumulative_preference_matrices(utilities, [1, 2, 3])

    assert np.allclose(matrices[1][0], [[0.0, 1.0], [0.0, 0.0]])
    assert np.allclose(matrices[2][0], [[0.0, 0.5], [0.5, 0.0]])
    assert np.allclose(matrices[3][0], [[0.0, 2 / 3], [1 / 3, 0.0]])


def test_squared_difference_sum_uses_corresponding_elements() -> None:
    first = np.array([[[0.0, 1.0], [0.0, 0.0]]])
    second = np.array([[[0.0, 0.5], [0.5, 0.0]]])

    actor_sums, combined = comparison.squared_difference_sums(first, second)

    assert np.allclose(actor_sums, [0.5])
    assert combined == 0.5


def test_distance_tables_include_full_reference_zero() -> None:
    matrices = {
        1: np.array(
            [
                [[0.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ]
        ),
        2: np.array(
            [
                [[0.0, 0.5], [0.5, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ]
        ),
    }

    all_pairs, consecutive, reference = comparison.build_distance_tables(matrices)

    assert len(all_pairs) == 1
    assert all_pairs.iloc[0]["combined_squared_difference_sum"] == 0.5
    assert len(consecutive) == 1
    assert consecutive.iloc[0]["seed_count_a"] == 1
    assert consecutive.iloc[0]["seed_count_b"] == 2
    assert len(reference) == 2
    assert reference.iloc[-1]["combined_squared_difference_sum"] == 0.0
