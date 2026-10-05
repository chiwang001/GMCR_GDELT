from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


INSTALL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = next(
    path
    for path in (
        INSTALL_ROOT / "analyze_equilibrium_robustness.py",
        INSTALL_ROOT / "cod" / "analyze_equilibrium_robustness.py",
    )
    if path.exists()
)
SPEC = importlib.util.spec_from_file_location("equilibrium_robustness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def available_project_root() -> Path | None:
    configured = os.environ.get("GMCR_PROJECT_ROOT")
    candidates = [
        Path(configured) if configured else None,
        INSTALL_ROOT if (INSTALL_ROOT / "cod").is_dir() else None,
        INSTALL_ROOT.parent / "tmp" / "GMCR_GDELT",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if (candidate / "DynamicGMCR").is_dir() or (
            (candidate / "cod").is_dir() and (candidate / "input").is_dir()
        ):
            return candidate
    return None


class CoreFunctionTests(unittest.TestCase):
    def test_pairwise_orientation_uses_rows_as_preferred_states(self) -> None:
        utilities = np.array(
            [
                [[3.0, 1.0]],
                [[2.0, 1.0]],
                [[0.0, 1.0]],
            ]
        )
        matrix = analysis.pairwise_matrix(analysis.pairwise_indicators(utilities))
        self.assertAlmostEqual(matrix[0, 0, 1], 2 / 3)
        self.assertAlmostEqual(matrix[0, 1, 0], 1 / 3)
        self.assertEqual(matrix[0, 0, 0], 0.0)

    def test_deterministic_sampling_is_without_replacement(self) -> None:
        first = analysis.deterministic_samples(20, [5, 10], 4, 123)
        second = analysis.deterministic_samples(20, [5, 10], 4, 123)
        for size in first:
            np.testing.assert_array_equal(first[size], second[size])
            for sample in first[size]:
                self.assertEqual(len(np.unique(sample)), size)

    def test_empty_union_jaccard_is_one(self) -> None:
        empty = np.zeros(4, dtype=bool)
        self.assertEqual(analysis.jaccard(empty, empty), 1.0)
        self.assertEqual(
            analysis.jaccard(np.array([True, False]), np.array([True, True])),
            0.5,
        )

    def test_conditional_certificate_checks_ui_gap_and_margin_clearance(self) -> None:
        margins = np.full((2, 2, 4), 0.20)
        margins[0, 0, 0] = 0.01
        local_gaps = np.array([[0.10, 0.03], [0.10, np.inf]])
        feasible = np.array([[1, 1], [1, 0]])
        global_ui, local_ui, sign_clear, certified = analysis.certificate_flags(
            margins, 0.05, 0.02, local_gaps, feasible
        )
        self.assertFalse(global_ui)
        self.assertTrue(local_ui[0, 0])
        self.assertFalse(local_ui[0, 1])
        self.assertFalse(sign_clear[0, 0, 0])
        self.assertFalse(certified[0, 0, 0])
        self.assertTrue(certified[0, 0, 1])
        self.assertTrue(certified[1, 1].all())

    def test_key_state_summary_schema(self) -> None:
        rows = []
        for state_id in analysis.KEY_STATES:
            for concept in analysis.CONCEPTS:
                for replicate in (1, 2):
                    rows.append(
                        {
                            "ensemble_size": 10,
                            "replicate": replicate,
                            "state_id": state_id,
                            "concept": concept,
                            "joint_margin": 0.1,
                            "joint_stable": True,
                            "joint_classification_preserved": True,
                            "both_actor_certificates": True,
                        }
                    )
        margins = np.full((2, 4, 4), 0.1)
        stable = np.ones_like(margins, dtype=bool)
        result = analysis.summarize_key_states(
            pd.DataFrame(rows), np.array(analysis.KEY_STATES), margins, stable
        )
        required = {
            "full_CN_margin",
            "full_US_margin",
            "full_joint_margin",
            "joint_margin_p05",
            "joint_margin_median",
            "joint_margin_p95",
            "joint_stability_rate",
            "joint_classification_preservation_rate",
            "certificate_holds_all_replicates",
        }
        self.assertTrue(required.issubset(result.columns))
        self.assertEqual(len(result), 16)


@unittest.skipIf(available_project_root() is None, "GMCR_GDELT checkout not available")
class RepositoryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert available_project_root() is not None
        cls.paths = analysis.discover_paths(available_project_root())
        cls.gmcr = analysis.import_probability_gmcr(cls.paths.code_dir)

    def test_full_ensemble_exactly_reconstructs_archived_current_matrices(self) -> None:
        utilities, _, state_ids, month = analysis.load_utility_ensemble(self.paths.utilities)
        reconstructed = analysis.pairwise_matrix(analysis.pairwise_indicators(utilities))
        stored = []
        for actor in analysis.ACTORS:
            ids, matrix = self.gmcr.load_probability_matrix(self.paths.matrices[actor])
            np.testing.assert_array_equal(ids, state_ids)
            stored.append(matrix)
        self.assertEqual(month, 202002)
        np.testing.assert_array_equal(reconstructed, np.stack(stored))

    def test_margin_wrapper_matches_existing_probability_gmcr_analysis(self) -> None:
        utilities, _, state_ids, _ = analysis.load_utility_ensemble(self.paths.utilities)
        matrix = analysis.pairwise_matrix(analysis.pairwise_indicators(utilities))
        neighbors = self.gmcr.load_neighbors(self.paths.graph, state_ids)
        margins, _ = analysis.compute_margins(self.gmcr, matrix, neighbors, 0.5)
        details, _, _ = self.gmcr.analyze(state_ids, matrix, neighbors, 0.5)
        for ai, actor in enumerate(analysis.ACTORS):
            actor_rows = details.loc[details["actor"] == actor].set_index("state_id")
            for si, state_id in enumerate(state_ids):
                for ci, concept in enumerate(analysis.CONCEPTS):
                    self.assertAlmostEqual(
                        margins[ai, si, ci],
                        float(actor_rows.loc[int(state_id), f"{concept}_margin"]),
                    )

    def test_empty_moves_and_missing_sanctions_follow_finite_convention(self) -> None:
        probabilities = np.zeros((2, 2, 2), dtype=float)
        empty_neighbors = [
            [np.array([], dtype=int), np.array([], dtype=int)],
            [np.array([], dtype=int), np.array([], dtype=int)],
        ]
        margins, improvements = self.gmcr.stability_margins(
            0, 0, probabilities, empty_neighbors, 0.5
        )
        np.testing.assert_array_equal(improvements, np.array([], dtype=int))
        np.testing.assert_allclose(margins, np.full(4, 0.5))

        probabilities[0, 1, 0] = 0.8
        missing_sanction_neighbors = [
            [np.array([1], dtype=int), np.array([], dtype=int)],
            [np.array([], dtype=int), np.array([], dtype=int)],
        ]
        margins, improvements = self.gmcr.stability_margins(
            0, 0, probabilities, missing_sanction_neighbors, 0.5
        )
        np.testing.assert_array_equal(improvements, np.array([1]))
        np.testing.assert_allclose(margins, np.array([-0.3, -0.5, -0.5, -0.5]))


if __name__ == "__main__":
    unittest.main()
