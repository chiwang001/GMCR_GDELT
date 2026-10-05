from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1] / "cod"
sys.path.insert(0, str(CODE_DIR))

import run_static_strategy_baseline as baseline


class StaticBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.input_dir = Path(__file__).resolve().parents[1] / "input"

    def test_loads_demo4_state_graph(self) -> None:
        data = baseline.load_data(self.input_dir)
        self.assertEqual(len(data.state_ids), 76)
        self.assertEqual(sum(len(x) for x in data.neighbors[0]), 229)
        self.assertEqual(sum(len(x) for x in data.neighbors[1]), 182)
        self.assertEqual(len(data.observed_months), 37)

    def test_default_strategy_priorities_define_static_preferences(self) -> None:
        data = baseline.load_data(self.input_dir)
        priorities = baseline.default_strategy_priorities()
        self.assertEqual(len([p for p in priorities if p.country == "CN"]), 8)
        self.assertEqual(len([p for p in priorities if p.country == "US"]), 8)
        self.assertTrue(all(p.explanation for p in priorities))
        utilities, satisfaction = baseline.compute_utilities(data.options, priorities)
        self.assertEqual(utilities.shape, (2, 76))
        self.assertEqual(satisfaction["CN"].shape, (76, 8))
        self.assertEqual(satisfaction["US"].shape, (76, 8))
        self.assertTrue(np.all(np.isin(satisfaction["CN"], [0, 1])))
        self.assertTrue(np.all(np.isin(satisfaction["US"], [0, 1])))
        weights = np.array([128, 64, 32, 16, 8, 4, 2, 1])
        np.testing.assert_array_equal(utilities[0], satisfaction["CN"] @ weights)
        np.testing.assert_array_equal(utilities[1], satisfaction["US"] @ weights)

    def test_transition_rows_are_conditional_markov_rows(self) -> None:
        data = baseline.load_data(self.input_dir)
        utilities, _ = baseline.compute_utilities(
            data.options, baseline.default_strategy_priorities()
        )
        actor_transition = baseline.transition_matrix(
            utilities, data.neighbors, 1e-9
        )
        probability, raw, _, totals = baseline.state_transition_matrix(
            actor_transition, data.transition_types
        )
        self.assertEqual(probability.shape, raw.shape)
        active = totals > baseline.EPS
        np.testing.assert_allclose(probability[active].sum(axis=1), 1.0)
        np.testing.assert_allclose(np.diag(probability), 0.0)

    def test_run_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = baseline.BaselineConfig(
                input_dir=self.input_dir,
                output_dir=Path(tmp),
            )
            summary = baseline.run_baseline(config)
            self.assertEqual(summary["model"], "static_strategy_priority_gmcr_baseline")
            self.assertEqual(summary["preference_method"], "GMCR strategy priority declarations")
            for filename in (
                "strategy_priority_declarations.csv",
                "state_priority_satisfaction.csv",
                "static_state_utilities.csv",
                "static_state_stability.csv",
                "historical_explanation.csv",
                "static_baseline_summary.json",
            ):
                self.assertTrue((Path(tmp) / filename).exists(), filename)
            document = json.loads(
                (Path(tmp) / "static_baseline_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(document["transitions"], 36)
            declarations = baseline.read_csv(
                Path(tmp) / "strategy_priority_declarations.csv"
            )
            self.assertEqual(
                declarations["lexicographic_weight"].tolist(),
                [128, 64, 32, 16, 8, 4, 2, 1] * 2,
            )


if __name__ == "__main__":
    unittest.main()
