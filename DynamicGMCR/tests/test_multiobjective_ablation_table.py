from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "cod"
sys.path.insert(0, str(CODE_DIR))

import generate_multiobjective_ablation_table as table  # noqa: E402


def test_multiobjective_table_has_all_models_and_metrics() -> None:
    path = Path(__file__).resolve().parents[1] / "results" / "ablation" / "full_budget" / "multiobjective_ablation_summary.csv"
    frame = pd.read_csv(path)
    assert frame["model"].tolist() == [
        "Full model",
        "No memory",
        "No forward-looking term",
        "No score disturbance",
        "No nonlinear saturation",
        "No stability penalty",
        "Static strategy-priority baseline",
    ]
    for metric in table.METRICS:
        assert f"{metric}_mean" in frame.columns
        assert f"{metric}_p05" in frame.columns
        assert f"{metric}_p95" in frame.columns


def test_metric_directions_and_static_undefined_terms() -> None:
    static = table.static_metrics()
    assert static["transition_support"] == 5 / 11
    assert static["mean_conditional_support"] > 0
    assert static["matrix_dispersion"] == 0
    assert pd.isna(static["forward_shortfall"])
    assert pd.isna(static["stability_shortfall"])
