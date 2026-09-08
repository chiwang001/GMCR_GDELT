"""Run controlled ablations of the DynamicGMCR core preference model.

Each variant reuses the solver's objective and optimizer with the same seed and
budget.  Ablations are implemented through model-data/config switches rather
than a second objective implementation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import solve_core_preference_model as solver


VARIANTS = (
    "full",
    "no_historical_memory",
    "no_forward_reward",
    "no_score_perturbation",
    "no_nonlinear_saturation",
    "no_stability_penalty",
)

DEFAULT_PRESET = "full"
PRESET_BUDGETS = {
    "quick": (24, 20, 20, 2),
    "standard": (60, 60, 500, 5),
    "full": (180, 250, 500, 25),
}


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run controlled ablation experiments for the DynamicGMCR core model."
    )
    parser.add_argument("--input-dir", type=Path, default=demo_dir / "input")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=demo_dir / "output" / "ablation" / "full_budget",
    )
    parser.add_argument(
        "--preset",
        choices=("quick", "standard", "full"),
        default=DEFAULT_PRESET,
        help=(
            "Optimizer budget shared by all variants (default: full; "
            "use quick only for a pipeline check)."
        ),
    )
    parser.add_argument("--seed", type=int, default=20250728)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated variant names."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute variants whose output directory already contains summary.json.",
    )
    return parser.parse_args()


def budget(preset: str) -> tuple[int, int, int, int]:
    try:
        return PRESET_BUDGETS[preset]
    except KeyError as exc:
        raise ValueError(
            f"Unknown preset {preset!r}; choose from {tuple(PRESET_BUDGETS)}"
        ) from exc


def build_config(args: argparse.Namespace, variant: str) -> solver.SolverConfig:
    population, generations, local_search_steps, progress_interval = budget(args.preset)
    config = solver.SolverConfig(
        input_dir=str(args.input_dir.resolve()),
        output_dir=str(args.output_dir.resolve()),
        seed=int(args.seed),
        population_size=population,
        generations=generations,
        local_search_steps=local_search_steps,
        progress_interval=progress_interval,
    )
    if variant == "no_forward_reward":
        config = replace(config, lambda_forward=0.0)
    elif variant == "no_score_perturbation":
        config = replace(config, delta=0.0)
    elif variant == "no_nonlinear_saturation":
        config = replace(config, nonlinear_saturation=False)
    elif variant == "no_stability_penalty":
        config = replace(config, lambda_stable=0.0)
    return config


def prepare_data(
    config: solver.SolverConfig, variant: str
) -> solver.ModelData:
    data = solver.load_model_data(config)
    if variant == "no_historical_memory":
        # Keep only the immediately preceding month's scaled event impact.
        data.lagged_impacts = data.impacts_scaled[:-1].copy()
    elif variant == "no_score_perturbation":
        data.score_perturbations = np.zeros_like(data.score_perturbations)
    return data


def clean_details(details: dict[str, Any]) -> dict[str, Any]:
    excluded = {"params", "h", "monthly_rows"}
    return {key: value for key, value in details.items() if key not in excluded}


def run_variant(
    args: argparse.Namespace, variant: str, variant_dir: Path
) -> dict[str, Any]:
    config = build_config(args, variant)
    data = prepare_data(config, variant)
    lower, upper = solver.parameter_bounds(config)
    started = time.perf_counter()
    best, _, de_history = solver.differential_evolution(data, config, lower, upper)
    best, _, local_history = solver.local_search(
        best, data, config, lower, upper
    )
    _, details = solver.evaluate(
        best, data, config, lower, upper, return_rows=True
    )
    elapsed = time.perf_counter() - started

    variant_dir.mkdir(parents=True, exist_ok=True)
    summary = clean_details(details)
    summary.update(
        {
            "variant": variant,
            "preset": args.preset,
            "seed": int(args.seed),
            "elapsed_seconds": float(elapsed),
            "config": asdict(config),
            "data_ablation": {
                "historical_memory": variant != "no_historical_memory",
                "score_perturbation": variant != "no_score_perturbation",
            "nonlinear_saturation": bool(config.nonlinear_saturation),
            "forward_term": float(config.lambda_forward) > 0.0,
            "stability_penalty": float(config.lambda_stable) > 0.0,
            },
        }
    )
    (variant_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (variant_dir / "parameters.json").write_text(
        json.dumps(
            {"variant": variant, "vector": [float(x) for x in best]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(details.get("monthly_rows", [])).to_csv(
        variant_dir / "monthly_fit.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(de_history + local_history).to_csv(
        variant_dir / "optimization_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def numeric_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "loss",
        "fit_objective",
        "log_transition_objective",
        "transition_probability_sum",
        "forward_shortfall_penalty",
        "seq_stability_penalty",
        "regularization_a",
        "regularization_mu",
        "regularization_alpha",
        "transition_observation_count",
        "forward_observation_count",
        "stability_actor_observation_count",
        "elapsed_seconds",
    )
    row = {"variant": summary["variant"]}
    for key in keys:
        if key in summary:
            row[key] = summary[key]
    return row


def main() -> None:
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(variants).difference(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; choose from {VARIANTS}")
    if not variants:
        raise ValueError("At least one variant is required.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        variant_dir = output_dir / variant
        summary_path = variant_dir / "summary.json"
        if summary_path.exists() and not args.force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"Skipping completed variant: {variant}")
        else:
            print(f"Running ablation variant: {variant}", flush=True)
            summary = run_variant(args, variant, variant_dir)
        summaries.append(summary)

    table = pd.DataFrame([numeric_summary(summary) for summary in summaries])
    full = table.loc[table["variant"] == "full"]
    if not full.empty:
        full_row = full.iloc[0]
        for column in table.columns:
            if column in {"variant", "elapsed_seconds"}:
                continue
            if pd.api.types.is_numeric_dtype(table[column]):
                table[f"delta_vs_full_{column}"] = table[column] - full_row[column]
    table.to_csv(output_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(output_dir),
        "preset": args.preset,
        "seed": int(args.seed),
        "variants": variants,
        "variant_definitions": {
            "full": "All core-model components enabled.",
            "no_historical_memory": "Replace the 0.6/0.3/0.1 distributed lag with the immediately preceding impact only.",
            "no_forward_reward": "Set lambda_forward to zero, removing the forward shortfall term.",
            "no_score_perturbation": "Set the ex ante state-score disturbance process to zero.",
            "no_nonlinear_saturation": "Replace tanh innovation saturation with an unsaturated linear innovation response.",
            "no_stability_penalty": "Set lambda_stable to zero, removing the unchanged-month SEQ stability shortfall from the objective.",
        },
        "aggregation": "All variants use the same optimizer seed and preset; deltas are reported against full.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'ablation_summary.csv'}")
    print(f"Wrote {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
