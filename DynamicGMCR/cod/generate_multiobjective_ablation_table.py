#!/usr/bin/env python3
"""Build a transparent multi-objective ablation table for DynamicGMCR.

The table keeps every objective dimension separate.  Full-model uncertainty is
estimated from all completed top-level seed runs.  Existing ablation outputs
are the full-budget, fixed-seed refits; their seed intervals therefore collapse
to the reported point until additional multi-seed ablation refits are run.
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
OUTPUT = ROOT / "output"
ABLATION = OUTPUT / "ablation" / "full_budget"
STATIC = OUTPUT / "static_preference_model"
# The public package keeps the checked-in static baseline under results/;
# callers can override this with --static-dir when rebuilding it elsewhere.
STATIC = ROOT / "results" / "static_baseline"
sys.path.insert(0, str(CODE_DIR))
import solve_core_preference_model as solver  # noqa: E402


CONCEPTS = ("nash", "gmr", "smr", "seq")
VARIANT_LABELS = {
    "full": "Full model",
    "no_historical_memory": "No memory",
    "no_forward_reward": "No forward-looking term",
    "no_score_perturbation": "No score disturbance",
    "no_nonlinear_saturation": "No nonlinear saturation",
    "no_stability_penalty": "No stability penalty",
    "static": "Static strategy-priority baseline",
}
METRICS = (
    "transition_support",
    "mean_conditional_support",
    "nash_retention",
    "gmr_retention",
    "smr_retention",
    "seq_retention",
    "forward_shortfall",
    "stability_shortfall",
    "matrix_dispersion",
)


def parse_flags(value: object) -> list[bool]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return [False] * 4
    flags: list[bool] = []
    for part in str(value).split("|")[:4]:
        try:
            flags.append(float(part) >= 0.0)
        except ValueError:
            flags.append(False)
    return flags + [False] * (4 - len(flags))


def fit_metrics(frame: pd.DataFrame, summary: dict[str, Any] | None = None) -> dict[str, float]:
    changed = frame[frame["observed_change"].astype(int).eq(1)].copy()
    stays = frame[frame["observed_change"].astype(int).eq(0)].copy()
    probabilities = pd.to_numeric(changed["transition_probability"], errors="coerce").dropna()
    result: dict[str, float] = {
        "transition_support": float(np.mean(probabilities.to_numpy() > 1.0e-8)) if len(probabilities) else np.nan,
        "mean_conditional_support": float(probabilities.mean()) if len(probabilities) else np.nan,
    }
    for index, concept in enumerate(CONCEPTS):
        joint: list[bool] = []
        for _, row in stays.iterrows():
            cn = parse_flags(row.get("cn_margins_Nash_GMR_SMR_SEQ"))
            us = parse_flags(row.get("us_margins_Nash_GMR_SMR_SEQ"))
            joint.append(bool(cn[index] and us[index]))
        result[f"{concept}_retention"] = float(np.mean(joint)) if joint else np.nan
    forward = pd.to_numeric(frame.get("forward_shortfall_penalty", pd.Series(dtype=float)), errors="coerce").dropna()
    result["forward_shortfall"] = float(forward.mean()) if len(forward) else float(summary.get("forward_shortfall_penalty", np.nan) if summary else np.nan)
    result["stability_shortfall"] = float(summary.get("seq_stability_penalty", np.nan) if summary else np.nan)
    return result


def matrix_dispersion(path: Path) -> float:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    values = frame.pivot_table(index="month", columns=["country", "driver_index"], values="preference_value", aggfunc="first")
    array = values.to_numpy(dtype=float)
    return float(np.mean(np.std(array, axis=0, ddof=0)))


def latest_file(directory: Path, pattern: str) -> Path | None:
    files = list(directory.glob(pattern))
    return max(files, key=lambda p: (p.stat().st_mtime_ns, str(p))) if files else None


def load_seed_metrics() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for seed_dir in sorted(OUTPUT.glob("seed_*"), key=lambda p: p.name):
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        run_dirs = [p for p in seed_dir.iterdir() if p.is_dir()]
        fit_path = latest_file(seed_dir, "**/monthly_fit_*.csv")
        summary_path = latest_file(seed_dir, "**/summary_*.json")
        pref_path = latest_file(seed_dir, "**/monthly_preferences_*.csv")
        if fit_path is None or summary_path is None or pref_path is None:
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            frame = pd.read_csv(fit_path, encoding="utf-8-sig")
            metrics = fit_metrics(frame, summary.get("objective", {}))
            metrics["forward_shortfall"] = float(summary["objective"]["forward_shortfall_penalty"])
            metrics["stability_shortfall"] = float(summary["objective"]["seq_stability_penalty"])
            metrics["matrix_dispersion"] = matrix_dispersion(pref_path)
            metrics["seed"] = seed
            rows.append(metrics)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    if not rows:
        raise FileNotFoundError("No complete seed-level dynamic results were found.")
    return pd.DataFrame(rows)


def ablation_metrics(variant: str) -> dict[str, float]:
    directory = ABLATION / variant
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(directory / "monthly_fit.csv", encoding="utf-8-sig")
    metrics = fit_metrics(frame, summary)
    config = solver.SolverConfig(**summary["config"])
    if variant == "no_forward_reward":
        config = config.__class__(**{**config.__dict__, "lambda_forward": 0.0})
    elif variant == "no_score_perturbation":
        config = config.__class__(**{**config.__dict__, "delta": 0.0})
    elif variant == "no_nonlinear_saturation":
        config = config.__class__(**{**config.__dict__, "nonlinear_saturation": False})
    elif variant == "no_stability_penalty":
        config = config.__class__(**{**config.__dict__, "lambda_stable": 0.0})
    data = solver.load_model_data(config)
    if variant == "no_historical_memory":
        data.lagged_impacts = data.impacts_scaled[:-1].copy()
    elif variant == "no_score_perturbation":
        data.score_perturbations = np.zeros_like(data.score_perturbations)
    vector = np.asarray(json.loads((directory / "parameters.json").read_text(encoding="utf-8"))["vector"], dtype=float)
    lower, upper = solver.parameter_bounds(config)
    _, details = solver.evaluate(vector, data, config, lower, upper, return_rows=False)
    metrics["forward_shortfall"] = float(details["forward_shortfall_penalty"])
    metrics["stability_shortfall"] = float(details["seq_stability_penalty"])
    params = details["h"]
    metrics["matrix_dispersion"] = float(np.mean(np.std(params, axis=0, ddof=0)))
    metrics["seed"] = int(summary.get("seed", 0))
    return metrics


def static_metrics() -> dict[str, float]:
    history = pd.read_csv(STATIC / "historical_explanation.csv", encoding="utf-8-sig")
    stability = pd.read_csv(STATIC / "static_state_stability.csv", encoding="utf-8-sig")
    changed = history[history["observed_change"].eq(1)]
    probabilities = pd.to_numeric(changed["observed_transition_probability"], errors="coerce")
    result = {
        "transition_support": float(np.mean(probabilities > 1.0e-8)),
        "mean_conditional_support": float(probabilities.mean()),
        "forward_shortfall": np.nan,
        "stability_shortfall": np.nan,
        "matrix_dispersion": 0.0,
    }
    stays = history[history["observed_change"].eq(0)]
    for concept in CONCEPTS:
        result[f"{concept}_retention"] = float(np.mean([bool(stability.loc[stability["state_id"].eq(int(row.start_state_id)), f"joint_{concept}"].iloc[0]) for row in stays.itertuples()]))
    return result


def aggregate(values: pd.DataFrame, model: str, source: str, note: str) -> dict[str, Any]:
    row: dict[str, Any] = {"model": model, "source": source, "seed_count": int(len(values)), "note": note}
    for metric in METRICS:
        array = values[metric].to_numpy(dtype=float)
        if np.all(np.isnan(array)):
            row[f"{metric}_mean"] = np.nan
            row[f"{metric}_p05"] = np.nan
            row[f"{metric}_p95"] = np.nan
        else:
            row[f"{metric}_mean"] = float(np.nanmean(array))
            row[f"{metric}_p05"] = float(np.nanquantile(array, 0.05))
            row[f"{metric}_p95"] = float(np.nanquantile(array, 0.95))
    return row


def fmt(mean: float, p05: float, p95: float, *, static: bool = False) -> str:
    if np.isnan(mean):
        return "—"
    if static:
        return f"{mean:.3f}"
    return f"{mean:.3f} [{p05:.3f}, {p95:.3f}]"


def main() -> int:
    global OUTPUT, ABLATION, STATIC
    parser = argparse.ArgumentParser(description="Generate the multi-objective ablation table.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT, help="Root containing seed_* dynamic runs.")
    parser.add_argument("--ablation-dir", type=Path, default=ABLATION, help="Full-budget ablation directory.")
    parser.add_argument("--static-dir", type=Path, default=STATIC, help="Static baseline result directory.")
    args = parser.parse_args()
    OUTPUT = args.output_root.resolve()
    ABLATION = args.ablation_dir.resolve()
    STATIC = args.static_dir.resolve()
    try:
        full_seeds = load_seed_metrics()
    except FileNotFoundError:
        # The compact public release omits the large seed-run directories but
        # checks in the fully aggregated table. Keep the command useful in
        # that release while requiring seed directories for a fresh rebuild.
        existing = ABLATION / "multiobjective_ablation_summary.csv"
        if existing.exists():
            print(
                f"No seed_* run directories found under {OUTPUT}; "
                f"retaining checked-in aggregate {existing}."
            )
            return 0
        raise
    rows = [aggregate(full_seeds, VARIANT_LABELS["full"], "500 top-level dynamic seeds", "Seed mean [P05, P95].")]
    for variant in ("no_historical_memory", "no_forward_reward", "no_score_perturbation", "no_nonlinear_saturation", "no_stability_penalty"):
        values = pd.DataFrame([ablation_metrics(variant)])
        rows.append(aggregate(values, VARIANT_LABELS[variant], "full-budget ablation", "One fixed optimizer seed; interval is degenerate until multi-seed refits are run."))
    rows.append(aggregate(pd.DataFrame([static_metrics()]), VARIANT_LABELS["static"], "static baseline", "Single fixed strategy-priority declaration model."))
    result = pd.DataFrame(rows)
    result.to_csv(ABLATION / "multiobjective_ablation_summary.csv", index=False, encoding="utf-8-sig", float_format="%.10f")

    columns = ["Model", "Transition support", "Mean conditional support", "Nash retention", "GMR retention", "SMR retention", "SEQ retention", "Forward shortfall", "Stability shortfall", "Matrix dispersion"]
    lines = [
        "# 多目标消融实验结果",
        "",
        "表 1 汇总 DynamicGMCR 动态偏好模型及静态策略优先基线的多目标评价。动态 Full model 报告 500 个已完成随机种子的均值及 5%–95% 分位区间；现有五个全预算消融结果由同一优化种子生成，因此其区间退化为单点，不能被解释为消融模型的跨种子不确定性。",
        "",
        "|" + "|".join(columns) + "|",
        "|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|",
    ]
    for _, row in result.iterrows():
        is_static = row["model"] == VARIANT_LABELS["static"]
        values = [row["model"]] + [fmt(float(row[f"{metric}_mean"]), float(row[f"{metric}_p05"]), float(row[f"{metric}_p95"]), static=is_static) for metric in METRICS]
        lines.append("|" + "|".join(values) + "|")
    lines += [
        "",
        "表注：transition support 与四类 retention rate 越高越好；mean conditional support 越高表示观测目标状态获得的条件支持越强；forward shortfall、stability shortfall 和 matrix dispersion 越低越好。forward shortfall 为变化月份前瞻共同收益不足的均值平方短缺，stability shortfall 为保持月份、双方 actor-level SEQ margin 短缺的均值平方项，matrix dispersion 为月度 2×5 偏好矩阵各元素时间标准差的平均值。静态模型不定义动态前瞻项和稳定性惩罚，故相应列记为“—”；其偏好矩阵固定，matrix dispersion 为 0。",
        "",
        "loss、transition objective 与 summed transition probability 未在表 1 中合并比较。loss 是包含转移拟合、前瞻短缺、稳定性短缺及正则项的综合优化损失；transition objective 是变化月份对数条件转移项的均值；summed transition probability 是变化月份观测转移概率的未归一化求和。三者量纲、聚合方式和优化含义不同，不能作为同一评价列或被再次压缩为单一总分。",
        "",
        "在不确定性充分的多目标报告中，本文保留各指标的独立方向与数值，不构造加权总分。该做法允许读者观察记忆、前瞻项、评分扰动、非线性饱和及稳定性惩罚对不同目标的权衡影响。",
    ]
    (ABLATION / "multiobjective_ablation_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {ABLATION / 'multiobjective_ablation_summary.csv'}")
    print(f"Wrote {ABLATION / 'multiobjective_ablation_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
