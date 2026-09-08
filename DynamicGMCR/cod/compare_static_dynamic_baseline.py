#!/usr/bin/env python3
"""Compare the declared-option static GMCR baseline with a DynamicGMCR dynamic run."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from static_figure_style import OKABE_ITO, configure_static_figure_style, save_static_figure


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "static_preference_model"
STATIC_HISTORY = OUTPUT / "historical_explanation.csv"
STATIC_STABILITY = OUTPUT / "static_state_stability.csv"


def choose_dynamic_run() -> tuple[Path, dict]:
    candidates: list[tuple[float, Path, dict]] = []
    for summary_path in (ROOT / "output").glob("*/summary_*.json"):
        if summary_path.parent.name == OUTPUT.name:
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if "dynamic-preference" not in str(summary.get("model", "")):
                continue
            loss = float(summary.get("objective", {}).get("loss", np.inf))
            fit_path = summary_path.parent / f"monthly_fit_{summary_path.stem.removeprefix('summary_')}.csv"
            if fit_path.exists():
                candidates.append((loss, fit_path, summary))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not candidates:
        raise FileNotFoundError("No DynamicGMCR dynamic monthly_fit output was found.")
    _, fit_path, summary = min(candidates, key=lambda item: item[0])
    return fit_path, summary


def parse_margin_flags(value: object) -> list[bool]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return [False] * 4
    parts = str(value).split("|")
    flags: list[bool] = []
    for part in parts[:4]:
        try:
            flags.append(float(part) >= 0.0)
        except ValueError:
            flags.append(False)
    return flags + [False] * (4 - len(flags))


def save_figure(fig: plt.Figure, stem: str) -> None:
    save_static_figure(fig, OUTPUT, stem)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_static_figure_style()

    static = pd.read_csv(STATIC_HISTORY, encoding="utf-8-sig")
    static_stability = pd.read_csv(STATIC_STABILITY, encoding="utf-8-sig").set_index("state_id")
    dynamic_path, dynamic_summary = choose_dynamic_run()
    dynamic = pd.read_csv(dynamic_path, encoding="utf-8-sig")
    dynamic = dynamic.rename(
        columns={
            "transition_probability": "dynamic_transition_probability",
            "transition_mechanism": "dynamic_transition_mechanism",
        }
    )
    dynamic["dynamic_seq_stable_both"] = dynamic["seq_stable_both"].fillna(0).astype(int)
    unchanged = dynamic["observed_change"].eq(0)
    dynamic["dynamic_joint_stability"] = 0
    for idx in dynamic.index[unchanged]:
        cn = parse_margin_flags(dynamic.at[idx, "cn_margins_Nash_GMR_SMR_SEQ"])
        us = parse_margin_flags(dynamic.at[idx, "us_margins_Nash_GMR_SMR_SEQ"])
        dynamic.at[idx, "dynamic_joint_stability"] = int(all(a and b for a, b in zip(cn, us)))
    keep_dynamic = [
        "month",
        "dynamic_transition_probability",
        "dynamic_transition_mechanism",
        "dynamic_seq_stable_both",
        "dynamic_joint_stability",
        "cn_stable_any",
        "us_stable_any",
    ]
    merged = static.merge(dynamic[keep_dynamic], on="month", how="left")
    merged["static_joint_seq_stability"] = merged.apply(
        lambda row: int(static_stability.loc[int(row["start_state_id"]), "joint_seq"])
        if int(row["observed_change"]) == 0 else 0,
        axis=1,
    )
    merged.to_csv(OUTPUT / "static_dynamic_monthly_comparison_single_run.csv", index=False, encoding="utf-8-sig")

    changed = merged[merged["observed_change"].eq(1)].copy()
    stays = merged[merged["observed_change"].eq(0)].copy()
    concepts = ("nash", "gmr", "smr", "seq")
    static_rates = []
    dynamic_rates = []
    for concept_idx, concept in enumerate(concepts):
        static_joint = []
        dynamic_joint = []
        for _, row in stays.iterrows():
            state = int(row["start_state_id"])
            s = static_stability.loc[state]
            static_joint.append(bool(s[f"joint_{concept}"]))
            cn = parse_margin_flags(dynamic.loc[dynamic["month"].eq(row["month"]), "cn_margins_Nash_GMR_SMR_SEQ"].iloc[0])
            us = parse_margin_flags(dynamic.loc[dynamic["month"].eq(row["month"]), "us_margins_Nash_GMR_SMR_SEQ"].iloc[0])
            dynamic_joint.append(bool(cn[concept_idx] and us[concept_idx]))
        static_rates.append(float(np.mean(static_joint)))
        dynamic_rates.append(float(np.mean(dynamic_joint)))

    summary = {
        "static_model": "declared-option strategy-priority GMCR",
        "dynamic_reference_run": str(dynamic_path.parent),
        "dynamic_loss": float(dynamic_summary.get("objective", {}).get("loss", np.nan)),
        "months": int(len(merged) + 1),
        "transitions": int(len(merged)),
        "changed_transitions": int(len(changed)),
        "unchanged_transitions": int(len(stays)),
        "static_changed_positive_rate": float((changed["observed_transition_probability"] > 1e-8).mean()),
        "dynamic_changed_positive_rate": float((changed["dynamic_transition_probability"] > 1e-8).mean()),
        "static_changed_mean_probability": float(changed["observed_transition_probability"].mean()),
        "dynamic_changed_mean_probability": float(changed["dynamic_transition_probability"].mean()),
        "static_stay_joint_seq_rate": float(stays["static_joint_seq_stability"].mean()),
        "static_stay_joint_any_rate": float(stays["stay_constraint_satisfied"].mean()),
        "dynamic_stay_joint_seq_rate": float(stays["dynamic_seq_stable_both"].mean()),
        "static_concept_joint_rates": dict(zip(concepts, static_rates)),
        "dynamic_concept_joint_rates": dict(zip(concepts, dynamic_rates)),
    }
    (OUTPUT / "static_dynamic_comparison_summary_single_run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"metric": "Changed transition positive-support rate", "static": summary["static_changed_positive_rate"], "dynamic": summary["dynamic_changed_positive_rate"]},
            {"metric": "Mean probability on observed changes", "static": summary["static_changed_mean_probability"], "dynamic": summary["dynamic_changed_mean_probability"]},
            {"metric": "Unchanged months jointly SEQ-stable", "static": summary["static_stay_joint_seq_rate"], "dynamic": summary["dynamic_stay_joint_seq_rate"]},
            *({"metric": f"Joint {c.upper()} hit rate on unchanged months", "static": static_rates[i], "dynamic": dynamic_rates[i]} for i, c in enumerate(concepts)),
        ]
    ).to_csv(OUTPUT / "static_dynamic_metric_comparison_single_run.csv", index=False, encoding="utf-8-sig")

    # Figure 1: conditional transition compatibility for the 11 observed changes.
    x = np.arange(len(changed))
    fig, ax = plt.subplots(figsize=(8.2, 3.6), constrained_layout=True)
    width = 0.38
    ax.bar(x - width / 2, changed["observed_transition_probability"], width, label="Static strategy-priority GMCR", color=OKABE_ITO["blue"])
    ax.bar(x + width / 2, changed["dynamic_transition_probability"], width, label="Dynamic preference GMCR", color=OKABE_ITO["orange"], hatch="//")
    ax.set_xticks(x, changed["month"].astype(str), rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Observed change month")
    ax.set_ylabel("Conditional transition support")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    save_figure(fig, "fig_static_dynamic_transition_support_single_run")

    # Figure 2: stability concept hit rates during unchanged months.
    fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    x = np.arange(len(concepts))
    ax.bar(x - width / 2, static_rates, width, label="Static", color=OKABE_ITO["blue"])
    ax.bar(x + width / 2, dynamic_rates, width, label="Dynamic", color=OKABE_ITO["orange"], hatch="//")
    ax.set_xticks(x, [c.upper() for c in concepts])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Joint stability hit rate")
    ax.set_xlabel("GMCR stability concept")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    save_figure(fig, "fig_static_dynamic_stability_concepts_single_run")

    # Figure 3: historical state path annotated by static explanation.
    path = pd.read_csv(ROOT / "input" / "monthly_key_states.csv", encoding="utf-8-sig")
    path["month"] = path["month"].astype(int)
    fig, ax = plt.subplots(figsize=(8.8, 3.8), constrained_layout=True)
    ax.step(path["month"].astype(str), path["state_id"], where="mid", color="#333333", linewidth=1.5)
    for _, row in changed.iterrows():
        supported = row["observed_transition_probability"] > 1e-8
        color = "#009E73" if supported else "#D55E00"
        marker = "o" if supported else "X"
        ax.scatter(str(int(row["month"])), row["observed_state_id"], color=color, marker=marker, s=36, zorder=3)
    ax.set_ylabel("Observed state ID")
    ax.set_xlabel("Month")
    ticks = np.linspace(0, len(path) - 1, min(10, len(path)), dtype=int)
    ax.set_xticks(ticks, path["month"].astype(str).iloc[ticks], rotation=45, ha="right")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#009E73", label="Positive static support", markersize=6), Line2D([0], [0], marker="X", color="w", markerfacecolor="#D55E00", label="Unsupported by static model", markersize=6)], frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    save_figure(fig, "fig_static_historical_path_explanation_single_run")

    print(f"Compared static baseline with dynamic run {dynamic_path.parent}; outputs written to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
