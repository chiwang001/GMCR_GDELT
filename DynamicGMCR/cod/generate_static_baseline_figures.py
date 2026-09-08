#!/usr/bin/env python3
"""Generate publication-ready figures for the DynamicGMCR static baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from static_figure_style import OKABE_ITO, configure_static_figure_style, save_static_figure


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "static_preference_model"


def load_results() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = json.loads((OUTPUT / "static_baseline_summary.json").read_text(encoding="utf-8"))
    history = pd.read_csv(OUTPUT / "historical_explanation.csv", encoding="utf-8-sig")
    stability = pd.read_csv(OUTPUT / "static_state_stability.csv", encoding="utf-8-sig")
    utilities = pd.read_csv(OUTPUT / "static_state_utilities.csv", encoding="utf-8-sig")
    return summary, history, stability, utilities


def draw_preference_workflow(summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 2.45), constrained_layout=True)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = [
        (0.015, 0.31, 0.205, 0.48, "Strategy-priority\ndeclarations", "8 ordered statements\nfor each decision maker", OKABE_ITO["sky"]),
        (0.275, 0.31, 0.185, 0.48, "Binary satisfaction", r"$b_{ip}(s)\in\{0,1\}$", OKABE_ITO["orange"]),
        (0.515, 0.31, 0.205, 0.48, "Lexicographic utility", r"$V_i(s)=\sum_{p=1}^{8}2^{8-p}b_{ip}(s)$", OKABE_ITO["green"]),
        (0.775, 0.21, 0.21, 0.68, "Static GMCR analysis", f"{summary['feasible_states']} feasible states\nNash / GMR / SMR / SEQ\nHistorical transition support", OKABE_ITO["blue"]),
    ]
    for x, y, width, height, heading, body, color in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=1.25, edgecolor=color, facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height * 0.68, heading, ha="center", va="center", fontsize=10, fontweight="bold", color=color, linespacing=1.1)
        ax.text(x + width / 2, y + height * 0.30, body, ha="center", va="center", fontsize=8.5, color=OKABE_ITO["black"], linespacing=1.25)
    for start, end in ((0.225, 0.27), (0.465, 0.51), (0.725, 0.77)):
        ax.add_patch(FancyArrowPatch((start, 0.55), (end, 0.55), arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color=OKABE_ITO["black"]))
    ax.text(0.5, 0.08, "Fixed preferences over the full sample; no monthly shocks or estimated preference weights", ha="center", va="center", fontsize=8.5, color="#555555")
    save_static_figure(fig, OUTPUT, "fig_static_preference_workflow")


def draw_performance(history: pd.DataFrame, stability: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)

    ax = axes[0]
    supported = np.array([
        history["transition_explanation"].eq("stable_status_quo").sum(),
        history["transition_explanation"].eq("immediate_utility_support").sum(),
    ])
    totals = np.array([
        history["observed_change"].eq(0).sum(),
        history["observed_change"].eq(1).sum(),
    ])
    unsupported = totals - supported
    x = np.arange(2)
    ax.bar(x, supported, width=0.58, color=OKABE_ITO["blue"], label="Explained")
    ax.bar(x, unsupported, width=0.58, bottom=supported, color=OKABE_ITO["light_gray"], edgecolor="#777777", linewidth=0.6, label="Unexplained")
    for xpos, value, total in zip(x, supported, totals):
        ax.text(xpos, total + 0.6, f"{value}/{total}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, ["Unchanged months", "Changed months"])
    ax.set_ylabel("Number of transitions")
    ax.set_ylim(0, 29)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.text(-0.13, 1.03, "A", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    ax = axes[1]
    concepts = ("nash", "gmr", "smr", "seq")
    counts = {
        "CN": np.array([stability[f"cn_{c}"].sum() for c in concepts]),
        "US": np.array([stability[f"us_{c}"].sum() for c in concepts]),
        "Joint": np.array([stability[f"joint_{c}"].sum() for c in concepts]),
    }
    width = 0.24
    x = np.arange(len(concepts))
    for offset, (label, values), color, hatch in zip(
        (-width, 0, width), counts.items(),
        (OKABE_ITO["sky"], OKABE_ITO["orange"], OKABE_ITO["green"]),
        ("", "//", ".."),
    ):
        ax.bar(x + offset, values, width, label=label, color=color, edgecolor="#4A4A4A", linewidth=0.45, hatch=hatch)
    ax.set_xticks(x, [c.upper() for c in concepts])
    ax.set_xlabel("GMCR stability concept")
    ax.set_ylabel("Number of stable states")
    ax.set_ylim(0, 82)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.text(-0.13, 1.03, "B", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    save_static_figure(fig, OUTPUT, "fig_static_baseline_performance")


def draw_utility_stability(stability: pd.DataFrame, utilities: pd.DataFrame) -> None:
    merged = utilities.merge(stability[["state_id", "joint_nash", "joint_gmr", "joint_smr", "joint_seq"]], on="state_id")
    joint_any = merged[["joint_nash", "joint_gmr", "joint_smr", "joint_seq"]].any(axis=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.65), constrained_layout=True)
    ax.scatter(merged.loc[~joint_any, "cn_utility"], merged.loc[~joint_any, "us_utility"], s=26, color=OKABE_ITO["light_gray"], edgecolor="#777777", linewidth=0.45, label="No joint stability")
    ax.scatter(merged.loc[joint_any, "cn_utility"], merged.loc[joint_any, "us_utility"], s=44, color=OKABE_ITO["green"], edgecolor=OKABE_ITO["black"], linewidth=0.55, label="Jointly stable")
    nash = merged[merged["joint_nash"]]
    ax.scatter(nash["cn_utility"], nash["us_utility"], s=90, marker="*", color=OKABE_ITO["vermillion"], edgecolor=OKABE_ITO["black"], linewidth=0.55, label="Joint Nash")
    for row in nash.itertuples():
        ax.annotate(f"S{int(row.state_id)}", (row.cn_utility, row.us_utility), xytext=(7, 7), textcoords="offset points", ha="left", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("CN lexicographic utility")
    ax.set_ylabel("US lexicographic utility")
    ax.set_xlim(-7, 239)
    ax.set_ylim(-7, 239)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="upper left")
    save_static_figure(fig, OUTPUT, "fig_static_state_utility_stability")


def main() -> int:
    global OUTPUT
    parser = argparse.ArgumentParser(description="Generate publication-ready static-baseline figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT,
        help="Directory containing static baseline CSV/JSON files and receiving figures.",
    )
    args = parser.parse_args()
    OUTPUT = args.output_dir.resolve()
    configure_static_figure_style()
    summary, history, stability, utilities = load_results()
    draw_preference_workflow(summary)
    draw_performance(history, stability)
    draw_utility_stability(stability, utilities)
    print(f"Static baseline figures written to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
