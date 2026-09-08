"""Create the figure for the 2020-02 reality-coded GMCR analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OPTIONS = ("U1", "U2", "U3", "U4", "C1", "C2", "C3", "C4")
CONCEPTS = ("Nash", "GMR", "SMR", "SEQ")


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent.parent
    default_dir = demo_dir / "output" / "preference_prediction" / "gmcr_stability_202002"
    parser = argparse.ArgumentParser(description="Plot historical conflict evolution and GMCR stability.")
    parser.add_argument("--input-dir", type=Path, default=demo_dir / "input")
    parser.add_argument("--stability-dir", type=Path, default=default_dir)
    parser.add_argument("--output", type=Path, default=default_dir / "conflict_evolution_analysis")
    return parser.parse_args()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    states = read(args.input_dir / "monthly_key_states.csv")
    table = read(args.input_dir / "state_table_cn.csv")
    stability = read(args.stability_dir / "stability_by_state.csv")

    option_columns = [next(column for column in table.columns if str(column).startswith(option)) for option in OPTIONS]
    option_values = table.set_index(table.columns[0]).loc[:, option_columns]
    option_values.columns = list(OPTIONS)
    # Observations end in 2020-01.  February is coded from the public policy
    # record (Phase-One implementation and reciprocal tariff reductions) as S73.
    observed_months = states["month"].astype(str).tolist()
    observed_ids = states["state_id"].astype(int).tolist()
    months = np.asarray(observed_months + ["2020-02"])
    y = np.asarray(observed_ids + [73])
    x = np.arange(len(months))
    comparison_ids = [57, 73, 76]
    state_matrix = option_values.loc[comparison_ids].to_numpy(float)
    comparison_flags = stability.set_index("state_id").loc[
        comparison_ids, [f"joint_{concept}_stable" for concept in CONCEPTS]
    ].astype(int).to_numpy()

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    fig = plt.figure(figsize=(11.2, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05), width_ratios=(1.0, 1.0))
    ax_path = fig.add_subplot(grid[0, :])
    ax_options = fig.add_subplot(grid[1, 0])
    ax_stability = fig.add_subplot(grid[1, 1])

    # Historical state path. Points are annotated only at changes to keep the
    # figure readable while retaining every monthly observation in the line.
    ax_path.step(x, y, where="mid", color="#1f4e79", linewidth=1.8)
    ax_path.scatter(x[:-1], y[:-1], color="#d55e00", s=18, zorder=3, label="Observed")
    ax_path.scatter(x[-1], y[-1], color="#009E73", marker="*", s=125, zorder=4,
                    label="2020-02 reality-coded S73")
    for index in range(len(months)):
        if index == 0 or y[index] != y[index - 1]:
            ax_path.annotate(str(y[index]), (x[index], y[index]), xytext=(0, 8),
                             textcoords="offset points", ha="center", fontsize=8)
    ax_path.axvline(x[-1], color="#009E73", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_path.set_xlim(-0.5, len(months) - 0.5)
    ax_path.set_ylim(0, max(y) + 8)
    ax_path.set_ylabel("State ID")
    ax_path.set_title("(a) Observed path and reality-coded February 2020 state")
    tick_positions = np.arange(0, len(months), 6)
    if tick_positions[-1] != len(months) - 1:
        tick_positions = np.append(tick_positions, len(months) - 1)
    ax_path.set_xticks(tick_positions, months[tick_positions], rotation=45, ha="right")
    ax_path.grid(axis="y", alpha=0.25)
    ax_path.legend(loc="upper left", frameon=False, ncol=2)

    image = ax_options.imshow(state_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax_options.set_yticks(np.arange(len(comparison_ids)), [f"S{state_id}" for state_id in comparison_ids])
    ax_options.set_xticks(np.arange(len(OPTIONS)), OPTIONS)
    ax_options.set_xlabel("Policy option")
    ax_options.set_title("(b) S73 and sensitivity-state option combinations")
    for row in range(len(comparison_ids)):
        for col in range(len(OPTIONS)):
            if state_matrix[row, col] > 0.5:
                ax_options.text(col, row, "1", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax_options, fraction=0.046, pad=0.04, ticks=[0, 1], label="Inactive / active")

    colors = ["#0072B2", "#009E73", "#D55E00"]
    positions = np.arange(len(CONCEPTS))
    width = 0.24
    for row, (state_id, color) in enumerate(zip(comparison_ids, colors)):
        bars = ax_stability.bar(positions + (row - 1) * width, comparison_flags[row],
                                width=width, color=color, label=f"S{state_id}")
        for bar, value in zip(bars, comparison_flags[row]):
            if value:
                ax_stability.text(bar.get_x() + bar.get_width() / 2, 1.03, "1",
                                  ha="center", va="bottom", fontsize=8)
    ax_stability.set_ylim(0, 1.28)
    ax_stability.set_ylabel("Joint stability (1=yes)")
    ax_stability.set_title("(c) Stability comparison under 202002 probabilities")
    ax_stability.set_xticks(positions, CONCEPTS)
    ax_stability.grid(axis="y", alpha=0.25)
    ax_stability.legend(loc="upper center", frameon=False, ncol=3)

    fig.suptitle("2020-02 reality state and probability-matrix GMCR assessment", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 300}), (".pdf", {}), (".svg", {})):
        fig.savefig(args.output.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
