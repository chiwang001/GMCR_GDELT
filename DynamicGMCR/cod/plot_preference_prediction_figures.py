"""Create publication-ready figures for the 2020-02 preference forecast."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
OUTPUT = ROOT / "output" / "preference_prediction"
OUTPUT.mkdir(parents=True, exist_ok=True)

M = 369
N_STATES = 76
STATE_IDS = np.arange(1, N_STATES + 1)
OKABE_ITO = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

# Use Times New Roman for every rendered text element, including SVG/PDF text.
mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.sans-serif": ["Times New Roman"],
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save one figure in vector and raster formats."""
    for suffix in (".svg", ".pdf", ".png"):
        fig.savefig(
            OUTPUT / f"{stem}{suffix}",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def state_labels() -> dict[int, str]:
    table = pd.read_csv(INPUT / "state_table_cn.csv", encoding="utf-8-sig")
    option_cols = ["U1[US]", "U2[US]", "U3[US]", "U4[US]", "C1[C0]", "C2[C0]", "C3[C0]", "C4[C0]"]
    labels: dict[int, str] = {}
    for _, row in table.iterrows():
        state = int(row.iloc[0])
        active = [col.split("[")[0] for col in option_cols if row[col] == 1]
        labels[state] = "+".join(active) if active else "No action"
    return labels


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, str]]:
    pref = pd.read_csv(
        OUTPUT / "predicted_preferences_by_seed_202002.csv", encoding="utf-8-sig"
    )
    util = pd.read_csv(
        OUTPUT / "predicted_utilities_by_seed_202002.csv", encoding="utf-8-sig"
    )
    pair = pd.read_csv(
        OUTPUT / "pairwise_preference_probabilities_202002.csv", encoding="utf-8-sig"
    )
    pref["forecast_preference"] = pd.to_numeric(pref["forecast_preference"])
    util["forecast_utility"] = pd.to_numeric(util["forecast_utility"])
    pair["preference_probability"] = pd.to_numeric(pair["preference_probability"])
    return pref, util, pair, state_labels()


def probability_matrices(pair: pd.DataFrame) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for actor in ("CN", "US"):
        sub = pair[pair["actor"] == actor]
        matrices[actor] = (
            sub.pivot(index="state_s", columns="state_q", values="preference_probability")
            .reindex(index=STATE_IDS, columns=STATE_IDS)
            .to_numpy()
        )
    return matrices


def mean_win_rates(pair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for actor in ("CN", "US"):
        mat = (
            pair[pair["actor"] == actor]
            .pivot(index="state_s", columns="state_q", values="preference_probability")
            .reindex(index=STATE_IDS, columns=STATE_IDS)
        )
        for state in STATE_IDS:
            values = mat.loc[state].drop(labels=state)
            rows.append({"actor": actor, "state_id": state, "mean_win_rate": values.mean()})
    return pd.DataFrame(rows)


def figure1_heatmaps(pair: pd.DataFrame) -> None:
    mats = probability_matrices(pair)
    # Reserve a dedicated third column for the shared colorbar so it never
    # overlaps the US heatmap or its x-axis label.
    fig = plt.figure(figsize=(9.6, 4.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.06), wspace=0.24)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1], sharex=None, sharey=None)]
    for ax, actor in zip(axes, ("CN", "US")):
        image = ax.imshow(mats[actor], vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Compared state q", fontsize=13)
        ax.set_ylabel("Focal state s", fontsize=13)
        ticks = np.arange(0, N_STATES, 10)
        ax.set_xticks(ticks, STATE_IDS[ticks])
        ax.set_yticks(ticks, STATE_IDS[ticks])
        ax.tick_params(length=2, labelsize=10.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_subplot(grid[0, 2])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("P(s > q)", labelpad=8, fontsize=13)
    cbar.ax.tick_params(labelsize=10.5)
    save_figure(fig, "fig1_preference_probability_heatmaps")


def figure2_scatter(pair: pd.DataFrame, labels: dict[int, str]) -> None:
    rates = mean_win_rates(pair).pivot(index="state_id", columns="actor", values="mean_win_rate")
    rho = spearmanr(rates["CN"], rates["US"]).statistic
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.scatter(
        rates["CN"], rates["US"], s=36, facecolor=OKABE_ITO["sky"],
        edgecolor=OKABE_ITO["blue"], linewidth=0.7, alpha=0.85, label="Other states"
    )
    key_states = {3: "State 3", 21: "State 21", 29: "State 29", 42: "State 42", 73: "State 73", 76: "State 76"}
    colors = {3: OKABE_ITO["green"], 21: OKABE_ITO["orange"], 29: OKABE_ITO["vermillion"], 42: OKABE_ITO["purple"], 73: OKABE_ITO["black"], 76: OKABE_ITO["blue"]}
    label_offsets = {
        3: (10, 8),
        21: (10, 9),
        29: (10, 8),
        42: (-58, -18),
        73: (10, 8),
        76: (8, -20),
    }
    for state, name in key_states.items():
        x, y = rates.loc[state, "CN"], rates.loc[state, "US"]
        ax.scatter([x], [y], s=60, color=colors[state], edgecolor="white", linewidth=0.9, zorder=3)
        ax.annotate(
            name,
            (x, y),
            xytext=label_offsets[state],
            textcoords="offset points",
            fontsize=11.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.2},
        )
    ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1.0, label="Equal preference")
    ax.axvspan(0, 0.2, color="#F2F2F2", zorder=0)
    ax.axhspan(0, 0.2, color="#F2F2F2", zorder=0)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="CN mean win rate", ylabel="US mean win rate")
    ax.tick_params(labelsize=11.5)
    ax.text(
        0.03,
        0.96,
        f"Spearman ρ = {rho:.3f}\nn = {N_STATES} states",
        transform=ax.transAxes,
        va="top",
        fontsize=11.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.25},
    )
    clean_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig2_cn_us_mean_win_rate_scatter")


def figure3_driver_intervals(pref: pd.DataFrame) -> None:
    driver_names = {
        1: "D1 Economic welfare",
        2: "D2 Strategic autonomy",
        3: "D3 Domestic legitimacy",
        4: "D4 Conflict control",
        5: "D5 Bargaining payoff",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.8), sharex=True, sharey=True)
    for ax, actor, color, title in zip(
        axes, ("CN", "US"), (OKABE_ITO["blue"], OKABE_ITO["orange"]), ("China (CN)", "United States (US)")
    ):
        sub = pref[pref["actor"] == actor]
        stats = sub.groupby("driver_index")["forecast_preference"].agg(
            mean="mean", q05=lambda x: x.quantile(0.05), q95=lambda x: x.quantile(0.95)
        )
        y = np.arange(1, 6)
        ax.errorbar(
            stats["mean"], y, xerr=[stats["mean"] - stats["q05"], stats["q95"] - stats["mean"]],
            fmt="o", color=color, ecolor=color, elinewidth=2.0, capsize=4, markersize=7
        )
        ax.axvline(0, color="#555555", linestyle="--", linewidth=0.8)
        ax.set_yticks(y, [driver_names[i] for i in y])
        ax.set_xlim(-3.3, 2.9)
        ax.set_xlabel("Forecast preference weight", fontsize=13)
        ax.tick_params(labelsize=10.5)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.6)
        ax.grid(axis="y", visible=False)
        ax.text(
            0.02,
            0.04,
            "Point: mean; bar: 5th-95th percentile",
            transform=ax.transAxes,
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.2},
        )
        clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, "fig3_driver_parameter_intervals")


def state73_ranks(util: pd.DataFrame, actor: str) -> pd.Series:
    wide = util[util["actor"] == actor].pivot(index="seed", columns="state_id", values="forecast_utility")
    return wide.rank(axis=1, ascending=False, method="average")[73]


def figure4_state73_rank_distribution(util: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.6), sharey=True)
    for ax, actor, color in zip(axes, ("CN", "US"), (OKABE_ITO["blue"], OKABE_ITO["orange"])):
        ranks = state73_ranks(util, actor)
        bins = np.arange(0.5, N_STATES + 1.5, 1)
        ax.axvspan(0.5, 5.5, color="#E8F2F8", zorder=0)
        ax.hist(ranks, bins=bins, color=color, alpha=0.82, edgecolor="white", linewidth=0.25)
        ax.axvline(ranks.mean(), color=OKABE_ITO["vermillion"], linewidth=1.2, label=f"Mean = {ranks.mean():.2f}")
        ax.axvline(ranks.median(), color=OKABE_ITO["green"], linewidth=1.2, linestyle="--", label=f"Median = {ranks.median():.0f}")
        ax.set_title(actor, pad=8)
        ax.set_xlabel("Rank of state 73 (1 = highest)")
        ax.set_xlim(0.5, N_STATES + 0.5)
        ax.set_xticks([1, 5, 10, 25, 50, 76])
        ax.text(0.98, 0.96, f"n = {len(ranks)}\nSD = {ranks.std():.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        clean_axes(ax)
        ax.legend(frameon=False, loc="upper center", fontsize=7)
    axes[0].set_ylabel("Seed count")
    fig.suptitle("Cross-seed rank distribution of the observed state", y=1.02, fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig4_state73_rank_distribution")


def figure5_difference_heatmap(pair: pd.DataFrame) -> None:
    mats = probability_matrices(pair)
    diff = mats["CN"] - mats["US"]
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    image = ax.imshow(diff, vmin=-1, vmax=1, cmap="PuOr", interpolation="nearest")
    ax.set_title("CN minus US preference probability", pad=9)
    ax.set_xlabel("Compared state q")
    ax.set_ylabel("Focal state s")
    ticks = np.arange(0, N_STATES, 10)
    ax.set_xticks(ticks, STATE_IDS[ticks])
    ax.set_yticks(ticks, STATE_IDS[ticks])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P_CN(s > q) - P_US(s > q)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save_figure(fig, "fig5_cn_minus_us_probability_difference_heatmap")


def figure6_pairwise_scatter(pair: pd.DataFrame) -> None:
    lookup = {(row.actor, int(row.state_s), int(row.state_q)): row.preference_probability for row in pair.itertuples()}

    def probability(actor: str, focal: int, compared: int) -> float:
        if focal == compared:
            return 0.0
        if focal < compared:
            return float(lookup[actor, focal, compared])
        # Off-diagonal ties are absent, so the reverse strict probability is one minus the stored value.
        return 1.0 - float(lookup[actor, compared, focal])

    rows = []
    for i, s in enumerate(STATE_IDS):
        for q in STATE_IDS[i + 1 :]:
            rows.append({"s": s, "q": q, "CN": probability("CN", int(s), int(q)), "US": probability("US", int(s), int(q))})
    data = pd.DataFrame(rows)
    strong_reverse = ((data["CN"] >= 0.8) & (data["US"] <= 0.2)) | ((data["CN"] <= 0.2) & (data["US"] >= 0.8))
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.scatter(data.loc[~strong_reverse, "CN"], data.loc[~strong_reverse, "US"], s=9, color=OKABE_ITO["sky"], alpha=0.45, label="Other pairs")
    ax.scatter(data.loc[strong_reverse, "CN"], data.loc[strong_reverse, "US"], s=13, color=OKABE_ITO["vermillion"], alpha=0.65, label=f"Strong reversal (n={strong_reverse.sum()})")
    for s, q in ((42, 21), (17, 21), (76, 73)):
        x = probability("CN", s, q)
        y = probability("US", s, q)
        ax.scatter([x], [y], s=45, color=OKABE_ITO["black"], zorder=3)
        ax.annotate(f"{s} vs {q}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=0.8)
    ax.axvline(0.2, color="#BBBBBB", linewidth=0.6, linestyle=":")
    ax.axvline(0.8, color="#BBBBBB", linewidth=0.6, linestyle=":")
    ax.axhline(0.2, color="#BBBBBB", linewidth=0.6, linestyle=":")
    ax.axhline(0.8, color="#BBBBBB", linewidth=0.6, linestyle=":")
    r = pearsonr(data["CN"], data["US"]).statistic
    ax.text(0.03, 0.96, f"Pearson r = {r:.3f}\nUnordered pairs = {len(data)}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="CN pairwise probability", ylabel="US pairwise probability")
    clean_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig6_pairwise_cn_us_probability_scatter")


def main() -> None:
    pref, util, pair, labels = read_data()
    figure1_heatmaps(pair)
    figure2_scatter(pair, labels)
    figure3_driver_intervals(pref)
    figure4_state73_rank_distribution(util)
    figure5_difference_heatmap(pair)
    figure6_pairwise_scatter(pair)
    print(f"Saved figures to {OUTPUT}")


if __name__ == "__main__":
    main()
