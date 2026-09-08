from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COUNTRIES = ["CN", "US"]
DOMAINS = ["diplomacy", "economy", "law", "politics", "security", "society_humanitarian"]
DRIVER_SHORT = ["D1", "D2", "D3", "D4", "D5"]
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
DOMAIN_LABELS = [
    "Diplomacy",
    "Economy",
    "Law",
    "Politics",
    "Security",
    "Society & humanitarian",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visualization figures for an existing DynamicGMCR solver output directory."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("DynamicGMCR") / "output",
        help="Root directory containing timestamped solver outputs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific timestamped output directory. If omitted, the latest directory is used.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("DynamicGMCR") / "input",
        help="DynamicGMCR input directory, used for observed state option-path visualization.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Figure formats to write.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs.")
    return parser.parse_args()


def latest_run_dir(output_root: Path) -> Path:
    candidates = [p for p in output_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {output_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def timestamp_from_run_dir(run_dir: Path) -> str:
    return run_dir.name


def find_required_file(run_dir: Path, stem: str, timestamp: str) -> Path:
    exact = run_dir / f"{stem}_{timestamp}.csv"
    if exact.exists():
        return exact
    matches = sorted(run_dir.glob(f"{stem}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"Missing {stem}_*.csv in {run_dir}")
    return matches[-1]


def find_json_file(run_dir: Path, stem: str, timestamp: str) -> Path | None:
    exact = run_dir / f"{stem}_{timestamp}.json"
    if exact.exists():
        return exact
    matches = sorted(run_dir.glob(f"{stem}_*.json"))
    return matches[-1] if matches else None


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#3B3B3B",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11.0,
            "axes.labelsize": 12.0,
            "axes.titlesize": 13.0,
            "axes.titleweight": "bold",
            "axes.titlepad": 9,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "savefig.transparent": False,
        }
    )
    return plt


def add_y_grid(ax) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.09,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def sparse_positions(count: int, max_ticks: int = 10) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=int)
    if count <= max_ticks:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, max_ticks, dtype=int))


def save_figure(fig, figures_dir: Path, name: str, formats: Iterable[str], dpi: int) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if not fig.get_constrained_layout():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout.*",
                category=UserWarning,
            )
            try:
                fig.tight_layout()
            except RuntimeError:
                pass
    for fmt in formats:
        path = figures_dir / f"{name}.{fmt}"
        kwargs = {
            "bbox_inches": "tight",
            "facecolor": "white",
            "metadata": {"Creator": "Matplotlib; Times New Roman publication style"},
        }
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        written.append(str(path))
    return written


def month_labels(months: pd.Series | np.ndarray) -> list[str]:
    labels: list[str] = []
    for month in months:
        value = str(int(month))
        labels.append(f"{value[:4]}-{value[4:6]}")
    return labels


def plot_optimization(history: pd.DataFrame, figures_dir: Path, timestamp: str, formats, dpi) -> list[str]:
    if history.empty:
        return []
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 3.55))

    de = history[history["phase"] == "differential_evolution"].copy()
    ls = history[history["phase"] == "local_search"].copy()
    if not de.empty:
        ax.plot(de["generation"], de["best_loss"], color=OKABE_ITO[0], lw=1.8, label="DE best")
        ax.plot(
            de["generation"],
            de["median_loss"],
            color=OKABE_ITO[5],
            lw=1.25,
            ls="--",
            label="DE median",
        )
    if not ls.empty:
        x_offset = float(de["generation"].max() + 1) if not de.empty else 0.0
        step_max = max(float(ls["step"].max()), 1.0)
        x = x_offset + ls["step"].astype(float) / step_max * max(len(de), 1)
        ax.axvspan(x_offset, float(x.max()), color=OKABE_ITO[4], alpha=0.055, linewidth=0)
        ax.axvline(x_offset, color="#777777", lw=0.8, ls=":")
        ax.text(
            x_offset,
            0.91,
            "Local search",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=10.5,
            color="#555555",
        )
        ax.plot(x, ls["best_loss"], color=OKABE_ITO[1], lw=1.8, label="Local search best")

    ax.set_xlabel("Generation-scale iteration")
    ax.set_ylabel("Loss")
    add_y_grid(ax)
    ax.margins(x=0.015)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01), handlelength=2.5)
    written = save_figure(fig, figures_dir, f"optimization_progress_{timestamp}", formats, dpi)
    plt.close(fig)
    return written


def plot_transition_fit(fit: pd.DataFrame, figures_dir: Path, timestamp: str, formats, dpi) -> list[str]:
    changed = fit[fit["observed_change"] == 1].copy()
    if changed.empty:
        return []
    plt = configure_matplotlib()
    changed["label"] = month_labels(changed["month"])
    x = np.arange(len(changed))

    probability = changed["transition_probability"].astype(float).to_numpy()
    shortfall = changed["forward_shortfall_penalty"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(8.0, 4.35))
    width = 0.30
    bar_offset = 0.22
    bars_probability = ax.bar(
        x - bar_offset,
        probability,
        width=width,
        color=OKABE_ITO[2],
        edgecolor="#00659C",
        linewidth=0.35,
        label="Compatibility score",
    )
    ax.set_ylim(0, max(1.05, float(probability.max()) * 1.16))
    ax.set_ylabel("Compatibility score")
    ax.set_xlabel("Changed month")
    ax.set_xticks(x)
    ax.set_xticklabels(changed["label"], rotation=45, ha="right")
    add_y_grid(ax)

    ax2 = ax.twinx()
    bars_shortfall = ax2.bar(
        x + bar_offset,
        shortfall,
        width=width,
        color=OKABE_ITO[4],
        edgecolor="#C78300",
        linewidth=0.35,
        label="Squared shortfall",
    )
    shortfall_max = float(shortfall.max()) if len(shortfall) else 0.0
    ax2.set_ylim(0, max(0.01, shortfall_max * 1.30))
    ax2.set_ylabel("Squared shortfall")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(OKABE_ITO[4])
    ax2.tick_params(axis="y", colors=OKABE_ITO[4])
    ax2.yaxis.label.set_color(OKABE_ITO[4])

    for bar, value in zip(bars_probability, probability):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ax.get_ylim()[1] * 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=OKABE_ITO[1],
        )
    for bar, value in zip(bars_shortfall, shortfall):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + ax2.get_ylim()[1] * 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#996600",
        )
    ax.legend(
        [bars_probability, bars_shortfall],
        ["Compatibility score", "Squared shortfall"],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        handlelength=1.6,
        columnspacing=1.8,
    )

    written = save_figure(fig, figures_dir, f"transition_fit_{timestamp}", formats, dpi)
    plt.close(fig)
    return written


def plot_stability(fit: pd.DataFrame, stable_margin: float, figures_dir: Path, timestamp: str, formats, dpi) -> list[str]:
    stay = fit[fit["observed_change"] == 0].copy()
    if stay.empty:
        return []
    plt = configure_matplotlib()
    stay["label"] = month_labels(stay["month"])
    x = np.arange(len(stay))
    ticks = sparse_positions(len(stay), 11)

    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    ax.axhspan(0, stable_margin, color="#BDBDBD", alpha=0.16, linewidth=0)
    ax.plot(x, stay["cn_primary_seq_margin"].astype(float), marker="o", ms=3.2, lw=1.5, color=OKABE_ITO[0], label="CN")
    ax.plot(x, stay["us_primary_seq_margin"].astype(float), marker="s", ms=3.2, lw=1.5, ls="--", color=OKABE_ITO[1], label="US")
    ax.axhline(0.0, color="#555555", ls=":", lw=1.0, label="SEQ stability boundary")
    ax.set_ylabel("SEQ stability margin")
    ax.set_xlabel("Unchanged month")
    ax.set_xticks(ticks)
    ax.set_xticklabels(stay["label"].iloc[ticks], rotation=45, ha="right")
    add_y_grid(ax)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))

    written = save_figure(fig, figures_dir, f"stability_margins_{timestamp}", formats, dpi)
    plt.close(fig)
    return written


def plot_preferences(pref: pd.DataFrame, figures_dir: Path, timestamp: str, formats, dpi) -> list[str]:
    if pref.empty:
        return []
    plt = configure_matplotlib()
    months = np.array(sorted(pref["month"].unique()))
    labels = [str(int(m)) for m in months]
    ticks = sparse_positions(len(months), 10)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True, gridspec_kw={"hspace": 0.16})
    for idx, country in enumerate(COUNTRIES):
        sub = pref[pref["country"] == country].copy()
        pivot = sub.pivot(index="month", columns="driver_index", values="preference_value").sort_index()
        for driver_idx in sorted(pivot.columns):
            axes[idx].plot(
                np.arange(len(pivot.index)),
                pivot[driver_idx],
                lw=1.55,
                color=OKABE_ITO[(int(driver_idx) - 1) % len(OKABE_ITO)],
                ls=LINE_STYLES[(int(driver_idx) - 1) % len(LINE_STYLES)],
                label=f"D{int(driver_idx)}",
            )
        axes[idx].set_ylabel("Preference value")
        add_y_grid(axes[idx])
        add_panel_label(axes[idx], chr(ord("A") + idx))

    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right")
    axes[-1].set_xlabel("Month")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        handlelength=2.8,
        columnspacing=1.8,
    )
    fig.subplots_adjust(top=0.90)

    written = save_figure(fig, figures_dir, f"preference_trajectories_{timestamp}", formats, dpi)
    plt.close(fig)
    return written


def plot_parameter_matrix(parameter_vector: pd.DataFrame, figures_dir: Path, timestamp: str, formats, dpi) -> list[str]:
    a = parameter_vector[parameter_vector["group"] == "A"].copy()
    if a.empty:
        return []
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=True)

    vmax = max(float(a["value"].abs().max()), 1.0)
    for idx, country in enumerate(COUNTRIES):
        sub = a[a["country"] == country].copy()
        matrix = sub.pivot(index="domain", columns="driver", values="value").reindex(DOMAINS)
        matrix = matrix.reindex(sorted(matrix.columns), axis=1)
        image = axes[idx].imshow(matrix.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        axes[idx].set_yticks(np.arange(len(matrix.index)))
        axes[idx].set_yticklabels(DOMAIN_LABELS)
        axes[idx].set_xticks(np.arange(len(matrix.columns)))
        axes[idx].set_xticklabels(DRIVER_SHORT[: len(matrix.columns)])
        axes[idx].tick_params(length=0)
        for spine in axes[idx].spines.values():
            spine.set_visible(False)
        add_panel_label(axes[idx], chr(ord("A") + idx))
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                val = float(matrix.values[r, c])
                text_color = "white" if abs(val) > 0.57 * vmax else "#222222"
                axes[idx].text(
                    c,
                    r,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color=text_color,
                )
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.87, pad=0.025, label="Coefficient")
    colorbar.outline.set_linewidth(0.6)

    written = save_figure(fig, figures_dir, f"parameter_matrix_A_{timestamp}", formats, dpi)
    plt.close(fig)
    return written


def plot_state_path(fit: pd.DataFrame, input_dir: Path, figures_dir: Path, timestamp: str, formats, dpi) -> list[str]:
    state_file = input_dir / "monthly_key_states.csv"
    if not state_file.exists():
        return []
    plt = configure_matplotlib()
    states = pd.read_csv(state_file).sort_values("month").copy()
    states["label"] = month_labels(states["month"])
    x = np.arange(len(states))
    ticks = sparse_positions(len(states), 11)
    state_values = states["state_id"].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    ax.step(x, state_values, where="post", color=OKABE_ITO[0], lw=1.8, label="State path")
    state_range = max(int(state_values.max() - state_values.min()), 1)
    label_offset = max(2.0, 0.03 * state_range)
    stage_start = 0
    for stage_end in range(1, len(state_values) + 1):
        if stage_end == len(state_values) or state_values[stage_end] != state_values[stage_start]:
            midpoint = (stage_start + stage_end - 1) / 2.0
            state_id = int(state_values[stage_start])
            ax.text(
                midpoint,
                state_id + label_offset,
                f"{state_id}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color="#222222",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.18},
                zorder=4,
            )
            stage_start = stage_end
    changed = fit[fit["observed_change"] == 1].copy()
    if not changed.empty:
        month_to_x = {int(month): idx for idx, month in enumerate(states["month"])}
        xs = [month_to_x[int(m)] for m in changed["month"] if int(m) in month_to_x]
        ys = [int(changed.loc[changed["month"] == int(m), "observed_state_id"].iloc[0]) for m in changed["month"] if int(m) in month_to_x]
        ax.scatter(
            xs,
            ys,
            color=OKABE_ITO[1],
            edgecolor="white",
            linewidth=0.6,
            s=34,
            zorder=3,
            label="Observed change",
        )
        ax.legend(ncol=2, loc="upper left")
    ax.set_ylabel("State ID")
    ax.set_xlabel("Month")
    ax.set_xticks(ticks)
    ax.set_xticklabels(states["label"].iloc[ticks], rotation=45, ha="right")
    ax.set_ylim(max(0, float(state_values.min()) - 8), float(state_values.max()) + label_offset + 6)
    add_y_grid(ax)
    ax.margins(x=0.015)

    written = save_figure(fig, figures_dir, f"observed_state_path_{timestamp}", formats, dpi)
    plt.close(fig)
    return written


def plot_utility(util: pd.DataFrame, figures_dir: Path, timestamp: str, formats, dpi) -> list[str]:
    if util.empty:
        return []
    plt = configure_matplotlib()
    pivot = util.pivot(index="month", columns="country", values="base_utility").sort_index()
    labels = [str(int(m)) for m in pivot.index]
    ticks = sparse_positions(len(pivot), 10)
    x = np.arange(len(pivot))

    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    for idx, country in enumerate(COUNTRIES):
        if country in pivot.columns:
            ax.plot(
                x,
                pivot[country],
                marker="o" if idx == 0 else "s",
                ms=3.1,
                lw=1.55,
                ls="-" if idx == 0 else "--",
                color=OKABE_ITO[idx],
                label=country,
            )
    ax.set_ylabel("Base utility")
    ax.set_xlabel("Month")
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right")
    add_y_grid(ax)
    ax.legend(ncol=2, loc="upper left")
    ax.margins(x=0.015)

    written = save_figure(fig, figures_dir, f"observed_state_utilities_{timestamp}", formats, dpi)
    plt.close(fig)
    return written


def plot_objective_decomposition(summary: dict, figures_dir: Path, timestamp: str, formats, dpi) -> list[str]:
    objective = summary.get("objective", {}) if summary else {}
    keys = [
        "log_transition_objective",
        "forward_shortfall_penalty",
        "seq_stability_penalty",
        "regularization_a",
        "regularization_mu",
        "regularization_alpha",
    ]
    if not all(k in objective for k in keys):
        return []
    plt = configure_matplotlib()
    config = summary.get("config", {})
    labels = ["Mean log compatibility", "Forward shortfall", "SEQ penalty", "A penalty", "mu penalty", "alpha penalty"]
    values = [
        float(objective["log_transition_objective"]),
        -float(config.get("lambda_forward", 0.0)) * float(objective["forward_shortfall_penalty"]),
        -float(config.get("lambda_stable", 0.0)) * float(objective["seq_stability_penalty"]),
        -float(config.get("lambda_a", 0.0)) * float(objective["regularization_a"]),
        -float(config.get("lambda_mu", 0.0)) * float(objective["regularization_mu"]),
        -float(config.get("lambda_alpha", 0.0)) * float(objective["regularization_alpha"]),
    ]
    colors = [OKABE_ITO[0] if v >= 0 else OKABE_ITO[1] for v in values]

    fig, ax = plt.subplots(figsize=(6.8, 3.35))
    y = np.arange(len(values))
    bars = ax.barh(y, values, color=colors, alpha=0.92, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Objective contribution")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.7)
    offset = max(abs(value) for value in values) * 0.025
    for bar, value in zip(bars, values):
        ax.text(
            value + (offset if value >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=10.5,
            color="#333333",
        )
    ax.margins(x=0.11)

    written = save_figure(fig, figures_dir, f"objective_decomposition_{timestamp}", formats, dpi)
    plt.close(fig)
    return written


def plot_state_option_matrix(input_dir: Path, figures_dir: Path, timestamp: str, formats, dpi) -> list[str]:
    state_file = input_dir / "monthly_key_states.csv"
    table_file = input_dir / "state_table_cn.csv"
    if not state_file.exists() or not table_file.exists():
        return []
    states = pd.read_csv(state_file).sort_values("month").copy()
    table = pd.read_csv(table_file)
    state_col = table.columns[0]
    option_cols = list(table.columns[1:9])
    merged = states.merge(table[[state_col] + option_cols], left_on="state_id", right_on=state_col, how="left")
    if merged[option_cols].isna().all().all():
        return []

    plt = configure_matplotlib()
    values = merged[option_cols].astype(float).to_numpy().T
    labels = month_labels(merged["month"])
    ticks = sparse_positions(len(labels), 10)

    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    binary_cmap = ListedColormap(["#F2F2F2", OKABE_ITO[0]])
    image = ax.imshow(values, aspect="auto", cmap=binary_cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(np.arange(len(option_cols)))
    ax.set_yticklabels([c.replace("[US]", " (US)").replace("[C0]", " (CN)") for c in option_cols])
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right")
    ax.set_xlabel("Month")
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.025, ticks=[0, 1], label="Activation")
    colorbar.ax.set_yticklabels(["Inactive", "Active"])
    colorbar.outline.set_linewidth(0.6)

    written = save_figure(fig, figures_dir, f"observed_option_activation_{timestamp}", formats, dpi)
    plt.close(fig)
    return written


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir else latest_run_dir(args.output_root)
    run_dir = run_dir.resolve()
    timestamp = timestamp_from_run_dir(run_dir)
    figures_dir = run_dir / "figures"

    summary_path = find_json_file(run_dir, "summary", timestamp)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path else {}
    stable_margin = float(summary.get("config", {}).get("stable_margin", 0.02))

    fit = pd.read_csv(find_required_file(run_dir, "monthly_fit", timestamp))
    pref = pd.read_csv(find_required_file(run_dir, "monthly_preferences", timestamp))
    param = pd.read_csv(find_required_file(run_dir, "parameter_vector", timestamp))
    history_file = find_required_file(run_dir, "optimization_history", timestamp)
    history = pd.read_csv(history_file) if history_file.exists() else pd.DataFrame()
    util_file = find_required_file(run_dir, "observed_state_utilities", timestamp)
    util = pd.read_csv(util_file) if util_file.exists() else pd.DataFrame()

    created: list[str] = []
    created.extend(plot_optimization(history, figures_dir, timestamp, args.formats, args.dpi))
    created.extend(plot_transition_fit(fit, figures_dir, timestamp, args.formats, args.dpi))
    created.extend(plot_stability(fit, stable_margin, figures_dir, timestamp, args.formats, args.dpi))
    created.extend(plot_preferences(pref, figures_dir, timestamp, args.formats, args.dpi))
    created.extend(plot_parameter_matrix(param, figures_dir, timestamp, args.formats, args.dpi))
    created.extend(plot_state_path(fit, args.input_dir, figures_dir, timestamp, args.formats, args.dpi))
    created.extend(plot_utility(util, figures_dir, timestamp, args.formats, args.dpi))
    created.extend(plot_objective_decomposition(summary, figures_dir, timestamp, args.formats, args.dpi))
    created.extend(plot_state_option_matrix(args.input_dir, figures_dir, timestamp, args.formats, args.dpi))

    manifest = {
        "timestamp_day_hour_minute": timestamp,
        "run_dir": str(run_dir),
        "figure_count": len(created),
        "figures": [str(Path(path).relative_to(run_dir)) for path in created],
    }
    manifest_path = run_dir / f"visualization_manifest_{timestamp}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Created {len(created)} figure files under {figures_dir}")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
