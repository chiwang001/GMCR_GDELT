"""Visualize actor/state stability classifications produced by DynamicGMCR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ACTORS = ("CN", "US")
CONCEPTS = ("Nash", "GMR", "SMR", "SEQ")
TRUE_COLOR = "#0072B2"
FALSE_COLOR = "#F2F2F2"
ACCENT_COLOR = "#D55E00"
TEXT_COLOR = "#303030"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize historical and predicted state-level GMCR stability."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing state_stability_types_*.csv.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "pdf"),
        choices=("png", "pdf", "svg"),
        help="Figure formats to create (default: png pdf).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.5,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.7,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )
    return plt


def find_result_file(run_dir: Path, stem: str, timestamp: str) -> Path:
    exact = run_dir / f"{stem}_{timestamp}.csv"
    if exact.exists():
        return exact
    matches = sorted(run_dir.glob(f"{stem}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"Missing {stem}_*.csv in {run_dir}")
    return matches[-1]


def month_label(month: int) -> str:
    value = str(int(month))
    return f"{value[:4]}-{value[4:]}"


def sparse_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, maximum, dtype=int))


def validate_classification(data: pd.DataFrame) -> None:
    required = {"month", "period", "state_id", "is_observed_state"}
    for actor in ACTORS:
        for concept in CONCEPTS:
            required.add(f"{actor}_{concept}_stable")
            required.add(f"{actor}_{concept}_margin")
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Classification table is missing columns: {missing}")

    duplicated = data.duplicated(["month", "state_id"])
    if duplicated.any():
        raise ValueError("Classification table has duplicate month/state rows.")


def ordered_axes(data: pd.DataFrame) -> tuple[list[int], list[int]]:
    months = [int(value) for value in data["month"].drop_duplicates()]
    states = sorted(int(value) for value in data["state_id"].unique())
    expected_rows = len(months) * len(states)
    if len(data) != expected_rows:
        raise ValueError(
            f"Expected a complete month/state grid with {expected_rows} rows, "
            f"found {len(data)}."
        )
    return months, states


def bool_matrix(
    data: pd.DataFrame, column: str, months: list[int], states: list[int]
) -> np.ndarray:
    frame = data.pivot(index="state_id", columns="month", values=column)
    return frame.reindex(index=states, columns=months).astype(bool).to_numpy(dtype=int)


def save_figure(
    fig,
    figures_dir: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for fmt in formats:
        path = figures_dir / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = max(150, int(dpi))
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    return paths


def plot_history_panels(
    data: pd.DataFrame,
    figures_dir: Path,
    timestamp: str,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt = configure_matplotlib()
    months, states = ordered_axes(data)
    month_positions = {month: index for index, month in enumerate(months)}
    state_positions = {state: index for index, state in enumerate(states)}
    observed = data.loc[data["is_observed_state"].astype(bool)]
    observed_x = [month_positions[int(value)] for value in observed["month"]]
    observed_y = [state_positions[int(value)] for value in observed["state_id"]]
    prediction_indices = [
        index
        for index, month in enumerate(months)
        if (data.loc[data["month"] == month, "period"].iloc[0] == "prediction")
    ]

    fig, axes = plt.subplots(
        len(CONCEPTS), len(ACTORS), figsize=(10.2, 12.2), sharex=True, sharey=True
    )
    cmap = ListedColormap([FALSE_COLOR, TRUE_COLOR])
    x_ticks = sparse_indices(len(months), 10)
    y_ticks = sparse_indices(len(states), 16)

    for row_index, concept in enumerate(CONCEPTS):
        for column_index, actor in enumerate(ACTORS):
            ax = axes[row_index, column_index]
            values = bool_matrix(
                data, f"{actor}_{concept}_stable", months, states
            )
            ax.imshow(
                values,
                aspect="auto",
                origin="upper",
                interpolation="nearest",
                cmap=cmap,
                vmin=0,
                vmax=1,
                rasterized=True,
            )
            ax.scatter(
                observed_x,
                observed_y,
                s=12,
                marker="s",
                facecolors="none",
                edgecolors=ACCENT_COLOR,
                linewidths=0.65,
                zorder=3,
            )
            for prediction_index in prediction_indices:
                ax.axvline(
                    prediction_index - 0.5,
                    color=ACCENT_COLOR,
                    linewidth=1.0,
                    linestyle="--",
                )
            ax.set_title(f"{actor} - {concept}", pad=5)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(states[index]) for index in y_ticks])
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(
                [month_label(months[index]) for index in x_ticks],
                rotation=45,
                ha="right",
            )
            ax.tick_params(length=0)
            if column_index == 0:
                ax.set_ylabel("State ID")
            if row_index == len(CONCEPTS) - 1:
                ax.set_xlabel("Month")

    legend_items = [
        Patch(facecolor=TRUE_COLOR, edgecolor="none", label="Stable"),
        Patch(facecolor=FALSE_COLOR, edgecolor="#BBBBBB", label="Not stable"),
        Line2D(
            [0],
            [0],
            marker="s",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor=ACCENT_COLOR,
            linestyle="none",
            label="Observed state",
        ),
        Line2D(
            [0],
            [0],
            color=ACCENT_COLOR,
            linestyle="--",
            linewidth=1.0,
            label="Forecast boundary",
        ),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.008),
    )
    fig.suptitle("State stability by actor, concept, and month", y=0.995, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.965, bottom=0.075, hspace=0.22, wspace=0.08)
    paths = save_figure(
        fig,
        figures_dir,
        f"stability_status_over_time_{timestamp}",
        formats,
        dpi,
    )
    plt.close(fig)
    return paths


def prediction_columns() -> list[tuple[str, str, str]]:
    return [
        (actor, concept, f"{actor}_{concept}_stable")
        for actor in ACTORS
        for concept in CONCEPTS
    ]


def prediction_matrix(
    prediction: pd.DataFrame, suffix: str
) -> tuple[np.ndarray, list[int], list[str]]:
    prediction = prediction.sort_values("state_id")
    states = prediction["state_id"].astype(int).tolist()
    columns = prediction_columns()
    labels = [f"{actor}\n{concept}" for actor, concept, _ in columns]
    values = np.column_stack(
        [prediction[column.replace("_stable", suffix)].to_numpy() for _, _, column in columns]
    )
    return values, states, labels


def decorate_prediction_axis(ax, states: list[int], labels: list[str], reference_state: int) -> None:
    from matplotlib.patches import Rectangle

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", which="both", top=True, bottom=False, length=0, pad=5)
    ax.set_yticks(np.arange(len(states)))
    ax.set_yticklabels([str(state) for state in states], fontsize=5.7)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylabel("State ID")
    ax.axvline(3.5, color="white", linewidth=2.0)
    ax.axvline(3.5, color="#555555", linewidth=0.7)
    if reference_state in states:
        row = states.index(reference_state)
        ax.add_patch(
            Rectangle(
                (-0.5, row - 0.5),
                len(labels),
                1,
                fill=False,
                edgecolor=ACCENT_COLOR,
                linewidth=1.3,
                clip_on=False,
            )
        )


def plot_prediction_status(
    prediction: pd.DataFrame,
    reference_state: int,
    prediction_month: int,
    figures_dir: Path,
    timestamp: str,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    plt = configure_matplotlib()
    values, states, labels = prediction_matrix(prediction, "_stable")
    fig, ax = plt.subplots(figsize=(6.2, 11.2))
    image = ax.imshow(
        values.astype(int),
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap=ListedColormap([FALSE_COLOR, TRUE_COLOR]),
        vmin=0,
        vmax=1,
    )
    del image
    decorate_prediction_axis(ax, states, labels, reference_state)
    ax.set_title(
        f"State-level stability classification, {month_label(prediction_month)}",
        pad=36,
        fontsize=10,
    )
    ax.legend(
        handles=[
            Patch(facecolor=TRUE_COLOR, edgecolor="none", label="Stable"),
            Patch(facecolor=FALSE_COLOR, edgecolor="#BBBBBB", label="Not stable"),
            Patch(facecolor="none", edgecolor=ACCENT_COLOR, label=f"Last observed state ({reference_state})"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.91, bottom=0.075)
    paths = save_figure(
        fig,
        figures_dir,
        f"prediction_{prediction_month}_stability_status_{timestamp}",
        formats,
        dpi,
    )
    plt.close(fig)
    return paths


def plot_prediction_margins(
    prediction: pd.DataFrame,
    reference_state: int,
    prediction_month: int,
    figures_dir: Path,
    timestamp: str,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    from matplotlib.colors import TwoSlopeNorm

    plt = configure_matplotlib()
    values, states, labels = prediction_matrix(prediction, "_margin")
    bound = max(1.0, float(np.nanmax(np.abs(values))))
    fig, ax = plt.subplots(figsize=(6.7, 11.2))
    image = ax.imshow(
        values.astype(float),
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="PuOr",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
    )
    decorate_prediction_axis(ax, states, labels, reference_state)
    ax.set_title(
        f"State-level stability margins, {month_label(prediction_month)}",
        pad=36,
        fontsize=10,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.025, shrink=0.72)
    colorbar.set_label("Stability margin")
    colorbar.ax.axhline(0, color="#333333", linewidth=0.7)
    fig.subplots_adjust(left=0.14, right=0.88, top=0.91, bottom=0.045)
    paths = save_figure(
        fig,
        figures_dir,
        f"prediction_{prediction_month}_stability_margins_{timestamp}",
        formats,
        dpi,
    )
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    timestamp = run_dir.name
    result_path = find_result_file(run_dir, "state_stability_types", timestamp)
    data = pd.read_csv(result_path, encoding="utf-8-sig")
    validate_classification(data)

    prediction = data.loc[data["period"] == "prediction"].copy()
    prediction_months = prediction["month"].astype(int).unique()
    if len(prediction_months) != 1:
        raise ValueError("Expected exactly one prediction month in the result table.")
    prediction_month = int(prediction_months[0])
    historical_observed = data.loc[
        (data["period"] == "historical") & data["is_observed_state"].astype(bool)
    ]
    if historical_observed.empty:
        raise ValueError("No historical observed state is marked in the result table.")
    reference_state = int(historical_observed.iloc[-1]["state_id"])

    figures_dir = run_dir / "figures" / "stability_states"
    created: list[str] = []
    created.extend(
        plot_history_panels(data, figures_dir, timestamp, args.formats, args.dpi)
    )
    created.extend(
        plot_prediction_status(
            prediction,
            reference_state,
            prediction_month,
            figures_dir,
            timestamp,
            args.formats,
            args.dpi,
        )
    )
    created.extend(
        plot_prediction_margins(
            prediction,
            reference_state,
            prediction_month,
            figures_dir,
            timestamp,
            args.formats,
            args.dpi,
        )
    )

    manifest_path = figures_dir / f"stability_visualization_manifest_{timestamp}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "timestamp_day_hour_minute": timestamp,
                "source": str(result_path),
                "prediction_month": prediction_month,
                "last_observed_state": reference_state,
                "figure_count": len(created),
                "figures": [str(Path(path).relative_to(run_dir)) for path in created],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Created {len(created)} figure files in {figures_dir}")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
