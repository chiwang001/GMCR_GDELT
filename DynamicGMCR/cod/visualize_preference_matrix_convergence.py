"""Visualize convergence of preference matrices across nested seed counts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ACTORS = ("CN", "US")
SERIES = (
    ("CN", "CN_squared_difference_sum", "#0072B2", "-", "o"),
    ("US", "US_squared_difference_sum", "#D55E00", "--", "s"),
    (
        "CN + US",
        "combined_squared_difference_sum",
        "#1A1A1A",
        "-.",
        "^",
    ),
)
REQUIRED_COLUMNS = {
    "seed_count_a",
    "seed_count_b",
    "CN_squared_difference_sum",
    "US_squared_difference_sum",
    "combined_squared_difference_sum",
}


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Create publication-ready preference-matrix convergence figures."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=demo_dir / "output" / "preference_matrix_convergence",
        help="Directory containing matrix distance CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure directory; defaults to <result-dir>/figures.",
    )
    parser.add_argument(
        "--prediction-month",
        type=int,
        default=None,
        help="Prediction month in YYYYMM form; inferred from the manifest by default.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
        help="Output formats (default: png pdf).",
    )
    parser.add_argument(
        "--dpi", type=int, default=600, help="Raster resolution (default: 600 DPI)."
    )
    return parser.parse_args()


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.sans-serif": ["Times New Roman"],
            "font.size": 12.0,
            "axes.titlesize": 14.0,
            "axes.labelsize": 13.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 10.5,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )
    return plt


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_prediction_month(result_dir: Path, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    manifests = sorted(result_dir.glob("matrix_distance_manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(
            f"No matrix_distance_manifest_*.json found in {result_dir}."
        )
    payload = read_json(manifests[-1])
    if "prediction_month" not in payload:
        match = re.search(r"(\d{6})$", manifests[-1].stem)
        if match is None:
            raise ValueError("Cannot infer prediction month from the manifest.")
        return int(match.group(1))
    return int(payload["prediction_month"])


def load_distance_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(REQUIRED_COLUMNS.difference(table.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if table.empty:
        raise ValueError(f"{path.name} is empty.")
    if table.duplicated(["seed_count_a", "seed_count_b"]).any():
        raise ValueError(f"{path.name} contains duplicate seed-count pairs.")
    values = table[list(REQUIRED_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{path.name} contains invalid numeric values.")
    distance_columns = [column for column in REQUIRED_COLUMNS if "difference" in column]
    if (table[distance_columns] < 0).any().any():
        raise ValueError(f"{path.name} contains negative squared distances.")
    return table.sort_values(["seed_count_a", "seed_count_b"]).reset_index(drop=True)


def sparse_ticks(values: Iterable[int], maximum: int = 8) -> list[int]:
    ordered = list(values)
    if len(ordered) <= maximum:
        return ordered
    indices = np.unique(np.linspace(0, len(ordered) - 1, maximum, dtype=int))
    return [int(ordered[index]) for index in indices]


def plot_distance_lines(ax, data: pd.DataFrame, x_column: str) -> None:
    for label, column, color, line_style, marker in SERIES:
        values = data[column].to_numpy(dtype=float)
        positive = values > 0.0
        ax.plot(
            data.loc[positive, x_column],
            values[positive],
            label=label,
            color=color,
            linestyle=line_style,
            marker=marker,
            linewidth=1.8,
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.0,
        )
    ax.set_yscale("log")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.75)
    ax.set_axisbelow(True)


def build_combined_distance_grid(
    all_pairs: pd.DataFrame,
) -> tuple[list[int], np.ndarray]:
    counts = sorted(
        set(all_pairs["seed_count_a"].astype(int)).union(
            all_pairs["seed_count_b"].astype(int)
        )
    )
    index = {count: position for position, count in enumerate(counts)}
    grid = np.full((len(counts), len(counts)), np.nan, dtype=float)
    for row in all_pairs.itertuples(index=False):
        a = index[int(row.seed_count_a)]
        b = index[int(row.seed_count_b)]
        value = float(row.combined_squared_difference_sum)
        grid[a, b] = value
        grid[b, a] = value
    return counts, grid


def create_figure(
    reference: pd.DataFrame,
    consecutive: pd.DataFrame,
    all_pairs: pd.DataFrame,
    full_count: int,
):
    plt = configure_matplotlib()
    from matplotlib.colors import LogNorm

    fig = plt.figure(figsize=(8.4, 7.2), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
    ax_reference = fig.add_subplot(grid_spec[0, 0])
    ax_consecutive = fig.add_subplot(grid_spec[0, 1])
    ax_heatmap = fig.add_subplot(grid_spec[1, :])

    plot_distance_lines(ax_reference, reference, "seed_count_a")
    ax_reference.set_xlabel("Number of seeds")
    ax_reference.set_ylabel("Squared-difference sum")
    ax_reference.set_title(f"Distance to the {full_count}-seed matrix", loc="left")
    ax_reference.set_xticks(sparse_ticks(reference["seed_count_a"].astype(int)))
    ax_reference.legend(loc="upper right", handlelength=2.4)

    plot_distance_lines(ax_consecutive, consecutive, "seed_count_b")
    ax_consecutive.set_xlabel("Larger nested sample size")
    ax_consecutive.set_ylabel("Squared-difference sum")
    ax_consecutive.set_title("Distance between consecutive sample sizes", loc="left")
    ax_consecutive.set_xticks(sparse_ticks(consecutive["seed_count_b"].astype(int)))
    ax_consecutive.legend(loc="upper right", handlelength=2.4)

    counts, distance_grid = build_combined_distance_grid(all_pairs)
    positive_values = distance_grid[np.isfinite(distance_grid) & (distance_grid > 0)]
    if positive_values.size == 0:
        raise ValueError("All pairwise combined distances are zero.")
    masked = np.ma.masked_invalid(distance_grid)
    image = ax_heatmap.imshow(
        masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="cividis",
        norm=LogNorm(vmin=float(positive_values.min()), vmax=float(positive_values.max())),
    )
    tick_values = sparse_ticks(counts, maximum=10)
    tick_positions = [counts.index(value) for value in tick_values]
    ax_heatmap.set_xticks(tick_positions)
    ax_heatmap.set_xticklabels(tick_values)
    ax_heatmap.set_yticks(tick_positions)
    ax_heatmap.set_yticklabels(tick_values)
    ax_heatmap.set_xlabel("Number of seeds in matrix B")
    ax_heatmap.set_ylabel("Number of seeds in matrix A")
    ax_heatmap.set_title("Combined distance for all seed-count pairs", loc="left")
    for spine in ax_heatmap.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax_heatmap, pad=0.018, shrink=0.92)
    colorbar.set_label("CN + US squared-difference sum")
    colorbar.outline.set_linewidth(0.6)

    for label, axis in zip(("A", "B", "C"), (ax_reference, ax_consecutive, ax_heatmap)):
        axis.text(
            -0.12 if label != "C" else -0.06,
            1.07,
            label,
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
        )
    return plt, fig


def save_figure(fig, output_dir: Path, stem: str, formats: Iterable[str], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = int(dpi)
        fig.savefig(path, **kwargs)
        written.append(path)
    return written


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else result_dir / "figures"
    )
    try:
        prediction_month = infer_prediction_month(result_dir, args.prediction_month)
        all_pairs = load_distance_table(
            result_dir / f"matrix_distances_all_pairs_{prediction_month}.csv"
        )
        consecutive = load_distance_table(
            result_dir
            / f"matrix_distances_consecutive_seed_counts_{prediction_month}.csv"
        )
        reference = load_distance_table(
            result_dir
            / f"matrix_distances_to_full_seed_matrix_{prediction_month}.csv"
        )
        full_count = int(reference["seed_count_b"].max())
        plt, figure = create_figure(reference, consecutive, all_pairs, full_count)
        stem = f"preference_matrix_convergence_{prediction_month}"
        written = save_figure(figure, output_dir, stem, args.formats, args.dpi)
        plt.close(figure)

        manifest = {
            "prediction_month": prediction_month,
            "full_seed_count": full_count,
            "figure_description": {
                "A": "Distance from each nested seed-count matrix to the full matrix.",
                "B": "Distance between consecutive nested seed-count matrices.",
                "C": "Combined CN and US distance for every seed-count pair.",
            },
            "distance": "sum of squared corresponding matrix-element differences",
            "y_scale_panels_A_B": "logarithmic; zero full-to-full point omitted",
            "color_scale_panel_C": "logarithmic cividis",
            "formats": list(args.formats),
            "dpi_png": int(args.dpi),
            "figures": [str(path.resolve()) for path in written],
        }
        manifest_path = output_dir / f"visualization_manifest_{prediction_month}.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    for path in written:
        print(f"Wrote {path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
