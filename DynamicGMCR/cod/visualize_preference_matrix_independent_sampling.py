"""Create publication-ready figures for independent seed sampling robustness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=demo_dir / "output" / "preference_matrix_independent_sampling",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prediction-month", type=int, default=202002)
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
    )
    parser.add_argument("--dpi", type=int, default=600)
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


def quantile_table(replicates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for count, group in replicates.groupby("seed_count", sort=True):
        row = {"seed_count": int(count), "replicates": int(len(group))}
        for actor in ("CN", "US", "combined"):
            col = f"{actor}_squared_distance_to_full"
            row[f"{actor}_mean"] = float(group[col].mean())
            row[f"{actor}_p05"] = float(group[col].quantile(0.05))
            row[f"{actor}_p95"] = float(group[col].quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(fig, output_dir: Path, stem: str, formats, dpi: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def create_figure(summary: pd.DataFrame, quantiles: pd.DataFrame):
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), constrained_layout=True)
    colors = {"CN": "#0072B2", "US": "#D55E00", "combined": "#009E73"}
    labels = {"CN": "CN", "US": "US", "combined": "CN + US"}
    positive_q = quantiles[quantiles.seed_count < quantiles.seed_count.max()]

    ax = axes[0]
    x = positive_q.seed_count.to_numpy()
    mean = positive_q.combined_mean.to_numpy()
    lo = positive_q.combined_p05.to_numpy()
    hi = positive_q.combined_p95.to_numpy()
    ax.fill_between(x, lo, hi, color=colors["combined"], alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=colors["combined"], marker="o", markersize=3.0,
            linewidth=1.35, markerfacecolor="white", label="Mean")
    ax.set_yscale("log")
    ax.set_xlabel("Number of seeds")
    ax.set_ylabel("Squared difference to full matrix")
    ax.set_title("Independent sampling distance", loc="left")
    ax.text(0.98, 0.06, "Shading: 5th–95th percentile", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.75)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    for actor in ("CN", "US"):
        y = positive_q[f"{actor}_mean"].to_numpy()
        lower = positive_q[f"{actor}_p05"].to_numpy()
        upper = positive_q[f"{actor}_p95"].to_numpy()
        ax.fill_between(x, lower, upper, color=colors[actor], alpha=0.12, linewidth=0)
        ax.plot(x, y, color=colors[actor], marker="o", markersize=2.8,
                linewidth=1.25, markerfacecolor="white", label=labels[actor])
    ax.set_yscale("log")
    ax.set_xlabel("Number of seeds")
    ax.set_ylabel("Squared difference to full matrix")
    ax.set_title("Actor-specific distance", loc="left")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.75)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[2]
    s = summary[summary.seed_count < summary.seed_count.max()]
    ax.plot(s.seed_count, s.CN_mean_element_sd, color=colors["CN"], marker="o",
            markersize=2.8, linewidth=1.25, markerfacecolor="white", label="CN")
    ax.plot(s.seed_count, s.US_mean_element_sd, color=colors["US"], marker="s",
            markersize=2.8, linewidth=1.25, markerfacecolor="white", label="US")
    ax.set_yscale("log")
    ax.set_xlabel("Number of seeds")
    ax.set_ylabel("Mean element-wise SD")
    ax.set_title("Within-size sampling variability", loc="left")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.75)
    ax.legend(frameon=False, loc="upper right")

    for label, axis in zip(("A", "B", "C"), axes):
        axis.text(-0.14, 1.07, label, transform=axis.transAxes,
                  fontsize=13, fontweight="bold", va="top")
    return plt, fig


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else result_dir / "figures"
    summary_path = result_dir / f"independent_sampling_summary_{args.prediction_month}.csv"
    replicate_path = result_dir / f"independent_sampling_replicates_{args.prediction_month}.csv"
    summary = pd.read_csv(summary_path)
    replicates = pd.read_csv(replicate_path)
    quantiles = quantile_table(replicates)
    quantile_path = result_dir / f"independent_sampling_quantiles_{args.prediction_month}.csv"
    quantiles.to_csv(quantile_path, index=False, encoding="utf-8-sig")
    plt, fig = create_figure(summary, quantiles)
    stem = f"preference_matrix_independent_sampling_{args.prediction_month}"
    paths = save_figure(fig, output_dir, stem, args.formats, args.dpi)
    plt.close(fig)
    manifest = {
        "prediction_month": args.prediction_month,
        "figure_description": {
            "A": "Mean combined distance to the 369-seed reference with 5th-95th percentile band.",
            "B": "Actor-specific CN and US distances with 5th-95th percentile bands.",
            "C": "Mean element-wise standard deviation across 100 independent replicates.",
        },
        "replicates_per_seed_count": 100,
        "distance": "sum of squared corresponding matrix-element differences",
        "interval": "5th-95th percentile across independent replicates",
        "y_scale": "logarithmic; 369-seed census point omitted from plotted curves",
        "formats": list(args.formats),
        "dpi_png": int(args.dpi),
        "figures": [str(path.resolve()) for path in paths],
    }
    manifest_path = output_dir / f"visualization_manifest_{args.prediction_month}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in paths:
        print(f"Wrote {path}")
    print(f"Wrote {quantile_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
