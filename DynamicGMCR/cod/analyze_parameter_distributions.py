"""Analyze the 42-dimensional core parameter vector across optimizer seeds.

Each ``parameter_vector_*.csv`` contains 72 structural rows: 42 estimated
model parameters and 30 theoretically fixed-zero A entries.  This script
keeps only the 42 rows marked ``estimated=1`` and summarizes their sampling
distribution across the completed ``seed_<number>`` runs.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Publication-ready typography: keep text editable in vector exports.
matplotlib.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.sans-serif": ["Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


SEED_PATTERN = re.compile(r"seed_(\d+)$")
REQUIRED_COLUMNS = {
    "index",
    "group",
    "country",
    "domain",
    "driver",
    "value",
    "estimated",
    "constraint",
}
GROUP_ORDER = ("A", "alpha", "mu")
ACTOR_ORDER = ("CN", "US")
DRIVER_CODES = ("D1", "D2", "D3", "D4", "D5")


def driver_code(value: object) -> str:
    """Return the compact D1-D5 code from a driver label."""
    text = str(value).strip()
    match = re.match(r"(D[1-5])", text)
    return match.group(1) if match else text


def parse_args() -> argparse.Namespace:
    demo_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Summarize parameter distributions across optimizer seeds."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=demo_dir / "output",
        help="Directory containing seed_<number> result directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=demo_dir / "output" / "parameter_distribution_core42",
        help="Directory for samples, summaries, figures, and the report.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=500,
        help="Expected number of seed directories with parameter vectors.",
    )
    parser.add_argument(
        "--expected-parameters",
        type=int,
        default=42,
        help="Expected number of estimated core parameters per seed.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Continue with a warning when the completed seed count differs.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Only write CSV/JSON/Markdown outputs.",
    )
    return parser.parse_args()


def discover_vectors(results_dir: Path) -> tuple[list[tuple[int, Path]], list[dict[str, object]]]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    selected: list[tuple[int, Path]] = []
    notes: list[dict[str, object]] = []
    for seed_dir in sorted(results_dir.iterdir(), key=lambda p: p.name):
        match = SEED_PATTERN.fullmatch(seed_dir.name)
        if not seed_dir.is_dir() or match is None:
            continue
        seed = int(match.group(1))
        candidates = list(seed_dir.rglob("parameter_vector_*.csv"))
        if not candidates:
            notes.append({"seed": seed, "reason": "no parameter_vector CSV"})
            continue
        candidates.sort(key=lambda p: (p.stat().st_mtime_ns, str(p)))
        if len(candidates) > 1:
            notes.append(
                {
                    "seed": seed,
                    "reason": "multiple parameter_vector CSV files; newest selected",
                    "candidate_count": len(candidates),
                }
            )
        selected.append((seed, candidates[-1]))
    selected.sort(key=lambda item: item[0])
    return selected, notes


def read_vector(seed: int, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame.copy()
    frame["seed"] = seed
    frame["parameter_file"] = str(path.resolve())
    frame["group"] = frame["group"].astype(str).str.strip()
    frame["country"] = frame["country"].fillna("").astype(str).str.strip()
    frame["domain"] = frame["domain"].fillna("").astype(str).str.strip()
    frame["driver"] = frame["driver"].fillna("").astype(str).str.strip()
    frame["driver"] = frame["driver"].map(driver_code)
    frame["index"] = pd.to_numeric(frame["index"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["estimated"] = pd.to_numeric(frame["estimated"], errors="coerce")
    if frame["value"].isna().any() or frame["estimated"].isna().any():
        raise ValueError(f"{path}: value or estimated contains non-numeric entries")
    if not np.isfinite(frame["value"].to_numpy(float)).all():
        raise ValueError(f"{path}: value contains non-finite entries")
    if not frame["estimated"].isin([0, 1]).all():
        raise ValueError(f"{path}: estimated must contain only 0 or 1")
    estimated_index = frame.loc[frame["estimated"] == 1, "index"]
    if estimated_index.isna().any() or not np.equal(estimated_index, np.floor(estimated_index)).all():
        raise ValueError(f"{path}: estimated rows must have integer core parameter indices")
    return frame[
        [
            "seed",
            "parameter_file",
            "index",
            "group",
            "country",
            "domain",
            "driver",
            "value",
            "estimated",
            "constraint",
        ]
    ]


def parameter_label(row: pd.Series) -> str:
    group = row["group"]
    if group == "A":
        return f"A[{row['country']},{row['domain']},{driver_code(row['driver'])}]"
    if group == "mu":
        return f"mu[{row['country']},{driver_code(row['driver'])}]"
    if group == "alpha":
        return f"alpha[{row['country']}]"
    return str(group)


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in samples.groupby(
        ["index", "group", "country", "domain", "driver"], dropna=False, sort=False
    ):
        core_index, group_name, country, domain, driver = keys
        values = group["value"].astype(float)
        estimated_values = group.loc[group["estimated"] == 1, "value"].astype(float)
        if estimated_values.empty:
            estimated_values = pd.Series(dtype=float)
        rows.append(
            {
                "index": int(core_index),
                "group": group_name,
                "country": country,
                "domain": domain,
                "driver": driver,
                "parameter": parameter_label(group.iloc[0]),
                "n_total": int(values.size),
                "n_estimated": int(estimated_values.size),
                "estimated_fraction": float(group["estimated"].mean()),
                "mean": float(estimated_values.mean()) if not estimated_values.empty else 0.0,
                "std": float(estimated_values.std(ddof=1)) if estimated_values.size > 1 else 0.0,
                "q05": float(estimated_values.quantile(0.05)) if not estimated_values.empty else 0.0,
                "median": float(estimated_values.median()) if not estimated_values.empty else 0.0,
                "q95": float(estimated_values.quantile(0.95)) if not estimated_values.empty else 0.0,
                "min": float(estimated_values.min()) if not estimated_values.empty else 0.0,
                "max": float(estimated_values.max()) if not estimated_values.empty else 0.0,
                "positive_fraction": float((estimated_values > 0).mean())
                if not estimated_values.empty
                else 0.0,
                "negative_fraction": float((estimated_values < 0).mean())
                if not estimated_values.empty
                else 0.0,
                "zero_fraction": float((estimated_values == 0).mean())
                if not estimated_values.empty
                else 1.0,
            }
        )
    result = pd.DataFrame(rows)
    result = result.sort_values(["index"], kind="stable")
    return result.reset_index(drop=True)


def plot_distributions(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    written: list[str] = []
    # One compact interval plot for the actor-level alpha and mu parameters.
    small = summary[summary["group"].isin(["alpha", "mu"])].copy()
    small["label"] = small.apply(
        lambda row: f"{row['country']} {driver_code(row['driver'])}" if row["group"] == "mu" else f"{row['country']} alpha",
        axis=1,
    )
    small = small.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    y = np.arange(len(small))
    ax.hlines(y, small["q05"], small["q95"], color="#2C7FB8", linewidth=2.4)
    ax.scatter(small["mean"], y, color="#D95F0E", s=42, zorder=3)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_yticks(y, small["label"])
    ax.set_xlabel("Estimated value")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.8)
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"parameter_distribution_alpha_mu.{suffix}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
        written.append(path.name)
    plt.close(fig)

    # Heatmap-like matrix of active A means, with one panel per actor.
    active = summary[(summary["group"] == "A") & (summary["n_estimated"] > 0)].copy()
    if not active.empty:
        domains = list(dict.fromkeys(active["domain"].tolist()))
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.4), sharey=True)
        max_abs = float(np.nanmax(np.abs(active[["q05", "q95"]].to_numpy())))
        max_abs = max(max_abs, 1.0)
        for axis, actor in zip(axes, ACTOR_ORDER):
            sub = active[active["country"] == actor]
            matrix = sub.pivot(index="domain", columns="driver", values="mean").reindex(
                index=domains, columns=DRIVER_CODES
            )
            image = axis.imshow(matrix.to_numpy(float), cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
            axis.set_xticks(range(len(DRIVER_CODES)), DRIVER_CODES)
            axis.set_yticks(range(len(domains)), domains)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix.iloc[i, j]
                    if pd.notna(value):
                        axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10)
        fig.subplots_adjust(left=0.16, right=0.86, bottom=0.14, top=0.96, wspace=0.16)
        colorbar_axis = fig.add_axes([0.88, 0.20, 0.025, 0.60])
        fig.colorbar(image, cax=colorbar_axis, label="Mean A coefficient")
        for suffix in ("png", "pdf", "svg"):
            path = output_dir / f"parameter_distribution_A_means.{suffix}"
            fig.savefig(path, dpi=600, bbox_inches="tight")
            written.append(path.name)
        plt.close(fig)
    return written


def _write_report_legacy(
    output_dir: Path,
    seed_count: int,
    notes: list[dict[str, object]],
    summary: pd.DataFrame,
    figure_names: Iterable[str],
) -> None:
    lines = [
        "# 500个随机种子参数分布分析",
        "",
        f"- 有效种子数：{seed_count}",
        "- 参数来源：每个 `seed_<number>` 目录中按修改时间选择的最新 `parameter_vector_*.csv`。",
        "- 统计口径：均值、样本标准差、5%-95%分位数均仅对 `estimated=1` 的参数计算。",
        "- `estimated=0` 的固定零项保留在样本文件中，但不参与有效参数分布统计。",
        "",
        "## 输出文件",
        "",
        "- `parameter_samples.csv`：逐种子参数长表。",
        "- `parameter_distribution_summary.csv`：参数分布汇总。",
        "- `parameter_distribution_manifest.json`：输入、样本和校验信息。",
    ]
    if figure_names:
        lines.extend(["- " + name for name in figure_names])
    if notes:
        lines.extend(["", "## 审计备注", ""])
        lines.extend(f"- 种子{note['seed']}：{note['reason']}" for note in notes)
    lines.extend(
        [
            "",
            "## 主要汇总",
            "",
            "| 参数组 | 决策者 | 参数 | 均值 | 标准差 | 5%-95%区间 | 估计比例 | 正值比例 | 负值比例 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['group']} | {row['country']} | {row['parameter']} | "
            f"{row['mean']:.4f} | {row['std']:.4f} | "
            f"[{row['q05']:.4f}, {row['q95']:.4f}] | "
            f"{row['estimated_fraction']:.1%} | {row['positive_fraction']:.1%} | "
            f"{row['negative_fraction']:.1%} |"
        )
    (output_dir / "parameter_distribution_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_report(
    output_dir: Path,
    seed_count: int,
    notes: list[dict[str, object]],
    summary: pd.DataFrame,
    figure_names: Iterable[str],
) -> None:
    """Write a UTF-8 Markdown report with readable labels."""
    lines = [
        "# 500个随机种子参数分布分析",
        "",
        f"- 有效种子数：{seed_count}",
        "- 参数来源：每个 `seed_<number>` 目录中按修改时间选取最新的 `parameter_vector_*.csv`。",
        "- 统计口径：均值、样本标准差和5%–95%分位数仅对 `estimated=1` 的参数计算。",
        "- `estimated=0` 的结构固定项保留在样本文件中，但不参与有效参数的分布统计。",
        "",
        "## 输出文件",
        "",
        "- `parameter_samples.csv`：逐种子参数长表。",
        "- `parameter_distribution_summary.csv`：参数分布汇总。",
        "- `parameter_distribution_manifest.json`：输入、样本和校验信息。",
    ]
    if figure_names:
        lines.extend("- " + name for name in figure_names)
    if notes:
        lines.extend(["", "## 审计备注", ""])
        lines.extend(f"- 种子{note['seed']}：{note['reason']}" for note in notes)
    lines.extend(
        [
            "",
            "## 主要汇总",
            "",
            "| 参数组 | 决策者 | 参数 | 均值 | 标准差 | 5%–95%区间 | 估计比例 | 正值比例 | 负值比例 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['group']} | {row['country']} | {row['parameter']} | "
            f"{row['mean']:.4f} | {row['std']:.4f} | "
            f"[{row['q05']:.4f}, {row['q95']:.4f}] | "
            f"{row['estimated_fraction']:.1%} | {row['positive_fraction']:.1%} | "
            f"{row['negative_fraction']:.1%} |"
        )
    (output_dir / "parameter_distribution_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, notes = discover_vectors(args.results_dir.resolve())
    if not selected:
        raise SystemExit("No parameter_vector_*.csv files were found.")
    if args.expected_seeds is not None and len(selected) != args.expected_seeds:
        message = f"Expected {args.expected_seeds} completed seeds, found {len(selected)}."
        if not args.allow_count_mismatch:
            raise SystemExit(message + " Use --allow-count-mismatch to continue.")
        print("Warning: " + message)

    all_samples = pd.concat([read_vector(seed, path) for seed, path in selected], ignore_index=True)
    samples = all_samples.loc[all_samples["estimated"] == 1].copy()
    if samples.empty:
        raise SystemExit("No estimated core parameters were found.")
    per_seed = samples.groupby("seed")["index"].agg(["count", "nunique"])
    invalid_seed_counts = per_seed[
        (per_seed["count"] != per_seed["nunique"])
        | (per_seed["nunique"] != args.expected_parameters)
    ]
    if not invalid_seed_counts.empty:
        message = (
            f"Expected {args.expected_parameters} unique core parameters per seed; "
            f"invalid counts found for {len(invalid_seed_counts)} seeds."
        )
        if not args.allow_count_mismatch:
            raise SystemExit(message + " Use --allow-count-mismatch to continue.")
        print("Warning: " + message)
    if samples.duplicated(["seed", "index"]).any():
        raise SystemExit("Core parameter indices are not unique within at least one seed.")
    samples["index"] = samples["index"].astype(int)
    samples["parameter"] = samples.apply(parameter_label, axis=1)
    sample_columns = [
        "seed",
        "parameter_file",
        "index",
        "group",
        "country",
        "domain",
        "driver",
        "parameter",
        "value",
        "estimated",
        "constraint",
    ]
    samples[sample_columns].to_csv(
        output_dir / "parameter_samples.csv", index=False, encoding="utf-8-sig", float_format="%.10g"
    )
    summary = summarize(samples)
    summary.to_csv(
        output_dir / "parameter_distribution_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    figure_names = [] if args.no_figures else plot_distributions(summary, output_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(args.results_dir.resolve()),
        "output_dir": str(output_dir),
        "expected_seed_count": args.expected_seeds,
        "completed_seed_count": len(selected),
        "expected_core_parameter_count": args.expected_parameters,
        "raw_parameter_row_count": int(len(all_samples)),
        "sample_row_count": int(len(samples)),
        "parameter_summary_row_count": int(len(summary)),
        "groups": GROUP_ORDER,
        "actors": ACTOR_ORDER,
        "statistics_use_estimated_only": True,
        "fixed_or_nonestimated_rows_excluded": int((all_samples["estimated"] == 0).sum()),
        "selected_seeds": [seed for seed, _ in selected],
        "notes": notes,
        "generated_files": [
            "parameter_samples.csv",
            "parameter_distribution_summary.csv",
            "parameter_distribution_manifest.json",
            "parameter_distribution_report.md",
            *figure_names,
        ],
    }
    (output_dir / "parameter_distribution_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readable_report(output_dir, len(selected), notes, summary, figure_names)
    print(
        f"Analyzed {len(selected)} seeds and {len(samples)} parameter rows; "
        f"wrote outputs to {output_dir}"
    )


def write_readable_report(
    output_dir: Path,
    seed_count: int,
    notes: list[dict[str, object]],
    summary: pd.DataFrame,
    figure_names: Iterable[str],
) -> None:
    """Write the report with Unicode escapes so the source remains ASCII-safe."""
    lines = [
        "# 500\u4e2a\u968f\u673a\u79cd\u5b50\u768442\u7ef4\u6838\u5fc3\u53c2\u6570\u5206\u5e03\u5206\u6790",
        "",
        f"- \u6709\u6548\u79cd\u5b50\u6570\uff1a{seed_count}",
        "- \u53c2\u6570\u6765\u6e90\uff1a\u6bcf\u4e2a `seed_<number>` \u76ee\u5f55\u4e2d\u6309\u4fee\u6539\u65f6\u95f4\u9009\u53d6\u6700\u65b0\u7684 `parameter_vector_*.csv`\u3002",
        "- \u7edf\u8ba1\u53e3\u5f84\uff1a\u5747\u503c\u3001\u6837\u672c\u6807\u51c6\u5dee\u548c5%\u201395%\u5206\u4f4d\u6570\u4ec5\u5bf9 `estimated=1` \u7684\u53c2\u6570\u8ba1\u7b97\u3002",
        "- \u7ed3\u6784\u56fa\u5b9a\u7684 `estimated=0` \u9879\u4e0d\u8ba1\u5165\u6838\u5fc3\u53c2\u6570\u5206\u5e03\uff0c\u4f46\u5728\u6e05\u5355\u4e2d\u5355\u72ec\u8bb0\u5f55\u3002",
        "",
        "## \u8f93\u51fa\u6587\u4ef6",
        "",
        "- `parameter_samples.csv`：\u9010\u79cd\u5b50\u53c2\u6570\u957f\u8868\u3002",
        "- `parameter_distribution_summary.csv`：\u53c2\u6570\u5206\u5e03\u6c47\u603b\u3002",
        "- `parameter_distribution_manifest.json`：\u8f93\u5165\u3001\u6837\u672c\u548c\u6821\u9a8c\u4fe1\u606f\u3002",
    ]
    if figure_names:
        lines.extend("- " + name for name in figure_names)
    if notes:
        lines.extend(["", "## \u5ba1\u8ba1\u5907\u6ce8", ""])
        lines.extend(f"- \u79cd\u5b50{note['seed']}：{note['reason']}" for note in notes)
    lines.extend(
        [
            "",
            "## \u4e3b\u8981\u6c47\u603b",
            "",
            "| \u53c2\u6570\u7ec4 | \u51b3\u7b56\u8005 | \u53c2\u6570 | \u5747\u503c | \u6807\u51c6\u5dee | 5%\u201395%\u533a\u95f4 | \u4f30\u8ba1\u6bd4\u4f8b | \u6b63\u503c\u6bd4\u4f8b | \u8d1f\u503c\u6bd4\u4f8b |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['group']} | {row['country']} | {row['parameter']} | "
            f"{row['mean']:.4f} | {row['std']:.4f} | "
            f"[{row['q05']:.4f}, {row['q95']:.4f}] | "
            f"{row['estimated_fraction']:.1%} | {row['positive_fraction']:.1%} | "
            f"{row['negative_fraction']:.1%} |"
        )
    (output_dir / "parameter_distribution_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
