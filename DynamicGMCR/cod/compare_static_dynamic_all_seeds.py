#!/usr/bin/env python3
"""Compare the declared-option static baseline with all completed seed runs."""

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
SEED_RE = re.compile(r"seed_(\d+)$")
CONCEPTS = ("nash", "gmr", "smr", "seq")
STATIC_COLOR = OKABE_ITO["blue"]
DYNAMIC_COLOR = OKABE_ITO["orange"]
PATH_POSITIVE_COLOR = OKABE_ITO["green"]
PATH_UNSUPPORTED_COLOR = OKABE_ITO["vermillion"]


def parse_flags(value: object) -> list[bool]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return [False] * 4
    result = []
    for part in str(value).split("|")[:4]:
        try:
            result.append(float(part) >= 0.0)
        except ValueError:
            result.append(False)
    return result + [False] * (4 - len(result))


def load_all_seed_fits() -> tuple[pd.DataFrame, list[int], list[int]]:
    frames: list[pd.DataFrame] = []
    selected: list[int] = []
    skipped: list[int] = []
    for seed_dir in sorted((ROOT / "output").glob("seed_*"), key=lambda p: p.name):
        match = SEED_RE.fullmatch(seed_dir.name)
        if not match:
            continue
        seed = int(match.group(1))
        files = list(seed_dir.rglob("monthly_fit_*.csv"))
        if not files:
            skipped.append(seed)
            continue
        fit_path = max(files, key=lambda p: (p.stat().st_mtime_ns, str(p)))
        frame = pd.read_csv(fit_path, encoding="utf-8-sig")
        frame["seed"] = seed
        if len(frame) != 36:
            raise ValueError(f"Seed {seed} has {len(frame)} transitions, expected 36.")
        frames.append(frame)
        selected.append(seed)
    if not frames:
        raise FileNotFoundError("No seed monthly_fit files found.")
    return pd.concat(frames, ignore_index=True), sorted(selected), sorted(skipped)


def save_figure(fig: plt.Figure, stem: str) -> None:
    save_static_figure(fig, OUTPUT, stem)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_static_figure_style()
    static = pd.read_csv(OUTPUT / "historical_explanation.csv", encoding="utf-8-sig")
    stability = pd.read_csv(OUTPUT / "static_state_stability.csv", encoding="utf-8-sig").set_index("state_id")
    dynamic, seeds, skipped = load_all_seed_fits()
    changed_months = static.loc[static["observed_change"].eq(1), "month"].astype(int).tolist()
    unchanged_months = static.loc[static["observed_change"].eq(0), "month"].astype(int).tolist()

    # Seed-level metrics provide a distribution rather than a single selected run.
    seed_rows: list[dict[str, float | int]] = []
    for seed, frame in dynamic.groupby("seed", sort=True):
        ch = frame[frame["observed_change"].eq(1)]
        st = frame[frame["observed_change"].eq(0)]
        concept_values = {c: [] for c in CONCEPTS}
        for _, row in st.iterrows():
            cn = parse_flags(row.get("cn_margins_Nash_GMR_SMR_SEQ"))
            us = parse_flags(row.get("us_margins_Nash_GMR_SMR_SEQ"))
            for i, c in enumerate(CONCEPTS):
                concept_values[c].append(float(cn[i] and us[i]))
        record: dict[str, float | int] = {
            "seed": int(seed),
            "changed_positive_rate": float((ch["transition_probability"] > 1e-8).mean()),
            "changed_mean_probability": float(ch["transition_probability"].mean()),
            "stay_seq_rate": float(st["seq_stable_both"].fillna(0).mean()),
        }
        for c in CONCEPTS:
            record[f"stay_{c}_rate"] = float(np.mean(concept_values[c]))
        seed_rows.append(record)
    seed_metrics = pd.DataFrame(seed_rows)
    seed_metrics.to_csv(OUTPUT / "dynamic_all_seed_metrics.csv", index=False, encoding="utf-8-sig")

    def stats(column: str) -> tuple[float, float, float]:
        values = seed_metrics[column].to_numpy(dtype=float)
        return float(np.median(values)), float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))

    # Month-level median and 5%-95% interval for plotting and audit.
    month_rows: list[dict[str, float | int]] = []
    for month in sorted(set(changed_months + unchanged_months)):
        srow = static.loc[static["month"].eq(month)].iloc[0].to_dict()
        drows = dynamic.loc[dynamic["month"].eq(month)]
        record = dict(srow)
        record["dynamic_seed_count"] = int(drows["seed"].nunique())
        if month in changed_months:
            values = drows["transition_probability"].astype(float).to_numpy()
            record.update({
                "dynamic_transition_probability": float(np.median(values)),
                "dynamic_transition_probability_q05": float(np.quantile(values, 0.05)),
                "dynamic_transition_probability_q95": float(np.quantile(values, 0.95)),
                "dynamic_positive_support_rate": float(np.mean(values > 1e-8)),
            })
        else:
            values = drows["seq_stable_both"].fillna(0).astype(float).to_numpy()
            record.update({
                "dynamic_seq_stability_rate": float(np.mean(values)),
                "dynamic_transition_probability": np.nan,
                "dynamic_transition_probability_q05": np.nan,
                "dynamic_transition_probability_q95": np.nan,
                "dynamic_positive_support_rate": np.nan,
            })
        month_rows.append(record)
    monthly = pd.DataFrame(month_rows)
    monthly.to_csv(OUTPUT / "static_dynamic_monthly_comparison.csv", index=False, encoding="utf-8-sig")

    static_changed = static[static["observed_change"].eq(1)]
    static_stays = static[static["observed_change"].eq(0)]
    static_concept_rates = {
        c: float(np.mean([bool(stability.loc[int(row.start_state_id), f"joint_{c}"]) for _, row in static_stays.iterrows()]))
        for c in CONCEPTS
    }
    dynamic_summary = {"seed_count": len(seeds), "skipped_seed_count": len(skipped)}
    for key in ("changed_positive_rate", "changed_mean_probability", "stay_seq_rate"):
        med, q05, q95 = stats(key)
        dynamic_summary[f"{key}_median"] = med
        dynamic_summary[f"{key}_q05"] = q05
        dynamic_summary[f"{key}_q95"] = q95
    for c in CONCEPTS:
        med, q05, q95 = stats(f"stay_{c}_rate")
        dynamic_summary[f"stay_{c}_rate_median"] = med
        dynamic_summary[f"stay_{c}_rate_q05"] = q05
        dynamic_summary[f"stay_{c}_rate_q95"] = q95
    summary = {
        "static_model": "strategy-priority declaration GMCR",
        "dynamic_reference": "all completed seed monthly_fit files",
        "dynamic_seed_count": len(seeds),
        "skipped_seed_count": len(skipped),
        "months": 37, "transitions": 36,
        "changed_transitions": 11, "unchanged_transitions": 25,
        "static_changed_positive_rate": float((static_changed["observed_transition_probability"] > 1e-8).mean()),
        "static_changed_mean_probability": float(static_changed["observed_transition_probability"].mean()),
        "static_stay_seq_rate": float(np.mean([bool(stability.loc[int(row.start_state_id), "joint_seq"]) for _, row in static_stays.iterrows()])),
        "static_concept_joint_rates": static_concept_rates,
        "dynamic_all_seed": dynamic_summary,
        "selected_seeds": seeds,
        "skipped_seeds": skipped,
    }
    (OUTPUT / "static_dynamic_comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_rows = [
        {"metric": "Changed transition positive-support rate", "static": summary["static_changed_positive_rate"], "dynamic_median": dynamic_summary["changed_positive_rate_median"], "dynamic_q05": dynamic_summary["changed_positive_rate_q05"], "dynamic_q95": dynamic_summary["changed_positive_rate_q95"]},
        {"metric": "Mean probability on observed changes", "static": summary["static_changed_mean_probability"], "dynamic_median": dynamic_summary["changed_mean_probability_median"], "dynamic_q05": dynamic_summary["changed_mean_probability_q05"], "dynamic_q95": dynamic_summary["changed_mean_probability_q95"]},
        {"metric": "Unchanged months jointly SEQ-stable", "static": summary["static_stay_seq_rate"], "dynamic_median": dynamic_summary["stay_seq_rate_median"], "dynamic_q05": dynamic_summary["stay_seq_rate_q05"], "dynamic_q95": dynamic_summary["stay_seq_rate_q95"]},
    ]
    for c in CONCEPTS:
        metric_rows.append({"metric": f"Joint {c.upper()} hit rate on unchanged months", "static": static_concept_rates[c], "dynamic_median": dynamic_summary[f"stay_{c}_rate_median"], "dynamic_q05": dynamic_summary[f"stay_{c}_rate_q05"], "dynamic_q95": dynamic_summary[f"stay_{c}_rate_q95"]})
    pd.DataFrame(metric_rows).to_csv(OUTPUT / "static_dynamic_metric_comparison.csv", index=False, encoding="utf-8-sig")

    changed = monthly[monthly["observed_change"].eq(1)].reset_index(drop=True)
    x = np.arange(len(changed)); width = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.0), constrained_layout=True)
    ax.bar(x - width / 2, changed["observed_transition_probability"], width, label="Static strategy-priority GMCR", color=STATIC_COLOR, edgecolor="#4A4A4A", linewidth=0.45)
    med = changed["dynamic_transition_probability"].to_numpy(float)
    low = med - changed["dynamic_transition_probability_q05"].to_numpy(float)
    high = changed["dynamic_transition_probability_q95"].to_numpy(float) - med
    ax.bar(
        x + width / 2,
        med,
        width,
        yerr=np.vstack([low, high]),
        capsize=3,
        label=f"Dynamic GMCR ({len(seeds)} seeds)",
        color=DYNAMIC_COLOR,
        edgecolor="#4A4A4A",
        linewidth=0.45,
        hatch="//",
    )
    ax.set_xticks(x, changed["month"].astype(str), rotation=45, ha="right"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Observed change month"); ax.set_ylabel("Conditional transition support")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    save_figure(fig, "fig_static_dynamic_transition_support")

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True); x = np.arange(4)
    sm = np.array([static_concept_rates[c] for c in CONCEPTS]); dm = np.array([dynamic_summary[f"stay_{c}_rate_median"] for c in CONCEPTS])
    dl = dm - np.array([dynamic_summary[f"stay_{c}_rate_q05"] for c in CONCEPTS]); dh = np.array([dynamic_summary[f"stay_{c}_rate_q95"] for c in CONCEPTS]) - dm
    ax.bar(x - width / 2, sm, width, label="Static", color=STATIC_COLOR, edgecolor="#4A4A4A", linewidth=0.45)
    ax.bar(
        x + width / 2,
        dm,
        width,
        yerr=np.vstack([dl, dh]),
        capsize=3,
        label=f"Dynamic ({len(seeds)} seeds)",
        color=DYNAMIC_COLOR,
        edgecolor="#4A4A4A",
        linewidth=0.45,
        hatch="//",
    )
    ax.set_xticks(x, [c.upper() for c in CONCEPTS]); ax.set_ylim(0, 1.05); ax.set_ylabel("Joint stability hit rate"); ax.set_xlabel("GMCR stability concept")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    save_figure(fig, "fig_static_dynamic_stability_concepts")

    path = pd.read_csv(ROOT / "input" / "monthly_key_states.csv", encoding="utf-8-sig")
    path_x = np.arange(len(path))
    month_to_x = {int(month): pos for pos, month in enumerate(path["month"])}
    fig, ax = plt.subplots(figsize=(9.4, 4.0), constrained_layout=True)
    ax.step(path_x, path["state_id"], where="mid", color=OKABE_ITO["black"], linewidth=1.8)
    for _, row in static_changed.iterrows():
        supported = row["observed_transition_probability"] > 1e-8
        color = PATH_POSITIVE_COLOR if supported else PATH_UNSUPPORTED_COLOR
        marker = "o" if supported else "X"
        ax.scatter(month_to_x[int(row["month"])], row["observed_state_id"], color=color, marker=marker, edgecolor="white", linewidth=0.6, s=42, zorder=3)
    ax.set_ylabel("Observed state ID"); ax.set_xlabel("Month")
    ticks = np.linspace(0, len(path) - 1, min(10, len(path)), dtype=int); ax.set_xticks(ticks, path["month"].astype(str).iloc[ticks], rotation=45, ha="right")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=PATH_POSITIVE_COLOR, label="Positive static support", markersize=6), Line2D([0], [0], marker="X", color="w", markerfacecolor=PATH_UNSUPPORTED_COLOR, label="Unsupported by static model", markersize=6)], frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    save_figure(fig, "fig_static_historical_path_explanation")
    print(f"Compared static baseline with all {len(seeds)} completed random seeds; outputs written to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
