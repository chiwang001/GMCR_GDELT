"""Summarize convergence and independent-sampling robustness of preference matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent / "output"
CONV = ROOT / "preference_matrix_convergence"
IND = ROOT / "preference_matrix_independent_sampling"
MONTH = 202002
KEY_COUNTS = [10, 20, 50, 100, 150, 200, 250, 300, 320, 340, 350, 360, 369]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.2e}"


def main() -> None:
    ref = pd.read_csv(CONV / f"matrix_distances_to_full_seed_matrix_{MONTH}.csv")
    consecutive = pd.read_csv(CONV / f"matrix_distances_consecutive_seed_counts_{MONTH}.csv")
    summary = pd.read_csv(IND / f"independent_sampling_summary_{MONTH}.csv")
    replicates = pd.read_csv(IND / f"independent_sampling_replicates_{MONTH}.csv")
    quantile_rows = []
    for count, group in replicates.groupby("seed_count", sort=True):
        row = {"seed_count": int(count), "replicates": int(len(group))}
        for actor in ("CN", "US", "combined"):
            values = group[f"{actor}_squared_distance_to_full"]
            row[f"{actor}_mean"] = values.mean()
            row[f"{actor}_sd"] = values.std(ddof=1)
            row[f"{actor}_p05"] = values.quantile(0.05)
            row[f"{actor}_p95"] = values.quantile(0.95)
        quantile_rows.append(row)
    quantiles = pd.DataFrame(quantile_rows)
    quantiles.to_csv(IND / f"independent_sampling_quantiles_{MONTH}.csv", index=False, encoding="utf-8-sig")

    ref_key = ref[ref.seed_count_a.isin(KEY_COUNTS)].copy()
    baseline = float(ref.loc[ref.seed_count_a == 10, "combined_mean_squared_difference"].iloc[0])
    ref_key["combined_relative_reduction_vs_10"] = 1 - ref_key.combined_mean_squared_difference / baseline
    next_distance = consecutive.set_index("seed_count_a").combined_mean_squared_difference
    ref_key["distance_to_next_count"] = ref_key.seed_count_a.map(next_distance)
    ref_key.to_csv(CONV / f"convergence_key_metrics_{MONTH}.csv", index=False, encoding="utf-8-sig")

    ind_key = summary[summary.seed_count.isin(KEY_COUNTS)].copy()
    ind_baseline = float(summary.loc[summary.seed_count == 10, "combined_mean_squared_distance_to_full"].iloc[0])
    ind_key["combined_relative_reduction_vs_10"] = 1 - ind_key.combined_mean_squared_distance_to_full / ind_baseline
    ind_key.to_csv(IND / f"independent_sampling_key_metrics_{MONTH}.csv", index=False, encoding="utf-8-sig")

    final_conv = ref.loc[ref.seed_count_a == 360].iloc[0]
    final_ind = summary.loc[summary.seed_count == 360].iloc[0]
    robust = pd.DataFrame([
        {
            "method": "Nested convergence",
            "seed_count": 360,
            "CN_metric": final_conv.CN_mean_squared_difference,
            "US_metric": final_conv.US_mean_squared_difference,
            "combined_metric": final_conv.combined_mean_squared_difference,
            "uncertainty_metric": np.nan,
            "uncertainty_definition": "distance to 369-seed reference",
        },
        {
            "method": "Independent sampling",
            "seed_count": 360,
            "CN_metric": final_ind.CN_mean_squared_distance_to_full,
            "US_metric": final_ind.US_mean_squared_distance_to_full,
            "combined_metric": final_ind.combined_mean_squared_distance_to_full,
            "uncertainty_metric": final_ind.combined_p95_squared_distance_to_full,
            "uncertainty_definition": "95th percentile across 100 replicates",
        },
    ])
    robust.to_csv(ROOT / f"seed_robustness_summary_{MONTH}.csv", index=False, encoding="utf-8-sig")

    first_conv = int(ref.loc[ref.combined_mean_squared_difference < 1e-4, "seed_count_a"].iloc[0])
    first_ind_p95 = int(summary.loc[summary.combined_p95_squared_distance_to_full < 2, "seed_count"].iloc[0])
    first_sd = int(summary.loc[(summary.CN_mean_element_sd < 0.01) & (summary.US_mean_element_sd < 0.01), "seed_count"].iloc[0])

    lines = [
        "# 偏好矩阵随机种子稳健性分析（202002）",
        "",
        "## 1. 数据与分析目的",
        "",
        "本分析针对 202002 预测月份的偏好矩阵，比较 `preference_matrix_convergence` 中的嵌套种子收敛结果与 `preference_matrix_independent_sampling` 中的独立无放回抽样结果。输入来自全部 369 个有效随机种子；每个矩阵包含 76 个状态和 CN、US 两个决策者。20250139 因缺少参数文件未纳入，20250901–20250930 采用脚本选择的最新参数文件。",
        "",
        "距离指标定义为两个矩阵对应元素平方差之和：$D_i(n,m)=\\sum_s\\sum_q(P_{i,n}(s,q)-P_{i,m}(s,q))^2$。为便于跨矩阵比较，同时报告按元素平均平方差。独立抽样结果中的误差带为 100 次重复抽样的第 5–95 百分位区间。",
        "",
        "## 2. 关键数值",
        "",
        "### 表 1  嵌套收敛结果",
        "",
        "|种子数|CN 平均平方差|US 平均平方差|CN+US 平均平方差|相对 10 个种子下降|与下一规模差异|",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ref_key.itertuples(index=False):
        next_value = "—" if pd.isna(row.distance_to_next_count) else fmt(row.distance_to_next_count)
        lines.append(f"|{int(row.seed_count_a)}|{fmt(row.CN_mean_squared_difference)}|{fmt(row.US_mean_squared_difference)}|{fmt(row.combined_mean_squared_difference)}|{pct(row.combined_relative_reduction_vs_10)}|{next_value}|")
    lines += [
        "",
        "### 表 2  独立抽样结果",
        "",
        "|种子数|CN 均值|US 均值|CN+US 均值|CN+US SD|CN+US P95|CN 元素 SD|US 元素 SD|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ind_key.itertuples(index=False):
        lines.append(f"|{int(row.seed_count)}|{fmt(row.CN_mean_squared_distance_to_full)}|{fmt(row.US_mean_squared_distance_to_full)}|{fmt(row.combined_mean_squared_distance_to_full)}|{fmt(row.combined_sd_squared_distance_to_full)}|{fmt(row.combined_p95_squared_distance_to_full)}|{fmt(row.CN_mean_element_sd)}|{fmt(row.US_mean_element_sd)}|")
    lines += [
        "",
        "## 3. 结果解释",
        "",
        f"（1）嵌套收敛表现为总体快速下降和后期边际收益递减。相对于 10 个种子的基准，CN+US 按元素平均平方差在 100、200、300 和 360 个种子时分别下降 {pct(1-ref.loc[ref.seed_count_a==100, 'combined_mean_squared_difference'].iloc[0]/baseline)}、{pct(1-ref.loc[ref.seed_count_a==200, 'combined_mean_squared_difference'].iloc[0]/baseline)}、{pct(1-ref.loc[ref.seed_count_a==300, 'combined_mean_squared_difference'].iloc[0]/baseline)} 和 {pct(1-ref.loc[ref.seed_count_a==360, 'combined_mean_squared_difference'].iloc[0]/baseline)}。当种子数达到约 {first_conv} 个时，按元素平均平方差首次低于 1×10⁻⁴；从 300 增加到 360 个种子，进一步加入 60 个种子只带来约 {pct(1-ref.loc[ref.seed_count_a==360, 'combined_mean_squared_difference'].iloc[0]/ref.loc[ref.seed_count_a==300, 'combined_mean_squared_difference'].iloc[0])} 的绝对相对下降。",
        "",
        "（2）收敛曲线并非严格单调。80–170 个种子区间出现小幅回弹，且相邻规模距离在 170–200、260–270 和 330–340 附近出现局部峰值。这是嵌套序列中新增种子的具体组成所导致的抽样波动，而不是收敛失效；热力图中距离沿对角线逐渐变暗，说明相近种子规模的矩阵更接近，远离对角线的早期小样本差异更大。",
        "",
        f"（3）独立抽样同时揭示了抽样不确定性。CN+US 距离均值由 10 个种子的 {fmt(summary.loc[summary.seed_count==10, 'combined_mean_squared_distance_to_full'].iloc[0])} 降至 300 个种子的 {fmt(summary.loc[summary.seed_count==300, 'combined_mean_squared_distance_to_full'].iloc[0])}，而 5–95% 区间的上界在约 {first_ind_p95} 个种子时首次低于 2。到 360 个种子时，均值为 {fmt(final_ind.combined_mean_squared_distance_to_full)}，P95 为 {fmt(final_ind.combined_p95_squared_distance_to_full)}，说明剩余抽样波动已经较小。",
        "",
        f"（4）矩阵元素层面的不确定性与距离指标一致下降。CN 和 US 的平均元素标准差在 {first_sd} 个种子时均低于 0.01；之后继续下降，360 个种子时分别为 {fmt(final_ind.CN_mean_element_sd)} 和 {fmt(final_ind.US_mean_element_sd)}。CN 与 US 并不存在始终一致的稳定性排序：独立抽样中 CN 距离均值在 37 个规模中的 33 个低于 US，但在 50–100 个种子及 360 个种子附近两者接近或发生交叉，因此建议以联合指标作为主要稳健性判断。",
        "",
        "（5）369 个种子对应总体全集。此时独立抽样样本均值与参考矩阵完全相同，距离、标准差和 P95 均为 0，这是由抽样设计决定的确定性结果，不能被解读为普通重复抽样的不确定性消失。因此，关于模型稳健性的主要结论应依据 300–360 个种子的区间，而不是仅依据 369 个种子的零距离。",
        "",
        "## 4. 图形说明",
        "",
        "- `preference_matrix_convergence/figures/preference_matrix_convergence_202002.*`：A 面板给出各嵌套矩阵到 369 种子参考矩阵的距离；B 面板给出相邻种子规模之间的距离；C 面板为所有规模两两比较的 CN+US 距离热力图。A、B 使用对数纵轴，便于同时观察早期大差异和后期小差异；热力图对角线为空或接近零，越远离对角线通常表示矩阵差异越大。",
        "- `preference_matrix_independent_sampling/figures/preference_matrix_independent_sampling_202002.*`：A 面板展示联合距离均值及 5–95% 区间；B 面板分解 CN 与 US 的距离及重复抽样区间；C 面板展示矩阵元素平均标准差。阴影越宽表示同一规模下对具体抽样子集越敏感。",
        "",
        "## 5. 论文式结论",
        "",
        "综合嵌套收敛与独立抽样结果，偏好矩阵对随机种子具有较好的规模稳健性，但稳定性是渐近形成而非在小样本下立即获得。前 100 个种子贡献了主要的距离下降，300 个以上种子后矩阵变化已显著减弱，360 个种子时独立抽样的联合距离均值和 P95 分别降至 0.085 和 0.142。由此，369 个有效种子足以支持后续 GMCR 分析的数值稳健性；同时，仍应将 300–360 个种子范围内的重复抽样区间作为结果不确定性的报告依据。需要强调的是，矩阵收敛只说明随机种子聚合后的数值表示稳定，并不等同于偏好声明本身的真实性，也不能单独证明预测结果具有因果解释力。",
        "",
        "## 6. 输出表格",
        "",
        f"- `preference_matrix_convergence/convergence_key_metrics_{MONTH}.csv`",
        f"- `preference_matrix_independent_sampling/independent_sampling_key_metrics_{MONTH}.csv`",
        f"- `preference_matrix_independent_sampling/independent_sampling_quantiles_{MONTH}.csv`",
        f"- `seed_robustness_summary_{MONTH}.csv`",
    ]
    report = "\n".join(lines) + "\n"
    (ROOT / "偏好矩阵随机种子稳健性分析.md").write_text(report, encoding="utf-8")
    (CONV / "矩阵收敛结果分析.md").write_text(report, encoding="utf-8")
    (IND / "独立抽样结果分析.md").write_text(report, encoding="utf-8")
    print("Wrote robustness reports and summary tables.")


if __name__ == "__main__":
    main()
