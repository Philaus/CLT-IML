import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

def calculate_correlation_matrix(data):

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    correlation_matrix = data[numeric_columns].corr(method="pearson")
    p_value_matrix = pd.DataFrame(
        np.zeros((len(numeric_columns), len(numeric_columns))),
        columns=numeric_columns,
        index=numeric_columns,
    )

    for i, col1 in enumerate(numeric_columns):
        for j, col2 in enumerate(numeric_columns):
            if i <= j:
                if len(data) > 2:  # 至少需要3个点来计算相关系数
                    corr, p_value = pearsonr(data[col1].values, data[col2].values)
                    p_value_matrix.iloc[i, j] = p_value
                    p_value_matrix.iloc[j, i] = p_value
                else:
                    p_value_matrix.iloc[i, j] = np.nan
                    p_value_matrix.iloc[j, i] = np.nan

    return correlation_matrix, p_value_matrix, numeric_columns


def plot_correlation_heatmap(
    correlation_matrix, p_value_matrix, title="Pearson correlation matrix"
):
    """绘制相关性热图"""

    label_mapping = {
        "r1": r"$r_1$",
        "r2": r"$r_2$",
        "r12": r"$r_{12}$",
        "s1": r"$s_1$",
        "s2": r"$s_2$",
        "crash_percentage": "Crash Percentage",
    }
    corr_mapped = correlation_matrix.rename(index=label_mapping, columns=label_mapping)
    fig, ax = plt.subplots(figsize=(6.5, 1.7))

    sns.heatmap(
        corr_mapped,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        ax=ax,
        cbar=True,
        # 将 colorbar 放到顶部，并设置横向展示
        cbar_kws={
            "orientation": "horizontal",
            "location": "top",
            "pad": 0.08,
            "shrink": 0.9,
        },
        annot_kws={"size": 17, "weight": "regular"},
        linewidths=0.5,
        linecolor="white",
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    ax.tick_params(
        axis="x", which="both", direction="out", length=4, width=1.2, labelsize=16
    )
    ax.tick_params(
        axis="y", which="both", direction="out", length=4, width=1.2, labelsize=12
    )

    plt.xticks(rotation=0, ha="center")
    plt.yticks(rotation=90)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14, direction="out", length=4, width=1.2)
    cbar.outline.set_linewidth(1.2)

    plt.savefig("correlation_matrix.png", dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":

    data = pd.read_csv("pressure_crash_cls.csv")
    r12_values = data['r2'] - data['r1']
    r2_position = data.columns.get_loc('r2')
    data.insert(r2_position + 1, 'r12', r12_values)

    # 计算相关性矩阵和p值矩阵
    correlation_matrix, p_value_matrix, numeric_columns = calculate_correlation_matrix(
        data
    )
    corr_sub = correlation_matrix.iloc[-1:, 0:5]
    p_sub = p_value_matrix.iloc[-1:, 0:5]
    plot_correlation_heatmap(corr_sub, p_sub)
    p_value_matrix.to_csv(f"correlation_analysis_p_values.csv")

    print("\n强相关性分析 (|r| > 0.7):")
    strong_correlations = []
    for i, col1 in enumerate(numeric_columns):
        for j, col2 in enumerate(numeric_columns):
            if i < j:
                corr = correlation_matrix.iloc[i, j]
                p_val = p_value_matrix.iloc[i, j]
                if abs(corr) > 0.7 and p_val < 0.05:
                    strong_correlations.append((col1, col2, corr, p_val))

    if strong_correlations:
        for col1, col2, corr, p_val in strong_correlations:
            print(f"{col1} - {col2}: r = {corr:.3f}, p = {p_val:.2e}")
    else:
        print("没有找到 |r| > 0.7 且 p < 0.05 的强相关性变量对")
