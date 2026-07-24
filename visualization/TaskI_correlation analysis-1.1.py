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
                if len(data) > 2:  # At least three points are required.
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
    """Plot the correlation heatmap."""

    label_mapping = {
        "r1": r"$r_1$",
        "r2": r"$r_2$",
        "r12": r"$r_{12}$",
        "s1": r"$s_1$",
        "s2": r"$s_2$",
        "p0": r"$p_0$",
        "Wt_Inner_max": r"$W_{Bi}^{\mathrm{max}}$",
        "Wt_Outer_max": r"$W_{Bo}^{\mathrm{max}}$",
        "gamma": r"$\gamma$",
        "Ekmax": r"$E_k^{\mathrm{max}}$",
    }
    corr_mapped = correlation_matrix.rename(index=label_mapping, columns=label_mapping)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    sns.heatmap(
        corr_mapped,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        ax=ax,
        cbar=True,
        # Place the color bar horizontally at the top.
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
        axis="both", which="both", direction="out", length=4, width=1.2, labelsize=16
    )

    plt.xticks(rotation=0, ha="center")
    plt.yticks(rotation=0)

    # ax.set_xticklabels(ax.get_xticklabels(), fontdict={'family': 'Arial'})
    # ax.set_yticklabels(ax.get_yticklabels(), fontdict={'family': 'Arial'})

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14, direction="out", length=4, width=1.2)
    cbar.outline.set_linewidth(1.2)

    plt.savefig("correlation_matrix.png", dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":

    # data1 = pd.read_csv("Double_Tearing_Train_Database_by_p0.csv")
    # data2 = pd.read_csv("TMONet-test_by_p0.csv")
    # data = pd.concat([data1, data2], ignore_index=True)
    data = pd.read_csv("Double_Tearing_Train_Database_Bisland_Ek.csv")
    if "folder_name" in data.columns:
        data = data.drop(columns=["folder_name"])
    r12_values = data['r2'] - data['r1']
    r2_position = data.columns.get_loc('r2')
    data.insert(r2_position + 1, 'r12', r12_values)

    # Calculate correlation and p-value matrices.
    correlation_matrix, p_value_matrix, numeric_columns = calculate_correlation_matrix(data)
    # p_value_matrix.to_csv("correlation_analysis_p_values.csv")

    independent_vars = ["r1", "r2", "r12", "s1", "s2", "p0"]
    dependent_vars = ["Wt_Inner_max", "Wt_Outer_max", "gamma", "Ekmax"]

    if independent_vars and dependent_vars:
        # Slice with dependent variables on y and independent variables on x.
        corr_sub = correlation_matrix.loc[dependent_vars, independent_vars]
        p_sub = p_value_matrix.loc[dependent_vars, independent_vars]

        plot_correlation_heatmap(corr_sub, p_sub)

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
            print(f"{col1} - {col2}: r = {corr:.3f}, p = {p_val:.4f}")
    else:
        print("没有找到 |r| > 0.7 且 p < 0.05 的强相关性变量对")
