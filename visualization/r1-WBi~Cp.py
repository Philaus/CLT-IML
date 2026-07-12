import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


def r1WtinmaxCp(df):

    df["X_axis"] = df["r1"] - df["Wt_Inner_max"]

    x_data = df["X_axis"]
    y_data = df["crash_percentage"]

    plt.figure(figsize=(9, 6), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid")  # 使用干净的网格背景

    # 绘制数据散点图
    plt.scatter(
        x_data,
        y_data,
        color="#1f77b4",  # 经典科学蓝
        alpha=0.7,  # 半透明度，方便观察重叠点
        edgecolors="w",  # 白色边缘，使散点更清晰
        linewidths=0.5,
        s=40,  # 散点大小
        label="Simulation Cases",
    )

    plt.axhline(
        y=0.085,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label="Crash Threshold",
    )

    plt.xlabel(r"$r_1 -  W_{Bi}^{\mathrm{max}}$", fontsize=12)
    plt.ylabel("Crash Percentage", fontsize=12)

    plt.tick_params(labelsize=10)
    plt.legend(loc="best", frameon=True, shadow=False, fontsize=10)
    plt.tight_layout()
    plt.savefig("r1-Wt_inmax~Cp.png", dpi=300, bbox_inches="tight")
    plt.show()


def r1_Wt_scatter_colored(df):

    x_data = df["r1"]
    x = df["Wt_Inner_max"]
    c_data = df["crash_percentage"]  # 用于上色的第三维数据
    y_data = x

    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=600)
    norm = mcolors.TwoSlopeNorm(vcenter=0.085, vmin=c_data.min(), vmax=c_data.max())
    cmap = plt.cm.get_cmap("RdBu_r")
    scatter = ax.scatter(
        x_data,
        y_data,
        c=c_data,
        cmap=cmap,
        norm=norm,
        alpha=0.85,
        edgecolors="black",  # 黑色细边框比白色边缘更具学术感
        linewidths=0.4,
        s=35,
        # label="Simulation Cases",
        zorder=3,
    )
    mn = min(x_data.min(), y_data.min())
    mx_val = max(x_data.max(), y_data.max())
    ax.plot(
        [mn, mx_val],
        [mn, mx_val],
        color="crimson",
        linestyle="--",
        linewidth=1.3,
        label=r"$r_1 = W_{Bi}^{\mathrm{max}}$",
        zorder=4,
    )

    ax.set_xlabel(r"$r_1$", fontsize=11, labelpad=6)
    ax.set_ylabel(r"$W_{Bi}^{\mathrm{max}}$", fontsize=11, labelpad=6)
    ax.set_xlim(mn * 0.95, mx_val * 1.05)
    ax.set_ylim(mn * 0.95, mx_val * 1.05)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Crash Percentage", fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.axhline(0.085, color="black", linestyle="-", linewidth=1)
    cbar.ax.text(
        1.08,
        0.085,
        "8.5%",
        va="center",
        ha="left",
        color="black",
        transform=cbar.ax.get_yaxis_transform(),
        fontsize=10,
    )
    ax.tick_params(
        axis="both", which="major", direction="in", labelsize=10, width=1.0, length=4
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    ax.grid(
        visible=True,
        which="major",
        axis="both",
        color="#E5E5E5",
        linestyle="--",
        linewidth=0.6,
        zorder=0,
    )

    ax.legend(frameon=False, fontsize=9, loc="center right")
    plt.tight_layout()
    plt.savefig("r1_Wt_scatter_colored0.png", dpi=600, bbox_inches="tight")


def plot_confusion_matrix(df):
    """
    根据判定标准绘制混淆矩阵
    标准: r1 - Wt_Inner_max > 0  --> 预测 Safe (<8.5%)
    """
    # 1. 建立二分类标签 (1 代表 Safe 正常类, 0 代表 Crash 崩溃类)
    y_true = (df["crash_percentage"] < 0.085).astype(int)
    x = df["Wt_Inner_max"]
    y_pred = ((df["r1"] - x / (1 + 2 * x)) > 0).astype(int)
    # 打印四项指标
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=600)
    categories = [
        "Central\nCrash\n" + r"($\geq$8.5%)",
        "Off-axis\nCrash\n" + r"($<$8.5%)",
    ]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        annot_kws={"size": 13, "weight": "bold"},
        cbar=False,
        linewidths=1.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel("Prediction", fontsize=8, labelpad=4)
    ax.set_ylabel("Real", fontsize=8, labelpad=4)

    plt.xticks(rotation=0, fontsize=7)
    plt.yticks(rotation=0, fontsize=7, va="center")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    plt.tight_layout()
    plt.savefig(
        "confusion_matrix_taskIvsII.png", dpi=600, bbox_inches="tight", transparent=True
    )


if __name__ == "__main__":

    file_name = "r1-Wt_inmax~Cp.csv"
    df = pd.read_csv(file_name)

    # r1WtinmaxCp(df)
    r1_Wt_scatter_colored(df)
    plot_confusion_matrix(df)
