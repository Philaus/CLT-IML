import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def extract_and_plot_csv_data():
    # 1. 定义需要提取的目标列名
    target_columns = ["Fr0", "Fdelta", "FA0", "Flamuta"]
    extracted_data = []

    # 获取当前工作目录
    current_dir = Path(".")

    print("开始遍历文件夹并提取数据...")

    # 2. 遍历当前目录下的所有子文件夹（仅限一层）
    for subdir in current_dir.iterdir():
        if subdir.is_dir():
            if subdir.name == "不好":
                continue
            # 寻找子文件夹下的所有 .csv 文件
            for csv_file in subdir.glob("*.csv"):
                try:
                    # 读取 CSV 文件
                    df = pd.read_csv(csv_file)

                    # 检查是否包含所有需要的列
                    if all(col in df.columns for col in target_columns):
                        # 提取第一行（索引为0）的数据
                        row_data = df[target_columns].iloc[0].to_dict()
                        # 记录文件名和文件夹名，方便后续追溯（可选）
                        row_data["source_file"] = f"{subdir.name}/{csv_file.name}"
                        extracted_data.append(row_data)
                except Exception as e:
                    print(f"读取文件 {csv_file} 时出错: {e}")

    # 3. 检查是否成功提取到数据
    if not extracted_data:
        print("未找到包含指定列的 CSV 数据，请检查列名或文件路径。")
        return

    # 将数据转换为 DataFrame 方便处理
    result_df = pd.DataFrame(extracted_data)
    print(f"成功提取了 {len(result_df)} 个 CSV 文件的数据。")

    # 4. 开始绘制散点图
    # 创建一个 2x2 的子图画布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # 将 2x2 矩阵展平为一维数组，方便循环

    # 设置图表样式（可选）
    (
        plt.style.use("seaborn-v0_8-whitegrid")
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else plt.style.use("default")
    )

    # 为每个变量绘制独立的散点图
    for i, col in enumerate(target_columns):
        ax = axes[i]
        # X轴使用数据的索引（第几个文件），Y轴为变量值
        ax.scatter(
            result_df.index,
            result_df[col],
            color="royalblue",
            s=50,
            alpha=0.8,
            edgecolors="none",
        )

        # 设置标题和标签
        ax.set_title(f"Scatter Plot of {col}", fontsize=14, fontweight="bold")
        ax.set_xlabel("File Index", fontsize=11)
        ax.set_ylabel(f"Value ({col})", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)

        # 如果数据量较少，可以把文件名作为 X 轴刻度（可选）
        if len(result_df) <= 15:
            ax.set_xticks(result_df.index)
            ax.set_xticklabels(
                result_df["source_file"], rotation=45, ha="right", fontsize=8
            )

    # 调整布局防止重叠
    plt.tight_layout()

    # 保存图片到当前目录（可选）
    plt.savefig("csv_data_scatter_plots.png", dpi=300)
    print("散点图已绘制完成，并保存为 'csv_data_scatter_plots.png'")

    # 显示图表
    plt.show()


if __name__ == "__main__":
    extract_and_plot_csv_data()
