import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]  # 设置中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def extract_data(folder_name, index):
    """
    从单个数据文件中提取中心区域的平均压强值
    """
    index = str(index).zfill(3)

    # 读取背景压强
    pt0 = pd.read_csv(
        f"{folder_name}/pt0.dat",
        delimiter=" ",
        header=None,
        skipinitialspace=True,
    )
    column_1 = pt0.values[:, 0]
    p0 = column_1.reshape(256, 256)

    # 读取压强数据文件
    filename = f"{folder_name}/x12d{index}"
    df = pd.read_csv(filename, delimiter=" ", header=None, skipinitialspace=True)
    data = df.values
    column_2 = data[:, 1]
    data_matrix = column_2.reshape(256, 256)
    data_matrix = p0 + data_matrix  # 叠加背景压强

    # 读取网格坐标
    grid_xx = pd.read_csv(
        f"{folder_name}/gridxx.dat", delimiter=" ", header=None, skipinitialspace=True
    )
    grid_zz = pd.read_csv(
        f"{folder_name}/gridzz.dat", delimiter=" ", header=None, skipinitialspace=True
    )

    # 创建网格
    X, Z = np.meshgrid(grid_xx, grid_zz, indexing="ij")
    X_flat = X.flatten()
    Z_flat = Z.flatten()
    values_flat = data_matrix.flatten()

    # 创建DataFrame并调整坐标
    df_coords = pd.DataFrame({"X": X_flat, "Z": Z_flat, "Value": values_flat})
    df_coords["X"] -= 2.766
    df_coords[["Z", "X"]] = df_coords[["X", "Z"]]  # 交换X和Z坐标

    # 计算中心区域平均压强（半径0.05的圆内）
    center_x, center_z = 0.065, 0
    radius = 0.065
    distances = np.sqrt(
        (df_coords["X"] - center_x) ** 2 + (df_coords["Z"] - center_z) ** 2
    )
    in_circle_mask = distances <= radius
    mean_value = df_coords.loc[in_circle_mask, "Value"].mean()

    return mean_value


def get_pressure_time_series(folder_name):
    """
    获取一个文件夹的所有压强数据和时间序列
    """
    # 获取所有数据文件
    folder_path = Path(folder_name)
    tk_files = [f for f in folder_path.glob("x12d*") if len(f.stem) == 7]
    count = len(tk_files)

    # 提取每个时间点的中心压强
    p_axis_mean = []
    for i in range(count):
        try:
            pressure = extract_data(folder_name, i)
            p_axis_mean.append(pressure)
        except Exception as e:
            print(f"  提取 {folder_name}/x12d{i:03d} 时出错: {e}")
            p_axis_mean.append(np.nan)

    # 读取时间数据
    try:
        nstt = np.loadtxt(f"{folder_name}/nstt.dat")
        time = nstt[:, 1]
        # 确保时间序列长度与压强序列一致
        if len(time) > len(p_axis_mean):
            time = time[: len(p_axis_mean)]
        elif len(time) < len(p_axis_mean):
            p_axis_mean = p_axis_mean[: len(time)]
    except Exception as e:
        print(f"  读取时间数据时出错: {e}")
        # 生成默认时间序列
        time = np.arange(len(p_axis_mean)) * 100  # 假设时间间隔为100

    return np.array(p_axis_mean), np.array(time)


def plot_comparison(folder1, folder2, label1=None, label2=None):
    """
    绘制两个情况的压强变化对比图
    """
    if label1 is None:
        label1 = os.path.basename(folder1)
    if label2 is None:
        label2 = os.path.basename(folder2)

    print(f"正在处理案例1: {folder1}")
    pressure1, time1 = get_pressure_time_series(folder1)

    print(f"正在处理案例2: {folder2}")
    pressure2, time2 = get_pressure_time_series(folder2)

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制两条曲线
    (line1,) = plt.plot(time1, pressure1, "b-", linewidth=2.8, alpha=0.8, label=label1)
    (line2,) = plt.plot(time2, pressure2, "r-", linewidth=2.8, alpha=0.8, label=label2)

    # 设置标题和标签
    plt.title("Core Pressure Evolution", fontsize=25, fontweight="bold")
    plt.xlabel("Time/t_A", fontsize=25)
    plt.ylabel("Core Pressure", fontsize=25)
    plt.xlim(3900, 7500)
    plt.legend(fontsize=20, loc="best")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.tight_layout()
    output_filename = f"pressure_comparison_{label1}_vs_{label2}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\n对比图已保存至: {output_filename}")
    plt.show()

    return pressure1, time1, pressure2, time2


if __name__ == "__main__":
    """
    主函数：绘制两个文件夹数据的对比图
    """
    print("=" * 60)
    print("中心压强对比图生成程序")
    print("=" * 60)

    # 这里需要指定两个要对比的文件夹路径
    # 请根据你的实际情况修改这些路径
    folder1 = "case-8"  # 第一个案例的文件夹路径
    folder2 = "case-0502"  # 第二个案例的文件夹路径

    # 为每个案例指定标签（显示在图例中）
    label1 = "Case B-1108"
    label2 = "Case B-0502"

    try:
        plot_comparison(folder1, folder2, label1, label2)
        print("\n对比图生成完成！")
    except Exception as e:
        print(f"生成对比图时出错: {e}")
