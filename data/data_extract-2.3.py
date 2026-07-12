import os
import csv
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import interpolate
import matplotlib.pyplot as plt


# ========================================================================================
# 2026.5.5 在2.1基础上修改，将饱和动能特征换为内外有理面上的磁岛宽度
# 2026.5.18 不再需要读取二维场文件
# 2026.5.31 修正绘图出图
# 2026.6.2 添加回饱和动能
# ========================================================================================


def extract_data_to_csv(folder_name, index):
    filename = f"{folder_name}/x12d{index}"
    df = pd.read_csv(filename, delimiter=" ", header=None, skipinitialspace=True)
    data = df.values
    # column_11 = data[:, 10] # 提取v_r
    column_11 = data[:, 6]  # 提取B_y
    data_matrix = column_11.reshape(256, 256)
    grid_xx = pd.read_csv(
        f"{folder_name}/gridxx.dat", delimiter=" ", header=None, skipinitialspace=True
    )
    grid_zz = pd.read_csv(
        f"{folder_name}/gridzz.dat", delimiter=" ", header=None, skipinitialspace=True
    )

    X, Z = np.meshgrid(grid_xx, grid_zz, indexing="ij")
    X_flat = X.flatten()
    Z_flat = Z.flatten()
    normalized_matrix = data_matrix * (3e-5 / np.max(data_matrix))
    values_flat = normalized_matrix.flatten()
    df_complete = pd.DataFrame({"R": X_flat, "Z": Z_flat, "Value": values_flat})
    # df_complete["X"] -= 2.766
    # df_complete[["Z", "X"]] = df_complete[["X", "Z"]]
    os.makedirs("data_frame_Bisland", exist_ok=True)

    # 将路径分隔符替换为下划线或其他合法字符
    folder_name = folder_name.replace("/", "__").replace("\\", "__")

    df_complete.to_csv(f"data_frame_Bisland/{folder_name}.csv", index=False)


def process_energy_files(root_dir="."):
    """
    在根目录下遍历所有文件夹，处理每个数据文件

    Args:
        root_dir: 根目录路径，默认为当前目录
    """
    root_path = Path(root_dir)

    # 查找所有包含 energy.dat文件的文件夹
    energy_files = list(root_path.rglob("energy.dat"))

    if not energy_files:
        print(f"在目录 {root_dir} 中未找到任何energy.dat文件")
        return {}

    print(f"检索到 {len(energy_files)} 组文件\n\n")
    for file_path in energy_files:
        try:
            # 获取相对路径（用于标识不同的算例）
            relative_path = file_path.relative_to(root_path)
            folder_name = str(relative_path.parent)

            print(f"开始处理: {folder_name}")

            # 加载输出文件
            energy = np.genfromtxt(file_path, skip_footer=1)

            kinetic_energy = energy[:, 2]
            max_index = np.argmax(kinetic_energy)
            if max_index == len(kinetic_energy) - 1:
                max_kinetic_energy = kinetic_energy[-1]
            else:
                start = max(0, max_index - 150)
                end = min(len(kinetic_energy), max_index + 150)
                max_kinetic_energy = np.mean(kinetic_energy[start:end])

            # --- 读取 _Wt_data.csv 并提取最大磁岛宽度 ---
            wt_csv_files = list(Path(folder_name).glob("*_Wt_data.csv"))
            if wt_csv_files:
                wt_df = pd.read_csv(wt_csv_files[0])
                wt_in_max = wt_df['Wt_Inner'].max()
                wt_out_max = wt_df['Wt_Outer'].max()
            else:
                wt_in_max, wt_out_max = 0.0, 0.0

            # 线性增长率识别模块
            growth_rate = energy[:, 4]
            scan_range = 1000
            scan_amplitude = 0.06
            l = len(growth_rate)
            for i in range(l):
                flag = 1
                if (
                    abs((growth_rate[i] - growth_rate[i + scan_range]) / growth_rate[i])
                    < scan_amplitude
                ):
                    for j in range(scan_range):
                        if (
                            abs(
                                (growth_rate[i] - growth_rate[i + scan_range - j])
                                / growth_rate[i]
                            )
                            > scan_amplitude
                        ):
                            flag = 0
                            break
                    if flag:
                        ave_growth_rate = np.mean(growth_rate[i : i + scan_range])
                        # folder_path = Path(folder_name)
                        # tk_files = [
                        #     f for f in folder_path.glob("x12d*") if len(f.stem) == 7
                        # ]
                        # count = len(tk_files)
                        # index = str(round(i / l * count)).zfill(3)
                        # if index == "000":
                        #     index = "001"
                        # extract_data_to_csv(folder_name, index)
                        break

            # 加载平衡文件
            qpg = np.loadtxt(f"{folder_name}/q_p_g.dat")
            q = qpg[:, 2]
            psi = qpg[:, 1]
            p0 = qpg[0, 4]
            psiN = -(psi - psi[0]) / psi[0]
            r = np.sqrt(psiN)

            # 有理面参数模块
            solutions = []
            rational_surface = 2
            for i in range(len(q) - 1):
                if (q[i] - rational_surface) * (
                    q[i + 1] - rational_surface
                ) < 0:  # 检测跨过有理面的区间
                    # 在该区间内插值求精确解
                    interp_func = interpolate.interp1d(q[i : i + 2], r[i : i + 2])
                    solutions.append(interp_func(rational_surface))
                    if len(solutions) == 2:
                        break
            if len(solutions) == 2:
                # r12=solutions[1]-solutions[0]
                dq_dr = interpolate.interp1d(r, np.gradient(q, r), kind="linear")
                s1 = dq_dr(solutions[0]) * solutions[0] / rational_surface
                s2 = dq_dr(solutions[1]) * solutions[1] / rational_surface
            else:
                print("  没有检测到双磁剪切")

            # 存储结果
            csv_file = "Double_Tearing_Train_Database_Bisland.csv"
            # 如果文件不存在，先写入表头
            if not os.path.exists(csv_file):
                with open(csv_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "p0",
                            "r1",
                            "r2",
                            "s1",
                            "s2",
                            "Wt_Inner_max",
                            "Wt_Outer_max",
                            "gamma",
                            "Ekmax",
                            "folder_name",
                        ]
                    )
            # 追加数据行
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        f"{p0}",
                        f"{solutions[0]}",
                        f"{solutions[1]}",
                        f"{s1}",
                        f"{s2}",
                        f"{wt_in_max}",
                        f"{wt_out_max}",
                        f"{ave_growth_rate}",
                        f"{max_kinetic_energy}",
                        folder_name,
                    ]
                )

            # 绘制增长率检测图
            plot_test(
                wt_df["Wt_Inner"],
                wt_df["Wt_Outer"],
                growth_rate,
                scan_range,
                ave_growth_rate,
                folder_name,
                r,
                q,
            )

        except Exception as e:
            print(f"  处理文件 {folder_name} 时出错: {e}")


def plot_test(
    wt_in_max,
    wt_out_max,
    growth_rate,
    scan_range,
    ave_growth_rate,
    folder_name,
    r,
    q,
):

    # 设置字体大小
    title_fontsize = 30
    label_fontsize = 25
    tick_fontsize = 25
    legend_fontsize = 25

    fig, axs = plt.subplots(1, 3, figsize=(30, 7))
    ax1, ax2, ax3 = axs.flat

    # 图1：q-profile
    ax1.plot(r, q)
    ax1.axhline(y=2, color="r", linestyle="--", label="q=2 Rational Surface")
    ax1.set_xlabel("r", fontsize=label_fontsize)
    ax1.set_ylabel("q", fontsize=label_fontsize)
    ax1.set_title("q-profile", fontsize=title_fontsize)
    ax1.legend(fontsize=legend_fontsize)
    ax1.grid(True)
    ax1.tick_params(axis="both", labelsize=tick_fontsize)

    # 图2：Island Width
    ax2.plot(wt_in_max, label="In", color="royalblue", linewidth=2)
    ax2.plot(wt_out_max, label="Out", color="crimson", linestyle="--", linewidth=2)
    ax2.set_ylabel("Island Width $W_T$", fontsize=label_fontsize)
    ax2.set_title("Magnetic Island Evolution", fontsize=title_fontsize)
    ax2.set_yscale("linear")
    ax2.grid(True, ls="--", alpha=0.7)
    ax2.tick_params(axis="both", labelsize=tick_fontsize)
    ax2.legend(fontsize=tick_fontsize, loc="best", frameon=True)

    # 图3：Growth Rate
    ax3.plot(growth_rate[500 :: scan_range // 10])
    ax3.axhline(y=ave_growth_rate, color="r", linestyle="--", label="Linar Growth Rate")
    ax3.set_ylabel("Growth Rate", fontsize=label_fontsize)
    ax3.set_title("Growth Rate", fontsize=title_fontsize)
    ax3.legend(fontsize=legend_fontsize)
    ax3.grid(True)
    ax3.tick_params(axis="both", labelsize=tick_fontsize)

    # 图4：2D Distribution
    folder_name_cleaned = folder_name.replace("/", "__").replace("\\", "__")
    # df = pd.read_csv(f"data_frame_Bisland/{folder_name_cleaned}.csv")
    # x = df.iloc[:, 0]
    # z = df.iloc[:, 1]
    # value = df.iloc[:, 2]
    # scatter = ax4.scatter(x, z, c=value, cmap="rainbow")
    # ax4.set_xlabel("R", fontsize=label_fontsize)
    # ax4.set_ylabel("Z", fontsize=label_fontsize)
    # ax4.set_title("By 2D Distribution", fontsize=title_fontsize)
    # ax4.set_aspect("equal")
    # ax4.tick_params(axis="both", labelsize=tick_fontsize)

    # ax4.ticklabel_format(style="sci", axis="both", scilimits=(0, 0), useMathText=True)
    # ax4.xaxis.get_offset_text().set_fontsize(tick_fontsize)
    # ax4.yaxis.get_offset_text().set_fontsize(tick_fontsize)

    # # 颜色条
    # cbar = plt.colorbar(scatter, ax=ax4)
    # cbar.set_label("Value", fontsize=label_fontsize)
    # cbar.ax.tick_params(labelsize=tick_fontsize)
    # cbar.formatter.set_powerlimits((0, 0))
    # cbar.formatter.set_useMathText(True)
    # cbar.update_ticks()
    # if cbar.ax.yaxis.get_offset_text():
    #     cbar.ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)

    plt.tight_layout()
    os.makedirs("data_frame_Bisland", exist_ok=True)
    plt.savefig(f"data_frame_Bisland/{folder_name_cleaned}.png")
    plt.close()


def split_test_data():

    random.seed(42)

    df = pd.read_csv("Double_Tearing_Train_Database_Bislan—traind.csv")
    # 随机选择10%的行作为测试集
    total_rows = len(df)
    test_size = max(1, int(total_rows * 0.1))  # 确保至少有一行
    test_indices = random.sample(range(total_rows), test_size)

    # 提取测试集数据
    test_df = df.iloc[test_indices].copy()

    # 从原始数据框中删除测试集数据
    train_df = df.drop(test_indices).reset_index(drop=True)

    # 保存训练集数据（删除测试集后的数据）
    train_df.to_csv("Double_Tearing_Train_Database_Bisland—train.csv", index=False)
    # 保存测试集数据
    test_df.to_csv("TMONet-test_Bisland—train.csv", index=False)
    # 创建测试集文件夹
    test_folder = "TMON-test_Bisland—train"
    os.makedirs(test_folder, exist_ok=True)

    # 移动对应的CSV和PNG文件
    data_frame_folder = "./data_frame_Bisland—train"

    moved_files = []
    for folder_name in test_df["folder_name"]:
        folder_name = folder_name.replace("/", "__").replace("\\", "__")
        # 构建源文件路径
        csv_source = os.path.join(data_frame_folder, f"{folder_name}.csv")
        png_source = os.path.join(data_frame_folder, f"{folder_name}.png")

        # 构建目标文件路径
        csv_target = os.path.join(test_folder, f"{folder_name}.csv")
        png_target = os.path.join(test_folder, f"{folder_name}.png")

        # 移动文件
        if os.path.exists(csv_source):
            shutil.move(csv_source, csv_target)
            moved_files.append(f"{folder_name}.csv")

        if os.path.exists(png_source):
            shutil.move(png_source, png_target)
            moved_files.append(f"{folder_name}.png")

    print(f"原始数据总数: {total_rows}")
    print(f"测试集数: {len(test_df)}")
    print(f"训练集数: {len(train_df)}")


if __name__ == "__main__":
    """主函数"""
    print("开始提取...")
    print("=" * 50)

    # 处理所有数据文件
    process_energy_files()
    print(f"\n结果保存到 'Transferlearning_Database_Bisland—train.csv'")

    # print(f"\n开始划分训练集和测试集")
    # split_test_data()
    # print(f"\n划分完成，测试集保存到 'TMONet-test_Bisland—train.csv' 和 'TMON-test_Bisland—train'")
