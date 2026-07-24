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
# 2026.3.31
# Convert all plot coordinates to SI units and use a fixed Alfvén time.
# Adjust the growth-rate selection logic for the eta10=1e-13 case.
# Annotate the linear phase in panel 3 and refine the axis formatting.
# ========================================================================================


def extract_data_to_csv(folder_name, index):
    filename = f"{folder_name}/x12d{index}"
    df = pd.read_csv(filename, delimiter=" ", header=None, skipinitialspace=True)
    data = df.values
    # column_11 = data[:, 10] # Extract v_r.
    column_11 = data[:, 6]  # Extract B_y.
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
    df_complete = pd.DataFrame({"X": X_flat, "Z": Z_flat, "Value": values_flat})
    df_complete["X"] -= 2.766
    df_complete[["Z", "X"]] = df_complete[["X", "Z"]]
    os.makedirs("data_frame_by_p0", exist_ok=True)

    # Replace path separators with underscores or another valid character.
    folder_name = folder_name.replace("/", "__").replace("\\", "__")

    df_complete.to_csv(f"data_frame_by_p0/{folder_name}.csv", index=False)


def process_energy_files(root_dir="."):
    """
    Traverse folders under the root directory and process each data file.

    Args:
        root_dir: Root-directory path; defaults to the current directory.
    """
    root_path = Path(root_dir)

    # Find all folders containing an energy.dat file.
    energy_files = list(root_path.rglob("energy.dat"))

    if not energy_files:
        print(f"在目录 {root_dir} 中未找到任何energy.dat文件")
        return {}

    print(f"检索到 {len(energy_files)} 组文件\n\n")
    for file_path in energy_files:
        try:
            # Get the relative path used to identify each case.
            relative_path = file_path.relative_to(root_path)
            folder_name = str(relative_path.parent)

            print(f"开始处理: {folder_name}")

            # Load the output files.
            energy = np.genfromtxt(file_path, skip_footer=1)

            kinetic_energy = energy[:, 2]
            max_index = np.argmax(kinetic_energy)
            if max_index == len(kinetic_energy) - 1:
                max_kinetic_energy = kinetic_energy[-1]
            else:
                start = max(0, max_index - 150)
                end = min(len(kinetic_energy), max_index + 150)
                max_kinetic_energy = np.mean(kinetic_energy[start:end])

            # Linear growth-rate detection.
            growth_rate = energy[:, 4]
            scan_range = 5000
            scan_amplitude = 0.04
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
                        folder_path = Path(folder_name)
                        tk_files = [
                            f for f in folder_path.glob("x12d*") if len(f.stem) == 7
                        ]
                        count = len(tk_files)
                        index = str(round(i / l * count)).zfill(3)
                        if index == "000":
                            index = "001"
                        extract_data_to_csv(folder_name, index)
                        break

            # Load the simulation output file.
            # if os.path.exists(f"{folder_name}/tokRZ_mpi_pll_acc_merged_v20.f90"):
            #     f90_path = f"{folder_name}/tokRZ_mpi_pll_acc_merged_v20.f90"
            # else:
            #     f90_path = f"{folder_name}/tokRZ_mpi_pll_acc_merged.f90"
            # with open(f90_path, "r") as f:
            #     flag1 = 0
            #     flag2 = 0
            #     for line in f:
            #         if flag1 and flag2:
            #             break
            #         if "eta0=" in line and "!" not in line:
            #             match = re.search(r"eta0=([0-9.eE+-]+)", line)
            #             if match:
            #                 eta = float(match.group(1))
            #                 flag1 = 1
            #         if "kap0=" in line and "!" not in line:
            #             match = re.search(r"kap0=([0-9.eE+-]+)", line)
            #             if match:
            #                 kappa = float(match.group(1))
            #                 flag2 = 1

            # Load the equilibrium file.
            qpg = np.loadtxt(f"{folder_name}/q_p_g.dat")
            q = qpg[:, 2]
            psi = qpg[:, 1]
            # q0 = qpg[0, 2]
            p0 = qpg[0, 4]
            psiN = -(psi - psi[0]) / psi[0]
            r = np.sqrt(psiN)

            # Rational-surface parameter calculation.
            solutions = []
            rational_surface = 2
            for i in range(len(q) - 1):
                if (q[i] - rational_surface) * (
                    q[i + 1] - rational_surface
                ) < 0:  # Detect an interval that crosses the rational surface.
                    # Interpolate within the interval to obtain the precise root.
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

            # Store the results.
            csv_file = "Double_Tearing_Train_Database_by_p0.csv"
            # Write the header first if the output file does not exist.
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
                            "E_kmax",
                            "gamma",
                            "folder_name",
                        ]
                    )
            # Append the data row.
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        f"{p0}",
                        f"{solutions[0]}",
                        f"{solutions[1]}",
                        f"{s1}",
                        f"{s2}",
                        f"{max_kinetic_energy}",
                        f"{ave_growth_rate}",
                        folder_name,
                    ]
                )

            # Plot the growth-rate detection result.
            plot_test(
                kinetic_energy,
                growth_rate,
                scan_range,
                ave_growth_rate,
                folder_name,
                r,
                q,
                max_kinetic_energy,
                i,
            )

        except Exception as e:
            print(f"  处理文件 {folder_name} 时出错: {e}")


def plot_test(
    kinetic_energy,
    growth_rate,
    scan_range,
    ave_growth_rate,
    folder_name,
    r,
    q,
    max_kinetic_energy,
    linear_idx,
):

    # Set font sizes.
    title_fontsize = 30
    label_fontsize = 25
    tick_fontsize = 25
    legend_fontsize = 25
    alfven_time = 1.1e-7 * 1e3

    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    ax1, ax2, ax3, ax4 = axs.flat

    # Panel 1: q profile.
    ax1.plot(r, q)
    ax1.axhline(y=2, color="r", linestyle="--", label="q=2 Rational Surface")
    ax1.set_xlim(0, 0.94)
    ax1.set_xlabel("radius/m", fontsize=label_fontsize)
    ax1.set_ylabel("q", fontsize=label_fontsize)
    ax1.set_title("q-profile", fontsize=title_fontsize)
    ax1.legend(fontsize=legend_fontsize)
    ax1.grid(True)
    ax1.tick_params(axis="both", labelsize=tick_fontsize)

    # Panel 2: kinetic-energy evolution.
    time_ax2 = np.arange(1000, len(kinetic_energy)) * alfven_time
    ax2.plot(time_ax2, kinetic_energy[1000:])
    ax2.axhline(y=max_kinetic_energy, color="r", linestyle="--", label="Max Ek")
    ax2.set_xlabel("time/ms", fontsize=label_fontsize)
    ax2.set_ylabel("Kinetic Energy", fontsize=label_fontsize)
    ax2.set_title("Kinetic Energy Evolution", fontsize=title_fontsize)
    ax2.set_yscale("log")  # Use a logarithmic scale.
    ax2.legend(fontsize=legend_fontsize)
    ax2.grid(True, which="both", ls="--")  # Add a logarithmic grid.
    ax2.tick_params(axis="both", labelsize=tick_fontsize)

    # Panel 3: growth rate.
    indices = np.arange(500, len(growth_rate), scan_range // 10)
    time_ax3 = indices * alfven_time
    ax3.plot(time_ax3, growth_rate[500 :: scan_range // 10])

    dot_x = linear_idx * alfven_time * 100
    dot_y = growth_rate[linear_idx * 100]
    ax3.plot(dot_x, dot_y, "ro", markersize=12)

    ax3.axhline(y=ave_growth_rate, color="r", linestyle="--", label="Linar Growth Rate")
    ax3.set_xlabel("time/ms", fontsize=label_fontsize)
    ax3.set_ylabel("Growth Rate", fontsize=label_fontsize)
    ax3.set_title("Growth Rate", fontsize=title_fontsize)
    ax3.legend(fontsize=legend_fontsize)
    ax3.grid(True)
    ax3.tick_params(axis="both", labelsize=tick_fontsize)
    ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    ax3.yaxis.get_offset_text().set_fontsize(tick_fontsize)

    # Panel 4: 2D distribution.
    folder_name_cleaned = folder_name.replace("/", "__").replace("\\", "__")
    df = pd.read_csv(f"data_frame_by_p0/{folder_name_cleaned}.csv")
    x = df.iloc[:, 0]
    z = df.iloc[:, 1]
    value = df.iloc[:, 2]
    scatter = ax4.scatter(x, z, c=value, cmap="rainbow")
    ax4.set_xlim(-0.94, 0.94)
    ax4.set_ylim(-0.94, 0.94)
    ax4.set_aspect("equal")
    ax4.set_xlabel("R/m", fontsize=label_fontsize)
    ax4.set_ylabel("Z/m", fontsize=label_fontsize)
    ax4.set_title("By 2D Distribution", fontsize=title_fontsize)
    ax4.tick_params(axis="both", labelsize=tick_fontsize)

    # Color bar.
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label("Value", fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.update_ticks()
    if cbar.ax.yaxis.get_offset_text():
        cbar.ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)

    plt.tight_layout()
    plt.savefig(f"data_frame_by_p0/{folder_name_cleaned}.png")
    plt.close()


def split_test_data():

    random.seed(42)

    df = pd.read_csv("Double_Tearing_Train_Database_by_p0.csv")
    # Randomly select 10% of rows for the test set.
    total_rows = len(df)
    test_size = max(1, int(total_rows * 0.1))  # Ensure at least one row.
    test_indices = random.sample(range(total_rows), test_size)

    # Extract the test-set data.
    test_df = df.iloc[test_indices].copy()

    # Remove the test-set rows from the original DataFrame.
    train_df = df.drop(test_indices).reset_index(drop=True)

    # Save the training data after removing the test set.
    train_df.to_csv("Double_Tearing_Train_Database_by_p0.csv", index=False)
    # Save the test-set data.
    test_df.to_csv("TMONet-test_by_p0.csv", index=False)
    # Create the test-set folder.
    test_folder = "TMON-test_by_p0"
    os.makedirs(test_folder, exist_ok=True)

    # Move the corresponding CSV and PNG files.
    data_frame_folder = "./data_frame_by_p0"

    moved_files = []
    for folder_name in test_df["folder_name"]:
        folder_name = folder_name.replace("/", "__").replace("\\", "__")
        # Build the source-file path.
        csv_source = os.path.join(data_frame_folder, f"{folder_name}.csv")
        png_source = os.path.join(data_frame_folder, f"{folder_name}.png")

        # Build the destination-file path.
        csv_target = os.path.join(test_folder, f"{folder_name}.csv")
        png_target = os.path.join(test_folder, f"{folder_name}.png")

        # Move the file.
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
    """Run the extraction workflow."""
    print("开始提取...")
    print("=" * 50)

    # Process all data files.
    process_energy_files()
    print(
        f"\n结果保存到 'Double_Tearing_Train_Database_by_p0.csv' 和 'data_frame_by_p0'"
    )

    print(f"\n开始划分训练集和测试集")
    split_test_data()
    print(f"\n划分完成，测试集保存到 'TMONet-test_by_p0.csv' 和 'TMON-test_by_p0'")
