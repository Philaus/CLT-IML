import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]  # Configure fonts that support Chinese text.
plt.rcParams["axes.unicode_minus"] = False  # Render minus signs correctly.


def extract_data(folder_name, index):
    """
    Extract the mean central-region pressure from one data file.
    """
    index = str(index).zfill(3)

    # Read the background pressure.
    pt0 = pd.read_csv(
        f"{folder_name}/pt0.dat",
        delimiter=" ",
        header=None,
        skipinitialspace=True,
    )
    column_1 = pt0.values[:, 0]
    p0 = column_1.reshape(256, 256)

    # Read the pressure-data file.
    filename = f"{folder_name}/x12d{index}"
    df = pd.read_csv(filename, delimiter=" ", header=None, skipinitialspace=True)
    data = df.values
    column_2 = data[:, 1]
    data_matrix = column_2.reshape(256, 256)
    data_matrix = p0 + data_matrix  # Add the background pressure.

    # Read the grid coordinates.
    grid_xx = pd.read_csv(
        f"{folder_name}/gridxx.dat", delimiter=" ", header=None, skipinitialspace=True
    )
    grid_zz = pd.read_csv(
        f"{folder_name}/gridzz.dat", delimiter=" ", header=None, skipinitialspace=True
    )

    # Create the grid.
    X, Z = np.meshgrid(grid_xx, grid_zz, indexing="ij")
    X_flat = X.flatten()
    Z_flat = Z.flatten()
    values_flat = data_matrix.flatten()

    # Create the DataFrame and adjust the coordinates.
    df_coords = pd.DataFrame({"X": X_flat, "Z": Z_flat, "Value": values_flat})
    df_coords["X"] -= 2.766
    df_coords[["Z", "X"]] = df_coords[["X", "Z"]]  # Swap the X and Z coordinates.

    # Calculate the mean pressure inside the central circle of radius 0.05.
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
    Load all pressure data and the time series from one folder.
    """
    # Get all data files.
    folder_path = Path(folder_name)
    tk_files = [f for f in folder_path.glob("x12d*") if len(f.stem) == 7]
    count = len(tk_files)

    # Extract the central pressure at each time point.
    p_axis_mean = []
    for i in range(count):
        try:
            pressure = extract_data(folder_name, i)
            p_axis_mean.append(pressure)
        except Exception as e:
            print(f"  提取 {folder_name}/x12d{i:03d} 时出错: {e}")
            p_axis_mean.append(np.nan)

    # Read the time data.
    try:
        nstt = np.loadtxt(f"{folder_name}/nstt.dat")
        time = nstt[:, 1]
        # Ensure that the time and pressure series have equal lengths.
        if len(time) > len(p_axis_mean):
            time = time[: len(p_axis_mean)]
        elif len(time) < len(p_axis_mean):
            p_axis_mean = p_axis_mean[: len(time)]
    except Exception as e:
        print(f"  读取时间数据时出错: {e}")
        # Generate the default time series.
        time = np.arange(len(p_axis_mean)) * 100  # Assume a time interval of 100.

    return np.array(p_axis_mean), np.array(time)


def plot_comparison(folder1, folder2, label1=None, label2=None):
    """
    Plot a pressure-history comparison for two cases.
    """
    if label1 is None:
        label1 = os.path.basename(folder1)
    if label2 is None:
        label2 = os.path.basename(folder2)

    print(f"正在处理案例1: {folder1}")
    pressure1, time1 = get_pressure_time_series(folder1)

    print(f"正在处理案例2: {folder2}")
    pressure2, time2 = get_pressure_time_series(folder2)

    # Create the figure.
    plt.figure(figsize=(12, 8))

    # Plot the two curves.
    (line1,) = plt.plot(time1, pressure1, "b-", linewidth=2.8, alpha=0.8, label=label1)
    (line2,) = plt.plot(time2, pressure2, "r-", linewidth=2.8, alpha=0.8, label=label2)

    # Set the title and labels.
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
    Plot a comparison of data from two folders.
    """
    print("=" * 60)
    print("中心压强对比图生成程序")
    print("=" * 60)

    # Specify the two folder paths to compare.
    # Adjust these paths for the cases being analyzed.
    folder1 = "case-8"  # Folder path for the first case.
    folder2 = "case-0502"  # Folder path for the second case.

    # Assign a legend label to each case.
    label1 = "Case B-1108"
    label2 = "Case B-0502"

    try:
        plot_comparison(folder1, folder2, label1, label2)
        print("\n对比图生成完成！")
    except Exception as e:
        print(f"生成对比图时出错: {e}")
