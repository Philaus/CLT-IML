import os
import csv
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import interpolate
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]  # Configure fonts that support Chinese text.
plt.rcParams["axes.unicode_minus"] = False  # Render minus signs correctly.


def extract_data(folder_name, index):

    index = str(index).zfill(3)
    pt0 = pd.read_csv(
        f"{folder_name}/pt0.dat",
        delimiter=" ",
        header=None,
        skipinitialspace=True,
    )
    column_1 = pt0.values[:, 0]
    p0 = column_1.reshape(256, 256)

    filename = f"{folder_name}/x12d{index}"
    df = pd.read_csv(filename, delimiter=" ", header=None, skipinitialspace=True)
    data = df.values
    column_2 = data[:, 1]
    data_matrix = column_2.reshape(256, 256)
    data_matrix = p0 + data_matrix
    grid_xx = pd.read_csv(
        f"{folder_name}/gridxx.dat", delimiter=" ", header=None, skipinitialspace=True
    )
    grid_zz = pd.read_csv(
        f"{folder_name}/gridzz.dat", delimiter=" ", header=None, skipinitialspace=True
    )

    X, Z = np.meshgrid(grid_xx, grid_zz, indexing="ij")
    X_flat = X.flatten()
    Z_flat = Z.flatten()
    values_flat = data_matrix.flatten()
    df = pd.DataFrame({"X": X_flat, "Z": Z_flat, "Value": values_flat})

    df["X"] -= 2.766
    df[["Z", "X"]] = df[["X", "Z"]]

    center_x, center_z = 0.065, 0
    radius = 0.065
    distances = np.sqrt((df["X"] - center_x) ** 2 + (df["Z"] - center_z) ** 2)
    in_circle_mask = distances <= radius
    mean_value = df.loc[in_circle_mask, "Value"].mean()

    return mean_value


def process_data(root_dir="."):
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

            # Load the equilibrium file.
            qpg = np.loadtxt(f"{folder_name}/q_p_g.dat")
            q = qpg[:, 2]
            psi = qpg[:, 1]
            q0 = qpg[0, 2]
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

            # Determine the pressure-crash type.
            folder_path = Path(folder_name)
            tk_files = [f for f in folder_path.glob("x12d*") if len(f.stem) == 7]
            count = len(tk_files)
            p_axis_mean = []
            for i in range(count):
                p_axis_mean.append(extract_data(folder_name, i))

            time_window = 400
            crash_percentage = 0.0
            index = None
            nstt = np.loadtxt(f"{folder_name}/nstt.dat")
            time = nstt[:, 1]
            index_window = int(time_window / (time[1] - time[0]))
            for i in range(int(len(p_axis_mean) / 3), len(p_axis_mean) - index_window):
                start_pressure = p_axis_mean[i]
                min_pressure = min(p_axis_mean[i : i + index_window])
                drop_percentage = (start_pressure - min_pressure) / start_pressure
                if drop_percentage > crash_percentage:
                    crash_percentage = drop_percentage
                    index = i

            plot_pressure_with_detection(
                p_axis_mean, time, crash_percentage, index, index_window, folder_name
            )

            # Store the results.
            csv_file = "pressure_crash_cls.csv"
            # Write the header first if the output file does not exist.
            if not os.path.exists(csv_file):
                with open(csv_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "q0",
                            "r1",
                            "r2",
                            "s1",
                            "s2",
                            "crash_percentage",
                            "folder_name",
                        ]
                    )
            # Append the data row.
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        q0,
                        solutions[0],
                        solutions[1],
                        s1,
                        s2,
                        crash_percentage,
                        folder_name,
                    ]
                )

        except Exception as e:
            print(f"  处理文件 {folder_name} 时出错: {e}")


def plot_pressure_with_detection(
    p, time, crash_percentage, drop_index, time_window, folder_name
):
    """
    Plot pressure and annotate the state and abrupt-drop location.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, p, "b-", linewidth=2, label="中心压强")

    # Set the title and labels.
    plt.title(f"{folder_name},crash_rate={crash_percentage:.1%}", fontsize=14)
    plt.xlabel("时间", fontsize=12)
    plt.ylabel("中心压强", fontsize=12)
    plt.grid(True, alpha=0.3)

    if drop_index is not None:
        plt.axvline(
            x=time[drop_index],
            color="orange",
            linestyle="--",
            linewidth=2,
        )
        plt.axvline(
            x=time[drop_index + time_window],
            color="orange",
            linestyle="--",
            linewidth=2,
        )
    plt.legend()
    plt.tight_layout()
    os.makedirs("pressure", exist_ok=True)
    folder_name = folder_name.replace("/", "__").replace("\\", "__")
    plt.savefig(f"pressure/{folder_name}.png")
    plt.close()


def split_test_data():

    random.seed(42)

    df = pd.read_csv("pressure_crash_cls.csv")
    # Randomly select 10% of rows for the test set.
    total_rows = len(df)
    test_size = max(1, int(total_rows * 0.1))  # Ensure at least one row.
    test_indices = random.sample(range(total_rows), test_size)

    # Extract the test-set data.
    test_df = df.iloc[test_indices].copy()

    # Remove the test-set rows from the original DataFrame.
    train_df = df.drop(test_indices).reset_index(drop=True)

    # Save the training data after removing the test set.
    train_df.to_csv("pressure_crash_cls.csv", index=False)
    # Save the test-set data.
    test_df.to_csv("pressure_test.csv", index=False)
    # Create the test-set folder.
    test_folder = "pressure_test"
    os.makedirs(test_folder, exist_ok=True)

    # Move the corresponding CSV and PNG files.
    data_frame_folder = "./pressure"

    moved_files = []
    for folder_name in test_df["folder_name"]:
        folder_name = folder_name.replace("/", "__").replace("\\", "__")
        # Build the source-file path.
        png_source = os.path.join(data_frame_folder, f"{folder_name}.png")
        png_target = os.path.join(test_folder, f"{folder_name}.png")
        # Move the file.
        if os.path.exists(png_source):
            shutil.move(png_source, png_target)
            moved_files.append(f"{folder_name}.png")

    print(f"原始数据总数: {total_rows}")
    print(f"测试集数: {len(test_df)}")
    print(f"训练集数: {len(train_df)}")


if __name__ == "__main__":

    print("开始提取...")
    print("=" * 50)

    process_data()
    print(f"\n结果保存到 'pressure_crash_cls.csv' 和 'pressure'")
