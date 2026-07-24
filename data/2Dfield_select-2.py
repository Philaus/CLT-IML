import pandas as pd
import numpy as np
import os
from tqdm import tqdm


# ===========================================================================================
# Compute and save gradients for every grid cell, oversample large-gradient
# points, and retain gradient values only for the top 20% of the dataset.
# 2025-12-18: Changed the sampling logic to take logarithms before averaging.
# ===========================================================================================


def calculate_grid_gradient(df, radius=0.95):
    """
    Compute gradients on Cartesian-grid data while respecting a circular boundary.
    """
    # Get and sort the unique X and Z coordinates.
    x_coords = np.sort(df["X"].unique())
    z_coords = np.sort(df["Z"].unique())

    # Calculate the grid spacing.
    dx = x_coords[1] - x_coords[0]
    dz = z_coords[1] - z_coords[0]

    # Create the grid matrix.
    grid_values = np.full((len(z_coords), len(x_coords)), np.nan)
    point_indices = np.full((len(z_coords), len(x_coords)), -1, dtype=int)

    # Populate grid values and record their indices.
    for idx, row in df.iterrows():
        x_idx = np.where(x_coords == row["X"])[0][0]
        z_idx = np.where(z_coords == row["Z"])[0][0]
        grid_values[z_idx, x_idx] = row["Value"]
        point_indices[z_idx, x_idx] = idx

    # Calculate gradients using central differences.
    grad_x = np.zeros_like(grid_values)
    grad_z = np.zeros_like(grid_values)

    # X-direction gradient (central differences for interior points).
    for i in range(len(z_coords)):
        for j in range(1, len(x_coords) - 1):
            if not np.isnan(grid_values[i, j - 1]) and not np.isnan(
                grid_values[i, j + 1]
            ):
                grad_x[i, j] = (grid_values[i, j + 1] - grid_values[i, j - 1]) / (
                    2 * dx
                )

    # Z-direction gradient (central differences for interior points).
    for i in range(1, len(z_coords) - 1):
        for j in range(len(x_coords)):
            if not np.isnan(grid_values[i - 1, j]) and not np.isnan(
                grid_values[i + 1, j]
            ):
                grad_z[i, j] = (grid_values[i + 1, j] - grid_values[i - 1, j]) / (
                    2 * dz
                )

    # Calculate the gradient magnitude.
    gradient_magnitude = np.sqrt(grad_x**2 + grad_z**2)

    # Add the gradient information back to the DataFrame.
    df_with_grad = df.copy()
    df_with_grad["Gradient"] = 0.0

    for i in range(len(z_coords)):
        for j in range(len(x_coords)):
            idx = point_indices[i, j]
            if idx != -1:
                dist_to_center = np.sqrt(x_coords[j] ** 2 + z_coords[i] ** 2)
                if dist_to_center < radius:  # Maximum radius from the magnetic axis.
                    df_with_grad.at[idx, "Gradient"] = gradient_magnitude[i, j]

    return df_with_grad


def sample_vr_data(
    input_folder,
    output_folder,
    samples_per_file=1024 * 40,
    extra_gradient_samples=1024 * 5,
    random_seed=42,
):
    """
    Perform adaptive sampling of the vr data.
    Args:
        input_folder: Path to the input folder.
        output_folder: Path to the output folder.
        q_params_file: Path to the q-parameter file.
        samples_per_file: Number of points sampled from each file.
        extra_gradient_samples: Additional samples from high-gradient regions.
        random_seed: Random seed.
    """

    np.random.seed(random_seed)

    # Create the output folder.
    os.makedirs(output_folder, exist_ok=True)

    # Get all input files and sort them deterministically.
    input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".csv")])
    print(f"找到 {len(input_files)} 个CSV文件")

    for i, filename in enumerate(tqdm(input_files, desc="处理文件中")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        df = pd.read_csv(input_path)
        df = df[df["Value"] != 0]  # Remove all invalid data.
        df["X"] -= 2.766
        df[["Z", "X"]] = df[["X", "Z"]]

        df = calculate_grid_gradient(df)

        # vr_mean = abs(df["Value"]).mean()
        # high_vr_mask = abs(df["Value"]) > vr_mean
        # low_vr_mask = abs(df["Value"]) <= vr_mean
        # high_vr_data = df[high_vr_mask]
        # low_vr_data = df[low_vr_mask]

        # Partition the data using the mean of the log-transformed values.
        log_values = np.log1p(np.abs(df["Value"]) / 1e-6)
        log_mean = log_values.mean()
        high_vr_mask = log_values > log_mean
        low_vr_mask = log_values <= log_mean
        high_vr_data = df[high_vr_mask]
        low_vr_data = df[low_vr_mask]

        n_high = int(samples_per_file * 3 / 8)  # Sampling allocation for high-value regions.
        n_low = samples_per_file - n_high

        # Random sampling.
        if n_high > 0:
            high_vr_sampled = high_vr_data.sample(
                n=n_high, random_state=random_seed + i
            )
        else:
            high_vr_sampled = pd.DataFrame(columns=df.columns)

        if n_low > 0:
            low_vr_sampled = low_vr_data.sample(
                n=n_low, random_state=random_seed + i + 1000
            )
        else:
            low_vr_sampled = pd.DataFrame(columns=df.columns)

        # Combine the sampling results.
        sampled_df = pd.concat([high_vr_sampled, low_vr_sampled], ignore_index=True)
        print(f"\n[文件: {filename}]")
        print(
            f"  -> 根据对数平均值划分：高值区域共有 {len(high_vr_data)} 个点，低值区域共有 {len(low_vr_data)} 个点"
        )

        # Step 2: Sample additional points by gradient from unselected points.
        remaining_points = df[~df.index.isin(sampled_df.index)]

        if len(remaining_points) > 0:
            if remaining_points["Gradient"].max() > 0:  # Ensure that valid gradients exist.
                grad_mean = remaining_points["Gradient"].mean()
                high_grad_data = remaining_points[
                    remaining_points["Gradient"] > grad_mean
                ]

                print(
                    f"  -> 在未选中的剩余点中：高梯度（大于剩余点梯度均值）区域共有 {len(high_grad_data)} 个点"
                )

                # Draw additional samples from the high-gradient region.
                if len(high_grad_data) > 0:
                    n_gradient = min(extra_gradient_samples, len(high_grad_data))
                    gradient_sampled = high_grad_data.sample(
                        n=n_gradient,
                        random_state=random_seed + i + extra_gradient_samples,
                    )
                else:
                    gradient_sampled = pd.DataFrame(columns=df.columns)
            else:
                gradient_sampled = pd.DataFrame(columns=df.columns)
        else:
            gradient_sampled = pd.DataFrame(columns=df.columns)

        # Combine all sampled points.
        final_sampled_df = pd.concat([sampled_df, gradient_sampled], ignore_index=True)

        # Calculate the threshold for the top 20%.
        threshold = final_sampled_df["Gradient"].quantile(0.8)
        # Set Gradient to None for the remaining 80%.
        final_sampled_df.loc[final_sampled_df["Gradient"] <= threshold, "Gradient"] = (
            None
        )

        # Shuffle the row order.
        final_sampled_df = final_sampled_df.sample(
            frac=1, random_state=random_seed + i
        ).reset_index(drop=True)
        # Save the sampled data.
        final_sampled_df.to_csv(output_path, index=False)

    print(f"\n采样完成! 所有文件已保存到: {output_folder}")


if __name__ == "__main__":

    input_folder = "data_frame_by_p0"
    # output_folder = "selected_By_gradient_p0"
    output_folder = "selected_D5"
    sample_vr_data(input_folder, output_folder)

    # input_folder = "TMON-test_by_p0"
    # output_folder = "selected_test_By_gradient_p0"
    # output_folder = "selected_test_B9_633"
    # sample_vr_data(input_folder, output_folder)
