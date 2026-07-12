import pandas as pd
import numpy as np
import os
from tqdm import tqdm


# ===========================================================================================
# 在每一格上都计算梯度并保存，会补充抽取大梯度的点，并且最后按照梯度大小，数据集只记录前20%的梯度值
# 2025.12.18 取样逻辑改变，先对值取对数再计算平均值
# ===========================================================================================


def calculate_grid_gradient(df, radius=0.95):
    """
    针对笛卡尔网格数据计算梯度，考虑圆形边界
    """
    # 获取唯一的X和Z坐标并排序
    x_coords = np.sort(df["X"].unique())
    z_coords = np.sort(df["Z"].unique())

    # 计算网格间距
    dx = x_coords[1] - x_coords[0]
    dz = z_coords[1] - z_coords[0]

    # 创建网格矩阵
    grid_values = np.full((len(z_coords), len(x_coords)), np.nan)
    point_indices = np.full((len(z_coords), len(x_coords)), -1, dtype=int)

    # 填充网格值并记录索引
    for idx, row in df.iterrows():
        x_idx = np.where(x_coords == row["X"])[0][0]
        z_idx = np.where(z_coords == row["Z"])[0][0]
        grid_values[z_idx, x_idx] = row["Value"]
        point_indices[z_idx, x_idx] = idx

    # 计算梯度（使用中心差分）
    grad_x = np.zeros_like(grid_values)
    grad_z = np.zeros_like(grid_values)

    # X方向梯度 (内部点使用中心差分)
    for i in range(len(z_coords)):
        for j in range(1, len(x_coords) - 1):
            if not np.isnan(grid_values[i, j - 1]) and not np.isnan(
                grid_values[i, j + 1]
            ):
                grad_x[i, j] = (grid_values[i, j + 1] - grid_values[i, j - 1]) / (
                    2 * dx
                )

    # Z方向梯度 (内部点使用中心差分)
    for i in range(1, len(z_coords) - 1):
        for j in range(len(x_coords)):
            if not np.isnan(grid_values[i - 1, j]) and not np.isnan(
                grid_values[i + 1, j]
            ):
                grad_z[i, j] = (grid_values[i + 1, j] - grid_values[i - 1, j]) / (
                    2 * dz
                )

    # 计算梯度模
    gradient_magnitude = np.sqrt(grad_x**2 + grad_z**2)

    # 将梯度信息添加回DataFrame
    df_with_grad = df.copy()
    df_with_grad["Gradient"] = 0.0

    for i in range(len(z_coords)):
        for j in range(len(x_coords)):
            idx = point_indices[i, j]
            if idx != -1:
                dist_to_center = np.sqrt(x_coords[j] ** 2 + z_coords[i] ** 2)
                if dist_to_center < radius:  # 离磁轴半径最大值
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
    对vr数据进行智能采样
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        q_params_file: q参数文件路径
        samples_per_file: 每个文件采样点数
        extra_gradient_samples: 大梯度区域额外采样点数
        random_seed: 随机种子
    """

    np.random.seed(random_seed)

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有输入文件并按顺序排序
    input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".csv")])
    print(f"找到 {len(input_files)} 个CSV文件")

    for i, filename in enumerate(tqdm(input_files, desc="处理文件中")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        df = pd.read_csv(input_path)
        df = df[df["Value"] != 0]  # 删除所有无效数据
        df["X"] -= 2.766
        df[["Z", "X"]] = df[["X", "Z"]]

        df = calculate_grid_gradient(df)

        # vr_mean = abs(df["Value"]).mean()
        # high_vr_mask = abs(df["Value"]) > vr_mean
        # low_vr_mask = abs(df["Value"]) <= vr_mean
        # high_vr_data = df[high_vr_mask]
        # low_vr_data = df[low_vr_mask]

        # 根据对数化的值的平均值划分数据
        log_values = np.log1p(np.abs(df["Value"]) / 1e-6)
        log_mean = log_values.mean()
        high_vr_mask = log_values > log_mean
        low_vr_mask = log_values <= log_mean
        high_vr_data = df[high_vr_mask]
        low_vr_data = df[low_vr_mask]

        n_high = int(samples_per_file * 3 / 8)  # 高值区域的采样百分比
        n_low = samples_per_file - n_high

        # 随机采样
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

        # 合并采样结果
        sampled_df = pd.concat([high_vr_sampled, low_vr_sampled], ignore_index=True)
        print(f"\n[文件: {filename}]")
        print(
            f"  -> 根据对数平均值划分：高值区域共有 {len(high_vr_data)} 个点，低值区域共有 {len(low_vr_data)} 个点"
        )

        # 第二步：额外基于梯度采样（从尚未被选中的点中选取）
        remaining_points = df[~df.index.isin(sampled_df.index)]

        if len(remaining_points) > 0:
            if remaining_points["Gradient"].max() > 0:  # 确保有有效梯度
                grad_mean = remaining_points["Gradient"].mean()
                high_grad_data = remaining_points[
                    remaining_points["Gradient"] > grad_mean
                ]

                print(
                    f"  -> 在未选中的剩余点中：高梯度（大于剩余点梯度均值）区域共有 {len(high_grad_data)} 个点"
                )

                # 从高梯度区域采样额外点数
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

        # 合并所有采样点
        final_sampled_df = pd.concat([sampled_df, gradient_sampled], ignore_index=True)

        # 计算前20%的阈值
        threshold = final_sampled_df["Gradient"].quantile(0.8)
        # 将后80%的Gradient值设置为None
        final_sampled_df.loc[final_sampled_df["Gradient"] <= threshold, "Gradient"] = (
            None
        )

        # 打乱顺序
        final_sampled_df = final_sampled_df.sample(
            frac=1, random_state=random_seed + i
        ).reset_index(drop=True)
        # 保存采样后的数据
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
