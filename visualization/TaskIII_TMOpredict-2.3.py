import torch
import numpy as np
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity as ssimfun
from sklearn.metrics import r2_score
from matplotlib.colors import LogNorm
import statistics


plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]  # 设置中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ========================================================================================
# 2025.12.11 回归模型B33，数据集归一化到3e-5，增加输入参数p0
# 2026.05.14 增加了详细的用时统计
# 2026.05.14 修复了每个算例重复读取模型和归一化器的问题，并且不再划分batch，全部一次性送入gpu推理
# 2026.6.3 大幅优化了图片和文字的风格和排版
# ========================================================================================


class AsymmetricMinMaxScaler:
    """
    简化版不对称归一化器，适用于1D数据
    """

    def __init__(self, feature_range=(-1, 1)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        X = np.array(X).flatten()
        self.target_min_, self.target_max_ = self.feature_range

        # 正值统计
        positive_data = X[X > 0]
        self.pos_max_ = np.max(positive_data) if len(positive_data) > 0 else 1.0

        # 负值统计
        negative_data = X[X < 0]
        self.neg_min_ = np.min(negative_data) if len(negative_data) > 0 else -1.0

        return self

    def transform(self, X):
        X = np.array(X).flatten()
        result = np.zeros_like(X)

        # 正值部分
        positive_mask = X > 0
        result[positive_mask] = (X[positive_mask] / self.pos_max_) * self.target_max_

        # 负值部分
        negative_mask = X < 0
        result[negative_mask] = (X[negative_mask] / abs(self.neg_min_)) * abs(
            self.target_min_
        )

        return result.reshape(-1, 1) if len(result.shape) == 1 else result

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        X = np.array(X).flatten()
        result = np.zeros_like(X)

        # 正值部分逆变换
        positive_mask = X > 0
        result[positive_mask] = (X[positive_mask] / self.target_max_) * self.pos_max_

        # 负值部分逆变换
        negative_mask = X < 0
        result[negative_mask] = (X[negative_mask] / abs(self.target_min_)) * abs(
            self.neg_min_
        )

        return result.reshape(-1, 1) if len(result.shape) == 1 else result


class SinActivation(nn.Module):
    """自定义sin激活函数"""

    def forward(self, x):
        return torch.sin(x)


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self):
        self.q_scaler = StandardScaler()
        self.coord_scaler = StandardScaler()
        self.vr_scaler = AsymmetricMinMaxScaler(feature_range=(-1, 1))
        self.eps = 1e-6  # 控制对数变换平滑度

    def fit_transform(self, q_params, spatial_coords, vr_values):
        """拟合并转换数据"""
        # 1. 归一化q参数
        q_params_scaled = self.q_scaler.fit_transform(q_params)

        # 2. 归一化空间坐标 (X, Z)
        coords_scaled = self.coord_scaler.fit_transform(spatial_coords)

        # 3. 对vr值进行对数变换 + 归一化
        vr_transformed = self._transform_vr(vr_values, fit=True)

        return q_params_scaled, coords_scaled, vr_transformed

    def transform(self, q_params, spatial_coords, vr_values):
        """转换新数据"""
        q_params_scaled = self.q_scaler.transform(q_params)
        coords_scaled = self.coord_scaler.transform(spatial_coords)
        vr_transformed = self._transform_vr(vr_values, fit=False)

        return q_params_scaled, coords_scaled, vr_transformed

    def _transform_vr(self, vr_values, fit=False):
        """对 vr 值进行带符号对数变换并归一化到 [-1, 1]"""
        vr_values = np.array(vr_values).flatten()
        vr_log = np.sign(vr_values) * np.log1p(np.abs(vr_values) / self.eps)

        # 归一化到 [-1, 1]
        if fit:
            vr_scaled = self.vr_scaler.fit_transform(vr_log.reshape(-1, 1)).flatten()
        else:
            vr_scaled = self.vr_scaler.transform(vr_log.reshape(-1, 1)).flatten()

        return vr_scaled

    def _transform_true_vr(self, vr_values, fit=False):
        """对 vr 直接归一化到 [-1, 1]"""
        vr_values = np.array(vr_values).flatten()

        # 归一化到 [-1, 1]
        if fit:
            vr_scaled = self.vr_scaler.fit_transform(vr_values.reshape(-1, 1)).flatten()
        else:
            vr_scaled = self.vr_scaler.transform(vr_values.reshape(-1, 1)).flatten()

        return vr_scaled

    def inverse_transform_vr(self, vr_scaled):
        """将归一化后的 vr 值反变换回原始尺度"""
        # 1. 反归一化
        vr_log = self.vr_scaler.inverse_transform(vr_scaled.reshape(-1, 1)).flatten()

        # 2. 逆带符号对数变换
        vr_original = np.sign(vr_log) * self.eps * (np.expm1(np.abs(vr_log)))

        return vr_original

    def get_preprocessing_info(self):
        """获取预处理信息"""
        return {
            "q_scaler_mean": self.q_scaler.mean_,
            "q_scaler_scale": self.q_scaler.scale_,
            "coord_scaler_mean": self.coord_scaler.mean_,
            "coord_scaler_scale": self.coord_scaler.scale_,
            "vr_scaler_mean": self.vr_scaler.mean_,
            "vr_scaler_scale": self.vr_scaler.scale_,
        }


class TMONet(nn.Module):

    def __init__(
        self,
        branch_input_dim=5,  # r1, r2, s1, s2, p0
        trunk_input_dim=2,  # R, Z
        hidden_dim1=200,  # Brunch 隐藏层大小
        hidden_dim2=110,  # Trunk 隐藏层大小
        output_dim=90,  # 输出内积层大小
        branch_depth=4,
        trunk_depth=4,
        dropout_rate=0.08,
    ):
        super(TMONet, self).__init__()

        # Branch Network: 处理q函数参数 (r1, r2, s1, s2, p0)
        self.branch_net = self._build_mlp(
            input_dim=branch_input_dim,
            hidden_dim=hidden_dim1,
            output_dim=output_dim,
            depth=branch_depth,
            dropout_rate=dropout_rate,
        )

        # Trunk Network: 处理空间坐标 (R, Z)
        self.trunk_net = self._build_mlp(
            input_dim=trunk_input_dim,
            hidden_dim=hidden_dim2,
            output_dim=output_dim,
            depth=trunk_depth,
            dropout_rate=dropout_rate,
        )

        # 偏置项
        self.bias = nn.Parameter(torch.zeros(1))

    def _build_mlp(self, input_dim, hidden_dim, output_dim, depth, dropout_rate=0.15):
        layers = []
        current_dim = input_dim

        # 1. 只有当 depth >= 2 时，才有必要构建隐藏层
        if depth >= 2:
            for i in range(depth - 1):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(SinActivation())

                # 只有不是最后一层隐藏层时才加 Dropout
                # 最后一层隐藏层的索引是 depth - 2
                if i < depth - 2:
                    layers.append(nn.Dropout(dropout_rate))

                current_dim = hidden_dim

        # 2. 输出层（直接将最后的特征映射到物理量量级上）
        layers.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, q_params, spatial_coords):
        """
        Args:
            q_params: [batch_size, 5] - (r1, r2, s1, s2, p0)
            spatial_coords: [batch_size, 2] - 空间坐标 (R, Z)
        Returns:
            vr_pred: [batch_size, 1] - 预测的vr值
        """
        branch_output = self.branch_net(q_params)  # [batch_size, output_dim]
        trunk_output = self.trunk_net(spatial_coords)  # [batch_size, output_dim]

        vr_pred = (
            torch.sum(branch_output * trunk_output, dim=1, keepdim=True) + self.bias
        )

        return vr_pred


def predict_vr(model, preprocessor, q_params, spatial_coords, device):
    """
    使用训练好的模型预测vr值
    Args:
        model_path: 模型文件路径
        preprocessor_path: 预处理器文件路径
        q_params: [n_samples, 5] - q函数参数 (r1, r2, s1, s2, p0)
        spatial_coords: [n_samples, 2] - 空间坐标 (R, Z)
    Returns:
        vr_predictions: [n_samples] - 预测的vr值
    """
    model.eval()

    # 数据预处理 (CPU上进行)
    q_params_2d = q_params.reshape(-1, 5) if q_params.ndim == 1 else q_params
    q_scaled = preprocessor.q_scaler.transform(q_params_2d)
    coords_scaled = preprocessor.coord_scaler.transform(spatial_coords)

    # 直接将基础尺寸的数据转为tensor并移至GPU
    q_tensor = torch.FloatTensor(q_scaled).to(device)
    coords_tensor = torch.FloatTensor(coords_scaled).to(device)

    # 在 GPU 上利用 expand 扩展维度
    n_coords = coords_tensor.shape[0]
    q_tensor_expanded = q_tensor.expand(n_coords, -1)

    # 预测
    with torch.no_grad():
        vr_pred_scaled = model(q_tensor_expanded, coords_tensor)

    # 移回CPU并反归一化
    vr_pred = preprocessor.inverse_transform_vr(vr_pred_scaled.cpu().numpy())

    return vr_pred


def batch_predict(
    model,  # 传入已实例化的模型
    preprocessor,  # 传入已实例化的预处理器
    q_test,
    vr_df,
    device,
    grid_size=256,
):
    x = np.linspace(-1.0, 1.0, grid_size)
    y = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(x, y)

    radius = 1.0
    distance_from_center = np.sqrt(X**2 + Y**2)
    circle_mask = distance_from_center <= radius
    coords_flat = np.column_stack([X[circle_mask], Y[circle_mask]])

    epoch_start = time.time()

    # 将所有点一次性送入预测
    batch_pred = predict_vr(model, preprocessor, q_test, coords_flat, device)
    vr_pred_flat = batch_pred.flatten()

    epoch_time = time.time() - epoch_start
    print(f"single case time_cost: {epoch_time:.4f}s")

    # 构建网格
    preprocessor_temp = DataPreprocessor()
    vr_true_flat = preprocessor_temp._transform_true_vr(
        np.array(vr_pred_flat), fit=True
    )
    vr_pred_grid = np.full(X.shape, np.nan)
    vr_pred_grid[circle_mask] = vr_true_flat

    true_points = vr_df.iloc[:, :2].values
    true_values = preprocessor_temp._transform_true_vr(
        vr_df.iloc[:, 2].values, fit=True
    )
    vr_true_grid = griddata(true_points, true_values, (X, Y), method="linear")

    error_grid = np.abs(vr_pred_grid - vr_true_grid)

    return vr_pred_grid, vr_true_grid, error_grid, epoch_time


def create_2d_distribution_plot(
    model,
    preprocessor,
    q_test,
    vr_df,
    device,
    grid_size=256,
    save_path="vr_prediction_comparison.png",
):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    vr_pred_grid, vr_true_grid, error_grid, epoch_time = batch_predict(
        model,
        preprocessor,
        q_test,
        vr_df,
        device,
        grid_size=grid_size,
    )

    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(
        2, 5,
        height_ratios=[1.2, 1],
        width_ratios=[1.0, 0.0 ,1.0, 0.1, 1.0],  # 控制 4 列的宽度比例
        hspace=0.23,
        wspace=0.18,  # 这个 wspace 现在主要控制图1和图2之间的基础间隙
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 4])
    ax4 = fig.add_subplot(gs[1, :])

    # 子图1：预测值分布
    im1 = ax1.imshow(
        vr_pred_grid,
        extent=[-0.94, 0.94, -0.94, 0.94],
        origin="lower",
        cmap="RdBu_r",
        aspect="equal",
        vmin=-1,
        vmax=1,
    )
    ax1.set_title(
        r"$B_y$ Predict Dist.", fontsize=22, fontweight="bold"
    )
    ax1.set_xlabel("R", fontsize=20)
    ax1.set_ylabel("Z", fontsize=20)
    ax1.set_xticks([-0.5, 0, 0.5])
    ax1.set_yticks([-0.5, 0, 0.5])
    ax1.tick_params(axis="both", labelsize=18)

    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar1.ax.tick_params(labelsize=18)
    # cbar1.set_label(r"$B_y$ Predict", fontsize=14)

    # 子图2：真实值分布
    im2 = ax2.imshow(
        vr_true_grid,
        extent=[-0.94, 0.94, -0.94, 0.94],
        origin="lower",
        cmap="RdBu_r",
        aspect="equal",
        vmin=-1,
        vmax=1,
    )
    ax2.set_title(r"$B_y$ Real Dist.", fontsize=22, fontweight="bold")
    ax2.set_xlabel("R", fontsize=20)
    # ax2.set_ylabel("Z", fontsize=20)
    ax2.set_xticks([-0.5, 0, 0.5])
    ax2.set_yticks([-0.5, 0, 0.5])
    ax2.tick_params(axis="both", labelsize=18)

    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar2.ax.tick_params(labelsize=18)
    cbar2.set_label(r"$B_y$ Uniformed Value", fontsize=18)

    # 子图3：误差分布
    # 使用发散色彩映射来显示正负误差
    im3 = ax3.imshow(
        error_grid,
        extent=[-0.94, 0.94, -0.94, 0.94],
        origin="lower",
        cmap="YlOrRd",
        aspect="equal",
    )
    ax3.set_title("Relative Residual Dist.", fontsize=22, fontweight="bold")
    ax3.set_xlabel("R", fontsize=20)
    # ax3.set_ylabel("Z", fontsize=20)
    # ax3.set_xticks([-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9])
    # ax3.set_yticks([-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9])
    ax3.set_xticks([-0.5, 0, 0.5])
    ax3.set_yticks([-0.5, 0, 0.5])
    ax3.tick_params(axis="both", labelsize=18)

    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=18)
    cbar3.set_label("Residual", fontsize=18)

    # 计算当前图像的均方误差和 PSNR SSIM
    # 为了排除数据矩阵中大量的NaN值，创建有效值掩码
    valid_mask = ~(np.isnan(vr_true_grid) | np.isnan(vr_pred_grid))
    # 提取有效值
    truth_valid = vr_true_grid[valid_mask]
    pred_valid = vr_pred_grid[valid_mask]
    # 计算实际数据范围
    data_range_actual = max(
        truth_valid.max() - truth_valid.min(), pred_valid.max() - pred_valid.min()
    )

    mse = np.nanmean(error_grid**2)
    psnr = 20 * np.log10(data_range_actual) - 10 * np.log10(mse)
    # 计算SSIM（需要将1D数组重新组织为2D）
    ssim = ssimfun(truth_valid, pred_valid, data_range=data_range_actual)

    mae = np.nanmean(np.abs(error_grid))
    stats_text = f"MSE: {mse:.2e}\nMAE: {mae:.2e}"
    ax3.text(
        0.02, 0.98,
        stats_text,
        transform=ax3.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="#E0E0E0", alpha=0.9
        ),
        fontsize=14,
    )

    x = np.linspace(-0.94, 0.94, grid_size)
    y = np.linspace(-0.94, 0.94, grid_size)
    X, Y = np.meshgrid(x, y)
    z_zero_idx = np.argmin(np.abs(y - 0))  # 找到最接近Z=0的索引
    r_coords = x  # R坐标
    pred_z0 = vr_pred_grid[z_zero_idx, :]  # Z=0的预测值
    true_z0 = vr_true_grid[z_zero_idx, :]  # Z=0的真实值

    # # 子图4：绘制Z=0的R-vr曲线
    ax4.plot(
        r_coords, true_z0, color="#2A729E", linestyle="-", linewidth=2.5, label="Real"
    )
    ax4.plot(
        r_coords,
        pred_z0,
        color="#D55E00",
        linestyle="--",
        linewidth=2.5,
        label="Predict",
    )
    ax4.set_xlabel("R", fontsize=22)
    ax4.set_ylabel(r"$B_y$", fontsize=22)
    ax4.set_title(r"$B_y$ Dist. on $Z=0$ profile", fontsize=24, fontweight="bold")
    ax4.grid(True, alpha=0.4, linestyle=":")
    ax4.tick_params(axis="both", labelsize=18)
    ax4.legend(fontsize=20, frameon=False)

    # plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"均方误差 (MSE): {mse:.2e}")
    print(f"平均绝对误差 (MAE): {mae:.2e}")
    print(f"psnr: {psnr:.2f}")
    print(f"ssim: {ssim:.4f}")

    return vr_pred_grid, vr_true_grid, error_grid, psnr, ssim, epoch_time


def plot_psnr_ssim_histogram(psnr_list, ssim_list, save_path=None):
    """
    绘制PSNR和SSIM的统计分布直方图
    """
    if save_path is None:
        save_path = f"psnr_ssim.png"

    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 绘制PSNR分布直方图
    n_psnr, bins_psnr, patches_psnr = ax1.hist(
        psnr_list, bins=8, color="skyblue", edgecolor="navy", alpha=0.7
    )
    ax1.set_xlabel("PSNR (dB)", fontsize=18)
    ax1.set_ylabel("Counts", fontsize=18)
    ax1.set_title("PSNR Distribution", fontsize=18)
    ax1.set_xlim(30, 45)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="both", labelsize=18)

    # 在PSNR直方图上添加统计信息
    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list)
    ax1.text(
        0.02,
        0.98,
        f"Mean: {psnr_mean:.2f} dB\nStd: {psnr_std:.2f} dB",
        transform=ax1.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 绘制SSIM分布直方图
    ax2.hist(ssim_list, bins=8, color="lightcoral", edgecolor="darkred", alpha=0.7)
    ax2.set_xlabel("SSIM", fontsize=18)
    ax2.set_ylabel("Counts", fontsize=18)
    ax2.set_title("SSIM Distribution", fontsize=18)
    ax2.set_xlim(0.8, 1)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="both", labelsize=18)

    # 在SSIM直方图上添加统计信息
    ssim_mean = np.mean(ssim_list)
    ssim_std = np.std(ssim_list)
    ax2.text(
        0.02, 0.98,
        f"Mean: {ssim_mean:.4f}\nStd: {ssim_std:.4f}",
        transform=ax2.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"PSNR_ave: {psnr_mean:.2f} dB\nSSIM_ave: {ssim_mean:.4f}")


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    preprocessor = torch.load(
        "preprocessor_CE1.pth", map_location=device, weights_only=False
    )
    checkpoint = torch.load(
        "TMONet_model_CE1.pth", map_location=device, weights_only=False
    )
    model = TMONet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv("TMONet-test_by_p0.csv")
    num_rows = len(df)

    epoch_start = time.time()

    all_psnr = []
    all_ssim = []

    all_predictions = []
    all_targets = []

    all_predict_time = []

    for row_index in range(num_rows):

        print(f"\n{'='*60}")
        print(f"Predicting Case {row_index+1}/{num_rows}")
        print(f"{'='*60}")

        # epoch_start_load = time.time()
        q_test = df.iloc[row_index].values[:5]

        folder_name = df.iloc[row_index].values[-1]
        folder_name = folder_name.replace("/", "__").replace("\\", "__")
        vr_df = pd.read_csv(os.path.join("TMON-test_by_p0", f"{folder_name}.csv"))
        vr_df = vr_df[vr_df["Value"] != 0]
        vr_df["X"] -= 2.766
        vr_df[["Z", "X"]] = vr_df[["X", "Z"]]

        # epoch_time_load = time.time() - epoch_start_load
        # print(f"dataload time_cost: {epoch_time_load:.3f}s")

        vr_pred_grid, vr_true_grid, error_grid, psnr, ssim, epoch_time_single = (
            create_2d_distribution_plot(
                model,
                preprocessor,
                q_test,
                vr_df,
                device,
                grid_size=256,
                save_path=f"distribution_plot_{row_index}.png",
            )
        )
        all_predict_time.append(epoch_time_single)

        valid_mask = ~(np.isnan(vr_pred_grid) | np.isnan(vr_true_grid))
        current_predictions = vr_pred_grid[valid_mask].flatten()
        current_targets = vr_true_grid[valid_mask].flatten()

        all_predictions.extend(current_predictions)
        all_targets.extend(current_targets)
        all_psnr.append(psnr)
        all_ssim.append(ssim)

    all_predict_time_ms = [t * 1000 for t in all_predict_time]
    avg_time = statistics.mean(all_predict_time_ms)
    max_time = max(all_predict_time_ms)
    min_time = min(all_predict_time_ms)
    print(f"平均预测时间: {avg_time:.1f}ms")
    print(f"最大预测时间: {max_time:.1f}ms，最小预测时间: {min_time:.1f}ms")

    epoch_start_plot = time.time()
    plot_psnr_ssim_histogram(all_psnr, all_ssim)

    epoch_time = time.time() - epoch_start
    print(f"time_cost: {epoch_time:.1f}s")

    # 绘制所有测试集的预测-真实值分布图
    plt.figure(figsize=(8, 6))
    counts, xedges, yedges, im = plt.hist2d(
        all_targets, all_predictions, bins=300, cmap="viridis", norm=LogNorm()
    )

    cbar = plt.colorbar(im)
    cbar.set_label("Point Count", fontsize=18)
    cbar.ax.tick_params(labelsize=18)  # 颜色条刻度字号

    min_val = min(min(all_targets), min(all_predictions))
    max_val = max(max(all_targets), max(all_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=1)

    r2 = r2_score(all_targets, all_predictions)
    print(f"所有测试集整体R²: {r2:.4f}")

    plt.xlabel("Real B_y values", fontsize=18)
    plt.ylabel("Predicted B_y Values", fontsize=18)
    plt.title(f"Prediction vs Real (R2 = {r2:.4f})", fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis="both", labelsize=20)

    plt.tight_layout()
    plt.savefig("all_test_sets_prediction_vs_real.png", dpi=600, bbox_inches="tight")
    plt.close()

    epoch_time_plot = time.time() - epoch_start_plot
    print(f"plot_all time_cost: {epoch_time_plot:.2f}s")
