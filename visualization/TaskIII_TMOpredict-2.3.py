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
]  # Configure fonts that support Chinese text.
plt.rcParams["axes.unicode_minus"] = False  # Render minus signs correctly.

# ========================================================================================
# 2025-12-11: B33 regression model; normalized data to 3e-5 and added p0.
# 2026-05-14: Added detailed timing statistics.
# 2026-05-14: Load the model/preprocessor once and infer each case in one GPU batch.
# 2026-06-03: Improved figure and typography styling.
# ========================================================================================


class AsymmetricMinMaxScaler:
    """
    Simplified asymmetric normalizer for 1D data.
    """

    def __init__(self, feature_range=(-1, 1)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        X = np.array(X).flatten()
        self.target_min_, self.target_max_ = self.feature_range

        # Positive-value statistics.
        positive_data = X[X > 0]
        self.pos_max_ = np.max(positive_data) if len(positive_data) > 0 else 1.0

        # Negative-value statistics.
        negative_data = X[X < 0]
        self.neg_min_ = np.min(negative_data) if len(negative_data) > 0 else -1.0

        return self

    def transform(self, X):
        X = np.array(X).flatten()
        result = np.zeros_like(X)

        # Positive values.
        positive_mask = X > 0
        result[positive_mask] = (X[positive_mask] / self.pos_max_) * self.target_max_

        # Negative values.
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

        # Inverse-transform positive values.
        positive_mask = X > 0
        result[positive_mask] = (X[positive_mask] / self.target_max_) * self.pos_max_

        # Inverse-transform negative values.
        negative_mask = X < 0
        result[negative_mask] = (X[negative_mask] / abs(self.target_min_)) * abs(
            self.neg_min_
        )

        return result.reshape(-1, 1) if len(result.shape) == 1 else result


class SinActivation(nn.Module):
    """Custom sine activation."""

    def forward(self, x):
        return torch.sin(x)


class DataPreprocessor:
    """Data preprocessor."""

    def __init__(self):
        self.q_scaler = StandardScaler()
        self.coord_scaler = StandardScaler()
        self.vr_scaler = AsymmetricMinMaxScaler(feature_range=(-1, 1))
        self.eps = 1e-6  # Control log-transform smoothness.

    def fit_transform(self, q_params, spatial_coords, vr_values):
        """Fit preprocessing and transform data."""
        # 1. Normalize q parameters.
        q_params_scaled = self.q_scaler.fit_transform(q_params)

        # 2. Normalize spatial coordinates (X, Z).
        coords_scaled = self.coord_scaler.fit_transform(spatial_coords)

        # 3. Log-transform and normalize vr values.
        vr_transformed = self._transform_vr(vr_values, fit=True)

        return q_params_scaled, coords_scaled, vr_transformed

    def transform(self, q_params, spatial_coords, vr_values):
        """Transform new data."""
        q_params_scaled = self.q_scaler.transform(q_params)
        coords_scaled = self.coord_scaler.transform(spatial_coords)
        vr_transformed = self._transform_vr(vr_values, fit=False)

        return q_params_scaled, coords_scaled, vr_transformed

    def _transform_vr(self, vr_values, fit=False):
        """Apply a signed log transform and normalize vr to [-1, 1]."""
        vr_values = np.array(vr_values).flatten()
        vr_log = np.sign(vr_values) * np.log1p(np.abs(vr_values) / self.eps)

        # Normalize to [-1, 1].
        if fit:
            vr_scaled = self.vr_scaler.fit_transform(vr_log.reshape(-1, 1)).flatten()
        else:
            vr_scaled = self.vr_scaler.transform(vr_log.reshape(-1, 1)).flatten()

        return vr_scaled

    def _transform_true_vr(self, vr_values, fit=False):
        """Normalize vr directly to [-1, 1]."""
        vr_values = np.array(vr_values).flatten()

        # Normalize to [-1, 1].
        if fit:
            vr_scaled = self.vr_scaler.fit_transform(vr_values.reshape(-1, 1)).flatten()
        else:
            vr_scaled = self.vr_scaler.transform(vr_values.reshape(-1, 1)).flatten()

        return vr_scaled

    def inverse_transform_vr(self, vr_scaled):
        """Inverse-transform normalized vr values to their original scale."""
        # 1. Invert normalization.
        vr_log = self.vr_scaler.inverse_transform(vr_scaled.reshape(-1, 1)).flatten()

        # 2. Invert the signed log transform.
        vr_original = np.sign(vr_log) * self.eps * (np.expm1(np.abs(vr_log)))

        return vr_original

    def get_preprocessing_info(self):
        """Return preprocessing metadata."""
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
        hidden_dim1=200,  # Branch hidden dimension.
        hidden_dim2=110,  # Trunk hidden dimension.
        output_dim=90,  # Inner-product output dimension.
        branch_depth=4,
        trunk_depth=4,
        dropout_rate=0.08,
    ):
        super(TMONet, self).__init__()

        # Branch network: process q-function parameters.
        self.branch_net = self._build_mlp(
            input_dim=branch_input_dim,
            hidden_dim=hidden_dim1,
            output_dim=output_dim,
            depth=branch_depth,
            dropout_rate=dropout_rate,
        )

        # Trunk network: process spatial coordinates.
        self.trunk_net = self._build_mlp(
            input_dim=trunk_input_dim,
            hidden_dim=hidden_dim2,
            output_dim=output_dim,
            depth=trunk_depth,
            dropout_rate=dropout_rate,
        )

        # Bias term.
        self.bias = nn.Parameter(torch.zeros(1))

    def _build_mlp(self, input_dim, hidden_dim, output_dim, depth, dropout_rate=0.15):
        layers = []
        current_dim = input_dim

        # 1. Build hidden layers only when depth >= 2.
        if depth >= 2:
            for i in range(depth - 1):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(SinActivation())

                # Add dropout except after the final hidden layer.
                # The final hidden-layer index is depth - 2.
                if i < depth - 2:
                    layers.append(nn.Dropout(dropout_rate))

                current_dim = hidden_dim

        # 2. Map final features directly to the physical-output scale.
        layers.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, q_params, spatial_coords):
        """
        Args:
            q_params: [batch_size, 5] - (r1, r2, s1, s2, p0)
            spatial_coords: [batch_size, 2] - Spatial coordinates (R, Z).
        Returns:
            vr_pred: [batch_size, 1] - Predicted vr values.
        """
        branch_output = self.branch_net(q_params)  # [batch_size, output_dim]
        trunk_output = self.trunk_net(spatial_coords)  # [batch_size, output_dim]

        vr_pred = (
            torch.sum(branch_output * trunk_output, dim=1, keepdim=True) + self.bias
        )

        return vr_pred


def predict_vr(model, preprocessor, q_params, spatial_coords, device):
    """
    Predict vr values with the trained model.
    Args:
        model_path: Path to the model file.
        preprocessor_path: Path to the preprocessor file.
        q_params: [n_samples, 5] - q-function parameters.
        spatial_coords: [n_samples, 2] - Spatial coordinates (R, Z).
    Returns:
        vr_predictions: [n_samples] - Predicted vr values.
    """
    model.eval()

    # Preprocess data on the CPU.
    q_params_2d = q_params.reshape(-1, 5) if q_params.ndim == 1 else q_params
    q_scaled = preprocessor.q_scaler.transform(q_params_2d)
    coords_scaled = preprocessor.coord_scaler.transform(spatial_coords)

    # Convert base-sized data to tensors and move them to the GPU.
    q_tensor = torch.FloatTensor(q_scaled).to(device)
    coords_tensor = torch.FloatTensor(coords_scaled).to(device)

    # Expand dimensions on the GPU.
    n_coords = coords_tensor.shape[0]
    q_tensor_expanded = q_tensor.expand(n_coords, -1)

    # Predict.
    with torch.no_grad():
        vr_pred_scaled = model(q_tensor_expanded, coords_tensor)

    # Move to the CPU and invert normalization.
    vr_pred = preprocessor.inverse_transform_vr(vr_pred_scaled.cpu().numpy())

    return vr_pred


def batch_predict(
    model,  # Pre-instantiated model.
    preprocessor,  # Pre-instantiated preprocessor.
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

    # Predict all points in one call.
    batch_pred = predict_vr(model, preprocessor, q_test, coords_flat, device)
    vr_pred_flat = batch_pred.flatten()

    epoch_time = time.time() - epoch_start
    print(f"single case time_cost: {epoch_time:.4f}s")

    # Build the grid.
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
        width_ratios=[1.0, 0.0 ,1.0, 0.1, 1.0],  # Control column-width ratios.
        hspace=0.23,
        wspace=0.18,  # Primarily controls spacing between panels 1 and 2.
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 4])
    ax4 = fig.add_subplot(gs[1, :])

    # Panel 1: predicted-value distribution.
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

    # Panel 2: target-value distribution.
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

    # Panel 3: error distribution.
    # Use a diverging color map for positive and negative errors.
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

    # Calculate MSE, PSNR, and SSIM for the current field.
    # Mask the numerous NaN values in the data matrix.
    valid_mask = ~(np.isnan(vr_true_grid) | np.isnan(vr_pred_grid))
    # Extract valid values.
    truth_valid = vr_true_grid[valid_mask]
    pred_valid = vr_pred_grid[valid_mask]
    # Calculate the actual data range.
    data_range_actual = max(
        truth_valid.max() - truth_valid.min(), pred_valid.max() - pred_valid.min()
    )

    mse = np.nanmean(error_grid**2)
    psnr = 20 * np.log10(data_range_actual) - 10 * np.log10(mse)
    # Calculate SSIM after reshaping the 1D array to 2D.
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
    z_zero_idx = np.argmin(np.abs(y - 0))  # Find the index closest to Z=0.
    r_coords = x  # R coordinates.
    pred_z0 = vr_pred_grid[z_zero_idx, :]  # Predictions at Z=0.
    true_z0 = vr_true_grid[z_zero_idx, :]  # Targets at Z=0.

    # # Panel 4: plot the R-vr curve at Z=0.
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
    Plot histograms of the PSNR and SSIM distributions.
    """
    if save_path is None:
        save_path = f"psnr_ssim.png"

    # Create the figure and subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the PSNR histogram.
    n_psnr, bins_psnr, patches_psnr = ax1.hist(
        psnr_list, bins=8, color="skyblue", edgecolor="navy", alpha=0.7
    )
    ax1.set_xlabel("PSNR (dB)", fontsize=18)
    ax1.set_ylabel("Counts", fontsize=18)
    ax1.set_title("PSNR Distribution", fontsize=18)
    ax1.set_xlim(30, 45)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="both", labelsize=18)

    # Add summary statistics to the PSNR histogram.
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

    # Plot the SSIM histogram.
    ax2.hist(ssim_list, bins=8, color="lightcoral", edgecolor="darkred", alpha=0.7)
    ax2.set_xlabel("SSIM", fontsize=18)
    ax2.set_ylabel("Counts", fontsize=18)
    ax2.set_title("SSIM Distribution", fontsize=18)
    ax2.set_xlim(0.8, 1)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="both", labelsize=18)

    # Add summary statistics to the SSIM histogram.
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

    # Plot predicted-versus-target values for all test sets.
    plt.figure(figsize=(8, 6))
    counts, xedges, yedges, im = plt.hist2d(
        all_targets, all_predictions, bins=300, cmap="viridis", norm=LogNorm()
    )

    cbar = plt.colorbar(im)
    cbar.set_label("Point Count", fontsize=18)
    cbar.ax.tick_params(labelsize=18)  # Color-bar tick-label size.

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
