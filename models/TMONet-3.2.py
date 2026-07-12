import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]  # 设置中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ========================================================================================
# 2026.3.29
# 大幅修改了梯度损失的计算方式，改为使用 PyTorch 的自动微分功能精确计算空间梯度
# 在计算梯度损失时正确处理了梯度的物理量纲和归一化问题
# 现在将首先按照工况划分训练集和验证集，然后再将二维数据展开为数据点
# ========================================================================================


class StructuralLoss(nn.Module):

    def __init__(
        self,
        alpha_start=0,
        alpha_end=1e-3,  # 最终梯度权重
        start_epoch=200,
        transition_epochs=800,
        preprocessor=None,
    ):
        super().__init__()
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.start_epoch = start_epoch
        self.transition_epochs = transition_epochs
        self.preprocessor = preprocessor
        self.mse = nn.MSELoss()
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """在训练循环中每个epoch开始时调用"""
        self.current_epoch = epoch

    def get_current_alpha(self):
        """计算当前epoch的alpha值"""
        if self.current_epoch < self.start_epoch:
            return self.alpha_start  # 前start_epoch个epoch完全使用MSE
        elif self.current_epoch < self.start_epoch + self.transition_epochs:
            # start_epoch-transition_epochs 个epoch线性衰减从1.0到alpha_end
            progress = (self.current_epoch - self.start_epoch) / self.transition_epochs
            return self.alpha_start - (self.alpha_start - self.alpha_end) * progress
        # if self.current_epoch > 200:
        #     return 0
        else:
            # transition_epochs之后维持alpha_end
            return self.alpha_end

    def forward(
        self, pred, target, model, q_params, coords, true_gradients, has_gradient
    ):
        current_alpha = self.get_current_alpha()
        mse_loss = self.mse(pred, target)

        # 只在有梯度信息的点上计算梯度损失  只有alpha不为0时才计算梯度损失
        if has_gradient.any() and current_alpha > 1e-8:
            grad_loss = self.calculate_gradient_loss(
                model, q_params, coords, true_gradients, has_gradient
            )
            # 添加调试信息
            # if torch.is_tensor(grad_loss) and grad_loss > 0:
            #     print(
            #         f"Epoch {self.current_epoch}: Alpha={current_alpha:.3f}, MSE: {mse_loss.item():.4f}, Grad: {grad_loss.item():.4f}"
            #     )
        else:
            grad_loss = torch.tensor(0.0).to(pred.device)

        combined_loss = mse_loss + current_alpha * (grad_loss - mse_loss)

        return combined_loss, mse_loss, grad_loss

    def calculate_gradient_loss(
        self, model, q_params, coords, true_gradients, has_gradient
    ):
        # 只选择有梯度信息的点
        grad_mask = has_gradient
        if not grad_mask.any():
            return torch.tensor(0.0).to(coords.device)

        q_params_grad = q_params[grad_mask]
        # 1. 启用坐标的梯度追踪 (AutoDiff 的核心)
        coords_grad = coords[grad_mask].clone().requires_grad_(True)
        true_gradients_grad = true_gradients[grad_mask]
        device = coords.device

        # 2. 前向传播，得到归一化后的预测值 [batch_size, 1]
        pred_scaled = model(q_params_grad, coords_grad)

        vr_scaler = self.preprocessor.vr_scaler
        vr_pos_max = torch.tensor(
            vr_scaler.pos_max_, dtype=torch.float32, device=device
        )
        vr_neg_min = torch.tensor(
            vr_scaler.neg_min_, dtype=torch.float32, device=device
        )
        vr_target_max = torch.tensor(
            vr_scaler.target_max_, dtype=torch.float32, device=device
        )
        vr_target_min = torch.tensor(
            vr_scaler.target_min_, dtype=torch.float32, device=device
        )

        pred_log = torch.zeros_like(pred_scaled)
        pos_mask = pred_scaled > 0
        neg_mask = pred_scaled < 0

        pred_log[pos_mask] = (pred_scaled[pos_mask] / vr_target_max) * vr_pos_max
        pred_log[neg_mask] = (
            pred_scaled[neg_mask] / torch.abs(vr_target_min)
        ) * torch.abs(vr_neg_min)

        pred_log = torch.clamp(pred_log, min=-50.0, max=50.0)
        pred_raw = (
            torch.sign(pred_log)
            * self.preprocessor.eps
            * torch.expm1(torch.abs(pred_log))
        )
        grad_outputs = torch.ones_like(pred_raw)
        gradients_scaled_coords = torch.autograd.grad(
            outputs=pred_raw,
            inputs=coords_grad,
            grad_outputs=grad_outputs,
            create_graph=True,  # 必须为True，保证梯度损失能继续反向传播给模型权重
            retain_graph=True,
            only_inputs=True,
        )[
            0
        ]  # 形状 [batch_size, 2]

        # 提取空间坐标的缩放系数 (StandardScaler 的 scale_)
        coord_scale = torch.tensor(
            self.preprocessor.coord_scaler.scale_, device=device, dtype=torch.float32
        )

        # 修正梯度量纲：根据链式法则得到针对物理坐标的导数
        grad_x_phys = gradients_scaled_coords[:, 0] / coord_scale[0]
        grad_z_phys = gradients_scaled_coords[:, 1] / coord_scale[1]

        # 计算物理空间的梯度模长 (加 1e-8 防止 sqrt(0) 处导数为 NaN)
        pred_grad_magnitude_raw = torch.sqrt(grad_x_phys**2 + grad_z_phys**2 + 1e-8)

        pred_grad_log = torch.sign(pred_grad_magnitude_raw) * torch.log1p(
            torch.abs(pred_grad_magnitude_raw) / self.preprocessor.grad_eps
        )

        grad_scaler = self.preprocessor.grad_scaler
        pos_max = torch.tensor(grad_scaler.pos_max_, dtype=torch.float32, device=device)
        neg_min = torch.tensor(grad_scaler.neg_min_, dtype=torch.float32, device=device)
        target_max = torch.tensor(
            grad_scaler.target_max_, dtype=torch.float32, device=device
        )
        target_min = torch.tensor(
            grad_scaler.target_min_, dtype=torch.float32, device=device
        )

        pred_grad_normalized = torch.zeros_like(pred_grad_log)
        pos_mask_grad = pred_grad_log > 0
        neg_mask_grad = pred_grad_log < 0

        pred_grad_normalized[pos_mask_grad] = (
            pred_grad_log[pos_mask_grad] / pos_max
        ) * target_max
        pred_grad_normalized[neg_mask_grad] = (
            pred_grad_log[neg_mask_grad] / torch.abs(neg_min)
        ) * torch.abs(target_min)

        # 最终计算 MSE Loss
        grad_loss = self.mse(pred_grad_normalized, true_gradients_grad)

        return grad_loss


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


class VRDataset(Dataset):
    def __init__(self, q_params, spatial_coords, vr_values, gradients=None):
        self.q_params = torch.FloatTensor(q_params)
        self.spatial_coords = torch.FloatTensor(spatial_coords)
        self.vr_values = torch.FloatTensor(vr_values.reshape(-1, 1))

        # 处理梯度：只有部分点有梯度信息，其余为NaN或0
        if gradients is not None:
            self.gradients = torch.FloatTensor(gradients).flatten()
            # 创建梯度掩码：只有梯度值>0的点才需要计算梯度损失
            self.has_gradient = (self.gradients > 0) & (~torch.isnan(self.gradients))
        else:
            self.gradients = None
            self.has_gradient = None

    def __len__(self):
        return len(self.q_params)

    def __getitem__(self, idx):
        if self.gradients is not None and self.has_gradient is not None:
            return (
                self.q_params[idx],
                self.spatial_coords[idx],
                self.vr_values[idx],
                self.gradients[idx],
                self.has_gradient[idx],
            )
        else:
            return self.q_params[idx], self.spatial_coords[idx], self.vr_values[idx]


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self):
        self.q_scaler = StandardScaler()
        self.coord_scaler = StandardScaler()
        self.vr_scaler = AsymmetricMinMaxScaler(feature_range=(-1, 1))
        self.grad_scaler = AsymmetricMinMaxScaler(feature_range=(-1, 1))

        self.eps = 1e-6  # 控制对数变换平滑度
        self.grad_eps = 1e-4

    def fit_transform(self, q_params, spatial_coords, vr_values, gradients=None):
        """拟合并转换数据"""
        # 1. 归一化q参数
        q_params_scaled = self.q_scaler.fit_transform(q_params)

        # 2. 归一化空间坐标
        coords_scaled = self.coord_scaler.fit_transform(spatial_coords)

        # 3. 对vr值进行对数变换 + 归一化
        vr_values = np.array(vr_values).flatten()
        vr_log = np.sign(vr_values) * np.log1p(np.abs(vr_values) / self.eps)
        vr_scaled = self.vr_scaler.fit_transform(vr_log.reshape(-1, 1)).flatten()

        # 4. 对梯度值进行相同的预处理（如果提供）
        if gradients is not None:
            gradients = np.array(gradients).flatten()
            grad_log = np.sign(gradients) * np.log1p(np.abs(gradients) / self.grad_eps)
            grad_scaled = self.grad_scaler.fit_transform(
                grad_log.reshape(-1, 1)
            ).flatten()
        else:
            grad_scaled = None

        return q_params_scaled, coords_scaled, vr_scaled, grad_scaled

    def transform(self, q_params, spatial_coords, vr_values, gradients=None):
        """转换新数据"""
        # 使用已经拟合的scaler进行转换
        q_params_scaled = self.q_scaler.transform(q_params)
        coords_scaled = self.coord_scaler.transform(spatial_coords)

        # 对vr值应用相同的变换
        vr_values = np.array(vr_values).flatten()
        vr_log = np.sign(vr_values) * np.log1p(np.abs(vr_values) / self.eps)
        vr_scaled = self.vr_scaler.transform(vr_log.reshape(-1, 1)).flatten()

        # 对梯度值应用相同的变换（如果提供）
        if gradients is not None:
            gradients = np.array(gradients).flatten()
            grad_log = np.sign(gradients) * np.log1p(np.abs(gradients) / self.grad_eps)
            grad_scaled = self.grad_scaler.transform(grad_log.reshape(-1, 1)).flatten()
        else:
            grad_scaled = None

        return q_params_scaled, coords_scaled, vr_scaled, grad_scaled

    def inverse_transform_gradients(self, grad_scaled):
        """将归一化后的梯度值反变换回原始尺度"""
        # 1. 反归一化
        grad_log = self.grad_scaler.inverse_transform(
            grad_scaled.reshape(-1, 1)
        ).flatten()

        # 2. 逆带符号对数变换
        grad_original = np.sign(grad_log) * self.grad_eps * (np.expm1(np.abs(grad_log)))

        return grad_original

    def inverse_transform_vr(self, vr_scaled):
        """将归一化后的 vr 值反变换回原始尺度"""
        # 1. 反归一化
        vr_log = self.vr_scaler.inverse_transform(vr_scaled.reshape(-1, 1)).flatten()

        # 2. 逆带符号对数变换
        vr_original = np.sign(vr_log) * self.eps * (np.expm1(np.abs(vr_log)))

        return vr_original


def load_data_from_df(q_params_df, vr_folder_path, desc="加载数据"):
    """
    根据传入的参数表，按工况加载并展开二维数据点
    """
    all_q_params = []
    all_spatial_coords = []
    all_vr_values = []
    all_gradients = []

    for i in tqdm(range(len(q_params_df)), desc=desc):
        # 读取对应工况的采样数据
        row = q_params_df.iloc[i]
        folder_name = row.values[-1]
        folder_name = str(folder_name).replace("/", "__").replace("\\", "__")
        vr_df = pd.read_csv(os.path.join(vr_folder_path, f"{folder_name}.csv"))

        X = vr_df["X"].values
        Z = vr_df["Z"].values
        vr = vr_df["Value"].values
        gradients = vr_df["Gradient"].values

        # 获取对应的q参数
        q_params = row.values[:5]

        # 为每个坐标点重复相同的q参数
        n_points = len(X)
        repeated_q_params = np.tile(q_params, (n_points, 1))

        # 组合空间坐标
        spatial_coords = np.column_stack([X, Z])

        # 添加到总数据集
        all_q_params.append(repeated_q_params)
        all_spatial_coords.append(spatial_coords)
        all_vr_values.append(vr)
        all_gradients.append(gradients)

    # 合并所有数据点
    q_params_combined = np.vstack(all_q_params)
    coords_combined = np.vstack(all_spatial_coords)
    vr_combined = np.hstack(all_vr_values)
    gradients_combined = np.hstack(all_gradients)

    return q_params_combined, coords_combined, vr_combined, gradients_combined


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=1000,
    patience=60,
    scheduler=None,  # 学习率调度器
    grad_clip=1.0,  # 梯度裁剪阈值
):
    """训练模型"""
    train_losses = []
    value_losses = []
    grad_losses = []
    val_losses = []
    learning_rates = []  # 记录学习率变化
    best_val_loss = float("inf")
    patience_counter = 0

    # 在训练循环开始前初始化计时列表
    epoch_times = []

    for epoch in range(num_epochs):

        epoch_start = time.time()
        criterion.set_epoch(epoch)

        # 训练阶段
        model.train()
        train_loss = 0.0
        v_loss = 0.0
        g_loss = 0.0
        # accumulation_steps = 4
        # for i,(q_params, coords, vr_true) in train_loader:
        for batch_data in train_loader:
            if len(batch_data) == 5:  # 有梯度信息和掩码
                q_params, coords, vr_true, gradients, has_gradient = batch_data
                q_params, coords, vr_true, gradients, has_gradient = (
                    q_params.to(device),
                    coords.to(device),
                    vr_true.to(device),
                    gradients.to(device),
                    has_gradient.to(device),
                )

                optimizer.zero_grad()
                vr_pred = model(q_params, coords)
                loss, value_loss, grad_loss = criterion(
                    vr_pred, vr_true, model, q_params, coords, gradients, has_gradient
                )
            else:  # 没有梯度信息
                q_params, coords, vr_true = batch_data
                q_params, coords, vr_true = (
                    q_params.to(device),
                    coords.to(device),
                    vr_true.to(device),
                )

                optimizer.zero_grad()
                vr_pred = model(q_params, coords)
                loss, value_loss, grad_loss = criterion(vr_pred, vr_true)

            # loss = loss / accumulation_steps
            loss.backward()
            # if (i + 1) % accumulation_steps == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            optimizer.step()
            train_loss += loss.item()
            v_loss += value_loss.item()
            g_loss += grad_loss.item()

            # 添加梯度裁剪
            # if grad_clip > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        # 更新学习率（如果有调度器）
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # 对于ReduceLROnPlateau，需要传入验证loss
                if epoch > 0:  # 至少有一个验证loss
                    scheduler.step(val_losses[-1])
            else:
                scheduler.step()

            # 记录当前学习率
            current_lr = optimizer.param_groups[0]["lr"]
            learning_rates.append(current_lr)

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        v_loss /= len(train_loader)
        value_losses.append(v_loss)
        g_loss /= len(train_loader)
        grad_losses.append(g_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 5:
                    q_params, coords, vr_true, gradients, has_gradient = batch_data
                    q_params, coords, vr_true, gradients, has_gradient = (
                        q_params.to(device),
                        coords.to(device),
                        vr_true.to(device),
                        gradients.to(device),
                        has_gradient.to(device),
                    )
                    vr_pred = model(q_params, coords)
                    val_loss += F.mse_loss(vr_pred, vr_true).item()
                else:
                    q_params, coords, vr_true = batch_data
                    q_params, coords, vr_true = (
                        q_params.to(device),
                        coords.to(device),
                        vr_true.to(device),
                    )
                    vr_pred = model(q_params, coords)
                    val_loss += F.mse_loss(vr_pred, vr_true).item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        # 显示当前步信息
        lr_info = ""
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            lr_info = f", LR: {current_lr:.1e}"

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        if epoch >= 100:
            print(
                f"Epoch [{epoch}] - Train Loss: {train_loss:.6f}, Value Loss: {v_loss:.6f}, Grad Loss: {g_loss:.6f}, Val Loss: {val_loss:.6f}{lr_info}, time_cost: {epoch_time:.2f}s"
            )
        else:
            print(
                f"Epoch [{epoch}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}{lr_info}, time_cost: {epoch_time:.2f}s"
            )

        # 早停和模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # 处理并行模型的状态字典
            model_state_dict = model.state_dict()
            if isinstance(model, nn.DataParallel):
                model_state_dict = model.module.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_state_dict,  # 使用处理后的状态字典
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "preprocessor": preprocessor,
                    "model_config": {
                        "branch_input_dim": 5,
                        "trunk_input_dim": 2,
                        "hidden_dim1": 200,
                        "hidden_dim2": 110,
                        "output_dim": 90,
                        "branch_depth": 4,
                        "trunk_depth": 4,
                    },
                },
                "TaskIII_TMOCE1.pth",
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return (
        train_losses,
        val_losses,
        learning_rates,
        epoch_times,
        value_losses,
        grad_losses,
    )


def plot_training(train_losses, val_losses, learning_rates, value_losses, grad_losses):
    """训练曲线图"""

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    fig, ax1 = plt.figure(figsize=(7, 4)), plt.gca()

    # 左侧y轴：损失值
    ax1.set_xlabel("Epoch", fontsize=18)
    ax1.set_ylabel("Loss", color="#1f77b4", fontsize=18)

    # 绘制损失曲线
    # line1 = ax1.plot(
    #     train_losses, label="Training Loss", color="#1f77b4", linewidth=2.5, alpha=0.9
    # )
    line2 = ax1.plot(
        val_losses, label="Validation Loss", color="#ff7f0e", linewidth=2.5, alpha=0.9
    )
    line4 = ax1.plot(
        value_losses, label="Value Loss", color="#33ff33", linewidth=2.5, alpha=0.9
    )
    line5 = ax1.plot(
        grad_losses,
        label="Gradient Loss",
        color="#6b16ff",
        linewidth=2.5,
        alpha=0.9,
    )
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(plt.LogFormatter(10, labelOnlyBase=False))
    ax1.grid(True, alpha=0.2)

    # 标记最佳验证损失点
    best_epoch = np.argmin(val_losses)
    best_val_loss = val_losses[best_epoch]
    ax1.scatter(
        best_epoch,
        best_val_loss,
        color="red",
        s=80,
        zorder=5,
        label=f"Best Val Loss: {best_val_loss:.4f}",
    )

    # 右侧y轴：学习率
    ax2 = ax1.twinx()
    ax2.set_ylabel("Learning Rate", color="#d62728", fontsize=12)

    # 绘制学习率曲线
    line3 = ax2.plot(
        learning_rates,
        label="Learning Rate",
        color="#d62728",
        linewidth=2,
        linestyle="--",
        alpha=0.8,
    )
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(plt.LogFormatter(10, labelOnlyBase=False))

    # 合并图例
    lines = line2 + line3 + line4 + line5
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=14)

    # 添加标题和信息
    # plt.title("TMONet Training Progress: Loss and Learning Rate", fontsize=18, pad=20)

    # 添加信息文本框
    # info_text = (
    #     f"Training Loss: {train_losses[-1]:.6f}\nValidation Loss: {val_losses[-1]:.6f}"
    # )
    # ax1.text(
    #     0.02,
    #     0.98,
    #     info_text,
    #     transform=ax1.transAxes,
    #     verticalalignment="top",
    #     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    #     fontsize=14,
    # )

    fig.tight_layout()
    plt.savefig("training_curves.png", dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")


def plot_epoch_times(epoch_times):
    """绘制epoch用时曲线"""
    del epoch_times[0]
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_times, color="#2E86AB", linewidth=2.5)
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.title("Epoch Training Time")
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    avg_time = sum(epoch_times) / len(epoch_times)
    total_time = sum(epoch_times)
    plt.text(
        0.02,
        0.98,
        f"Avg: {avg_time:.2f}s\nTotal: {total_time/60:.1f}min",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("epoch_times.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Average epoch time: {avg_time:.2f}s")
    print(f"Total training time: {total_time/60:.1f}min")


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    vr_folder_path = "selected_B9_633"
    q_params_file_path = "Double_Tearing_Train_Database_by_p0.csv"

    q_params_df_full = pd.read_csv(q_params_file_path)
    # 将独立的物理算例划分为训练算例和验证算例 (80% / 20%)
    train_cases_df, val_cases_df = train_test_split(
        q_params_df_full, test_size=0.2, random_state=42
    )
    print(
        f"训练集包含 {len(train_cases_df)} 个算例，验证集包含 {len(val_cases_df)} 个算例"
    )

    # 分别加载并展开训练集和验证集的数据点
    print("\nExtracting training points...")
    q_train_raw, coords_train_raw, vr_train_raw, grad_train_raw = load_data_from_df(
        train_cases_df, vr_folder_path, desc="加载训练集"
    )
    print("\nExtracting validation points...")
    q_val_raw, coords_val_raw, vr_val_raw, grad_val_raw = load_data_from_df(
        val_cases_df, vr_folder_path, desc="加载验证集"
    )

    preprocessor = DataPreprocessor()
    # 只在训练集上拟合（fit）和转换（transform）
    q_train, coords_train, vr_train, grad_train = preprocessor.fit_transform(
        q_train_raw, coords_train_raw, vr_train_raw, grad_train_raw
    )
    torch.save(preprocessor, "TMO_preprocessor_CE1.pth")
    # 验证集只能使用训练集的标准进行转换（transform）
    q_val, coords_val, vr_val, grad_val = preprocessor.transform(
        q_val_raw, coords_val_raw, vr_val_raw, grad_val_raw
    )

    # 创建数据集和数据加载器
    train_dataset = VRDataset(q_train, coords_train, vr_train, grad_train)
    val_dataset = VRDataset(q_val, coords_val, vr_val, grad_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=True,
        num_workers=16,  # 添加多进程数据加载
        pin_memory=True,  # 启用锁页内存，加速GPU传输
        persistent_workers=True,  # 保持worker进程，避免重复创建
        prefetch_factor=2,  # 预取更多batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # 初始化模型
    model = TMONet(
        branch_input_dim=5,  # r1, r2, s1, s2, p0
        trunk_input_dim=2,  # R, Z
        hidden_dim1=200,  # Brunch 隐藏层大小
        hidden_dim2=110,  # Trunk 隐藏层大小
        output_dim=90,  # 输出内积层大小
        branch_depth=4,  # 特别注意：这里的depth是实际隐藏层数+1
        trunk_depth=4,
        dropout_rate=0.08,
    ).to(device)

    # 启用数据并行 - 仅在多卡机器启用
    # if torch.cuda.device_count() > 1:
    #     print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
    #     model = nn.DataParallel(model)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=5e-4, betas=(0.9, 0.999)
    )

    # 添加学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,  # 学习率减小因子
        patience=10,  # epoch 忍耐值
        min_lr=1e-6,  # 最小学习率
    )

    # 训练模型
    print("Starting training...")
    train_losses, val_losses, learning_rates, epoch_times, value_losses, grad_losses = (
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=StructuralLoss(
                alpha_start=0,
                alpha_end=0.15,  # 最终梯度权重
                start_epoch=100,  # 100个epoch后开始引入梯度约束
                transition_epochs=300,  # 线性过渡区间长
                preprocessor=preprocessor,
            ),  # 加入梯度相似性损失
            optimizer=optimizer,
            device=device,
            num_epochs=1000,
            patience=60,
            scheduler=scheduler,
            grad_clip=1.0,  # 梯度裁剪阈值
        )
    )

    # 保存训练数据
    csv_path = "losses_info_CE5.csv"
    df = pd.DataFrame(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "learning_rates": learning_rates,
            "epoch_times": epoch_times,
            "value_losses": value_losses,
            "grad_losses": grad_losses,
        }
    )
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # df_loaded = pd.read_csv(csv_path, encoding="utf-8")
    # train_losses = df_loaded["train_losses"].tolist()
    # val_losses = df_loaded["val_losses"].tolist()
    # learning_rates = df_loaded["learning_rates"].tolist()
    # value_losses = df_loaded["value_losses"].tolist()
    # grad_losses = df_loaded["grad_losses"].tolist()

    # 绘制训练曲线
    plot_training(train_losses, val_losses, learning_rates, value_losses, grad_losses)
    # plot_epoch_times(epoch_times)
    print("Training completed!")
