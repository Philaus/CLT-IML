import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import time
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.ticker as ticker

# ================================================================================
# 2026.5.6 将饱和动能特征替换为内外磁岛宽度
# 2026.5.20 优化程序数据加载和cpu-gpu交互
# 2026.5.31 修复了模型隐藏层和dropout定义的逻辑
# 2026.6.2 将饱和动能加回来
# ================================================================================


def preprocess_data(dataset):

    output_columns = ["Wt_Inner_max", "Wt_Outer_max", "gamma", "Ekmax"]
    normal_columns = ["r1", "r2", "s1", "s2", "p0"]

    # 提取特征
    normal_features = dataset[normal_columns].values
    # 标准化特征
    normal_scaler = StandardScaler()
    features_scaled = normal_scaler.fit_transform(normal_features)

    # 处理输出标签
    labels = dataset[output_columns].values

    # labels 进行标准化
    label_scaler = StandardScaler()
    labels_scaled = label_scaler.fit_transform(labels)

    # 返回处理后的数据和所有标准化器
    return features_scaled, labels_scaled, normal_scaler, label_scaler


class AbaloneDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float)
        self.y = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rates=None):
        super(MLP, self).__init__()

        if dropout_rates is None:
            dropout_rates = [0.2] * len(hidden_sizes)
        elif isinstance(dropout_rates, (int, float)):
            # 如果传入的是单个数（如 0.08），自动扩展为 [0.08, 0.08, ...]
            dropout_rates = [dropout_rates] * len(hidden_sizes)
        elif len(dropout_rates) == 1:
            # 如果传入的是单元素列表（如 [0.08]），自动复制对齐，例如 [0.08, 0.08]
            dropout_rates = dropout_rates * len(hidden_sizes)

        # 创建多层网络
        layers = []
        prev_size = input_size

        for i, (hidden_size, dropout_rate) in enumerate(
            zip(hidden_sizes, dropout_rates)
        ):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())

            # 只在不是最后一个隐藏层的情况下添加 Dropout
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # 输出层（无激活函数）
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0, path="model.pth"):
        """
        Args:
            patience (int): 验证集损失不再改善后等待的epoch数
            verbose (bool): 是否打印早停信息
            delta (float): 被认为有改善的最小变化量
            path (str): 保存最佳模型的路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        """在验证损失减小时保存模型"""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path}"
            )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            },
            self.path,
        )
        self.val_loss_min = val_loss


def train_epoch(net, device, train_iter, loss_fn, optimizer):
    # 将模型设置为训练模式
    net.train()
    metrics = Accumulator(2)
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        # 计算梯度并更新参数
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metrics.add(loss.item() * len(y), len(y))
    train_loss = metrics[0] / metrics[1]
    return train_loss


@torch.no_grad()
def eval_model(net, device, test_iter, loss_fn):
    net.eval()
    metrics = Accumulator(2)
    for X, y in test_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        metrics.add(loss.item() * len(y), len(y))
    test_loss = metrics[0] / metrics[1]
    return test_loss


def split_data_by_fold(data, fold):
    """
    根据折数划分数据为训练、验证、测试集 (8:2:1比例)
    Args:
        data: 完整数据集
        fold: 当前折数 (0到n_folds-1)
        n_folds: 总折数
    Returns:
        train_data, val_data, test_data
    """
    n_samples = len(data)

    # 计算各集合大小 (按8:2:1比例)
    test_size = n_samples // 11
    val_size = 2 * test_size

    # 确保随机但可重复的划分
    np.random.seed(42)
    indices = np.random.permutation(n_samples)

    # 计算测试集的起始位置
    test_start = fold * test_size
    test_end = test_start + test_size

    # 计算验证集的起始位置（在测试集之后）
    val_start = test_end
    val_end = val_start + val_size

    # 提取测试集
    test_indices = indices[test_start:test_end]
    test_data = data.iloc[test_indices].reset_index(drop=True)

    # 提取验证集
    val_indices = indices[val_start:val_end]
    val_data = data.iloc[val_indices].reset_index(drop=True)

    # 剩余为训练集
    train_indices = np.concatenate([indices[:test_start], indices[val_end:]])
    train_data = data.iloc[train_indices].reset_index(drop=True)

    return train_data, val_data, test_data


def train_and_test_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    loss_fn,
    optimizer,
    scheduler,
    epochs,
    patience,
    label_scaler,  # 只需要label_scaler进行逆变换
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 初始化早停
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_ls, val_ls = [], []
    train_start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, device, train_loader, loss_fn, optimizer)
        val_loss = eval_model(model, device, val_loader, loss_fn)

        train_ls.append(train_loss)
        val_ls.append(val_loss)

        print(
            f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        # 早停检查
        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    # 加载最佳模型
    checkpoint = torch.load("model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Loaded best model from epoch {checkpoint['epoch']} with val loss: {checkpoint['val_loss']:.6f}"
    )

    # 进行测试集预测
    model.eval()

    # 改用 PyTorch 列表在显存/内存中暂存 Tensor
    preds_accum = []
    trues_accum = []

    # 在开始前同步一下 GPU，确保计时准确
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    predict_start_time = time.time()

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_pred = model(X_test)

            # 极其重要：只收集原始张量，绝不在循环内调用 .cpu().numpy()
            preds_accum.append(y_pred.cpu())
            trues_accum.append(y_test)

    # 在循环外部，一次性完成大矩阵的合并、转 numpy 和逆变换（向量化操作，速度极快）
    test_predictions = []
    test_true_values = []

    if preds_accum:
        # 一次性拼接并转为 numpy
        y_pred_all = torch.cat(preds_accum, dim=0).numpy()
        y_test_all = torch.cat(trues_accum, dim=0).numpy()

        # 大矩阵整体逆变换
        y_pred_orig = label_scaler.inverse_transform(y_pred_all)
        y_test_orig = label_scaler.inverse_transform(y_test_all)

        y_pred_orig[:, 3] = 10 ** y_pred_orig[:, 3]
        y_test_orig[:, 3] = 10 ** y_test_orig[:, 3]

        test_predictions.extend(y_pred_orig)
        test_true_values.extend(y_test_orig)

    # 预测结束，同步并记录时间
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    predict_end_time = time.time()
    total_predict_time = (predict_end_time - predict_start_time) * 1000

    # 计算平均单点预测时间
    n_test_points = len(test_predictions)
    time_per_point = total_predict_time / n_test_points if n_test_points > 0 else 0

    print(f"训练用时: {train_time:.2f} 秒")
    print(f"总预测用时: {total_predict_time:.4f} 毫秒")
    print(f"测试集点数: {n_test_points}")
    print(f"单点预测平均用时: {time_per_point:.4f} 毫秒")

    # 测试预测完成后，清理DataLoader
    try:
        del train_loader, val_loader, test_loader
    except:
        pass
    # 强制垃圾回收
    import gc

    gc.collect()
    # 如果是CUDA，清理缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        train_ls,
        val_ls,
        model,
        test_predictions,
        test_true_values,
        train_time,
        time_per_point,
    )


def evaluate_test_set(predictions, true_values):
    """
    计算测试集的评估指标
    """
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    # 计算相对误差
    relative_errors = np.abs((predictions - true_values) / true_values) * 100
    avg_errors = np.mean(relative_errors, axis=0)

    # 计算R²分数
    r2_wt_inner = r2_score(true_values[:, 0], predictions[:, 0])
    r2_wt_outer = r2_score(true_values[:, 1], predictions[:, 1])
    r2_gamma = r2_score(true_values[:, 2], predictions[:, 2])
    r2_ekmax = r2_score(true_values[:, 3], predictions[:, 3])

    return {
        "predictions": predictions,
        "true_values": true_values,
        "avg_errors": avg_errors,
        "r2_wt_inner": r2_wt_inner,
        "r2_wt_outer": r2_wt_outer,
        "r2_gamma": r2_gamma,
        "r2_ekmax": r2_ekmax,
    }


def plot_all_folds_results(fold_results):
    """
    绘制所有折的测试结果
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    # 合并所有折的结果
    all_pred = np.concatenate([r["predictions"] for r in fold_results])
    all_true = np.concatenate([r["true_values"] for r in fold_results])

    # 计算整体指标
    overall_errors = np.abs((all_pred - all_true) / all_true) * 100

    overall_errors_final = overall_errors
    all_true_final = all_true
    all_pred_final = all_pred

    overall_avg_errors_final = np.mean(overall_errors_final, axis=0)
    overall_r2_wt_inner_final = r2_score(all_true_final[:, 0], all_pred_final[:, 0])
    overall_r2_wt_outer_final = r2_score(all_true_final[:, 1], all_pred_final[:, 1])
    overall_r2_gamma_final = r2_score(all_true_final[:, 2], all_pred_final[:, 2])
    overall_r2_ekmax_final = r2_score(all_true_final[:, 3], all_pred_final[:, 3])

    print(f"\n过滤后指标:")
    print(
        f"Wt_Inner - R²: {overall_r2_wt_inner_final:.4f}, 平均相对误差: {overall_avg_errors_final[0]:.2f}%"
    )
    print(
        f"Wt_Outer - R²: {overall_r2_wt_outer_final:.4f}, 平均相对误差: {overall_avg_errors_final[1]:.2f}%"
    )
    print(
        f"gamma - R²: {overall_r2_gamma_final:.4f}, 平均相对误差: {overall_avg_errors_final[2]:.2f}%"
    )
    print(
        f"Ekmax - R²: {overall_r2_ekmax_final:.4f}, 平均相对误差: {overall_avg_errors_final[3]:.2f}%"
    )

    # 创建图形
    plt.figure(figsize=(14, 12))

    def format_sci(x, _):
        return "{:.1e}".format(x).replace("e-0", "e-").replace("e+0", "e")

    # 1. Wt_Inner散点图
    plt.subplot(2, 2, 1)
    min_val = min(all_true_final[:, 0].min(), all_pred_final[:, 0].min())
    max_val = max(all_true_final[:, 0].max(), all_pred_final[:, 0].max())
    plt.scatter(
        all_true_final[:, 0],
        all_pred_final[:, 0],
        alpha=0.7,
    )
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, alpha=0.8)
    plt.text(
        0.02,
        0.98,
        f"R²: {overall_r2_wt_inner_final:.4f}\nAvg Error: {overall_avg_errors_final[0]:.2f}%",
        transform=plt.gca().transAxes,
        fontsize=16,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8, edgecolor="gray"),
    )
    plt.xlabel("Real $W_{Bi}^{\mathrm{max}}$", fontsize=16)
    plt.ylabel("$W_{Bi}^{\mathrm{max}}$ Prediction", fontsize=16)
    plt.title("$W_{Bi}^{\mathrm{max}}$: All Folds Test Results", fontsize=18)
    plt.tick_params(axis="both", labelsize=16)
    plt.grid(True, alpha=0.3, linestyle="--")

    # 2. Wt_Outer散点图
    plt.subplot(2, 2, 2)
    min_val = min(all_true_final[:, 1].min(), all_pred_final[:, 1].min())
    max_val = max(all_true_final[:, 1].max(), all_pred_final[:, 1].max())
    plt.scatter(
        all_true_final[:, 1],
        all_pred_final[:, 1],
        alpha=0.7,
    )
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, alpha=0.8)
    plt.text(
        0.02,
        0.98,
        f"R²: {overall_r2_wt_outer_final:.4f}\nAvg Error: {overall_avg_errors_final[1]:.2f}%",
        transform=plt.gca().transAxes,
        fontsize=16,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8, edgecolor="gray"),
    )
    plt.xlabel("Real $W_{Bo}^{\mathrm{max}}$", fontsize=16)
    plt.ylabel("$W_{Bo}^{\mathrm{max}}$ Prediction", fontsize=16)
    plt.title("$W_{Bo}^{\mathrm{max}}$: All Folds Test Results", fontsize=18)
    plt.tick_params(axis="both", labelsize=16)
    plt.grid(True, alpha=0.3, linestyle="--")

    # 3. gamma散点图
    plt.subplot(2, 2, 3)
    min_val = min(all_true_final[:, 2].min(), all_pred_final[:, 2].min())
    max_val = max(all_true_final[:, 2].max(), all_pred_final[:, 2].max())
    plt.scatter(all_true_final[:, 2], all_pred_final[:, 2], alpha=0.7)
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, alpha=0.8)
    plt.text(
        0.02,
        0.98,
        f"R²: {overall_r2_gamma_final:.4f}\nAvg Error: {overall_avg_errors_final[2]:.2f}%",
        transform=plt.gca().transAxes,
        fontsize=16,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8, edgecolor="gray"),
    )
    plt.xlabel("Real $\gamma$", fontsize=16)
    plt.ylabel("$\gamma$ Prediction", fontsize=16)
    plt.title("$\gamma$: All Folds Test Results", fontsize=18)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_sci))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_sci))
    ax.xaxis.offsetText.set_fontsize(16)
    ax.yaxis.offsetText.set_fontsize(16)
    plt.tick_params(axis="both", labelsize=16)
    plt.grid(True, alpha=0.3, linestyle="--")

    # 4. Ekmax散点图
    plt.subplot(2, 2, 4)
    min_val = min(all_true_final[:, 3].min(), all_pred_final[:, 3].min())
    max_val = max(all_true_final[:, 3].max(), all_pred_final[:, 3].max())
    plt.scatter(all_true_final[:, 3], all_pred_final[:, 3], alpha=0.7)
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, alpha=0.8)
    plt.text(
        0.02,
        0.98,
        f"R²: {overall_r2_ekmax_final:.4f}\nAvg Error: {overall_avg_errors_final[3]:.2f}%",
        transform=plt.gca().transAxes,
        fontsize=16,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8, edgecolor="gray"),
    )
    plt.xlabel("Real $E_k^{\mathrm{max}}$", fontsize=16)
    plt.ylabel("$E_k^{\mathrm{max}}$ Prediction", fontsize=16)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("$E_k^{\mathrm{max}}$: All Folds Test Results", fontsize=18)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_sci))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_sci))
    ax.xaxis.offsetText.set_fontsize(16)
    ax.yaxis.offsetText.set_fontsize(16)
    plt.tick_params(axis="both", labelsize=16)
    plt.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    # plt.savefig("cross_validation_results—5.2.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("cross_validation_results-5.2.png", dpi=600, bbox_inches="tight")
    # plt.savefig(
    #     "cross_validation_results—5.eps", format="eps", dpi=600, bbox_inches="tight"
    # )
    plt.show()


if __name__ == "__main__":
    # ==================== 数据准备 ====================
    full_dataset = pd.read_csv("Double_Tearing_Train_Database_Bisland_Ek.csv")
    full_dataset["Ekmax"] = np.log10(full_dataset["Ekmax"])

    # ==================== 10折交叉验证 ====================
    n_folds = 10
    all_test_predictions = []
    all_test_true_values = []
    fold_results = []

    # 添加时间记录列表
    all_train_times = []
    all_time_per_point = []

    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)

    best_val_loss = float("inf")
    best_model_state = None
    best_fold = 0

    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_folds}")
        print(f"{'='*60}")

        # 划分数据
        train_data, val_data, test_data = split_data_by_fold(full_dataset, fold)

        # 预处理（每折单独进行）
        X_train, y_train, normal_scaler, label_scaler = preprocess_data(train_data)

        # 对验证集使用相同的标准化器
        normal_features_val = val_data[["r1", "r2", "s1", "s2", "p0"]].values
        X_val = normal_scaler.transform(normal_features_val)
        labels_val = val_data[["Wt_Inner_max", "Wt_Outer_max", "gamma", "Ekmax"]].values
        y_val = label_scaler.transform(labels_val)

        # 对测试集使用相同的标准化器
        normal_features_test = test_data[["r1", "r2", "s1", "s2", "p0"]].values
        X_test = normal_scaler.transform(normal_features_test)
        labels_test = test_data[["Wt_Inner_max", "Wt_Outer_max", "gamma", "Ekmax"]].values
        y_test = label_scaler.transform(labels_test)

        # 创建数据加载器
        train_dataset = AbaloneDataset(X_train, y_train)
        val_dataset = AbaloneDataset(X_val, y_val)
        test_dataset = AbaloneDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,         
            shuffle=True,          
            num_workers=0,      
            pin_memory=True,
            drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=64,        
            shuffle=False,         
            num_workers=0,         
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,       
            shuffle=False,         
            num_workers=0,     
            pin_memory=True
        )

        input_size = X_train.shape[1]
        output_size = 4
        hidden_sizes = [128,64]
        dropout_rates = [0.1]

        model = MLP(input_size, hidden_sizes, output_size, dropout_rates)

        # 设置优化器和调度器
        lr = 5e-3
        weight_decay = 1e-5
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        # 训练和测试
        loss_fn = nn.MSELoss()
        (
            train_losses,
            val_losses,
            model,
            test_predictions,
            test_true_values,
            train_time,
            time_per_point,
        ) = train_and_test_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            loss_fn,
            optimizer,
            scheduler,
            epochs=400,
            patience=40,
            label_scaler=label_scaler,
        )
        # 记录时间信息
        all_train_times.append(train_time)
        all_time_per_point.append(time_per_point)

        # 获取当前折的最小验证损失
        current_min_val_loss = min(val_losses)
        # 检查是否是最佳模型
        if current_min_val_loss < best_val_loss:
            best_val_loss = current_min_val_loss
            best_model_state = model.state_dict().copy()
            best_fold = fold + 1
            print(f"🎯 新的最佳模型！Fold {best_fold}, Val Loss: {best_val_loss:.6f}")

        # 评估当前折
        fold_result = evaluate_test_set(test_predictions, test_true_values)
        fold_result["fold"] = fold + 1
        fold_result["train_time"] = train_time  # 添加训练用时
        fold_result["time_per_point"] = time_per_point  # 添加单点预测用时
        fold_results.append(fold_result)

        print(
            f"Fold {fold+1} 完成: "
            f"Wt_Inner R²={fold_result['r2_wt_inner']:.4f}, "
            f"Wt_Outer R²={fold_result['r2_wt_outer']:.4f}, "
            f"gamma R²={fold_result['r2_gamma']:.4f}, "
            f"Ekmax R²={fold_result['r2_ekmax']:.4f}, "
            f"Avg Errors={fold_result['avg_errors']}, "
        )

    # 保存最佳模型
    if best_model_state is not None:
        torch.save(best_model_state, "TaskI_10folds.pth")
    print(f"\n{'='*60}")
    print("交叉验证完成")
    print(f"{'='*60}")

    print(
        f"  平均训练用时: {np.mean(all_train_times):.2f} ± {np.std(all_train_times):.2f} 秒"
    )
    print(
        f"  平均单点预测用时: {np.mean(all_time_per_point):.4f} ± {np.std(all_time_per_point):.4f} 毫秒"
    )

    # 绘制所有折的结果
    plot_all_folds_results(fold_results)

    # # ======================================================================
    # # 独立测试模块
    # # ======================================================================
    # print(f"\n{'='*60}")
    # print("开始在独立测试集上进行预测与测试")
    # print(f"{'='*60}")

    # # 1. 读取并处理独立测试集数据
    # indep_dataset = pd.read_csv("Double_Tearing_Train_Database_Bisland-16.csv")
    # indep_dataset["Ekmax"] = np.log10(indep_dataset["Ekmax"])

    # # 2. 获取预处理 Scaler (利用原有的 full_dataset 重新拟合，确保特征和标签的缩放尺度一致)
    # _, _, global_normal_scaler, global_label_scaler = preprocess_data(full_dataset)

    # # 3. 提取特征并使用相同的标准化器进行缩放
    # normal_features_indep = indep_dataset[["r1", "r2", "s1", "s2", "p0"]].values
    # X_indep = global_normal_scaler.transform(normal_features_indep)

    # labels_indep = indep_dataset[
    #     ["Wt_Inner_max", "Wt_Outer_max", "gamma", "Ekmax"]
    # ].values
    # y_indep = global_label_scaler.transform(labels_indep)

    # # 4. 构建独立测试集的 DataLoader
    # indep_test_dataset = AbaloneDataset(X_indep, y_indep)
    # indep_test_loader = DataLoader(
    #     indep_test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True
    # )

    # # 5. 初始化模型结构并加载已保存的最佳模型权重
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_size = X_indep.shape[1]
    # output_size = 4
    # hidden_sizes = [128, 64]
    # dropout_rates = [0.1]

    # loaded_model = MLP(input_size, hidden_sizes, output_size, dropout_rates)
    # loaded_model.load_state_dict(torch.load("TaskI_10folds.pth"))
    # loaded_model.to(device)
    # loaded_model.eval()

    # # 6. 执行前向预测
    # preds_accum = []
    # trues_accum = []

    # with torch.no_grad():
    #     for X_batch, y_batch in indep_test_loader:
    #         X_batch = X_batch.to(device)
    #         y_pred = loaded_model(X_batch)
    #         preds_accum.append(y_pred.cpu())
    #         trues_accum.append(y_batch)

    # # 7. 拼接矩阵并进行逆变换还原真实物理尺度
    # y_pred_all = torch.cat(preds_accum, dim=0).numpy()
    # y_test_all = torch.cat(trues_accum, dim=0).numpy()

    # y_pred_orig = global_label_scaler.inverse_transform(y_pred_all)
    # y_test_orig = global_label_scaler.inverse_transform(y_test_all)

    # # 还原 log10(Ekmax)
    # y_pred_orig[:, 3] = 10 ** y_pred_orig[:, 3]
    # y_test_orig[:, 3] = 10 ** y_test_orig[:, 3]

    # # 8. 复用原有的评估函数计算 R² 和平均相对误差
    # indep_result = evaluate_test_set(y_pred_orig, y_test_orig)

    # print("\n独立测试集 (Bisland-16) 评估指标:")
    # print(
    #     f"Wt_Inner - R²: {indep_result['r2_wt_inner']:.4f}, 平均相对误差: {indep_result['avg_errors'][0]:.2f}%"
    # )
    # print(
    #     f"Wt_Outer - R²: {indep_result['r2_wt_outer']:.4f}, 平均相对误差: {indep_result['avg_errors'][1]:.2f}%"
    # )
    # print(
    #     f"gamma    - R²: {indep_result['r2_gamma']:.4f}, 平均相对误差: {indep_result['avg_errors'][2]:.2f}%"
    # )
    # print(
    #     f"Ekmax    - R²: {indep_result['r2_ekmax']:.4f}, 平均相对误差: {indep_result['avg_errors'][3]:.2f}%"
    # )

    # # 9. 复用原有绘图函数绘制散点图
    # # 注意：这里将结果打包成 list `[indep_result]` 传入，以适配原函数的数据结构
    # print("\n正在生成独立测试集预测效果散点图...")
    # plot_all_folds_results([indep_result])
