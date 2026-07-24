import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import matplotlib.ticker as ticker

plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold" 

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)


def test_zero_shot_performance():
    """
    Evaluate the original model on the full target domain without fine-tuning.
    Fit normalization on only three random target samples to prevent leakage.
    """
    print(
        "\n" + "=" * 20 + " 开始测试：零微调直接预测（Zero-Shot, 3点基准） " + "=" * 20
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_all_raw, y_all_raw = load_full_data()
    # --- Fit the normalizer on only three samples. ---
    np.random.seed(42)
    sample_size = 3
    sample_indices = np.random.choice(len(X_all_raw), size=sample_size, replace=False)
    X_sample_for_scaler = X_all_raw[sample_indices]
    y_sample_for_scaler = y_all_raw[sample_indices]

    # 2. Fit and transform.
    # Fit on three samples, then transform the full target domain.
    normal_scaler = StandardScaler()
    normal_scaler.fit(X_sample_for_scaler)
    X_all = normal_scaler.transform(X_all_raw)
    label_scaler = StandardScaler()
    label_scaler.fit(y_sample_for_scaler)

    # 3. Instantiate the network and load source-domain weights.
    model = MLP(
        input_size=5, hidden_sizes=[128, 64], output_size=4, dropout_rates=[0.1]
    ).to(device)
    model.load_state_dict(torch.load("TaskI_10folds.pth", map_location=device))

    # 4. Evaluate without backpropagation or weight updates.
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_all, dtype=torch.float).to(device)
        y_pred = model(X_tensor).cpu().numpy()
        y_pred_orig = label_scaler.inverse_transform(y_pred)

        # Invert normalization to recover physical units.
        y_pred_orig = label_scaler.inverse_transform(y_pred)
        y_pred_orig[:, 3] = 10 ** y_pred_orig[:, 3]
        y_all_raw[:, 3] = 10 ** y_all_raw[:, 3]

    # 5. Calculate metrics without fine-tuning.
    r2_inner = r2_score(y_all_raw[:, 0], y_pred_orig[:, 0])
    r2_outer = r2_score(y_all_raw[:, 1], y_pred_orig[:, 1])
    r2_gamma = r2_score(y_all_raw[:, 2], y_pred_orig[:, 2])
    r2_ekmax = r2_score(y_all_raw[:, 3], y_pred_orig[:, 3])

    errors = np.abs((y_pred_orig - y_all_raw) / y_all_raw) * 100
    avg_errors = np.mean(errors, axis=0)

    print("\n" + "="*50)
    print("【零微调直接预测结果 (Zero-Shot Regression)】")
    print(f"Wt_Inner - R2: {r2_inner:.4f}, Avg Error: {avg_errors[0]:.2f}%")
    print(f"Wt_Outer - R2: {r2_outer:.4f}, Avg Error: {avg_errors[1]:.2f}%")
    print(f"gamma    - R2: {r2_gamma:.4f}, Avg Error: {avg_errors[2]:.2f}%")
    print(f"Ekmax    - R2: {r2_ekmax:.4f}, Avg Error: {avg_errors[3]:.2f}%")
    print("="*50)

    # Plot the no-fine-tuning comparison.
    var_names = [
        "$W_{ti}\mathrm{max}$",
        "$W_{to}\mathrm{max}$",
        "$\gamma$",
        "$E_k\mathrm{max}$",
    ]
    # r2_scores = [r2_inner, r2_outer, r2_gamma, r2_ekmax]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    for i in range(4):
        true_val = y_all_raw[:, i]
        pred_val = y_pred_orig[:, i]

        axes[i].scatter(
            true_val,
            pred_val,
            s=90,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.8,
            zorder=3,
        )
        min_val = min(true_val.min(), pred_val.min())
        max_val = max(true_val.max(), pred_val.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, alpha=0.8)

        axes[i].text(
            0.02,
            0.98,
            f"Avg Error: {avg_errors[i]:.2f}%",
            transform=axes[i].transAxes,
            fontsize=16,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8, edgecolor="gray"),
        )

        axes[i].set_xlabel(f"Real {var_names[i]}", fontsize=16)
        axes[i].set_ylabel(f"Predicted {var_names[i]}", fontsize=16)
        # axes[i].set_title(f"{var_names[i]}", fontsize=18)
        axes[i].tick_params(axis="both", labelsize=16)
        axes[i].grid(True, alpha=0.3, linestyle="--")

        if var_names[i] in ["$\gamma$", "$E_k\mathrm{max}$"]:
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits(
                (-2, 2)
            )  # Use scientific notation outside 10^-2 to 10^2.
            axes[i].xaxis.set_major_formatter(formatter)
            axes[i].yaxis.set_major_formatter(formatter)

            # Adjust the scientific-notation exponent label size.
            axes[i].xaxis.offsetText.set_fontsize(14)
            axes[i].yaxis.offsetText.set_fontsize(14)

    plt.tight_layout()
    plt.savefig('zero_shot_extrapolation_plot.png', dpi=600, bbox_inches="tight")
    print("零微调散点图已成功保存为 'zero_shot_extrapolation_plot.png'")
    plt.show()


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rates=None):
        super(MLP, self).__init__()

        if dropout_rates is None:
            dropout_rates = [0.2] * len(hidden_sizes)
        elif isinstance(dropout_rates, (int, float)):
            # Expand a scalar to match all layers.
            dropout_rates = [dropout_rates] * len(hidden_sizes)
        elif len(dropout_rates) == 1:
            # Replicate a one-element list to match all layers.
            dropout_rates = dropout_rates * len(hidden_sizes)

        # Build the multilayer network.
        layers = []
        prev_size = input_size

        for i, (hidden_size, dropout_rate) in enumerate(
            zip(hidden_sizes, dropout_rates)
        ):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())

            # Add dropout except after the final hidden layer.
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer without activation.
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DTMDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float)
        self.y = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EarlyStopping:
    def __init__(self, patience=25, delta=0):
        """
        Args:
            patience (int): Maximum epochs without loss improvement.
            delta (float): Minimum significant improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None  # Hold the best weights in memory.

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            # Capture the current best weights.
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0


def load_full_data():
    full_data = pd.read_csv("Transferlearning_Database_Bisland_Database_Bisland.csv")

    full_data["Ekmax"] = np.log10(full_data["Ekmax"])
    normal_columns = ["r1", "r2", "s1", "s2", "p0"]
    output_columns = ["Wt_Inner_max", "Wt_Outer_max", "gamma", "Ekmax"]

    X_all = full_data[normal_columns].values
    y_all = full_data[output_columns].values
    return X_all, y_all


def different_testdata():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the complete target-domain dataset.
    X_all, y_all = load_full_data()
    total_samples = len(X_all)
    print(f"Total Target Domain samples available: {total_samples}")

    summary_log_file = "transfer_learning_summary_report.csv"

    # Number of repeated random splits to average.
    num_repeats = 8

    # 2. Iterate over test-set sizes.
    for test_size in range(1, 22):
        train_size = total_samples - test_size
        print(f"\n" + "="*25 + f" 测试集大小: {test_size} (微调训练集: {train_size}) " + "="*25)

        # Accumulate metrics from the 10 repeated experiments.
        repeat_r2_inner, repeat_r2_outer, repeat_r2_gamma, repeat_r2_Ekmax = [], [], [], []
        repeat_errors = []

        for repeat in range(num_repeats):
            # Use a different train_test_split seed for each split.
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X_all, y_all, test_size=test_size, random_state=42 + repeat, shuffle=True
            )

            # 3. Fit scalers only on training data to prevent leakage.
            normal_scaler = StandardScaler()
            X_train = normal_scaler.fit_transform(X_train_raw)
            X_test = normal_scaler.transform(X_test_raw)

            label_scaler = StandardScaler()
            y_train = label_scaler.fit_transform(y_train_raw)
            y_test = label_scaler.transform(y_test_raw)

            train_loader = DataLoader(DTMDataset(X_train, y_train), batch_size=min(8, len(X_train)), shuffle=True)

            # 4. Recreate the model and load pretrained weights.
            model = MLP(input_size=5, hidden_sizes=[128, 64], output_size=4, dropout_rates=[0.1]).to(device)
            model.load_state_dict(torch.load("TaskI_10folds.pth", map_location=device))

            # 5. Freezing strategy.
            for param in model.parameters():
                param.requires_grad = False
            for param in model.network[-1].parameters():
                param.requires_grad = True

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=1e-3, 
                weight_decay=1e-3
            )
            loss_fn = nn.MSELoss()

            # early_stopping = EarlyStopping(patience=20, delta=0)

            # 6. Fine-tuning.
            epochs = 300
            model.train()
            # total_train_loss = 0.0
            for epoch in range(1, epochs + 1):
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    y_hat = model(X_batch)
                    loss = loss_fn(y_hat, y_batch)
                    loss.backward()
                    optimizer.step()

                #     total_train_loss += loss.item() * len(y_batch)

                # avg_train_loss = total_train_loss / len(X_train)

                # # --- Apply early stopping. ---
                # early_stopping(avg_train_loss, model)
                # if early_stopping.early_stop:
                #     break

            # 7. Evaluate on the current random test set.
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(device)
                y_test_pred = model(X_test_tensor).cpu().numpy()

                y_pred_orig = label_scaler.inverse_transform(y_test_pred)
                y_true_orig = label_scaler.inverse_transform(y_test)

                # # ================= Diagnostic metrics in log space =================
                # y_pred_log_space = label_scaler.inverse_transform(y_test_pred)
                # y_true_log_space = label_scaler.inverse_transform(y_test)
                # log_space_errors = np.mean(
                #     np.abs((y_pred_log_space - y_true_log_space) / y_true_log_space)
                #     * 100,
                #     axis=0,
                # )
                # print(
                #     f"    [Log-space diagnostic] Ekmax relative error: {log_space_errors[3]:.2f}%"
                # )
                # # ================================================================

                # Calculate metrics for the current split.
                # Guard against undefined r2_score when test_size=1.
                if test_size > 1:
                    repeat_r2_inner.append(r2_score(y_true_orig[:, 0], y_pred_orig[:, 0]))
                    repeat_r2_outer.append(r2_score(y_true_orig[:, 1], y_pred_orig[:, 1]))
                    repeat_r2_gamma.append(r2_score(y_true_orig[:, 2], y_pred_orig[:, 2]))
                    repeat_r2_Ekmax.append(r2_score(y_true_orig[:, 3], y_pred_orig[:, 3]))
                else:
                    repeat_r2_inner.append(np.nan)
                    repeat_r2_outer.append(np.nan)
                    repeat_r2_gamma.append(np.nan)
                    repeat_r2_Ekmax.append(np.nan)

                # Calculate mean relative error.
                fold_errors = np.mean(np.abs((y_pred_orig - y_true_orig) / y_true_orig) * 100, axis=0)
                repeat_errors.append(fold_errors)

                y_pred_orig[:, 3] = 10 ** y_pred_orig[:, 3]
                y_true_orig[:, 3] = 10 ** y_true_orig[:, 3]

        # 8. Average performance across 10 experiments.
        mean_errors = np.mean(repeat_errors, axis=0)
        mean_r2_inner = np.nanmean(repeat_r2_inner) if test_size > 1 else np.nan
        mean_r2_outer = np.nanmean(repeat_r2_outer) if test_size > 1 else np.nan
        mean_r2_gamma = np.nanmean(repeat_r2_gamma) if test_size > 1 else np.nan
        mean_r2_Ekmax = np.nanmean(repeat_r2_Ekmax) if test_size > 1 else np.nan

        print(
            f"--> [结果汇总] 测试点数: {test_size} | 平均误差: Inner: {mean_errors[0]:.2f}%, Outer: {mean_errors[1]:.2f}%, Gamma: {mean_errors[2]:.2f}%, Ekmax: {mean_errors[3]:.2f}%"
        )

        # 9. Write the summary CSV with separate train/test sample counts.
        experiment_summary = {
            "K_Folds": f"Random_{num_repeats}Mix",  # Mark repeated shuffled experiments.
            "Transfer_data_amount": train_size,  # Fine-tuning sample count.
            "Global_R2_Wt_Inner": (
                round(mean_r2_inner, 4) if not np.isnan(mean_r2_inner) else "N/A"
            ),
            "Global_R2_Wt_Outer": (
                round(mean_r2_outer, 4) if not np.isnan(mean_r2_outer) else "N/A"
            ),
            "Global_R2_gamma": (
                round(mean_r2_gamma, 4) if not np.isnan(mean_r2_gamma) else "N/A"
            ),
            "Global_R2_Ekmax": (
                round(mean_r2_Ekmax, 4) if not np.isnan(mean_r2_Ekmax) else "N/A"
            ),
            "Global_Error_Wt_Inner(%)": round(mean_errors[0], 2),
            "Global_Error_Wt_Outer(%)": round(mean_errors[1], 2),
            "Global_Error_gamma(%)": round(mean_errors[2], 2),
            "Global_Error_Ekmax(%)": round(mean_errors[3], 2),
        }

        df_summary = pd.DataFrame([experiment_summary])
        df_summary.to_csv(
            summary_log_file, 
            mode='a', 
            header=not os.path.exists(summary_log_file), 
            index=False, 
            encoding='utf-8'
        )

    print(f"\n数据效率消融实验完成！总成绩单已追加保存至 {summary_log_file}")


def different_Kfold():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load target-domain data.
    X_all, y_all = load_full_data()

    # 2. Initialize cross-validation.
    kf = KFold(n_splits=8, shuffle=True, random_state=42)

    # Collect fold predictions and targets for the final plot.
    all_y_true_orig = []
    all_y_pred_orig = []

    # Start cross-validation.
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        print(f"\n" + "="*20 + f" FOLD {fold + 1} / {kf.n_splits} " + "="*20)

        # Split the current fold strictly by index.
        X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]
        y_train_raw, y_test_raw = y_all[train_idx], y_all[test_idx]

        # Fit scalers only on this fold's training set.
        normal_scaler = StandardScaler()
        X_train = normal_scaler.fit_transform(X_train_raw)
        X_test = normal_scaler.transform(X_test_raw)

        label_scaler = StandardScaler()
        y_train = label_scaler.fit_transform(y_train_raw)
        y_test = label_scaler.transform(y_test_raw)

        # Build this fold's DataLoader.
        train_loader = DataLoader(DTMDataset(X_train, y_train), batch_size=32, shuffle=True)

        # 3. Reload the model for each fold to prevent weight leakage.

        model = MLP(input_size=5, hidden_sizes=[128, 64], output_size=4, dropout_rates=[0.1]).to(device)

        try:
            model.load_state_dict(torch.load("TaskI_10folds.pth", map_location=device))
        except FileNotFoundError:
            print("ERROR: Pre-trained weights not found. Please check the path.")
            exit()

        # 4. Freeze all weights except the final layer.
        for param in model.parameters():
            param.requires_grad = False
        for param in model.network[-1].parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=1e-3, 
            weight_decay=1e-3
        )
        loss_fn = nn.MSELoss()

        # early_stopping = EarlyStopping(patience=20, delta=0)

        # 5. Fine-tune this fold.
        epochs = 300
        model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_hat = model(X_batch)
                loss = loss_fn(y_hat, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_batch)

            avg_loss = total_loss / len(X_train)
            if epoch % 10 == 0 or epoch == epochs:
                print(f"Epoch {epoch:3d}/{epochs} - Train Loss: {avg_loss:.6f}")

            # # --- Apply early stopping. ---
            # early_stopping(avg_loss, model)
            # if early_stopping.early_stop:
            #     break

        # 6. Evaluate this fold's independent test set.
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(device)
            y_test_pred = model(X_test_tensor).cpu().numpy()

            # Use this fold's label_scaler to recover physical units.
            y_pred_orig = label_scaler.inverse_transform(y_test_pred)
            y_true_orig = label_scaler.inverse_transform(y_test)

            # Store physical targets and predictions globally.
            all_y_true_orig.append(y_true_orig)
            all_y_pred_orig.append(y_pred_orig)
            y_pred_orig[:, 3] = 10 ** y_pred_orig[:, 3]
            y_true_orig[:, 3] = 10 ** y_true_orig[:, 3]

    # ====================================================================
    # 7. Aggregate K-fold results and evaluate global metrics.
    # ====================================================================
    all_y_true_orig = np.vstack(all_y_true_orig)
    all_y_pred_orig = np.vstack(all_y_pred_orig)

    # Calculate aggregate cross-validation R².
    r2_wt_inner = r2_score(all_y_true_orig[:, 0], all_y_pred_orig[:, 0])
    r2_wt_outer = r2_score(all_y_true_orig[:, 1], all_y_pred_orig[:, 1])
    r2_gamma = r2_score(all_y_true_orig[:, 2], all_y_pred_orig[:, 2])
    r2_Ekmax = r2_score(all_y_true_orig[:, 3], all_y_pred_orig[:, 3])

    # Calculate aggregate mean relative error.
    errors = np.abs((all_y_pred_orig - all_y_true_orig) / all_y_true_orig) * 100
    avg_errors = np.mean(errors, axis=0)

    print("\n" + "="*50)
    print(f"{kf.n_splits}-Fold Cross Validation Final Evaluation Results:")
    print(f"Wt_Inner - Global CV R2: {r2_wt_inner:.4f}, Avg Error: {avg_errors[0]:.2f}%")
    print(f"Wt_Outer - Global CV R2: {r2_wt_outer:.4f}, Avg Error: {avg_errors[1]:.2f}%")
    print(f"gamma    - Global CV R2: {r2_gamma:.4f}, Avg Error: {avg_errors[2]:.2f}%")
    print(f"Ekmax    - Global CV R2: {r2_Ekmax:.4f}, Avg Error: {avg_errors[3]:.2f}%")
    print("="*50)

    summary_log_file = "transfer_learning_summary_report.csv"
    # Get K from kf.n_splits.
    experiment_summary = {
        "K_Folds": kf.n_splits,  # Record K for this run.
        "Total_Samples": len(X_all),  # Record total target-domain samples.
        "Global_R2_Wt_Inner": round(r2_wt_inner, 4),
        "Global_R2_Wt_Outer": round(r2_wt_outer, 4),
        "Global_R2_gamma": round(r2_gamma, 4),
        "Global_R2_Ekmax": round(r2_Ekmax, 4),
        "Global_Error_Wt_Inner(%)": round(avg_errors[0], 2),
        "Global_Error_Wt_Outer(%)": round(avg_errors[1], 2),
        "Global_Error_gamma(%)": round(avg_errors[2], 2),
        "Global_Error_Ekmax(%)": round(avg_errors[3], 2),
    }
    # Convert to a DataFrame and append.
    df_summary = pd.DataFrame([experiment_summary])
    df_summary.to_csv(
        summary_log_file, 
        mode='a', 
        header=not os.path.exists(summary_log_file),  # Write a header only for a new file.
        index=False, 
        encoding='utf-8'
    )
    print(f"\n[数据记录] 本次 {kf.n_splits} 折总体评估结果已成功追加至 {summary_log_file}")

    # 8. Plot physical-value comparisons across all folds.
    var_names = [
        "$W_{Bi}^{\mathrm{max}}$",
        "$W_{Bo}^{\mathrm{max}}$",
        "$\gamma$",
        "$E_k^{\mathrm{max}}$",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i in range(4):
        true_val = all_y_true_orig[:, i]
        pred_val = all_y_pred_orig[:, i]
        current_r2 = r2_score(true_val, pred_val)

        # 1. Plot the test-point distribution.
        axes[i].scatter(true_val, pred_val, alpha=0.7)

        # 2. Plot the perfect-prediction reference y=x.
        min_val = min(true_val.min(), pred_val.min())
        max_val = max(true_val.max(), pred_val.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, alpha=0.8)

        # 3. Add the annotation box.
        axes[i].text(
            0.02,
            0.98,
            f"R2: {current_r2:.4f}\nAvg Error: {avg_errors[i]:.2f}%",
            transform=axes[i].transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8, edgecolor="gray"),
        )

        axes[i].set_xlabel(f"Real {var_names[i]}", fontsize=16)
        axes[i].set_ylabel(f"Predicted {var_names[i]}", fontsize=16)
        axes[i].set_title(f"{var_names[i]}: All Folds Test Results", fontsize=18)
        axes[i].tick_params(axis="both", labelsize=16)
        axes[i].grid(True, alpha=0.3, linestyle="--")

        if i == 2:
            def format_sci(x, _):
                return "{:.1e}".format(x).replace("e-0", "e-").replace("e+0", "e")

            axes[i].xaxis.set_major_formatter(ticker.FuncFormatter(format_sci))
            axes[i].yaxis.set_major_formatter(ticker.FuncFormatter(format_sci))
            axes[i].xaxis.offsetText.set_fontsize(16)
            axes[i].yaxis.offsetText.set_fontsize(16)

    plt.tight_layout()
    # Build the output name from the fold count.
    save_name = f'transfer_scatter_plot_fold{kf.n_splits}_CV.png'
    plt.savefig(save_name, dpi=600, bbox_inches="tight")
    print(f"全局交叉验证散点图已成功保存为 '{save_name}'")
    plt.show()

if __name__ == "__main__":

    different_testdata()
    # different_Kfold()
    # test_zero_shot_performance()
