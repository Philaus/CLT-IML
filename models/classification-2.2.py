import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.metrics import r2_score
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import time
from matplotlib.lines import Line2D
import traceback
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio

# plt.rcParams["font.sans-serif"] = [
#     "SimHei",
#     "Microsoft YaHei",
#     "DejaVu Sans",
# ]  # 设置中文字体
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

# ====================================================================
# 2026.1.24 删除了不再使用的模块
# 添加扫描s2参数绘制多个r1-r2图功能，复用了原有遍历预测和绘图代码
# 增加了直接读取模型pkl文件功能，无需每次都重新训练
# 2026.1.28 添加扫描三维空间绘制分界面功能
# ====================================================================


class PlasmaDisruptionRegressor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ["r1", "r2", "s1", "s2"]
        self.crash_threshold = 0.085

        # 自定义评分函数
        self.custom_scorer = make_scorer(self._weighted_accuracy)

    def _weighted_accuracy(self, y_true, y_pred):

        y_pred_binary = (y_pred > self.crash_threshold).astype(int)
        y_true_binary = (y_true > self.crash_threshold).astype(int)

        accuracy = np.mean(y_pred_binary == y_true_binary)

        return accuracy

    def load_data(self, file_path):
        """加载数据"""
        data = pd.read_csv(file_path)
        X = data[self.feature_names].values
        y = data["crash_percentage"].values

        total_samples = len(y)
        crash_above = np.sum(y > self.crash_threshold)
        print(f"数据统计:")
        print(f"  总样本数: {total_samples}")
        print(f"  中心崩塌样本数: {crash_above} ({crash_above/total_samples:.1%})")
        return X, y

    def train_model(self, X, y, use_grid_search=True):
        """训练回归模型"""
        print("\n开始训练回归模型...")
        start_time = time.time()

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        if use_grid_search:
            print("正在进行超参数搜索...")
            fixed_params = {
                "max_features": 0.8,
                "min_samples_leaf": 2,
                "min_samples_split": 3,
                "max_samples": 1.0,
                "random_state": 42,
                "bootstrap": True,
                "oob_score": True,
                "n_jobs": -1,
            }

            search_spaces = {
                "max_depth": Integer(8, 15),  # 树最大深度
                "n_estimators": Integer(150, 250),  # 森林中树的数量
                # "max_features": Real(0.7, 0.95),  # 寻找最佳分割时考虑的特征比例
                # "min_samples_leaf": Integer(1, 3),  # 叶节点所需的最小样本数
                # "min_samples_split": Integer(2, 4),  # 分裂内部节点所需最小样本数
                # "max_samples": Real(0.85, 1.0),  # 样本采样比例
            }

            base_model = RandomForestRegressor(**fixed_params)
            opt = BayesSearchCV(
                base_model,
                search_spaces,
                n_iter=20,  # 迭代次数
                cv=10,  # 10 折交叉验证
                random_state=42,
                n_jobs=-1,  # 使用全部 CPU 核心
                scoring="r2",
            )
            opt.fit(X_scaled, y)
            self.model = opt.best_estimator_

            y_pred_full = self.model.predict(X_scaled)
            self._plot_results(y, y_pred_full, "-search")

            end_time = time.time()
            training_time = end_time - start_time
            print(f"训练用时: {training_time:.2f} 秒")  # 这是10次训练的用时
            self.last_training_time = training_time

            print(f"最佳参数: {opt.best_params_}")
            print(f"最佳交叉验证加权准确率: {opt.best_score_:.4f}")

        else:
            print("使用给定参数训练10个模型用于集成预测...")
            self.models = []  # 存储10个模型
            kf = KFold(n_splits=10, shuffle=True, random_state=42)

            cv_scores = []
            y_pred_ensemble = np.zeros_like(y)
            counts = np.zeros_like(y)  # 记录每个样本被预测的次数

            n_estimators = 237

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # 创建并训练模型
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=11,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    random_state=42 + fold,  # 不同的随机种子
                )
                model.fit(X_train, y_train)

                # 验证集评估
                y_val_pred = model.predict(X_val)
                y_val_pred_binary = (y_val_pred > self.crash_threshold).astype(int)
                y_val_binary = (y_val > self.crash_threshold).astype(int)

                val_accuracy = np.mean(y_val_pred_binary == y_val_binary)
                cv_scores.append(val_accuracy)

                y_pred_ensemble[val_idx] += y_val_pred
                counts[val_idx] += 1

                self.models.append(model)

                print(f"  折 {fold+1} 验证准确率: {val_accuracy:.3f}")

            r2 = r2_score(y, y_pred_ensemble)
            print(f"\n回归模型性能:")
            print(f"R² 分数: {r2:.4f}")
            y_pred_ensemble /= counts
            self._plot_results(y, y_pred_ensemble, n_estimators)
            print(
                f"10折交叉验证平均准确率: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})"
            )

            # 为了保持接口一致，设置一个主模型（使用第一个模型）
            self.model = self.models[0]

    def predict(self, parameters):
        """
        预测新样本 - 使用10折集成预测
        """
        if self.model is None:
            raise ValueError("没有找到模型文件")

        start_time = time.time()
        input_data = np.array(parameters).reshape(1, -1)
        input_scaled = self.scaler.transform(input_data)

        # 如果有多模型，使用集成预测，否则使用单模型
        if hasattr(self, "models") and self.models:
            # 10折集成预测
            predictions = []
            for model in self.models:
                pred = model.predict(input_scaled)[0]
                predictions.append(pred)

            crash_percentage = np.mean(predictions)
            # 计算预测的标准差作为置信度指标
            pred_std = np.std(predictions)
        else:
            # 单模型预测
            crash_percentage = self.model.predict(input_scaled)[0]
            pred_std = 0.0  # 单模型没有标准差

        # 根据阈值判断是否发生明显崩塌
        if crash_percentage > self.crash_threshold:
            confidence_level = "高" if pred_std < 0.05 else "中"
            result = {
                "crash_percentage": crash_percentage,
                "has_crash": True,
                "confidence": f"明显崩塌 ({crash_percentage:.1%})",
                "prediction_std": pred_std,
                "confidence_level": confidence_level,
            }
        else:
            confidence_level = "高" if pred_std < 0.02 else "中"
            result = {
                "crash_percentage": 0.0,
                "has_crash": False,
                "confidence": "无显著崩塌",
                "prediction_std": pred_std,
                "confidence_level": confidence_level,
            }

        end_time = time.time()
        prediction_time = end_time - start_time
        # 添加到返回结果中
        result["prediction_time"] = f"{prediction_time:.6f}秒"
        # 如果记录了训练时间，也添加到结果中
        if hasattr(self, "last_training_time"):
            result["training_time"] = f"{self.last_training_time:.2f}秒"

        return result

    def save_model(self, filepath):
        """保存训练好的模型"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "crash_threshold": self.crash_threshold,
        }

        # 如果有多模型，也保存它们
        if hasattr(self, "models") and self.models:
            model_data["models"] = self.models

        joblib.dump(model_data, filepath)

    def _plot_results(self, y_test, y_pred, n_estimators):
        """绘制回归结果"""
        fig, axes = plt.subplots(1, 1, figsize=(5, 4.5))
        r2 = r2_score(y_test, y_pred)
        mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        y_pred_binary = (y_pred > self.crash_threshold).astype(int)
        y_test_binary = (y_test > self.crash_threshold).astype(int)
        accuracy = np.mean(y_pred_binary == y_test_binary) * 100

        # 预测 vs 实际散点图
        axes.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5, zorder=3)
        max_val = max(y_test.max(), y_pred.max())
        axes.plot([0, max_val], [0, max_val], "r--", lw=1.5, zorder=2)

        box_text = (
            r"$R^2$"
            + f"       : {r2:.4f}\n"
            + r"MRE     "
            + f" : {mre:.2f}%\n"
            + r"Accuracy"
            + f" : {accuracy:.2f}%\n"
            + r"with threshold ="
            + f" {self.crash_threshold*100:.1f}%"
        )

        axes.text(
            0.05, 0.95, box_text,
            transform=axes.transAxes,
            fontsize=12,
            fontfamily="monospace",
            ha="left",
            va="top",
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.8,
                edgecolor="#E5E5E5",
                linewidth=0.8
            )
        )

        axes.set_xlabel("Real Crash Percentage", fontsize=14)
        axes.set_ylabel("Crash Percentage Prediction", fontsize=14)
        # axes.set_title(
        #     f"Predicted vs Real\nR2 = {r2:.4f}, Accuracy = {accuracy:.4f}", fontsize=20
        # )
        axes.set_xlim(0, max_val * 1.05)
        axes.set_ylim(0, max_val * 1.05)

        axes.grid(True, alpha=0.2, linestyle="--", zorder=0)
        axes.tick_params(axis="both", labelsize=13, direction="in")
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            f"regression_results{n_estimators}.png", dpi=600, bbox_inches="tight"
        )
        plt.show()
        plt.close()

    def evaluate_thresholds(self, X, y, thresholds=None):
        """评估不同阈值下的性能并绘制ROC曲线"""
        if thresholds is None:
            thresholds = np.arange(0.07, 0.14, 0.005)

        X_scaled = self.scaler.transform(X)
        y_pred_proba = self.model.predict(X_scaled)

        # 存储结果
        results = []
        confusion_matrices = []

        for threshold in thresholds:
            y_pred_binary = (y_pred_proba > threshold).astype(int)
            y_true_binary = (y > threshold).astype(int)

            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

            # 计算各项指标
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 灵敏度
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            results.append(
                {
                    "threshold": threshold,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "specificity": specificity,
                    "f1": f1,
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                }
            )
            confusion_matrices.append((tn, fp, fn, tp))

        # 绘制性能指标随阈值变化
        self._plot_metrics_vs_threshold(results)

        # 绘制最佳阈值处的混淆矩阵
        best_result = max(results, key=lambda x: x["f1"])
        self._plot_confusion_matrix(
            best_result, confusion_matrices[results.index(best_result)]
        )

        return results

    def scan_all_parameter_pairs(
        self, r_step=0.01, s_step=0.05, model_path="TaskII_PCcls.pkl"
    ):
        """
        扫描四个参数的所有两两组合，绘制二维分布图

        参数:
        - r_step: r参数的网格步长
        - s_step: s参数的网格步长
        """

        if self.model is None:
            print(f"当前未加载模型，尝试从 {model_path} 读取...")
            try:
                # 10折集成预测，结果取平均
                model_data = joblib.load(model_path)
                self.model = model_data["model"]
                self.scaler = model_data["scaler"]
                # 恢复其他重要属性
                if "crash_threshold" in model_data:
                    self.crash_threshold = model_data["crash_threshold"]
                # 如果是集成模型，恢复模型列表
                if "models" in model_data:
                    self.models = model_data["models"]
                print("模型加载成功！")
            except Exception as e:
                # 打印错误堆栈以便调试
                traceback.print_exc()
                raise ValueError(f"模型加载失败，请检查文件是否存在: {e}")

        # 定义参数范围
        r1_range = (0.23, 0.38)
        r2_range = (0.52, 0.67)
        s1_range = (-0.8, -0.45)
        s2_range = (0.6, 1.0)

        r1_fixed = 0.27
        r2_fixed = 0.6
        s1_fixed = -0.55
        s2_fixed = 0.7

        # -------------------------------------------
        # 扫描不同的参数子空间
        combinations = [
            ("r1", "r2", r1_range, r2_range, s1_fixed, s2_fixed, "s1", "s2"),  # 676 点
            ("r1", "s1", r1_range, s1_range, r2_fixed, s2_fixed, "r2", "s2"),  # 338 点
            ("r1", "s2", r1_range, s2_range, r2_fixed, s1_fixed, "r2", "s1"),  # 390 点
            # ("r2", "s1", r2_range, s1_range, r1_fixed, s2_fixed, "r1", "s2"),
            # ("r2", "s2", r2_range, s2_range, r1_fixed, s1_fixed, "r1", "s1"),
            # ("s1", "s2", s1_range, s2_range, r1_fixed, r2_fixed, "r1", "r2"),
        ]
        # -------------------------------------------
        # 扫描s2取值，给出r1-r2子空间的系列
        # s2_values = np.arange(0.5, 1.2 + 0.01, 0.025)
        # combinations = [
        #     ("r1", "r2", r1_range, r2_range, s1_fixed, s2_val, "s1", "s2")
        #     for s2_val in s2_values
        # ]
        # -------------------------------------------

        results = {}
        for num, (
            x_param,
            y_param,
            x_range,
            y_range,
            fixed1,
            fixed2,
            fixed_param1,
            fixed_param2,
        ) in enumerate(combinations, 1):
            print(f"\n{'='*60}")
            print(f"组合 {num}/{len(combinations)}: {x_param}-{y_param} 平面")
            print(f"{'='*60}")

            # 根据参数类型选择步长
            x_step = r_step if x_param.startswith("r") else s_step
            y_step = r_step if y_param.startswith("r") else s_step

            # 创建网格
            x_values = np.arange(x_range[0], x_range[1] + x_step / 2, x_step)
            y_values = np.arange(y_range[0], y_range[1] + y_step / 2, y_step)

            # 处理递减范围（如s1从-0.9到-0.3）
            if y_range[0] > y_range[1]:
                y_values = np.arange(y_range[0], y_range[1] - y_step / 2, -y_step)

            X, Y = np.meshgrid(x_values, y_values)

            # 初始化预测矩阵
            pc_predictions = np.zeros_like(X)

            total_points = len(x_values) * len(y_values)
            print(f"网格大小: {X.shape}")

            combo_start_time = time.time()
            # first_run = True

            # 遍历网格进行预测
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    # 根据当前组合构造特征向量
                    features = {}

                    # 设置变化的参数
                    features[x_param] = X[i, j]
                    features[y_param] = Y[i, j]

                    # 设置固定参数
                    if fixed_param1 == "r1" or fixed_param2 == "r1":
                        features["r1"] = r1_fixed
                    elif fixed_param1 == "r2" or fixed_param2 == "r2":
                        features["r2"] = r2_fixed
                    elif fixed_param1 == "s1" or fixed_param2 == "s1":
                        features["s1"] = s1_fixed
                    elif fixed_param1 == "s2" or fixed_param2 == "s2":
                        features["s2"] = s2_fixed

                    # 处理第二个固定参数
                    if fixed_param1 == "r1" and "r1" not in features:
                        features["r1"] = fixed1
                    elif fixed_param1 == "r2" and "r2" not in features:
                        features["r2"] = fixed1
                    elif fixed_param1 == "s1" and "s1" not in features:
                        features["s1"] = fixed1
                    elif fixed_param1 == "s2" and "s2" not in features:
                        features["s2"] = fixed1

                    if fixed_param2 == "r1" and "r1" not in features:
                        features["r1"] = fixed2
                    elif fixed_param2 == "r2" and "r2" not in features:
                        features["r2"] = fixed2
                    elif fixed_param2 == "s1" and "s1" not in features:
                        features["s1"] = fixed2
                    elif fixed_param2 == "s2" and "s2" not in features:
                        features["s2"] = fixed2

                    # 按正确顺序构造输入数组
                    input_array = np.array(
                        [
                            [
                                features.get("r1", 0),
                                features.get("r2", 0),
                                features.get("s1", 0),
                                features.get("s2", 0),
                            ]
                        ]
                    )

                    # 标准化
                    # onepoint_start_time = time.time()
                    input_scaled = self.scaler.transform(input_array)

                    # 预测
                    if hasattr(self, "models") and self.models:
                        predictions = []
                        for model in self.models:
                            pred = model.predict(input_scaled)[0]
                            predictions.append(pred)
                        pc_value = np.mean(predictions)
                    else:
                        pc_value = self.model.predict(input_scaled)[0]

                    pc_predictions[i, j] = pc_value

                    # onepoint_end_time = time.time()
                    # if first_run:
                    #     prediction_time = onepoint_end_time - onepoint_start_time
                    #     print(f"单点用时: {prediction_time:.2e} 秒")
                    #     first_run = False

            # 记录当前组合的结束时间
            combo_end_time = time.time()
            combo_time = combo_end_time - combo_start_time

            # 计算平均预测时间
            avg_prediction_time = combo_time / total_points
            print(f"当前组合预测用时: {combo_time:.2f} 秒")
            print(f"平均每个点预测用时: {avg_prediction_time:.2e} 秒")

            # 绘制当前组合的二维图
            self._plot_parameter_pair(
                X, Y, pc_predictions, x_param, y_param, num, fixed2
            )

            # 保存结果
            results[f"{x_param}_{y_param}"] = {
                "X": X,
                "Y": Y,
                "pc": pc_predictions,
                "x_param": x_param,
                "y_param": y_param,
                "fixed_params": {fixed_param1: fixed1, fixed_param2: fixed2},
            }
        return results

    def _plot_parameter_pair(
        self, X, Y, pc_predictions, x_param, y_param, comb_idx, fixed2
    ):
        """绘制单个参数对的二维分布图"""

        def to_latex_format(param_name):
            # 如果名字最后一位是数字，比如 r1 -> r_1, s2 -> s_2
            if param_name[-1].isdigit():
                return f"{param_name[:-1]}_{param_name[-1]}"
            return param_name  # 如果没有数字则保持原样

        all_fixed_values = {"r1": 0.27, "r2": 0.6, "s1": -0.55, "s2": 0.7}
        remaining_fixed = {
            k: v for k, v in all_fixed_values.items() if k != x_param and k != y_param
        }

        fig, axes = plt.subplots(1, 1, figsize=(7, 6))
        if pc_predictions.shape[0] >= 2 and pc_predictions.shape[1] >= 2:
            contour = axes.contourf(
                X,
                Y,
                pc_predictions,
                levels=np.linspace(0, 0.2, 51),
                cmap="OrRd",
                alpha=0.8,
                vmin=0,
                vmax=0.2,
                extend="max",
            )
            # 绘制阈值线
            axes.contour(
                X,
                Y,
                pc_predictions,
                levels=[self.crash_threshold],
                colors="black",
                linewidths=2.5,
                linestyles="--",
            )
            # 显示图例
            threshold_percent = self.crash_threshold * 100
            legend_label = f"C.P.={threshold_percent:.1f}%"
            proxy_line = Line2D([0], [0], color="black", lw=2.8, linestyle="--")
            axes.legend(
                [proxy_line],
                [legend_label],
                prop={"weight": "bold", "size": 18},
                loc="upper right",
                frameon=True,
                facecolor="white",
                edgecolor="#E0E0E0",
                handlelength=1.5,
            )

            fixed_items = [f"${to_latex_format(k)}$={v}" for k, v in remaining_fixed.items()]
            info_text = f"Fixed:\n{', '.join(fixed_items)}"
            anchored_box = AnchoredText(
                info_text,
                loc="upper right",  # 让它的锚定策略和图例一样，默认往右上靠
                prop=dict(size=16, weight="bold"),  # 字体加大加粗
                frameon=True,
                # borderpad 控制文本框与坐标轴上、右边界的距离
                # 图例默认的 borderpad 是 0.4，这里设为 0.4 能让二者右侧完美绝对对齐！
                borderpad=0.4,
            )
            box_frame = anchored_box.patch
            box_frame.set_facecolor("white")
            box_frame.set_edgecolor("#E0E0E0")
            box_frame.set_boxstyle("round,pad=0.5")
            box_frame.set_alpha(0.95)
            fig.canvas.draw()  # 触发一次画布刷新，计算图例的真实动态高度
            legend_height = axes.get_legend().get_window_extent().height
            anchored_box.set_bbox_to_anchor((0.96, 0.85), transform=axes.transAxes)
            axes.add_artist(anchored_box)

            latex_x = to_latex_format(x_param)
            latex_y = to_latex_format(y_param)
            axes.set_xlabel(
                f"${latex_x}$", fontsize=20, fontweight="bold"
            )
            axes.set_ylabel(
                f"${latex_y}$", fontsize=20, fontweight="bold"
            )
            axes.set_title(
                f"${latex_x}$-${latex_y}$ Subspace", fontsize=22, fontweight="bold"
            )
            axes.tick_params(axis="both", labelsize=18)
            for spine in axes.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(2.5)
        else:
            axes.text(
                0.5,
                0.5,
                f"数据不足\n形状: {pc_predictions.shape}",
                ha="center",
                va="center",
            )

        axes.grid(True, alpha=0.3)
        if "contour" in locals():
            # 修改颜色条设置
            cbar = plt.colorbar(contour, ax=axes, ticks=np.linspace(0, 0.2, 5))
            cbar.ax.tick_params(size=18)
            cbar.set_label("Crash Percentage", size=20)
            cbar.outline.set_linewidth(2.5)
        plt.tight_layout()

        # -------------------------------------------
        # 采用扫描s2绘制r1-r2图时添加s2指示条
        # fig.subplots_adjust(bottom=0.2)
        # ax_pos = fig.add_axes([0.3, 0.07, 0.4, 0.01])
        # s2_min, s2_max = 0.5, 1.2
        # # 绘制背景基准线
        # ax_pos.axhline(0, color="lightgray", linewidth=4, zorder=1)
        # # 绘制当前位置指示点
        # ax_pos.plot(fixed2, 0, "ro", markersize=10, clip_on=False, zorder=2)
        # # 设置坐标轴样式
        # ax_pos.set_xlim(s2_min, s2_max)
        # ax_pos.set_xticks([s2_min, s2_max])
        # ax_pos.set_xticklabels([f"s2={s2_min}", f"s2={s2_max}"], fontsize=18)
        # ax_pos.set_yticks([])
        # # 隐藏边框
        # for spine in ax_pos.spines.values():
        #     spine.set_visible(False)
        # ax_pos.patch.set_alpha(0)
        # -------------------------------------------

        # 保存图像
        filename = f"pc_distribution_{x_param}_{y_param}_{comb_idx}.png"
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.show()
        print(f"已保存: {filename}")

    def _plot_metrics_vs_threshold(self, results):
        """绘制指标随阈值变化图"""
        thresholds = [r["threshold"] for r in results]
        metrics = ["accuracy", "precision", "recall", "f1"]
        labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

        nature_colors = ["#2A729E", "#D55E00", "#009E73", "#56B4E9"]
        line_styles = ["-", "-", "-", "-"]
        markers = ["o", "s", "^", "p"]

        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=600)
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            values = [r[metric] for r in results]
            ax.plot(
                thresholds,
                values,
                label=label,
                color=nature_colors[i],
                linestyle=line_styles[i],
                linewidth=1.5,
                marker=markers[i],
                markersize=4.5,
                markerfacecolor="white",   # 空心标点让多线图更通透、不拥挤
                markeredgewidth=1.5,
                zorder=3,                  # 确保曲线在网格线上方
                clip_on=True,
            )

        ax.set_title("Performance Metrics vs Threshold", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Threshold", fontsize=11, labelpad=8)
        ax.set_ylabel("Score", fontsize=11, labelpad=8)
        ax.tick_params(
            axis="both", which="major", direction="in", labelsize=10, width=1.0, length=4
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("black")
        ax.legend(frameon=False, fontsize=9, loc="best", handlelength=2.5)
        ax.grid(
            visible=True,
            which="major",
            axis="both",
            color="#E5E5E5",
            linestyle="--",
            linewidth=0.6,
            zorder=0,
        )
        plt.tight_layout()
        plt.savefig("metrics_vs_threshold.png", dpi=600, bbox_inches="tight")
        plt.show()
        plt.close()

    def _plot_confusion_matrix(self, best_result, cm):
        """绘制混淆矩阵"""
        tn, fp, fn, tp = cm

        plt.figure(figsize=(6, 5))
        cm_matrix = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(
            cm_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted No", "Predicted Yes"],
            yticklabels=["Actual No", "Actual Yes"],
            annot_kws={"size": 20},
        )
        plt.title(
            f'Confusion Matrix (Threshold = {best_result["threshold"]:.1%})\n'
            f'F1-Score = {best_result["f1"]:.3f}',
            fontsize=20,
        )
        plt.xlabel("Predicted Label", fontsize=18)
        plt.ylabel("True Label", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=600, bbox_inches="tight")
        plt.show()

    def plot_3d_boundary(self, model_path="TaskII_PCcls.pkl", save_gif=True):
        """
        扫描 r1-r2-s2 三维空间，绘制 crash probability = 0.085 的分界面
        固定 s1 = -0.55
        """
        def to_latex_format(param_name):
            if param_name[-1].isdigit():
                return f"{param_name[:-1]}_{param_name[-1]}"
            return param_name

        # 1. 确保模型已加载
        if self.model is None:
            data = joblib.load(model_path)
            self.model, self.scaler = data["model"], data["scaler"]
            if "models" in data:
                self.models = data["models"]

        # 2. 定义三维网格参数
        r1_vals = np.arange(0.23, 0.33 + 0.01, 0.01)
        r2_vals = np.arange(0.52, 0.66 + 0.01, 0.01)
        s2_vals = np.arange(0.6, 0.95 + 0.05, 0.05)
        s1_fixed = -0.55

        # 生成网格 (indexing='ij' 确保坐标顺序对应 r1, r2, s2)
        R1, R2, S2 = np.meshgrid(r1_vals, r2_vals, s2_vals, indexing="ij")

        # 3. 构造输入并进行批量预测
        # 展平以便一次性输入模型
        grid_points = np.column_stack(
            [R1.ravel(), R2.ravel(), np.full(R1.size, s1_fixed), S2.ravel()]
        )
        input_scaled = self.scaler.transform(grid_points)

        # 集成预测
        if hasattr(self, "models") and self.models:
            preds = np.mean([m.predict(input_scaled) for m in self.models], axis=0)
        else:
            preds = self.model.predict(input_scaled)

        # 恢复为三维体数据形状
        volume = preds.reshape(R1.shape)

        # 4. 使用 Marching Cubes 提取等值面 (阈值=0.085)
        try:
            verts, faces, _, _ = measure.marching_cubes(
                volume, self.crash_threshold, step_size=1
            )
        except RuntimeError:
            print("未在当前空间内找到指定阈值的等值面。")
            return

        # 5. 坐标变换：将索引坐标转换回物理坐标 (r1, r2, s2)
        # verts 的列顺序对应 indexing='ij' 下的 axis 0(r1), 1(r2), 2(s2)
        real_verts = np.zeros_like(verts)
        real_verts[:, 0] = verts[:, 0] * 0.01 + 0.23  # r1
        real_verts[:, 1] = verts[:, 1] * 0.01 + 0.52  # r2
        real_verts[:, 2] = verts[:, 2] * 0.05 + 0.6  # s2

        # 6. 3D 绘图
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 创建多边形集合
        mesh = Poly3DCollection(real_verts[faces], alpha=0.7)
        mesh.set_facecolor("coral")
        mesh.set_edgecolor("black")
        mesh.set_linewidth(0.1)
        ax.add_collection3d(mesh)

        # 设置坐标轴范围和标签
        ax.set_xlim(0.23, 0.34)
        ax.set_ylim(0.52, 0.67)
        ax.set_zlim(0.6, 1.0)
        ax.set_xlabel(
            f"${to_latex_format('r1')}$", fontsize=20, fontweight="bold", labelpad=14
        )
        ax.set_ylabel(
            f"${to_latex_format('r2')}$", fontsize=20, fontweight="bold", labelpad=14
        )
        ax.set_zlabel(
            f"${to_latex_format('s2')}$", fontsize=20, fontweight="bold", labelpad=14
        )
        threshold_percent = self.crash_threshold * 100
        latex_s1 = to_latex_format("s1")
        ax.set_title(
            f"3D Crash Boundary (C.P.={threshold_percent:.1f}%)\nFixed ${latex_s1}$={s1_fixed}",
            fontsize=22,
            fontweight="bold",
            pad=10,  # 增加标题间距防止与 3D 轴重叠
        )
        for axis_obj in [ax.xaxis, ax.yaxis, ax.zaxis]:
            for label in axis_obj.get_ticklabels():
                label.set_fontsize(18)
                label.set_fontweight("bold")

        if save_gif:
            print("正在生成旋转动画帧...")
            frames = []
            # 设定旋转路径：水平旋转 360 度
            azimuths = np.arange(0, 360, 2)  # 每隔2度采一帧，共180帧

            for i, azim in enumerate(azimuths):
                ax.view_init(elev=30, azim=azim)  # 改变视角

                # 将当前画布转换为图片数组
                fig.canvas.draw()
                # 使用 buffer_rgba 代替 tostring_rgb
                rgba_buffer = fig.canvas.buffer_rgba()
                # 直接转换为 numpy 数组，它会自动处理高 DPI 带来的维度变化
                image = np.array(rgba_buffer)
                # 丢弃 Alpha 通道（RGBA -> RGB），imageio 存 GIF 通常只需要 RGB
                image = image[:, :, :3]
                frames.append(image)

                if i % 10 == 0:
                    print(f"已处理 {i}/{len(azimuths)} 帧")

            # 保存为 GIF (duration 为帧间隔时间，单位秒)
            imageio.mimsave("3d_boundary_rotation.gif", frames, fps=15, loop=0)
            print("动画已保存为: 3d_boundary_rotation.gif")
        else:
            # 原有的静态保存逻辑
            ax.view_init(elev=30, azim=-60)
            plt.savefig("3d_crash_boundary.png", dpi=300)
            plt.show()


if __name__ == "__main__":

    regressor = PlasmaDisruptionRegressor()
    # -------------------------------------------
    X, y = regressor.load_data("TaskII_PCcls.csv")
    regressor.train_model(X, y, use_grid_search=True)
    # regressor.train_model(X, y, use_grid_search=False)

    # use_grid_search = bool(int(input("是否进行超参搜索？(0-否, 1-是): ")))
    # regressor.train_model(X, y, use_grid_search=use_grid_search)

    regressor.save_model("TaskII_PCcls.pkl")

    print("\n开始阈值扫描和ROC分析...")
    results = regressor.evaluate_thresholds(X, y)
    best_result = max(results, key=lambda x: x["f1"])
    print(f"\n最佳阈值: {best_result['threshold']:.3f}")
    print(f"最佳F1分数: {best_result['f1']:.3f}")
    # -------------------------------------------

    # 预测并利用扫描结果绘制参数空间分布图
    # regressor.scan_all_parameter_pairs(r_step=0.01, s_step=0.05)

    # 预测三维参数空间并绘制二维分界面
    # regressor.plot_3d_boundary()
