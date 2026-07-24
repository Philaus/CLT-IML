import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Interpolate and plot the selected perturbation-field data to assess sampling.

# Set the figure style.
plt.style.use("default")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

target_folder = "selected_B9_633"
csv_files = glob.glob(os.path.join(target_folder, "*.csv"))

for csv_file in csv_files:
    try:
        # Read the CSV file.
        df = pd.read_csv(csv_file)

        # Extract data.
        R = df["X"].values * 0.94
        Z = df["Z"].values * 0.94
        values = df["Value"].values

        # Create the figure.
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

        # Scatter plot showing the sampled-data distribution.
        ax1.tick_params(axis="both", direction="in", length=2, width=1, labelsize=12)
        scatter1 = ax1.scatter(
            R, Z, c=values, cmap="viridis", s=10, alpha=0.8, vmin=-3e-5, vmax=3e-5
        )
        ax1.set_xlabel("R", fontsize=20)
        ax1.set_ylabel("Z", fontsize=20)
        ax1.set_title("Selected Data Points Distribution", fontsize=20)
        ax1.tick_params(axis="both", labelsize=20)
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)

        circle = plt.Circle((0, 0), 0.94, fill=False, color="red", linewidth=2)
        ax1.add_patch(circle)
        ax1.set_xlim(-0.94, 0.94)
        ax1.set_ylim(-0.94, 0.94)
        cbar = plt.colorbar(scatter1, ax=ax1, label="Value")
        cbar.set_ticks([-3e-5, -2e-5, -1e-5, 0, 1e-5, 2e-5, 3e-5])
        cbar.ax.tick_params(labelsize=20)  # Color-bar tick-label size.
        cbar.set_label("Value", fontsize=20)  # Color-bar label size.

        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_filename = os.path.join(target_folder, base_name + ".png")
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"已生成: {output_filename}")

    except Exception as e:
        print(f"处理文件 {csv_file} 时出错: {str(e)}")
        import traceback

        traceback.print_exc()
        continue

print("所有文件处理完成！")
