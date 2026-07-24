import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_nature_style_lines(excel_path):
    df = pd.read_excel(excel_path)
    df = df.iloc[:, :5]
    x_col = df.columns[0]
    y_cols = df.columns[1:5]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False  # Render minus signs correctly.

    nature_colors = ["#2A729E", "#D55E00", "#009E73", "#56B4E9"]
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "p"]

    fig, ax = plt.subplots(figsize=(5, 3), dpi=600)

    for i, y_col in enumerate(y_cols):
        label_text = f"{y_col}"

        ax.plot(
            df[x_col],
            df[y_col],
            label=label_text,
            color=nature_colors[i],
            linestyle=line_styles[i],
            linewidth=1.5,
            marker=markers[i],
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            clip_on=True,
        )

    ax.set_xlabel(f"{x_col}", fontsize=11, fontdict={"weight": "normal"}, labelpad=8)
    ax.set_ylabel(
        "MRE Performance(%)",
        fontsize=11,
        fontdict={"weight": "normal"},
        labelpad=8,
    )
    ax.set_xlim(2, 23)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylim(1, 20)
    ax.yaxis.set_major_locator(MultipleLocator(2))

    # Adjust tick size and direction: inward ticks and heavier lines.
    ax.tick_params(
        axis="both", which="major", direction="in", labelsize=10, width=1.0, length=4
    )
    ax.tick_params(axis="both", which="minor", direction="in", width=0.8, length=2)
    # ax.minorticks_on()

    # Use heavier black borders instead of Matplotlib's default light spines.
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # 3. Remove the legend border and place it automatically.
    ax.legend(frameon=False, fontsize=9, loc="best", handlelength=2.5)

    ax.grid(
        visible=True, 
        which='major',       # Draw the grid only at major ticks.
        axis='both',         # Draw on both axes.
        color='#E5E5E5',     # Very light gray.
        linestyle='--',      # Dashed line.
        linewidth=0.6,       # Thin line.
        zorder=0             # Keep grid lines behind the data.
    )

    box_text = "Zero-Shot Performance (MRE):\n"
    box_text += r"  $W_{Bi}^{\mathrm{max}}$" + "    $\geq$"+" 14.06%\n"
    box_text += r"  $W_{Bo}^{\mathrm{max}}$" + "    $\geq$"+" 7.55%\n"
    box_text += r"  $\gamma$" + "           $\geq$"+" 45.29%\n"
    box_text += r"  $E_k^{\mathrm{max}}$" + "     $\geq$"+" 140.02%"
    ax.text(
        0.25,
        0.95,
        box_text,
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top",
        zorder=5,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.75,
            edgecolor="#E5E5E5",
            linewidth=0.8,
        ),
    )
    plt.tight_layout()
    output_png = "performance vs data.png"

    plt.savefig(output_png, dpi=600, bbox_inches="tight")
    # plt.show()
    plt.close()


if __name__ == "__main__":

    excel_file_path = "performance vs data.xlsx"
    plot_nature_style_lines(excel_file_path)
