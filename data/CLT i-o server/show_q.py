import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob


def extract_data_from_dat(filename):

    data = np.loadtxt(filename)
    y_data = data[:, 2]
    x_data = np.linspace(0, 1, len(y_data))
    return x_data, y_data


def custom_function(r, qc, Fr0, Flamuta, FA0, r_delta, Fdelta):

    term1 = qc * (1 + (r / Fr0) ** (2 * Flamuta)) ** (1 / Flamuta)
    term2 = (1 + FA0 * np.exp(-(((r - r_delta) / Fdelta) ** 2))) / (1 + FA0)
    return term1 * term2


if __name__ == "__main__":

    filename = "q_p_g.dat"

    try:
        x_data, y_data = extract_data_from_dat(filename)
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, "b-", linewidth=2, label="Data from file")

        r = np.linspace(0, 1, 300)

        qc = 4.0
        r_delta = 0.0

        csv_files = glob.glob("*.csv")
        df = pd.read_csv(csv_files)
        Fr0, Fdelta, FA0, Flamuta = df[["Fr0", "Fdelta", "FA0", "Flamuta"]].values.T

        q_values = custom_function(r, qc, Fr0, Flamuta, FA0, r_delta, Fdelta)

        plt.plot(r, q_values, "r--", linewidth=2, label="Theoretical function")
        plt.legend()
        plt.xlabel("r")
        plt.ylabel("q")
        plt.title("Data from File vs Theoretical Function")
        plt.grid(True, alpha=0.3)
        plt.savefig('q.png', dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"错误：找不到文件 {filename}")
    except Exception as e:
        print(f"处理文件时发生错误：{e}")