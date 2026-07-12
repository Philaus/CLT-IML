import os
import pandas as pd
import glob
import shutil


def extract_csv_data():
    """
    扫描当前文件夹中的CSV文件
    """
    csv_files = glob.glob("*.csv")
    extracted_data = {}
    r12 = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        data_row = df.iloc[0]

        Fr0 = data_row.iloc[4]
        Fdelta = data_row.iloc[5]
        FA0 = data_row.iloc[6]
        Flamuta = data_row.iloc[8]

        r2 = data_row.iloc[1]
        r1 = data_row.iloc[0]

        extracted_data[csv_file] = {
            "Fr0": Fr0,
            "Fdelta": Fdelta,
            "FA0": FA0,
            "Flamuta": Flamuta,
        }
        r12.append(r2 - r1)
        print(f"{csv_file}: r12={r2 - r1:.2f}")

    return extracted_data, r12


def process_files(index):
    """
    主处理函数：为每个CSV文件创建文件夹并修改文件
    """
    # 提取CSV数据
    csv_data, r12 = extract_csv_data()
    cnt = 1

    # 为每个CSV文件创建文件夹并处理
    for i, (csv_file, data) in enumerate(csv_data.items(), 0):
        # 创建文件夹名（1-***格式）
        folder_name = (
            f"{index}{str(cnt).zfill(2)}-{os.path.splitext(csv_file)[0]}-r12={r12[i]:.2f}"
        )
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # 修改 inequ 和 run.sh 文件
        modify_inequ(data, folder_name, cnt, index)

        # 修改 eq_transp.F 和 tokRZ_mpi_pll_acc_merged.f90 文件
        modify_eq_transp(data, folder_name, cnt, r12)

        shutil.copy2(csv_file, os.path.join(folder_name, csv_file))
        shutil.copy2("show_q.py", os.path.join(folder_name, "show_q.py"))
        cnt += 1


def modify_inequ(data, folder_name, cnt, index):
    """
    修改inequ文件并复制到目标文件夹
    """
    with open("inequ", "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 修改第11行（索引10）
    if len(lines) > 10:
        new_line = f"12        4.00      {data['Fr0']:.4f}    {data['Flamuta']:.4f}    {data['FA0']:07.4f}    2.0      +1.         1.0\n"
        lines[10] = new_line
    # 写入修改后的内容到目标文件夹
    target_file = os.path.join(folder_name, "inequ")
    with open(target_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    """
    修改run.sh文件并复制到目标文件夹
    """
    with open("run.sh", "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) > 5:
        new_line = f"mv a.out case{index}{cnt:02d}.out\n"
        lines[5] = new_line
    if len(lines) > 6:
        new_line = f"nohup mpirun -n 1 case{index}{cnt:02d}.out &\n"
        lines[6] = new_line
    target_file = os.path.join(folder_name, "run.sh")
    with open(target_file, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)


def modify_eq_transp(data, folder_name, cnt, r12):
    """
    修改eq_transp.F文件并复制到目标文件夹
    """
    with open("eq_transp.F", "r", encoding="utf-8") as f:
        content = f.read()
    old_pattern = "     &    *(1+qdp0*exp(-ybar/0.195**2))/(1+qdp0)"
    new_pattern = f"     &    *(1+qdp0*exp(-ybar/{data['Fdelta']}**2))/(1+qdp0)"
    content = content.replace(old_pattern, new_pattern)
    # 写入修改后的内容到目标文件夹
    target_file = os.path.join(folder_name, "eq_transp.F")
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(content)

    """
    修改tokRZ_mpi_pll_acc_merged.f90文件并复制到目标文件夹
    """
    with open("tokRZ_mpi_pll_acc_merged.f90", "r") as f:
        content = f.read()
    old_pattern = "       !$acc set device_num(nrank+8)"
    new_pattern = f"       !$acc set device_num(nrank+{(cnt-1)%10})"
    content = content.replace(old_pattern, new_pattern)

    r = int(r12[cnt - 1] * 100)
    if r < 16:
        eta10 = 9e-6
    elif r <= 20:
        eta10 = 1e-5
    elif r <= 26:
        eta10 = 2e-5
    elif r <= 29:
        eta10 = 3e-5
    elif r <= 32:
        eta10 = 6e-5
    elif r <= 36:
        eta10 = 8e-5
    else:
        eta10 = 1e-4

    old_pattern = "      eta10=8e-5"
    new_pattern = f"      eta10={eta10:.0e}"
    content = content.replace(old_pattern, new_pattern)

    target_file = os.path.join(folder_name, "tokRZ_mpi_pll_acc_merged.f90")
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(content)

    shutil.copy("wsdiagncd_allin2_mnmode.f90", folder_name)


if __name__ == "__main__":

    index = "11"
    process_files(index)
