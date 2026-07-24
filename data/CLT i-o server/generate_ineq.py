import os
import pandas as pd
import glob
import shutil


def extract_csv_data():
    """
    Scan CSV files in the current folder.
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
    Create a folder for each CSV file and prepare the required files.
    """
    # Extract CSV data.
    csv_data, r12 = extract_csv_data()
    cnt = 1

    # Create and process a folder for each CSV file.
    for i, (csv_file, data) in enumerate(csv_data.items(), 0):
        # Create the folder name using the 1-*** format.
        folder_name = (
            f"{index}{str(cnt).zfill(2)}-{os.path.splitext(csv_file)[0]}-r12={r12[i]:.2f}"
        )
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Modify the inequ and run.sh files.
        modify_inequ(data, folder_name, cnt, index)

        # Modify the eq_transp.F and tokRZ_mpi_pll_acc_merged.f90 files.
        modify_eq_transp(data, folder_name, cnt, r12)

        shutil.copy2(csv_file, os.path.join(folder_name, csv_file))
        shutil.copy2("show_q.py", os.path.join(folder_name, "show_q.py"))
        cnt += 1


def modify_inequ(data, folder_name, cnt, index):
    """
    Modify the inequ file and copy it to the target folder.
    """
    with open("inequ", "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Modify line 11 (index 10).
    if len(lines) > 10:
        new_line = f"12        4.00      {data['Fr0']:.4f}    {data['Flamuta']:.4f}    {data['FA0']:07.4f}    2.0      +1.         1.0\n"
        lines[10] = new_line
    # Write the modified content to the target folder.
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
    Modify the eq_transp.F file and copy it to the target folder.
    """
    with open("eq_transp.F", "r", encoding="utf-8") as f:
        content = f.read()
    old_pattern = "     &    *(1+qdp0*exp(-ybar/0.195**2))/(1+qdp0)"
    new_pattern = f"     &    *(1+qdp0*exp(-ybar/{data['Fdelta']}**2))/(1+qdp0)"
    content = content.replace(old_pattern, new_pattern)
    # Write the modified content to the target folder.
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
