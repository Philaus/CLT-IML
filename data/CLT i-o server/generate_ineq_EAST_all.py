import os
import shutil
from pathlib import Path

# ================= Configuration =================
# Replacement content for inequ.
NEW_INEQU_LINE_5 = (
    "03        0.70      .00000001 300.0     2200      0.0                 .200\n"
)
NEW_INEQU_LINE_7 = (
    "05        1.8500    +1.850    0.4500    1.9       2.0       1.0       0.50\n"
)

# Files to copy.
FILES_TO_COPY = [
    "tokRZ_mpi_pll_acc_merged.f90",
    "wsdiagncd_allin2_mnmode.f90",
    "wsdiagncd_allin2_mnmode_outBrmn.f90",
    "run4090.sh",
    "dia_main.out",
    "dia_Bmn.out",
]
# ===========================================

# current_dir_full = ["batch11" ,"batch22", "batch33", "batch44"]
current_dir_full = ["batch44"]
def batch_process():

    for current_dir in current_dir_full:
        # Get all subfolders in the current directory.
        subdirs = [
            os.path.join(current_dir, d)
            for d in os.listdir(current_dir)
            if os.path.isdir(os.path.join(current_dir, d))
        ]

        print(f"检测到 {len(subdirs)} 个子文件夹，开始处理...")

        for i, subdir in enumerate(subdirs):
            # 1. Calculate the sequence number (cnt).
            # Use (i + 1).zfill(2) for names such as case01 and case02.
            cnt = str(i + 1).zfill(2)

            # Cycle device labels from 2 through 9 for GPU binding.
            device_id = 2 + (i % 8)

            print(f"---> 正在处理: {subdir} (序号: {cnt}, GPU ID: {device_id})")

            # 2. Process the inequ file.
            inequ_path = os.path.join(subdir, "inequ")
            if os.path.exists(inequ_path):
                with open(inequ_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if len(lines) >= 7:
                    lines[4] = NEW_INEQU_LINE_5
                    lines[6] = NEW_INEQU_LINE_7
                    with open(inequ_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)

            # 3. Copy files before modifying the copied versions.
            for file_name in FILES_TO_COPY:
                if os.path.exists(file_name):
                    shutil.copy2(file_name, subdir)

            # # 4. Modify the device number in the copied tokRZ...f90 file.
            # fortran_path = os.path.join(subdir, "tokRZ_mpi_pll_acc_merged.f90")
            # if os.path.exists(fortran_path):
            #     with open(fortran_path, "r", encoding="utf-8") as f:
            #         content = f.read()
            #     old_target = "!$acc set device_num(nrank+3)"
            #     new_target = f"!$acc set device_num(nrank+{device_id})"
            #     if old_target in content:
            #         content = content.replace(old_target, new_target)
            #         with open(fortran_path, "w", encoding="utf-8") as f:
            #             f.write(content)

            # 4+. Modify the device number and eta10 variable in the copied tokRZ...f90 file.
            fortran_path = os.path.join(subdir, "tokRZ_mpi_pll_acc_merged.f90")
            if os.path.exists(fortran_path):
                with open(fortran_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # --- A. Modify the device number. ---
                old_target = "!$acc set device_num(nrank+3)"
                new_target = f"!$acc set device_num(nrank+{device_id})"
                content = content.replace(old_target, new_target)

                # --- B. Calculate r from the CSV and modify eta10. ---
                import glob
                import csv
                
                # Find CSV files in the subdirectory.
                csv_files = glob.glob(os.path.join(subdir, "*.csv"))
                if csv_files:
                    try:
                        with open(csv_files[0], mode='r', encoding='utf-8') as f_csv:
                            reader = csv.DictReader(f_csv)
                            row = next(reader)
                            # Calculate r = Fr2 - Fr1.
                            r = float(row['Fr2']) - float(row['Fr1'])
                        
                        # Determine the eta10 string from the configured logic.
                        if r < 0.16:
                            eta10_val = "9e-6"
                        elif r <= 0.20:
                            eta10_val = "1e-5"
                        elif r <= 0.26:
                            eta10_val = "2e-5"
                        elif r <= 0.29:
                            eta10_val = "3e-5"
                        elif r <= 0.32:
                            eta10_val = "6e-5"
                        elif r <= 0.36:
                            eta10_val = "8e-5"
                        else:
                            eta10_val = "1e-4"
                        
                        # Perform the replacement using the required string format.
                        content = content.replace("      eta10=6e-05", f"      eta10={eta10_val}")
                        print(f"r: {r:.4f} | eta10 更新为: {eta10_val}")
                    except Exception as e:
                        print(f"读取 CSV 或处理数据出错: {e}")

                # Save the modified file.
                with open(fortran_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # 5. Dynamically modify lines 12 and 13 of run.sh.
            run_sh_path = os.path.join(subdir, "run2080.sh")
            if os.path.exists(run_sh_path):
                with open(run_sh_path, "r", encoding="utf-8") as f:
                    run_lines = f.readlines()

                # Build the case{cnt} string dynamically.
                case_name = f"case{cnt}.out"
                run_lines[9] = (
                    f"mpif90fftwopenacc tokRZ_mpi_pll_acc_merged.f90 -o {case_name}\n"
                )
                run_lines[10] = (
                    f"nohup mpirun -np 1 --bind-to socket --mca oob_tcp_if_include lo {case_name} &\n"
                )

                with open(run_sh_path, "w", encoding="utf-8") as f:
                    f.writelines(run_lines)
                with open(run_sh_path, 'r', encoding='utf-8') as f:
                    content = f.read().replace('\r\n', '\n')
                with open(run_sh_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(content)

            Path(subdir, "a.out").unlink(missing_ok=True)

    print("\n所有子文件夹处理完毕。")


if __name__ == "__main__":
    batch_process()
