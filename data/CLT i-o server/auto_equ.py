import os
import shutil
import time
from pathlib import Path

def automate_simulation():
    # 1. Configure paths.
    # Get the directory containing this Python script (the workspace root).
    base_dir = Path.cwd()
    # Set the share-folder path (the share folder one level above this directory).
    # share_dir = base_dir.parent / "share"
    share_dir = base_dir / "share"

    # List of target files to monitor.
    target_files = ['psi_xz.dat', 'q_p_g.dat', 'wch.dat', 'ploteq.pdf']
    # Source files to copy into the share folder.
    source_files = ['eq_transp.F', 'inequ']

    # 2. Scan and iterate over subfolders.
    # Exclude entries that are not folders.
    subfolders = [f for f in base_dir.iterdir() if f.is_dir()]

    # Sort by name to ensure execution from case 1 through case N.
    # subfolders.sort()

    for folder in subfolders:
        print(f"\n>>> 正在处理工况: {folder.name}")

        # --- A. Copy files to the share folder. ---
        for file_name in source_files:
            src_file = folder / file_name
            if src_file.exists():
                shutil.copy(src_file, share_dir / file_name)
                print(f"   已复制 {file_name} 到 share")
            else:
                print(f"   警告: {src_file} 不存在，跳过该文件")

        # --- B. Monitor the share folder and wait for generated files. ---
        print(f"   等待平衡程序计算和提交数据文件...")
        while True:
            # Check whether all target files have been generated.
            all_exist = all((share_dir / f).exists() for f in target_files)

            if all_exist:
                # A new file may still be receiving data; wait three seconds for writes to finish.
                time.sleep(3)
                print("   检测到所有结果文件，开始移动...")
                break

            # Check every two seconds.
            time.sleep(2)

        # --- C. Move the four generated files back into the subfolder. ---
        for file_name in target_files:
            target_path = share_dir / file_name
            destination_path = folder / file_name

            # Move rather than copy the file to free space in the share folder.
            try:
                shutil.move(str(target_path), str(destination_path))
                print(f"   已移动 {file_name} -> {folder.name}")
            except Exception as e:
                print(f"   移动 {file_name} 时出错: {e}")

        # --- D. Cleanup: remove this iteration's input files from the share folder. ---
        for file_name in source_files:
            file_to_delete = share_dir / file_name
            if file_to_delete.exists():
                file_to_delete.unlink()  # unlink() removes a file.
                print(f"   已从 share 中清除旧的 {file_name}")

    print("\n========================================")
    print("所有工况遍历完成！")

if __name__ == "__main__":
    automate_simulation()
