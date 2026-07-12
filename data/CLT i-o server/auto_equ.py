import os
import shutil
import time
from pathlib import Path

def automate_simulation():
    # 1. 设置路径
    # 获取当前 Python 脚本所在的目录 (工作区根目录)
    base_dir = Path.cwd()
    # 设定 share 文件夹路径 (当前目录的上一级的 share 文件夹)
    # share_dir = base_dir.parent / "share"
    share_dir = base_dir / "share"

    # 待监控的目标文件列表
    target_files = ['psi_xz.dat', 'q_p_g.dat', 'wch.dat', 'ploteq.pdf']
    # 需要复制到 share 的源文件
    source_files = ['eq_transp.F', 'inequ']

    # 2. 扫描并遍历子文件夹
    # 过滤掉非文件夹对象
    subfolders = [f for f in base_dir.iterdir() if f.is_dir()]

    # 可以按名称排序，确保从 1 号到 N 号顺序执行
    # subfolders.sort()

    for folder in subfolders:
        print(f"\n>>> 正在处理工况: {folder.name}")

        # --- A. 复制文件到 share ---
        for file_name in source_files:
            src_file = folder / file_name
            if src_file.exists():
                shutil.copy(src_file, share_dir / file_name)
                print(f"   已复制 {file_name} 到 share")
            else:
                print(f"   警告: {src_file} 不存在，跳过该文件")

        # --- B. 监控 share 文件夹并等待生成 ---
        print(f"   等待平衡程序计算和提交数据文件...")
        while True:
            # 检查是否所有目标文件都已生成
            all_exist = all((share_dir / f).exists() for f in target_files)

            if all_exist:
                # 额外的小技巧：有时文件刚创建但内容还没写完，稍微等 3 秒确保写入完成
                time.sleep(3)
                print("   检测到所有结果文件，开始移动...")
                break

            # 每隔 2 秒检查一次
            time.sleep(2)

        # --- C. 将生成的四个文件移动回子文件夹 ---
        for file_name in target_files:
            target_path = share_dir / file_name
            destination_path = folder / file_name

            # 使用 move 移动文件（相当于剪切，清理 share 空间）
            try:
                shutil.move(str(target_path), str(destination_path))
                print(f"   已移动 {file_name} -> {folder.name}")
            except Exception as e:
                print(f"   移动 {file_name} 时出错: {e}")

        # --- D. 清理工作：删除本轮在 share 文件夹中的输入文件 ---
        for file_name in source_files:
            file_to_delete = share_dir / file_name
            if file_to_delete.exists():
                file_to_delete.unlink()  # unlink() 专门用于删除文件
                print(f"   已从 share 中清除旧的 {file_name}")

    print("\n========================================")
    print("所有工况遍历完成！")

if __name__ == "__main__":
    automate_simulation()
