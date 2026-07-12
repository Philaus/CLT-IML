import os
import shutil

if __name__ == "__main__":
    # 定义要复制的文件列表
    files_to_copy = [
        "paper_fft_gr_figure.m",
        "plot_growth.m", 
        "plot_ke_fft.m",
        "plot_x12d.m"
    ]

    root_dir = os.getcwd()
    # 遍历根目录下的所有项目
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)

        if os.path.isdir(item_path):
            print(f"\n处理文件夹: {item}")
            try:
                # 1. 在文件夹内创建case文件夹
                case_dir = os.path.join(item_path, "case")
                if not os.path.exists(case_dir):
                    os.makedirs(case_dir)
                    print(f"  创建文件夹: {case_dir}")

                # 2. 移动原文件夹内的所有文件到case文件夹
                moved_files = []
                for file_item in os.listdir(item_path):
                    file_item_path = os.path.join(item_path, file_item)

                    # 跳过case文件夹本身和要复制的文件（如果已存在）
                    if (file_item == "case" or 
                        (os.path.isfile(file_item_path) and file_item in files_to_copy)):
                        continue

                    # 移动文件或文件夹到case目录
                    destination = os.path.join(case_dir, file_item)
                    shutil.move(file_item_path, destination)
                    moved_files.append(file_item)

                # 3. 复制指定的四个文件到当前文件夹
                copied_files = []
                for file_name in files_to_copy:
                    source_file = os.path.join(root_dir, file_name)
                    dest_file = os.path.join(item_path, file_name)

                    if os.path.exists(source_file):
                        shutil.copy2(source_file, dest_file)
                        copied_files.append(file_name)
                    else:
                        print(f"  警告: 源文件 {file_name} 不存在，跳过复制")

                if copied_files:
                    print(f"  复制文件: {', '.join(copied_files)}")

                print(f"  完成处理: {item}")

            except Exception as e:
                print(f"  处理文件夹 {item} 时出错: {e}")