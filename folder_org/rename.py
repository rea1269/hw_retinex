import os
import argparse

def batch_rename(folder_path, prefix="LOLv1"):
    """
    将指定文件夹下的纯数字文件名修改为 '前缀_数字' 的形式
    """
    if not os.path.exists(folder_path):
        print(f"错误：路径 '{folder_path}' 不存在，请检查路径是否正确。")
        return

    file_list = os.listdir(folder_path)
    
    success_count = 0
    skip_count = 0
    
    print(f"开始处理文件夹: {folder_path}")
    print(f"设定英文前缀为: {prefix}")
    print("-" * 50)
    
    for item in file_list:
        old_path = os.path.join(folder_path, item)
        
        if os.path.isdir(old_path):
            continue

        filename, extension = os.path.splitext(item)
        
        if filename.isdigit() and not filename.startswith(f"{prefix}_"):
            new_name = f"{prefix}_{filename}{extension}"
            new_path = os.path.join(folder_path, new_name)
            
            try:
                os.rename(old_path, new_path)
                print(f"成功: {item}  ==>  {new_name}")
                success_count += 1
            except Exception as e:
                print(f"失败: 重命名 {item} 时发生错误: {e}")
        else:
            skip_count += 1
            
    print("-" * 50)
    print(f"处理完成！成功重命名了 {success_count} 个文件，跳过了 {skip_count} 个不符合条件的文件。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量将纯数字文件名重命名为 '前缀_数字' 格式")
    
    parser.add_argument('--path', '-p', type=str, required=True, 
                        help="目标文件夹的绝对路径或相对路径")
    
    parser.add_argument('--prefix', '-f', type=str, default="LOLv1", 
                        help="要添加的英文前缀 (默认: LOLv1)")
    
    args = parser.parse_args()

    batch_rename(args.path, prefix=args.prefix)