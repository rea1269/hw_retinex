import os
import shutil
import argparse
from pathlib import Path

def parse_filename_info(filename: str):
    """
    解析文件名，提取出数字后缀以及自集成状态(sem/no_sem)
    例如: 'v1_no_sem_tar_ex_1.png' -> num_suffix='1', sem_status='no_sem'
         'FiveK_sem_no_tar_hi_in_10.png' -> num_suffix='10', sem_status='sem'
    """
    name_without_ext = Path(filename).stem
    segments = name_without_ext.split('_')
    
    # 1. 提取最后的数字后缀
    last_segment = segments[-1]
    num_suffix = last_segment if last_segment.isdigit() else ""
    
    # 2. 提取自集成状态 (通过检查文件名中是否包含特定片段)
    # 优先匹配 no_sem，然后再匹配 sem
    if "no_sem" in name_without_ext:
        sem_status = "no_sem"
    elif "sem" in name_without_ext:
        sem_status = "sem"
    else:
        sem_status = "unknown" # 容错处理
        
    return num_suffix, sem_status

def classify_and_copy_files(src_dir: Path, dst_root: Path, move_mode=False):
    """
    扫描 src_dir 下所有的 png 文件，根据数字后缀和 sem/no_sem 状态进行二级目录归类
    """
    if not src_dir.is_dir():
        print(f"[ERROR] 输入源路径不存在: {src_dir}")
        return

    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"📂 开始扫描源目录: {src_dir}")
    print(f"📦 目标二级归类目录: {dst_root}")
    print(f"🔄 运行模式: {'【剪切/移动】' if move_mode else '【复制/保留原文件】'}")
    print("-" * 60)

    success_count = 0
    skip_count = 0

    # 深度递归扫描所有文件
    for p in src_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
            
        if p.name.startswith("._") or p.name.lower() == ".ds_store":
            continue

        # 1. 解析文件名中的关键信息
        num_suffix, sem_status = parse_filename_info(p.name)
        
        if not num_suffix:
            print(f"[WARN] 文件 '{p.name}' 结尾不是纯数字，已跳过。")
            skip_count += 1
            continue

        # 2. 核心改进：构建二级子文件夹路径（例如: result_mydata/1/no_sem）
        target_sub_dir = dst_root / num_suffix / sem_status
        target_sub_dir.mkdir(parents=True, exist_ok=True) # 自动级联创建多层文件夹

        # 3. 拼接最终的目标文件路径
        target_file_path = target_sub_dir / p.name

        try:
            if move_mode:
                shutil.move(str(p), str(target_file_path))
            else:
                shutil.copy2(str(p), str(target_file_path))
                
            print(f"归类成功: {p.name}  ==>  result_mydata/{num_suffix}/{sem_status}/")
            success_count += 1
        except Exception as e:
            print(f"[FAILED] 处理文件 {p.name} 时发生错误: {e}")

    print("-" * 60)
    print(f"🎉 归类完成！共成功聚合了 {success_count} 个文件，跳过/不符文件 {skip_count} 个。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将重命名后的多层级图片按照数字后缀和sem状态聚合到二级文件夹中")
    
    parser.add_argument('--src', '-s', type=Path, default=Path("result/mydata_png"),
                        help="包含已重命名文件的根目录 (默认: result/mydata_png)")
    
    parser.add_argument('--dst', '-d', type=Path, default=Path("result_mydata"),
                        help="分类后存放的根目录 (默认: result_mydata)")
    
    parser.add_argument('--move', action='store_true',
                        help="加入此参数将使用【剪切】而非【复制】")

    args = parser.parse_args()
    classify_and_copy_files(args.src.resolve(), args.dst.resolve(), move_mode=args.move)