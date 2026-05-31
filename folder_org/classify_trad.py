import os
import shutil
import argparse
from pathlib import Path

# 第一层数据集文件夹的映射表，用来规范化 result_trad 下的根文件夹名称
DATASET_MAPPING = {
    "FiveK": "FiveK",
    "LOLv1": "LOLv1",
    "LOLv2": "LOLv2",
    "mydata_png": "mydata_png"
}

def parse_traditional_filename(filename: str):
    """
    从重命名后的传统方法文件名中提取核心信息
    例如: 'v1_clahe_1.png'       -> dataset_key='LOLv1', num_suffix='1'
         'FiveK_combined_10.png' -> dataset_key='FiveK', num_suffix='10'
         'v2_real_gamma_5.png'   -> dataset_key='LOLv2', num_suffix='5'
         'my_tar_hi_msr_12.png'  -> dataset_key='mydata_png', num_suffix='12'
    """
    name_without_ext = Path(filename).stem
    segments = name_without_ext.split('_')
    
    # 1. 提取最后的数字后缀
    last_segment = segments[-1]
    num_suffix = last_segment if last_segment.isdigit() else ""
    
    # 2. 识别它属于哪一个数据集 (根据文件名前缀包含的关键字)
    # 这里的判断依据源自我们之前 rename_traditional.py 写入的前缀
    if name_without_ext.startswith("v1_"):
        dataset_key = "LOLv1"
    elif name_without_ext.startswith("FiveK_"):
        dataset_key = "FiveK"
    elif name_without_ext.startswith("v2_"):
        dataset_key = "LOLv2"
    elif name_without_ext.startswith("my_"):
        dataset_key = "mydata_png"
    else:
        dataset_key = "unknown_dataset"
        
    return num_suffix, dataset_key

def classify_traditional_files(src_dir: Path, dst_root: Path, move_mode=False):
    """
    递归扫描传统方法目录，按 [数据集名称/数字后缀] 的结构独立聚合
    """
    if not src_dir.is_dir():
        print(f"[ERROR] 输入源路径不存在: {src_dir}")
        return

    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"📂 开始扫描传统方法源目录: {src_dir}")
    print(f"📦 目标隔离归类目录: {dst_root}")
    print(f"🔄 运行模式: {'【剪切/移动】' if move_mode else '【复制/保留原文件】'}")
    print("-" * 60)

    success_count = 0
    skip_count = 0

    # 深度递归扫描所有文件
    for p in src_dir.rglob("*"):
        # 只处理已经经过重命名、带有前缀且以数字结尾的 png 文件
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
            
        if p.name.startswith("._") or p.name.lower() == ".ds_store":
            continue

        # 1. 解析文件名
        num_suffix, dataset_key = parse_traditional_filename(p.name)
        
        if not num_suffix:
            print(f"[WARN] 文件 '{p.name}' 结尾不是纯数字，已跳过。")
            skip_count += 1
            continue
            
        if dataset_key == "unknown_dataset":
            print(f"[WARN] 无法识别文件 '{p.name}' 对应的数据集归属，已跳过。")
            skip_count += 1
            continue

        # 2. 核心改进：构建隔离的二级子文件夹路径
        # 结构为：result_trad / 数据集名称 / 数字后缀
        target_sub_dir = dst_root / dataset_key / num_suffix
        target_sub_dir.mkdir(parents=True, exist_ok=True) # 自动级联创建

        # 3. 拼接最终的目标路径
        target_file_path = target_sub_dir / p.name

        try:
            if move_mode:
                shutil.move(str(p), str(target_file_path))
            else:
                shutil.copy2(str(p), str(target_file_path))
                
            print(f"归类成功: {p.name}  ==>  {dst_root.name}/{dataset_key}/{num_suffix}/")
            success_count += 1
        except Exception as e:
            print(f"[FAILED] 处理文件 {p.name} 时发生错误: {e}")

    print("-" * 60)
    print(f"🎉 归类完成！共成功隔离聚合了 {success_count} 个传统方法文件，跳过 {skip_count} 个。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将传统方法结果按[数据集/数字后缀]进行相互隔离的聚合")
    
    parser.add_argument('--src', '-s', type=Path, default=Path("result/traditional"),
                        help="重命名后的传统方法根目录 (默认: result/traditional)")
    
    parser.add_argument('--dst', '-d', type=Path, default=Path("result_trad"),
                        help="分类后存放的根目录 (默认: result_trad)")
    
    parser.add_argument('--move', action='store_true',
                        help="加入此参数将使用【剪切】而非【复制】")

    args = parser.parse_args()
    classify_traditional_files(args.src.resolve(), args.dst.resolve(), move_mode=args.move)