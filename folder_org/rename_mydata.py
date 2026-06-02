import os
import argparse
from pathlib import Path

# 核心映射字典
MAPPING = {
    "FiveK": "FiveK",
    "LOLv1": "v1",
    "LOL_v2_real": "v2_real",
    "LOL_v2_synthetic": "v2_syn",
    "NTIRE": "NTIRE",
    "SDSD_indoor": "SD_in",
    "SDSD_outdoor": "SD_out",
    "SID": "SID",
    "SMID": "SMID",
    
    "out_sem": "no_sem",
    "outsem": "no_sem",
    "sem": "sem",
    
    "without_targets": "no_tar",
    "with_targets": "tar",

    "extreme_lowlight_texture": "ex",
    "high_contrast_complex_light": "hi",

    "input": "in"
}

def get_prefix_from_path(file_path: Path, root_path: Path) -> str:
    """
    计算文件相对于根目录的各级父目录名称，并将其映射转换为缩写前缀
    """
    # 获取文件相对于 root_path 的相对路径
    try:
        rel_path = file_path.relative_to(root_path)
    except ValueError:
        return ""

    parts = rel_path.parts[:-1] 
    
    prefix_segments = []
    for part in parts:
        translated = MAPPING.get(part, part)
        prefix_segments.append(translated)
        
    return "_".join(prefix_segments)

def batch_rename_by_path(root_dir: Path):
    if not root_dir.is_dir():
        print(f"[ERROR] 目标路径不存在: {root_dir}")
        return

    print(f"🚀 开始扫描并重命名目录: {root_dir}")
    print("-" * 60)
    
    success_count = 0
    skip_count = 0

    for p in root_dir.rglob("*"):
        if not p.is_file():
            continue
            
        if p.name.startswith("._") or p.name.lower() == ".ds_store":
            continue
            
        prefix = get_prefix_from_path(p, root_dir)
        if not prefix:
            continue

        filename = p.stem
        extension = p.suffix

        if filename.startswith(f"{prefix}_"):
            skip_count += 1
            continue
            
        new_name = f"{prefix}_{filename}{extension}"
        new_path = p.parent / new_name
        
        try:
            p.rename(new_path)
            print(f"成功: .../{p.parent.name}/{p.name}  ==>  {new_name}")
            success_count += 1
        except Exception as e:
            print(f"[FAILED] 无法重命名 {p.name}: {e}")
            
    print("-" * 60)
    print(f"🎉 处理完成！共成功修改 {success_count} 个文件，自动跳过 {skip_count} 个已修改/不符文件。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据文件所处的多层目录结构深度自动加缩写前缀")
    parser.add_argument('--path', '-p', type=Path, default=Path("result/mydata_png"),
                        help="包含各个数据集的根目录路径 (默认: result/mydata_png)")
    
    args = parser.parse_args()
    batch_rename_by_path(args.path.resolve())