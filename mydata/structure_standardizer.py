#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集规范化整理脚本: 适配方案二（多场景独立子集结构）

功能概述
--------
本脚本专门用于将自定义的 `mydata` 原始结构自动整理并划分到 `mydata_standard` 目录中。
完全适配方案二的要求，保持多场景独立：

  mydata_standard/
  ├─ extreme_lowlight/
  │  ├─ Train/
  │  │  ├─ input/
  │  │  └─ target/
  │  └─ Test/
  │     ├─ input/
  │     └─ target/
  └─ high_contrast/
     ├─ Train/
     │  ├─ input/
     │  └─ target/
     └─ Test/
        ├─ input/
        └─ target/

划分机制
--------
  1. 对于 with_targets 下的两个场景：
     - 将原本的 input/ 和 target/ 中的文件按指定的比例（默认 85% 训练，15% 测试）
       随机且同步地划分到对应场景的 Train/ 和 Test/ 中。
     - 确保 input 和 target 中的配对文件名完全一致、严格对齐。
  2. 自动创建所有级联的多层文件夹。

运行方式
--------
  在项目根目录下直接运行（或通过 python 解释器执行）：
    python mydata/structure_standardizer.py

参数说明
--------
  --src_dir       原始 mydata 路径，默认为 './mydata'
  --dst_dir       目标整理路径，默认为 './mydata_standard'
  --train_ratio   训练集所占比例，默认为 0.85 (即 85% 训练，15% 测试)
  --seed          随机数种子，默认为 42，确保可复现性
"""

import argparse
import shutil
import random
from pathlib import Path

def setup_args():
    parser = argparse.ArgumentParser(description="将 mydata 自动整理并划分为方案二的标准格式")
    parser.add_argument("--src_dir", type=Path, default=Path("./mydata"), help="原始 mydata 目录路径")
    parser.add_argument("--dst_dir", type=Path, default=Path("./data/mydata_standard"), help="生成的标准数据集目录路径")
    parser.add_argument("--train_ratio", type=float, default=0.85, help="训练集划分比例 (0.0 到 1.0 之间)")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子，用于保证划分结果可复现")
    return parser.parse_args()

def clean_and_sort_files(folder_path: Path) -> list[Path]:
    """收集有效的图片/数据文件，跳过系统隐藏文件"""
    if not folder_path.exists():
        return []
    valid_suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.dng'}
    files = []
    for p in folder_path.iterdir():
        if p.is_file() and p.suffix.lower() in valid_suffixes and not p.name.startswith("._"):
            files.append(p)
    return sorted(files)

def split_and_copy_scene(src_scene_path: Path, dst_scene_root: Path, train_ratio: float):
    """对单个场景的数据进行同步的 Train/Test 随机划分与复制"""
    src_input_dir = src_scene_path / "input"
    src_target_dir = src_scene_path / "target"

    if not src_input_dir.exists() or not src_target_dir.exists():
        print(f"[WARN] 场景 {src_scene_path.name} 缺失 input 或 target 子文件夹，跳过。")
        return

    # 收集输入和对应的标签
    input_files = clean_and_sort_files(src_input_dir)
    
    if not input_files:
        print(f"[WARN] 场景 {src_scene_path.name}/input 下未找到有效文件。")
        return

    # 过滤出严格配对的 target 文件
    paired_pairs = []
    for in_file in input_files:
        # 在 target 中寻找同名文件
        target_candidates = list(src_target_dir.glob(f"{in_file.stem}.*"))
        # 过滤掉 macOS 临时文件
        target_candidates = [c for c in target_candidates if not c.name.startswith("._")]
        
        if target_candidates:
            paired_pairs.append((in_file, target_candidates[0]))
        else:
            print(f"[WARN] 未能在 target 中找到与 {in_file.name} 配对的文件，该样本将被舍弃。")

    # 随机打乱配对序列
    random.shuffle(paired_pairs)

    # 计算切分界限
    num_samples = len(paired_pairs)
    num_train = int(num_samples * train_ratio)

    train_pairs = paired_pairs[:num_train]
    test_pairs = paired_pairs[num_train:]

    # 定义目标子目录
    dirs_to_create = [
        dst_scene_root / "Train" / "input",
        dst_scene_root / "Train" / "target",
        dst_scene_root / "Test" / "input",
        dst_scene_root / "Test" / "target"
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    # 复制训练集
    for in_p, tg_p in train_pairs:
        shutil.copy2(in_p, dst_scene_root / "Train" / "input" / in_p.name)
        shutil.copy2(tg_p, dst_scene_root / "Train" / "target" / tg_p.name)

    # 复制测试集
    for in_p, tg_p in test_pairs:
        shutil.copy2(in_p, dst_scene_root / "Test" / "input" / in_p.name)
        shutil.copy2(tg_p, dst_scene_root / "Test" / "target" / tg_p.name)

    print(f"场景 [{src_scene_path.name}] 划分完成:")
    print(f"   - 总配对样本数: {num_samples}")
    print(f"   - 划分至 Train : {len(train_pairs)} 对 (写入 -> {dst_scene_root / 'Train'})")
    print(f"   - 划分至 Test  : {len(test_pairs)} 对 (写入 -> {dst_scene_root / 'Test'})")

def main():
    args = setup_args()
    
    # 设置随机种子，保证每次划分结果一致
    random.seed(args.seed)

    src_root = args.src_dir.resolve()
    dst_root = args.dst_dir.resolve()

    print("\n" + "="*60)
    print(" 开始执行数据集标准格式整理 (方案二: 多场景独立子集)")
    print(f" 原始目录: {src_root}")
    print(f" 目标目录: {dst_root}")
    print(f" 划分比例: Train = {args.train_ratio*100:.1f}%, Test = {(1-args.train_ratio)*100:.1f}%")
    print("="*60)

    with_targets_dir = src_root / "with_targets"
    if not with_targets_dir.is_dir():
        print(f"[ERROR] 未能在原始目录下找到 'with_targets' 文件夹: {with_targets_dir}")
        return 1

    # 1. 映射并处理 extreme_lowlight_texture
    src_extreme = with_targets_dir / "extreme_lowlight_texture"
    dst_extreme = dst_root / "extreme_lowlight"
    if src_extreme.is_dir():
        split_and_copy_scene(src_extreme, dst_extreme, args.train_ratio)
    else:
        print(f"[INFO] 未检测到场景: {src_extreme.name}")

    # 2. 映射并处理 high_contrast_complex_light
    src_contrast = with_targets_dir / "high_contrast_complex_light"
    dst_contrast = dst_root / "high_contrast"
    if src_contrast.is_dir():
        split_and_copy_scene(src_contrast, dst_contrast, args.train_ratio)
    else:
        print(f"[INFO] 未检测到场景: {src_contrast.name}")

    print("\n🎉 所有配对场景已整理并切分完毕！符合方案二标准格式。")
    print("="*60 + "\n")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())