#!/usr/bin/env python3
r"""DNG (RAW) 批量转 PNG 脚本 —— Retinexformer 自定义数据预处理

功能概述
--------
Designed for computational photography workflows (e.g. Retinexformer) where the
demosaiced RGB must preserve the original under-exposure and sensor noise.

递归扫描指定目录下的 .dng 文件，经 rawpy 去马赛克后输出 PNG，供低光照增强
模型（Retinexformer）测试使用。转换过程严格保留欠曝特征与传感器噪声：

  - no_auto_bright=True   禁用自动提亮
  - use_camera_wb=True    使用相机记录的白平衡
  - 默认将最长边缩放到 600 px（与 LOL-v1 实验尺度 400×600 一致）

推荐工作流
----------
  1. 本脚本：mydata/*.dng  →  mydata_png/*.png
  2. 推理脚本：mydata/inference_mydata_png.py

运行环境
--------
  conda 环境：Retinexformer（Python 3.9 + PyTorch 2）
  额外依赖：  pip install rawpy

  激活环境：
    conda activate Retinexformer

工作目录（重要）
--------------
  请在项目根目录下运行（推荐）：

    .

  脚本支持绝对路径传参，因此从其他目录调用也可以，但建议始终 cd 到根目录，
  便于与项目内其他路径保持一致。

输入目录结构（示例）
------------------
  mydata\
  ├── with_targets\
  │   ├── extreme_lowlight_texture\
  │   │   ├── input\*.dng
  │   │   └── target\*.dng
  │   └── high_contrast_complex_light\
  │       ├── input\*.dng
  │       └── target\*.dng
  └── without_targets\
      └── *.dng

输出目录结构
------------
  输出会镜像输入的相对路径，仅扩展名变为 .png，例如：

  mydata\with_targets\extreme_lowlight_texture\input\1.dng
    → mydata_png\with_targets\extreme_lowlight_texture\input\1.png

常用命令（CMD，项目根目录下执行）
--------------------------------
  :: 转换整个 mydata（最常用）
  python mydata\convert_dng_to_png.py ^
    --input_dir mydata ^
    --output_dir mydata_png ^
    --overwrite

  :: 只转换无 GT 的数据
  python mydata\convert_dng_to_png.py ^
    --input_dir mydata\without_targets ^
    --output_dir mydata_png\without_targets ^
    --overwrite

  :: 只转换有 GT 的数据
  python mydata\convert_dng_to_png.py ^
    --input_dir mydata\with_targets ^
    --output_dir mydata_png\with_targets ^
    --overwrite

  :: 严格对齐 LOL-v1 尺寸 400×600（宽×高，可能改变宽高比）
  python mydata\convert_dng_to_png.py ^
    --input_dir mydata ^
    --output_dir mydata_png ^
    --target_width 600 --target_height 400 ^
    --overwrite

  :: 保留原始分辨率（不缩放，输出较慢、文件较大）
  python mydata\convert_dng_to_png.py ^
    --input_dir mydata ^
    --output_dir mydata_png_full ^
    --no_resize --bit_depth 16

参数说明
--------
  --input_dir       必填。DNG 源文件根目录（递归搜索）
  --output_dir      必填。PNG 输出根目录（自动创建）
  --bit_depth       输出位深，8 或 16，默认 8（兼容 Retinexformer）
  --max_long_edge   最长边像素，默认 600；设为 0 等同不缩放
  --target_width    精确输出宽度，须与 --target_height 同时指定
  --target_height   精确输出高度，须与 --target_width 同时指定
  --no_resize       禁用缩放，保留去马赛克后的原始分辨率
  --overwrite       覆盖已存在的 PNG；默认跳过已有文件


注意
----
  - CMD 中续行符为 ^，PowerShell 中为 `（反引号），请勿混用。
  - 会自动跳过 macOS 产生的 ._*.dng 垃圾文件。
  - 修改缩放参数后请重新转换并加 --overwrite，否则旧 PNG 会被跳过。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rawpy
from tqdm import tqdm


def collect_dng_files(input_dir: Path) -> list[Path]:
    """Recursively collect .dng files (case-insensitive), skip macOS junk."""
    files: list[Path] = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".dng":
            continue
        if path.name.startswith("._"):
            continue
        files.append(path)
    return sorted(files)


def resize_image(
    rgb: np.ndarray,
    max_long_edge: int | None,
    target_width: int | None,
    target_height: int | None,
) -> np.ndarray:
    """Resize demosaiced RGB to experiment-friendly resolution."""
    height, width = rgb.shape[:2]

    if target_width is not None and target_height is not None:
        if (width, height) == (target_width, target_height):
            return rgb
        return cv2.resize(
            rgb,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

    if not max_long_edge or max(height, width) <= max_long_edge:
        return rgb

    scale = max_long_edge / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(
        rgb,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def convert_dng_to_png(
    src_path: Path,
    dst_path: Path,
    bit_depth: int,
    max_long_edge: int | None,
    target_width: int | None,
    target_height: int | None,
) -> None:
    """Convert a single DNG file to PNG without auto-brightening."""
    with rawpy.imread(str(src_path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=bit_depth,
            output_color=rawpy.ColorSpace.sRGB,
        )

    rgb = resize_image(rgb, max_long_edge, target_width, target_height)

    if bit_depth == 16:
        if rgb.dtype != np.uint16:
            rgb = np.clip(rgb, 0, 65535).astype(np.uint16)
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(dst_path), bgr):
        raise RuntimeError(f"Failed to write PNG: {dst_path}")


def build_output_path(
    src_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    rel_path = src_path.relative_to(input_dir)
    return output_dir / rel_path.with_suffix(".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch convert DNG (RAW) images to lossless PNG while preserving "
            "under-exposure and sensor noise for low-light enhancement testing."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root directory containing DNG files (searched recursively).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Root directory for output PNG files (mirrors input structure).",
    )
    parser.add_argument(
        "--bit_depth",
        type=int,
        choices=[8, 16],
        default=8,
        help="Output PNG bit depth. Default: 8 (compatible with Retinexformer /255 loading).",
    )
    parser.add_argument(
        "--max_long_edge",
        type=int,
        default=600,
        help=(
            "Resize so the longest side equals this value while keeping aspect ratio. "
            "LOL-v1 uses 400x600; default 600 matches that scale. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=None,
        help="Optional exact output width. Must be used together with --target_height.",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=None,
        help="Optional exact output height. Must be used together with --target_width.",
    )
    parser.add_argument(
        "--no_resize",
        action="store_true",
        help="Disable resizing and keep the full demosaiced resolution.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files. By default, existing files are skipped.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    if (args.target_width is None) ^ (args.target_height is None):
        print(
            "[ERROR] --target_width and --target_height must be provided together.",
            file=sys.stderr,
        )
        return 1

    max_long_edge = None if args.no_resize else args.max_long_edge
    if max_long_edge == 0:
        max_long_edge = None

    output_dir.mkdir(parents=True, exist_ok=True)

    dng_files = collect_dng_files(input_dir)
    if not dng_files:
        print(f"[WARN] No .dng files found under: {input_dir}")
        return 0

    failed: list[tuple[Path, str]] = []
    skipped = 0

    for src_path in tqdm(dng_files, desc="Converting DNG -> PNG", unit="file"):
        dst_path = build_output_path(src_path, input_dir, output_dir)

        if dst_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            convert_dng_to_png(
                src_path,
                dst_path,
                args.bit_depth,
                max_long_edge,
                args.target_width,
                args.target_height,
            )
        except Exception as exc:  # noqa: BLE001 - batch job should continue on failure
            failed.append((src_path, str(exc)))

    print(f"\nDone. Total: {len(dng_files)}, Skipped: {skipped}, Failed: {len(failed)}")
    if failed:
        print("\nFailed files:")
        for src_path, reason in failed:
            print(f"  - {src_path}: {reason}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
