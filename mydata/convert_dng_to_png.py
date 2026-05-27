#!/usr/bin/env python3
r"""DNG (RAW) 批量转 PNG 脚本 —— Retinexformer 自定义数据预处理

功能概述
--------
Designed for computational photography workflows (e.g. Retinexformer) where the
demosaiced RGB must preserve the original under-exposure and sensor noise.

递归扫描指定目录下的 .dng / .png / .jpg 文件并输出 PNG，供低光照增强
模型（Retinexformer）测试使用。

  - .dng：经 rawpy 去马赛克后输出 PNG（见下方 tone_mode 说明）
  - .png / .jpg / .heic：不做 RAW 处理，仅按参数缩放后保存为 PNG

DNG 转换过程严格保留欠曝特征与传感器噪声：

  - 默认 --tone_mode preserve：no_auto_bright=True，保留欠曝（适合 Retinexformer）
  - --tone_mode preview：启用 dcraw 自动提亮，接近多数 RAW 查看器
  - --tone_mode embedded：使用 DNG 内嵌预览图，最接近 Mac 预览/Finder 快览
  - use_camera_wb=True    使用相机记录的白平衡（embedded 模式除外）
  - 默认将最长边缩放到 600 px（与 LOL-v1 实验尺度 400×600 一致）

推荐工作流
----------
  1. 本脚本：mydata/*.{dng,png,jpg}  →  mydata_png/*.png
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

  mydata\with_targets\extreme_lowlight_texture\input\1.jpg
    → mydata_png\with_targets\extreme_lowlight_texture\input\1.png  （仅缩放）

常用命令（CMD，项目根目录下执行）
--------------------------------
  :: 转换整个 mydata（最常用，保留欠曝，供 Retinexformer）
  python mydata\convert_dng_to_png.py ^
    --input_dir mydata ^
    --output_dir mydata_png ^
    --overwrite

  :: 让 PNG 亮度接近 Mac/系统直接预览 DNG 的效果
  python mydata\convert_dng_to_png.py ^
    --input_dir mydata ^
    --output_dir mydata_png ^
    --tone_mode preview ^
    --overwrite

  :: 使用 DNG 内嵌预览（与 Finder 快览最接近，分辨率可能较低）
  python mydata\convert_dng_to_png.py ^
    --input_dir mydata ^
    --output_dir mydata_png ^
    --tone_mode embedded ^
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
  --input_dir       必填。源图像根目录（递归搜索 .dng / .png / .jpg / .heic）
  --output_dir      必填。PNG 输出根目录（自动创建）
  --tone_mode       色调模式：preserve（默认）/ preview / embedded
  --bit_depth       输出位深，8 或 16，默认 8（兼容 Retinexformer）
  --max_long_edge   最长边像素，默认 600；设为 0 等同不缩放
  --target_width    精确输出宽度，须与 --target_height 同时指定
  --target_height   精确输出高度，须与 --target_width 同时指定
  --no_resize       禁用缩放，保留去马赛克后的原始分辨率
  --overwrite       覆盖已存在的 PNG；默认跳过已有文件


注意
----
  - CMD 中续行符为 ^，PowerShell 中为 `（反引号），请勿混用。
  - 会自动跳过 macOS 产生的 ._ 开头垃圾文件。
  - --tone_mode / --bit_depth 仅对 .dng 生效；其余格式只缩放，保留原位深。
  - iPhone 导出的 HEIC 若误命名为 .png，macOS 会自动用 sips 解码。
  - 修改缩放参数后请重新转换并加 --overwrite，否则旧 PNG 会被跳过。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rawpy
from tqdm import tqdm

ToneMode = Literal["preserve", "preview", "embedded"]
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".heic", ".heif"}
SUPPORTED_SUFFIXES = {".dng", *RASTER_SUFFIXES}
HEIF_BRANDS = {
    b"heic",
    b"heix",
    b"hevc",
    b"hevx",
    b"mif1",
    b"msf1",
    b"avif",
    b"av01",
}


def collect_image_files(input_dir: Path) -> list[Path]:
    """Recursively collect supported images, skip macOS junk."""
    files: list[Path] = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.name.startswith("._"):
            continue
        files.append(path)
    return sorted(files)


def detect_container_brand(src_path: Path) -> bytes | None:
    """Return ISO-BMFF brand from file header, e.g. b'heic'."""
    with src_path.open("rb") as handle:
        header = handle.read(32)
    if len(header) < 12 or header[4:8] != b"ftyp":
        return None
    return header[8:12]


def is_heif_container(src_path: Path) -> bool:
    brand = detect_container_brand(src_path)
    return brand in HEIF_BRANDS if brand is not None else False


def load_raster_via_sips(src_path: Path) -> np.ndarray:
    """Decode images with macOS built-in sips (HEIC/HEIF and other formats)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            ["sips", "-s", "format", "png", str(src_path), "--out", str(tmp_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(stderr or "sips conversion failed")

        image = cv2.imread(str(tmp_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError("sips produced an unreadable PNG")
        return image
    finally:
        tmp_path.unlink(missing_ok=True)


def load_raster_image(src_path: Path) -> np.ndarray:
    """Load PNG/JPG/HEIC raster data, with macOS sips fallback for HEIF."""
    image = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if image is not None:
        return image

    if sys.platform == "darwin":
        try:
            return load_raster_via_sips(src_path)
        except RuntimeError as exc:
            if is_heif_container(src_path):
                raise RuntimeError(
                    f"Failed to decode HEIF/HEIC image (misnamed as {src_path.suffix}?): {src_path}"
                ) from exc

    if is_heif_container(src_path):
        raise RuntimeError(
            f"File is HEIF/HEIC, not {src_path.suffix}: {src_path}. "
            "Rename to .heic or convert on macOS."
        )

    raise RuntimeError(f"Failed to read image: {src_path}")


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


def load_embedded_preview(src_path: Path) -> np.ndarray:
    """Load the embedded JPEG/bitmap preview baked into the DNG."""
    with rawpy.imread(str(src_path)) as raw:
        thumb = raw.extract_thumb()

    if thumb.format == rawpy.ThumbFormat.JPEG:
        rgb = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to decode embedded JPEG preview: {src_path}")
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if thumb.format == rawpy.ThumbFormat.BITMAP:
        rgb = np.asarray(thumb.data)
        if rgb.ndim == 2:
            return cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        if rgb.shape[2] == 4:
            return cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        return rgb

    raise RuntimeError(f"Unsupported embedded preview format: {thumb.format}")


def develop_raw_rgb(raw: rawpy.RawPy, tone_mode: ToneMode, bit_depth: int) -> np.ndarray:
    """Demosaic RAW with tone settings matching the selected preview mode."""
    if tone_mode == "preserve":
        return raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=bit_depth,
            output_color=rawpy.ColorSpace.sRGB,
        )

    return raw.postprocess(
        use_camera_wb=True,
        no_auto_bright=False,
        output_bps=bit_depth,
        output_color=rawpy.ColorSpace.sRGB,
    )


def convert_raster_to_png(
    src_path: Path,
    dst_path: Path,
    max_long_edge: int | None,
    target_width: int | None,
    target_height: int | None,
) -> None:
    """Resize an existing PNG/JPG/HEIC and save as PNG without tone processing."""
    image = load_raster_image(src_path)

    image = resize_image(image, max_long_edge, target_width, target_height)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst_path), image):
        raise RuntimeError(f"Failed to write PNG: {dst_path}")


def convert_dng_to_png(
    src_path: Path,
    dst_path: Path,
    tone_mode: ToneMode,
    bit_depth: int,
    max_long_edge: int | None,
    target_width: int | None,
    target_height: int | None,
) -> None:
    """Convert a single DNG file to PNG via rawpy."""
    if tone_mode == "embedded":
        rgb = load_embedded_preview(src_path)
    else:
        with rawpy.imread(str(src_path)) as raw:
            rgb = develop_raw_rgb(raw, tone_mode, bit_depth)

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


def convert_image_file(
    src_path: Path,
    dst_path: Path,
    tone_mode: ToneMode,
    bit_depth: int,
    max_long_edge: int | None,
    target_width: int | None,
    target_height: int | None,
) -> None:
    suffix = src_path.suffix.lower()
    if suffix == ".dng":
        convert_dng_to_png(
            src_path,
            dst_path,
            tone_mode,
            bit_depth,
            max_long_edge,
            target_width,
            target_height,
        )
        return

    if suffix in RASTER_SUFFIXES:
        convert_raster_to_png(
            src_path,
            dst_path,
            max_long_edge,
            target_width,
            target_height,
        )
        return

    raise ValueError(f"Unsupported image format: {src_path}")


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
            "Batch convert DNG/PNG/JPG/HEIC images to PNG. DNG uses rawpy development; "
            "other formats are resized only."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root directory containing DNG/PNG/JPG/HEIC files (searched recursively).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Root directory for output PNG files (mirrors input structure).",
    )
    parser.add_argument(
        "--tone_mode",
        choices=["preserve", "preview", "embedded"],
        default="preserve",
        help=(
            "DNG only. preserve=keep under-exposure for Retinexformer; "
            "preview=auto-bright demosaic like most RAW viewers; "
            "embedded=use in-file preview JPEG (closest to macOS Preview)."
        ),
    )
    parser.add_argument(
        "--bit_depth",
        type=int,
        choices=[8, 16],
        default=8,
        help="DNG output bit depth. Default: 8 (compatible with Retinexformer /255 loading).",
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

    tone_mode: ToneMode = args.tone_mode
    if tone_mode != "preserve":
        print(f"Tone mode: {tone_mode} (brighter, closer to system DNG preview)")

    image_files = collect_image_files(input_dir)
    if not image_files:
        print(f"[WARN] No supported images found under: {input_dir}")
        return 0

    failed: list[tuple[Path, str]] = []
    skipped = 0

    for src_path in tqdm(image_files, desc="Converting -> PNG", unit="file"):
        dst_path = build_output_path(src_path, input_dir, output_dir)

        if dst_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            convert_image_file(
                src_path,
                dst_path,
                tone_mode,
                args.bit_depth,
                max_long_edge,
                args.target_width,
                args.target_height,
            )
        except Exception as exc:  # noqa: BLE001 - batch job should continue on failure
            failed.append((src_path, str(exc)))

    print(f"\nDone. Total: {len(image_files)}, Skipped: {skipped}, Failed: {len(failed)}")
    if failed:
        print("\nFailed files:")
        for src_path, reason in failed:
            print(f"  - {src_path}: {reason}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
