#!/usr/bin/env python3
r"""Retinexformer 自定义 PNG 数据批量推理脚本

功能概述
--------
对 convert_dng_to_png.py 生成的 mydata_png 目录中的低光照图像，使用
Retinexformer（LOL-v1 预训练权重）进行增强推理，输出增强后的 PNG。

  - 自动跳过 target/、gt/ 目录（仅对 input 或无 GT 的图像推理）
  - 支持 8-bit / 16-bit PNG 输入
  - 大图（边长 ≥ 3000）自动分块推理
  - 若存在配对 GT（input/1.png ↔ target/1.png），自动计算 PSNR / SSIM

推荐工作流
----------
  1. 预处理：mydata/convert_dng_to_png.py   （DNG → PNG）
  2. 本脚本：  mydata/inference_mydata_png.py （PNG → 增强结果）

运行环境
--------
  conda 环境：Retinexformer（需已安装 basicsr，见项目 README）
  预训练权重：pretrained_weights/LOL_v1.pth

  激活环境：
    conda activate Retinexformer

  若报 ModuleNotFoundError: No module named 'basicsr'，在项目根目录执行：
    pip install -e . --no-build-isolation --config-settings "--global-option=--no_cuda_ext"

工作目录（重要）
--------------
  请在项目根目录下运行（推荐）：

    .

  脚本会自动将项目根目录加入 sys.path，因此必须从根目录运行，或保证
  Options/、pretrained_weights/、basicsr/ 等相对路径可访问。
  所有路径参数均支持绝对路径。

输入目录结构（示例）
------------------
  mydata_png\
  ├── without_targets\
  │   └── *.png                  ← 会被推理
  └── with_targets\
      ├── extreme_lowlight_texture\
      │   ├── input\*.png        ← 会被推理
      │   └── target\*.png       ← 跳过（仅用于评估）
      └── high_contrast_complex_light\
          ├── input\*.png
          └── target\*.png

输出目录结构
------------
  输出镜像输入相对路径，例如：

  mydata_png\without_targets\1.png
    → results\mydata_png\without_targets\1.png

常用命令（CMD，项目根目录下执行）
--------------------------------
  :: 默认推理（使用内置默认路径，最简单）
  python mydata\inference_mydata_png.py

  :: 显式指定输入/输出路径
  python mydata\inference_mydata_png.py ^
    --input_dir mydata_png ^
    --output_dir results\mydata_png ^
    --weights pretrained_weights\LOL_v1.pth ^
    --opt Options\RetinexFormer_LOL_v1.yml ^
    --gpus 0

  :: 启用自集成（效果更好，速度约为 8 倍慢）
  python mydata\inference_mydata_png.py --self_ensemble

  :: 若 conda activate 后 python 路径不对，可用 conda run 指定环境
  conda run -n Retinexformer python mydata\inference_mydata_png.py

参数说明
--------
  --input_dir       PNG 输入根目录，默认 mydata_png/
  --output_dir      增强结果输出目录，默认 results/mydata_png/
  --opt             模型配置 YAML，默认 Options/RetinexFormer_LOL_v1.yml
  --weights         预训练权重，默认 pretrained_weights/LOL_v1.pth
  --gpus            CUDA GPU 编号，默认 0；Mac 上自动使用 MPS，否则回退 CPU
  --self_ensemble   启用自集成测试策略（可选，更慢）

注意
----
  - CMD 中续行符为 ^，PowerShell 中为 `（反引号），请勿混用。
  - 请写成一行命令，或正确使用续行符，避免只执行第一行。
  - 使用 convert_dng_to_png.py 默认缩放（最长边 600）后，推理速度远快于 4K 原图。
  - 有 GT 的子集会打印平均 PSNR / SSIM；without_targets 仅保存增强图。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from natsort import natsorted
from skimage import img_as_ubyte
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Enhancement"))

import utils  # noqa: E402
from basicsr.models import create_model  # noqa: E402
from basicsr.utils.options import parse  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FACTOR = 4
SPLIT_THRESHOLD = 3000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Retinexformer inference on mydata_png."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=ROOT / "mydata_png",
        help="Root directory of converted PNG images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "results" / "mydata_png",
        help="Directory to save enhanced images.",
    )
    parser.add_argument(
        "--opt",
        type=Path,
        default=ROOT / "Options" / "RetinexFormer_LOL_v1.yml",
        help="Model config YAML.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "pretrained_weights" / "LOL_v1.pth",
        help="Pretrained checkpoint path.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="CUDA device id(s), e.g. 0 (ignored on Mac MPS / CPU)",
    )
    parser.add_argument(
        "--self_ensemble",
        action="store_true",
        help="Use self-ensemble for better results (slower).",
    )
    return parser.parse_args()


def resolve_device(gpus: str) -> torch.device:
    """Select device: CUDA (unchanged) > Apple MPS > CPU."""
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def collect_input_images(input_dir: Path) -> list[Path]:
    """Collect inference inputs, skipping GT/target folders."""
    images: list[Path] = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = {part.lower() for part in path.parts}
        if "target" in parts or "gt" in parts:
            continue
        if path.name.startswith("._"):
            continue
        images.append(path)
    return natsorted(images)


def target_path_for_input(input_path: Path) -> Path | None:
    """Map .../input/foo.png -> .../target/foo.png when target exists."""
    parts = list(input_path.parts)
    lowered = [part.lower() for part in parts]
    if "input" not in lowered:
        return None
    idx = lowered.index("input")
    parts[idx] = "target"
    candidate = Path(*parts)
    return candidate if candidate.is_file() else None


def load_image_normalized(image_path: Path) -> np.ndarray:
    """Load PNG/JPG and normalize to [0, 1], supporting 8-bit and 16-bit."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    return image.astype(np.float32) / 255.0


def self_ensemble(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x

    outputs = []
    for hflip in (False, True):
        for vflip in (False, True):
            for rotate in (False, True):
                outputs.append(forward_transformed(x, hflip, vflip, rotate, model))
    return torch.mean(torch.stack(outputs), dim=0)

#111111:去红绿噪声
def chroma_denoise_post_process(img_rgb_float):
    """
    针对低光增强后残留红绿彩噪的即插即用后处理插件
    - 输入: img_rgb_float (0.0 ~ 1.0 的 NumPy 矩阵, RGB 顺序)
    - 原理: 转换到 Lab 空间，保持代表亮度的 L 通道不改变，只对代表色彩的 a, b 通道进行双边滤波
    """
    img_uint8 = (np.clip(img_rgb_float, 0, 1) * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 对 a (红绿) 和 b (黄蓝) 色彩通道执行高保真双border滤波
    a_denoised = cv2.bilateralFilter(a_channel, d=9, sigmaColor=25, sigmaSpace=15)
    b_denoised = cv2.bilateralFilter(b_channel, d=9, sigmaColor=25, sigmaSpace=15)

    lab_processed = cv2.merge((l_channel, a_denoised, b_denoised))
    bgr_processed = cv2.cvtColor(lab_processed, cv2.COLOR_Lab2BGR)
    rgb_processed = cv2.cvtColor(bgr_processed, cv2.COLOR_BGR2RGB)

    return rgb_processed.astype(np.float32) / 255.0

#11111:清晰化函数
def texture_sharpen_post_process(img_rgb_float, sigma=1.0, strength=0.8):
    """
    针对低光增强后画面模糊、纹理涂抹的即插即用清晰化插件 (USM 方案)
    - 输入: img_rgb_float (0.0 ~ 1.0 的 NumPy 矩阵, RGB 顺序)
    - sigma: 控制清晰化的纹理尺度 (越大则越偏向粗边缘, 越小则偏向微小毛刺)
    - strength: 增强强度，通常在 0.3 ~ 1.2 之间。过大会产生白边（Halo 效应）
    """
    # 转为 uint8 以保证 OpenCV 兼容性
    img_uint8 = (np.clip(img_rgb_float, 0, 1) * 255.0).astype(np.uint8)
    
    # 转换到 YCrCb 颜色空间，只对代表亮度的 Y 通道做清晰化，避免放大色彩噪声
    ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)
    
    # 执行 Unsharp Masking (USM) 算法
    # 1. 提取低频平滑图
    gaussian = cv2.GaussianBlur(y_channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # 2. 混合公式：输出 = 原图 + strength * (原图 - 模糊图)
    y_sharpened = cv2.addWeighted(y_channel, 1.0 + strength, gaussian, -strength, 0)
    
    # 合并通道并转回 RGB
    ycrcb_processed = cv2.merge((y_sharpened, cr, cb))
    rgb_processed = cv2.cvtColor(ycrcb_processed, cv2.COLOR_YCrCb2RGB)
    
    return rgb_processed.astype(np.float32) / 255.0

def run_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
    use_self_ensemble: bool,
) -> torch.Tensor:
    _, _, h, w = input_tensor.shape
    pad_h = (FACTOR - h % FACTOR) % FACTOR
    pad_w = (FACTOR - w % FACTOR) % FACTOR
    input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    if h < SPLIT_THRESHOLD and w < SPLIT_THRESHOLD:
        if use_self_ensemble:
            restored = self_ensemble(input_tensor, model)
        else:
            restored = model(input_tensor)
    else:
        input_1 = input_tensor[:, :, :, 1::2]
        input_2 = input_tensor[:, :, :, 0::2]
        if use_self_ensemble:
            restored_1 = self_ensemble(input_1, model)
            restored_2 = self_ensemble(input_2, model)
        else:
            restored_1 = model(input_1)
            restored_2 = model(input_2)
        restored = torch.zeros_like(input_tensor)
        restored[:, :, :, 1::2] = restored_1
        restored[:, :, :, 0::2] = restored_2

    return torch.clamp(restored[:, :, :h, :w], 0, 1)


def build_model(opt_path: Path, weights_path: Path, device: torch.device) -> nn.Module:
    opt = parse(str(opt_path), is_train=False)
    opt["dist"] = False
    # basicsr 根据 num_gpu 决定是否走 CUDA；非 CUDA 时必须置 0，避免 Mac 上触发 cuda
    if device.type != "cuda":
        opt["num_gpu"] = 0

    yaml_cfg = yaml.load(open(opt_path, mode="r", encoding="utf-8"), Loader=yaml.SafeLoader)
    yaml_cfg["network_g"].pop("type", None)

    model = create_model(opt).net_g
    checkpoint = torch.load(weights_path, map_location="cpu")

    try:
        #11111
        #model.load_state_dict(checkpoint["params"])
    # ----------- 【终极兼容：清洗多卡前缀 + 自动过滤 LoRA 冗余键】 -----------
        from collections import OrderedDict
        
        raw_state_dict = checkpoint["params"] if "params" in checkpoint else checkpoint
        state_dict = OrderedDict()
        
        for k, v in raw_state_dict.items():
            # 1. 自动移除多卡训练生成的 module. 前缀
            name = k[7:] if k.startswith('module.') else k
            
            # 2. 如果本地网络是原版结构（不支持 LoRA），则跳过权重里的 lora 辅助层
            if 'lora' in name:
                continue
                
            state_dict[name] = v
            
        # 使用 strict=False 允许平稳加载基础权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"提示：网络缺失了以下基础键 (可能影响效果): {missing_keys[:3]}... (共{len(missing_keys)}个)")
        if unexpected_keys:
            print(f"提示：权重中含有未匹配的冗余键: {unexpected_keys[:3]}... (共{len(unexpected_keys)}个)")
        # ----------------------------------------------------------------------

    except RuntimeError:
        state_dict = {"module." + k: v for k, v in checkpoint["params"].items()}
        model.load_state_dict(state_dict)

    model.to(device)
    if device.type == "cuda":
        model = nn.DataParallel(model)
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        print(f"[ERROR] input_dir not found: {input_dir}")
        return 1
    if not args.weights.is_file():
        print(f"[ERROR] weights not found: {args.weights}")
        return 1
    if not args.opt.is_file():
        print(f"[ERROR] config not found: {args.opt}")
        return 1

    device = resolve_device(args.gpus)
    print(f"Using device: {device}")
    print(f"Loading weights: {args.weights}")

    image_paths = collect_input_images(input_dir)
    if not image_paths:
        print(f"[WARN] No input images found under: {input_dir}")
        return 0

    model = build_model(args.opt, args.weights, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    psnr_values: list[float] = []
    ssim_values: list[float] = []

    with torch.inference_mode():
        for image_path in tqdm(image_paths, desc="Enhancing", unit="img"):
            rel_path = image_path.relative_to(input_dir)
            save_path = output_dir / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            image = load_image_normalized(image_path)
            input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)

            restored = run_model(model, input_tensor, args.self_ensemble)
            restored_np = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
            #111111：去红绿噪声
            restored_np = chroma_denoise_post_process(restored_np)
            #restored_np = texture_sharpen_post_process(restored_np, sigma=0.5, strength=0.8)
            
            utils.save_img(str(save_path), img_as_ubyte(restored_np))

            gt_path = target_path_for_input(image_path)
            if gt_path is not None:
                target = load_image_normalized(gt_path)
                psnr_values.append(utils.PSNR(target, restored_np))
                ssim_values.append(
                    utils.calculate_ssim(
                        img_as_ubyte(target),
                        img_as_ubyte(restored_np),
                    )
                )

            empty_device_cache(device)

    print(f"\nSaved {len(image_paths)} enhanced image(s) to: {output_dir}")
    if psnr_values:
        print(f"PSNR (with GT): {np.mean(psnr_values):.4f}")
        print(f"SSIM (with GT): {np.mean(ssim_values):.4f}")
        print(f"Evaluated pairs: {len(psnr_values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
