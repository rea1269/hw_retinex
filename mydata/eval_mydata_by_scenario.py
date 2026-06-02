#!/usr/bin/env python3
"""Compute per-scenario PSNR/SSIM for mydata_png inference outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage import img_as_ubyte

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Enhancement"))
import utils  # noqa: E402

SCENARIOS = {
    "extreme_lowlight_texture": "极限暗光",
    "high_contrast_complex_light": "复杂光源",
}


def load_norm(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    return image.astype(np.float32) / 255.0


def eval_result_dir(result_dir: Path, gt_root: Path, fmt: str = "jpg") -> dict[str, tuple[float, float, int]]:
    stats: dict[str, tuple[float, float, int]] = {}
    for scen in SCENARIOS:
        target_dir = gt_root / fmt / "with_targets" / scen / "target"
        psnrs: list[float] = []
        ssims: list[float] = []
        for gt_path in sorted(target_dir.glob("*.png")):
            pred = result_dir / fmt / "with_targets" / scen / "input" / gt_path.name
            if not pred.is_file():
                continue
            gt = load_norm(gt_path)
            pred_img = load_norm(pred)
            psnrs.append(utils.PSNR(gt, pred_img))
            ssims.append(utils.calculate_ssim(img_as_ubyte(gt), img_as_ubyte(pred_img)))
        if psnrs:
            stats[scen] = (float(np.mean(psnrs)), float(np.mean(ssims)), len(psnrs))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=ROOT / "results")
    parser.add_argument("--gt_root", type=Path, default=ROOT / "mydata_png")
    parser.add_argument("--fmt", default="jpg", choices=("jpg", "dng"))
    args = parser.parse_args()

    dirs = sorted(p for p in args.results_root.glob("mydata_png*") if p.is_dir())
    if not dirs:
        print("No results/mydata_png* directories found.", file=sys.stderr)
        return 1

    for result_dir in dirs:
        print(f"\n[{result_dir.name}]")
        stats = eval_result_dir(result_dir, args.gt_root, args.fmt)
        for scen, label in SCENARIOS.items():
            if scen not in stats:
                print(f"  {label}: (no pairs)")
                continue
            psnr, ssim, n = stats[scen]
            print(f"  {label}: PSNR={psnr:.2f} SSIM={ssim:.3f} (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
