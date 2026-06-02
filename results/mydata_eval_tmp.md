# mydata 评测结果（临时）

> 生成时间：2026-06-02  
> 环境：`Retinexformer` conda 环境  
> 模型：LOL_v1 / LOL_v2_real（预训练，不含 LoRA）  
> 推理：无 Self-Ensemble  
> 预处理：DNG/JPG → `mydata_png/`（最长边 600，DNG 保留欠曝）

## 实验设置

| 项目 | 配置 |
|---|---|
| 环境 | `conda run -n Retinexformer` |
| 模型 1 | `LOL_v1.pth` + `Options/RetinexFormer_LOL_v1.yml` |
| 模型 2 | `LOL_v2_real.pth` + `Options/RetinexFormer_LOL_v2_real.yml` |
| 评测样本 | 14 对有 GT（极限暗光 9 对 + 复杂光源 5 对） |

---

## 1. mydata（DNG）

### 全量 GT 均值（14 对）

| 模型 | PSNR | SSIM |
|---|---|---|
| **LOL_v1** | **15.81** | **0.605** |
| **LOL_v2_real** | 14.46 | 0.602 |

### 分场景

| 场景 | LOL_v1 PSNR (SSIM) | LOL_v2_real PSNR (SSIM) |
|---|---|---|
| 极限暗光（n=9） | 13.65 (0.559) | **14.03 (0.590)** |
| 复杂光源（n=5） | **19.69 (0.687)** | 15.25 (0.623) |

输出目录：

- `results/mydata_png/dng/`（LOL_v1）
- `results/mydata_png_lolv2_real/dng/`（LOL_v2_real）

---

## 2. mydata（JPG）

### 全量 GT 均值（14 对）

| 模型 | PSNR | SSIM |
|---|---|---|
| **LOL_v1** | **15.58** | 0.649 |
| **LOL_v2_real** | 15.52 | **0.681** |

### 分场景

| 场景 | LOL_v1 PSNR (SSIM) | LOL_v2_real PSNR (SSIM) |
|---|---|---|
| 极限暗光（n=9） | 15.01 (0.642) | **15.85 (0.670)** |
| 复杂光源（n=5） | **16.59 (0.662)** | 14.91 (0.701) |

输出目录：

- `results/mydata_png/jpg/`（LOL_v1）
- `results/mydata_png_lolv2_real/jpg/`（LOL_v2_real）

---

## 简要结论

1. **JPG**：LOL_v2_real 整体 SSIM 更高（0.681 vs 0.649），极限暗光 PSNR/SSIM 也更高；LOL_v1 在复杂光源 PSNR 更高（16.59 vs 14.91）。
2. **DNG**：LOL_v1 整体 PSNR 更高（15.81 vs 14.46），复杂光源优势明显（19.69 vs 15.25）；极限暗光 LOL_v2_real 略好。
3. DNG 与 JPG 表现差异较大，说明 RAW 解码与 JPG 直出存在 domain gap。

---

## 复现命令

```powershell
# DNG + LOL_v1
conda run -n Retinexformer python mydata/inference_mydata_png.py --input_dir mydata_png/dng --output_dir results/mydata_png/dng --weights pretrained_weights/LOL_v1.pth --opt Options/RetinexFormer_LOL_v1.yml

# DNG + LOL_v2_real
conda run -n Retinexformer python mydata/inference_mydata_png.py --input_dir mydata_png/dng --output_dir results/mydata_png_lolv2_real/dng --weights pretrained_weights/LOL_v2_real.pth --opt Options/RetinexFormer_LOL_v2_real.yml

# JPG + LOL_v1
conda run -n Retinexformer python mydata/inference_mydata_png.py --input_dir mydata_png/jpg --output_dir results/mydata_png/jpg --weights pretrained_weights/LOL_v1.pth --opt Options/RetinexFormer_LOL_v1.yml

# JPG + LOL_v2_real
conda run -n Retinexformer python mydata/inference_mydata_png.py --input_dir mydata_png/jpg --output_dir results/mydata_png_lolv2_real/jpg --weights pretrained_weights/LOL_v2_real.pth --opt Options/RetinexFormer_LOL_v2_real.yml

# 分场景评估
conda run -n Retinexformer python mydata/eval_mydata_by_scenario.py --results_root results --gt_root mydata_png --fmt dng
conda run -n Retinexformer python mydata/eval_mydata_by_scenario.py --results_root results --gt_root mydata_png --fmt jpg
```
