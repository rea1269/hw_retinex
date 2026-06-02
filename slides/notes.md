# Retinexformer

# 复现（实现了5个数据集）

名词解释：

**参数self\-ensemble**: We add the self\-ensemble strategy in the testing code to derive better results\.

**PSNR:峰值信噪比**

越大越好：PSNR 的单位是分贝（dB），数值越大，说明图像失真越小，噪点越少，画质越纯净。

数值范围含义：

- 低于 20 dB：图像质量非常差，提亮后满是彩噪或严重偏色。

- 20 \~ 30 dB：处于可以接受的范围，也是大多数传统算法或基础轻量模型在极暗场景下的表现。

- 高于 30 dB：图像质量非常好，肉眼几乎察觉不到它和原版清晰大片的像素级差异。

**SSIM:结构相似性指数**

越大越好：SSIM 的取值范围严格在 $[0, 1]$ 之间。数值越接近 1，代表两张图片的结构越相似，视觉效果越逼真、越自然。

数值范围含义：

- 低于 0\.7：物体边缘模糊、线条扭曲或者出现了大面积的人工伪影。

- 0\.8 \~ 0\.9：大部分物体的轮廓和细节得到了恢复。

- 高于 0\.95：两张图的纹理结构几乎完全一致，达到了高保真级别。

||No self\_ensemble<br>PSNR|No self\_ensemble<br>SSIM|self\_ensemble<br>PSNR|self\_ensemble<br>SSIM|
|---|---|---|---|---|
|LOL\_V1|25\.154999|0\.845462|25\.265992|0\.853085|
|LOL\_V2\_real|22\.795579|0\.839678|22\.821343|0\.849842|
|LOL\_V2\_syn|25\.672992|0\.929550|25\.830671|0\.934126|
|FiveK|24\.937761|0\.906930|24\.967888|0\.908813|
|SID|24\.439999|0\.680029|24\.359135|0\.680809|
|SMID|太大了，不下了||||
|SDSD\-indoor|文件损坏||||
|SDSD\-outdoor|太大了，不下了||||
|NTIRE 2024|数据链接无法下载||||



# 改进

### 非神经网络的传统方法对比

#### Gamma 校正（Gamma Correction）——全局亮度映射

基本原理：Gamma校正是一种全局非线性变换。它通过一个幂函数公式来改变图像中每个像素的灰度值：

$O = \left(\frac{I}{255}\right)^\gamma \times 255$  其中 $I$ 是输入像素值，$O$ 是输出像素值

工作机制：

- 当 $\gamma < 1$ 时，该曲线在暗区（低像素值）的斜率非常大。这意味着它会拉伸暗部的灰度范围，把原本聚集在极暗区域的像素强行推向亮区，从而实现整体提亮。

特点与局限：计算速度极快，但因为是全局一刀切，缺乏对图像内容的感知。如果盲目整体提亮，原本就比较亮的地方容易过曝（泛白失真），并且隐藏在暗处的硬件噪声也会被等比例放大。

#### CLAHE（限制对比度自适应直方图均衡化）—— 局部对比度自适应

- 基本原理：传统的直方图均衡化（HE）是在全局拉伸亮度，容易导致图像失真。CLAHE 在其基础上做了两项重大改进：自适应（Adaptive）和限制对比度（Contrast Limiting）。

- 工作机制：

    1. 分块处理（自适应）：它不看整张图，而是把图像分成许多不重叠的小网格（如 8×8 的 `tile\_grid\_size`）。在每个小网格内部独立计算直方图并进行均衡化，这使得它能很好地提取局部细节。

    2. 限制对比度（裁剪阈值）：如果一个小网格内的像素颜色太单一，直方图就会出现一个极高的尖峰，直接均衡化会导致该区域对比度爆炸、噪点严重。CLAHE 设置了一个裁剪阈值（`clip\_limit`），把超过阈值的尖峰像素“切掉”，并均匀分散到其他灰度级中，从而抑制了噪声的过度放大。

    3. 双线性插值：最后通过插值算法消除块与块之间的边界效应，让图像过渡自然。

- 特点与局限：在保持图像色彩（在 HSV 空间操作）和提升局部纹理细节方面表现极佳。但它的核心是“拉开明暗边界”，如果输入图片整体都极其黑暗（如 LOL 数据集），在限制了对比度放大后，它能借调的“亮部像素”太少，导致整体提亮幅度非常有限。

#### Gamma \+ CLAHE 结合 —— 粗细结合的协同增强

- 基本原理：这是一种“全局粗调 \+ 局部微调”的串联级联架构，旨在结合前两者的优势，互补其短板。

- 工作机制：

    1. 第一阶段（Gamma 负责打底）：先利用 Gamma 校正进行全局粗暴提亮。这一步的意义在于把原本锁死在 `\[0, 30\]` 极窄范围内的暗部像素，整体拉伸拓展到 `\[0, 150\]` 的中高光照区间。虽然此时图像可能看起来色彩寡淡、灰蒙蒙的，但它成功为下一步开辟了“施展空间”。

    2. 第二阶段（CLAHE 负责润色）：将 Gamma 提亮后的图像转入 HSV 空间，此时亮度通道（V 通道）已经有了丰富的动态范围。CLAHE 此时登场，针对这些被拉开的新像素进行局部自适应调整，把因为粗暴提亮而变得扁平、模糊的细节和纹理重新剥离出来，拉开局部对比度。

- 特点：针对 LOL 这种极端暗光的硬核场景，组合拳的效果显著优于前两者。它既实现了传统算法所能达到的最大提亮幅度和视野可见度（由 Gamma 提供），又最大程度地保留了物体的边缘和局部反差（由 CLAHE 提供）。

### 总结对比表（方便放入报告）

## 场景化微调（hw\_retinex-master，训练已完成）

代码目录：`hw_retinex-master/`（相对仓库根目录）。相对主复现代码 `basicsr/` 的改动：在 Denoiser 的 IG-MSA / IGAB FFN 中注入 LoRA 旁路，并新增多损失训练逻辑。

### 动机

- 自采 mydata 上，公开集预训练 checkpoint 指标差异大（如 FiveK SSIM $\approx$ 0.21 vs SDSD\_outdoor $\approx$ 0.65），存在 **domain gap**。
- 目标：在保留 Retinex 分解能力的前提下，以**参数高效**方式适配场景相关噪声与纹理（集中在 Denoiser）。

### LoRA 微调策略

实现位置：`basicsr/models/image_restoration_model.py` → `freeze_backbone_unfreeze_lora()`。

- **冻结**：Estimator 全部参数 + Denoiser 全部预训练主干（`requires_grad=False`）。
- **可训练**：名称含 `lora_` 的参数（LoRA 旁路）。
- **LoRA 注入**（`basicsr/models/archs/RetinexFormer_arch.py`）：
  - IG-MSA：`to_q_lora` / `to_k_lora` / `to_v_lora` / `proj_lora`（`LoRALinear`，$r=16$，`lora_alpha=8`）。
  - IGAB FeedForward：`lora_0` / `lora_4`（`LoRAConv1x1_Only`，与首尾 $1\times1$ Conv 并联）。
- Estimator **无 LoRA 层**，训练时始终冻结。
- 预训练加载：`strict_load_g: false`（checkpoint 不含 LoRA 键）。

### 训练配置

| 项 | 值 |
|---|---|
| 训练入口 | `python basicsr/train.py --opt Options/RetinexFormer_LOL_v2_real_finetune.yml` |
| 测试入口 | `python Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real_finetune.yml --weights 21.13/best_psnr_21.13_4000.pth --dataset LOL_v2_real` |
| 预训练权重 | `LOL_v2_real.pth` |
| 训练数据 | LOL-v2 Real captured Train（Normal / Low） |
| 验证数据 | LOL-v2 Real captured Test |
| 总迭代 | 20,000 |
| 优化器 | Adam，lr = $5\times10^{-5}$ |
| Patch | 256×256，batch = 32，Mixup（beta=1.2） |
| 梯度裁剪 | 0.01 |

> **说明**：当前 yml 指向 LOL-v2 公开集，用于验证 LoRA 微调流程；未在 mydata 分场景上继续微调。

### 损失函数

配置见 `Options/RetinexFormer_LOL_v2_real_finetune.yml`：

- **l\_pix**（幻灯片 loss\_I-pix）：L1Loss，weight = 1.0
- **l\_ssim**（幻灯片 loss\_I-ssim）：SSIMLoss，$(1-\mathrm{SSIM})\times 0.2$

实现：`basicsr/models/losses/losses.py`（新增 SSIMLoss 等）。

### 模块分工（架构回顾）

RetinexFormer 核心设计思想是将经典的 **Retinex 理论**（图像分解为光照图与反射率）与现代 **Transformer** 架构相结合。模型主要由两大核心子模块串联而成：

\[低光照输入图像\] ─\&gt; \[光照估计器 Estimator\] ─\&gt; 输出物理提亮图 \(input\_img\) \&amp; 光照特征 \(illu\_fea\)                                                                                                    │                               │                                                                                                                        v                               v                                                                                                             \[细节修复去噪器 Denoiser\] ───\&gt; \[高清输出图像\]

幻灯片符号对应：\(\overline{L}\) = `illu_map`（提亮图），\(F_{\mathrm{illu}}\) = `illu_fea`（光照引导特征）。

**Illumination\_Estimator（光照估计器）**

- **定位**：模型的“视觉感知前端”，负责建模光照分布。

- **机制**：计算输入彩色图像的通道均值（`mean\(dim=1\)`）得到亮度单通道图，并与原图拼接（4\-Channel 输入）。通过经典的 $1\times1$ 卷积与 $5\times5$ 深度可分离卷积，解耦出：

    1. `illu\_map`（光照图）：用于对原图进行物理层面的初步提亮。

    2. `illu\_fea`（光照特征）：富含亮度上下文信息，作为强力先验输入后续的去噪器。

**Denoiser（细节修复与去噪器）**

- **定位**：模型的“后道细节恢复器”。

- **机制**：由于物理提亮会不可避免地放大暗部噪声与伪影，该模块采用类似 **U\-Net** 的 Encoder\-Decoder（编码器\-解码器）架构。通过下采样捕捉全局大局观，上采样恢复分辨率，并借助跳跃连接（Skip Connection）确保浅层纹理细节不丢失。

**IG\-MSA（光照引导自注意力机制）**

普通的 Self\-Attention 在图像恢复任务中易导致边缘模糊，RetinexFormer 对此进行了针对性重构：

- **核心操作**：在计算全局自注意力前，直接将前端估计出的光照特征 `illu\_attn` 与 Value 矩阵进行哈达玛积交互（`v = v \* illu\_attn`）。

- **技术优势**：使 Attention 矩阵天然具备**光照感知能力**。网络能自适应地识别出“哪些区域原本极暗、需要分配更高的注意力权重以修复噪声”，实现局部自适应的精细化表征。

**IGAB 模块中的 FeedForward 优化**

- **机制**：传统的 Transformer 采用纯全连接层（MLP）作为前馈网络，而这里的 `FeedForward` 被改造成了由 $1\times1$ 卷积与 $3\times3$ **深度可分离卷积（Groups Conv）** 组合的结构。

- **技术优势**：显着增强了网络保持图像局部空间连续性的能力，这对于图像去噪和边缘保持至关重要。

### 训练曲线与验证指标

本地图片：`slides/img/loss_I-pix.png`，`slides/img/loss_I-ssim.png`

| 设置 | PSNR | SSIM | 备注 |
|---|---|---|---|
| 预训练 LOL\_v2\_real（零样本） | 22.80 | 0.840 | 复现表，无 SE |
| LoRA 微调 best | **21.13** | **0.841** | iter 4000，`best_psnr_21.13_4000.pth` |

# 数据采集相关

## 数据采集

|有GT||无GT|
|---|---|---|
|极限暗光纹理类|高反差复杂光源类||
|9对|5对|4对|

跑自采数据流程：

DNG 原图（`mydata/`）→ `convert\_dng\_to\_png\.py` → PNG（`mydata\_png/`）→ `inference\_mydata\_png\.py` → 增强结果（`results/mydata\_png/`）

## 用mydata数据集跑的结果：

|预训练权重和模型配置|Without self\-ensemble||With self\-ensemble||
|---|---|---|---|---|
||PSNR\(with GT\)|SSIM\(with GT\)|PSNR\(with GT\)|SSIM\(with GT\)|
|LOL\_v1\.pth|15\.8222|0\.6084|15\.9053|0\.6223|
|LOL\_v2\_real|14\.5201|0\.6061|14\.4307|0\.6137|
|LOL\_v2\_synthetic|19\.5011|0\.5759|19\.4783|0\.5856|
|FiveK|16\.2962|0\.2061|16\.2775|0\.2058|
|NTIRE|16\.2959|0\.5633|16\.2940|0\.5657|
|SDSD\_indoor|18\.8688|0\.6043|19\.0728|0\.6203|
|SDSD\_outdoor|19\.0794|0\.6431|19\.2033|0\.6462|
|SID|16\.2693|0\.6383|16\.3640|0\.6494|
|SMID|18\.8043|0\.4507|18\.9316|0\.4563|



