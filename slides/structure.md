Slides structure（5 min，共 9 页）

1.  首页                                    ~20s
2.  问题与任务                                ~25s
    低光增强背景 + 本作业：复现 / 自采验证 / 改进
3.  Retinex 理论                              ~35s
    I = R × L，一页图 + 直觉解释
4.  模型结构                                  ~45s
    Estimator → Denoiser pipeline 图；IG-MSA 一句话带过
5.  公开数据集复现与 Self-ensemble              ~60s
    左：5 数据集 PSNR/SSIM 表（with / without）
    右：self-ensemble 原理 + 提升幅度小结
6.  自采数据：流程与结果                        ~70s
    上：两类场景（极限暗光 / 高反差）、有/无 GT、DNG→PNG→推理流程
    下：多 checkpoint 对比表 + 1–2 张定性对比图（必放图）
7.  分析：Domain Gap 与传统方法对比            ~40s
    【Domain Gap】（notes §mydata 结果）
    · 同一 mydata，不同预训练 checkpoint 指标差异极大
      （例：FiveK SSIM≈0.21 vs SDSD_outdoor SSIM≈0.65；LOL_v1 PSNR≈15.8）
    · 说明公开集训练的泛化不足，需针对自采场景改进
    【传统 baseline 对比】（notes §改进；实现见 Enhancement/traditional_baseline.py）
    · 放 notes 总结对比表（四列：算法 / 核心思想 / 优点 / 面对 LOL 的痛点）
      - Gamma：全局幂函数映射，快、提亮明显 → 亮部过曝、暗部噪点放大
      - CLAHE：分块直方图均衡 + clip_limit，保局部细节 → 极暗场景提亮幅度有限
      - Gamma+CLAHE：先全局拉伸再 HSV 局部润色，优于单方法
        → 仍无法消除提亮带来的红绿彩噪
    · 结论：传统方法可作 baseline，但彩噪/感知质量瓶颈需深度学习（RetinexFormer）
8.  改进：特定场景后训练计划                    ~35s
    【动机】传统方法 + 零样本预训练均不足以覆盖自采两类场景（notes §数据采集）
    · 极限暗光纹理类（9 对 GT）/ 高反差复杂光源类（5 对 GT）
    【策略】（notes §后训练初步计划）
    · RetinexFormer 分工：Estimator 物理提亮 → Denoiser 细节修复与去噪
    · 冻结 self.estimator（光照估计器），仅微调 Denoiser（去噪器）
      — 理由：光照分解较通用；场景相关的纹理/噪声模式集中在 Denoiser
    · 实现：解析权重 name 选择性冻结；训练入口 models/image_restoration_model.py
    · 目标：分别在两类场景上 fine-tune，缩小 domain gap、提升 mydata PSNR/SSIM
    （若时间紧：口头说明「初步计划，进行中」即可，不必展开 IG-MSA 细节）
9.  总结与分工                                ~30s
10. 感谢倾听
时间分配：方法（2–4）~1:45 | 实验（5–6）~2:10 | 分析改进（7–8）~1:15 | 首尾 ~55s
