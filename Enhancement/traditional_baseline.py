import cv2
import numpy as np
import os
import glob
import argparse

'''
 传统 MSR 和 RetinexFormer 一脉相承，
 其本质都是在估计环境光照并反推反射率 $R$。
 传统 MSR 提亮效果非常惊艳，
 但它会因为没有对内容的感知能力，将暗处的红绿噪点一同呈对数级放大。
 RetinexFormer 的 Denoiser 引入了深度注意力机制，解决了传统 MSR 的这一痛点。
'''

def enhance_gamma(image, gamma=0.4):
    """
    Gamma 校正提亮
    gamma < 1 : 提亮图像
    gamma > 1 : 变暗图像
    """
    # 建立查找表 (Look-Up Table, LUT) 加速计算
    table = np.array([((i / 255.0) ** gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # 利用查找表进行映射
    return cv2.LUT(image, table)

def enhance_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE 局部直方图均衡化 (在 HSV 空间操作以保持色彩)
    """
    # 1. 转换到 HSV 颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 2. 创建 CLAHE 对象并应用于 V (亮度) 通道
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v_clahe = clahe.apply(v)
    
    # 3. 合并通道并转回 BGR
    hsv_clahe = cv2.merge((h, s, v_clahe))
    result = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    return result

def enhance_gamma_plus_clahe(image, gamma=0.4, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    先用标准 Gamma 强力提亮，再用 CLAHE 优化局部对比度
    """
    # 1. 先进行 Gamma 提亮
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    bright_img = cv2.LUT(image, table)
    
    # 2. 将提亮后的图片转到 HSV 空间，用 CLAHE 优化亮度层的对比度
    hsv = cv2.cvtColor(bright_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v_clahe = clahe.apply(v)
    
    hsv_clahe = cv2.merge((h, s, v_clahe))
    return cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)


# 多尺度 Retinex (Multi-Scale Retinex, MSR) 核心算法实现
def enhance_msr(image, scales=[30, 80, 250], weights=[1/3, 1/3, 1/3], dynamic_range=1.0):
    """
    多尺度 Retinex (MSR) 低光照图像增强算法
    - scales: 高斯模糊的标准差列表，分别捕获小、中、大尺度的光照背景
    - weights: 每个尺度的加权权重
    - dynamic_range: 最后的色彩拉伸控制系数（增益因子）
    """
    # 将图像转换为 float64 防溢出，并加 1.0 防止 log(0) 导致 NAN
    img_bgr = image.astype(np.float64) + 1.0
    msr_log = np.zeros_like(img_bgr)
    
    # 对 B, G, R 三个通道分别执行多尺度 Retinex 解耦
    for channel in range(3):
        img_channel = img_bgr[:, :, channel]
        r_channel_log = np.zeros_like(img_channel)
        
        # 遍历不同尺度的高斯核，计算对数域反射率 R = log(I) - log(L)
        for scale, weight in zip(scales, weights):
            # 高斯模糊得到照度估计 L
            blur = cv2.GaussianBlur(img_channel, (0, 0), scale)
            blur[blur <= 0] = 1.0 # 稳健性防御
            
            # 累加各尺度下的反射率成分
            r_channel_log += weight * (np.log10(img_channel) - np.log10(blur))
            
        msr_log[:, :, channel] = r_channel_log

    # 简易而高效的色彩恢复映射 (Simplest Color Balance 变体)
    # 将对数域反射率线性拉伸并截断到 0-255，强化对比度
    msr_res = np.zeros_like(msr_log)
    for channel in range(3):
        ch_min = np.min(msr_log[:, :, channel])
        ch_max = np.max(msr_log[:, :, channel])
        
        # 线性动态映射
        msr_res[:, :, channel] = (msr_log[:, :, channel] - ch_min) / (ch_max - ch_min) * 255.0
        
    # 转换为标准的 uint8 图像格式
    return np.clip(msr_res * dynamic_range, 0, 255).astype(np.uint8)



def process_dataset(input_dir, base_output_dir):
    """
    批量处理 LOL 数据集（包含 Gamma、CLAHE、Gamma+CLAHE、MSR）
    """
    # 规范化路径
    input_dir = os.path.normpath(input_dir)
    
    # 自动提取 'data/' 后面的所有相对路径作为输出的子目录
    # 如输入 "../data/LOLv2/Real_captured/Test/Low"
    # parts 切出 ["LOLv2", "Real_captured", "Test", "Low"]
    path_parts = input_dir.split(os.sep)
    if "data" in path_parts:
        data_index = path_parts.index("data")
        # 取出 'data' 之后的所有层级
        relative_layers = path_parts[data_index + 1:]
    else:
        # 如果输入的路径没包含 'data'，就取最后三层作为目录结构
        relative_layers = path_parts[-3:] if len(path_parts) >= 3 else path_parts[-1:]

    # 动态拼接成结果根目录：results/traditional/LOLv2/Real_captured/Test/Low/
    dataset_root = os.path.join(base_output_dir, "traditional", *relative_layers)
    
    #output_dir_gamma = os.path.join(dataset_root, "gamma")
    #output_dir_clahe = os.path.join(dataset_root, "clahe")
    output_dir_combined = os.path.join(dataset_root, "combined")
    output_dir_msr = os.path.join(dataset_root, "msr")

    os.makedirs(output_dir_msr, exist_ok=True)
    #os.makedirs(output_dir_gamma, exist_ok=True)
    #os.makedirs(output_dir_clahe, exist_ok=True)
    os.makedirs(output_dir_combined, exist_ok=True) 
    
    # 支持常见的图片格式
    image_paths = glob.glob(os.path.join(input_dir, "*.png")) + \
                  glob.glob(os.path.join(input_dir, "*.jpg"))
    
    print(f"找到 {len(image_paths)} 张待处理图片。")
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"无法读取图片: {filename}")
            continue
            
        # 1. 纯 Gamma 校正
        #img_gamma = enhance_gamma(img, gamma=0.4) 
        #gamma_save_path = os.path.join(output_dir_gamma, filename)
        #cv2.imwrite(gamma_save_path, img_gamma)
        
        # 2. 纯 CLAHE 增强
        #img_clahe = enhance_clahe(img, clip_limit=2.5) 
        #clahe_save_path = os.path.join(output_dir_clahe, filename)
        #cv2.imwrite(clahe_save_path, img_clahe)
        
        # 3. Gamma + CLAHE 
        # 默认 gamma=0.4 并设置 clip_limit=3.0 强化对比度
        img_combined = enhance_gamma_plus_clahe(img, gamma=0.4, clip_limit=3.0)
        combined_save_path = os.path.join(output_dir_combined, filename)
        cv2.imwrite(combined_save_path, img_combined)

        # 4. 推理 MSR 算法并存储结果
        img_msr = enhance_msr(img, scales=[15, 80, 250], weights=[1/3, 1/3, 1/3])
        msr_save_path = os.path.join(output_dir_msr, filename)
        cv2.imwrite(msr_save_path, img_msr)
        
    print("处理完成！所有结果已成功保存。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="传统低照度图像增强算法命令行工具")
    
    parser.add_argument("--input", type=str, required=True, 
                        help="测试集暗光图片所在的文件夹路径 (例如: ../data/LOLv1/Test/input)")
    parser.add_argument("--output_base", type=str, default="../results/traditional", 
                        help="输出结果的根目录，默认为 ../results")
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output_base)