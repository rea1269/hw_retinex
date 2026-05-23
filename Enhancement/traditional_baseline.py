import cv2
import numpy as np
import os
import glob
import argparse

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


def process_dataset(input_dir, base_output_dir):
    """
    批量处理 LOL 数据集（包含 Gamma、CLAHE 以及 Gamma+CLAHE）
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
    
    output_dir_gamma = os.path.join(dataset_root, "gamma")
    output_dir_clahe = os.path.join(dataset_root, "clahe")
    output_dir_combined = os.path.join(dataset_root, "combined")

    os.makedirs(output_dir_gamma, exist_ok=True)
    os.makedirs(output_dir_clahe, exist_ok=True)
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
        img_gamma = enhance_gamma(img, gamma=0.4) 
        gamma_save_path = os.path.join(output_dir_gamma, filename)
        cv2.imwrite(gamma_save_path, img_gamma)
        
        # 2. 纯 CLAHE 增强
        img_clahe = enhance_clahe(img, clip_limit=2.5) 
        clahe_save_path = os.path.join(output_dir_clahe, filename)
        cv2.imwrite(clahe_save_path, img_clahe)
        
        # 3. Gamma + CLAHE 
        # 默认 gamma=0.4 并设置 clip_limit=3.0 强化对比度
        img_combined = enhance_gamma_plus_clahe(img, gamma=0.4, clip_limit=3.0)
        combined_save_path = os.path.join(output_dir_combined, filename)
        cv2.imwrite(combined_save_path, img_combined)
        
    print("处理完成！所有结果已成功保存。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="传统低照度图像增强算法命令行工具")
    
    parser.add_argument("--input", type=str, required=True, 
                        help="测试集暗光图片所在的文件夹路径 (例如: ../data/LOLv1/Test/input)")
    parser.add_argument("--output_base", type=str, default="../results", 
                        help="输出结果的根目录，默认为 ../results")
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output_base)