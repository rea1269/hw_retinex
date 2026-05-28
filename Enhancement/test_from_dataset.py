from ast import arg
import numpy as np
import os
import cv2
import utils
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics

# BasicSR 框架
from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse

# 进行自增强
def self_ensemble(x, model):#[B, C, H, W]
    #翻转后输入模型得到输出,再翻转回去,最后平均
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
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

def single_image_enhance(model, img_path, output_path, self_ensemble_flag=False, factor=4):
    """
    对单张图片进行增强
    Args:
        model: 加载好的模型
        img_path: 输入图片路径
        output_path: 输出图片路径
        self_ensemble_flag: 是否使用self-ensemble
        factor: padding的倍数
    Returns:
        restored: 增强后的图像 (numpy array)
    """
    # 检查输入图片是否存在
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Input image not found: {img_path}")
    
    # 读取图片
    img = np.float32(utils.load_img(img_path)) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)
    input_ = img.unsqueeze(0).cuda()
    
    # Padding in case images are not multiples of factor
    b, c, h, w = input_.shape
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    
    # 处理大图
    if h < 3000 and w < 3000:
        if self_ensemble_flag:
            restored = self_ensemble(input_, model)
        else:
            restored = model(input_)
    else:
        # split and test for large images
        input_1 = input_[:, :, :, 1::2]
        input_2 = input_[:, :, :, 0::2]
        if self_ensemble_flag:
            restored_1 = self_ensemble(input_1, model)
            restored_2 = self_ensemble(input_2, model)
        else:
            restored_1 = model(input_1)
            restored_2 = model(input_2)
        restored = torch.zeros_like(input_)
        restored[:, :, :, 1::2] = restored_1
        restored[:, :, :, 0::2] = restored_2
    
    # Unpad images to original dimensions
    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    utils.save_img(output_path, img_as_ubyte(restored))
    print(f"Enhanced image saved to: {output_path}")
    
    return restored

def calculate_metrics(restored, target):
    """
    计算PSNR和SSIM指标
    Args:
        restored: 增强后的图像
        target: 真实图像
    Returns:
        psnr_val, ssim_val
    """
    psnr_val = utils.PSNR(target, restored)
    ssim_val = utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored))
    return psnr_val, ssim_val

parser = argparse.ArgumentParser(description='Image Enhancement using Retinexformer')
parser.add_argument('--input_dir', default='./Enhancement/Datasets',type=str, help='Directory of validation images')#输入
parser.add_argument('--result_dir', default='./results/',type=str, help='Directory for results')#输出

parser.add_argument('--opt', type=str, default='Options/RetinexFormer_LOL_v2_real.yml', help='Path to option YAML file.')#模型配置文件yaml
parser.add_argument('--weights', default='pretrained_weights/LOL_v2_real.pth',type=str, help='Path to weights')#模型权重文件pth

parser.add_argument('--dataset', default='LOL_v2_real', type=str,help='Test Dataset')#测试数据集名称
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')#使用的GPU设备

parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')#使用GT的平均值来校正模型的输出
parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble to obtain better results')#使用自增强来获得更好的结果

parser.add_argument('--input_image', type=str, default='', help='Single image path for enhancement (bypasses dataset testing)')#单张图片路径
parser.add_argument('--target_image', type=str, default='', help='Ground truth image path for metric calculation')#GT图片路径
parser.add_argument('--output_image', type=str, default='./enhanced_output.png', help='Output path for single image enhancement')#输出图片路径
parser.add_argument('--calc_metrics', action='store_true', help='Calculate PSNR and SSIM (requires target image)')#是否计算PSNR和SSIM
args = parser.parse_args()

# 指定 gpu
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")
import yaml
from yaml import CLoader as Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False
# 需要改成utf-8编码，否则在Windows系统上会报错 #
x = yaml.load(open(args.opt, mode='r', encoding='utf-8'), Loader=Loader)
s = x['network_g'].pop('type')
##########################
model_restoration = create_model(opt).net_g
# 加载模型
checkpoint = torch.load(weights)
try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)
print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# ========== 单张图片处理模式 ==========
if args.input_image != '':
    print(f"Single image mode: processing {args.input_image}")
    print(f"Self-ensemble: {args.self_ensemble}")

    # 设置PKU数据文件夹路径
    pku_data_dir = os.path.join('data', 'PKU')
    pku_result_dir = os.path.join(pku_data_dir, 'result')
    
    # 构建完整的输入图片路径
    if os.path.isabs(args.input_image):
        input_img_path = args.input_image
    else:
        input_img_path = os.path.join(pku_data_dir, args.input_image)
    
    print(f"Reading input image from: {input_img_path}")
    
    # 构建输出图片路径
    if args.output_image == './enhanced_output.png' or args.output_image == '':
        os.makedirs(pku_result_dir, exist_ok=True)
        model_name = os.path.basename(args.weights).replace('.pth', '')
        input_basename = os.path.splitext(os.path.basename(input_img_path))[0]
        input_ext = os.path.splitext(os.path.basename(input_img_path))[1]
        output_filename = f"{input_basename}_{model_name}{input_ext}"
        output_img_path = os.path.join(pku_result_dir, output_filename)
    elif os.path.isabs(args.output_image):
        output_img_path = args.output_image
    else:
        os.makedirs(pku_result_dir, exist_ok=True)
        output_img_path = os.path.join(pku_result_dir, args.output_image)
    
    print(f"Output will be saved to: {output_img_path}")
    
    # 增强单张图片
    restored_img = single_image_enhance(
        model_restoration, 
        input_img_path, 
        output_img_path,
        self_ensemble_flag=args.self_ensemble
    )
    
    # 如果提供了GT图片且需要计算指标
    if args.calc_metrics and args.target_image != '':
        # 构建GT图片路径
        if os.path.isabs(args.target_image):
            target_img_path = args.target_image
        else:
            target_img_path = os.path.join(pku_data_dir, args.target_image)
        
        print(f"Reading target image from: {target_img_path}")
        
        if not os.path.exists(target_img_path):
            print(f"Warning: Target image not found at {target_img_path}")
        else:
            target_img = np.float32(utils.load_img(target_img_path)) / 255.
        
        # 如果启用了GT_mean校正
        if args.GT_mean:
            mean_restored = cv2.cvtColor(restored_img.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            mean_target = cv2.cvtColor(target_img.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            restored_img = np.clip(restored_img * (mean_target / mean_restored), 0, 1)
        
        psnr_val, ssim_val = calculate_metrics(restored_img, target_img)
        print(f"PSNR: {psnr_val:.4f}")
        print(f"SSIM: {ssim_val:.4f}")

        # 如果有GT_mean校正，重新保存校正后的图片
        if args.GT_mean:
            utils.save_img(output_img_path, img_as_ubyte(restored_img))
            print(f"GT_mean corrected image saved to: {output_img_path}")

    elif args.calc_metrics and args.target_image == '':
        print("Warning: calc_metrics is enabled but no target_image provided. Skipping metric calculation.")
    
    print("Single image processing completed!")
    exit(0)

# ========== 数据集批量处理模式 ==========
# 生成输出结果的文件
factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')

os.makedirs(result_dir, exist_ok=True)

psnr = []
ssim = []
if dataset in ['SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor']:#如果数据集是SID、SMID、SDSD_indoor或SDSD_outdoor，则使用BasicSR的数据加载方式进行测试
    os.makedirs(result_dir_input, exist_ok=True)
    os.makedirs(result_dir_gt, exist_ok=True)

    if dataset == 'SID':from basicsr.data.SID_image_dataset import Dataset_SIDImage as Dataset
    elif dataset == 'SMID':from basicsr.data.SMID_image_dataset import Dataset_SMIDImage as Dataset
    else:from basicsr.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset

    opt = opt['datasets']['val']
    opt['phase'] = 'test'
    if opt.get('scale') is None:opt['scale'] = 1
    if '~' in opt['dataroot_gt']:opt['dataroot_gt'] = os.path.expanduser('~') + opt['dataroot_gt'][1:]
    if '~' in opt['dataroot_lq']:opt['dataroot_lq'] = os.path.expanduser('~') + opt['dataroot_lq'][1:]
    dataset = Dataset(opt)
    print(f'test dataset length: {len(dataset)}')

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_batch['lq']
            input_save = data_batch['lq'].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            target = data_batch['gt'].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            # Padding in case images are not multiples of 4
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if args.self_ensemble:
                restored = self_ensemble(input_, model_restoration)
            else:
                restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored)))
            type_id = os.path.dirname(inp_path).split('/')[-1]
            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_input, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_gt, type_id), exist_ok=True)
            utils.save_img((os.path.join(result_dir, type_id, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            utils.save_img((os.path.join(result_dir_input, type_id, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(input_save))
            utils.save_img((os.path.join(result_dir_gt, type_id, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(target))
else:
    input_dir = opt['datasets']['val']['dataroot_lq']
    target_dir = opt['datasets']['val']['dataroot_gt']
    print(input_dir)
    print(target_dir)

    input_paths = natsorted(glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))
    target_paths = natsorted(glob(os.path.join(target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')))

    with torch.inference_mode():
        for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(target_paths)):

            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(inp_path)) / 255.
            target = np.float32(utils.load_img(tar_path)) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 4
            b, c, h, w = input_.shape
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
            # 处理大图
            if h < 3000 and w < 3000:
                if args.self_ensemble:
                    restored = self_ensemble(input_, model_restoration)
                else:
                    restored = model_restoration(input_)
            else:
                # split and test
                input_1 = input_[:, :, :, 1::2]
                input_2 = input_[:, :, :, 0::2]
                if args.self_ensemble:
                    restored_1 = self_ensemble(input_1, model_restoration)
                    restored_2 = self_ensemble(input_2, model_restoration)
                else:
                    restored_1 = model_restoration(input_1)
                    restored_2 = model_restoration(input_2)
                restored = torch.zeros_like(input_)
                restored[:, :, :, 1::2] = restored_1
                restored[:, :, :, 0::2] = restored_2

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored)))
            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))

psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
print("PSNR: %f " % (psnr))#峰值信噪比
print("SSIM: %f " % (ssim))#结构相似性指数
