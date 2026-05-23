import warnings
import os

# 1. 强行忽略所有烦人的 PyTorch 警告
warnings.filterwarnings("ignore", category=UserWarning)

# 2. 强行指定 CUDA 算力兼容目标
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
import sys

# 【核心】强行把当前虚拟环境的底层库路径，硬塞到 Windows 搜寻链的最顶端
env_path = r"C:\Users\Lenovo\anaconda3\envs\retinex"
os.environ["PATH"] = os.path.join(env_path, "Library", "bin") + os.path.pathsep + os.environ["PATH"]

try:
    import torch
    print("🎉 恭喜！PyTorch 成功加载！")
    print("GPU 是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前显卡:", torch.cuda.get_device_name(0))
except Exception as e:
    print("依然报错，错误信息为:", e)