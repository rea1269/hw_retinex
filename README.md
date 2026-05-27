### Introduction
This is a baseline and toolbox for wide-range low-light image enhancement. This repo **supports over 15 benchmarks** and extremely high-resolution (up to 4000x6000) low-light enhancement. Our method Retinexformer **won the second place** in the [NTIRE 2024 Challenge on Low Light Enhancement](https://codalab.lisn.upsaclay.fr/competitions/17640).  If you find this repo useful, please give it a star ⭐ and consider citing our paper in your research. Thank you.

### Results
- Results on LOL-v1, LOL-v2-real, LOL-v2-synthetic, SID, SMID, SDSD-in, SDSD-out, and MIT Adobe FiveK datasets can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1DC6A-I9S7yJ-pmMVTLAHaw?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1UCpHh3MkV4bxzWgiiULnb3BOPWS_8crP?usp=drive_link)

- Results on LOL-v1, LOL-v2-real, and LOL-v2-synthetic datasets with the same test setting as KinD, LLFlow, and recent diffusion models can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1Kbq9pASf1O_0Y9QMc88obQ?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1_ugNFblIYOCIam4cVJiXVX1aBGEYhn1o?usp=drive_link).

- Results on the NTIRE 2024 low-light enhancement dataset can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1DC6A-I9S7yJ-pmMVTLAHaw?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1bzICalpU1RprfnepLsMyCYT569UWTTlV?usp=sharing)

- Results on LIME, NPE, MEF, DICM, and VV datasets can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1cqBwmuXk83h6u1NZJVbfkg?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1rWa_WRX5bqlW2HnBNMUGFKWrou7gIQpO?usp=drive_link)

- Results on ExDark nighttime object detection can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1ZvoPzYQePRc80-o7rrJuRQ?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1nZQnwKkGvswv--JunzgBLTXVRlOdcxb6?usp=sharing). Please use [this repo](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/tree/main/IAT_high/IAT_mmdetection) to run experiments on the ExDark dataset

- Results of some compared baseline methods are shared on [Google Drive](https://drive.google.com/drive/folders/1P75bv6jBp8UxcLDhqMACvIQXck2sS9da?usp=sharing) and [Baidu Disk](https://pan.baidu.com/s/1ksGGV6nMyVVE6OqGuyO8_w?pwd=cyh2)

## 1. Create Environment

We suggest you use pytorch 1.11 to re-implement the results in our ICCV 2023 paper and pytorch 2 to re-implement the results in NTIRE 2024 Challenge because pytorch 2 can save more memory in mix-precision training.

### 1.1 Install the environment with Pytorch 1.11

- Make Conda Environment
```
conda create -n Retinexformer python=3.7
```

- Install Dependencies
```
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```

### 1.2 Install the environment with Pytorch 2

- Make Conda Environment
```
conda create -n torch2 python=3.9 -y
conda activate torch2
```

- Install Dependencies
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```

## 2. Prepare Dataset
Download the following datasets:

LOL-v1 [Baidu Disk](https://pan.baidu.com/s/1ZAC9TWR-YeuLIkWs3L7z4g?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing)

LOL-v2 [Baidu Disk](https://pan.baidu.com/s/1X4HykuVL_1WyB3LWJJhBQg?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing)

SID [Baidu Disk](https://pan.baidu.com/share/init?surl=HRr-5LJO0V0CWqtoctQp9w) (code: `gplv`), [Google Drive](https://drive.google.com/drive/folders/1eQ-5Z303sbASEvsgCBSDbhijzLTWQJtR?usp=share_link&pli=1)

SMID [Baidu Disk](https://pan.baidu.com/share/init?surl=Qol_4GsIjGDR8UT9IRZbBQ) (code: `btux`), [Google Drive](https://drive.google.com/drive/folders/1OV4XgVhipsRqjbp8SYr-4Rpk3mPwvdvG)

SDSD-indoor [Baidu Disk](https://pan.baidu.com/s/1rfRzshGNcL0MX5soRNuwTA?errmsg=Auth+Login+Params+Not+Corret&errno=2&ssnerror=0#list/path=%2F) (code: `jo1v`), [Google Drive](https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6)

SDSD-outdoor [Baidu Disk](https://pan.baidu.com/share/init?surl=JzDQnFov-u6aBPPgjSzSxQ) (code: `uibk`), [Google Drive](https://drive.google.com/drive/folders/14TF0f9YQwZEntry06M93AMd70WH00Mg6)

MIT-Adobe FiveK [Baidu Disk](https://pan.baidu.com/s/1ajax7N9JmttTwY84-8URxA?pwd=cyh2) (code:`cyh2`), [Google Drive](https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view?usp=sharing), [Official](https://data.csail.mit.edu/graphics/fivek/)

NTIRE 2024 [Baidu Disk](https://pan.baidu.com/s/1Tl-LUhwsPh6XFA2SqR5c8Q?pwd=cyh2) (code:`cyh2`), Google Drive links for [training input](https://drive.google.com/file/d/1Js9yHmV0xAWhT5oJKzfx6oOr_7k5hcNg/view), [training GT](https://drive.google.com/file/d/1PUJgJiEyrIj5TgwcQlFvVGuIe3_PXMLY/view), and [mini-val set](https://drive.google.com/drive/folders/1M-WVWToH1HhMtmQlYrb8qlNCQgi0kG3y?usp=sharing).

## 3. Testing

Download our models from [Baidu Disk](https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link). Put them in folder `pretrained_weights`

```shell
# activate the environment
conda activate Retinexformer

# LOL-v1
python Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1

# LOL-v2-real
python Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real

python Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real_finetune.yml --weights 21.13/best_psnr_21.13_4000.pth --dataset LOL_v2_real

# LOL-v2-synthetic
python Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic

# SID
python Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SID.yml --weights pretrained_weights/SID.pth --dataset SID

# SMID
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SMID.yml --weights pretrained_weights/SMID.pth --dataset SMID

# SDSD-indoor
python Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SDSD_indoor.yml --weights pretrained_weights/SDSD_indoor.pth --dataset SDSD_indoor

# SDSD-outdoor
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_SDSD_outdoor.yml --weights pretrained_weights/SDSD_outdoor.pth --dataset SDSD_outdoor

# FiveK
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_FiveK.yml --weights pretrained_weights/FiveK.pth --dataset FiveK

# NTIRE
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_NTIRE.yml --weights pretrained_weights/NTIRE.pth --dataset NTIRE --self_ensemble

# MST_Plus_Plus trained with 4 GPUs on NTIRE 
python3 Enhancement/test_from_dataset.py --opt Options/MST_Plus_Plus_NTIRE_4x1800.yml --weights pretrained_weights/MST_Plus_Plus_4x1800.pth --dataset NTIRE --self_ensemble

# MST_Plus_Plus trained with 8 GPUs on NTIRE 
python3 Enhancement/test_from_dataset.py --opt Options/MST_Plus_Plus_NTIRE_8x1150.yml --weights pretrained_weights/MST_Plus_Plus_8x1150.pth --dataset NTIRE --self_ensemble

```
- #### Self-ensemble testing strategy
We add the self-ensemble strategy in the testing code to derive better results. Just add a `--self_ensemble` action at the end of the above test command to use it.

- #### The same test setting as LLFlow, KinD, and recent diffusion models
We provide the same test setting as LLFlow, KinD, and recent diffusion models. Please note that we do not suggest this test setting because it uses the mean of ground truth to enhance the output of the model. But if you want to follow this test setting, just add a `--GT_mean` action at the end of the above test command as

```shell
# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1 --GT_mean

# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real --GT_mean

# LOL-v2-synthetic
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic --GT_mean
```

## 4. Training

Feel free to check our training logs from [Baidu Disk](https://pan.baidu.com/s/16NtLba_ANe3Vzji-eZ1xAA?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/drive/folders/1HU_wEn_95Hakxi_ze-pS6Htikmml5MTA?usp=sharing)

We suggest you use the environment with pytorch 2 to train our model on the NTIRE 2024 dataset and the environment with pytorch 1.11 to train our model on other datasets.

```shell
# activate the enviroment
conda activate Retinexformer

# LOL-v1
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v1.yml

# LOL-v2-real
python basicsr/train.py --opt Options/RetinexFormer_LOL_v2_real_finetune.yml

# LOL-v2-synthetic
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml

# SID
python3 basicsr/train.py --opt Options/RetinexFormer_SID.yml

# SMID
python3 basicsr/train.py --opt Options/RetinexFormer_SMID.yml

# SDSD-indoor
python3 basicsr/train.py --opt Options/RetinexFormer_SDSD_indoor.yml

# SDSD-outdoor
python3 basicsr/train.py --opt Options/RetinexFormer_SDSD_outdoor.yml

# FiveK
python3 basicsr/train.py --opt Options/RetinexFormer_FiveK.yml
```

Train our Retinexformer and MST++ with the distributed data parallel (DDP) strategy of pytorch on the NTIRE 2024 Low-Light Enhancement dataset. Please note that we use the mix-precision strategy in the training process, which is controlled by the bool hyperparameter `use_amp`  in the config file.

```shell
# activate the enviroment
conda activate torch2

# Train Retinexformer with 8 GPUs on NTIRE
bash train_multigpu.sh Options/RetinexFormer_NTIRE.yml 0,1,2,3,4,5,6,7 4321

# Train MST++ with 4 GPUs on NTIRE
bash train_multigpu.sh Options/RetinexFormer_NTIRE_4x1800.yml 0,1,2,3,4,5,6,7 4329

# Train MST++ with 8 GPUs on NTIRE
bash train_multigpu.sh Options/MST_Plus_Plus_NTIRE_8x1150.yml 0,1,2,3,4,5,6,7 4343

```