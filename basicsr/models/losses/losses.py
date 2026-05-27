import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import torchvision.models as models
from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss   #把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

@weighted_loss   #把 mse_loss 作为 weighted_loss 的输入
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. 'f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight

    def gaussian_window(self, window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window @ _1D_window.t()
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window_size=11):
        (_, channel, _, _) = img1.size()
        window = self.create_window(window_size, channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = (
            (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ) / (
            (mu1_sq + mu2_sq + C1) *
            (sigma1_sq + sigma2_sq + C2)
        )

        return ssim_map.mean()

    def forward(self, pred, target):
        loss = 1 - self.ssim(pred, target)
        return self.loss_weight * loss

class VGGLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(VGGLoss, self).__init__()

        vgg = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg[:35])).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        pred_feat = self.feature_extractor(pred)
        target_feat = self.feature_extractor(target)

        loss = F.l1_loss(pred_feat, target_feat)
        return self.loss_weight * loss

class FrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(FrequencyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        pred_real = pred_fft.real
        pred_imag = pred_fft.imag

        target_real = target_fft.real
        target_imag = target_fft.imag

        loss_real = F.l1_loss(pred_real, target_real)
        loss_imag = F.l1_loss(pred_imag, target_imag)

        loss = loss_real + loss_imag
        return self.loss_weight * loss
