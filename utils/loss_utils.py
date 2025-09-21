#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def tv_loss_module(img):
    #assume img to be shape of (cxhxw)
    img = img.squeeze(0)
    loss = torch.mean(torch.abs(img[:, :, :-1] - img[:, :, 1:])) + torch.mean(torch.abs(img[:, :-1, :] - img[:, 1:, :]))
    return loss

def cross_modal_edge_loss(img1, img2):
    #Assume img1 and img2 are of shape (C, H, W)
    grad_x1 = img1[:, :, :-1] - img1[:, :, 1:]
    grad_x2 = img2[:, :, :-1] - img2[:, :, 1:]
    
    grad_y1 = img1[:, :-1, :] - img1[:, 1:, :]
    grad_y2 = img2[:, :-1, :] - img2[:, 1:, :]
    
    loss_x = torch.abs(grad_x1 - grad_x2).mean()
    loss_y = torch.abs(grad_y1 - grad_y2).mean()
    total_loss = loss_x + loss_y
    return total_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
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

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# def DCLoss(img, valid_rays, patch_size):
#     """
#     calculating dark channel of image, the image shape is of N*C*W*H
#     """
#     img[~valid_rays] = 1e3
#     img = img.t().view(3,-1,patch_size,patch_size)
#     img = img.transpose(0,1)
#     maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, 0, 0))
#     dc = -maxpool(-img[:, None, :, :, :]).flatten()
#     dc = dc[dc < 999]
#     dc = dc**2
#     return dc.mean()

class DCLoss(nn.Module):
    def __init__(self, patch_size=16):
        super(DCLoss, self).__init__()
        self.patch_size = patch_size
        self.maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=patch_size, 
                                   padding=(0, patch_size//2, patch_size//2))

    def forward(self, img):
        """
        calculating dark channel of image, the image shape is of N*C*W*H
        """
        img = img.unsqueeze(0)
        img[img<0.01] = 1e3
        maxpool = nn.MaxPool3d((3, self.patch_size, self.patch_size), stride=1, padding=(0, self.patch_size//2, self.patch_size//2))
        dc = -maxpool(0-img[:, None, :, :, :])
        dc = dc[dc < 999]   
        return (dc**2).mean()

class MixtureNLLLoss(nn.Module):
    def __init__(self, fixed_mu1=1e-5, initial_pi=0.5, epsilon=1e-6):
        super(MixtureNLLLoss, self).__init__()
        self.logit_pi = nn.Parameter(torch.logit(torch.tensor(initial_pi)))
        #initialize mu2 to 1e-3
        self.mu2 = nn.Parameter(torch.tensor(1e-3))
        self.fixed_mu1 = fixed_mu1
        #initialize sigma1 and sigma2 to 1e-3
        self.log_sigma1 = nn.Parameter(torch.tensor(-6.9078))
        self.log_sigma2 = nn.Parameter(torch.tensor(-6.9078))
        self.epsilon = epsilon

    def forward(self, logits):
        #calculate the log likelihood of the mixture model
        pi = 0.5
        mu1 = self.fixed_mu1
        mu2 = self.mu2
        sigma1 = torch.exp(self.log_sigma1)
        sigma2 = torch.exp(self.log_sigma2)
        #calculate the log likelihood of the mixture model
        log_likelihood = torch.log(pi * torch.exp(-0.5 * (logits - mu1) ** 2 / (sigma1 ** 2)) / (torch.pi*sigma1*math.sqrt(2) + self.epsilon) + (1 - pi) * torch.exp(-0.5 * (logits - mu2) ** 2 / (sigma2 ** 2)) / (torch.pi*sigma2*math.sqrt(2) + self.epsilon) + self.epsilon)
        return -log_likelihood.mean()
        