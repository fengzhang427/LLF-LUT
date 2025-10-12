import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


class Halo_Loss(nn.Module):
    def __init__(self, num_lap=3):
        super(Halo_Loss, self).__init__()
        self.num_lap = num_lap
        self.max_pool = nn.MaxPool2d(5)

    def forward(self, img_pyr, ref_img, ref_pyr):
        bin_edge = torch.abs(gradit(ref_pyr[-1], 7))
        bin_edge[bin_edge < 1.5] = 0
        bin_edge[bin_edge > 0] = 1
        combine_B = torch.abs(gradit(bin_edge, 25))
        combine_B[combine_B > 15] = 0
        combine_B[combine_B > 0] = 1
        bin_edge = bin_edge * combine_B
        bin_edge = self.max_pool(bin_edge)
        halo_loss = 0
        for k in range(self.num_lap):
            gradit_v_ = gradit_v(img_pyr[k])
            gradit_v_map_ = gradit_v_map(
                F.interpolate(ref_img, size=(img_pyr[k].shape[-2], img_pyr[k].shape[-1]), mode='bilinear', align_corners=True))
            bin_edge_ = F.interpolate(bin_edge, size=(gradit_v_.shape[-2], gradit_v_.shape[-1]), mode='bilinear', align_corners=True)
            halo_loss += torch.mean(torch.mean(gradit_v_ * gradit_v_map_ * bin_edge_))
            gradit_h_ = gradit_h(img_pyr[k])
            gradit_h_map_ = gradit_h_map(
                F.interpolate(ref_img, size=(img_pyr[k].shape[-2], img_pyr[k].shape[-1]), mode='bilinear', align_corners=True))
            bin_edge_ = F.interpolate(bin_edge, size=(gradit_h_.shape[-2], gradit_h_.shape[-1]), mode='bilinear', align_corners=True)
            halo_loss += torch.mean(torch.mean(gradit_h_ * gradit_h_map_ * bin_edge_))
        return halo_loss


def gradit(img, size=7):
    conv_op = nn.Conv2d(1, 1, size, 1, size // 2, bias=False, padding_mode='replicate')
    log_kernel = LoG(size, 0.6).reshape((1, 1, size, size))
    conv_op.weight.data = torch.from_numpy(log_kernel)
    conv_op.weight.requires_grad = False
    conv_op.cuda()
    if img.shape[1] == 1:
        gray_img = img
    else:
        gray_img = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
        gray_img = gray_img.unsqueeze(0)
    edge_detect = conv_op(gray_img)
    return torch.abs(edge_detect)


def gradit_v_map(img, alpha=0.1):
    gray_img = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
    v_map = torch.abs(gray_img[:, :-1, :] - gray_img[:, 1:, :])
    v_map[v_map < alpha] = 1
    v_map[v_map < 1] = 0
    return v_map


def gradit_v(img):
    v_map = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    return v_map


def gradit_h(img):
    h_map = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    return h_map


def gradit_h_map(img, alpha=0.1):
    gray_img = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
    v_map = torch.abs(gray_img[:, :, :-1] - gray_img[:, :, 1:])
    v_map[v_map < alpha] = 1
    v_map[v_map < 1] = 0
    return v_map


def LoG(size=7, sigma=0.6):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    t = (x ** 2 + y ** 2) / (2.0 * sigma ** 2)
    g = - (1 / np.pi / sigma ** 4) * (1 - t) * np.exp(-t)
    g = g / g.sum()
    return g.astype('float32')
