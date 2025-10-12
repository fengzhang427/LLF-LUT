from abc import ABC
import torch
import trilinear
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


# class Tri_linear(torch.nn.Module, ABC):
#     def __init__(self):
#         super(Tri_linear, self).__init__()
#
#     @staticmethod
#     def forward(lut, img):
#         # scale im between -1 and 1 since its used as grid input in grid_sample
#         img = (img - .5) * 2.
#
#         # grid_sample expects NxDxHxWx3 (1x1xHxWx3)
#         img = img.permute(0, 2, 3, 1)[:, None]
#
#         # add batch dim to LUT
#         lut = lut[None]
#
#         # grid sample
#         result = F.grid_sample(lut, img, mode='bilinear', padding_mode='border', align_corners=True)
#
#         # drop added dimensions and permute back
#         result = result[:, :, 0]
#         return lut, result


class Generator3DLUT_identity(nn.Module, ABC):
    def __init__(self, dim=33):
        global file
        super(Generator3DLUT_identity, self).__init__()
        if dim == 33:
            file = open("./models/IdentityLUT33.txt", 'r')
        elif dim == 64:
            file = open("./models/IdentityLUT64.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)

        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k
                    x = lines[n].split()
                    buffer[0, i, j, k] = float(x[0])
                    buffer[1, i, j, k] = float(x[1])
                    buffer[2, i, j, k] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.Tri_linear = Tri_linear()

    def forward(self, x):
        _, output = self.Tri_linear(self.LUT, x)
        # self.LUT, output = self.Tri_linear(self.LUT, x)
        return output


class Generator3DLUT_zero(nn.Module, ABC):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)
        # self.LUT = nn.Parameter(torch.tensor(self.LUT))
        self.LUT = nn.Parameter(self.LUT.clone().detach().requires_grad_(True))
        self.Tri_linear = Tri_linear()

    def forward(self, x):
        _, output = self.Tri_linear(self.LUT, x)
        return output


class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut=None, x=None):
        x = x.contiguous()
        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)
        assert 1 == trilinear.forward(lut,
                                      x,
                                      output,
                                      dim,
                                      shift,
                                      binsize,
                                      W,
                                      H,
                                      batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        ctx.save_for_backward(*variables)
        return lut, output

    @staticmethod
    def backward(ctx, lut_grad=None, x_grad=None):
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])
        assert 1 == trilinear.backward(x,
                                       x_grad,
                                       lut_grad,
                                       dim,
                                       shift,
                                       binsize,
                                       W,
                                       H,
                                       batch)
        return lut_grad, x_grad


class Tri_linear(torch.nn.Module, ABC):
    def __init__(self):
        super(Tri_linear, self).__init__()

    @staticmethod
    def forward(lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)


class TV_3D(nn.Module, ABC):
    def __init__(self, dim=33):
        super(TV_3D, self).__init__()
        self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float)
        self.weight_r[:, :, :, (0, dim - 2)] *= 2.0
        self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float)
        self.weight_g[:, :, (0, dim - 2), :] *= 2.0
        self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float)
        self.weight_b[:, (0, dim - 2), :, :] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, lut):
        dif_r = lut.LUT[:, :, :, :-1] - lut.LUT[:, :, :, 1:]
        dif_g = lut.LUT[:, :, :-1, :] - lut.LUT[:, :, 1:, :]
        dif_b = lut.LUT[:, :-1, :, :] - lut.LUT[:, 1:, :, :]
        tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
            torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))
        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))
        return tv, mn
