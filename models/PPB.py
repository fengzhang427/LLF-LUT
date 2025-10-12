import torch.nn as nn
import torch.nn.functional as F
import torch
import kornia
from .layers import ResidualBlock, ResidualBlock_1


def remapping(img_gauss, img_lpr, sigma, fact, N):
    discretisation = torch.linspace(0, 1, N)
    discretisation_step = discretisation[1]
    for ref in discretisation:
        img_remap = fact * (img_lpr - ref) * torch.exp(
            -(img_lpr - ref) * (img_lpr - ref) * (2 * sigma * sigma))
        img_lpr = img_lpr + (torch.abs(img_gauss - ref) < discretisation_step) * img_remap * (
                1 - torch.abs(img_gauss - ref) / discretisation_step)
    return img_lpr


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, channels=3, device=torch.device('cuda')):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(device=device, channels=channels)

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)],
                       dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def gauss_decom(self, img):
        current = img
        pyr = [img]
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            pyr.append(down)
            current = down
        return pyr

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


class HFBlock(nn.Module):
    def __init__(self, num_residual_blocks, lap_layer=3):
        super(HFBlock, self).__init__()
        self.high_freq_block = None
        self.lap_layer = lap_layer

        model = [nn.Conv2d(in_channels=10, out_channels=64, kernel_size=1, padding=0, stride=1,
                           groups=1,
                           bias=True),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        for _ in range(num_residual_blocks):
            # model += [ResidualBlock_1(64)]
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, padding=0, stride=1,
                            groups=1,
                            bias=True)]
        self.model = nn.Sequential(*model)

        self.high_freq_blocks = nn.ModuleList()
        for i in range(lap_layer - 1):
            high_freq_block = nn.Sequential(
                nn.Conv2d(in_channels=9, out_channels=16, kernel_size=1, padding=0, stride=1,
                          groups=1,
                          bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1,
                          groups=16,
                          bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, padding=0, stride=1,
                          groups=1,
                          bias=True))
            self.high_freq_blocks.append(high_freq_block)
            # setattr(self, 'high_freq_block_{}'.format(str(i)), high_freq_block)
            # self.add_module('high_freq_block_{}'.format(str(i)), high_freq_block)

    def forward(self, high_with_low, gauss_lut, pyr_lut, enhanced_low):
        pyr_reconstruct_list = []
        fact_sigma = self.model(high_with_low)
        fact = fact_sigma[:, 0, :, :]
        fact = fact.unsqueeze(1)
        sigma = fact_sigma[:, 1, :, :]
        sigma = sigma.unsqueeze(1)
        pyr_reconstruct_ori = remapping(gauss_lut[-2], pyr_lut[-2], sigma, fact, 10)

        pyr_reconstruct = pyr_reconstruct_ori
        up_enhanced = enhanced_low

        for i in range(self.lap_layer - 1):
            up = nn.functional.interpolate(up_enhanced, size=(pyr_lut[-2 - i].shape[2], pyr_lut[-2 - i].shape[3]))
            up_enhanced = up + pyr_reconstruct
            up_enhanced = nn.functional.interpolate(up_enhanced,
                                                    size=(pyr_lut[-3 - i].shape[2], pyr_lut[-3 - i].shape[3]))
            pyr_reconstruct = nn.functional.interpolate(pyr_reconstruct,
                                                        size=(pyr_lut[-3 - i].shape[2], pyr_lut[-3 - i].shape[3]))
            # self.high_freq_block = getattr(self, 'high_freq_block_{}'.format(str(i)))
            concat_high = torch.cat([up_enhanced, pyr_lut[-3 - i], pyr_reconstruct], 1)
            fact_sigma = self.high_freq_blocks[i](concat_high)
            fact = fact_sigma[:, 0, :, :]
            fact = fact.unsqueeze(1)
            sigma = fact_sigma[:, 1, :, :]
            sigma = sigma.unsqueeze(1)
            pyr_reconstruct = remapping(gauss_lut[-3 - i], pyr_lut[-3 - i], sigma, fact, 10)

            setattr(self, 'pyr_reconstruct_{}'.format(str(i)), pyr_reconstruct)

        for i in reversed(range(self.lap_layer - 1)):
            pyr_reconstruct = getattr(self, 'pyr_reconstruct_{}'.format(str(i)))
            pyr_reconstruct_list.append(pyr_reconstruct)

        pyr_reconstruct_list.append(pyr_reconstruct_ori)
        pyr_reconstruct_list.append(enhanced_low)
        return pyr_reconstruct_list


class PPB(nn.Module):
    def __init__(self, config):
        super(PPB, self).__init__()
        num_residual_blocks = config['num_residual_blocks']
        num_lap = config['num_lap']
        self.block = HFBlock(num_residual_blocks, num_lap)

    def forward(self, gauss_input, pyr_input, enhanced_low):
        low_freq_gray = kornia.color.rgb_to_grayscale(enhanced_low)
        edge_map = kornia.filters.canny(low_freq_gray)[1]
        low_freq_up = nn.functional.interpolate(enhanced_low, size=(pyr_input[-2].shape[2], pyr_input[-2].shape[3]))
        gauss_input_up = nn.functional.interpolate(gauss_input[-1],
                                                   size=(pyr_input[-2].shape[2], pyr_input[-2].shape[3]))
        edge_map_up = nn.functional.interpolate(edge_map, size=(pyr_input[-2].shape[2], pyr_input[-2].shape[3]))
        concat_imgs = torch.cat([gauss_input[-2], edge_map_up, low_freq_up, gauss_input_up], 1)
        pyr_reconstruct_results = self.block(concat_imgs, gauss_input, pyr_input, enhanced_low)
        return pyr_reconstruct_results
