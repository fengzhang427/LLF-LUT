import torch
import torch.nn as nn
import torch.nn.functional as F
from .Spatial_Transformer import Spatial_Transformer
from .PPB import PPB, Lap_Pyramid_Conv
from .LUT import Generator3DLUT_identity, Generator3DLUT_zero, TV_3D


class LLF_LUT(nn.Module):
    def __init__(self, config):
        super(LLF_LUT, self).__init__()
        self.transformer_config = config['transformer']
        self.filter_config = config['filter']
        self.LUT_config = config['LUT']
        self.pad_size = self.filter_config['low_freq_resolution']
        self.device = torch.device('cuda' if config['gpu_ids'] is not None else 'cpu')

        # define transformer model
        self.transformer = Spatial_Transformer(
            in_chans=self.transformer_config['input_channel'],
            embed_dim=self.transformer_config['embed_dim'],
            num_classes=self.transformer_config['num_classes'],
            out_chans=self.transformer_config['output_channel'],
            depths=self.transformer_config['depths'],
            num_heads=self.transformer_config['num_heads'],
            window_sizes=self.transformer_config['window_sizes'],
            back_RBs=self.transformer_config['back_RBs'],
            recon_type=self.transformer_config['recon_type']
        )

        # define Laplacian filter
        self.laplacian_filter = PPB(self.filter_config)
        self.pyramid = Lap_Pyramid_Conv(self.filter_config['num_lap'], self.filter_config['channels'], self.device)

        # define learnable LUTs
        self.LUT0 = Generator3DLUT_identity(dim=self.LUT_config['LUT_dim'])
        self.LUT1 = Generator3DLUT_zero(dim=self.LUT_config['LUT_dim'])
        self.LUT2 = Generator3DLUT_zero(dim=self.LUT_config['LUT_dim'])

        # Load TV_loss
        self.TV3 = TV_3D(dim=self.LUT_config['LUT_dim'])
        cuda = True if config['gpu_ids'] is not None else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.TV3.weight_r = self.TV3.weight_r.type(Tensor)
        self.TV3.weight_g = self.TV3.weight_g.type(Tensor)
        self.TV3.weight_b = self.TV3.weight_b.type(Tensor)
    #
    # def check_image_size(self, x):
    #     _, _, h, w = x.size()
    #     mod_pad_h = self.pad_size - h
    #     mod_pad_w = self.pad_size - w
    #     try:
    #         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
    #     except BaseException:
    #         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant")
    #     return x

    def forward(self, input_image):
        B, C, H, W = input_image.size()

        pyr_input = self.pyramid.pyramid_decom(input_image)
        pyr_input_low = pyr_input[-1]
        pred_weight, pred_weight_point = self.transformer(pyr_input_low)

        enhanced_low0 = self.LUT0(pyr_input_low)
        enhanced_low1 = self.LUT1(pyr_input_low)
        enhanced_low2 = self.LUT2(pyr_input_low)
        enhanced_full0 = self.LUT0(input_image)
        enhanced_full1 = self.LUT1(input_image)
        enhanced_full2 = self.LUT2(input_image)

        # pred_weight = F.interpolate(pred_weight, size=(pyr_input_low.shape[2], pyr_input_low.shape[3]), mode='bicubic',
        #                             align_corners=True)

        enhanced_low = pred_weight[:, :3] * enhanced_low0 + pred_weight[:, 3:6] * enhanced_low1 + \
                       pred_weight[:, 6:9] * enhanced_low2
        enhanced_full = pred_weight_point[:, 0] * enhanced_full0 + pred_weight_point[:, 1] * enhanced_full1 + \
                        pred_weight_point[:, 2] * enhanced_full2

        # remapping function
        gauss_enhanced_full = self.pyramid.gauss_decom(enhanced_full)
        pyr_enhanced_full = self.pyramid.pyramid_decom(enhanced_full)
        pyr_reconstruct_results = self.laplacian_filter(gauss_enhanced_full, pyr_enhanced_full, enhanced_low)
        enhanced_image = self.pyramid.pyramid_recons(pyr_reconstruct_results)

        # define smooth loss and tv loss
        weights_norm = torch.mean(pred_weight ** 2)
        # weights_norm = torch.mean(pred_weight_point ** 2)
        tv0, mn0 = self.TV3(self.LUT0)
        tv1, mn1 = self.TV3(self.LUT1)
        tv2, mn2 = self.TV3(self.LUT2)
        tv_cons = tv0 + tv1 + tv2
        mn_cons = mn0 + mn1 + mn2

        loss_smooth = weights_norm + tv_cons if self.LUT_config['lambda_smooth'] > 0 else 0
        loss_mono = mn_cons if self.LUT_config['lambda_mono'] > 0 else 0
        loss_LUT = self.LUT_config['lambda_smooth'] * loss_smooth + self.LUT_config['lambda_mono'] * loss_mono

        return enhanced_image, pyr_reconstruct_results, loss_LUT
