"""Real-time Spatial Temporal Transformer.
"""
import numpy as np
import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
from .layers import make_layer, ResidualBlock_noBN, EncoderLayer, DecoderLayer, InputProj, Downsample, Upsample
from timm.models.layers import trunc_normal_


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class Spatial_Transformer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, num_classes=3, out_chans=3,
                 depths=None,
                 num_heads=None,
                 window_sizes=None,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 back_RBs=0, recon_type="enhancement"):
        """
        Args:
            in_chans (int, optional): Number of input image channels. Defaults to 3.
            embed_dim (int, optional): Number of projection output channels. Defaults to 32.
            num_classes (int, optional): Number of weight predictions. Defaults to 3.
            out_chans (int, optional): Number of output image channels. Defaults to 3.
            depths (list[int], optional): Depths of each Transformer stage. Defaults to [2, 2, 2, 2, 2, 2, 2, 2].
            num_heads (list[int], optional): Number of attention head of each stage. Defaults to [2, 4, 8, 16, 16, 8, 4, 2].
            window_size (tuple[int], optional): Window size. Defaults to (8, 8).
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4..
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop_rate (float, optional): Dropout rate. Defaults to 0.
            attn_drop_rate (float, optional): Attention dropout rate. Defaults to 0.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.1.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            patch_norm (bool, optional): If True, add normalization after patch embedding. Defaults to True.
            back_RBs (int, optional): Number of residual blocks for super resolution. Defaults to 10.
            recon_type (str): The reconstruction module.
        """
        super(Spatial_Transformer, self).__init__()

        if depths is None:
            depths = [1, 1, 1, 1, 1, 1, 1, 1]
        if num_heads is None:
            num_heads = [2, 4, 8, 16, 16, 8, 4, 2]
        if window_sizes is None:
            window_sizes = [(4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4)]
        self.num_layers = len(depths)
        self.num_enc_layers = self.num_layers // 2
        self.num_dec_layers = self.num_layers // 2
        self.scale = 2 ** (self.num_enc_layers - 1)
        self.image_size = 4 * self.scale
        dec_depths = depths[self.num_enc_layers:]
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.in_channels = in_chans
        self.out_channels = out_chans
        self.recon = recon_type

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth 
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        dec_dpr = enc_dpr[::-1]

        self.input_proj = InputProj(in_channels=self.in_channels, embed_dim=embed_dim,
                                    kernel_size=3, stride=1, act_layer=nn.LeakyReLU)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i_layer in range(self.num_enc_layers):
            encoder_layer = EncoderLayer(
                dim=embed_dim,
                depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_sizes[i_layer], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=enc_dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer
            )
            downsample = Downsample(embed_dim, embed_dim)
            self.encoder_layers.append(encoder_layer)
            self.downsample.append(downsample)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i_layer in range(self.num_dec_layers):
            decoder_layer = DecoderLayer(
                dim=embed_dim,
                depth=depths[i_layer + self.num_enc_layers],
                num_heads=num_heads[i_layer + self.num_enc_layers],
                window_size=window_sizes[i_layer + self.num_enc_layers], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dec_dpr[sum(dec_depths[:i_layer]):sum(dec_depths[:i_layer + 1])],
                norm_layer=norm_layer
            )
            self.decoder_layers.append(decoder_layer)
            if i_layer != self.num_dec_layers - 1:
                upsample = Upsample(embed_dim, embed_dim)
                self.upsample.append(upsample)

        # Weight Prediction
        self.head = nn.Linear(embed_dim * 64, num_classes) if num_classes > 0 else nn.Identity()

        # Reconstruction block
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=embed_dim)
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, back_RBs)
        self.recon_conv = nn.Conv2d(embed_dim, 64, 3, 1, 1, bias=True)
        # Upsampling
        self.upconv1 = nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(embed_dim, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, self.out_channels, 3, 1, 1, bias=True)

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.pad_size = 8

        self.apply(_init_weights)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = self.pad_size - h
        mod_pad_w = self.pad_size - w
        try:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        except BaseException:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant")
        return x

    def forward(self, x):
        B, C, H, W = x.size()
        upsample_x = F.interpolate(x, (H * 4, W * 4), mode='bilinear', align_corners=False)
        res_x = x

        x = self.input_proj(x)  # B, C, H, W

        if H < W and H < self.image_size:
            scale = self.image_size / H
            Hp, Wp = round(H * scale), round(W * scale)
        elif H > W and W < self.image_size:
            scale = self.image_size / W
            Hp, Wp = round(H * scale), round(W * scale)
        else:
            Hp, Wp = H, W
        Hp = int(np.ceil(Hp / self.scale)) * self.scale
        Wp = int(np.ceil(Wp / self.scale)) * self.scale
        x = F.pad(x, (0, Wp - W, 0, Hp - H))

        encoder_features = []
        for i_layer in range(self.num_enc_layers):
            x = self.encoder_layers[i_layer](x)
            encoder_features.append(x)
            if i_layer != self.num_enc_layers - 1:
                x = self.downsample[i_layer](x)

        # latent_x = x
        latent_x = self.check_image_size(x)
        class_num = self.head(latent_x.flatten()).unsqueeze(0)
        y = x
        # y = latent_x

        for i_layer in range(self.num_dec_layers):
            y = self.decoder_layers[i_layer](y, encoder_features[-i_layer - 1])
            if i_layer != self.num_dec_layers - 1:
                y = self.upsample[i_layer](y)

        y = y[:, :, :H, :W].contiguous()
        out = self.recon_trunk(y)

        if self.recon == "Super-resolution":
            # Super-resolution
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

            out = self.lrelu(self.HRconv(out))
            out = self.conv_last(out)
            _, _, H, W = out.size()
            outs = out.view(B, -1, H, W)
            outs = outs + upsample_x.permute(0, 2, 1, 3, 4)
        else:
            if self.in_channels == self.out_channels:
                outs = res_x + self.conv_last(self.recon_conv(out))
            else:
                outs = self.conv_last(self.recon_conv(out))

        return outs, class_num


if __name__ == '__main__':
    model = Spatial_Transformer(
        in_chans=3,
        embed_dim=32,
        num_classes=3,
        out_chans=9,
        depths=[1, 1, 1, 1, 1, 1, 1, 1],
        num_heads=[2, 4, 8, 16, 16, 8, 4, 2],
        window_sizes=[[4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]],
        back_RBs=0,
        recon_type="enhancement"
    )
    x = torch.randn(1, 3, 64, 64)
    img, score = model(x)
    print(score)
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    print(f"Number of parameters {num_params / 10 ** 6: 0.2f}")
