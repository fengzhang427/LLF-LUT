import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def window_partition(x, window_size):
    """Partition the input sequences into several windows along spatial
    dimensions.

    Args:
        x (torch.Tensor): (B, H, W, C)
        window_size (tuple[int]): Window size

    Returns:
        windows: (B*nW, Wh, Ww, C)
    """
    B, H, W, C = x.shape
    # B, num_Hwin, Wh, num_Wwin, Ww, C
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, B, H, W):
    """Reverse window partition.

    Args:
        windows (torch.Tensor): (B*nW, Wh, Ww, C)
        window_size (tuple[int]): Window size
        B (int): Number of batches
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Adjust window size and shift size based on the size of the input.

    Args:
        x_size (tuple[int]): The shape of x.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int], optional): Shift size. Defaults to None.

    Returns:
        use_window_size: Window size for use.
        use_shift_size: Shift size for use.
    """
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self/cross attention (W-MSA/W-MCA) module with relative 
    position bias. 
    It supports both of shifted and non-shifted window.
    """

    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Initialization function.

        Args:
            dim (int): Number of input channels.
            window_size (tuple[int]): The size of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            attn_drop (float, optional): Dropout ratio of attention weight. Defaults to 0.0
            proj_drop (float, optional): Dropout ratio of output. Defaults to 0.0
        """
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads  # nH
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # * 2*Wh-1 * 2*Ww-1, nH

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_q = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_kv = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_q_flatten = torch.flatten(coords_q, 1)  # 2, D1*Wh*Ww
        coords_kv_flatten = torch.flatten(coords_kv, 1)  # 2, D2*Wh*Ww
        relative_coords = coords_q_flatten[:, :, None] - coords_kv_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv=None, mask=None):
        """Forward function.

        Args:
            q (torch.Tensor): (B*nW, Wh*Ww, C)
            kv (torch.Tensor): (B*nW, Wh*Ww, C). Defaults to None.
            mask (torch.Tensor, optional): Mask for shifted window attention (nW, Wh*Ww, Wh*Ww). Defaults to None.

        Returns:
            torch.Tensor: (B*nW, Wh*Ww, C)
        """
        kv = q if kv is None else kv
        B_, N1, C = q.shape  # N1 = Wh*Ww, B_ = B*nW
        B_, N2, C = kv.shape  # N2 = Wh*Ww, B_ = B*nW

        q = self.q(q).reshape(B_, N1, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(kv).reshape(B_, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]  # B_, nH, N1(2), C
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, nH, N1, N2

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N1, N2, -1)  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, Wh*Ww, Wh*Ww

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N1, N2) + mask.unsqueeze(1).unsqueeze(0)  # B, nW, nH, Wh*Ww, Wh*Ww
            attn = attn.view(-1, self.num_heads, N1, N2)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class matmul(nn.Module):
    def __init__(self):
        super(matmul, self).__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x


class FourierWindowAttention(nn.Module):
    """Window based multi-head self/cross attention (W-MSA/W-MCA) module with relative
    position bias.
    It supports both of shifted and non-shifted window.
    """

    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Initialization function.

        Args:
            dim (int): Number of input channels.
            window_size (tuple[int]): The size of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            attn_drop (float, optional): Dropout ratio of attention weight. Defaults to 0.0
            proj_drop (float, optional): Dropout ratio of output. Defaults to 0.0
        """
        super(FourierWindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads  # nH
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # * 2*Wh-1 * 2*Ww-1, nH

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_q = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_kv = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_q_flatten = torch.flatten(coords_q, 1)  # 2, D1*Wh*Ww
        coords_kv_flatten = torch.flatten(coords_kv, 1)  # 2, D2*Wh*Ww
        relative_coords = coords_q_flatten[:, :, None] - coords_kv_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.mat = matmul()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.

        Args:
            x (torch.Tensor): (B*nW, Wh*Ww, C)
            mask (torch.Tensor, optional): Mask for shifted window attention (nW, Wh*Ww, Wh*Ww). Defaults to None.

        Returns:
            torch.Tensor: (B*nW, Wh*Ww, C)
        """
        B, N, C = x.shape  # N1 = Wh*Ww, B_ = B*nW

        x_f = torch.fft.rfftn(x, dim=-2, norm='ortho')
        x_f = torch.cat([x_f.real, x_f.imag], dim=-2)

        B_, N_, C_ = x_f.shape

        # Spatial Attention
        qkv_spatial = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_spatial[0], qkv_spatial[1], qkv_spatial[2]  # B_, nH, N1(2), C
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1)  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, Wh*Ww, Wh*Ww
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)  # B, nW, nH, Wh*Ww, Wh*Ww
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)

        # Spectral Attention
        qkv_spectral = self.qkv(x_f).reshape(B_, N_, 3, self.num_heads, C_ // self.num_heads).permute(2, 0, 3, 1, 4)
        q_f, k_f, v_f = qkv_spectral[0], qkv_spectral[1], qkv_spectral[2]  # make torchscript happy (cannot use tensor as tuple)
        attn_f = (self.mat(q_f, k_f.transpose(-2, -1))) * self.scale
        attn_f = self.softmax(attn_f)
        attn_f = self.attn_drop(attn_f)
        x_f = self.mat(attn_f, v_f).transpose(1, 2).reshape(B_, N_, C_)
        x_f = torch.complex(*x_f.split(N_ // 2, dim=-2))
        x_f = torch.fft.irfftn(x_f, s=N, dim=-2, norm='ortho')

        x = x + x_f

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class EncoderTransformerBlock(nn.Module):
    """spatial encoder transformer block.
    """

    def __init__(self, dim, num_heads, window_size=(8, 8),
                 shift_size=(0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        """Initialization function.

        Args:
            dim (int): Number of input channels. 
            num_heads (int): Number of attention heads.
            window_size (tuple[int], optional): Window size. Defaults to 8.
            shift_size (tuple[int], optional): Shift size. Defaults to 0.
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional):  Stochastic depth rate. Defaults to 0.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super(EncoderTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-win_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        # self.attn = WindowAttention(
        #     dim,
        #     window_size=self.window_size, num_heads=num_heads,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
        #     proj_drop=drop
        # )
        self.attn = FourierWindowAttention(
            dim,
            window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        """Forward function.

        Args:
            x (torch.Tensor): (B, H, W, C)
            mask_matrix (torch.Tensor): (nW*B, Wh*Ww, Wh*Ww)

        Returns:
            torch.Tensor: (B, H, W, C)
        """
        B, H, W, C = x.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        shortcut = x
        x = self.norm1(x)

        # Padding
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, window_size[0] * window_size[1], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)[0]  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], C)
        shifted_x = window_reverse(attn_windows, window_size, B, Hp, Wp)  # B, H, W, C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DecoderTransformerBlock(nn.Module):
    """spatial decoder transformer block.
    """

    def __init__(self, dim, num_heads, window_size=(8, 8),
                 shift_size=(0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        """Initialization function.

        Args:
            dim (int): Number of input channels. 
            num_heads (int): Number of attention heads.
            window_size (tuple[int], optional): Window size. Defaults to 8.
            shift_size (tuple[int], optional): Shift size. Defaults to 0.
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional):  Stochastic depth rate. Defaults to 0.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super(DecoderTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-win_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn1 = WindowAttention(
            dim, window_size=self.window_size,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.attn2 = WindowAttention(
            dim, window_size=self.window_size,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_kv, mask_matrix_q, mask_matrix_qkv):
        """Forward function.

        Args:
            x (torch.Tensor): (B, H, W, C)
            attn_kv (torch.Tensor): (B, H, W, C)
            mask_matrix_q (torch.Tensor): (nW*B, Wh*Ww, Wh*Ww)
            mask_matrix_qkv (torch.Tensor): (nW*B, Wh*Ww, Wh*Ww)

        Returns:
            torch.Tensor: (B, H, W, C)
        """
        B, H, W, C = x.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        shortcut = x
        x = self.norm1(x)

        # Padding
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask_q = mask_matrix_q
            attn_mask_qkv = mask_matrix_qkv
        else:
            shifted_x = x
            attn_mask_q = None
            attn_mask_qkv = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, window_size[0] * window_size[1], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA for query
        attn_windows = self.attn1(x_windows, mask=attn_mask_q)[0]  # nW*B, window_size*window_size, C
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], C)
        shifted_x = window_reverse(attn_windows, window_size, B, Hp, Wp)  # B, Hp, Wp, C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.norm2(x)
        attn_kv = self.norm_kv(attn_kv)
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
        attn_kv = F.pad(attn_kv, (0, 0, 0, pad_r, 0, pad_b, 0, 0))

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            shifted_attn_kv = torch.roll(attn_kv, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask_q = mask_matrix_q
            attn_mask_qkv = mask_matrix_qkv
        else:
            shifted_x = x
            shifted_attn_kv = attn_kv
            attn_mask_q = None
            attn_mask_qkv = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C
        attn_kv_windows = window_partition(shifted_attn_kv, window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, window_size[0] * window_size[1], C)  # nW*B, window_size*window_size, C
        attn_kv_windows = attn_kv_windows.view(-1, window_size[0] * window_size[1],
                                               C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn2(x_windows, attn_kv_windows, mask=attn_mask_qkv)[0]  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], C)
        shifted_x = window_reverse(attn_windows, window_size, B, Hp, Wp)  # B, H, W, C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x


class EncoderLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=(8, 8),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        """Encoder layer

        Args:
            dim (int): Number of feature channels
            depth (int): Depths of this stage.
            num_heads (int): Number of attention head.
            window_size (tuple[int], optional): Window size. Defaults to (8, 8).
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional): Stochastic depth rate. Defaults to 0.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super(EncoderLayer, self).__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            EncoderTransformerBlock(dim=dim, num_heads=num_heads,
                                    window_size=window_size,
                                    shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): (B, C, H, W)

        Returns:
            torch.Tensor: (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C

        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1, H, W, 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # nW, Wh, Ww, 1
        mask_windows = mask_windows.view(-1, window_size[0] * window_size[1])  # nW, Wh*Ww
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, Wh*Ww, Wh*Ww
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.permute(0, 3, 1, 2)  # B, D, C, H, W

        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=(8, 8),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        """Decoder layer

        Args:
            dim (int): Number of feature channels
            depth (int): Depths of this stage.
            num_heads (int): Number of attention head.
            window_size (tuple[int], optional): Window size. Defaults to (8, 8).
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional): Stochastic depth rate. Defaults to 0.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super(DecoderLayer, self).__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            DecoderTransformerBlock(dim=dim, num_heads=num_heads,
                                    window_size=window_size,
                                    shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, attn_kv):
        """Forward function.

        Args:
            x (torch.Tensor): (B, C, H, W)
            attn_kv (torch.Tensor): (B, C, H, W)

        Returns:
            torch.Tensor: (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        attn_kv = attn_kv.permute(0, 2, 3, 1)  # B, H, W, C

        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]

        img_mask_q = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1, H, W, 1
        img_mask_kv = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1, H, W, 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask_q[:, h, w, :] = cnt
                img_mask_kv[:, h, w, :] = cnt
                cnt += 1

        mask_windows_q = window_partition(img_mask_q, window_size)  # nW, Wh, Ww, 1
        mask_windows_kv = window_partition(img_mask_kv, window_size)  # nW, Wh, Ww, 1
        mask_windows_q = mask_windows_q.view(-1, window_size[0] * window_size[1])  # nW, Wh*Ww
        mask_windows_kv = mask_windows_kv.view(-1, window_size[0] * window_size[1])  # nW, Wh*Ww
        attn_mask_q = mask_windows_q.unsqueeze(1) - mask_windows_q.unsqueeze(2)  # nW, Wh*Ww, Wh*Ww
        attn_mask_qkv = mask_windows_kv.unsqueeze(1) - mask_windows_q.unsqueeze(2)  # nW, Wh*Ww, Wh*Ww
        attn_mask_q = attn_mask_q.masked_fill(attn_mask_q != 0, float(-100.0)).masked_fill(attn_mask_q == 0, float(0.0))
        attn_mask_qkv = attn_mask_qkv.masked_fill(attn_mask_qkv != 0, float(-100.0)).masked_fill(attn_mask_qkv == 0,
                                                                                                 float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_kv, attn_mask_q, attn_mask_qkv)

        x = x.permute(0, 3, 1, 2)  # B, C, H, W

        return x


class InputProj(nn.Module):
    """Image input projection

    Args:
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of output channels. Default: 32.
        kernel_size (int): Size of the convolution kernel. Default: 3
        stride (int): Stride of the convolution. Default: 1
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        act_layer (nn.Module): Activation layer. Default: nn.LeakyReLU.
    """

    def __init__(self, in_channels=3, embed_dim=32, kernel_size=3, stride=1,
                 norm_layer=None, act_layer=nn.LeakyReLU):
        super(InputProj, self).__init__()

        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): (B, C, H, W)

        Returns:
            torch.Tensor: (B, C, H, W)
        """
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): (B, C, H, W)

        Returns:
            torch.Tensor: (B, C, H, W)
        """
        out = self.conv(x)
        return out


class Upsample(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2),
        )

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): (B, C, H, W)

        Returns:
            torch.Tensor: (B, C, H, W)
        """
        out = self.deconv(x)
        return out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ResidualBlock_1(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock_1, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, drop_out_rate=0.):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, padding=0,
                               stride=1,
                               groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, padding=1,
                               stride=1,
                               groups=1,
                               bias=True)

        self.conv3 = nn.Conv2d(in_channels=in_features // 2, out_channels=in_features, kernel_size=1, padding=0,
                               stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_features // 2, out_channels=in_features // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        self.norm1 = LayerNorm(in_features, data_format="channels_first")

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, in_features, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        return inp + x * self.beta


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
