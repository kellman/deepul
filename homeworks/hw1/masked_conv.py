import torch
from torch import nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """
    Masked convolution for PixelCNN.
    mask_type: 'A' (exclude center) or 'B' (include center)
    """
    def __init__(self, in_channels, out_channels, kernel_size, mask_type='B', stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        assert mask_type in ('A', 'B')
        kh, kw = self.weight.shape[2], self.weight.shape[3]
        center_h, center_w = kh // 2, kw // 2

        mask = torch.ones_like(self.weight.data)
        # zero out rows below center
        mask[:, :, center_h+1:, :] = 0
        # zero out columns to the right of center in center row
        mask[:, :, center_h, center_w+1:] = 0
        if mask_type == 'A':
            # exclude center pixel
            mask[:, :, center_h, center_w] = 0

        # register so it moves with device/dtype and isn't trainable
        self.register_buffer('mask', mask)

    def forward(self, x):
        masked_weight = self.weight * self.mask
        return F.conv2d(x, masked_weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

class MaskedResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.convdw1 = MaskedConv2d(channels, channels//2, 1, mask_type='B', padding=0)
        self.ln1 = nn.LayerNorm(channels)
        self.conv1 = MaskedConv2d(channels//2, channels//2, kernel_size, mask_type='B', padding=kernel_size//2)
        self.ln2 = nn.LayerNorm(channels)
        self.convdw2 = MaskedConv2d(channels//2, channels, 1, mask_type='B', padding=0)
        self.ln3 = nn.LayerNorm(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.ln1(self.convdw1(x)))
        out = self.relu(self.ln2(self.conv1(out)))
        out = self.relu(self.ln3(self.convdw2(out)))
        return out + x # Residual connection
