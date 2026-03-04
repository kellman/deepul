import numpy

import torch
from torch import nn
import torch.nn.functional as F
from masked_conv import MaskedConv2d, MaskedResidualBlock


class ColorPixelCNN(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=120, kernel_size=7, num_layers=10, bits_per_channel=2):
        super().__init__()
        self.in_channels = in_channels
        self.channels_per_color = 2 ** bits_per_channel
        self.output_channels = in_channels * self.channels_per_color

        layers = nn.Sequential()
        # first layer with mask type A to exclude current pixel
        layers.append(MaskedConv2d(in_channels, hidden_channels, kernel_size, mask_type='A', padding=kernel_size//2))
        layers.append(nn.LayerNorm(hidden_channels))
        layers.append(nn.ReLU(inplace=True))

        # subsequent layers with mask type B to include current pixel
        for i in range(num_layers - 1):
            layers.append(MaskedResidualBlock(hidden_channels, kernel_size=kernel_size))

        # final layers to produce output logits for each pixel value
        layers.append(MaskedConv2d(hidden_channels, hidden_channels, 1, mask_type='B'))
        layers.append(nn.ReLU(inplace=True))
        layers.append(MaskedConv2d(hidden_channels, self.output_channels, 1, mask_type='B'))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self._normalize_inputs(x)
        preds = self.net(x) # (B, 12, H, W) for 2 bits per channel
        preds = self._normalize_preds(preds)
        return preds

    def _normalize_inputs(self, x):
        # scale pixel values from [0, 1] to [-1, 1]
        return 2.0 * x - 1.0

    def _normalize_preds(self, preds):
        # normalize logits to get probabilities for each color channel
        for c in range(self.in_channels):
            start = c * self.channels_per_color
            end = (c + 1) * self.channels_per_color
            preds[:, start:end, :, :] = F.softmax(preds[:, start:end, :, :], dim=1)
        return preds

    def loss(self, preds, targets):
        """
        preds: (B, output_channels, H, W) - output logits for each pixel value (4 values per channel)
        targets: (B, in_channels, H, W) - original pixel values in [0, 1]
        """
        loss = 0.0
        for c in range(self.in_channels):
            start = c * self.channels_per_color
            end = (c + 1) * self.channels_per_color
            loss += F.cross_entropy(preds[:, start:end, :, :], targets[:, c, :, :].long(), reduction='sum')
        return loss / self.in_channels

    def sample(self, num_samples):
        pass
