import numpy

import torch
from torch import nn
import torch.nn.functional as F
from masked_conv import MaskedConv2d, MaskedResidualBlock

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_features):
        super().__init__(num_features)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ColorPixelCNN(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=120, kernel_size=7, num_layers=8, bits_per_channel=2):
        super().__init__()
        self.in_channels = in_channels
        self.channels_per_color = 2 ** bits_per_channel
        self.output_channels = in_channels * self.channels_per_color

        layers = nn.Sequential()
        # first layer with mask type A to exclude current pixel
        layers.append(MaskedConv2d(in_channels, hidden_channels, kernel_size, mask_type='A', padding=kernel_size//2))
        layers.append(LayerNorm2d(hidden_channels))
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
        preds = self.net(x)
        return preds

    def _normalize_inputs(self, x):
        x = x.float() # ensure input is float
        x = x / self.channels_per_color # scale pixel values to [0, 1]
        x = 2.0 * x - 1.0 # scale pixel values from [0, 1] to [-1, 1]
        return x

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
            loss += F.cross_entropy(preds[:, start:end, :, :], targets[:, c, :, :].long(), reduction='mean')
        return loss / self.in_channels

    def sample(self, num_samples, height, width, device='cuda'):
        """
        Generate new samples from the model.

        args:
            num_samples: number of samples to generate
            height: height of generated images
            width: width of generated images
            device: device to run sampling on
        
        returns:
            samples: (num_samples, in_channels, height, width) generated images with pixel values in [0, channels_per_color]
        """
        samples = torch.zeros((num_samples, self.in_channels, height, width), device=device)
        init_value = torch.randint(0, self.channels_per_color, (num_samples, self.in_channels), device=device).float()
        samples[:, :, 0, 0] = init_value

        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    preds = self.forward(samples) # (num_samples, output_channels, height, width)
                    for c in range(self.in_channels):
                        start = c * self.channels_per_color
                        end = (c + 1) * self.channels_per_color
                        pixel_preds = preds[:, start:end, i, j].softmax(dim=1) # (num_samples, channels_per_color)
                        pixel_values = torch.multinomial(pixel_preds, num_samples=1).squeeze(1) # (num_samples,)
                        samples[:, c, i, j] = pixel_values.float()
        return samples

