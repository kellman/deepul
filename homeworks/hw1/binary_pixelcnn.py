import numpy

import torch
from torch import nn
import torch.nn.functional as F

class BinaryPixelCNN(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, output_channels=1, kernel_size=7, num_layers=6):
        super().__init__()
        layers = nn.Sequential()
        for i in range(num_layers):
            mask_type = 'A' if i == 0 else 'B'
            layers.append(MaskedConv2d(in_channels if i == 0 else hidden_channels, hidden_channels, kernel_size, mask_type=mask_type, padding=kernel_size//2))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(MaskedConv2d(hidden_channels, hidden_channels, 1, mask_type='B'))
        layers.append(nn.ReLU(inplace=True))
        layers.append(MaskedConv2d(hidden_channels, output_channels, 1, mask_type='B'))  # output logits for each pixel value
        layers.append(nn.Sigmoid())  # scale to [0, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(2.0 * x - 1.0)
    
    def sample(self, num_samples, img_height, img_width, device='cuda'):
        """
        Generate samples from a trained PixelCNN model.
        
        Args:
            model: Trained PixelCNN model
            num_samples: Number of images to generate
            img_height, img_width: Image dimensions
            device: Device to generate on
        
        Returns:
            samples: (num_samples, 1, img_height, img_width) tensor with values in [0, 1]
        """
        samples = torch.zeros(num_samples, 1, img_height, img_width, device=device)
        
        with torch.no_grad():
            # Iterate through each pixel position (row-major order)
            for i in range(img_height):
                for j in range(img_width):
                    # Get model predictions for current partial image
                    logits = self.forward(samples)  # (num_samples, 1, H, W)
                    
                    # Extract predictions for pixel at (i, j)
                    pixel_probs = logits[:, 0, i, j]  # (num_samples,)
                    
                    # Sample from Bernoulli distribution or use threshold
                    samples[:, 0, i, j] = torch.bernoulli(pixel_probs)
        return samples

def mixture_logistics(
    x: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    scales: torch.Tensor,
    max_val: int,
) -> torch.Tensor:
    """
    x: An (n,) numpy array of integers in {0, ..., d-1}
    mus: An (k,) numpy array of means for each logistic component
    sigmas: An (k,) numpy array of standard deviations for each logistic component
    scales: An (k,) numpy array of scales for each logistic component

    Returns
    - An (n,) numpy array of probabilities p_theta(x_i) for each x_i in x
    """
    prob = torch.zeros_like(x)
    for i in range(len(mus)):
        sig1 = torch.sigmoid((x + 0.5 - mus[i]) / sigmas[i]) * (x + 0.5 < max_val) + 1 * (x + 0.5 >= max_val)
        sig2 = torch.sigmoid((x - 0.5 - mus[i]) / sigmas[i]) * (x - 0.5 >= 0) + 0 * (x - 0.5 < 0)
        prob += scales[i] * (sig1 - sig2)
    return prob

class DiscreteMixtureLogistic(nn.Module):
    def __init__(self, K: int, max_val: int):
        super().__init__()
        self.K = K
        self.max_val = max_val
        self.mus = nn.Parameter(torch.rand(1, K) * max_val)  # [1, K] - means initialized randomly in [0, max_val]
        self.log_scales = nn.Parameter(torch.rand(1, K))
        self.logit_weights = nn.Parameter(torch.randn(1, K))

    def prob(self, x):
        """
        Computes the probability of discrete x under the mixture of logistics model.
        x: [batch, 1] - target values

        returns:
        - An (n,) numpy array of probabilities p_theta(x_i) for each x_i in x
        """
        with torch.no_grad():
            return mixture_logistics(
                x,
                self.mus.squeeze(0),
                torch.exp(self.log_scales).squeeze(0),
                F.softmax(self.logit_weights, dim=-1).squeeze(0),
                self.max_val
            )

    def forward(self, x):
        """
        x: [batch, 1] - target values (e.g. scaled to [-1, 1])
        """
        # 1. Get scales and inverse scales
        scales = torch.exp(self.log_scales)
        inv_scales = torch.exp(-self.log_scales)
        
        # 2. Calculate centered values at bin edges
        # Assuming x is scaled to [-1, 1] and bin width is 1
        centered_x = x - self.mus
        plus_in = inv_scales * (centered_x + 0.5)
        minus_in = inv_scales * (centered_x - 0.5)
        
        # 3. Stable log-CDF difference (The "Log-Sigmoid" trick)
        # log(sigmoid(plus) - sigmoid(minus))
        # For very large values, we approximate using the softplus derivative logic
        cdf_plus = torch.sigmoid(plus_in)
        cdf_minus = torch.sigmoid(minus_in)

        # Probabilities in log space for each component k
        # We use a clamp/softplus approach for the edges (0 and 255)
        log_probs_k = torch.log(torch.clamp(cdf_plus - cdf_minus, min=1e-12))
        
        # 4. Add log mixture weights (log_pi)
        # log_pi = log_softmax(logit_weights)
        log_pi = F.log_softmax(self.logit_weights, dim=-1)
        
        # 5. Final Negative Log Likelihood using LogSumExp
        # NLL = -log( sum( exp(log_pi + log_probs_k) ) )
        nll = -torch.logsumexp(log_pi + log_probs_k, dim=-1)
        return torch.mean(nll)
