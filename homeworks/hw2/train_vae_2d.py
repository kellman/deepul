import torch
from torch import nn
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
import numpy as np
import os

from deepul.hw2_helper import *


def log_normal(x, mean, log_var, eps=1e-5):
    c = - 0.5 * np.log(2*np.pi)
    return c - log_var/2 - (x - mean)**2 / (2 * torch.exp(log_var) + eps)


class VAE(nn.Module):
    def __init__(self, input_features, latent_features):
        super().__init__()

        self.input_features = input_features
        self.latent_features = latent_features

        # We encode the data onto the latent space using two linear layers
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=(self.latent_features*2))
        )
        
        # The latent code must be decoded into the original image
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.input_features)
        )
    
    def forward(self, x):
        # encode
        parameters = self.encoder(x)
        mu, log_var = torch.chunk(parameters, 2, dim=-1)

        # reparameterization trick
        z = self.reparameterize(mu, log_var)

        # decode
        x_sample = self.decoder(z)

        x_mu, x_log_var = torch.chunk(x_sample, 2, dim=-1)
        
        outputs["x_hat"] = self.reparameterize(x_mu,x_log_var)  
        outputs["z"] = z
        outputs["mu"] = mu
        outputs["log_var"] = log_var
        outputs["x_mu"] = x_mu
        outputs["x_log_var"] = x_log_var
        return outputs

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def sample(self, z):
        # z is drawn from random distribution
        x_sample = self.decoder(z)
        x_mu, x_log_var = torch.chunk(x_sample, 2, dim=-1)
        full_generation_path = self.reparameterize(x_mu, x_log_var)  
        without_decoder_noise = x_mu
        return full_generation_path, without_decoder_noise

    def loss(self, x, x_mu, x_log_var):

        # log reconstruction loss
        R = torch.sum(-1 * log_normal(x, x_mu, x_log_var))


        



def main():
    ### Visualize data
    # visualize_q1_data('a', 1)
    # plt.savefig('q1a_data.png')
    # visualize_q1_data('b', 1)
    # plt.savefig('q1b_data.png')

    ### Get data
    train_data, test_data = q1_sample_data('a', 1)
    train_data = torch.Tensor(train_data)
    test_data = torch.Tensor(test_data)
    print(train_data.shape)
    print(test_data.shape)

    ### initialize model
    model = VAE(2, 2)

    tmp = train_data[100:300, :]
    print(tmp.shape)
    output = model(tmp)
    print(output.shape)




if __name__ == '__main__':
    main()