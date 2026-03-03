from deepul.hw1_helper import visualize_q2a_data
from pixelcnn import PixelCNN, mixture_logistics

import torch
from torch import nn
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
import numpy as np
import os


def load_data():
    shape_data = np.load('/workspace/deepul/homeworks/hw1/data/mnist.pkl', allow_pickle=True)
    train_data, test_data, train_labels, test_labels = shape_data['train'], shape_data['test'], shape_data['train_labels'], shape_data['test_labels']
    return train_data, test_data

def visualize_data(data, output_name='shapes_visualization.png'):
    """data: (B, 1, H, W) numpy array of pixel values in [0, 1]
    """
    # visualize first 25 images from train_data (shape: B,1,H,W)
    imgs = data[:25, 0, :, :] # (N, H, W)
    rows = cols = 5

    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < imgs.shape[0]:
            ax.imshow(imgs[i], cmap='gray', vmin=0, vmax=np.max(data))
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_name)
    plt.close()

def visualize_loss(train_losses, val_losses, test_losses, output_name='loss_curves.png'):
    plt.figure(figsize=(10, 5))
    epoch = np.linspace(0, len(test_losses) - 1, len(test_losses))
    iterations = np.linspace(0, len(test_losses) - 1, len(train_losses))
    plt.plot(iterations, np.log10(np.array(train_losses) + 1e-12), label='Train Loss')
    plt.plot(epoch, np.log10(np.array(val_losses) + 1e-12), label='Validation Loss')
    plt.plot(epoch, np.log10(np.array(test_losses) + 1e-12), label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('Training, Validation, Testing Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(output_name)
    plt.close()

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, data:torch.Tensor):
        data -= torch.min(data)
        data /= torch.max(data)
        self.data = data.float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    output_folder = 'q2b_results'
    os.makedirs(output_folder, exist_ok=True)

    train_data, test_data = load_data()
    train_data = train_data.transpose((0, 3, 1, 2)) # (B, H, W, C) -> (B, C, H, W)
    test_data = test_data.transpose((0, 3, 1, 2))
    visualize_data(test_data, output_name=f'{output_folder}/test_examples.png')

    N_batch_size = 128
    N_epochs = 30
    lr = 1e-3
    val_proportion = 0.1
    val_size = int(len(train_data) * val_proportion)
    torch.random.manual_seed(42)

    # Convert data to tensors
    val_data = train_data[-val_size:]
    train_data = train_data[:-val_size]

    test_data = torch.from_numpy(test_data).float()
    val_data = torch.from_numpy(val_data).float()
    train_data = torch.from_numpy(train_data).float()
    print(f'Train data shape: {train_data.shape}')
    print(f'Val data shape: {val_data.shape}')
    print(f'Test data shape: {test_data.shape}')

    train_dataset = BinaryDataset(train_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=N_batch_size, shuffle=True)

    val_dataset = BinaryDataset(val_data)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=N_batch_size, shuffle=False)

    test_dataset = BinaryDataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=N_batch_size, shuffle=False)

    model = PixelCNN(num_layers=10)
    model.train().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, test_losses = [], [], []

    for epoch in range(N_epochs):
        model.train()
        for idx, batch in tqdm.tqdm(enumerate(train_dataloader)):
            batch = batch.cuda()
            predictions = model(batch)
            loss = F.binary_cross_entropy(predictions, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().cpu().numpy())

        model.eval()
        with torch.no_grad():
            # generate samples and save them
            N_samples = 25
            samples = model.sample(N_samples, train_data.shape[2], train_data.shape[3]).cpu().numpy() # (25, 1, H, W)
            visualize_data(samples, output_name=f'{output_folder}/samples_epoch_{epoch}.png')
        
        with torch.no_grad():
            val_loss = 0
            for val_batch in val_dataloader:
                val_batch = val_batch.cuda()
                val_predictions = model(val_batch)
                val_loss += F.binary_cross_entropy(val_predictions, val_batch)
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss.detach().cpu().numpy())
            
            test_loss = 0
            for test_batch in test_dataloader:
                test_batch = test_batch.cuda()
                test_predictions = model(test_batch)
                test_loss += F.binary_cross_entropy(test_predictions, test_batch)
            test_loss /= len(test_dataloader)
            test_losses.append(test_loss.detach().cpu().numpy())
        print(f'val loss: {val_losses[-1]:.4f}, test loss: {test_losses[-1]:.4f}')
    visualize_loss(train_losses, val_losses, test_losses, output_name=f'{output_folder}/loss_curves.png')

if __name__ == '__main__':
    main()