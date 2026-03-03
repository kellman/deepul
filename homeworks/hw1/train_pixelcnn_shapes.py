from deepul.hw1_helper import visualize_q2a_data
from pixelcnn import PixelCNN, mixture_logistics

import torch
from torch import nn
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
import numpy as np


def load_data():
    shape_data = np.load('/workspace/deepul/homeworks/hw1/data/shapes.pkl', allow_pickle=True)
    train_data, test_data, train_labels, test_labels = shape_data['train'], shape_data['test'], shape_data['train_labels'], shape_data['test_labels']
    return train_data, test_data

def visualize_data(data):
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
    plt.savefig('shapes_visualization.png')

def visualize_loss(train_losses, val_losses, test_losses):
    plt.figure(figsize=(10, 5))
    epoch = np.linspace(0, len(test_losses) - 1, len(test_losses))
    iterations = np.linspace(0, len(test_losses) - 1, len(train_losses))
    plt.plot(iterations, train_losses, label='Train Loss')
    plt.plot(epoch, val_losses, label='Validation Loss')
    plt.plot(epoch, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation, Testing Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('loss_curves.png')

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
    train_data, test_data = load_data()
    train_data = train_data.transpose((0, 3, 1, 2)) # (B, H, W, C) -> (B, C, H, W)
    test_data = test_data.transpose((0, 3, 1, 2))
    visualize_data(train_data)

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

    model = PixelCNN()
    model.train().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, test_losses = [], [], []

    for epoch in range(N_epochs):
        for idx, batch in tqdm.tqdm(enumerate(train_dataloader)):
            batch = batch.cuda()
            predictions = model(batch)
            loss = F.binary_cross_entropy(predictions, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().cpu().numpy())
        
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
    visualize_loss(train_losses, val_losses, test_losses)

if __name__ == '__main__':
    main()