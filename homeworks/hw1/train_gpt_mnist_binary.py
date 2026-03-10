from masked_gpt import GPT, sinusoidal_positional_encoding

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
        return self.data[idx].view(-1, self.data.shape[-1]) # flatten the image to a sequence (C, H*W)


def main():
    output_folder = 'q3a_binary_mnist_results'
    os.makedirs(output_folder, exist_ok=True)

    train_data, test_data = load_data()
    _, H, W, C = train_data.shape

    N_batch_size = 128
    N_epochs = 100
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

    # Visualize real data
    visualize_data(train_data[:25].permute(0, 3, 1, 2).cpu().numpy(), output_name=f'{output_folder}/real_shapes.png')

    train_dataset = BinaryDataset(train_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=N_batch_size, shuffle=True)

    val_dataset = BinaryDataset(val_data)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=N_batch_size, shuffle=False)

    test_dataset = BinaryDataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=N_batch_size, shuffle=False)

    d_model = 128
    n_heads = 4
    n_layers = 2
    d_ff = 512
    d_output = 1

    model = GPT(d_model, n_heads, d_ff, n_layers, d_output, max_seq_len=H*W).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, test_losses = [], [], []

    # test the sampler
    with torch.no_grad():
        N_samples = 25
        model.eval()
        samples = model.sample(N_samples)
        samples = samples.view(N_samples, H, W, 1).permute(0, 3, 1, 2).cpu().float().numpy()
        visualize_data(samples, output_name=f'{output_folder}/sample_gen_test.png')

    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(current_step - num_warmup_steps) / float(max(1, num_total_steps - num_warmup_steps)))))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    num_warmup_steps = 1000
    num_total_steps = len(train_dataloader) * N_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps)

    for epoch in range(N_epochs):
        model.train()
        for idx, batch in tqdm.tqdm(enumerate(train_dataloader)):
            # prepare data
            batch = batch.cuda()
            # compute model and loss
            full_batch = torch.cat([model.start_token.expand(batch.shape[0], -1, -1), batch], dim=1)
            predictions = model(full_batch)  # (B, H*W, d_output)
            predictions = predictions[:, :-1, :]
            loss = F.binary_cross_entropy(predictions, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.detach().cpu().numpy())
        model.eval()
        with torch.no_grad():
            # generate samples and save them
            N_samples = 25
            samples = model.sample(N_samples)
            samples = samples.view(N_samples, H, W, 1).permute(0, 3, 1, 2).cpu().float().numpy()
            visualize_data(samples, output_name=f'{output_folder}/samples_epoch_{epoch}.png')
        
        with torch.no_grad():
            val_loss = 0
            for val_batch in val_dataloader:
                val_batch = val_batch.cuda()
                val_full_batch = torch.cat([model.start_token.expand(val_batch.shape[0], -1, -1), val_batch], dim=1)
                val_predictions = model(val_full_batch)
                val_predictions = val_predictions[:, :-1, :]
                val_loss += F.binary_cross_entropy(val_predictions, val_batch)
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss.detach().cpu().numpy())
            
            test_loss = 0
            for test_batch in test_dataloader:
                test_batch = test_batch.cuda()
                test_full_batch = torch.cat([model.start_token.expand(test_batch.shape[0], -1, -1), test_batch], dim=1)
                test_predictions = model(test_full_batch)
                test_predictions = test_predictions[:, :-1, :]
                test_loss += F.binary_cross_entropy(test_predictions, test_batch)
            test_loss /= len(test_dataloader)
            test_losses.append(test_loss.detach().cpu().numpy())
        print(f'val loss: {val_losses[-1]:.4f}, test loss: {test_losses[-1]:.4f}')
        visualize_loss(train_losses, val_losses, test_losses, output_name=f'{output_folder}/loss_curves.png')
        torch.save(model.state_dict(), f'{output_folder}/model_{epoch}.pt')
if __name__ == '__main__':
    main()