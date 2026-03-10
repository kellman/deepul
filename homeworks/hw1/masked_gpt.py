import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_attention(q, k, v, mask):
    """
    q, k, v: (batch_size, seq_len, d_model) tensors
    mask: (seq_len, seq_len) binary tensor where mask[i, j] = 0 if position j should not be attended to when processing position i

    Returns:
    - output: (batch_size, seq_len, d_model) tensor of attention outputs
    """
    score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.shape[-1], dtype=torch.float32))
    score = score.masked_fill(mask == 0, float('-inf'))  # mask
    attn_weights = torch.softmax(score, dim=-1)  # (batch_size, seq_len, seq_len)
    output = torch.matmul(attn_weights, v)  # (batch_size, seq_len, d_model)
    return output


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_head)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_head)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_head)

        attn_output = masked_attention(q, k, v, mask)  # (batch_size, num_heads, seq_len, d_head)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)
        output = self.out_linear(attn_output)  # (batch_size, seq_len, d_model)
        return output

class MaskedTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MaskedMultiHeadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, d_output, max_seq_len=400):
        """
        Generative transformer with masked self-attention

        Args:
            d_model: dimension of model embeddings
            num_heads: number of attention heads
            d_ff: dimension of feedforward network
            num_layers: number of transformer blocks
            d_output: dimension of output

        Return: 
            output: (batch_size, seq_len, d_output) tensor of transformer outputs
        """
        super().__init__()
        self.stem = nn.Linear(1, d_model)  # project input to d_model dimensions
        self.layers = nn.ModuleList([MaskedTransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_output)
        )

        self.d_output = d_output
        self.max_seq_len = max_seq_len

        pos_enc = sinusoidal_positional_encoding(max_seq_len + 1, d_model)  # Precompute positional encodings for a maximum sequence length of 400
        self.register_buffer('pos_enc', pos_enc)

        start_token = torch.zeros((1, 1, 1), dtype=torch.float32)
        self.register_buffer('start_token', start_token)

        mask = torch.tril(torch.ones((max_seq_len + 1, max_seq_len + 1), dtype=torch.bool))  # Causal mask for maximum sequence length
        self.register_buffer('mask', mask)

    def forward(self, x):
        # x is (batch_size, seq_len + 1, 1), with x[:, 0] = start_token
        x = 2 * x - 1
        x = self.stem(x)  # (batch_size, seq_len + 1, d_model)
        x += self.pos_enc[:x.size(1), :].unsqueeze(0)  # Add positional encoding

        for layer in self.layers:
            x = layer(x, self.mask[:x.size(1), :x.size(1)])

        x = self.head(x)

        if self.d_output == 1:
            x = torch.sigmoid(x)
        else:
            x = F.softmax(x, dim=-1)
        return x
    
    def sample(self, N):
        samples = torch.zeros(N, self.max_seq_len, 1).cuda()
        current = self.start_token.expand(N, -1, -1).cuda()  # (N, 1, 1)
        for i in range(self.max_seq_len):
            pred = self.forward(current)  # current is (N, i+1, 1), start + i samples
            next_pred = pred[:, -1, 0]  # prediction for the next (i-th) pixel
            next_sample = torch.bernoulli(next_pred).unsqueeze(-1)  # (N, 1)
            samples[:, i, :] = next_sample
            current = torch.cat([current, next_sample.unsqueeze(-1)], dim=1)
        return samples

def sinusoidal_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe