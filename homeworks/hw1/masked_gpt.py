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
        attn_output = self.attn(x, mask)
        x = self.ln1(x + attn_output) 
        ff_output = self.ff(x)
        x = self.ln2(x + ff_output)
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
        self.head = nn.Linear(d_model, d_output)

        self.d_output = d_output
        self.max_seq_len = max_seq_len

        pos_enc = sinusoidal_positional_encoding(max_seq_len + 1, d_model)  # Precompute positional encodings for a maximum sequence length of 400
        self.register_buffer('pos_enc', pos_enc)

        start_token = torch.zeros((1, 1, 1), dtype=torch.float32)
        self.register_buffer('start_token', start_token)

        mask = torch.tril(torch.ones((max_seq_len + 1, max_seq_len + 1), dtype=torch.bool))  # Causal mask for maximum sequence length
        self.register_buffer('mask', mask)

    def forward(self, x):
        x = 2 * x - 1
        x = torch.cat([self.start_token.expand(x.shape[0], -1, -1), x], dim=1)  # Prepend start token, resulting in shape (batch_size, seq_len + 1, d_model)

        x = self.stem(x)  # (batch_size, seq_len, d_model)
        x += self.pos_enc[:x.size(1), :].unsqueeze(0)  # Add positional encoding

        for layer in self.layers:
            x = layer(x, self.mask)

        x = self.head(x)

        if self.d_output == 1:
            x = torch.sigmoid(x)
        else:
            x = F.softmax(x, dim=-1)
        return x[:, 1:, :]  # Remove the start token output
    
    def sample(self, N):
        # check that this is correctly sampling the modele
        samples = torch.randint(2, (N, self.max_seq_len, 1)).cuda()
        for i in range(self.max_seq_len):
            pred = self.forward(samples)

            # get current value and sample
            if self.d_output == 1:
                samples[:, i, 0] = torch.bernoulli(pred[:, i, 0])
            else:
                raise "not yet implemented for multinomials"
        return samples

def sinusoidal_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe