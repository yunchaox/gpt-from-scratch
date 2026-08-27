import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self,dim: int, eps: float=1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scale (gamma) initialized to 1s
        self.gamma = nn.Parameter(torch.ones(dim))
        # Learnable shift (beta) initialized to 0s
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalizes the features (the embedding dimension C) for each token independently across its sequence.
        # Computes mean and variance over the last dimension (C) per position.
        # keepdim=True preserves the number of dimensions for broadcasting. 
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        # Scale and shift
        out = self.gamma * x_hat + self.beta
        return out
        
               
class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.register_buffer(
            'mask', 
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # split along the last dimension into three (Q, K, V)
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # compute the scaled dot-product attention
        # scale Q by 1/sqrt(head_dim)
        attn_score = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        # Apply causal mask
        attn_score = attn_score.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # Softmax to get probabilities
        attn_probs = F.softmax(attn_score, dim=-1)
        # Multiply by V
        out = attn_probs @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = LayerNorm(dim=d_model)
        self.attn = CausalMultiHeadAttention(d_model, n_heads, block_size)
        self.ln2 = LayerNorm(dim=d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        n_heads: int, 
        block_size: int, 
        n_layers: int, 
        dropout: float):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # takes the last block of tokens, of shape (B, block_size)
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            # takes the logits of the last position
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            # sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
            