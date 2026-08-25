import torch

batch_size = 64
block_size = 256
max_iters = 3000
eval_interval = 500
lr = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
d_model = 384
n_heads = 6
n_layers = 6
train_ratio = 0.9
max_new_tokens = 500
dropout = 0.2

seed = 1337
