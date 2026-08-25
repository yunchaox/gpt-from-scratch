import os
import urllib.request
import torch

from hyperparameters import train_ratio, block_size, device, batch_size

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
PATH = "input.txt"

def load_text():
    if not os.path.exists(PATH):
        urllib.request.urlretrieve(url, PATH)
    with open(PATH, "r", encoding="utf-8") as f:
        return f.read()

text = load_text()
# Tokenization.
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Character to integer mapping.
stoi = {c: i for i, c in enumerate(chars)}
# Integer to character mapping.
itos = {i: c for i, c in enumerate(chars)}

def encode(s: str) -> list[int]:
    """ Encodes a string into a list of integers using the stoi mapping. """
    return [stoi[c] for c in s]

def decode(l: list[int]) -> str:
    """ Decodes a list of integers into a string using the itos mapping. """
    return "".join(itos[i] for i in l)

# Converts to torch.Tensor, splits into train/val, and returns a helper function to get a random batch of data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(train_ratio * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data_source = train_data if split == "train" else val_data
    # Generates a random starting positions for the sequences in one training batch from [0, high), with size (batch_size, ).
    ix = torch.randint(len(data_source) - block_size, (batch_size, ))
    # For each starting position, stack the next 'block_size' elements to form the training sequence.
    x = torch.stack([data_source[i: i + block_size] for i in ix])
    # The target sequence is the same as the training sequence, but shifted by one position to the right.
    y = torch.stack([data_source[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
