from model import GPTLanguageModel
from hyperparameters import device, block_size, max_new_tokens, d_model, n_heads, n_layers, dropout
import torch
from data import decode, vocab_size

@torch.no_grad()
def generate():
    model = GPTLanguageModel(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_heads=n_heads, 
        block_size=block_size, 
        n_layers=n_layers, 
        dropout=dropout
        ).to(device)

    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device) 
    out = model.generate(context, max_new_tokens=max_new_tokens)
    text = decode(out[0].tolist())
    print(text)
    with open("sample.txt", "w") as f:
        f.write(text)

if __name__ == "__main__":
    generate()