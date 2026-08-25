from hyperparameters import device, max_iters, eval_interval, seed, lr, d_model, n_heads, block_size, n_layers, dropout
from model import GPTLanguageModel
from data import get_batch, vocab_size
import torch

def train():
    torch.manual_seed(seed)
    model = GPTLanguageModel(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_heads=n_heads, 
        block_size=block_size, 
        n_layers=n_layers, 
        dropout=dropout
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    ### Iterates over max_iters, sampling and training the model with batches of data
    for step in range(max_iters):
        # Generates a random batch of training data.
        xb, yb = get_batch("train")
        # Computes the logits and loss of the model.
        logits, loss = model(xb, yb)
        # Resets the gradients of the model's parameters to zero.
        optimizer.zero_grad()
        # Computes the gradient of the loss with respect to the model's parameters.
        loss.backward()
        # Updates the model's parameters to minimize the loss.
        optimizer.step()
        # Outputs the loss every eval_interval steps or when the loop finishes.
        if step % eval_interval == 0 or step == max_iters - 1:
            print(f"step {step}: loss {loss.item()}")
    torch.save(model.state_dict(), "model.pt")
    print("Model saved to model.pt")

if __name__ == "__main__":
    train()
            