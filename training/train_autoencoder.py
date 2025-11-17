# training/train_autoencoder.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.dataset import get_dataloaders
from models.autoencoder import ConvAutoencoder

def main():
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders(batch_size=64)

    model = ConvAutoencoder(latent_dim=64).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")

    for epoch in range(1, 21):
        model.train()
        train_loss = 0.0
        n = 0
        for x, _ in tqdm(train_loader, desc=f"Train AE {epoch}"):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n += x.size(0)
        train_loss /= n

        # валід
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, _ = model(x)
                loss = criterion(x_hat, x)
                val_loss += loss.item() * x.size(0)
                n_val += x.size(0)
        val_loss /= n_val

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/autoencoder_best.pth")
            print("==> Saved best AE")

if __name__ == "__main__":
    main()
