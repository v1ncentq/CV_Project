# training/train_baseline.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.dataset import get_dataloaders
from utils.train_utils import train_with_early_stopping
from models.custom_cnn import CustomCNN
from config import DEVICE

def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results/curves", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)

    train_loader, val_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    model = CustomCNN(num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model, history, best_val_acc = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=30,       # >= 20
        patience=5,
        checkpoint_path="checkpoints/baseline_best.pth",
    )

    print("Best val accuracy:", best_val_acc)

    # Learning curves
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Baseline CNN Loss")
    plt.savefig("results/curves/baseline_loss.png")

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Baseline CNN Accuracy")
    plt.savefig("results/curves/baseline_acc.png")

if __name__ == "__main__":
    main()
