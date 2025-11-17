# utils/train_utils.py
import time
from copy import deepcopy
import torch
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_corrects, n = 0.0, 0, 0

    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss     += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        n += inputs.size(0)

    return running_loss / n, running_corrects / n


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, running_corrects, n = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Val", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss     += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            n += inputs.size(0)

    return running_loss / n, running_corrects / n


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=30,
    patience=5,
    checkpoint_path="checkpoints/baseline_best.pth",
):
    best_val_acc = 0.0
    best_model_wts = deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        since = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | "
            f"Epoch time: {time.time() - since:.1f}s"
        )

        # Early stopping по val_loss (або по val_acc — як хочеш)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = deepcopy(model.state_dict())
            torch.save(best_model_wts, checkpoint_path)
            print(f"==> New best model saved to {checkpoint_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)
    return model, history, best_val_acc
