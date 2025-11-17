# experiments/transfer_comparison.py

import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from config import DEVICE
from utils.dataset import get_dataloaders
from utils.train_utils import train_with_early_stopping
from models.transfer_models import get_resnet18, get_efficientnet_b0


def build_model(model_name: str, num_classes: int, feature_extract: bool):
    """Створення потрібної transfer-моделі."""
    if model_name == "resnet18":
        model = get_resnet18(num_classes=num_classes, feature_extract=feature_extract)
    elif model_name == "efficientnet_b0":
        model = get_efficientnet_b0(num_classes=num_classes, feature_extract=feature_extract)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model


def make_optimizer_with_param_groups(
    model,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float = 1e-4,
):
    """Різні learning rate для backbone і голови класифікатора."""
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name or "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Using device:", DEVICE)

    # 1) Дані
    train_loader, val_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    print("Classes:", class_names)

    # 2) Набір експериментів: модель + режим + гіперпараметри
    experiments = [
        {
            "model_name": "resnet18",
            "mode": "fe",          # feature extraction
            "lr_backbone": 1e-4,
            "lr_head": 1e-3,
            "epochs": 20,
            "patience": 5,
        },
        {
            "model_name": "resnet18",
            "mode": "ft",          # fine-tuning
            "lr_backbone": 1e-5,
            "lr_head": 5e-4,
            "epochs": 20,
            "patience": 5,
        },
        {
            "model_name": "efficientnet_b0",
            "mode": "fe",
            "lr_backbone": 1e-4,
            "lr_head": 1e-3,
            "epochs": 20,
            "patience": 5,
        },
        {
            "model_name": "efficientnet_b0",
            "mode": "ft",
            "lr_backbone": 1e-5,
            "lr_head": 5e-4,
            "epochs": 20,
            "patience": 5,
        },
    ]

    results = []

    # 3) Прогін усіх експериментів
    for exp in experiments:
        model_name = exp["model_name"]
        mode = exp["mode"]
        feature_extract = mode == "fe"

        print("\n" + "=" * 80)
        print(f"Experiment: model={model_name}, mode={mode}")
        print("=" * 80)

        # Модель
        model = build_model(model_name, num_classes, feature_extract).to(DEVICE)

        # Критерій + оптимізатор
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = make_optimizer_with_param_groups(
            model,
            lr_backbone=exp["lr_backbone"],
            lr_head=exp["lr_head"],
            weight_decay=1e-4,
        )

        # Шлях до чекпоінта для цієї конфігурації
        ckpt_name = f"{model_name}_{mode}_best.pth"
        ckpt_path = os.path.join("checkpoints", ckpt_name)

        # Навчання з early stopping
        start_time = time.time()
        model, history, best_val_acc = train_with_early_stopping(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            DEVICE,
            num_epochs=exp["epochs"],
            patience=exp["patience"],
            checkpoint_path=ckpt_path,
        )
        total_time_min = (time.time() - start_time) / 60.0
        n_params = count_parameters(model)

        print(f"Best val acc for {model_name} ({mode}): {best_val_acc:.4f}")
        print(f"Trainable params: {n_params}")
        print(f"Training time: {total_time_min:.2f} min")

        results.append(
            {
                "model_name": model_name,
                "mode": mode,
                "val_acc": best_val_acc,
                "params": n_params,
                "train_time_min": total_time_min,
                "ckpt_path": ckpt_path,
            }
        )

    # 4) Таблиця з усіма результатами
    df = pd.DataFrame(results)
    summary_path = "results/transfer_summary.csv"
    df.to_csv(summary_path, index=False)
    print("\nAll experiments summary saved to:", summary_path)
    print(df)

    # 5) Вибір найкращої моделі за val_acc
    best_idx = df["val_acc"].idxmax()
    best_row = df.iloc[best_idx]

    print("\nBest experiment:")
    print(best_row)

    # 6) Копіюємо її чекпоінт у transfer_best.pth
    src_ckpt = best_row["ckpt_path"]
    dst_ckpt = os.path.join("checkpoints", "transfer_best.pth")
    shutil.copy(src_ckpt, dst_ckpt)
    print(f"\nSaved best transfer model to {dst_ckpt}")


if __name__ == "__main__":
    main()
