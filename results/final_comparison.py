# results/final_comparison.py

import os
import sys
import time

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# --- додамо корінь проєкту в sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DEVICE
from utils.dataset import get_dataloaders
from models.custom_cnn import CustomCNN
from models.transfer_models import get_resnet18, get_efficientnet_b0


# ----------------- ХЕЛПЕРИ ----------------- #

def build_model_from_name(model_name: str, num_classes: int):
    """
    Створює потрібну архітектуру за назвою моделі.

    Приклади назв:
      - "baseline_cnn"
      - "resnet18"
      - "efficientnet_b0"
      - "efficient_b0_ft_advanced_l2_ls" (важливе тут "efficient")
    """
    name = model_name.lower()

    if "baseline" in name or "custom" in name:
        model = CustomCNN(num_classes=num_classes)

    elif "resnet18" in name:
        # feature_extract тут не критично для оцінки (впливає лише на requires_grad)
        model = get_resnet18(num_classes=num_classes)

    elif "efficientnet" in name or "efficient" in name:
        model = get_efficientnet_b0(num_classes=num_classes)

    else:
        raise ValueError(f"Не знаю, як створити модель для model_name='{model_name}'")

    return model


def evaluate_model(model: nn.Module, loader, device):
    """Повертає (avg_loss, accuracy) на переданому loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_all_summaries():
    """
    Читає summary-таблиці:
      - results/transfer_summary.csv        (transfer learning)
      - results/augmentation_summary.csv    (augmentation/regularization, якщо є)

    УВАГА: results/models_summary.csv більше НЕ використовуємо, бо там каша.
    """
    paths = [
        ("results/transfer_summary.csv", "transfer"),
        ("results/augmentation_summary.csv", "augmentation"),
    ]

    dfs = []
    for path, category in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["category"] = category
            dfs.append(df)
        else:
            print(f"[info] Summary file not found, skipping: {path}")

    if not dfs:
        raise RuntimeError("Не знайдено жодної summary-таблиці (transfer/augmentation).")

    all_df = pd.concat(dfs, ignore_index=True)

    if "ckpt_path" not in all_df.columns:
        raise RuntimeError("Summary-файли повинні містити колонку 'ckpt_path'.")

    # Якщо немає колонки test_acc — створимо
    if "test_acc" not in all_df.columns:
        all_df["test_acc"] = None

    # Якщо немає колонки mode — теж створимо (буде NaN)
    if "mode" not in all_df.columns:
        all_df["mode"] = None

    return all_df


# ----------------- MAIN ----------------- #

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)

    print("Using device:", DEVICE)

    # 1) Дані: seg_test будемо вважати тестовою вибіркою
    _, test_loader, class_names = get_dataloaders(
        batch_size=64,
        img_size=150,
        aug_type="baseline",
    )
    num_classes = len(class_names)
    print("Test size:", len(test_loader.dataset))
    print("Classes:", class_names)

    # 2) Підтягуємо всі моделі з transfer/augmentation summary
    df = load_all_summaries()

    # 3) Додаємо baseline CNN, якщо є checkpoints/baseline_best.pth
    baseline_ckpt = os.path.join("checkpoints", "baseline_best.pth")
    if os.path.exists(baseline_ckpt):
        print("\n[info] Додаємо baseline CNN до фінальної таблиці...")
        baseline_model = CustomCNN(num_classes=num_classes).to(DEVICE)
        baseline_state = torch.load(baseline_ckpt, map_location=DEVICE)
        baseline_model.load_state_dict(baseline_state)

        # Оцінимо baseline на test як і всі інші
        _, baseline_acc = evaluate_model(baseline_model, test_loader, DEVICE)
        baseline_params = count_parameters(baseline_model)

        baseline_row = {
            "model_name": "baseline_cnn",
            "val_acc": baseline_acc,    # формально це теж acc на seg_test
            "test_acc": None,           # перезапишемо нижче в циклі
            "params": baseline_params,
            "train_time_min": None,     # можна заповнити вручну, якщо знаєш
            "ckpt_path": baseline_ckpt,
            "category": "baseline",
            "mode": None,
        }

        df = pd.concat([df, pd.DataFrame([baseline_row])], ignore_index=True)
    else:
        print("\n[info] baseline_best.pth не знайдено — baseline CNN не буде включено.")

    print("\nSummary tables merged:")
    print(df)

    # 4) Оцінюємо test_acc для КОЖНОЇ моделі з доступним checkpoint-ом
    test_accs = []

    for idx, row in df.iterrows():
        model_name = row["model_name"]
        ckpt_path = row.get("ckpt_path", None)

        print("\n" + "-" * 80)
        print(f"[{idx}] Model: {model_name}")
        print(f"Category: {row.get('category', 'N/A')}")
        print(f"Checkpoint: {ckpt_path}")

        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"[warn] Checkpoint not found, skipping test eval.")
            test_accs.append(None)
            continue

        # Створюємо архітектуру
        try:
            model = build_model_from_name(model_name, num_classes).to(DEVICE)
        except ValueError as e:
            print(f"[warn] {e} — пропускаємо цю модель.")
            test_accs.append(None)
            continue

        # Завантажуємо ваги — якщо state_dict не підходить, пропускаємо
        try:
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"[error] Failed to load checkpoint into '{model_name}': {e}")
            test_accs.append(None)
            continue

        # Оцінка на test
        start = time.time()
        test_loss, test_acc = evaluate_model(model, test_loader, DEVICE)
        elapsed_min = (time.time() - start) / 60.0

        # Параметри (якщо params порожній/NaN — перезаписуємо)
        params = count_parameters(model)
        df.loc[idx, "params"] = params

        print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f} (eval time {elapsed_min:.2f} min)")
        test_accs.append(test_acc)

    df["test_acc"] = test_accs

    # 5) Зберігаємо фінальну таблицю
    final_table_path = "results/final_comparison.csv"
    df.to_csv(final_table_path, index=False)
    print("\nFinal comparison table saved to:", final_table_path)
    print(df)

    # 6) Обираємо найкращу модель за test_acc
    df_valid = df.dropna(subset=["test_acc"])
    if df_valid.empty:
        print("\n[warn] Немає жодної моделі з порахованим test_acc — Confusion Matrix не буде побудовано.")
        return

    best_idx = df_valid["test_acc"].idxmax()
    best_row = df_valid.loc[best_idx]

    print("\nBest model by test_acc:")
    print(best_row)

    best_model_name = best_row["model_name"]
    best_ckpt_path = best_row["ckpt_path"]

    # 7) Створюємо Confusion Matrix для найкращої моделі
    print(f"\nBuilding confusion matrix for best model: {best_model_name}")
    best_model = build_model_from_name(best_model_name, num_classes).to(DEVICE)
    best_state = torch.load(best_ckpt_path, map_location=DEVICE)
    best_model.load_state_dict(best_state)
    best_model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = best_model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print("\nClassification report for best model:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")
    plt.title(f"Confusion Matrix – {best_model_name}")
    plt.tight_layout()

    cm_path = "results/confusion_matrices/best_model_confusion_test.png"
    plt.savefig(cm_path)
    print("\nSaved confusion matrix to:", cm_path)


if __name__ == "__main__":
    main()
