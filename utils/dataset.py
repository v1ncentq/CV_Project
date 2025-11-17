# utils/dataset.py
import os
from torch.utils.data import DataLoader
from torchvision import datasets

from augmentation.advanced_aug import (
    get_baseline_train_transform,
    get_advanced_train_transform,
    get_eval_transform,
)


def get_transforms(img_size=150, mode="train", aug_type="baseline"):
    """
    aug_type: "baseline" або "advanced"
    mode: "train" або "val"
    """
    if mode == "train":
        if aug_type == "advanced":
            return get_advanced_train_transform(img_size)
        else:
            return get_baseline_train_transform(img_size)
    else:
        return get_eval_transform(img_size)


def _auto_find_split_dir(search_root: str, split_name: str) -> str:
    """
    Шукає директорію з назвою split_name (seg_train/seg_test) десь всередині search_root.
    Повертає перший знайдений шлях, у якому є підпапки-класи.

    Наприклад, знайде:
    - data/intel/seg_train/seg_train
    - data/intel/seg_train
    - data/seg_train
    - будь-який інший .../seg_train з підпапками класів
    """
    search_root = os.path.abspath(search_root)

    # Спочатку пробуємо кілька типових варіантів
    candidates = [
        os.path.join(search_root, "intel", split_name, split_name),
        os.path.join(search_root, "intel", split_name),
        os.path.join(search_root, split_name, split_name),
        os.path.join(search_root, split_name),
    ]

    for path in candidates:
        if os.path.isdir(path):
            # перевіримо, що всередині є хоч якісь підпапки
            subdirs = [
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
            ]
            if subdirs:
                return path

    # Якщо не спрацювали типові варіанти – робимо повний обхід дерева
    for root, dirs, files in os.walk(search_root):
        if os.path.basename(root) == split_name:
            subdirs = [
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]
            if subdirs:
                return root

    raise FileNotFoundError(
        f"Не знайдено директорію для {split_name} всередині {search_root}.\n"
        f"Перевір структуру папок у 'data/' та назву датасету."
    )


def get_dataloaders(
    data_root="data",       # тепер шукаємо від 'data', а не 'data/intel'
    batch_size=64,
    img_size=150,
    num_workers=4,
    aug_type="baseline",
):
    # знайдемо реальні шляхи до train/val
    train_dir = _auto_find_split_dir(data_root, "seg_train")
    val_dir   = _auto_find_split_dir(data_root, "seg_test")

    print("Train dir:", train_dir)
    print("Val dir:", val_dir)

    train_ds = datasets.ImageFolder(
        train_dir,
        transform=get_transforms(img_size=img_size, mode="train", aug_type=aug_type)
    )
    val_ds = datasets.ImageFolder(
        val_dir,
        transform=get_transforms(img_size=img_size, mode="val", aug_type=aug_type)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, train_ds.classes
