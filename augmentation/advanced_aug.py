# augmentation/advanced_aug.py

from torchvision import transforms


def get_baseline_train_transform(img_size: int = 150):
    """Базова аугментація: тільки фліп + нормалізація."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_advanced_train_transform(img_size: int = 150):
    """
    Advanced-аугментація:
    - випадковий кроп/масштаб
    - повороти/зсуви/шеар
    - зміна яскравості/контрасту/насиченості
    - перспективні спотворення
    - RandomErasing
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            )
        ], p=0.7),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10,
        ),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.4),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.4),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform(img_size: int = 150):
    """Те, що застосовується до val / test: без аугментацій."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
