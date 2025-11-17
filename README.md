# 🌄 Intel Image Classification – CNNs, Transfer Learning & Autoencoders

Deep Learning / Computer Vision project for the **Intel Image Classification** dataset.  
The goal is to:

- build a **baseline CNN** from scratch;
- compare several **transfer learning** approaches (ResNet18, EfficientNet-B0);
- study **data augmentation & regularization**;
- train an **autoencoder** on the same dataset and explore its applications;
- make a **final comparison** of all classification models.

---

## 🧩 Dataset

Default dataset: **Intel Image Classification** (6 classes of natural scenes):

- `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`

You can download it from Kaggle and unpack into `data/` directory.  
The code automatically searches for `seg_train` and `seg_test` folders inside `data/...`,  
so the following layouts will work, for example:

```text
data/
  intel/
    seg_train/
      seg_train/
        buildings/
        forest/
        ...
    seg_test/
      seg_test/
        buildings/
        forest/
        ...

🏗 Project Structure
CV_Project/
├── config.py                     # DEVICE (cpu/cuda) detection and basic config
├── data/                         # Dataset root (not in repo)
│   └── intel/...
├── models/
│   ├── custom_cnn.py             # Baseline CNN architecture
│   ├── transfer_models.py        # Transfer learning models (ResNet18, EfficientNet-B0)
│   └── autoencoder.py            # Conv autoencoder for images
├── utils/
│   ├── dataset.py                # get_dataloaders, transforms (baseline/advanced)
│   ├── train_utils.py            # training loops, early stopping, history
│   └── eval_utils.py             # evaluation helpers (metrics, confusion matrix, etc.)
├── augmentation/
│   └── advanced_aug.py           # advanced data augmentation pipeline
├── training/
│   ├── train_baseline.py         # training baseline CNN
│   └── train_transfer.py         # training single EfficientNet-B0 FT + saving curves
├── experiments/
│   ├── transfer_comparison.py    # compare multiple transfer models/configs
│   └── augmentation_study.ipynb  # study augmentation & regularization configs
├── applications/
│   └── ae_applications.ipynb     # autoencoder training + anomaly detection, latent space, denoising
├── results/
│   ├── curves/                   # training curves (loss/acc)
│   ├── confusion_matrices/       # confusion matrix images
│   ├── transfer_summary.csv      # summary for transfer learning experiments
│   ├── augmentation_summary.csv  # summary for augmentation experiments (if used)
│   ├── final_comparison.csv      # final merged comparison of all models (Stage 6)
│   └── final_comparison.py       # script to build final comparison & confusion matrix
└── README.md

⚙️ Environment & Installation
1. Create virtual environment
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

2. Install PyTorch (with CUDA if available)

Example for CUDA 11.8 (adjust to your system / CUDA version):
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio

Or CPU-only (for testing):
pip install torch torchvision torchaudio

3. Install other dependencies
pip install -r requirements.txt
(If there is no requirements.txt, install manually: matplotlib, pandas, scikit-learn, etc.)

🚀 Stage 2 – Baseline CNN
Baseline model: custom convolutional network defined in models/custom_cnn.py.

Train baseline model

From project root:
# make sure virtualenv is active
python -m training.train_baseline

This script:

loads train/val data via get_dataloaders (baseline augmentations),
trains the custom CNN for ≥20 epochs with early stopping,
saves the best model to:

checkpoints/baseline_best.pth
saves training curves (loss/accuracy) into results/curves/.

🔁 Stage 3 – Transfer Learning (ResNet18 & EfficientNet-B0)

Transfer models are defined in:
models/transfer_models.py

Training & comparison script
python -m experiments.transfer_comparison

This script:

trains several transfer configs, for example:

resnet18 (feature extraction & fine-tuning),

efficientnet_b0 (feature extraction & fine-tuning),

uses different learning rates for backbone & classifier head,

applies early stopping and saves checkpoints as:
checkpoints/resnet18_fe_best.pth
checkpoints/resnet18_ft_best.pth
checkpoints/efficientnet_b0_fe_best.pth
checkpoints/efficientnet_b0_ft_best.pth
logs all experiments to:
results/transfer_summary.csv
with columns such as:
model_name, mode (fe/ft), val_acc, params, train_time_min, ckpt_path.

🎛 Stage 4 – Data Augmentation & Regularization
Advanced augmentation is implemented in:
augmentation/advanced_aug.py

Two regimes are supported in utils/dataset.py:

aug_type="baseline"
aug_type="advanced"

Augmentation & regularization study

Run and edit:
experiments/augmentation_study.ipynb

This notebook:
trains (usually EfficientNet-B0 FT) under different configs:
baseline vs advanced augmentation,
with/without L2 regularization (weight decay),
with/without label smoothing in CrossEntropyLoss.
saves curves to results/curves/efficient_b0_ft_<config>_*.png,
logs results for each config to:
results/augmentation_summary.csv

Columns include:
config, aug_type, weight_decay, label_smoothing, val_acc, params, train_time_min, ckpt_path.

You can additionally implement and test simple ensembles of top models in this notebook
(averaging logits from several checkpoints).

🧠 Stage 5 – Autoencoder
Conv autoencoder is implemented in:
models/autoencoder.py
Autoencoder is trained and analyzed in:
applications/ae_applications.ipynb

What the notebook does

Training:

uses same dataset (Intel), but with ToTensor() only (no Normalize),
trains a convolutional autoencoder with MSELoss + early stopping,
saves best weights to checkpoints/autoencoder_best.pth,
plots train/val loss curves.

Applications (at least two as required):

Anomaly detection:
compute reconstruction error (MSE per image) on validation/test set.
Images with highest reconstruction error are treated as “anomalies”.
The notebook visualizes original vs reconstructed images for top anomalies.

Latent space visualization:
encode images into latent vectors z, then apply t-SNE or PCA.
The 2D scatter plot shows how classes cluster in latent space.

Denoising (optional):
add noise to input images, run them through the autoencoder,
and visualize original → noisy → denoised triplets.

Autoencoder models are not included in the final classification comparison table.

📊 Stage 6 – Final Comparison & Confusion Matrix
Final evaluation script:
results/final_comparison.py
Run:
python results/final_comparison.py

This script:

1. Loads and merges summary files:

results/transfer_summary.csv
results/augmentation_summary.csv (if present)

2. Optionally adds baseline CNN from checkpoints/baseline_best.pth.

3. For each model row:

builds the corresponding architecture (CustomCNN, ResNet18, EfficientNet-B0),
loads weights from ckpt_path,
evaluates test accuracy on seg_test via get_dataloaders,
updates columns:
test_acc
params (number of trainable parameters)

4. Saves the final comparison table to:
results/final_comparison.csv

with columns like:
model_name
category (baseline / transfer / augmentation)
val_acc
test_acc
params
train_time_min
ckpt_path

5. Finds the best model by test_acc and:

computes predictions on the test set,
prints a classification_report,
builds and saves confusion matrix to:
results/confusion_matrices/best_model_confusion_test.png

📝 Summary of Findings (short)

Baseline Custom CNN reaches >60% accuracy, satisfying minimal requirements,
but is clearly outperformed by transfer learning.

ResNet18 and especially EfficientNet-B0 with fine-tuning achieve much higher validation/test accuracy.

Advanced augmentation + L2 + (optionally) label smoothing improves generalization and stabilizes training.

EfficientNet-B0 FT with proper augmentation/regularization is typically the best single model in terms of accuracy vs size.

Ensembling several strong models can slightly increase test accuracy further,
at the cost of higher inference time and memory.

Autoencoder is useful for:

anomaly detection via reconstruction error,
visualizing the structure of the dataset in latent space,
(optionally) image denoising.

🧷 Notes

This project is intended both as a coursework / lab project and as a clean example of:

building a baseline CNN,
migrating to transfer learning,
experimenting with augmentation & regularization,
and adding unsupervised learning (autoencoder) on top.

Feel free to adapt models/, experiments/, and applications/ to other image datasets
(just update get_dataloaders and class mappings).
