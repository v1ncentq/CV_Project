# models/transfer_models.py
import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting: bool):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_resnet18(num_classes: int, feature_extract: bool = True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    set_parameter_requires_grad(model, feature_extract)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def get_efficientnet_b0(num_classes: int, feature_extract: bool = True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    set_parameter_requires_grad(model, feature_extract)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
