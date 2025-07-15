# model.py
# --------------------------------------------
# Defines the image classification model architecture.
# Includes a utility function to load the model with saved weights.
# --------------------------------------------

import torch
import torch.nn as nn
from torchvision import models

# Class labels used in training
from . import CLASS_NAMES

def create_model(num_classes=4):
    """
    Returns a ResNet18 model adjusted for the specified number of output classes.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(weights_path, device=None):
    """
    Loads the trained model weights from file.

    Args:
        weights_path (str): Path to the .pt file
        device (torch.device): Optional torch device (CPU or CUDA)

    Returns:
        torch.nn.Module: The loaded model in evaluation mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = create_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model
